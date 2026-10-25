#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["open3d==0.19.0", "trimesh", "scipy", "pillow", "opencv-python>=4.8", "pyobjc-framework-Cocoa; sys_platform == 'darwin'"]
# ///
"""Polycam raw-data export -> data/room/room.glb + markers.json (handoffAB.md deliverables 1-2).

    uv run tools/scan_to_room.py data/room/scan_raw --marker-size 0.179

Scan in Polycam's LiDAR mode with Developer mode on, then Export -> Raw data. Pass the unzipped
folder (or the .zip). --marker-size is the black square measured off the print, in metres.

  1. load the mesh and the keyframes (images, phone poses, LiDAR depth)
  2. work out how Polycam's poses and mesh line up, by checking mesh depth against LiDAR depth
  3. find the markers in the keyframes and cast their corners onto the mesh
  4. set the world frame: click a room corner on the floor, then a point on each of its two walls.
     The clicks only need to be close: they snap to the scan's walls and to where those walls meet
  5. write room.glb (world frame, transforms baked, <= --max-tris), markers.json, and plan.png:
     a top-down picture with the origin and axes, to tape-measure checks.json from

If step 2 finds no convention that lines up, the export can't be trusted: fall back to clicking
marker corners by hand (CAMERA.md E3), then to our own ARKit app.
"""

import argparse
import sys
import zipfile
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import cv2
import numpy as np

from printed import TAG_DICT
from roomlib import (
    DATA, FLIP_YZ, Mesh, detect_markers, load_json, marker_detector, pick_points, pixel_rays,
    plane_rms, save_json, transform, world_from_clicks,
)

DEPTH_OK_M = 0.02  # median mesh-vs-LiDAR depth error that counts as lined up
LIDAR_MAX_M = 5.0  # beyond this, LiDAR depth is a guess
CHECK_FRAMES = 12
MIN_MARKER_PX = 25  # smaller detections are too coarse to place a corner
MIN_SIGHTINGS = 3
SPREAD_WARN_M = 0.01
SCALE_TOL = 0.02  # handoff: scale within 2%
WALL_SNAP_MAX_DEG = 30  # a click further off every wall than this is used as clicked
WALL_SEARCH_M = 0.75  # how far along the floor from a wall click to look for the wall's surface
WALL_BAND_M = (1.3, 2.3)  # height band where walls are clear of furniture
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass
class Frame:
    name: str
    image: Path
    K: np.ndarray
    size: tuple  # (width, height) the intrinsics refer to
    T: np.ndarray  # 4x4 from Polycam's t_ij fields; what it means is decided in step 2
    depth: Path | None
    confidence: Path | None


def find_mesh(folder):
    """The export's glTF mesh: raw.glb, raw.gltf in newer captures, or the glTF export's own name."""
    meshes = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".glb", ".gltf"})
    glb = next((p for p in meshes if p.stem == "raw"), meshes[0] if meshes else None)
    if glb is None:
        sys.exit(f"no .glb/.gltf mesh in {folder}: is this an unzipped Polycam raw-data export?")
    return glb


def load_polycam(folder):
    """(mesh vertices, faces, alignmentTransform, frames) from an unzipped raw-data export."""
    import trimesh

    glb = find_mesh(folder)

    kf = folder / "keyframes"
    images, cameras = kf / "corrected_images", kf / "corrected_cameras"
    if not images.exists() or not cameras.exists():
        print("  no corrected_* folders (scan too long for pose optimization): using raw ARKit poses")
        images, cameras = kf / "images", kf / "cameras"

    frames = []
    for img in sorted(p for p in images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
        cam = cameras / f"{img.stem}.json"
        if not cam.exists():
            continue
        c = load_json(cam)
        T = np.eye(4)
        T[:3] = [[c[f"t_{i}{j}"] for j in range(4)] for i in range(3)]
        K = np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1]], np.float64)
        depth, conf = kf / "depth" / f"{img.stem}.png", kf / "confidence" / f"{img.stem}.png"
        frames.append(Frame(img.stem, img, K, (int(c["width"]), int(c["height"])), T,
                            depth if depth.exists() else None, conf if conf.exists() else None))
    if not frames:
        sys.exit(f"no keyframes with cameras in {images.parent}")

    align = np.eye(4)
    info = folder / "mesh_info.json"
    if info.exists() and "alignmentTransform" in (j := load_json(info)):
        align = np.array(j["alignmentTransform"], np.float64).reshape(4, 4).T  # stored column-major

    mesh = trimesh.load(glb, force="mesh")
    return np.asarray(mesh.vertices), np.asarray(mesh.faces), align, frames


def pose(T, cam_to_world, arkit_axes):
    """Read Polycam's 4x4 one way -> OpenCV world->camera (R, t)."""
    if cam_to_world:
        M = T[:3, :3] @ FLIP_YZ if arkit_axes else T[:3, :3]  # columns are the camera's axes
        R = M.T
        return R, -R @ T[:3, 3]
    R, t = T[:3, :3], T[:3, 3]
    return (FLIP_YZ @ R, FLIP_YZ @ t) if arkit_axes else (R, t)


def scaled_K(K, from_size, to_size):
    K = K.copy()
    K[0] *= to_size[0] / from_size[0]
    K[1] *= to_size[1] / from_size[1]
    K[2] = [0, 0, 1]
    return K


def depth_errors(mesh, frame, R, t, along_z, step=4):
    """|mesh depth - LiDAR depth| over a grid of confident LiDAR pixels, plus the hit fraction."""
    depth = cv2.imread(str(frame.depth), cv2.IMREAD_UNCHANGED).astype(np.float64) / 1000
    h, w = depth.shape
    vs, us = np.mgrid[step // 2:h:step, step // 2:w:step]
    vs, us = vs.ravel(), us.ravel()
    lidar = depth[vs, us]
    keep = (lidar > 0) & (lidar < LIDAR_MAX_M)
    if frame.confidence is not None:
        conf = cv2.imread(str(frame.confidence), cv2.IMREAD_UNCHANGED)
        keep &= conf[vs, us] >= conf.max()  # highest confidence level present
    if keep.sum() < 50:
        return np.array([]), 0.0

    K = scaled_K(frame.K, frame.size, (w, h))
    o, d = pixel_rays(K, None, R, t, np.column_stack([us[keep], vs[keep]]))
    hit = mesh.cast(o, d)
    measured = hit * (d @ R.T)[:, 2] if along_z else hit  # LiDAR is depth along z, or range
    ok = np.isfinite(measured)
    return np.abs(measured[ok] - lidar[keep][ok]), ok.mean()


def choose_convention(vertices, faces, align, frames):
    """Try every reading of the export; keep the one where mesh depth matches LiDAR depth."""
    with_depth = [f for f in frames if f.depth is not None]
    if not with_depth:
        print("  WARNING: no LiDAR depth in the export, so alignment can't be checked; assuming the documented"
              " convention (camera->world, ARKit axes, inverse alignmentTransform)")
        return "inverse alignment", (True, True), Mesh(transform(np.linalg.inv(align), vertices), faces), None

    sample = [with_depth[i] for i in np.linspace(0, len(with_depth) - 1, min(CHECK_FRAMES, len(with_depth))).astype(int)]
    meshes = {"inverse alignment": np.linalg.inv(align), "as exported": np.eye(4), "alignment": align}
    if np.allclose(align, np.eye(4)):
        meshes = {"as exported": np.eye(4)}

    results = []
    for mesh_name, M in meshes.items():
        mesh = Mesh(transform(M, vertices), faces)
        for (c2w, arkit), along_z in product(product((True, False), (True, False)), (True, False)):
            errs, hits = [], []
            for f in sample:
                R, t = pose(f.T, c2w, arkit)
                e, h = depth_errors(mesh, f, R, t, along_z)
                errs.append(e)
                hits.append(h)
            errs = np.concatenate(errs)
            score = float(np.median(errs)) if len(errs) and np.mean(hits) > 0.5 else np.inf
            results.append((score, float(np.mean(hits)), mesh_name, (c2w, arkit), along_z, mesh))

    results.sort(key=lambda r: r[0])
    print("  mesh-vs-LiDAR depth, best readings first:")
    for score, hits, mesh_name, (c2w, arkit), along_z, _ in results[:4]:
        print(f"    {score * 100:7.2f} cm median  ({hits:.0%} rays hit)  mesh {mesh_name}, "
              f"{'camera->world' if c2w else 'world->camera'}, {'ARKit' if arkit else 'OpenCV'} axes, "
              f"depth {'along z' if along_z else 'as range'}")
    best = results[0]
    if not best[0] < DEPTH_OK_M:
        sys.exit(f"FAIL: no reading lines the mesh up with LiDAR depth within {DEPTH_OK_M * 100:.0f} cm. "
                 "Fall back to clicking marker corners (CAMERA.md E3).")
    return best[2], best[3], best[5], best[0]


def locate_markers(mesh, frames, convention):
    """Each marker's 4 corners on the mesh: the median over every keyframe that saw it clearly."""
    detector = marker_detector()
    seen = {}
    for f in frames:
        img = cv2.imread(str(f.image), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        K = scaled_K(f.K, f.size, (img.shape[1], img.shape[0]))
        R, t = pose(f.T, *convention)
        for marker_id, c in detect_markers(detector, img).items():
            if np.linalg.norm(c - np.roll(c, -1, axis=0), axis=1).mean() < MIN_MARKER_PX:
                continue
            pts = mesh.hit_points(*pixel_rays(K, None, R, t, c))
            if not np.isnan(pts).any():
                seen.setdefault(marker_id, []).append(pts)

    found = {}
    for marker_id, obs in sorted(seen.items()):
        obs = np.array(obs)
        corners = np.median(obs, axis=0)
        spread = float(np.median(np.linalg.norm(obs - corners, axis=2)))
        found[marker_id] = (corners, len(obs), spread)
    return found


def fit_floor(mesh, corner, up=np.array([0.0, 1.0, 0.0])):
    """Floor plane near the clicked corner -> (unit up normal, corner moved down onto the plane)."""
    import open3d as o3d

    v, n = mesh.vertices, mesh.vertex_normals()
    rel = v - corner
    near = np.linalg.norm(rel - np.outer(rel @ up, up), axis=1) < 2.0
    upward = near & (n @ up > 0.9)
    # The floor is the lowest broad flat surface near the corner; desks and seats are above it.
    floor_h = np.percentile(v[upward] @ up, 5) if upward.any() else corner @ up
    flat = upward & (np.abs(v @ up - floor_h) < 0.05)
    if flat.sum() < 100:
        print("  WARNING: little floor found near the corner click; using gravity as up")
        return up, corner
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(v[flat]))
    (a, b, c, d), _ = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    normal = np.array([a, b, c])
    scale = np.linalg.norm(normal)
    normal, d = normal / scale, d / scale
    if normal @ up < 0:
        normal, d = -normal, -d
    return normal, corner - (normal @ corner + d) * normal


def wall_angle(tm):
    """The room's wall direction in the floor plane (radians), from every vertical surface's normal,
    weighted by area. Walls meet at right angles, so this plus multiples of 90 degrees covers them all.
    Polycam's frames are gravity-aligned, so vertical means a normal with no Y."""
    n, w = tm.face_normals, tm.area_faces
    vertical = np.abs(n[:, 1]) < 0.2
    theta = np.arctan2(n[vertical, 2], n[vertical, 0])
    return 0.25 * np.arctan2((w[vertical] * np.sin(4 * theta)).sum(), (w[vertical] * np.cos(4 * theta)).sum())


def snap(direction, base):
    """The wall direction nearest a clicked one -> (unit vector, degrees it moved)."""
    angle = np.arctan2(direction[2], direction[0])
    snapped = base + np.round((angle - base) / (np.pi / 2)) * np.pi / 2
    return np.array([np.cos(snapped), 0.0, np.sin(snapped)]), float(np.degrees(abs(angle - snapped)))


def wall_offset(tm, inward, near, up, floor_h):
    """d in inward . x = d for the wall through a click, `inward` pointing into the room: the area-weighted
    median of surfaces facing into the room within WALL_SEARCH_M of the click (across the floor), using
    the band WALL_BAND_M above the floor where walls are clear of furniture. Falls back to the click."""
    centres, normals, areas = tm.triangles_center, tm.face_normals, tm.area_faces
    rel = centres - near
    across = np.linalg.norm(rel - np.outer(rel @ up, up), axis=1)
    height = centres @ up - floor_h
    sel = ((np.abs(normals @ up) < 0.2) & (normals @ inward > np.cos(np.radians(20))) & (across < WALL_SEARCH_M)
           & (height > WALL_BAND_M[0]) & (height < WALL_BAND_M[1]))
    if sel.sum() < 10:
        return float(inward @ near), False
    offsets, weights = centres[sel] @ inward, areas[sel]
    order = np.argsort(offsets)
    half = np.searchsorted(np.cumsum(weights[order]), weights.sum() / 2)
    return float(offsets[order][half]), True


def square_up(mesh, corner, along_a, along_b, up):
    """Clicks -> the exact frame: wall directions snapped to the scan's walls, and the corner moved to
    where the two walls and the floor meet. Returns (corner, a point along wall A, one along wall B)."""
    import trimesh

    tm = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
    base = wall_angle(tm)
    (dir_a, moved_a), (dir_b, moved_b) = snap(along_a - corner, base), snap(along_b - corner, base)
    if max(moved_a, moved_b) > WALL_SNAP_MAX_DEG or abs(dir_a @ dir_b) > 0.5:
        print(f"  WARNING: clicks are {moved_a:.0f} and {moved_b:.0f} degrees from the scan's walls, or along the "
              "same wall; using them as clicked")
        return corner, along_a, along_b
    # From the corner the room lies between the two walls, so each wall's inward direction is the other's.
    floor_h = up @ corner
    (d_a, found_a), (d_b, found_b) = (wall_offset(tm, dir_b, along_a, up, floor_h),
                                      wall_offset(tm, dir_a, along_b, up, floor_h))
    exact = np.linalg.solve(np.vstack([dir_b, dir_a, up]), [d_a, d_b, up @ corner])
    print(f"  wall directions snapped by {moved_a:.1f} and {moved_b:.1f} degrees; corner moved "
          f"{np.linalg.norm(exact - corner) * 100:.0f} cm to where the walls meet"
          + ("" if found_a and found_b else " (a wall wasn't found near its click; used the click)"))
    return exact, exact + dir_a, exact + dir_b


def plan_png(path, glb, mesh, T, world_markers, ppm=100):
    """Top-down picture of the room in the world frame, with the origin, the X and Z axes and the
    markers drawn on: what to tape-measure from."""
    import trimesh

    V = transform(T, mesh.vertices)
    world = Mesh(V, mesh.faces)
    lo, hi = V[:, [0, 2]].min(axis=0) - 0.3, V[:, [0, 2]].max(axis=0) + 0.3
    W, H = ((hi - lo) * ppm).astype(int)
    X, Z = np.meshgrid(lo[0] + (np.arange(W) + 0.5) / ppm, lo[1] + (np.arange(H) + 0.5) / ppm)
    origins = np.column_stack([X.ravel(), np.full(X.size, 1.1), Z.ravel()])  # look down from 1.1 m
    t, tri, bary = world.cast_detail(origins, np.tile([0.0, -1.0, 0.0], (len(origins), 1)))
    hit = np.isfinite(t)
    img = np.full((len(origins), 3), 30, np.uint8)
    visual = trimesh.load(glb, force="mesh").visual
    material = getattr(visual, "material", None)
    texture = getattr(material, "baseColorTexture", None) or getattr(material, "image", None)
    if getattr(visual, "uv", None) is not None and texture is not None:
        tex = np.asarray(texture.convert("RGB"))[..., ::-1]  # BGR for OpenCV
        corners = mesh.faces[tri[hit]]
        w1, w2 = bary[hit, 0], bary[hit, 1]
        uv = ((1 - w1 - w2)[:, None] * visual.uv[corners[:, 0]] + w1[:, None] * visual.uv[corners[:, 1]]
              + w2[:, None] * visual.uv[corners[:, 2]])
        rows = np.clip(((1 - uv[:, 1]) * (tex.shape[0] - 1)).astype(int), 0, tex.shape[0] - 1)
        cols = np.clip((uv[:, 0] * (tex.shape[1] - 1)).astype(int), 0, tex.shape[1] - 1)
        img[hit] = tex[rows, cols]
    else:
        img[hit] = np.clip(255 - t[hit] * 150, 60, 255).astype(np.uint8)[:, None]  # higher is brighter
    img = np.ascontiguousarray(img.reshape(H, W, 3))

    def px(p):
        return int((p[0] - lo[0]) * ppm), int((p[2] - lo[1]) * ppm)

    for marker_id, c in world_markers.items():
        cv2.polylines(img, [np.array([px(p) for p in c], np.int32)], True, (255, 0, 255), 2)
        cv2.putText(img, str(marker_id), px(c.mean(axis=0) + [0.15, 0, 0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 0, 255), 2)
    for axis, colour in (("X", (0, 0, 255)), ("Z", (255, 120, 0))):
        tip = np.array([1.0, 0, 0]) if axis == "X" else np.array([0, 0, 1.0])
        cv2.arrowedLine(img, px((0, 0, 0)), px(tip), colour, 3, tipLength=0.15)
        label = px(tip * 0.5 + (np.array([0, 0, 0.25]) if axis == "X" else np.array([0.15, 0, 0])))
        cv2.putText(img, f"+{axis}", label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
    cv2.circle(img, px((0, 0, 0)), 7, (0, 255, 255), -1)
    for i, line in enumerate(("yellow: origin, the corner. Arrows are 1 m.",
                              "tape x from the wall along blue (Z), z from the wall along red (X)")):
        cv2.putText(img, line, (10, H - 30 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(str(path), img)
    print(f"wrote {path}")


def frame_picks(mesh, path, convention_key, repick):
    if path.exists() and not repick:
        saved = load_json(path)
        if saved.get("convention") == convention_key:
            print(f"  reusing the corner and wall clicks in {path} (--repick to redo)")
            return [np.array(saved[k]) for k in ("corner", "along_a", "along_b")]
    pts = pick_points(mesh, 3, "Click 3 points: (1) a room corner, on the floor where two walls meet, "
                               "(2) anywhere on one of those walls, (3) anywhere on the other wall")
    save_json(path, {"convention": convention_key, "corner": pts[0].tolist(),
                     "along_a": pts[1].tolist(), "along_b": pts[2].tolist()})
    return list(pts)


def decimate(vertices, faces, max_tris):
    if len(faces) <= max_tris:
        return vertices, faces
    import open3d as o3d

    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces))
    m = m.simplify_quadric_decimation(target_number_of_triangles=max_tris)
    m.remove_unreferenced_vertices()
    return np.asarray(m.vertices), np.asarray(m.triangles)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scan", type=Path, help="unzipped Polycam raw-data export, or the .zip")
    p.add_argument("--marker-size", type=float, required=True, help="black square measured off the print, metres")
    p.add_argument("--out", type=Path, default=DATA)
    p.add_argument("--max-tris", type=int, default=300_000)
    p.add_argument("--repick", action="store_true", help="redo the corner and wall clicks")
    args = p.parse_args()

    scan = args.scan
    if scan.suffix == ".zip":
        target = scan.with_suffix("")
        if not target.exists():
            zipfile.ZipFile(scan).extractall(target)
        roots = [d.parent for d in target.rglob("mesh_info.json")] + [d.parent for d in target.rglob("keyframes")]
        scan = roots[0] if roots else target

    print("1. loading the export")
    vertices, faces, align, frames = load_polycam(scan)
    print(f"  {len(faces):,} triangles, {len(frames)} keyframes")

    print("2. lining up poses and mesh")
    mesh_name, convention, mesh, depth_err = choose_convention(vertices, faces, align, frames)
    convention_key = f"{mesh_name}|{'c2w' if convention[0] else 'w2c'}|{'arkit' if convention[1] else 'opencv'}"

    print("3. finding markers")
    found = locate_markers(mesh, frames, convention)
    for marker_id, (corners, n, spread) in found.items():
        edges = np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)
        print(f"  marker {marker_id}: {n} sightings, corner spread {spread * 1000:.1f} mm, "
              f"edges {edges.mean() * 100:.1f} cm (measured {args.marker_size * 100:.1f} cm)")
    markers = {i: v for i, v in found.items() if v[1] >= MIN_SIGHTINGS}
    if len(markers) < len(found):
        print(f"  skipped markers seen fewer than {MIN_SIGHTINGS} times: {sorted(set(found) - set(markers))}")

    print("4. world frame")
    corner, along_a, along_b = frame_picks(mesh, args.out / "frame_picks.json", convention_key, args.repick)
    up, corner = fit_floor(mesh, corner)
    corner, along_a, along_b = square_up(mesh, corner, along_a, along_b, up)
    T = world_from_clicks(corner, along_a, along_b, up)

    print("5. writing")
    import trimesh

    world_v, world_f = decimate(transform(T, mesh.vertices), mesh.faces, args.max_tris)
    args.out.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(world_v, world_f, process=False).export(args.out / "room.glb")
    print(f"wrote {args.out / 'room.glb'} ({len(world_f):,} triangles)")

    world_markers = {i: transform(T, c) for i, (c, _, _) in markers.items()}
    save_json(args.out / "markers.json", {
        "dictionary": TAG_DICT,
        "marker_size_m": args.marker_size,
        "markers": {str(i): np.round(c, 4).tolist() for i, c in world_markers.items()},
    })
    plan_png(args.out / "plan.png", find_mesh(scan), mesh, T, world_markers)
    for i, c in world_markers.items():  # markers on the floor or the two origin walls should sit at 0
        planes = {"the floor (y)": c[:, 1], "the Z wall (x)": c[:, 0], "the X wall (z)": c[:, 2]}
        name, coord = min(planes.items(), key=lambda kv: np.abs(kv[1]).mean())
        print(f"  marker {i}: {np.abs(coord).mean() * 100:.1f} cm off {name}")

    warnings = []
    if len(markers) < 3:
        warnings.append(f"only {len(markers)} markers located; the handoff needs >= 3")
    elif plane_rms(np.concatenate(list(world_markers.values()))) < 0.05:
        warnings.append("every marker is on one plane; put at least one on a wall (or the floor)")
    report = {"convention": convention_key, "depth_median_m": depth_err, "markers": {}}
    for i, (c, n, spread) in markers.items():
        ratio = float(np.linalg.norm(c - np.roll(c, -1, axis=0), axis=1).mean() / args.marker_size)
        report["markers"][str(i)] = {"sightings": n, "spread_m": round(spread, 4), "edge_ratio": round(ratio, 4)}
        if abs(ratio - 1) > SCALE_TOL:
            warnings.append(f"marker {i} measures {ratio:.1%} of its printed size: scale or a bad sighting")
        if spread > SPREAD_WARN_M:
            warnings.append(f"marker {i} corners spread {spread * 1000:.0f} mm across sightings")
    report["warnings"] = warnings
    save_json(args.out / "scan_report.json", report)

    print("\n" + ("\n".join(f"WARNING: {w}" for w in warnings) if warnings else "all scan checks passed"))


if __name__ == "__main__":
    main()
