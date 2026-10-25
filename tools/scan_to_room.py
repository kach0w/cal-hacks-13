#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["open3d==0.19.0", "trimesh", "pillow", "opencv-python>=4.8"]
# ///
"""Polycam scan -> data/room/room.glb + markers.json + plan.png. Usually run by setup_room.py.

    uv run tools/scan_to_room.py                 # the export zip(s) dropped into data/room/
    uv run tools/scan_to_room.py path/to/export  # an unzipped export, or a zip

Scan in Polycam's LiDAR mode with Developer mode on; export Raw data (and glTF if the raw zip
has no mesh in it). No clicks or measurements: everything comes from the scan.

  1. load the mesh and the keyframes (images, phone poses, LiDAR depth)
  2. check Polycam's poses and mesh line up, by comparing mesh depth with LiDAR depth
  3. find the markers in the keyframes and cast their corners onto the mesh
  4. world frame: floor at y = 0, X and Z along the walls, origin at the corner of the floor's
     outline, so the room is at x, z >= 0
  5. write room.glb (transforms baked, <= --max-tris), markers.json, plan.png, scan_report.json
"""

import argparse
import shutil
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path

import cv2
import numpy as np

from printed import TAG_CM, TAG_DICT
from roomlib import (
    DATA, FLIP_YZ, Mesh, detect_markers, load_json, marker_detector, pixel_rays, plane_rms, save_json,
    transform,
)

DEPTH_OK_M = 0.02  # median mesh-vs-LiDAR depth error that counts as lined up
LIDAR_MAX_M = 5.0  # beyond this, LiDAR depth is a guess
CHECK_FRAMES = 12
MIN_MARKER_PX = 25  # smaller detections are too coarse to place a corner
MIN_SIGHTINGS = 3
SPREAD_WARN_M = 0.01
FLOOR_MARKER_M = 0.05  # a marker this close to y = 0 is on the floor
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
                 "Fall back to clicking marker corners (CAMERA.md E1).")
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


def wall_angle(tm):
    """The room's wall direction in the floor plane (radians), from every vertical surface's normal,
    weighted by area. Walls meet at right angles, so this plus multiples of 90 degrees covers them all.
    Polycam's frames are gravity-aligned, so vertical means a normal with no Y."""
    n, w = tm.face_normals, tm.area_faces
    vertical = np.abs(n[:, 1]) < 0.2
    theta = np.arctan2(n[vertical, 2], n[vertical, 0])
    return 0.25 * np.arctan2((w[vertical] * np.sin(4 * theta)).sum(), (w[vertical] * np.cos(4 * theta)).sum())


def world_frame(mesh):
    """The world frame from the scan alone -> 4x4 transform. Floor at y = 0 (the lowest height holding a
    good share of the upward-facing area: desks and seats are above it, stray bits below it are small),
    X and Z along the walls, origin at the corner of the floor's outline so the room is at x, z >= 0."""
    import open3d as o3d
    import trimesh

    tm = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
    normals, centres, areas = tm.face_normals, tm.triangles_center, tm.area_faces
    upward = normals[:, 1] > 0.9  # Polycam's frames are gravity-aligned, Y up
    heights = centres[upward, 1]
    bins = np.arange(heights.min(), heights.max() + 0.04, 0.02)
    hist, _ = np.histogram(heights, bins, weights=areas[upward])
    floor_h = bins[np.argmax(hist >= 0.25 * hist.max())] + 0.01
    flat = upward & (np.abs(centres[:, 1] - floor_h) < 0.05)

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(centres[flat]))
    (a, b, c, _), _ = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    y = np.array([a, b, c]) / np.linalg.norm([a, b, c])
    y = y if y[1] > 0 else -y
    theta = wall_angle(tm)
    x = np.array([np.cos(theta), 0.0, np.sin(theta)])
    x -= (x @ y) * y
    x /= np.linalg.norm(x)
    R = np.vstack([x, y, np.cross(x, y)])  # right-handed: X x Y = Z

    floor = tm.vertices[np.unique(tm.faces[flat])] @ R.T
    origin = [np.percentile(floor[:, 0], 0.5), np.median(floor[:, 1]), np.percentile(floor[:, 2], 0.5)]
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, -np.array(origin)
    return T


def plan_png(path, glb, mesh, T, world_markers, ppm=100):
    """Top-down picture of the room in the world frame, with the origin, the X and Z axes and the
    markers drawn on."""
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
    cv2.putText(img, "yellow: origin (floor, y = 0). Arrows are 1 m. Magenta: markers.", (10, H - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(str(path), img)
    print(f"wrote {path}")


def decimate(vertices, faces, max_tris):
    if len(faces) <= max_tris:
        return vertices, faces
    import open3d as o3d

    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces))
    m = m.simplify_quadric_decimation(target_number_of_triangles=max_tris)
    m.remove_unreferenced_vertices()
    return np.asarray(m.vertices), np.asarray(m.triangles)


def prepare_scan(path):
    """The unzipped export to use. Zips found in `path` (Polycam's raw-data zip, plus its glTF zip when
    the raw one has no mesh) are a new scan: they're unpacked into scan_raw/, an older scan_raw/ is kept
    under a timestamped name, and the zips move in with their scan so they're not unpacked twice."""
    path = Path(path)
    if path.is_dir() and (path / "keyframes").exists():
        return path
    zips = [path] if path.suffix == ".zip" else sorted(path.glob("*.zip"))
    target = (path.parent if path.suffix == ".zip" else path) / "scan_raw"
    if zips:
        if target.exists():
            target.rename(target.with_name(f"scan_raw_{datetime.now():%Y%m%d_%H%M%S}"))
        target.mkdir(parents=True)
        for z in zips:
            zipfile.ZipFile(z).extractall(target)
            shutil.move(str(z), target / z.name)
        print(f"  unpacked {', '.join(z.name for z in zips)} into {target}")
    roots = sorted(p.parent for p in target.rglob("keyframes")) if target.exists() else []
    if not roots:
        sys.exit(f"no scan in {path}: drop Polycam's Raw-data zip (and its glTF zip) there")
    scan = roots[0]
    if not any(p.suffix.lower() in {".glb", ".gltf"} for p in scan.iterdir()):
        mesh = next(target.rglob("*.glb"), None) or next(target.rglob("*.gltf"), None)
        if mesh is not None:
            shutil.move(str(mesh), scan / mesh.name)
    return scan


def process_scan(scan, out=DATA, marker_size=None, max_tris=300_000):
    """Steps 1-5 -> a summary dict for setup_room.py."""
    import trimesh

    print("1. loading the scan")
    vertices, faces, align, frames = load_polycam(scan)
    print(f"  {len(faces):,} triangles, {len(frames)} keyframes")

    print("2. checking poses and mesh line up")
    _, convention, mesh, depth_err = choose_convention(vertices, faces, align, frames)

    print("3. finding markers")
    found = locate_markers(mesh, frames, convention)
    markers = {i: v for i, v in found.items() if v[1] >= MIN_SIGHTINGS}
    if len(markers) < len(found):
        print(f"  seen too few times to use: {sorted(set(found) - set(markers))}")
    edges = [np.linalg.norm(c - np.roll(c, -1, axis=0), axis=1).mean() for c, _, _ in markers.values()]
    measured = float(np.median(edges)) if edges else TAG_CM / 100
    marker_size = marker_size or measured
    for i, (c, n, spread) in markers.items():
        print(f"  marker {i}: {n} sightings, corners agree to {spread * 1000:.0f} mm")
    if edges and abs(measured / (TAG_CM / 100) - 1) > 0.03:
        print(f"  markers measure {measured * 100:.1f} cm in the scan, printed at {TAG_CM:.0f} cm: the printer "
              "scaled them. That's fine; the scan's size is what's used")

    print("4. world frame")
    T = world_frame(mesh)
    world_markers = {i: transform(T, c) for i, (c, _, _) in markers.items()}
    for i, c in world_markers.items():
        height = c[:, 1].mean()
        where = f"on the floor ({abs(height) * 100:.1f} cm from y = 0)" if abs(height) < FLOOR_MARKER_M \
            else f"{height:.2f} m up"
        print(f"  marker {i}: {where}")

    print("5. writing")
    world_v, world_f = decimate(transform(T, mesh.vertices), mesh.faces, max_tris)
    out.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(world_v, world_f, process=False).export(out / "room.glb")
    print(f"wrote {out / 'room.glb'} ({len(world_f):,} triangles)")
    save_json(out / "markers.json", {
        "dictionary": TAG_DICT,
        "marker_size_m": round(marker_size, 4),
        "markers": {str(i): np.round(c, 4).tolist() for i, c in world_markers.items()},
    })
    plan_png(out / "plan.png", find_mesh(scan), mesh, T, world_markers)

    warnings = []
    if len(markers) < 3:
        warnings.append(f"only {len(markers)} markers found in the scan; the camera needs at least 3")
    elif plane_rms(np.concatenate(list(world_markers.values()))) < 0.05:
        warnings.append("every marker is on one plane; put at least one on a wall and one on the floor")
    for i, (_, _, spread) in markers.items():
        if spread > SPREAD_WARN_M:
            warnings.append(f"marker {i}'s corners only agree to {spread * 1000:.0f} mm across the scan")
    report = {
        "depth_median_m": depth_err,
        "marker_size_m": round(marker_size, 4),
        "markers": {str(i): {"sightings": n, "spread_m": round(spread, 4)} for i, (_, n, spread) in markers.items()},
        "warnings": warnings,
    }
    save_json(out / "scan_report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scan", nargs="?", type=Path, default=DATA,
                   help="folder with the export zip(s) (default data/room/), an unzipped export, or a zip")
    p.add_argument("--marker-size", type=float, help="black square in metres; default: measured from the scan")
    p.add_argument("--out", type=Path, default=DATA)
    p.add_argument("--max-tris", type=int, default=300_000)
    args = p.parse_args()
    report = process_scan(prepare_scan(args.scan), args.out, args.marker_size, args.max_tris)
    print("\n" + ("\n".join(f"WARNING: {w}" for w in report["warnings"]) or "all scan checks passed"))


if __name__ == "__main__":
    main()
