#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["open3d==0.19.0", "trimesh", "pillow", "opencv-python>=4.8"]
# ///
"""The room tools against a synthetic room, so they're checked before any real scan exists.

    uv run tests/test_room_tools.py

Builds a fake Polycam raw-data export (a box room with markers on the floor and walls, keyframes
rendered by ray casting, LiDAR depth, camera JSONs in Polycam's format) and runs scan_to_room's
steps on it. Then registers a fake webcam and calibrates a fake lens.
"""

import json
import sys
import tempfile
import traceback
import zipfile
from pathlib import Path

import cv2
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import calibrate_intrinsics as ci  # noqa: E402
import register_camera  # noqa: E402
import scan_to_room as s2r  # noqa: E402
from printed import TAG_DICT  # noqa: E402
from roomlib import (  # noqa: E402
    FLIP_YZ, Mesh, camera_center, floor_hit, pixel_rays, plane_rms, project, save_json, solve_pose,
    transform,
)

ROOM = (4.0, 2.5, 3.0)  # x, y (height), z
S = 0.18
# marker id -> (top-left corner, step along its top edge, step down its left edge), room coords
MARKERS = {
    0: ((1.0, 0.0, 1.0), (S, 0, 0), (0, 0, S)),  # floor
    1: ((2.0, 1.4, 0.0), (S, 0, 0), (0, -S, 0)),  # back wall, z = 0
    2: ((0.0, 1.3, 1.8), (0, 0, -S), (0, -S, 0)),  # left wall, x = 0
    3: ((3.0, 0.0, 2.2), (S, 0, 0), (0, 0, S)),  # floor, far
}
TRUTH = {i: np.array([tl, np.add(tl, r), np.add(np.add(tl, r), d), np.add(tl, d)], float)
         for i, (tl, r, d) in MARKERS.items()}
PAPER = 0.25  # white margin around each marker, as a fraction of its size
BITMAPS = {i: cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, TAG_DICT)), i, 120)
           for i in MARKERS}


def rot_y(deg, shift):
    a = np.radians(deg)
    T = np.eye(4)
    T[:3, :3] = [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]
    T[:3, 3] = shift
    return T


POSE_FRAME = rot_y(25, (0.7, -1.3, 0.4))  # room -> ARKit session frame (gravity-aligned, arbitrary heading)
ALIGN = rot_y(-40, (-0.2, 0.0, 1.1))  # session frame -> Polycam's axis-aligned raw.glb frame


def plane_grid(origin, u, v, normal, step=0.1):
    nu, nv = max(1, round(np.linalg.norm(u) / step)), max(1, round(np.linalg.norm(v) / step))
    i, j = np.meshgrid(np.arange(nu + 1), np.arange(nv + 1), indexing="ij")
    verts = (np.array(origin) + (i[..., None] / nu) * np.array(u) + (j[..., None] / nv) * np.array(v)).reshape(-1, 3)
    at = lambda a, b: a * (nv + 1) + b  # noqa: E731
    faces = np.array([f for a in range(nu) for b in range(nv)
                      for f in ([at(a, b), at(a + 1, b), at(a + 1, b + 1)], [at(a, b), at(a + 1, b + 1), at(a, b + 1)])])
    v0, v1, v2 = verts[faces[0]]
    if np.cross(v1 - v0, v2 - v0) @ normal < 0:
        faces = faces[:, ::-1]
    return verts, faces


def room_mesh():
    W, H, D = ROOM
    planes = [((0, 0, 0), (W, 0, 0), (0, 0, D), (0, 1, 0)), ((0, H, 0), (W, 0, 0), (0, 0, D), (0, -1, 0)),
              ((0, 0, 0), (W, 0, 0), (0, H, 0), (0, 0, 1)), ((0, 0, D), (W, 0, 0), (0, H, 0), (0, 0, -1)),
              ((0, 0, 0), (0, 0, D), (0, H, 0), (1, 0, 0)), ((W, 0, 0), (0, 0, D), (0, H, 0), (-1, 0, 0))]
    verts, faces = [], []
    for plane in planes:
        v, f = plane_grid(*plane)
        faces.append(f + sum(len(x) for x in verts))
        verts.append(v)
    return np.concatenate(verts), np.concatenate(faces)


def look_at(eye, target):
    f = np.subtract(target, eye) / np.linalg.norm(np.subtract(target, eye))
    r = np.cross(f, (0, 1, 0))
    r /= np.linalg.norm(r)
    R = np.vstack([r, np.cross(f, r), f])  # OpenCV: x right, y down, z forward
    return R, -R @ np.asarray(eye, float)


def render(mesh, K, R, t, size, ss=2):
    """Grey room, white paper, black-and-white markers. Supersampled, then averaged down."""
    w, h = size
    Ks = K.copy()
    Ks[0, 0], Ks[1, 1] = K[0, 0] * ss, K[1, 1] * ss
    Ks[0, 2], Ks[1, 2] = (K[0, 2] + 0.5) * ss - 0.5, (K[1, 2] + 0.5) * ss - 0.5
    vs, us = np.mgrid[0:h * ss, 0:w * ss]
    p = mesh.hit_points(*pixel_rays(Ks, None, R, t, np.column_stack([us.ravel(), vs.ravel()])))
    img = np.full(len(p), 140.0)
    with np.errstate(invalid="ignore"):
        img[np.abs(p[:, 1]) < 1e-3] = 100
        for i, c in TRUTH.items():
            right, down = c[1] - c[0], c[3] - c[0]
            normal = np.cross(right, down) / S ** 2
            rel = p - c[0]
            on = np.abs(rel @ normal) < 1e-4
            a, b = rel @ right / S ** 2, rel @ down / S ** 2
            img[on & (a > -PAPER) & (a < 1 + PAPER) & (b > -PAPER) & (b < 1 + PAPER)] = 235
            inside = on & (a >= 0) & (a < 1) & (b >= 0) & (b < 1)
            n = BITMAPS[i].shape[0]
            img[inside] = BITMAPS[i][(b[inside] * n).astype(int), (a[inside] * n).astype(int)]
    img = img.reshape(h * ss, w * ss).astype(np.float32)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA).astype(np.uint8)


def render_depth(mesh, K, R, t, size):
    """LiDAR-style depth along the camera's z axis, uint16 millimetres."""
    w, h = size
    vs, us = np.mgrid[0:h, 0:w]
    o, d = pixel_rays(K, None, R, t, np.column_stack([us.ravel(), vs.ravel()]))
    z = mesh.cast(o, d) * (d @ R.T)[:, 2]
    mm = np.where(np.isfinite(z), np.round(z * 1000), 0)
    return mm.reshape(h, w).astype(np.uint16)


def keyframe_views():
    """A few viewpoints per marker, ~1.2 m out from it, plus two overviews."""
    views = []
    for c in TRUTH.values():
        centre = c.mean(axis=0)
        normal = -np.cross(c[1] - c[0], c[3] - c[0]) / S ** 2  # out of the surface, into the room
        side = np.cross(normal, (0, 1, 0)) if abs(normal[1]) < 0.9 else np.array([1.0, 0, 0])
        for offset in (-0.4, 0.0, 0.4):
            eye = centre + 1.1 * normal + offset * side + (0.25 if abs(normal[1]) < 0.9 else 0) * np.array([0, 1, 0])
            if abs(normal[1]) > 0.9:  # floor marker: look down at it from the side
                eye += np.array([0, 0, 0.6])
            views.append((np.clip(eye, 0.2, np.array(ROOM) - 0.2), centre))
    views += [((3.6, 2.0, 2.6), (1.0, 0.5, 0.5)), ((0.4, 1.8, 2.7), (3.0, 0.5, 0.5))]
    return views


def make_export(folder, size=(960, 720), depth_size=(192, 144), mesh_name="raw.glb"):
    """A Polycam raw-data export of the synthetic room: what the real phone will hand us."""
    v, f = room_mesh()
    mesh = Mesh(v, f)
    trimesh.Trimesh(transform(ALIGN @ POSE_FRAME, v), f, process=False).export(folder / mesh_name)
    save_json(folder / "mesh_info.json", {"alignmentTransform": ALIGN.T.ravel().tolist()})  # column-major
    kf = folder / "keyframes"
    for d in ("corrected_images", "corrected_cameras", "depth", "confidence"):
        (kf / d).mkdir(parents=True)

    K = np.array([[760.0, 0, 480], [0, 760.0, 360], [0, 0, 1]])
    Kd = s2r.scaled_K(K, size, depth_size)
    for k, (eye, target) in enumerate(keyframe_views()):
        R, t = look_at(eye, target)
        name = f"{k:05d}"
        cv2.imwrite(str(kf / "corrected_images" / f"{name}.jpg"), render(mesh, K, R, t, size),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(kf / "depth" / f"{name}.png"), render_depth(mesh, Kd, R, t, depth_size))
        cv2.imwrite(str(kf / "confidence" / f"{name}.png"), np.full(depth_size[::-1], 255, np.uint8))
        c2w = np.eye(4)
        c2w[:3, :3], c2w[:3, 3] = R.T @ FLIP_YZ, camera_center(R, t)  # ARKit camera axes
        T = POSE_FRAME @ c2w
        save_json(kf / "corrected_cameras" / f"{name}.json", {
            "fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2],
            "width": size[0], "height": size[1], "blur_score": 200.0,
            **{f"t_{i}{j}": T[i, j] for i in range(3) for j in range(4)},
        })


def test_scan_to_room():
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        make_export(folder)
        vertices, faces, align, frames = s2r.load_polycam(folder)
        mesh_name, convention, mesh, err = s2r.choose_convention(vertices, faces, align, frames)
        assert (mesh_name, convention) == ("inverse alignment", (True, True)), (mesh_name, convention)
        assert err < 0.005, err

        found = s2r.locate_markers(mesh, frames, convention)
        assert set(found) == set(MARKERS), sorted(found)
        for i, (corners, n, spread) in found.items():
            off = np.linalg.norm(corners - transform(POSE_FRAME, TRUTH[i]), axis=1).max()
            assert n >= 3 and off < 0.005, (i, n, off)

        # No clicks: floor at y = 0, axes along the walls, origin at the floor's corner. For this box
        # room that is exactly the room's own frame.
        T = s2r.world_frame(mesh)
        assert np.isclose(np.linalg.det(T[:3, :3]), 1), "not right-handed"
        for i, (corners, _, _) in found.items():
            off = np.linalg.norm(transform(T, corners) - TRUTH[i], axis=1).max()
            assert off < 0.005, (i, off)


def test_gltf_mesh_in_a_zip():
    """Newer exports name the mesh raw.gltf, and the export usually arrives zipped."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        export, out = tmp / "export" / "Scan 2026-10-24", tmp / "room"
        export.mkdir(parents=True)
        make_export(export, mesh_name="raw.gltf")
        with zipfile.ZipFile(tmp / "scan.zip", "w") as z:
            for p in export.rglob("*"):
                z.write(p, p.relative_to(tmp / "export"))
        sys.argv = ["scan_to_room.py", str(tmp / "scan.zip"), "--out", str(out)]
        s2r.main()
        saved = json.loads((out / "markers.json").read_text())
        markers = saved["markers"]
        assert sorted(markers) == ["0", "1", "2", "3"], sorted(markers)
        assert abs(saved["marker_size_m"] - S) < 0.003, saved["marker_size_m"]  # measured from the scan
        assert np.linalg.norm(np.array(markers["2"]) - TRUTH[2], axis=1).max() < 0.005
        room = trimesh.load(out / "room.glb", force="mesh")
        assert abs(room.bounds[0][1]) < 0.005, room.bounds  # floor at y = 0


def test_wrong_readings_rejected():
    """A camera->world pose read as world->camera, or with the wrong axes, must not line up."""
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        make_export(folder)
        vertices, faces, align, frames = s2r.load_polycam(folder)
        mesh = Mesh(transform(np.linalg.inv(align), vertices), faces)
        f = frames[0]
        for reading in ((True, False), (False, True), (False, False)):
            errs, hits = s2r.depth_errors(mesh, f, *s2r.pose(f.T, *reading), along_z=True)
            assert hits < 0.5 or np.median(errs) > 0.05, reading


def test_register_camera():
    K = np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]])  # ~70 degrees wide, like the Brio
    dist = np.array([-0.08, 0.02, 0, 0, 0])
    R, t = look_at((3.85, 2.35, 0.15), (1.0, 0.0, 2.0))  # high in a corner, all four markers in view
    obj = np.concatenate(list(TRUTH.values()))
    img = project(K, dist, R, t, obj) + np.random.default_rng(1).normal(0, 0.3, (len(obj), 2))
    assert ((img >= 0) & (img < (1280, 720))).all(), "a marker is out of the fake webcam's view"

    R2, t2, rms, inliers = solve_pose(obj, img, K, dist)
    centre_off = np.linalg.norm(camera_center(R2, t2) - camera_center(R, t))
    angle = np.degrees(np.arccos(np.clip((np.trace(R2 @ R.T) - 1) / 2, -1, 1)))
    assert centre_off < 0.01 and angle < 0.5 and rms < 1 and len(inliers) == len(obj), (centre_off, angle, rms)

    floor_point = np.array([2.0, 0.0, 1.5])
    hit = floor_hit(K, dist, R, t, project(K, dist, R, t, [floor_point]))[0]
    assert np.linalg.norm(hit - floor_point) < 0.001, hit


def test_coplanar_detection():
    assert plane_rms(np.concatenate([TRUTH[0], TRUTH[3]])) < 0.05  # both on the floor
    assert plane_rms(np.concatenate(list(TRUTH.values()))) > 0.05  # floor + walls


def test_setup_flow():
    """setup_room's path: export zip dropped in the room folder -> scan processed -> webcam registered."""
    with tempfile.TemporaryDirectory() as tmp:
        room, export = Path(tmp) / "room", Path(tmp) / "export"
        room.mkdir()
        export.mkdir()
        make_export(export)
        with zipfile.ZipFile(room / "Scan.zip", "w") as z:
            for p in export.rglob("*"):
                z.write(p, p.relative_to(export))
        report = s2r.process_scan(s2r.prepare_scan(room), room)
        assert sorted(report["markers"]) == ["0", "1", "2", "3"], report["markers"]
        assert not (room / "Scan.zip").exists(), "the zip should move in with its scan"
        assert s2r.prepare_scan(room) == room / "scan_raw", "a second run should reuse the unpacked scan"

        K = np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]])
        R, t = look_at((3.85, 2.35, 0.15), (1.0, 0.0, 2.0))
        frame = cv2.cvtColor(render(Mesh(*room_mesh()), K, R, t, (1280, 720)), cv2.COLOR_GRAY2BGR)
        lens = {"K": K.tolist(), "dist": [0.0] * 5, "width": 1280, "height": 720}
        camera = register_camera.register([frame], lens, room)
        off = np.linalg.norm(camera["center"] - camera_center(R, t))
        assert off < 0.02 and camera["rms_px"] < 1, (off, camera["rms_px"])


def test_calibrate_intrinsics():
    board = ci.make_board()
    px_per_m = 100 / (ci.BOARD_SQUARE_CM / 100)
    cols, rows = ci.BOARD_SQUARES
    pad = 40
    art = cv2.copyMakeBorder(board.generateImage((cols * 100, rows * 100)), pad, pad, pad, pad,
                             cv2.BORDER_CONSTANT, value=255)
    K = np.array([[1000.0, 0, 640], [0, 1000.0, 360], [0, 0, 1]])
    # The board's origin is the outer edge of its first pixel, half a pixel before that pixel's centre.
    edge = (0.5 - pad) / px_per_m
    pixel_to_board = np.array([[1 / px_per_m, 0, edge], [0, 1 / px_per_m, edge], [0, 0, 1]])
    centre = np.array([cols, rows, 0]) * ci.BOARD_SQUARE_CM / 200
    rng = np.random.default_rng(2)
    detector = cv2.aruco.CharucoDetector(board)
    views = []
    while len(views) < 15:
        R = cv2.Rodrigues(np.r_[rng.uniform(-0.6, 0.6, 2), rng.uniform(-0.3, 0.3)])[0]
        z = rng.uniform(0.45, 0.8)
        pos = np.r_[rng.uniform(-0.25, 0.25) * z, rng.uniform(-0.12, 0.12) * z, z]
        t = pos - R @ centre
        H = K @ np.column_stack([R[:, 0], R[:, 1], t]) @ pixel_to_board
        frame = cv2.warpPerspective(art, H, (1280, 720), borderValue=90)
        if (found := ci.find_board(detector, frame)) is not None:
            views.append(found)
    rms, K2, dist, used = ci.calibrate(board, views, (1280, 720))
    assert rms < 0.5 and np.abs(K2 - K).max() < 5, (rms, K2)


if __name__ == "__main__":
    failed = 0
    for name, fn in [(n, f) for n, f in globals().items() if n.startswith("test_")]:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception:
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    sys.exit(1 if failed else 0)
