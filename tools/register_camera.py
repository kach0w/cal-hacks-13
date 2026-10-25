#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["open3d==0.19.0", "trimesh", "pillow", "opencv-python>=4.8"]
# ///
"""Work out where the webcam is in the room -> data/room/camera.json + camera_ref.jpg
(handoffAB.md deliverables 3-4).

    uv run tools/register_camera.py --camera brio
    uv run tools/register_camera.py --image frame.jpg      # from a saved 1280x720 frame

Needs intrinsics.json (calibrate_intrinsics.py) and markers.json (scan_to_room.py). Watches the
markers for ~2 s, matches their corners to markers.json and solves the camera's pose.
Re-run it whenever the camera is bumped.

Also writes camera_overlay.jpg: the markers, the mesh and the check points drawn where the
solved pose puts them. They should sit on the real ones.
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from roomlib import (
    DATA, Mesh, camera_center, detect_markers, load_json, marker_detector, plane_rms, project,
    save_json, solve_pose,
)
from webcam import open_camera

SECONDS = 2.0
RMS_OK_PX = 5.0  # handoff acceptance
LENS_OK_M = 0.05  # handoff acceptance
OVERLAY_VERTICES = 20_000


def capture(args):
    if args.image:
        frame = cv2.imread(str(args.image))
        if frame is None:
            sys.exit(f"can't read {args.image}")
        return [frame]
    cap = open_camera(args.camera, exact=True)
    for _ in range(10):  # let the stream settle
        cap.read()
    frames, end = [], time.time() + SECONDS
    while time.time() < end:
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return frames


def median_corners(frames, wanted):
    """{id: 4x2 median corners} for markers seen in at least a third of the frames."""
    detector = marker_detector()
    seen = {}
    for frame in frames:
        for marker_id, c in detect_markers(detector, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)).items():
            if marker_id in wanted:
                seen.setdefault(marker_id, []).append(c)
    need = max(1, len(frames) // 3)
    return {i: np.median(np.array(cs), axis=0) for i, cs in sorted(seen.items()) if len(cs) >= need}


def draw_overlay(frame, K, dist, R, t, detected, world, rms, data):
    view = frame.copy()
    h, w = view.shape[:2]
    mesh_path = data / "room.glb"
    if mesh_path.exists():
        mesh = Mesh.load(mesh_path)
        rng = np.random.default_rng(0)
        pts = mesh.vertices[rng.choice(len(mesh.vertices), min(OVERLAY_VERTICES, len(mesh.vertices)), replace=False)]
        in_front = (pts @ np.asarray(R).T + t)[:, 2] > 0.05
        pts = pts[in_front]
        uv = project(K, dist, R, t, pts)
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        shown = inside.copy()
        shown[inside] = mesh.visible(camera_center(R, t), pts[inside])
        for u, v in uv[shown].astype(int):
            cv2.circle(view, (u, v), 2, (255, 160, 0), -1)
    for marker_id, c in detected.items():
        for u, v in c:
            cv2.circle(view, (int(u), int(v)), 6, (0, 255, 0), 2)
    for marker_id, corners in world.items():
        uv = project(K, dist, R, t, corners)
        for u, v in uv:
            cv2.drawMarker(view, (int(u), int(v)), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
        cv2.putText(view, str(marker_id), tuple(uv.mean(axis=0).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    checks = data / "checks.json"
    if checks.exists():
        for name, xyz in load_json(checks).get("test_points", {}).items():
            u, v = project(K, dist, R, t, [xyz])[0].astype(int)
            cv2.circle(view, (u, v), 9, (0, 255, 255), 2)
            cv2.putText(view, name, (u + 12, v), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(view, f"green: detected  red: markers.json  blue: mesh  yellow: checks   rms {rms:.2f}px",
                (12, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return view


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", default="0", help="name (substring is fine) or index")
    p.add_argument("--image", type=Path, help="register from a saved frame instead of the live camera")
    p.add_argument("--min-markers", type=int, default=3, help="handoff: at least 3")
    p.add_argument("--data", type=Path, default=DATA)
    args = p.parse_args()
    data = args.data

    intr = load_json(data / "intrinsics.json")
    markers = load_json(data / "markers.json")
    K, dist = np.array(intr["K"], np.float64), np.array(intr["dist"], np.float64)
    world = {int(i): np.array(c, np.float64) for i, c in markers["markers"].items()}

    frames = capture(args)
    h, w = frames[0].shape[:2]
    if (w, h) != (intr["width"], intr["height"]):
        sys.exit(f"frames are {w}x{h} but the lens was calibrated at {intr['width']}x{intr['height']}")

    detected = median_corners(frames, set(world))
    print(f"markers seen: {sorted(detected)} (in markers.json: {sorted(world)})")
    if len(detected) < args.min_markers:
        sys.exit(f"need at least {args.min_markers} markers in view, saw {len(detected)}")

    used = sorted(detected)
    obj = np.concatenate([world[i] for i in used])
    img = np.concatenate([detected[i] for i in used])
    R, t, rms, inliers = solve_pose(obj, img, K, dist)
    center = camera_center(R, t)
    if len(inliers) < len(obj):
        print(f"WARNING: {len(obj) - len(inliers)} of {len(obj)} corners disagreed and were dropped")
    if plane_rms(obj) < 0.05:
        print("WARNING: every marker in view is on one plane; the pose is much weaker than with floor + wall")

    save_json(data / "camera.json", {
        "width": w, "height": h,
        "K": K.tolist(), "dist": dist.tolist(),
        "R": np.round(R, 7).tolist(), "t": np.round(t, 5).tolist(),
        "camera_center": np.round(center, 4).tolist(),
        "rms_px": round(rms, 3),
        "markers_used": used,
        "registered_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    })
    ref = frames[len(frames) // 2]
    cv2.imwrite(str(data / "camera_ref.jpg"), ref, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(data / "camera_overlay.jpg"), draw_overlay(ref, K, dist, R, t, detected, world, rms, data))
    print(f"wrote {data / 'camera_ref.jpg'} and camera_overlay.jpg")

    print(f"\nreprojection error {rms:.2f} px ({'ok' if rms < RMS_OK_PX else f'FAIL, need < {RMS_OK_PX}'})")
    print(f"camera at {np.round(center, 3).tolist()} m")
    checks = data / "checks.json"
    if checks.exists() and (lens := load_json(checks).get("lens_measured")):
        off = np.linalg.norm(center - np.array(lens))
        print(f"tape says {lens}: off by {off * 100:.1f} cm ({'ok' if off < LENS_OK_M else 'FAIL, need < 5 cm'})")


if __name__ == "__main__":
    main()
