#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["open3d==0.19.0", "trimesh", "pillow", "opencv-python>=4.8"]
# ///
"""Work out where the webcam is in the room -> data/room/camera.json + camera_ref.jpg. Usually run by
setup_room.py; run it alone to re-register after the webcam is bumped.

    uv run tools/register_camera.py --camera brio
    uv run tools/register_camera.py --image frame.jpg --lens data/cameras/brio-101.json

Needs the webcam's lens file (data/cameras/, from calibrate_intrinsics.py) and markers.json
(scan_to_room.py). Watches the markers for ~2 s and solves the webcam's pose from their corners.

Also writes camera_overlay.jpg: the markers and the scan drawn where the solved pose puts them.
They should sit on the real ones.
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from calibrate_intrinsics import camera_file
from roomlib import (
    DATA, Mesh, camera_center, detect_markers, load_json, marker_detector, plane_rms, project,
    save_json, solve_pose,
)
from webcam import open_camera, resolve_camera

SECONDS = 2.0
RMS_OK_PX = 5.0
MIN_MARKERS = 3
OVERLAY_VERTICES = 20_000


def capture(camera):
    cap = open_camera(camera, exact=True)
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
        pts = pts[(pts @ np.asarray(R).T + t)[:, 2] > 0.05]
        uv = project(K, dist, R, t, pts)
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        shown = inside.copy()
        shown[inside] = mesh.visible(camera_center(R, t), pts[inside])
        for u, v in uv[shown].astype(int):
            cv2.circle(view, (u, v), 2, (255, 160, 0), -1)
    for c in detected.values():
        for u, v in c:
            cv2.circle(view, (int(u), int(v)), 6, (0, 255, 0), 2)
    for marker_id, corners in world.items():
        uv = project(K, dist, R, t, corners)
        for u, v in uv:
            cv2.drawMarker(view, (int(u), int(v)), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
        cv2.putText(view, str(marker_id), tuple(uv.mean(axis=0).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.putText(view, f"green: markers seen  red: where the scan says they are  blue: the scan   fit {rms:.2f}px",
                (12, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return view


def register(frames, lens, data=DATA, min_markers=MIN_MARKERS):
    """Solve the webcam's pose from frames of the markers; write camera.json and the images -> summary."""
    K, dist = np.array(lens["K"], np.float64), np.array(lens["dist"], np.float64)
    world = {int(i): np.array(c, np.float64) for i, c in load_json(data / "markers.json")["markers"].items()}
    h, w = frames[0].shape[:2]
    if (w, h) != (lens["width"], lens["height"]):
        sys.exit(f"frames are {w}x{h} but the lens was calibrated at {lens['width']}x{lens['height']}")

    detected = median_corners(frames, set(world))
    print(f"  markers in view: {sorted(detected)} (in the scan: {sorted(world)})")
    if len(detected) < min_markers:
        sys.exit(f"the webcam needs at least {min_markers} markers in view and sees {len(detected)}. Move or add "
                 "markers where it sees them face-on (then rescan, so the scan knows where they are)")

    used = sorted(detected)
    obj = np.concatenate([world[i] for i in used])
    R, t, rms, inliers = solve_pose(obj, np.concatenate([detected[i] for i in used]), K, dist)
    center = camera_center(R, t)
    warnings = []
    if len(inliers) < len(obj):
        warnings.append(f"{len(obj) - len(inliers)} of {len(obj)} marker corners disagreed and were left out")
    if plane_rms(obj) < 0.05:
        warnings.append("every marker in view is on one plane; the pose is much weaker than with floor + wall")
    if rms > RMS_OK_PX:
        warnings.append(f"fit {rms:.1f} px is poor: a marker may have moved since the scan")
    if len(used) == MIN_MARKERS:
        warnings.append(f"only {MIN_MARKERS} markers in view; one or two more would make positions noticeably tighter")

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
    return {"center": center, "rms_px": rms, "markers_used": used, "warnings": warnings}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", default="0", help="name (substring is fine) or index")
    p.add_argument("--image", type=Path, help="register from a saved frame instead of the live camera")
    p.add_argument("--lens", type=Path, help="lens file; default data/cameras/<webcam>.json")
    p.add_argument("--data", type=Path, default=DATA)
    args = p.parse_args()

    lens_path = args.lens or camera_file(resolve_camera(args.camera)[1])
    if not lens_path.exists():
        sys.exit(f"no lens file at {lens_path}: run setup_room.py (or calibrate_intrinsics.py) first")
    if args.image:
        frame = cv2.imread(str(args.image))
        if frame is None:
            sys.exit(f"can't read {args.image}")
        frames = [frame]
    else:
        frames = capture(args.camera)
    summary = register(frames, load_json(lens_path), args.data)
    print(f"\nwebcam at {np.round(summary['center'], 2).tolist()} m, fit {summary['rms_px']:.2f} px")
    for warning in summary["warnings"]:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
