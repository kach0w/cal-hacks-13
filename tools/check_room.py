#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["open3d==0.19.0", "trimesh", "scipy", "pillow", "opencv-python>=4.8", "pyobjc-framework-Cocoa; sys_platform == 'darwin'"]
# ///
"""The handoff's acceptance table (handoffAB.md), to run together with B.

    uv run tools/check_room.py --wall-to-wall 4.12

Reads data/room/: camera.json, checks.json, camera_ref.jpg, room.glb, scan_report.json.
The first run asks you to click the base of each test object in camera_ref.jpg (kept in
check_pixels.json; --reclick to redo). --wall-to-wall is a tape-measured distance between two
opposite walls; you click one point on each wall in the mesh.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from roomlib import DATA, Mesh, camera_center, floor_hit, load_json, pick_points, project, save_json

CHECKS_TEMPLATE = """{
  "lens_measured": [x, y, z],
  "test_points": {
    "mid_floor": [x, 0, z],
    "far_corner_floor": [x, 0, z]
  }
}"""

HINTS = {
    "scale": "error grows with distance: scale, or the focal length in K",
    "rms": "a marker moved since the scan, a wrong marker size, or mismatched ids",
    "lens": "same offset everywhere: mesh origin/axes. Mirrored or behind a wall: corner order or the Y-up flip",
    "floor": "grows with distance: scale or K. Grows toward the image edges: dist not applied. "
             "Good in the middle, bad in a corner: too few markers on that side",
    "project": "same causes as the floor check",
}


def click_bases(image, names, path, reclick):
    saved = {} if reclick or not path.exists() else load_json(path)
    todo = [n for n in names if n not in saved]
    for name in todo:
        clicked = []
        view = image.copy()
        cv2.putText(view, f"click the BASE of the object at '{name}', then press any key (s to skip)",
                    (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.namedWindow("check")
        cv2.setMouseCallback("check", lambda e, u, v, *_: e == cv2.EVENT_LBUTTONDOWN and clicked.append((u, v)))
        while True:
            shown = view.copy()
            if clicked:
                cv2.drawMarker(shown, clicked[-1], (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            cv2.imshow("check", shown)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("s"):
                break
            if key != 255 and clicked:
                saved[name] = list(clicked[-1])
                break
    cv2.destroyAllWindows()
    if todo:
        save_json(path, saved)
    return saved


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wall-to-wall", type=float, help="tape-measured distance between two opposite walls, metres")
    p.add_argument("--reclick", action="store_true", help="redo the test-object clicks")
    p.add_argument("--data", type=Path, default=DATA)
    args = p.parse_args()
    data = args.data

    if not (data / "checks.json").exists():
        sys.exit(f"write {data / 'checks.json'} first, tape-measured from the origin corner:\n{CHECKS_TEMPLATE}")
    cam, checks = load_json(data / "camera.json"), load_json(data / "checks.json")
    K, dist = np.array(cam["K"]), np.array(cam["dist"])
    R, t = np.array(cam["R"]), np.array(cam["t"])
    rows = []  # (check, value, limit, passed, hint key)

    report = data / "scan_report.json"
    if report.exists() and (ratios := [m["edge_ratio"] for m in load_json(report)["markers"].values()]):
        off = abs(np.mean(ratios) - 1)
        rows.append(("scale: marker edges vs printed size", f"{off:.1%}", "< 2%", off < 0.02, "scale"))
    if args.wall_to_wall:
        a, b = pick_points(Mesh.load(data / "room.glb"), 2, "Click one point on each of the two opposite walls")
        d = np.abs(a - b)[[0, 2]].max()  # along whichever floor axis separates them
        off = abs(d / args.wall_to_wall - 1)
        rows.append(("scale: wall to wall vs tape", f"{d:.3f} m vs {args.wall_to_wall:.3f} m ({off:.1%})", "< 2%",
                     off < 0.02, "scale"))

    rows.append(("reprojection error (rms_px)", f"{cam['rms_px']:.2f} px", "< 5 px", cam["rms_px"] < 5, "rms"))
    if lens := checks.get("lens_measured"):
        off = np.linalg.norm(camera_center(R, t) - np.array(lens))
        rows.append(("camera_center vs tape", f"{off * 100:.1f} cm", "< 5 cm", off < 0.05, "lens"))

    points = checks.get("test_points", {})
    if points:
        image = cv2.imread(str(data / "camera_ref.jpg"))
        pixels = click_bases(image, list(points), data / "check_pixels.json", args.reclick)
        for name, xyz in points.items():
            if name not in pixels:
                continue
            limit = 0.10 if "far" in name else 0.05
            hit = floor_hit(K, dist, R, t, [pixels[name]])[0]
            off = np.linalg.norm((hit - np.array(xyz))[[0, 2]])
            rows.append((f"floor_hit on {name}", f"{off * 100:.1f} cm", f"< {limit * 100:.0f} cm", off < limit, "floor"))
            px = np.linalg.norm(project(K, dist, R, t, [xyz])[0] - pixels[name])
            rows.append((f"{name} projected into camera_ref", f"{px:.1f} px", "< 8 px", px < 8, "project"))

    width = max(len(r[0]) for r in rows)
    print()
    for check, value, limit, passed, _ in rows:
        print(f"{'PASS' if passed else 'FAIL'}  {check:<{width}}  {value:>24}  (need {limit})")
    failed = {r[4] for r in rows if not r[3]}
    for key in failed:
        print(f"  likely cause ({key}): {HINTS[key]}")
    print("\nall checks passed" if not failed else f"\n{sum(not r[3] for r in rows)} check(s) failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
