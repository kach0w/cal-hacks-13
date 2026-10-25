#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["open3d==0.19.0", "trimesh", "pillow", "opencv-python>=4.8", "pyobjc-framework-Cocoa; sys_platform == 'darwin'"]
# ///
"""Set up a room in one go: the scan, the webcam's lens, and where the webcam is.

    uv run tools/setup_room.py --camera brio

Before running:
  1. Markers up (printables/markers.pdf): at least 3 where the webcam sees them face-on, on the floor
     and a wall.
  2. Scan the room with Polycam (LiDAR mode, Developer mode on) and drop the export zip(s) into
     data/room/: the Raw-data zip, plus the glTF zip if the raw one has no mesh.
  3. Webcam in its final spot.

Then this:
  1. turns the scan into data/room/room.glb, markers.json and plan.png   (scan_to_room.py)
  2. reuses the webcam's lens file from data/cameras/, or, the first time, opens a guided
     calibration window: hold the calibration board up until the grid fills  (calibrate_intrinsics.py)
  3. finds the webcam in the room from the markers: camera.json, camera_overlay.jpg  (register_camera.py)

Re-run it after a new scan or a bumped webcam. --recalibrate redoes the lens.
"""

import argparse

import numpy as np

import calibrate_intrinsics
import register_camera
import scan_to_room
from roomlib import DATA, load_json
from webcam import resolve_camera


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", default="0", help="webcam name (substring is fine) or index")
    p.add_argument("--recalibrate", action="store_true", help="redo the webcam's lens calibration")
    args = p.parse_args()

    print("== 1/3  scan")
    scan = scan_to_room.process_scan(scan_to_room.prepare_scan(DATA))

    print("\n== 2/3  lens")
    _, name = resolve_camera(args.camera)
    lens_path = calibrate_intrinsics.camera_file(name)
    if lens_path.exists() and not args.recalibrate:
        print(f"  using {name}'s lens file, {lens_path}")
    else:
        print(f"  calibrating {name}: hold the board 40-60 cm from it and fill the grid")
        calibrate_intrinsics.run(args.camera, lens_path)

    print("\n== 3/3  webcam position")
    camera = register_camera.register(register_camera.capture(args.camera), load_json(lens_path))

    print("\n== done")
    print(f"  room:   {len(scan['markers'])} markers in the scan; top-down view in {DATA / 'plan.png'}")
    print(f"  webcam: at {np.round(camera['center'], 2).tolist()} m, fitted to markers {camera['markers_used']} "
          f"({camera['rms_px']:.1f} px); check {DATA / 'camera_overlay.jpg'}")
    for warning in scan["warnings"] + camera["warnings"]:
        print(f"  note:   {warning}")


if __name__ == "__main__":
    main()
