#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["opencv-python>=4.8", "pyobjc-framework-Cocoa; sys_platform == 'darwin'"]
# ///
"""Measure a webcam's lens (K, distortion) with the ChArUco board -> data/room/intrinsics.json.

    uv run tools/calibrate_intrinsics.py --camera brio
    uv run tools/calibrate_intrinsics.py --images data/room/calib_frames   # re-run on saved frames

Hold printables/calibration_board.pdf, mounted flat, in front of the camera: near, far, tilted,
and in every part of the frame, corners included. A frame is kept automatically when the board
is clear and somewhere new. Press c to calibrate once ~20 are kept, q to quit.

Run at 1280x720 with the focus the camera will keep. Changing resolution, focus or zoom later
invalidates this; exposure and white balance don't.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from printed import BOARD_DICT, BOARD_MARKER_CM, BOARD_SQUARE_CM, BOARD_SQUARES
from roomlib import DATA, bring_to_front, save_json
from webcam import HEIGHT, WIDTH, open_camera

MIN_CORNERS = 12  # of the board's 24 inner corners
MIN_MOVE_PX = 60  # how far the board must move before another frame is kept
TARGET_VIEWS = 20
RMS_WARN_PX = 1.0
OUTLIER_X = 2.0  # a view fitting this many times worse than the median is dropped
MIN_VIEWS = 10


def make_board():
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, BOARD_DICT))
    return cv2.aruco.CharucoBoard(BOARD_SQUARES, BOARD_SQUARE_CM / 100, BOARD_MARKER_CM / 100, dictionary)


def find_board(detector, gray):
    """(corners, ids) when enough of the board is visible, else None."""
    corners, ids, _, _ = detector.detectBoard(gray)
    if ids is None or len(ids) < MIN_CORNERS:
        return None
    return corners.reshape(-1, 1, 2).astype(np.float32), ids.reshape(-1, 1)  # one shape across OpenCV 4 and 5


def calibrate(board, views, size):
    """views: [(corners, ids)] -> (rms_px, K, dist[5], views used).

    Views that fit much worse than the rest (usually motion blur) are dropped and the lens re-solved.
    k3 stays 0, the usual choice for webcams: left free, it soaks up noise as extreme distortion."""
    obj, img = [], []
    for corners, ids in views:
        o, i = board.matchImagePoints(corners, ids)
        if len(o) >= 6:
            obj.append(o)
            img.append(i)
    if len(obj) < 6:
        sys.exit(f"only {len(obj)} usable views; need at least 6, ideally ~20")
    while True:
        rms, K, dist, _, _, _, _, per_view = cv2.calibrateCameraExtended(obj, img, size, None, None,
                                                                         flags=cv2.CALIB_FIX_K3)
        per_view = per_view.ravel()
        bad = per_view > max(OUTLIER_X * np.median(per_view), 1.0)
        if not bad.any() or len(obj) - bad.sum() < MIN_VIEWS:
            return float(rms), K, dist.ravel()[:5], len(obj)
        print(f"  dropping {bad.sum()} view(s) with errors of {per_view[bad].min():.1f}+ px (usually motion blur)")
        obj = [o for o, b in zip(obj, bad) if not b]
        img = [i for i, b in zip(img, bad) if not b]


def save(out, rms, K, dist, n, size):
    save_json(out, {
        "width": size[0], "height": size[1],
        "K": np.round(K, 4).tolist(), "dist": np.round(dist, 6).tolist(),
        "rms_px": round(rms, 3), "views": n,
        "calibrated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    })
    print(f"reprojection error {rms:.2f} px over {n} views")
    if rms > RMS_WARN_PX:
        print(f"WARNING: above {RMS_WARN_PX} px. Board not flat, motion blur, or too few tilted views; redo")


def from_images(folder, board, detector):
    views, size = [], None
    for path in sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}):
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        size = size or gray.shape[::-1]
        if gray.shape[::-1] != size:
            sys.exit(f"{path.name} is {gray.shape[1]}x{gray.shape[0]}, the others {size[0]}x{size[1]}")
        if (found := find_board(detector, gray)) is not None:
            views.append(found)
    return views, size


def from_camera(camera, board, detector, keep_dir):
    keep_dir.mkdir(parents=True, exist_ok=True)
    cap = open_camera(camera, exact=True)
    views, spots = [], []  # spots: (centre, size) of each kept view, to tell "somewhere new"
    while True:
        ok, frame = cap.read()
        if not ok:
            sys.exit("camera stopped returning frames")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found = find_board(detector, gray)
        view = frame.copy()
        if found is not None:
            pts = found[0].reshape(-1, 2)
            spot = np.r_[pts.mean(axis=0), np.ptp(pts, axis=0).mean()]
            if all(np.abs(spot - s).sum() > MIN_MOVE_PX for s in spots):
                spots.append(spot)
                views.append(found)
                cv2.imwrite(str(keep_dir / f"{len(views):02d}.png"), frame)
            cv2.aruco.drawDetectedCornersCharuco(view, found[0], found[1], (0, 255, 0))
        cv2.putText(view, f"kept {len(views)}/{TARGET_VIEWS}   c: calibrate   q: quit", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.imshow("calibrate", view)
        if not views and not spots:
            bring_to_front()
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            sys.exit("quit without calibrating")
        if key == ord("c"):
            cap.release()
            cv2.destroyAllWindows()
            return views, (WIDTH, HEIGHT)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", default="0", help="name (substring is fine) or index")
    p.add_argument("--images", type=Path, help="calibrate from saved frames instead of the live camera")
    p.add_argument("--out", type=Path, default=DATA / "intrinsics.json")
    args = p.parse_args()

    board = make_board()
    detector = cv2.aruco.CharucoDetector(board)
    if args.images:
        views, size = from_images(args.images, board, detector)
    else:
        views, size = from_camera(args.camera, board, detector, DATA / "calib_frames")
    rms, K, dist, used = calibrate(board, views, size)
    save(args.out, rms, K, dist, used, size)


if __name__ == "__main__":
    main()
