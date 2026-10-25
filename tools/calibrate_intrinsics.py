#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["opencv-python>=4.8", "pyobjc-framework-Cocoa; sys_platform == 'darwin'"]
# ///
"""Measure a webcam's lens once -> data/cameras/<webcam>.json, reused in every room. Usually run by
setup_room.py the first time it sees a webcam.

    uv run tools/calibrate_intrinsics.py --camera brio
    uv run tools/calibrate_intrinsics.py --images data/room/calib_frames --name brio-101  # saved frames

Hold printables/calibration_board.pdf, mounted flat, 40-60 cm from the webcam. The window shows a
3x3 grid: move the board until every square has turned green, and tilt it in a few shots. It
finishes by itself. Esc stops.

Calibrate at 1280x720 with the focus the webcam will keep. Changing resolution, focus or zoom later
invalidates this; exposure and white balance don't.
"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from printed import BOARD_DICT, BOARD_MARKER_CM, BOARD_SQUARE_CM, BOARD_SQUARES
from roomlib import CAMERAS, DATA, bring_to_front, save_json
from webcam import HEIGHT, WIDTH, open_camera, resolve_camera

MIN_CORNERS = 12  # of the board's 24 inner corners
MIN_MOVE_PX = 60  # how far the board must move before another shot is kept
GRID = 3
MIN_TILTED = 4  # shots with the board tilted at least TILT_DEG; they're what pin down the focal length
TILT_DEG = 20
MIN_SHOTS = 15
RMS_WARN_PX = 1.0
OUTLIER_X = 2.0  # a view fitting this many times worse than the median is dropped
MIN_VIEWS = 10


def camera_file(name):
    """data/cameras/<name>.json, e.g. 'Brio 101' -> brio-101.json."""
    return CAMERAS / f"{re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')}.json"


def make_board():
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, BOARD_DICT))
    return cv2.aruco.CharucoBoard(BOARD_SQUARES, BOARD_SQUARE_CM / 100, BOARD_MARKER_CM / 100, dictionary)


def find_board(detector, gray):
    """(corners, ids) when enough of the board is visible, else None."""
    corners, ids, _, _ = detector.detectBoard(gray)
    if ids is None or len(ids) < MIN_CORNERS:
        return None
    return corners.reshape(-1, 1, 2).astype(np.float32), ids.reshape(-1, 1)  # one shape across OpenCV 4 and 5


def tilt(board, view, size):
    """Rough angle between the board and the image plane, in degrees, with a guessed lens."""
    obj, img = board.matchImagePoints(*view)
    K = np.array([[size[0], 0, size[0] / 2], [0, size[0], size[1] / 2], [0, 0, 1.0]])
    ok, rvec, _ = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_IPPE)
    return float(np.degrees(np.arccos(abs(cv2.Rodrigues(rvec)[0][2, 2])))) if ok else 0.0


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


def save(out, name, rms, K, dist, n, size):
    save_json(out, {
        "camera": name, "width": size[0], "height": size[1],
        "K": np.round(K, 4).tolist(), "dist": np.round(dist, 6).tolist(),
        "rms_px": round(rms, 3), "views": n,
        "calibrated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    })
    print(f"lens: fx {K[0, 0]:.0f}, fy {K[1, 1]:.0f}, centre ({K[0, 2]:.0f}, {K[1, 2]:.0f}); "
          f"fit {rms:.2f} px over {n} shots")
    if rms > RMS_WARN_PX:
        print(f"WARNING: fit above {RMS_WARN_PX} px; the board may not be flat. Redo with --recalibrate")


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


def done(cells, tilted, shots):
    return cells.all() and tilted >= MIN_TILTED and shots >= MIN_SHOTS


def guided(camera, board, detector, keep_dir):
    """Live view with a grid that fills in; returns the kept views once the grid, tilts and count are done."""
    keep_dir.mkdir(parents=True, exist_ok=True)
    cap = open_camera(camera, exact=True)
    views, spots, cells, tilted = [], [], np.zeros((GRID, GRID), bool), 0
    raised = False
    while not done(cells, tilted, len(views)):
        ok, frame = cap.read()
        if not ok:
            sys.exit("camera stopped returning frames")
        found = find_board(detector, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        view = frame.copy()
        if found is not None:
            pts = found[0].reshape(-1, 2)
            centre = pts.mean(axis=0)
            spot = np.r_[centre, np.ptp(pts, axis=0).mean()]
            if all(np.abs(spot - s).sum() > MIN_MOVE_PX for s in spots):
                spots.append(spot)
                views.append(found)
                cells[min(int(centre[1] * GRID / HEIGHT), GRID - 1), min(int(centre[0] * GRID / WIDTH), GRID - 1)] = True
                tilted += tilt(board, found, (WIDTH, HEIGHT)) >= TILT_DEG
                cv2.imwrite(str(keep_dir / f"{len(views):02d}.png"), frame)
            cv2.aruco.drawDetectedCornersCharuco(view, found[0], found[1], (0, 255, 0))

        shade = view.copy()
        for r in range(GRID):
            for c in range(GRID):
                p1 = (c * WIDTH // GRID, r * HEIGHT // GRID)
                p2 = ((c + 1) * WIDTH // GRID, (r + 1) * HEIGHT // GRID)
                if cells[r, c]:
                    cv2.rectangle(shade, p1, p2, (0, 200, 0), -1)
                cv2.rectangle(view, p1, p2, (200, 200, 200), 1)
        view = cv2.addWeighted(shade, 0.25, view, 0.75, 0)
        for i, line in enumerate((
            "Move the board into every square, and tilt it in a few shots. Finishes by itself.",
            f"squares {cells.sum()}/{GRID * GRID}   tilted {min(tilted, MIN_TILTED)}/{MIN_TILTED}   "
            f"shots {len(views)}/{MIN_SHOTS}   (Esc stops)",
        )):
            cv2.putText(view, line, (16, 34 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4)
            cv2.putText(view, line, (16, 34 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        cv2.imshow("calibrate", view)
        if not raised:
            bring_to_front()
            raised = True
        if cv2.waitKey(1) & 0xFF == 27:
            sys.exit("stopped before the grid was covered; nothing saved")
    cap.release()
    cv2.destroyAllWindows()
    return views, (WIDTH, HEIGHT)


def run(camera, out=None):
    """Guided calibration for a webcam -> its lens file."""
    _, name = resolve_camera(camera)
    board = make_board()
    views, size = guided(camera, board, cv2.aruco.CharucoDetector(board), DATA / "calib_frames")
    rms, K, dist, used = calibrate(board, views, size)
    out = out or camera_file(name)
    save(out, name, rms, K, dist, used, size)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", default="0", help="name (substring is fine) or index")
    p.add_argument("--images", type=Path, help="calibrate from saved frames instead of the live camera")
    p.add_argument("--name", help="webcam name for the lens file, with --images")
    p.add_argument("--out", type=Path, help="default data/cameras/<webcam>.json")
    args = p.parse_args()

    if not args.images:
        run(args.camera, args.out)
        return
    board = make_board()
    views, size = from_images(args.images, board, cv2.aruco.CharucoDetector(board))
    rms, K, dist, used = calibrate(board, views, size)
    name = args.name or resolve_camera(args.camera)[1]
    save(args.out or camera_file(name), name, rms, K, dist, used, size)


if __name__ == "__main__":
    main()
