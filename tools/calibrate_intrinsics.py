"""Lens intrinsics K, dist for the corner camera (CAMERA.md E2): 9×6 checkerboard -> cv2.calibrateCamera.

    python -m tools.calibrate_intrinsics --room data/room --src 0        # live: SPACE capture, ESC/q solve
    python -m tools.calibrate_intrinsics --room data/room --images DIR   # offline, from saved views
    python -m tools.calibrate_intrinsics --room data/room --hfov 75      # emergency fallback, no board

Merges width/height/K/dist into ROOM/camera.json. New intrinsics invalidate the old pose, so R/t
are dropped: re-run tools.register_camera afterwards.
"""
import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

from recall.room import undistort
from tools import read_images, read_json, write_json

log = logging.getLogger(__name__)

BOARD = (9, 6)  # inner corners
MIN_VIEWS = 10
SIZE = (1280, 720)
POSE_KEYS = ("R", "t", "camera_center", "rms_px", "markers_used", "registered_at")
SUBPIX = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-3)
CORNER_FRAC = 0.2  # an image corner is covered by a board corner within this fraction of W and H of it


def find_board(img, board=BOARD, fast=False):
    """Sub-pixel inner corners (N, 2) of the checkerboard, or None."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | (cv2.CALIB_CB_FAST_CHECK if fast else 0)
    ok, corners = cv2.findChessboardCorners(gray, board, flags=flags)
    if not ok:
        return None
    return cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), SUBPIX).reshape(-1, 2)


def calibrate(views, size=SIZE, board=BOARD, square_m=0.025):
    """Corner sets from find_board -> (K, dist[5], rms_px)."""
    obj = np.zeros((board[0] * board[1], 3), np.float32)
    obj[:, :2] = np.mgrid[0:board[0], 0:board[1]].T.reshape(-1, 2) * square_m
    rms, K, dist, _, _ = cv2.calibrateCamera([obj] * len(views), [v.astype(np.float32) for v in views],
                                             size, None, None)
    return K, dist.ravel()[:5], float(rms)


def lens_problems(views, K, dist, size=SIZE):
    """Why the fitted lens may be wrong near the image corners (where dist matters most); empty if fine.

    dist is only constrained where the board was: without views reaching the corners, the polynomial
    can fold over before them, and those pixels then have no ray at all.
    """
    w, h = size
    corners = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], float)
    near = (np.abs(np.vstack(views)[:, None] - corners) < CORNER_FRAC * np.array(size)).all(2).any(0)
    problems = []
    if not near.all():
        names = np.array(["top-left", "top-right", "bottom-left", "bottom-right"])
        problems.append(f"no view puts the board in the {', '.join(names[~near])} corner(s) of the image")
    if np.isnan(undistort(corners, K, dist)).any():
        problems.append("dist folds over before the image corners: pixels there have no ray")
    return problems


def hfov_intrinsics(hfov_deg, size=SIZE):
    """Fallback pinhole: fx = fy = (W/2)/tan(HFOV/2), principal point at the image centre, no distortion."""
    w, h = size
    f = (w / 2) / np.tan(np.radians(hfov_deg) / 2)
    return np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1]]), np.zeros(5)


def write_intrinsics(room_dir, size, K, dist, rms=None):
    """Merge intrinsics into ROOM/camera.json, dropping any pose they invalidate."""
    path = Path(room_dir) / "camera.json"
    cam = read_json(path, {})
    if any(k in cam for k in POSE_KEYS):
        log.warning("new intrinsics invalidate the old pose: dropped R/t, re-run python -m tools.register_camera")
    cam = {k: v for k, v in cam.items() if k not in POSE_KEYS and k != "intrinsics_rms_px"}
    cam.update(width=size[0], height=size[1], K=np.asarray(K).tolist(), dist=np.asarray(dist).tolist())
    if rms is not None:
        cam["intrinsics_rms_px"] = rms
    write_json(path, cam)
    return path


def capture_live(index, board=BOARD):
    """Show the camera with detected corners; SPACE keeps a view, ESC/q finishes. Returns corner sets."""
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SIZE[1])
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    views = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise SystemExit(f"camera {index} gave no frame")
            if frame.shape[1::-1] != SIZE:
                raise SystemExit(f"camera gives {frame.shape[1]}x{frame.shape[0]}, need {SIZE[0]}x{SIZE[1]}")
            corners = find_board(frame, board, fast=True)
            show = frame.copy()
            if corners is not None:
                cv2.drawChessboardCorners(show, board, corners.reshape(-1, 1, 2), True)
            msg = f"{len(views)} views  SPACE capture  ESC/q {'solve' if len(views) >= MIN_VIEWS else 'quit'}"
            cv2.putText(show, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.imshow("calibrate intrinsics", show)
            key = cv2.waitKey(1) & 0xFF
            full = find_board(frame, board) if key == 32 and corners is not None else None
            if full is not None:
                views.append(full)
                log.info("view %d captured", len(views))
            elif key in (27, ord("q")):
                return views
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="Calibrate the corner camera's lens (K, dist).")
    ap.add_argument("--room", default="data/room")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--src", type=int, default=0, help="live camera index")
    src.add_argument("--images", help="folder of checkerboard views")
    src.add_argument("--hfov", type=float, help="emergency fallback: horizontal FOV in degrees")
    ap.add_argument("--square-mm", type=float, default=25.0, help="printed square size")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.hfov:
        K, dist = hfov_intrinsics(args.hfov)
        path = write_intrinsics(args.room, SIZE, K, dist)
        print(f"wrote {path} (HFOV {args.hfov}° fallback, no distortion)\nK = {np.round(K, 1).tolist()}")
        return
    if args.images:
        images = read_images(args.images)
        if any(img.shape[1::-1] != SIZE for img in images):
            raise SystemExit(f"all views must be {SIZE[0]}x{SIZE[1]}")
        views = [c for c in map(find_board, images) if c is not None]
        log.info("board found in %d of %d images", len(views), len(images))
    else:
        views = capture_live(args.src)
    if len(views) < MIN_VIEWS:
        raise SystemExit(f"need at least {MIN_VIEWS} views with the board, have {len(views)}")
    K, dist, rms = calibrate(views, square_m=args.square_mm / 1000)
    path = write_intrinsics(args.room, SIZE, K, dist, rms)
    print(f"wrote {path} from {len(views)} views: rms {rms:.3f} px\n"
          f"K = {np.round(K, 2).tolist()}\ndist = {np.round(dist, 5).tolist()}")
    if rms > 1.0:
        print("rms > 1 px: blurry or mis-detected views; recapture with the board sharp and still")
    for problem in lens_problems(views, K, dist):
        print(f"{problem}: recapture with the board close to every image corner")


if __name__ == "__main__":
    main()
