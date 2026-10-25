"""Which index is the corner camera? Grabs one frame from each camera index and shows them side by side.

    python -m tools.cameras [--max 4]

Writes data/cameras.jpg (one tile per index that gave a frame, labelled with its index and
resolution) and opens it in a window. Use that index as --src everywhere (or RECALL_SRC=<index>).
A camera gets FIRST_FRAME_S to start delivering, then SETTLE_S for its exposure.
"""
import argparse
import time

import cv2
import numpy as np

from recall import config
from recall.overlay import banner, close_windows, key

FIRST_FRAME_S = 5.0   # some cameras take seconds to start delivering
SETTLE_S = 1.0        # first frames are often black while exposure settles


def probe(index, w, h) -> tuple[bool, object]:
    """-> (did it open, a frame from SETTLE_S after its first one, or None)."""
    cap = cv2.VideoCapture(index)
    try:
        if not cap.isOpened():
            return False, None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        frame, deadline = None, time.time() + FIRST_FRAME_S
        while time.time() < deadline:
            ok, f = cap.read()
            if not ok:
                time.sleep(0.02)
                continue
            if frame is None:
                deadline = time.time() + SETTLE_S
            frame = f
        return True, frame
    finally:
        cap.release()


def main():
    ap = argparse.ArgumentParser(description="Show a frame from every camera index.")
    ap.add_argument("--max", type=int, default=4, help="highest index to try")
    config.add_args(ap)
    a = ap.parse_args()
    cfg = config.from_args(a)
    tiles, silent = [], 0
    for i in range(a.max + 1):
        opened, f = probe(i, cfg.capture_w, cfg.capture_h)
        if f is None:
            print(f"index {i}: " + (f"opened but gave no frame in {FIRST_FRAME_S:g} s" if opened else "did not open"))
            silent += opened
            continue
        h, w = f.shape[:2]
        note = "" if (w, h) == (cfg.capture_w, cfg.capture_h) else \
            f"  (not the {cfg.capture_w}x{cfg.capture_h} asked for: zones need one size everywhere)"
        print(f"index {i}: {w}x{h}{note}")
        tile = cv2.resize(f, (640, int(640 * h / w)))
        tiles.append(banner(tile, f"--src {i}   {w}x{h}"))
    if silent and not tiles:
        raise SystemExit(f"{silent} camera(s) opened but gave no frames: in use by another app? Run again")
    if not tiles:
        raise SystemExit("no camera opened (macOS: System Settings > Privacy > Camera > allow your terminal)")
    height = max(t.shape[0] for t in tiles)
    sheet = np.hstack([cv2.copyMakeBorder(t, 0, height - t.shape[0], 0, 4, cv2.BORDER_CONSTANT) for t in tiles])
    out = cfg.data_dir / "cameras.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet)
    print(f"saved {out}; any key in the window closes it")
    cv2.imshow("cameras", sheet)
    try:
        while not key(100):  # not waitKey(0): Ctrl-C would wait for a key too
            pass
    except KeyboardInterrupt:
        pass
    finally:
        close_windows()


if __name__ == "__main__":
    main()
