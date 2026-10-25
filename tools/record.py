"""Record the corner camera for replay (tasks 2-3).

    python -m tools.record data/demo1.mp4 [--src 0] [--fps 15] [--force]

A preview window shows REC and the elapsed time; q, ESC or Ctrl-C stops. The newest frame is
written for every 1/fps s of wall time, so video time is wall time (a slow camera or a slow loop
repeats frames instead of speeding the video up) and `python -m recall.live --src data/demo1.mp4`
replays it exactly as the camera saw it. Every frame is written at the first frame's size.
An existing file is replaced only with --force, so re-running the last command can't wipe a take.
"""
import argparse
import logging
import time
from pathlib import Path

import cv2

from recall import config
from recall.camera import CameraSource
from recall.overlay import QUIT, RED, banner, close_windows, key
from recall.zones import check_size

log = logging.getLogger("recall")
FIRST_FRAME_S = 5.0   # no frame this long after opening: the camera isn't delivering


def main():
    ap = argparse.ArgumentParser(description="Record the camera to a video file for replay.")
    ap.add_argument("out", type=Path, help="e.g. data/demo1.mp4")
    ap.add_argument("--src", help="camera index")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--force", action="store_true", help="replace OUT if it exists")
    config.add_args(ap)
    a = ap.parse_args()
    cfg = config.from_args(a, src=a.src)
    if not str(cfg.src).isdigit():
        raise SystemExit("--src must be a camera index")
    if a.out.exists() and not a.force:
        raise SystemExit(f"{a.out} exists: give another name, or add --force to replace it")
    a.out.parent.mkdir(parents=True, exist_ok=True)

    cam = CameraSource(int(cfg.src), cfg.capture_w, cfg.capture_h)
    writer, n, t0 = None, 0, None
    try:
        deadline = time.time() + FIRST_FRAME_S
        while (frame := cam.read()) is None:
            if time.time() > deadline:
                raise SystemExit(f"no frames from camera {cfg.src} in {FIRST_FRAME_S:.0f} s "
                                 "(macOS: allow camera access for the terminal)")
            time.sleep(0.05)
        h, w = frame.image.shape[:2]
        writer = cv2.VideoWriter(str(a.out), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (w, h))
        if not writer.isOpened():
            raise SystemExit(f"can't write {a.out}")
        log.info("recording %dx%d at %.0f fps to %s — q, ESC or Ctrl-C stops", w, h, a.fps, a.out)
        t0 = time.time()
        while True:
            frame = cam.read() or frame
            check_size(frame.image, cfg)
            img = frame.image if frame.image.shape[:2] == (h, w) else cv2.resize(frame.image, (w, h))
            while n <= (time.time() - t0) * a.fps:  # frame n is video time n / fps
                writer.write(img)
                n += 1
            cv2.imshow("record", banner(img.copy(), f"REC {time.time() - t0:5.1f}s  (q stops)", RED))
            if key(1) in QUIT:
                break
            time.sleep(max(0.0, t0 + n / a.fps - time.time()))  # hold the frame rate
    except KeyboardInterrupt:
        pass
    finally:
        if writer:
            writer.release()
        cam.close()
        close_windows()
    if t0:
        log.info("saved %s: %d frames, %.1f s", a.out, n, n / a.fps)


if __name__ == "__main__":
    main()
