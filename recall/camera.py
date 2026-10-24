"""Frame sources. A live camera grabs on its own thread and keeps only the newest frame; a video
file replays deterministically, one frame per LOOP_S of video time, stamped with when it was shot.

Both return Frame(image, ts, seq) from read(); a camera returns the same seq until a new frame
arrives (that is how the pipeline notices a stalled camera), a video returns None at the end.
"""
import logging
import os
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Frame:
    image: np.ndarray
    ts: float          # capture time, epoch seconds
    seq: int           # increases by one per new frame


class CameraSource:
    live = True

    def __init__(self, index: int, width=1280, height=720):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise SystemExit(f"camera {index} did not open (macOS: allow camera access for the terminal)")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._latest: Frame | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._grab, name="capture", daemon=True)
        self._thread.start()
        log.info("camera %d open at %dx%d", index, self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                 self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def _grab(self):
        seq = 0
        while not self._stop.is_set():
            ok, img = self.cap.read()
            if not ok:
                self._stop.wait(0.01)
                continue
            seq += 1
            with self._lock:
                self._latest = Frame(img, time.time(), seq)

    def read(self) -> Frame | None:
        with self._lock:
            return self._latest

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2.0)  # release only after the grab thread has left cap.read()
        if self._thread.is_alive():
            log.warning("camera read is hung; leaving the device for the OS to reclaim at exit")
            return
        self.cap.release()


class VideoSource:
    live = False

    def __init__(self, path: str, step_s: float):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise SystemExit(f"could not open {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = n / self.fps if 0 < n < 1e7 else 0.0  # some containers don't know their length
        # Frames carry the time they were shot (the file was finished at its mtime, never later
        # than now), so replays are repeatable and answers about one read "as of" the recording.
        self.t0 = min(os.path.getmtime(path), time.time()) - duration
        self.step_ms = step_s * 1000
        self.next_ms = 0.0
        self.pos = 0       # frames grabbed so far
        self.seq = 0

    def read(self) -> Frame | None:
        """The first frame at or after the next LOOP_S mark of video time; None at the end."""
        while self.cap.grab():   # sequential: exact, unlike seeking in H.264
            self.pos += 1
            ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)   # this frame's own timestamp (variable frame rate)
            if ms <= 0 and self.pos > 1:               # backend without timestamps
                ms = (self.pos - 1) * 1000 / self.fps
            if ms + 0.5 < self.next_ms:
                continue
            ok, img = self.cap.retrieve()
            if not ok:
                return None
            self.next_ms = ms + self.step_ms
            self.seq += 1
            return Frame(img, self.t0 + ms / 1000, self.seq)
        return None

    def close(self):
        self.cap.release()


def open_source(cfg):
    s = str(cfg.src)
    if s.isdigit():
        return CameraSource(int(s), cfg.capture_w, cfg.capture_h)
    return VideoSource(s, cfg.loop_s)
