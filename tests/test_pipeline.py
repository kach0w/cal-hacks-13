"""The detection thread: frozen camera, capture never blocked, video replay, clean shutdown."""
import threading
import time

import cv2
import numpy as np

from fakes import FRAME, KEYS_TABLE, S, kinds, world
from recall.camera import Frame, VideoSource
from recall.perception import Detections
from recall.pipeline import Pipeline
from recall.query import clock


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


class StubCamera:
    """A live source whose newest frame the test controls; read() repeats it until push()."""
    live = True

    def __init__(self):
        self.frame, self.seq = None, 0

    def push(self, ts, image=FRAME):
        self.seq += 1
        self.frame = Frame(image, ts, self.seq)

    def read(self):
        return self.frame

    def close(self):
        pass


class ScriptedDetector:
    def __init__(self, sightings=(), delay=0.0):
        self.sightings, self.delay, self.calls = list(sightings), delay, 0

    def detect(self, image, ts, raw=False):
        self.calls += 1
        time.sleep(self.delay)
        return Detections([S(s.object, s.bbox) for s in self.sightings])


def busy(i):
    """Frames that always differ, so the change gate runs detection every look."""
    img = np.zeros((72, 128, 3), np.uint8)
    img[:, (i * 13) % 128:] = 255
    return img


# --- frozen camera ----------------------------------------------------------------------------
def test_stalled_camera_freezes_memory_and_stamps_answers(tmp_path):
    cfg, store, mem, q = world(tmp_path, stale_after_s=5.0)
    cam, clk = StubCamera(), Clock()
    det = ScriptedDetector([S("keys", KEYS_TABLE)])
    p = Pipeline(cam, det, mem, cfg, clock=clk)

    for i in range(3):                                   # keys on the table, camera live
        cam.push(clk.t, busy(i))
        p.step()
        clk.t += 1
    last_real = cam.frame.ts
    assert q.where_is("keys", now=clk.t).text.startswith("Your keys are on the table")

    det.sightings = []                                   # keys would look "gone"...
    for _ in range(20):                                  # ...but the camera has stalled
        assert p.step() is None
        clk.t += 1
    assert p.frozen and det.calls == 3
    assert mem.state["keys"].status == "observed" and mem.state["keys"].misses == 0
    assert store.heartbeat() == {"last_frame_ts": last_real, "camera_ok": False}
    a = q.where_is("keys", now=clk.t)
    assert a.as_of == last_real and a.text.startswith(f"As of {clock(last_real)}: Your keys are on the table")
    assert "disappeared" not in [e.kind for e in store.events()]

    cam.push(clk.t, busy(99))                            # camera back: live again, rules resume
    p.step()
    assert not p.frozen and q.as_of(clk.t) is None
    for i in range(3):
        clk.t += 1
        cam.push(clk.t, busy(100 + i))
        p.step()
    assert kinds(store.events())[-1] == "disappeared:keys"


def test_short_stall_is_not_frozen(tmp_path):
    cfg, store, mem, q = world(tmp_path, stale_after_s=5.0)
    cam, clk = StubCamera(), Clock()
    p = Pipeline(cam, ScriptedDetector(), mem, cfg, clock=clk)
    cam.push(clk.t)
    p.step()
    clk.t += 4
    assert p.step() is None and not p.frozen               # same frame: skipped, not counted


def test_camera_that_never_delivers(tmp_path):
    cfg, store, mem, q = world(tmp_path, stale_after_s=2.0)
    clk = Clock()
    p = Pipeline(StubCamera(), ScriptedDetector(), mem, cfg, clock=clk)
    clk.t += 3
    assert p.step() is None and p.frozen
    assert store.heartbeat() == {"last_frame_ts": None, "camera_ok": False}


def test_static_scene_keeps_the_heartbeat_fresh(tmp_path):
    """The gate skips detection on a still scene, but answers must still read as live."""
    cfg, store, mem, q = world(tmp_path)
    cam, clk = StubCamera(), Clock()
    det = ScriptedDetector([S("keys", KEYS_TABLE)])
    p = Pipeline(cam, det, mem, cfg, clock=clk)
    for i in range(8):
        cam.push(clk.t)                                  # identical frames
        r = p.step()
        clk.t += 1
    assert r.reason == "static" and det.calls < 8
    assert store.heartbeat()["last_frame_ts"] == clk.t - 1 and q.as_of(clk.t) is None


# --- threads ------------------------------------------------------------------------------------
class ThreadedCamera(StubCamera):
    """Produces frames on its own thread, like CameraSource."""

    def __init__(self, hz=200):
        super().__init__()
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, args=(1 / hz,), daemon=True)
        self._t.start()

    def _run(self, dt):
        while not self._stop.is_set():
            self.push(time.time(), busy(self.seq))
            time.sleep(dt)

    def close(self):
        self._stop.set()
        self._t.join()


def test_detection_never_blocks_capture(tmp_path):
    cfg, store, mem, q = world(tmp_path, loop_s=0.05)
    cam = ThreadedCamera()
    det = ScriptedDetector([S("keys", KEYS_TABLE)], delay=0.3)   # a slow detector
    p = Pipeline(cam, det, mem, cfg)
    p.start()
    seqs = []
    for _ in range(10):                                   # the "window" keeps getting fresh frames
        time.sleep(0.1)
        seqs.append(cam.read().seq)
    t0 = time.time()
    p.stop()
    cam.close()
    assert time.time() - t0 < 1.0                         # stop waits for at most one in-flight pass
    assert p.finished.is_set() and 2 <= det.calls <= 5
    assert all(b > a for a, b in zip(seqs, seqs[1:]))     # capture advanced during every detect


class Flaky(ScriptedDetector):
    def __init__(self, fail_calls):
        super().__init__([S("keys", KEYS_TABLE)])
        self.fail_calls = fail_calls

    def detect(self, image, ts, raw=False):
        self.calls += 1
        if self.calls in self.fail_calls:
            raise RuntimeError("model fell over")
        return Detections([S("keys", KEYS_TABLE)])


def test_one_failed_pass_is_survived(tmp_path, caplog):
    cfg, store, mem, q = world(tmp_path, loop_s=0.02)
    cam = ThreadedCamera()
    p = Pipeline(cam, Flaky({1}), mem, cfg)
    p.start()
    time.sleep(0.3)
    p.stop()
    cam.close()
    assert not p.crashed and "detection pass failed (1 in a row)" in caplog.text
    assert kinds(store.events()) == ["appeared:keys"]


def test_repeated_failures_give_up_and_freeze(tmp_path, caplog):
    cfg, store, mem, q = world(tmp_path, loop_s=0.02)
    cam = ThreadedCamera()
    p = Pipeline(cam, Flaky(set(range(1, 100))), mem, cfg)
    p.start()
    assert p.finished.wait(2)
    cam.close()
    assert p.crashed and "giving up" in caplog.text
    assert store.heartbeat()["camera_ok"] is False         # answers now say "as of"


def test_stopping_marks_the_world_not_live(tmp_path):
    cfg, store, mem, q = world(tmp_path)
    cam, clk = StubCamera(), Clock()
    p = Pipeline(cam, ScriptedDetector([S("keys", KEYS_TABLE)]), mem, cfg, clock=clk)
    cam.push(clk.t)
    p.step()
    assert q.as_of(clk.t) is None
    p.start()
    p.stop()
    assert store.heartbeat() == {"last_frame_ts": clk.t, "camera_ok": False}
    assert q.as_of(clk.t) == clk.t


def test_skipped_passes_keep_the_last_boxes(tmp_path):
    """The --show overlay and /stream draw latest().detections: a still scene must keep its boxes."""
    cfg, store, mem, q = world(tmp_path)
    cam, clk = StubCamera(), Clock()
    p = Pipeline(cam, ScriptedDetector([S("keys", KEYS_TABLE)]), mem, cfg, clock=clk)
    for _ in range(8):
        cam.push(clk.t)
        p.step()
        clk.t += 1
    r = p.latest()
    assert r.reason == "static" and [s.object for s in r.detections.sightings] == ["keys"]


# --- video replay -------------------------------------------------------------------------------
def write_video(path, n=50, fps=10):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (128, 72))
    for i in range(n):
        w.write(np.full((72, 128, 3), i * 5, np.uint8))  # brightness encodes the frame index
    w.release()


def test_video_replay_is_sequential_and_repeatable(tmp_path):
    path = tmp_path / "demo.avi"
    write_video(path)
    runs = []
    for _ in range(2):
        src = VideoSource(str(path), step_s=1.0)
        frames = []
        while (f := src.read()) is not None:
            frames.append(f)
        src.close()
        runs.append([(f.seq, round(f.ts - src.t0, 3), int(round(f.image.mean() / 5))) for f in frames])
    assert runs[0] == runs[1]
    assert runs[0] == [(1, 0.0, 0), (2, 1.0, 10), (3, 2.0, 20), (4, 3.0, 30), (5, 4.0, 40)]
    assert abs(src.t0 + 5.0 - path.stat().st_mtime) < 1e-6          # the video ends at its mtime


def test_replay_runs_to_the_end_and_finishes(tmp_path):
    path = tmp_path / "demo.avi"
    write_video(path, n=40)
    cfg, store, mem, q = world(tmp_path)
    det = ScriptedDetector([S("keys", KEYS_TABLE)])
    p = Pipeline(VideoSource(str(path), 1.0), det, mem, cfg)
    p.start()
    assert p.finished.wait(5)
    assert det.calls >= 1 and kinds(store.events()) == ["appeared:keys"]


def test_video_from_the_future_is_clamped_to_now(tmp_path):
    import os
    path = tmp_path / "demo.avi"
    write_video(path, n=20)
    os.utime(path, (time.time() + 3600, time.time() + 3600))  # copied with a bad clock
    src = VideoSource(str(path), 1.0)
    assert src.t0 + 2.0 <= time.time() + 0.01


def test_video_without_timestamps_falls_back_to_frame_count():
    class Cap:  # a backend that reports POS_MSEC = 0
        def __init__(self):
            self.i = 0
        def grab(self):
            self.i += 1
            return self.i <= 30
        def get(self, prop):
            return 0.0
        def retrieve(self):
            return True, np.full((4, 4, 3), self.i, np.uint8)
    src = VideoSource.__new__(VideoSource)
    src.cap, src.fps, src.t0, src.step_ms, src.next_ms, src.pos, src.seq = Cap(), 10.0, 0.0, 1000.0, 0.0, 0, 0
    got = []
    while (f := src.read()) is not None:
        got.append((round(f.ts, 3), int(f.image[0, 0, 0])))
    assert got == [(0.0, 1), (1.0, 11), (2.0, 21)]
