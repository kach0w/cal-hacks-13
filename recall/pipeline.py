"""The detection thread (DESIGN §9.1 rule 1): perception never blocks capture.

Once per LOOP_S it takes the newest frame and runs  stall check -> change gate -> detect -> memory.
If a live camera delivers no new frame for STALE_AFTER_S the memory is frozen: no detection and no
misses counted (an unplugged camera must not make everything "disappear"), and the heartbeat says
camera_ok=false so every answer is stamped "as of" the last real frame. A video source is read
one frame per LOOP_S of video time, as fast as detection allows, and the thread ends with the file.
When the thread ends for any reason the heartbeat says camera_ok=false: the world is frozen.
"""
import logging
import threading
import time
from dataclasses import dataclass

from .camera import Frame
from .gate import ChangeGate
from .perception import Detections, Detector

log = logging.getLogger(__name__)
MAX_FAILURES = 5   # passes in a row that may raise before the thread gives up


@dataclass
class Result:
    frame: Frame
    reason: str                              # first | change | settling | heartbeat | static
    detections: Detections | None = None     # the latest boxes (carried over when the gate skips)


class Pipeline:
    def __init__(self, source, detector: Detector, memory, cfg, clock=time.time, on_result=None):
        """on_result(Result): called on the detection thread after every look (snapshots, /stream)."""
        self.source, self.detector, self.memory, self.cfg = source, detector, memory, cfg
        self.store, self.clock, self.on_result = memory.store, clock, on_result
        self._pace = None                    # (wall, video ts) of the first replayed frame
        self.gate = ChangeGate(cfg)
        self.frozen = False                  # live camera stalled; memory not updating
        self.crashed = False                 # gave up after MAX_FAILURES passes in a row raised
        self.last_frame_ts = None
        self._last_seq, self._last_new = None, clock()
        self._latest: Result | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.finished = threading.Event()    # the thread has exited (stopped, video ended, crashed)
        self._thread = threading.Thread(target=self._run, name="detection", daemon=True)
        if cfg.loop_s * 2 > cfg.stale_after_s:
            log.warning("LOOP_S %.1f is close to STALE_AFTER_S %.1f: answers may say 'as of' while live",
                        cfg.loop_s, cfg.stale_after_s)

    def start(self):
        self._thread.start()

    def stop(self, timeout=10.0):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout)  # lets an in-flight detect + memory write finish
            if self._thread.is_alive():
                log.warning("detection thread did not stop within %.0f s", timeout)

    def latest(self) -> Result | None:
        with self._lock:
            return self._latest

    def _run(self):
        failures = 0
        try:
            while not self._stop.is_set():
                t0 = self.clock()
                try:
                    result, failures = self.step(), 0
                except Exception:
                    failures += 1
                    log.exception("detection pass failed (%d in a row)", failures)
                    if failures >= MAX_FAILURES:
                        self.crashed = True
                        log.error("giving up: memory is no longer updating")
                        break
                    result = True  # a failed pass is not the end of a video
                if result is None and not self.source.live:
                    break  # end of the video
                if self.source.live:
                    self._stop.wait(max(0.0, self.cfg.loop_s - (self.clock() - t0)))
                elif self.cfg.replay_speed > 0 and isinstance(result, Result):
                    self._stop.wait(self._replay_delay(result.frame.ts))
        finally:
            if self.last_frame_ts is not None:
                self.store.beat(self.last_frame_ts, camera_ok=False)  # nobody is watching any more
            self.finished.set()

    def step(self) -> Result | None:
        """One look. None when there was no new frame (camera stalled, or the video ended)."""
        frame = self.source.read()
        if frame is None or frame.seq == self._last_seq:
            if self.source.live:
                self._check_stall()
            return None
        if self.frozen:
            log.warning("camera is back; memory live again")
            self.frozen = False
        self._last_seq, self._last_new, self.last_frame_ts = frame.seq, self.clock(), frame.ts

        run, why = self.gate.check(frame.image, frame.ts)
        if not run:
            self.store.beat(frame.ts)
            prev = self.latest()
            result = Result(frame, why, prev.detections if prev else None)  # still scene: same boxes
        else:
            dets = self.detector.detect(frame.image, frame.ts)
            self.gate.mark_detected(frame.image, frame.ts)
            self.memory.update(dets.sightings, frame.image, frame.ts, held=dets.held)
            result = Result(frame, why, dets)
            if log.isEnabledFor(logging.DEBUG):
                self._log_pass(result)
        with self._lock:
            self._latest = result
        if self.on_result:
            try:
                self.on_result(result)
            except Exception:  # a display hook must never stop the memory
                log.exception("on_result failed")
        return result

    def _replay_delay(self, ts) -> float:
        """Seconds to wait so a replay plays at REPLAY_SPEED x real time."""
        if self._pace is None:
            self._pace = (self.clock(), ts)
        wall0, ts0 = self._pace
        return max(0.0, wall0 + (ts - ts0) / self.cfg.replay_speed - self.clock())

    def _check_stall(self):
        if self.clock() - self._last_new < self.cfg.stale_after_s:
            return
        if not self.frozen:
            self.frozen = True
            log.warning("no new camera frame for %.0f s: memory frozen, answers will say 'as of'",
                        self.cfg.stale_after_s)
        self.store.beat(self.last_frame_ts, camera_ok=False)

    def _log_pass(self, r: Result):
        m = self.memory
        seen = ", ".join(f"{m.items[s.object].name}@{m.locate(s.bbox)[0] or '?'} {s.score:.2f}"
                         for s in r.detections.sightings) or "nothing"
        persons = f"  (+{len(r.detections.persons)} person)" if r.detections.persons else ""
        log.debug("[%-9s %.3f] seen: %s%s", r.reason, self.gate.last_fraction, seen, persons)
