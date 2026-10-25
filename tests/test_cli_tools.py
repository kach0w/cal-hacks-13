"""The interactive setup tools end to end (CAMERA.md §3-§5, §8 tasks 0-2): enroll calibrate, zones
onboard/manual/show/list, live --show, tools.cameras, tools.range_test capture, tools.record.

No window opens and no camera or model is touched: OpenCV's window calls, input() and `open` are
scripted (GUI), cameras are fakes or a small video in tmp_path, detectors return fixed boxes.
Every tool runs through call(), so a scripted Ctrl-C the tool should absorb fails the test instead
of stopping the pytest run, and fake cameras that would be read forever fail after `limit` reads."""
import builtins
import json
import logging
import shutil
import signal
import subprocess
import sys
import time

import cv2
import numpy as np
import pytest

from recall import config, zones
from recall.camera import Frame, VideoSource
from recall.models import Sighting, Zone
from recall.perception import Detections
from recall.store import Store, read_json, write_json_atomic

W, H = 320, 180


class GUI:
    """What the human sees and does. keys: one per waitKey while a window is open ('' = none, a float =
    that many seconds pass, an exception class is raised). replies: one per input(), which must never
    run while an OpenCV window is open (on macOS it would freeze). rois: what selectROIs returns.
    events: imshow window names and waitKey delays, in order."""

    def __init__(self, monkeypatch, keys=(), replies=(), rois=()):
        self.keys, self.replies, self.rois = list(keys), list(replies), rois
        self.open, self.titles, self.asked, self.opened, self.closes, self.events = set(), [], [], [], 0, []
        for name in ("imshow", "waitKey", "destroyAllWindows", "destroyWindow", "selectROIs"):
            monkeypatch.setattr(cv2, name, getattr(self, name))
        monkeypatch.setattr(builtins, "input", self.input)
        monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kw: self.opened.append([str(c) for c in cmd]))

    def imshow(self, name, img):
        self.open.add(name)
        self.titles.append(name)
        self.events.append(name)

    def waitKey(self, ms=0):
        self.events.append(ms)
        if not self.open:
            return -1                                   # no window, no key events
        if not self.keys:
            raise AssertionError("ran out of scripted keys")
        k = self.keys.pop(0)
        if isinstance(k, type):
            raise k()
        if isinstance(k, float):
            time.sleep(k)
            return -1
        return -1 if k == "" else ord(k) | 0x100000     # high bits set, as some backends do

    def destroyAllWindows(self):
        self.open.clear()
        self.closes += 1

    def destroyWindow(self, name):
        self.open.discard(name)

    def selectROIs(self, name, img, showCrosshair=True, fromCenter=False, printNotice=True):
        self.titles.append(name)
        self.open.add(name)
        return self.rois

    def input(self, prompt=""):
        assert not self.open, f"input({prompt!r}) while {self.open} is open: it freezes on macOS"
        self.asked.append(prompt)
        if not self.replies:
            raise AssertionError(f"no scripted reply for {prompt!r}")
        r = self.replies.pop(0)
        if isinstance(r, type):
            raise r()
        return r


class Cam:
    """A fake live camera: `none` reads of nothing, then a new frame every `every` reads, up to frame
    `stall` (then that one forever). Frame n is filled with 0 while n <= black, else 10 + n, so frames
    can be told apart; size(n) -> (w, h). Read more than `limit` times, it fails the test."""
    live = True

    def __init__(self, none=0, black=0, every=1, size=lambda n: (W, H), stall=None, limit=None):
        self.none, self.black, self.every, self.size = none, black, every, size
        self.stall, self.limit, self.reads, self.closed = stall, limit, 0, False

    def read(self):
        self.reads += 1
        if self.limit and self.reads > self.limit:
            raise AssertionError(f"still reading after {self.limit} reads: a wait without a time limit?")
        if self.reads <= self.none:
            return None
        n = (self.reads - self.none - 1) // self.every + 1
        n = min(n, self.stall or n)
        w, h = self.size(n)
        return Frame(np.full((h, w, 3), 0 if n <= self.black else min(255, 10 + n), np.uint8), 1e6 + n, n)

    def close(self):
        self.closed = True


class Det:
    """A detector that sees each item in `boxes` ({id: bbox}) on every frame; score(n) -> the score
    of its n-th detect call."""

    def __init__(self, boxes, cfg=None, score=lambda n: 0.5):
        self.boxes, self.cfg, self.score, self.calls = boxes, cfg, score, 0

    def detect(self, image, ts, raw=False):
        self.calls += 1
        return Detections([Sighting(oid, list(b), self.score(self.calls), ts) for oid, b in self.boxes.items()])

    def warmup(self, h, w):
        pass


@pytest.fixture
def tmp(tmp_path):
    """A scratch DATA_DIR with a copy of items.json; each test warns about frame sizes afresh."""
    shutil.copy(config.ROOT / "items.json", tmp_path / "items.json")
    zones._warn_size.cache_clear()
    old = signal.getsignal(signal.SIGTERM)       # live.main points SIGTERM at Ctrl-C's handler
    yield tmp_path
    signal.signal(signal.SIGTERM, old)


def call(main):
    """main(), failing the test if a scripted Ctrl-C escapes it (pytest would stop the whole run)."""
    try:
        return main()
    except KeyboardInterrupt:
        pytest.fail("Ctrl-C escaped the tool")


def argv(monkeypatch, tmp, *args):
    monkeypatch.setattr(sys, "argv", ["prog", *map(str, args), "--set", f"DATA_DIR={tmp}",
                                      "--set", f"ITEMS_PATH={tmp / 'items.json'}",
                                      "--set", f"CAPTURE_W={W}", "--set", f"CAPTURE_H={H}"])


def video(path, n=30, fps=10):
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (W, H))
    for i in range(n):
        img = np.full((H, W, 3), 120, np.uint8)
        cv2.rectangle(img, (5 * i, 60), (5 * i + 40, 120), (255, 255, 255), -1)
        out.write(img)
    out.release()
    return path


def items(tmp):
    return read_json(tmp / "items.json")


# --- zones -------------------------------------------------------------------------------------
def test_onboard_names_candidates_in_the_terminal_with_the_picture_in_preview(tmp, monkeypatch):
    from recall import perception
    dets = [{"label": "table", "bbox": [0, 90, 200, 180]},
            {"label": "table", "bbox": [200, 90, 320, 180]},          # same name: must be renamed
            {"label": "chair", "bbox": [10, 10, 30, 30]},             # too small to be furniture
            {"label": "shelf", "bbox": [0, 0, 320, 40]}]
    monkeypatch.setattr(perception, "detect_prompts", lambda img, prompts, cfg, conf: dets)
    monkeypatch.setattr(sys, "platform", "darwin")
    gui = GUI(monkeypatch,
              replies=["", "",                                        # [0] keep 'table', on
                       "", "Table", "side  table ", " Next To ",      # [1] two taken names, then one
                       "d",                                           # [2] shelf: delete
                       "y",                                           # draw manual zones too
                       "shelf", "in"],                                # the one box big enough
              rois=np.array([[250, 10, 100, 30], [5, 5, 3, 3]]))      # off the right edge; a slip
    argv(monkeypatch, tmp, "onboard", "--src", video(tmp / "table.avi"))
    call(zones.main)

    assert [z.to_dict() for z in zones.load_zones(tmp / "zones.json")] == [
        Zone("table", [0, 90, 200, 180], "on", "detected").to_dict(),
        Zone("side table", [200, 90, 320, 180], "next to", "detected").to_dict(),
        Zone("shelf", [250.0, 10.0, 320.0, 40.0], "in", "manual").to_dict()]
    assert gui.opened == [["open", str(tmp / "zones_candidates.jpg")], ["open", str(tmp / "zones_drawn.jpg")]]
    assert cv2.imread(str(tmp / "zones_candidates.jpg")).shape == (H, W, 3)
    assert gui.titles == ["draw zones"] and not gui.open and not gui.replies


def test_onboard_with_no_furniture_goes_straight_to_drawing(tmp, monkeypatch):
    from recall import perception
    monkeypatch.setattr(perception, "detect_prompts", lambda img, prompts, cfg, conf: [])
    src = video(tmp / "table.avi")
    zones.save_zones([Zone("counter", [0, 0, 100, 100])], tmp / "zones.json")

    gui = GUI(monkeypatch, rois=())                                    # ESC without a box
    argv(monkeypatch, tmp, "onboard", "--src", src)
    call(zones.main)
    assert gui.asked == [] and gui.titles == ["draw zones"]
    assert [z.name for z in zones.load_zones(tmp / "zones.json")] == ["counter"]   # not wiped

    gui = GUI(monkeypatch, rois=np.array([[10, 20, 200, 100]]), replies=["desk", ""])
    call(zones.main)
    assert [(z.name, z.prep, z.box) for z in zones.load_zones(tmp / "zones.json")] == [
        ("desk", "on", [10.0, 20.0, 210.0, 120.0])]


def test_manual_adds_to_the_zones_and_rejects_a_taken_name(tmp, monkeypatch, capsys):
    zones.save_zones([Zone("table", [0, 90, 200, 180])], tmp / "zones.json")
    gui = GUI(monkeypatch, rois=np.array([[210, 90, 100, 90], [0, 0, 50, 40]]),
              replies=["table", "counter", "", ""])                     # 2nd box: ENTER drops it
    argv(monkeypatch, tmp, "manual", "--src", video(tmp / "table.avi"))
    call(zones.main)
    assert [(z.name, z.source) for z in zones.load_zones(tmp / "zones.json")] == [
        ("table", "manual"), ("counter", "manual")]
    assert "already a zone called 'table'" in capsys.readouterr().out and not gui.open

    argv(monkeypatch, tmp, "list")
    call(zones.main)
    assert "counter" in capsys.readouterr().out


def test_ctrl_c_while_naming_changes_nothing(tmp, monkeypatch):
    from recall import perception
    monkeypatch.setattr(perception, "detect_prompts",
                        lambda img, prompts, cfg, conf: [{"label": "table", "bbox": [0, 90, 200, 180]}])
    zones.save_zones([Zone("counter", [0, 0, 100, 100])], tmp / "zones.json")
    gui = GUI(monkeypatch, replies=["", KeyboardInterrupt])
    argv(monkeypatch, tmp, "onboard", "--src", video(tmp / "table.avi"))
    with pytest.raises(SystemExit, match="stopped"):
        call(zones.main)
    assert [z.name for z in zones.load_zones(tmp / "zones.json")] == ["counter"] and not gui.open


def test_zones_show_quits_on_q_or_ctrl_c_and_releases_the_camera(tmp, monkeypatch, caplog):
    from recall import camera
    for keys in (["", "", "Q"], ["", "\x1b"], ["", KeyboardInterrupt]):
        cam = Cam(none=3, size=lambda n: (640, 360))
        monkeypatch.setattr(camera, "open_source", lambda cfg: cam)
        gui = GUI(monkeypatch, keys=keys)
        argv(monkeypatch, tmp, "show")
        call(zones.main)
        assert cam.closed and not gui.open and gui.closes and not gui.keys
    assert caplog.text.count("frames are 640x360, not CAPTURE_W x CAPTURE_H = 320x180") == 1


def test_grab_waits_out_a_camera_that_starts_dark(tmp, monkeypatch):
    from recall import camera
    cam = Cam(none=2, black=3)
    monkeypatch.setattr(camera, "open_source", lambda cfg: cam)
    monkeypatch.setattr(zones, "SETTLE_S", 0.0)
    img = zones._grab(config.Settings(data_dir=tmp, capture_w=W, capture_h=H))
    assert img.mean() > 0 and cam.closed


def test_grab_gives_up_on_a_camera_that_stays_dark_or_silent(tmp, monkeypatch):
    from recall import camera
    monkeypatch.setattr(zones, "SETTLE_S", 0.0)
    monkeypatch.setattr(zones, "DARK_WAIT_S", 0.2)
    cfg = config.Settings(data_dir=tmp, capture_w=W, capture_h=H)
    cam = Cam(black=10 ** 9, limit=30)                                  # lens cap on: the dark frame it is
    monkeypatch.setattr(camera, "open_source", lambda cfg: cam)
    assert zones._grab(cfg).mean() == 0 and cam.closed
    cam = Cam(none=10 ** 9, limit=30)
    monkeypatch.setattr(camera, "open_source", lambda cfg: cam)
    with pytest.raises(SystemExit, match="could not read a frame"):
        zones._grab(cfg)
    assert cam.closed


# --- enroll calibrate --------------------------------------------------------------------------
def test_calibrate_a_video_end_to_end(tmp, monkeypatch):
    from recall import enroll, perception
    det = Det({"keys": [300, 100, 340, 150]}, score=lambda n: round(0.9 - 0.1 * n, 2))
    monkeypatch.setattr(perception, "make_detector", lambda items, cfg: det)
    monkeypatch.setattr(enroll, "MIN_GAP_S", 0.05)
    stale = tmp / "enroll" / "keys" / "7.jpg"                           # from an older calibration
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"old")
    gui = GUI(monkeypatch, keys=[" ", " ",          # 2nd SPACE within MIN_GAP_S: ignored
                                 0.06, " ", 0.06, " ",   # keys done: scores 0.8, 0.5, 0.3
                                 " ", "N"])           # glasses never seen: SPACE does nothing, N next
    argv(monkeypatch, tmp, "calibrate", "keys", "glasses", "--src", video(tmp / "table.avi"),
         "--set", "LOOP_S=0.1")
    call(enroll.main)

    it = items(tmp)
    assert it["keys"]["min_score"] == 0.18                                 # 0.6 x the lowest accepted
    assert it["keys"]["ref_crops"] == ["enroll/keys/0.jpg", "enroll/keys/1.jpg", "enroll/keys/2.jpg"]
    assert sorted(p.name for p in (tmp / "enroll" / "keys").iterdir()) == ["0.jpg", "1.jpg", "2.jpg"]
    assert cv2.imread(str(tmp / "enroll" / "keys" / "0.jpg")).shape == (50, 20, 3)   # clipped to the image
    assert it["glasses"]["min_score"] is None and "ref_crops" not in it["glasses"]
    assert it["remote"] == json.loads((config.ROOT / "items.json").read_text())["remote"]
    assert not gui.open and not gui.keys and det.calls == 8


def test_calibrate_quit_keeps_the_samples_taken_and_releases(tmp, monkeypatch):
    from recall import camera, enroll, perception
    cam = Cam(none=2)
    monkeypatch.setattr(camera, "open_source", lambda cfg: cam)
    monkeypatch.setattr(perception, "make_detector",
                        lambda items, cfg: Det({"keys": [10, 10, 60, 40], "glasses": [80, 10, 120, 40]}))
    gui = GUI(monkeypatch, keys=["", " ", "q"])
    argv(monkeypatch, tmp, "calibrate", "keys", "glasses")
    call(enroll.main)
    it = items(tmp)
    assert it["keys"]["min_score"] == 0.3 and it["keys"]["ref_crops"] == ["enroll/keys/0.jpg"]
    assert it["glasses"]["min_score"] is None and cam.closed and not gui.open


def test_calibrate_video_ending_mid_item_keeps_its_samples(tmp, monkeypatch, capsys):
    from recall import enroll, perception
    monkeypatch.setattr(perception, "make_detector", lambda items, cfg: Det({"keys": [10, 10, 60, 40]}))
    gui = GUI(monkeypatch, keys=[" ", "", "", ""])
    argv(monkeypatch, tmp, "calibrate", "keys", "--src", video(tmp / "short.avi", n=4), "--set", "LOOP_S=0.1")
    call(enroll.main)
    assert items(tmp)["keys"]["min_score"] == 0.3 and "video ended" in capsys.readouterr().out
    assert not gui.open


def test_calibrate_ctrl_c_and_a_box_off_the_image(tmp, monkeypatch):
    from recall import camera, enroll, perception
    cam = Cam()
    monkeypatch.setattr(camera, "open_source", lambda cfg: cam)
    monkeypatch.setattr(perception, "make_detector", lambda items, cfg: Det({"keys": [400, 10, 460, 40]}))
    gui = GUI(monkeypatch, keys=[" ", " ", KeyboardInterrupt])          # nothing to crop: ignored
    argv(monkeypatch, tmp, "calibrate", "keys")
    with pytest.raises(SystemExit, match="stopped"):
        call(enroll.main)
    assert items(tmp)["keys"]["min_score"] is None and not (tmp / "enroll" / "keys").exists()
    assert cam.closed and not gui.open


def test_calibrate_ctrl_c_mid_recalibration_replaces_the_old_set_whole(tmp, monkeypatch, caplog):
    from recall import camera, enroll, perception
    crops = tmp / "enroll" / "keys"
    crops.mkdir(parents=True)
    for i in range(3):
        cv2.imwrite(str(crops / f"{i}.jpg"), np.zeros((5, 5, 3), np.uint8))
    data = items(tmp)
    data["keys"].update(min_score=0.3, ref_crops=[f"enroll/keys/{i}.jpg" for i in range(3)])
    write_json_atomic(tmp / "items.json", data)
    on_disk = []                                                        # the crops at each detect call

    def score(n):
        on_disk.append(sorted(p.name for p in crops.iterdir()))
        return 0.8
    cam = Cam(size=lambda n: (640, 360))
    monkeypatch.setattr(camera, "open_source", lambda cfg: cam)
    monkeypatch.setattr(perception, "make_detector", lambda items, cfg: Det({"keys": [10, 10, 60, 40]}, score=score))
    gui = GUI(monkeypatch, keys=[" ", "", KeyboardInterrupt])
    argv(monkeypatch, tmp, "calibrate", "keys")
    with pytest.raises(SystemExit, match="stopped"):
        call(enroll.main)

    assert on_disk == [["0.jpg", "1.jpg", "2.jpg"]] * 3                 # untouched while sampling
    it = items(tmp)["keys"]
    assert it["min_score"] == 0.48 and it["ref_crops"] == ["enroll/keys/0.jpg"]   # kept, as q would
    assert sorted(p.name for p in crops.iterdir()) == ["0.jpg"]
    assert cv2.imread(str(crops / "0.jpg")).shape == (30, 50, 3)        # the new sample
    assert cam.closed and not gui.open and "frames are 640x360" in caplog.text


# --- live --show -------------------------------------------------------------------------------
@pytest.mark.parametrize("keys", [[0.3, "", "q"], [0.3, "", "\x1b"], [0.3, "", KeyboardInterrupt]])
def test_live_show_window_stops_cleanly(tmp, monkeypatch, caplog, keys):
    from recall import live
    cam = Cam(size=lambda n: (640, 360))
    monkeypatch.setattr(live, "open_source", lambda cfg: cam)
    monkeypatch.setattr(live, "make_detector", lambda items, cfg: Det({"keys": [100, 100, 160, 140]}))
    gui = GUI(monkeypatch, keys=keys)
    argv(monkeypatch, tmp, "--show")
    call(live.main)
    assert cam.closed and not gui.open and gui.titles[0] == "recall" and not gui.keys
    store = Store(tmp)
    assert [e.kind for e in store.events()] == ["appeared"] and store.heartbeat()["camera_ok"] is False
    assert "frames are 640x360" in caplog.text


# --- tools.record ------------------------------------------------------------------------------
def recorded(path, fps):
    src = VideoSource(str(path), 1 / fps)
    frames = []
    while (f := src.read()) is not None:
        frames.append(f)
    src.close()
    return frames


@pytest.mark.parametrize("stop", ["q", KeyboardInterrupt])
def test_record_writes_a_video_that_replays_at_its_frame_rate(tmp, monkeypatch, caplog, stop):
    from tools import record
    caplog.set_level(logging.INFO)
    cam = Cam(none=3, size=lambda n: (W, H) if n < 4 else (W // 2, H // 2))   # the camera switches mode
    monkeypatch.setattr(record, "CameraSource", lambda index, w, h: cam)
    gui = GUI(monkeypatch, keys=[""] * 6 + [stop])
    out = tmp / "takes" / "demo1.mp4"
    argv(monkeypatch, tmp, out, "--fps", 20)
    call(record.main)

    frames = recorded(out, 20)
    n = int(caplog.text.split(f"saved {out}: ")[1].split(" frames")[0])
    assert len(frames) == n >= 7 and {f.image.shape for f in frames} == {(H, W, 3)}
    assert np.allclose(np.diff([f.ts for f in frames]), 1 / 20, atol=1e-3)
    assert cam.closed and not gui.open and "frames are 160x90" in caplog.text


def test_record_repeats_frames_when_the_loop_is_slow(tmp, monkeypatch):
    from tools import record
    monkeypatch.setattr(record, "CameraSource", lambda index, w, h: Cam())
    GUI(monkeypatch, keys=[0.2, 0.2, 0.2, "q"])                          # each preview takes 0.2 s
    argv(monkeypatch, tmp, tmp / "slow.mp4", "--fps", 20)
    t0 = time.time()
    call(record.main)
    assert abs(len(recorded(tmp / "slow.mp4", 20)) / 20 - (time.time() - t0)) < 0.15   # video time = wall time


def test_record_gives_up_on_a_camera_that_never_delivers(tmp, monkeypatch):
    from tools import record
    cam = Cam(none=10 ** 9, limit=100)
    monkeypatch.setattr(record, "CameraSource", lambda index, w, h: cam)
    monkeypatch.setattr(record, "FIRST_FRAME_S", 0.2)
    GUI(monkeypatch)
    argv(monkeypatch, tmp, tmp / "demo1.mp4")
    with pytest.raises(SystemExit, match="no frames from camera 0"):
        call(record.main)
    assert cam.closed and not (tmp / "demo1.mp4").exists()


def test_record_never_replaces_a_take_without_force(tmp, monkeypatch):
    from tools import record
    out = tmp / "demo1.mp4"
    out.write_bytes(b"a good take")
    monkeypatch.setattr(record, "CameraSource", lambda index, w, h: pytest.fail("camera opened"))
    GUI(monkeypatch)
    argv(monkeypatch, tmp, out)
    with pytest.raises(SystemExit, match="demo1.mp4 exists"):
        call(record.main)
    assert out.read_bytes() == b"a good take"

    monkeypatch.setattr(record, "CameraSource", lambda index, w, h: Cam())
    GUI(monkeypatch, keys=["", "q"])
    argv(monkeypatch, tmp, out, "--force")
    call(record.main)
    assert recorded(out, 15)


# --- tools.range_test capture ------------------------------------------------------------------
def range_test(monkeypatch, tmp, cam, *args):
    from recall import camera
    from tools import range_test
    monkeypatch.setattr(camera, "open_source", lambda cfg: cam)
    monkeypatch.setattr(range_test, "make_detector", lambda items, cfg: Det({"keys": [10, 10, 60, 40]}, cfg))
    argv(monkeypatch, tmp, *args, "--detectors", "yolo-world", "--imgsz", 640)
    call(range_test.main)


def test_range_capture_bursts_distinct_frames_and_keeps_station_order(tmp, monkeypatch, capsys, caplog):
    from tools.range_test import BURST_GAP_MS
    cam = Cam(every=3, size=lambda n: (640, 360))                       # a new frame every 3rd read
    gui = GUI(monkeypatch, keys=["", " ", "", "", "",                   # far: SPACE + 3 burst frames
                                 " ", "", "", "",                       # near
                                 "N"])                                  # mid: skip
    range_test(monkeypatch, tmp, cam, "capture", "--stations", "far=3,near=1,mid=2", "--burst", 3)
    out = tmp / "range_test"
    means = [cv2.imread(str(out / "near" / f"{i:02d}.jpg")).mean() for i in range(3)]
    assert means[0] < means[1] < means[2]                               # three different frames
    assert list(json.loads((out / "stations.json").read_text()).items()) == [("near", 1.0), ("far", 3.0)]
    assert not (out / "mid").exists() and (out / "report" / "near_yolo-world_640.jpg").exists()
    printed = capsys.readouterr().out
    assert printed.index("\n  near") < printed.index("\n  far")          # eval in distance order
    assert cam.closed and not gui.open and not gui.keys and "frames are 640x360" in caplog.text
    repaints = sum(a == "range test" and b == BURST_GAP_MS for a, b in zip(gui.events, gui.events[1:]))
    assert repaints == 6                                                # each burst frame shown + painted


def test_range_capture_quit_keeps_what_was_captured(tmp, monkeypatch, capsys):
    out = tmp / "range_test"
    (out / "near").mkdir(parents=True)
    (out / "near" / "05.jpg").write_bytes(b"old burst")                  # replaced by the new burst
    (out / "empty").mkdir()                                              # a station with no images
    gui = GUI(monkeypatch, keys=[" ", "", "", "q"])
    range_test(monkeypatch, tmp, Cam(), "capture", "--burst", 2)
    assert sorted(p.name for p in (out / "near").iterdir()) == ["00.jpg", "01.jpg"]
    assert json.loads((out / "stations.json").read_text()) == {"near": 1.0}
    printed = capsys.readouterr().out
    assert "skipping empty: no images" in printed and "near" in printed and not gui.open


def test_range_burst_gives_up_on_a_stalled_camera_and_keeps_the_earlier_one(tmp, monkeypatch):
    from tools import range_test as rt
    monkeypatch.setattr(rt, "NEW_FRAME_S", 0.2)
    near = tmp / "range_test" / "near"
    near.mkdir(parents=True)
    (near / "00.jpg").write_bytes(b"earlier burst")
    cam = Cam(stall=2, limit=300)                                       # two frames, then that one forever
    gui = GUI(monkeypatch, keys=[" ", ""])
    with pytest.raises(SystemExit, match="no new frame for 0.2 s"):
        range_test(monkeypatch, tmp, cam, "capture", "--burst", 3)
    assert [p.read_bytes() for p in near.iterdir()] == [b"earlier burst"]
    assert not (tmp / "range_test" / "stations.json").exists() and cam.closed and not gui.open


def test_range_capture_ctrl_c_releases_the_camera(tmp, monkeypatch):
    cam = Cam()
    gui = GUI(monkeypatch, keys=["", KeyboardInterrupt])
    with pytest.raises(SystemExit, match="stopped"):
        range_test(monkeypatch, tmp, cam, "capture")
    assert cam.closed and not gui.open


# --- tools.cameras -----------------------------------------------------------------------------
class Cap:
    """cv2.VideoCapture for a fake set of cameras: index -> (frame size, seconds before its first
    frame), or 'dead' (opens, never gives a frame)."""
    cams = {0: ((W, H), 0), 1: "dead", 2: ((640, 480), 0.2)}
    released = []

    def __init__(self, index):
        self.index, self.kind, self.t0 = index, self.cams.get(index), time.time()

    def isOpened(self):
        return self.kind is not None

    def set(self, prop, value):
        return True

    def read(self):
        if self.kind == "dead" or time.time() - self.t0 < self.kind[1]:
            return False, None
        (w, h), _ = self.kind
        return True, np.full((h, w, 3), 80, np.uint8)

    def release(self):
        self.released.append(self.index)


@pytest.mark.parametrize("keys", [["", "", "x"], ["", KeyboardInterrupt]])
def test_cameras_tiles_every_index_that_gives_a_frame(tmp, monkeypatch, capsys, keys):
    from tools import cameras
    monkeypatch.setattr(cv2, "VideoCapture", Cap)
    monkeypatch.setattr(cameras, "SETTLE_S", 0.01)
    monkeypatch.setattr(cameras, "FIRST_FRAME_S", 0.4)                 # index 2 starts after 0.2 s
    Cap.released = []
    gui = GUI(monkeypatch, keys=keys)
    argv(monkeypatch, tmp, "--max", 3)
    call(cameras.main)
    printed = capsys.readouterr().out
    assert "index 0: 320x180\n" in printed and "index 1: opened but gave no frame in 0.4 s" in printed
    assert "index 3: did not open" in printed
    assert "index 2: 640x480  (not the 320x180 asked for" in printed
    assert cv2.imread(str(tmp / "cameras.jpg")).shape == (480, 2 * 644, 3)
    assert Cap.released == [0, 1, 2, 3] and not gui.open and not gui.keys


@pytest.mark.parametrize("cams, why", [({}, "no camera opened .* allow your terminal"),
                                       ({0: "dead"}, r"1 camera\(s\) opened but gave no frames")])
def test_cameras_none_give_a_frame(tmp, monkeypatch, cams, why):
    from tools import cameras
    monkeypatch.setattr(cv2, "VideoCapture", Cap)
    monkeypatch.setattr(Cap, "cams", cams)
    monkeypatch.setattr(cameras, "FIRST_FRAME_S", 0.1)
    GUI(monkeypatch)
    argv(monkeypatch, tmp, "--max", 1)
    with pytest.raises(SystemExit, match=why) as e:
        call(cameras.main)
    assert ("Privacy" in str(e.value)) == (not cams)
