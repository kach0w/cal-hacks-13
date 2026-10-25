"""tools/tune.py: cached detections replayed through the real gate and memory give exactly the events
recall.live logs; expect files, scoring and the grid. The simulated table (tools/simulate.py) seen by
the real UltralyticsDetector code around a fake model: no camera, no model download."""
import os
import shutil
from dataclasses import replace

import numpy as np
import pytest

from recall import live
from recall.camera import VideoSource
from recall.config import ROOT, load_settings
from recall.enroll import load_items, save_items
from recall.gate import ChangeGate
from recall.models import Item, Transition
from recall.places import Places
from recall.perception import PERSON, UltralyticsDetector
from recall.store import Store
from recall.zones import save_zones
from tools import simulate, tune

IDS = ["keys", "glasses", "remote", "wallet", "bag"]       # items.json order: events of one pass follow it
ITEMS = {i: Item(i, i, f"a {i}", container=i == "bag") for i in IDS}
LABELS = IDS + [PERSON]

# What tools/simulate.py's script does (tests/test_simulate.py), as an expect file.
SCRIPT = """\
# the props appear
appeared keys kitchen counter
appeared glasses kitchen counter
appeared remote table
appeared wallet table
appeared bag table
moved keys table                    # via mid-air: one move, not two
# the hand over the remote at t=9 commits nothing
disappeared glasses
reappeared glasses table
inferred_contained glasses table    # into the bag
moved bag kitchen counter
moved glasses kitchen counter       # carried inside the bag
"""


class _T:  # what ultralytics tensors look like after .cpu().numpy()
    def __init__(self, a):
        self.a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


DECOY = [100, 420, 220, 500]   # a second "wallet", on the counter


class SceneModel:
    """Stands in for an ultralytics model: the script's boxes at second `sec`, scored by
    score(sec, item) (None = missed), the hand as a person, a weak decoy for the glasses and a wallet
    decoy scored by score(sec, "decoy")."""

    def __init__(self, truth, score):
        self.truth, self.score, self.sec = truth, score, 0

    def predict(self, image, conf, **kw):
        boxes, persons = self.truth[self.sec]
        rows = [(b, LABELS.index(oid), self.score(self.sec, oid)) for oid, b in boxes.items()]
        rows += [(p, LABELS.index(PERSON), 0.8) for p in persons] + [([10, 10, 60, 50], 1, 0.05)]
        rows += [(DECOY, LABELS.index("wallet"), self.score(self.sec, "decoy"))]
        # float32 scores kept only if above conf, compared in float32, best first: as ultralytics does
        rows = [(b, c, np.float32(s)) for b, c, s in rows if s is not None]
        rows = sorted((r for r in rows if r[2] > np.float32(conf)), key=lambda r: -r[2])
        b = type("Boxes", (), {"xyxy": _T([r[0] for r in rows]), "cls": _T([r[1] for r in rows]),
                               "conf": _T(np.array([r[2] for r in rows], np.float32))})
        return [type("Result", (), {"boxes": b})]


class SceneDetector(UltralyticsDetector):
    """The real detector code around SceneModel, which is told each frame's second of the script."""

    def __init__(self, video, items, cfg, score):
        _, truth, self.t0 = video
        self.scene, self.calls = SceneModel(truth, score), 0
        super().__init__(items, cfg, self.scene, LABELS)

    def detect(self, image, ts, raw=False):
        self.calls += 1
        self.scene.sec = int(ts - self.t0 + 0.01)   # the second of the script, also at LOOP_S=0.5
        return super().detect(image, ts, raw)


def clean(sec, oid):
    return None if oid == "decoy" else 0.9


MISSES = np.random.default_rng(3).random((simulate.LENGTH_S, 2)) < 0.2


def noisy(sec, oid):
    """Keys and remote missed at random; the wallet hovers around the 0.15 default threshold, and at
    20-23 s the decoy scores 0.0002 under it. Keyed by (second, item), so it doesn't matter which
    frames the gate lets through."""
    if oid == "decoy":
        return 0.1502 if 20 <= sec <= 23 else None
    if oid == "wallet":
        if 13 <= sec <= 15:
            return 0.1496                    # rounds to 0.15, is below it
        return 0.1504 if 20 <= sec <= 23 else 0.16 + 0.02 * (sec % 6)
    if oid in ("keys", "remote") and MISSES[sec, int(oid == "remote")]:
        return None
    return 0.9 - 0.01 * (sec % 5)


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    """The simulated table written at 2 fps (the loop samples 1 fps; 10 fps would only be slower)
    -> (path, {second: (boxes, persons)}, t0 of its frames)."""
    path = tmp_path_factory.mktemp("video") / "sim.mp4"
    scene = simulate.Scene(ROOT / "data" / "reference")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(simulate, "FPS", 2)
        simulate.write_video(scene, path)
    src = VideoSource(str(path), 1.0)
    src.close()
    return path, {sec: scene.render(sec)[1:] for sec in range(simulate.LENGTH_S)}, src.t0


@pytest.fixture
def base(tmp_path):
    """--set flags for a scratch data dir with the simulated zones and these items."""
    save_items(ITEMS, tmp_path / "items.json")
    save_zones(simulate.ZONES, tmp_path / "zones.json")
    return [f"DATA_DIR={tmp_path}", f"ITEMS_PATH={tmp_path / 'items.json'}", "DETECT_DEVICE=cpu"]


def logged(t):
    return t.kind, t.object, t.from_surface, t.to_surface, t.status, t.container, t.confidence, t.ts


# Settings and per-item thresholds, each run through recall.live and through the tune replay.
CASES = [([], {}),
         (["MISSING_AFTER_K=1", "MOVE_CONFIRM_N=1"], {}),
         (["CHANGE_FRACTION=0.03", "SETTLE_S=2", "HEARTBEAT_S=6"], {}),
         # the wallet's float32 0.2 is not above the model's conf=0.2 floor, though it is >= 0.2
         (["DEFAULT_MIN_SCORE=0.2", "MISSING_AFTER_K=2", "CONTAIN_OVERLAP=0.6", "ZONE_MARGIN_PX=60"], {}),
         (["DEFAULT_MIN_SCORE=0.1"], {"wallet": 0.15, "keys": 0.95}),
         # detect() keeps a later box scoring above the best one's rounded score (0.1502 > 0.15) only
         # if it passes the threshold: the decoy does at 0.15 (case 4), not at 0.1503
         (["DEFAULT_MIN_SCORE=0.1"], {"wallet": 0.1503}),
         # the frames read and the gate's tiny frames come from their own caches
         (["LOOP_S=0.5", "MISSING_AFTER_K=4"], {}),
         (["CHANGE_WIDTH=80", "CHANGE_PIXEL_DELTA=60", "CHANGE_FRACTION=0.002"], {})]


def test_replay_logs_exactly_what_live_logs(video, base, tmp_path):
    cfg = load_settings(base)
    det = SceneDetector(video, ITEMS, cfg, noisy)
    tune.cache(video[0], cfg, det)
    assert det.calls == simulate.LENGTH_S and det.model is det.scene   # the model is given back
    logs = []
    for n, (sets, thresholds) in enumerate(CASES):
        cfg = load_settings([*base, *sets])
        tune.cache(video[0], cfg, SceneDetector(video, ITEMS, cfg, noisy))   # new only for LOOP_S, CHANGE_WIDTH
        grid = tune.parse_grid([*sets, *(f"MIN_SCORE.{k}={v}" for k, v in thresholds.items())])
        [row] = tune.run([video[0]], [None], grid, base)
        items = {k: replace(it, min_score=thresholds.get(k)) for k, it in ITEMS.items()}
        store = Store(tmp_path / f"live{n}")
        live.run(cfg, items, Places(simulate.ZONES), VideoSource(str(video[0]), cfg.loop_s),
                 SceneDetector(video, items, cfg, noisy), store)
        logs.append([logged(t) for t in store.events()])
        assert [logged(t) for t in row.events[0]] == logs[-1], (sets, thresholds)
    assert len({tuple(log) for log in logs}) == len(CASES)          # every case changed something
    assert len(list(tmp_path.glob("tune/*.npz"))) == 3
    # The wallet's 0.1496 at 13-15 s is under its 0.15 although Sighting.score says 0.15; the decoy
    # wins at 20-23 s with a 0.15 threshold, not with 0.1503.
    wallet = lambda log: [(e[0], e[3], round(e[7] - video[2])) for e in log if e[1] == "wallet"]
    assert wallet(logs[4]) == [("appeared", "table", 0), ("disappeared", None, 15), ("reappeared", "table", 16),
                               ("moved", "kitchen counter", 21), ("moved", "table", 26)]
    assert wallet(logs[5]) == wallet(logs[4])[:3]


def test_cached_gate_sees_what_the_live_gate_sees(video, base):
    cfg = load_settings(base)
    times, smalls, *_ = tune.load(tune.cache(video[0], cfg, SceneDetector(video, ITEMS, cfg, clean)))
    src, gate, replayed = VideoSource(str(video[0]), cfg.loop_s), ChangeGate(cfg), tune.CachedGate(cfg)
    for ts, small in zip(times, smalls):
        f = src.read()
        assert f.ts == ts and gate.check(f.image, ts) == replayed.check(small, ts)
        assert gate.last_fraction == replayed.last_fraction
        gate.mark_detected(f.image, ts)                     # every frame: compare consecutive diffs
        replayed.mark_detected(small, ts)
    assert src.read() is None and len(times) == simulate.LENGTH_S


def test_expect_file(tmp_path):
    p = tmp_path / "demo.expect"
    p.write_text(SCRIPT)
    exp = tune.read_expect(p, ITEMS, Places(simulate.ZONES))
    assert len(exp) == 11 and exp[0] == ("appeared", "keys", "kitchen counter")
    assert exp[6] == ("disappeared", "glasses", None) and exp[-1] == ("moved", "glasses", "kitchen counter")
    # Lines that could never match, so would only ever count as missing (CAMERA.md's zone names are examples)
    for text, error in [
            ("appeared keys\nvanished keys", r"demo.expect:2: want '<kind> <item> \[place\]'"),
            ("appeared keys\n\nappeared glases", r"demo.expect:3: unknown item 'glases' \(items: keys, glasses"),
            ("moved keys counter", r"demo.expect:1: unknown place 'counter' \(zones: kitchen counter, table\)"),
            ("moved keys Table", "unknown place 'Table'"),
            ("disappeared keys table", "'disappeared' takes no place")]:
        p.write_text(text)
        with pytest.raises(SystemExit, match=error):
            tune.read_expect(p, ITEMS, Places(simulate.ZONES))


def event(kind, oid, zone, ts):
    return Transition(oid, oid, kind, None, zone, None, None, "observed", None, 0.9, ts, "", None)


def test_alignment_counts_missing_and_phantom():
    exp = [("appeared", "keys", None), ("moved", "keys", "table"), ("disappeared", "glasses", None)]
    act = [event("appeared", "keys", "kitchen counter", 1), event("disappeared", "remote", None, 2),
           event("moved", "keys", "table", 3), event("disappeared", "glasses", None, 4)]
    pairs = tune.align(exp, act)
    assert [e is None for e, _ in pairs] == [False, True, False, False] and [t for _, t in pairs] == act
    wrong_zone = [event("appeared", "keys", "table", 1), event("moved", "keys", "kitchen counter", 2)]
    assert [(e, t and tune.key(t)) for e, t in tune.align(exp, wrong_zone) if not (e and t)] == [   # both
        (None, ("moved", "keys", "kitchen counter")), (("moved", "keys", "table"), None),
        (("disappeared", "glasses", None), None)]
    assert len(tune.align([], act)) == 4 and tune.align(exp[:1], act[:1]) == [(exp[0], act[0])]
    twice = [act[0], replace(act[0], ts=9)]
    assert tune.align(exp[:1], twice) == [(exp[0], twice[0]), (None, twice[1])]     # the first one matches
    # One pass's events (one ts) match in any order; two passes' only in theirs.
    one_pass = [event("disappeared", "glasses", None, 5), event("moved", "keys", "table", 5)]
    assert tune.score(exp[1:], one_pass) == (0, 0) and tune.score(exp, one_pass) == (1, 0)
    assert tune.score(exp[1:], [replace(one_pass[0], ts=4), one_pass[1]]) == (1, 1)
    assert tune.score(exp[1:], [*one_pass, event("appeared", "bag", "table", 5)]) == (0, 1)


def test_grid_catches_the_hand_pass(video, base, tmp_path, capsys):
    """Two takes of one script, scored against one expect file (CAMERA.md §8)."""
    take2 = tmp_path / "take2.mp4"
    shutil.copy2(video[0], take2)                    # its own name and cache; the same frame times
    for v in (video[0], take2):
        tune.cache(v, load_settings(base), SceneDetector(video, ITEMS, load_settings(base), clean))
    (tmp_path / "demo.expect").write_text(SCRIPT)
    flags = [a for b in base for a in ("--set", b)]
    tune.main(["run", str(video[0]), str(take2), "--expect", str(tmp_path / "demo.expect"),
               "--grid", "missing_after_k=1,3", *flags])
    out = capsys.readouterr().out
    assert "sim.mp4: 0 missing, 0 phantom" in out and "take2.mp4: 0 missing, 0 phantom" in out
    assert "\n  + " not in out and "\n  - " not in out
    recommend = next(ln for ln in out.splitlines() if ln.startswith("recommend:"))
    assert "MISSING_AFTER_K" not in recommend and "--set DETECT_DEVICE=cpu" in recommend
    expected = tune.read_expect(tmp_path / "demo.expect", ITEMS, Places(simulate.ZONES))
    rows = tune.run([video[0]], [expected], tune.parse_grid(["MISSING_AFTER_K=1,3"]), base)
    assert [(r.values, r.scores) for r in rows] == [({"MISSING_AFTER_K": "3"}, [(0, 0)]),
                                                    ({"MISSING_AFTER_K": "1"}, [(0, 2)])]
    phantoms = [tune.key(t) for e, t in tune.align(expected, rows[1].events[0]) if e is None]
    assert phantoms == [("disappeared", "remote", None), ("reappeared", "remote", "table")]
    # The five props appear in the first pass: listed in any order, they still match.
    lines = SCRIPT.splitlines()
    (tmp_path / "demo.expect").write_text("\n".join([lines[5], *lines[:5], *lines[6:]]))
    reordered = tune.read_expect(tmp_path / "demo.expect", ITEMS, Places(simulate.ZONES))
    assert reordered[0] == ("appeared", "bag", "table") and tune.score(reordered, rows[0].events[0]) == (0, 0)
    tune.main(["run", str(take2), *flags])            # no expect file: its events, ready to save as one
    out = capsys.readouterr().out
    assert "take2.mp4: no expect file (--expect), not scored" in out and "nothing to score" in out
    assert "\n    appeared keys kitchen counter " in out and "recommend" not in out


def test_per_item_threshold_override(video, base, tmp_path, capsys):
    weak_wallet = lambda sec, oid: 0.12 if oid == "wallet" else clean(sec, oid)   # under the 0.15 default
    tune.cache(video[0], load_settings(base), SceneDetector(video, ITEMS, load_settings(base), weak_wallet))
    (tmp_path / "sim.expect").write_text(SCRIPT)
    expects = tune.expect_files([video[0]], tmp_path / "sim.expect", ITEMS, Places(simulate.ZONES))
    rows = tune.run([video[0]], expects, tune.parse_grid(["MIN_SCORE.Wallet=0.2,0.15,0.1"]), base)
    assert [(r.values["MIN_SCORE.wallet"], r.scores[0], r.changes) for r in rows] == [
        ("0.1", (0, 0), 1), ("0.15", (1, 0), 0), ("0.2", (1, 0), 1)]   # a tie goes to the current value
    tune.report(rows, [video[0]], expects)
    out = capsys.readouterr().out
    assert '- appeared wallet table' not in out and 'recommend in items.json: "wallet" min_score 0.1' in out
    with pytest.raises(SystemExit, match="under 0.01"):     # the cache has no box under detect(raw=True)'s conf
        tune.run([video[0]], [None], tune.parse_grid(["MIN_SCORE.keys=0.005"]), base)


def test_cache_is_reused(video, base, tmp_path, monkeypatch):
    cfg = load_settings(base)
    det = SceneDetector(video, ITEMS, cfg, clean)
    path = tune.cache(video[0], cfg, det)
    assert path.parent == tmp_path / "tune" and path.name.startswith(f"sim__yolo-world_{cfg.detect_imgsz}_")
    assert tune.cache(video[0], cfg, det) == path and det.calls == simulate.LENGTH_S
    monkeypatch.setattr(tune, "make_detector", lambda *a: pytest.fail("loaded a model for a cached video"))
    tune.main(["cache", str(video[0]), *[a for b in base for a in ("--set", b)]])
    # A new cache for whatever changes the frames read, their tiny copies or the boxes.
    items = load_items(cfg.items_path)
    for k, v in {"loop_s": 0.5, "change_width": 80, "detector": "yoloe", "detect_model": "other.pt",
                 "detect_imgsz": 640, "detect_device": "mps"}.items():
        assert tune.cache_path(video[0], replace(cfg, **{k: v}), items) != path, k
    copy = tmp_path / "sim.mp4"
    shutil.copy2(video[0], copy)
    assert tune.cache_path(copy, cfg, items) == path        # same name, size and time: the same recording
    os.utime(copy, (1e9, 1e9))
    assert tune.cache_path(copy, cfg, items) != path        # recorded again
    save_items({**ITEMS, "bag": replace(ITEMS["bag"], prompt="a brown paper bag")}, tmp_path / "items.json")
    assert tune.cache_path(video[0], cfg, load_items(cfg.items_path)) != path    # new prompt: new cache
    with pytest.raises(SystemExit, match="python -m tools.tune cache"):
        tune.run([video[0]], [None], {}, base)


def registered_room(room_dir):
    """A camera registered over a floor with a 0.75 m table on it (room.glb), and simulate's zones as
    surfaces: the floor under each zone box, 0.9 m high."""
    import json
    import trimesh
    from recall.room import Room
    room_dir.mkdir(parents=True, exist_ok=True)
    z = np.array([0, -0.6, 1.0]) / np.linalg.norm([0, -0.6, 1.0])
    x = np.array([-1.0, 0, 0])
    R = np.array([x, np.cross(z, x), z])
    (room_dir / "camera.json").write_text(json.dumps({
        "width": 1280, "height": 720, "K": [[900, 0, 640], [0, 900, 360], [0, 0, 1]], "dist": [0] * 5,
        "R": R.tolist(), "t": (-R @ [0, 2.0, -0.5]).tolist()}))
    floor = trimesh.Trimesh([[-5, 0, -5], [5, 0, -5], [5, 0, 10], [-5, 0, 10]], [[0, 2, 1], [0, 3, 2]])
    table = trimesh.creation.box(extents=[1.2, 0.75, 1.0])
    table.apply_translation([-0.9, 0.375, 3.0])
    trimesh.util.concatenate([floor, table]).export(room_dir / "room.glb")
    room, surfaces = Room.load(room_dir), []
    for zone in simulate.ZONES:
        x0, y0, x1, y1 = zone.box
        pts = np.array([room.floor_hit(u, v) for u, v in ((x0, y0), (x1, y0), (x1, y1 - 1), (x0, y1 - 1))])
        surfaces.append({"name": zone.name, "box_min": [pts[:, 0].min(), -0.1, pts[:, 2].min()],
                         "box_max": [pts[:, 0].max(), 0.9, pts[:, 2].max()], "prep": "on"})
    (room_dir / "surfaces.json").write_text(json.dumps(surfaces))


def test_replay_with_the_room_logs_exactly_what_live_logs(video, base, tmp_path):
    """Registered room + surfaces: tune sets up the room and places as recall.live.main does, and the
    events match whole, positions (from_xyz / to_xyz) included."""
    registered_room(load_settings(base).room_dir)
    whole = lambda t: {k: v for k, v in t.to_dict().items() if k != "photo_path"}
    for n, sets in enumerate([[], ["MISSING_AFTER_K=1", "MOVE_CONFIRM_N=1", "SURFACE_MARGIN_M=0.3"]]):
        cfg = load_settings([*base, *sets])
        tune.cache(video[0], cfg, SceneDetector(video, ITEMS, cfg, noisy))
        [row] = tune.run([video[0]], [None], tune.parse_grid(sets), base)
        room = live.load_room(cfg)
        places = live.setup_places(cfg, room)
        assert places.in_3d and room.source == "mesh"
        store = Store(tmp_path / f"live{n}")
        live.run(cfg, ITEMS, places, VideoSource(str(video[0]), cfg.loop_s), SceneDetector(video, ITEMS, cfg, noisy),
                 store, room)
        logged_live = [whole(t) for t in store.events()]
        assert any(e["to_xyz"] for e in logged_live) and any(e["from_xyz"] for e in logged_live)
        assert [whole(t) for t in row.events[0]] == logged_live, sets
