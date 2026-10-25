"""Task 3 (CAMERA.md §8): tune the gate and memory settings on recordings — offline, fast, exact.

    python -m tools.tune cache data/demo1.mp4 [data/demo2.mp4 ...] [--set DETECTOR=yoloe ...]
    python -m tools.tune run data/demo1.mp4 [...] [--expect data/demo1.expect]
        [--grid MISSING_AFTER_K=2,3,4] [--grid CONTAIN_OVERLAP=0.2,0.3] [--grid MIN_SCORE.glasses=0.1,0.2]

cache: every box the model returns (raw: any score) on each frame recall.live --src would read, +
the change gate's tiny frame -> data/tune/, keyed by video, detector, imgsz, prompts, LOOP_S and
CHANGE_WIDTH (cache each value of those you want to grid over).
run: the real ChangeGate, UltralyticsDetector.detect (on the cached boxes) and Memory, per
combination of --grid values (settings, or MIN_SCORE.<item>): exactly the events recall.live would
log, in milliseconds. Each video is scored against --expect (one file for every take), else its own
<video>.expect: one event per line, in order (one pass's events in any order), item ids from
items.json, place optional (a surface once the room is registered, else a zone, named as in
surfaces.json / zones.json). Prints a row per combination (fewest missing +
phantom events, then fewest changes, first), the best one's diff and its flags. A video without an
expect file prints its events in that format, ready to check and save as one:

    appeared keys kitchen counter
    moved keys table
    inferred_contained glasses table      # into the bag
"""
import argparse
import functools
import hashlib
import itertools
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, fields, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from recall import config
from recall.camera import VideoSource
from recall.enroll import load_items
from recall.gate import ChangeGate
from recall.memory import Memory
from recall.models import Transition
from recall.perception import DEFAULT_MODEL, UltralyticsDetector, _boxes, make_detector
from recall.store import Store
from recall.live import load_room, setup_places

log = logging.getLogger("recall")
KINDS = ("appeared", "reappeared", "moved", "disappeared", "inferred_contained")
PER_ITEM = "MIN_SCORE."
CACHE_KEY = ("loop_s", "change_width", "detector", "detect_model", "detect_imgsz", "detect_device")


class CachedGate(ChangeGate):
    """The real gate, fed the tiny blurred frames ChangeGate._small made when the cache was built."""

    def _small(self, small):
        return small


class Recorder:
    """Wraps a detector's model to keep every box it returns: [(xyxy, class, score)] per call."""

    def __init__(self, model):
        self.model, self.frames, self.conf = model, [], 0.0

    def predict(self, image, conf, **kw):
        results = self.model.predict(image, conf=conf, **kw)
        self.frames.append(list(_boxes(results[0])))
        self.conf = conf
        return results


class _Tensor(np.ndarray):
    """What _boxes() reads from a result: tensors, via .cpu().numpy()."""

    def cpu(self):
        return self

    def numpy(self):
        return self


class CachedModel:
    """The model on replay: this frame's cached boxes above conf, as ultralytics keeps them (float32
    score > float32 conf, in its order). The cache has no box under the conf it was made with."""

    def __init__(self, floor):
        self.floor, self.boxes = floor, None

    def predict(self, image, conf, **kw):
        if conf < self.floor:
            raise SystemExit(f"a min score under {self.floor} needs boxes the cache doesn't have")
        keep = self.boxes[2] > np.float32(conf)
        xyxy, cls, score = (a[keep].view(_Tensor) for a in self.boxes)
        return [SimpleNamespace(boxes=SimpleNamespace(xyxy=xyxy, cls=cls, conf=score))]


def settings(overrides) -> config.Settings:
    try:
        return config.load_settings(overrides)
    except ValueError as e:
        raise SystemExit(str(e))


def threshold(item, cfg) -> float:
    return cfg.default_min_score if item.min_score is None else item.min_score


def cache_path(video, cfg, items) -> Path:
    if not Path(video).is_file():
        raise SystemExit(f"no video {video}")
    model = cfg.detect_model or DEFAULT_MODEL.get(cfg.detector, "")
    stamp = lambda p: [p.stat().st_size, p.stat().st_mtime] if p.is_file() else None
    key = [[str(getattr(cfg, k)) for k in CACHE_KEY], model, stamp(Path(model)), stamp(Path(video)),
           [[oid, it.name, it.prompt] for oid, it in items.items()]]
    digest = hashlib.sha1(json.dumps(key).encode()).hexdigest()[:10]
    return cfg.data_dir / "tune" / f"{Path(video).stem}__{cfg.detector}_{cfg.detect_imgsz}_{digest}.npz"


def cache(video, cfg, detector) -> Path:
    """Detect (raw) once on each frame recall.live --src would read; nothing to do if cached already."""
    items = load_items(cfg.items_path)
    path = cache_path(video, cfg, items)
    if path.exists():
        log.info("already cached: %s", path)
        return path
    gate, src, t0, ts, small = ChangeGate(cfg), VideoSource(str(video), cfg.loop_s), time.time(), [], []
    rec = detector.model = Recorder(detector.model)
    try:
        while (f := src.read()) is not None:
            detector.detect(f.image, f.ts, raw=True)
            ts.append(f.ts)
            small.append(gate._small(f.image))
    finally:
        src.close()
        detector.model = rec.model
    rows = [(n, *box) for n, boxes in enumerate(rec.frames) for box in boxes]
    frame, xyxy, cls, score = zip(*rows) if rows else ([],) * 4
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(".part.npz")   # an interrupted run never leaves a cache that looks complete
    np.savez_compressed(part, ts=np.array(ts, float), small=np.array(small, np.uint8), frame=np.array(frame, int),
                        xyxy=np.array(xyxy, float).reshape(-1, 4), cls=np.array(cls, int),
                        score=np.array(score, np.float32), conf=rec.conf,
                        labels=np.array([label or "" for label in detector.labels], str))
    os.replace(part, path)
    log.info("cached %d frames of %s in %.1f s -> %s", len(ts), video, time.time() - t0, path)
    return path


def load(path):
    """-> (frame times, ChangeGate._small per frame, the model's (xyxy, class, score) arrays per frame,
    the detector's labels, the conf the boxes were asked for with)"""
    z = np.load(path)
    cut = np.searchsorted(z["frame"], np.arange(1, len(z["ts"])))
    boxes = list(zip(*(np.split(z[k], cut) for k in ("xyxy", "cls", "score"))))
    labels = [label or None for label in z["labels"].tolist()]
    return z["ts"].tolist(), z["small"], boxes, labels, float(z["conf"])


def replay(cached, cfg, items, places, room=None) -> list[Transition]:
    """Pipeline.step on a video, the model's boxes looked up; frame=None, so no photos."""
    ts, smalls, boxes, labels, conf = cached
    model, gate, events = CachedModel(conf), CachedGate(cfg), []
    detector = UltralyticsDetector(items, cfg, model, labels)
    with tempfile.TemporaryDirectory() as tmp:
        memory = Memory(items, places, Store(tmp), cfg, room.pixel_to_world if room else None)
        for t, small, frame_boxes in zip(ts, smalls, boxes):
            if gate.check(small, t)[0]:
                model.boxes = frame_boxes
                dets = detector.detect(None, t)
                gate.mark_detected(small, t)
                events += memory.update(dets.sightings, None, t, held=dets.held)
    return events


# --- expectations ---------------------------------------------------------------------------------
def read_expect(path, items, places) -> list[tuple]:
    """[(kind, item, to_place | None)] from '<kind> <item id> [place name]' lines."""
    names, out = places.names(), []

    def fail(why):
        raise SystemExit(f"{path}:{n}: {why}")

    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        words = line.split("#", 1)[0].split()
        if not words:
            continue
        kind, oid, place = words[0], words[1] if len(words) > 1 else None, " ".join(words[2:]) or None
        if kind not in KINDS or oid is None:
            fail(f"want '<kind> <item> [place]', kind one of {', '.join(KINDS)}")
        if oid not in items:
            fail(f"unknown item {oid!r} (items: {', '.join(items)})")
        if place and kind == "disappeared":
            fail("'disappeared' takes no place: the item is gone")
        if place and place not in names:
            kinds = "surfaces" if places.in_3d else "zones"
            fail(f"unknown place {place!r} ({kinds}: {', '.join(names) or 'none yet'})")
        out.append((kind, oid, place))
    return out


def key(t: Transition) -> tuple:
    return t.kind, t.object, t.to_surface


def align(expected, events) -> list[tuple]:
    """Most matches, in order -> [(expected | None, Transition | None)]. The events of one pass (one ts)
    match in any order: memory.update emits them in items.json order. No expected place matches any."""
    after = [(0, [(e, None) for e in expected[i:]]) for i in range(len(expected) + 1)]
    for one_pass in reversed([list(g) for _, g in itertools.groupby(events, key=lambda t: t.ts)]):
        after = _with_pass(expected, one_pass, after)
    return after[0][1]


def _with_pass(expected, group, after) -> list[tuple]:
    """after[i]: (matches, pairs) of expected[i:] against the later passes -> the same from this one."""
    ok = lambda e, t: e[:2] == (t.kind, t.object) and e[2] in (None, t.to_surface)

    @functools.cache
    def best(i, used):   # bit k of used: group[k] is matched already
        free = [(k, t) for k, t in enumerate(group) if not used >> k & 1]
        options = [(after[i][0], [(None, t) for _, t in free] + after[i][1])]   # the rest of the pass: phantoms
        if i < len(expected):
            m, pairs = best(i + 1, used)
            options.append((m, [(expected[i], None), *pairs]))                    # expected[i] missing
            for k, t in free:
                if ok(expected[i], t):
                    m, pairs = best(i + 1, used | 1 << k)
                    options.insert(0, (m + 1, [(expected[i], t), *pairs]))         # a match wins a tie
        return max(options, key=lambda o: o[0])

    return [best(i, 0) for i in range(len(expected) + 1)]


def score(expected, events) -> tuple[int, int]:
    """(missing, phantom) events."""
    pairs = align(expected, events)
    return sum(t is None for _, t in pairs), sum(e is None for e, _ in pairs)


# --- the grid -------------------------------------------------------------------------------------
@dataclass
class Row:
    values: dict           # {NAME: value as typed}
    cfg: config.Settings
    changes: int           # grid values that differ from the current ones
    events: list           # per video
    scores: list           # per video: (missing, phantom), None without an expect file
    total: int


def parse_grid(specs) -> dict[str, list[str]]:
    grid = {}
    for spec in specs:
        name, sep, values = spec.partition("=")
        name = name.strip().upper()
        name = PER_ITEM + name[len(PER_ITEM):].lower() if name.startswith(PER_ITEM) else name
        if not sep or name in grid:
            raise SystemExit(f"bad --grid {spec!r}: NAME=v1,v2,... once per name")
        grid[name] = [v.strip() for v in values.split(",")]
    return grid


def resolved(name, cfg, items):
    return threshold(items[name[len(PER_ITEM):]], cfg) if name.startswith(PER_ITEM) else getattr(cfg, name.lower())


def combos(grid, base_set):
    """Every combination of grid values -> (values, Settings, items with those thresholds)."""
    for values in itertools.product(*grid.values()):
        values = dict(zip(grid, values))
        cfg = settings([*base_set, *(f"{k}={v}" for k, v in values.items() if not k.startswith(PER_ITEM))])
        items = load_items(cfg.items_path)
        for k, v in values.items():
            if k.startswith(PER_ITEM):
                oid = k[len(PER_ITEM):]
                if oid not in items or not v.replace(".", "", 1).isdigit():
                    raise SystemExit(f"bad --grid {k}={v}: want MIN_SCORE.<item in {cfg.items_path}>=<number>")
                items[oid] = replace(items[oid], min_score=float(v))
        yield values, cfg, items


def run(videos, expects, grid, base_set, places=None, room=None) -> list[Row]:
    """Every video replayed under every combination, best first. places, room: as recall.live sets
    them up from the base settings (xyz + surfaces once registered); set up here if not given."""
    base, loaded, rows = settings(base_set), {}, []
    base_items = load_items(base.items_path)
    if places is None:
        room = load_room(base)
        places = setup_places(base, room)
    for values, cfg, items in list(combos(grid, base_set)):   # every combination checked before any replay
        events = []
        for video in videos:
            path = cache_path(video, cfg, items)
            if path not in loaded:
                if not path.exists():
                    live = settings([])
                    flags = "".join(f" --set {k.upper()}={getattr(cfg, k)}" for k in (*CACHE_KEY, "data_dir",
                                    "items_path") if getattr(cfg, k) != getattr(live, k))
                    raise SystemExit(f"{video} isn't cached for these settings; first run:\n"
                                     f"  python -m tools.tune cache {video}{flags}")
                loaded[path] = load(path)
            events.append(replay(loaded[path], cfg, items, places, room))
        changes = sum(resolved(k, cfg, items) != resolved(k, base, base_items) for k in values)
        scores = [None if exp is None else score(exp, ev) for exp, ev in zip(expects, events)]
        rows.append(Row(values, cfg, changes, events, scores, sum(sum(s) for s in scores if s)))
    return sorted(rows, key=lambda r: (r.total, r.changes))


def expect_files(videos, given, items, places) -> list:
    """Each video's expected events: the --expect file (one for every take), else its own
    <video>.expect, else None."""
    if given and not given.is_file():
        raise SystemExit(f"--expect {given}: no such file")
    paths = [given or v.with_suffix(".expect") for v in videos]
    return [read_expect(p, items, places) if p.is_file() else None for p in paths]


def report(rows, videos, expects):
    head = ["#", *rows[0].values, *(f"{v.name} missing/phantom" for v in videos), "total"]
    cells = [[str(n), *r.values.values(), *("-" if s is None else f"{s[0]}/{s[1]}" for s in r.scores),
              str(r.total)] for n, r in enumerate(rows, 1)]
    widths = [max(map(len, col)) for col in zip(head, *cells)]
    for cols in [head, *cells]:
        print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    best, text = rows[0], lambda k: " ".join(w for w in k if w)
    for video, exp, events in zip(videos, expects, best.events):
        if exp is None:
            print(f"\n== {video.name}: no expect file (--expect), not scored; its events")
            pairs = [(key(t), t) for t in events]
        else:
            print(f"\n== {video.name}: %d missing, %d phantom  (- expected, not logged; + logged, not expected)"
                  % score(exp, events))
            pairs = align(exp, events)
        for e, t in pairs:
            mark = " " if e and t else "+" if t else "-"
            print(f"  {mark} {text(key(t) if t else e):<44} # {t.time if t else 'missing'}")
    if all(e is None for e in expects):
        print("\nno expect files: nothing to score")
        return
    live, file_items = settings([]), load_items(best.cfg.items_path)
    flags = [f"--set {f.name.upper()}={getattr(best.cfg, f.name)}" for f in fields(best.cfg)
             if getattr(best.cfg, f.name) != getattr(live, f.name)]
    print("\nrecommend:", " ".join(flags) or "no changes to the settings")
    for k, v in best.values.items():
        if k.startswith(PER_ITEM) and float(v) != resolved(k, best.cfg, file_items):
            print(f"recommend in {best.cfg.items_path.name}: \"{k[len(PER_ITEM):]}\" min_score {v}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Tune gate + memory settings on cached detections (task 3).")
    ap.add_argument("cmd", choices=["cache", "run"])
    ap.add_argument("videos", nargs="+", type=Path)
    ap.add_argument("--expect", type=Path, metavar="FILE",
                    help="run: the expected events of every video (default: each one's own <video>.expect)")
    ap.add_argument("--grid", action="append", default=[], metavar="NAME=V1,V2,...",
                    help="run: values to try (repeatable); NAME is a setting or MIN_SCORE.<item>")
    config.add_args(ap)
    a = ap.parse_args(argv)
    cfg = config.from_args(a)
    if a.cmd == "cache":
        items, detector = load_items(cfg.items_path), None
        if not all([cache_path(v, cfg, items).exists() for v in a.videos]):   # every video checked first
            detector = make_detector(items, cfg)
            detector.warmup(cfg.capture_h, cfg.capture_w)
        for v in a.videos:
            cache(v, cfg, detector)
        return
    room = load_room(cfg)
    places = setup_places(cfg, room)
    expects = expect_files(a.videos, a.expect, load_items(cfg.items_path), places)
    t0 = time.time()
    rows = run(a.videos, expects, parse_grid(a.grid), a.set, places, room)
    log.info("replayed %d settings x %d videos in %.1f s", len(rows), len(a.videos), time.time() - t0)
    report(rows, a.videos, expects)


if __name__ == "__main__":
    main()
