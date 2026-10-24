"""The live loop: frame -> change gate -> detect -> zone -> memory -> event log.

    python -m recall.live                       # camera 0
    python -m recall.live --show                # + overlay window (q quits)
    python -m recall.live --src demo.mp4 --show # replay a recording into data/replay/
    python -m recall.live --src demo.mp4 --show --set REPLAY_SPEED=1    # watch it in real time
    python -m recall.live --src demo.mp4 --snapshots data/shots         # every pass as a jpg
    python -m recall.live --src demo.mp4 --set CONTAIN_OVERLAP=0.2 -v   # tune; -v logs every pass

Threads: capture (camera.py) keeps the newest frame; detection (pipeline.py) looks at it once per
LOOP_S; the main thread only draws the window (macOS needs that) and handles Ctrl-C. A video
replays as fast as detection allows into data/replay/, wiped first, so tuning runs are repeatable
and never touch the live memory (ask about it with: python -m recall.ask --replay ...).
"""
import argparse
import logging
import signal
from pathlib import Path

from . import config
from .camera import open_source
from .enroll import load_items
from .memory import Memory
from .models import clock_s
from .perception import make_detector
from .pipeline import Pipeline
from .query import Query
from .store import Store
from .zones import check_size, load_zones

log = logging.getLogger("recall")


def state_table(state, items) -> str:
    rows = []
    for oid in items:
        o = state[oid]
        if o.last_observed is None:
            rows.append(f"  {o.name:<12} {'—':<28} never seen")
            continue
        where = o.zone or "(no named area)"
        if o.status == "inferred":
            where = f"in {state[o.container].name} → {where}"
        rows.append(f"  {o.name:<12} {where:<28} {o.status:<9} {o.confidence:.2f}  since {clock_s(o.since)}")
    return "\n".join(rows)


def load_room(cfg):
    """The Part E hook: pixel_to_world once data/room/camera.json exists, else None."""
    if not (cfg.room_dir / "camera.json").exists():
        return None
    try:
        from .room import Room
        room = Room.load(cfg.room_dir)
    except Exception as e:  # a bad room file must not stop the demo; positions stay 2D
        log.warning("room not loaded (%s); xyz stays empty", e)
        return None
    log.info("room loaded from %s: xyz on", cfg.room_dir)
    return room.pixel_to_world


def show_window(pipeline, source, memory):
    import cv2
    from .overlay import QUIT, key, live_view
    while not pipeline.finished.is_set():
        r = pipeline.latest()
        frame = source.read() if source.live else (r.frame if r else None)  # never advance a video here
        if frame is not None:
            check_size(frame.image, pipeline.cfg)
            frozen_at = pipeline.last_frame_ts if pipeline.frozen else None
            cv2.imshow("recall", live_view(frame.image, r, memory.zones, memory.state, memory.items, frozen_at))
        if key(30) in QUIT:
            break


def run(cfg, items, zones, source, detector, store, show=False, snapshots=None) -> Pipeline:
    """Memory + the detection thread on `source`; blocks until the video ends, q, or Ctrl-C.
    snapshots: a folder that gets the overlay of every detection pass (no window needed)."""
    query = Query(cfg, items, store)

    def on_event(t):
        log.info("%-18s %s   still: %s", t.kind.upper(), query.describe(t, zones), t.still_path)

    memory = Memory(items, zones, store, cfg, pixel_to_world=load_room(cfg), on_event=on_event)
    on_result = None
    if snapshots:
        import cv2
        from .overlay import live_view
        snapshots.mkdir(parents=True, exist_ok=True)

        def on_result(r):
            if r.reason != "static":
                img = live_view(r.frame.image, r, zones, memory.state, items)
                cv2.imwrite(str(snapshots / f"{r.frame.seq:04d}_{r.reason}.jpg"), img)

    pipeline = Pipeline(source, detector, memory, cfg, on_result=on_result)
    old = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        pipeline.start()
        if show:
            show_window(pipeline, source, memory)
        else:
            while not pipeline.finished.wait(0.5):
                pass
    except KeyboardInterrupt:
        log.info("stopping")
    finally:
        for sig in old:  # a second Ctrl-C must not skip the cleanup
            signal.signal(sig, signal.SIG_IGN)
        pipeline.stop()
        source.close()
        if show:
            import cv2
            cv2.destroyAllWindows()
        log.info("final state:\n%s", state_table(memory.state, items))
        for sig, handler in old.items():
            signal.signal(sig, handler)
    return pipeline


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run the camera memory.")
    ap.add_argument("--src", help="camera index or a video file to replay (default: SRC setting)")
    ap.add_argument("--show", action="store_true", help="overlay window (q quits)")
    ap.add_argument("--snapshots", type=Path, metavar="DIR", help="save the overlay of every detection pass")
    config.add_args(ap)
    a = ap.parse_args(argv)
    cfg = config.from_args(a, src=a.src)
    signal.signal(signal.SIGTERM, signal.default_int_handler)  # kill -> same clean stop as Ctrl-C

    items = load_items(cfg.items_path)
    if not items:
        raise SystemExit(f"no items in {cfg.items_path}")
    zones = load_zones(cfg.zones_path)
    if not zones:
        log.warning("no zones yet: answers will say 'not in a named area'. Run: python -m recall.zones onboard")
    log.info("tracking: %s", ", ".join(it.name for it in items.values()))
    log.info("zones:    %s", ", ".join(z.name for z in zones) or "(none)")

    try:
        detector = make_detector(items, cfg)  # slow, may download: before anything is opened or wiped
        detector.warmup(cfg.capture_h, cfg.capture_w)
        source = open_source(cfg)
    except KeyboardInterrupt:
        raise SystemExit("stopped")
    store = Store(cfg.data_dir if source.live else cfg.replay_dir)
    if not source.live:
        store.clear()
        log.info("replaying %s into %s", cfg.src, store.root)
    if run(cfg, items, zones, source, detector, store, a.show, a.snapshots).crashed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
