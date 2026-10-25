"""The live loop: frame -> change gate -> detect -> place (+ xyz) -> memory -> event log.

    python -m recall.live                       # camera 0
    python -m recall.live --show                # + overlay window (q quits)
    python -m recall.live --web                 # + web page: board, ask box, guidance
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
from .places import Places
from .surfaces import load_surfaces
from .zones import check_size, load_zones

log = logging.getLogger("recall")


def state_table(state, items) -> str:
    rows = []
    for oid in items:
        o = state[oid]
        if o.last_observed is None:
            rows.append(f"  {o.name:<12} {'—':<28} never seen")
            continue
        where = o.surface or "(no named area)"
        if o.status == "inferred":
            where = f"in {state[o.container].name} → {where}"
        rows.append(f"  {o.name:<12} {where:<28} {o.status:<9} {o.confidence:.2f}  since {clock_s(o.since)}")
    return "\n".join(rows)


def load_room(cfg):
    """The Part E bridge (recall.room.Room) once data/room/camera.json exists, else None. Also None
    when camera.json was registered at another size than CAPTURE_W x CAPTURE_H: the lens
    calibration only holds at its own size (DESIGN §9), so every xyz would be wrong."""
    if not (cfg.room_dir / "camera.json").exists():
        return None
    try:
        from .room import Room
        room = Room.load(cfg.room_dir)
    except Exception as e:  # a bad room file must not stop the demo; positions stay 2D
        log.warning("room not loaded (%s); xyz stays empty", e)
        return None
    if (room.width, room.height) != (cfg.capture_w, cfg.capture_h):
        log.error("room not used: camera.json is registered at %dx%d but CAPTURE is %dx%d, so xyz stays "
                  "empty. Capture at %dx%d, or calibrate and register the camera again at the new size",
                  room.width, room.height, cfg.capture_w, cfg.capture_h, room.width, room.height)
        return None
    log.info("room loaded from %s: xyz on (source=%s)", cfg.room_dir, room.source)
    return room


def setup_places(cfg, room) -> Places:
    """Scanned surfaces once the room is registered and they're named, else the image zones."""
    zones, surfaces = load_zones(cfg.zones_path), load_surfaces(cfg.surfaces_path)
    if surfaces and room is None:
        log.warning("%s needs the registered room (data/room/camera.json): using image zones", cfg.surfaces_path)
        surfaces = []
    elif surfaces and room.source == "floor":  # no mesh: every point would land on the floor
        log.warning("%s needs room.glb, which wasn't read (%s): using image zones", cfg.surfaces_path,
                    room.mesh_error or "missing")
        surfaces = []
    elif room is not None and not surfaces:
        log.warning("room registered but no surfaces named: using image zones. Run: python -m recall.surfaces name")
    places = Places(zones, surfaces)
    if not places.names():
        log.warning("no places yet: answers will say 'not in a named area'. Run: python -m recall.zones onboard")
    log.info("places:   %s %s", "surfaces" if places.in_3d else "zones", ", ".join(places.names()) or "(none)")
    return places


def warn_unknown_places(memory):
    """Place names in the loaded state that no current place has: left from the other kind (image
    zones, before the room was registered) or from a renamed place. Not cleaned up: each such
    object logs a 'moved' to its new name when it's next seen (CAMERA.md §4)."""
    names = set(memory.places.names())
    old = sorted({p for o in memory.state.values() for p in (o.surface, o.pending_surface) if p and p not in names})
    if old:
        log.warning("%s names places that aren't today's %s: %s. Those objects log a 'moved' when next seen; "
                    "to start clean, stop and delete state.json, events.jsonl, heartbeat.json and photos/ there",
                    memory.store.root / memory.store.STATE, "surfaces" if memory.places.in_3d else "zones",
                    ", ".join(old))


def show_window(pipeline, source, memory, room=None):
    import cv2
    from .overlay import QUIT, key, live_view
    while not pipeline.finished.is_set():
        r = pipeline.latest()
        frame = source.read() if source.live else (r.frame if r else None)  # never advance a video here
        if frame is not None:
            check_size(frame.image, pipeline.cfg)
            frozen_at = pipeline.last_frame_ts if pipeline.frozen else None
            cv2.imshow("recall", live_view(frame.image, r, memory.places, memory.state, memory.items, frozen_at, room))
        if key(30) in QUIT:
            break


def run(cfg, items, places, source, detector, store, room=None, show=False, snapshots=None,
        web=False) -> Pipeline:
    """Memory + the detection thread on `source`; blocks until the video ends, q, or Ctrl-C.
    room: the registered room (xyz + surfaces), or None. snapshots: a folder that gets the
    overlay of every detection pass (no window needed)."""
    query = Query(cfg, items, store)

    def on_event(t):
        log.info("%-18s %s   photo: %s", t.kind.upper(), query.describe(t, places), t.photo_path)

    memory = Memory(items, places, store, cfg, pixel_to_world=room.pixel_to_world if room else None,
                    on_event=on_event)
    warn_unknown_places(memory)
    on_result = None
    if snapshots:
        import cv2
        from .overlay import live_view
        snapshots.mkdir(parents=True, exist_ok=True)

        def on_result(r):
            if r.reason != "static":
                img = live_view(r.frame.image, r, places, memory.state, items, room=room)
                cv2.imwrite(str(snapshots / f"{r.frame.seq:04d}_{r.reason}.jpg"), img)

    pipeline = Pipeline(source, detector, memory, cfg, on_result=on_result)
    old = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        pipeline.start()
        if web:
            from .web import Web
            print(f"Recall web page: {Web(cfg, items, memory, pipeline, places, room).start()}")
        if show:
            show_window(pipeline, source, memory, room)
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
    ap.add_argument("--web", action="store_true", help="web page: board, ask box, left/right guidance")
    ap.add_argument("--snapshots", type=Path, metavar="DIR", help="save the overlay of every detection pass")
    config.add_args(ap)
    a = ap.parse_args(argv)
    cfg = config.from_args(a, src=a.src)
    signal.signal(signal.SIGTERM, signal.default_int_handler)  # kill -> same clean stop as Ctrl-C

    items = load_items(cfg.items_path)
    if not items:
        raise SystemExit(f"no items in {cfg.items_path}")
    log.info("tracking: %s", ", ".join(it.name for it in items.values()))
    room = load_room(cfg)
    places = setup_places(cfg, room)

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
    if run(cfg, items, places, source, detector, store, room, a.show, a.snapshots, a.web).crashed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
