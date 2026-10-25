"""A made-up week for the demo, written into this laptop's memory (data/), nowhere else.

    python -m tools.seed_demo        # moves the current memory to data/before-demo-<time>/, writes the week
    (stop live first: it rewrites the memory files while it runs)

To go back: stop live, delete data/state.json, events.jsonl and heartbeat.json, and move the files
from data/before-demo-<time>/ back into data/.

The story: Mom lives alone. Over the last six days her glasses keep ending up in her blue bag,
because she packs it before going out and forgets she did; she finds them later at the dining
table. Her keys wander between the hall table, the kitchen counter and the window sill, and one
night the remote slips under a sofa cushion. This morning the glasses went into the bag again.
Every event carries the object's position in the room (x, y, z in metres, y up); like the live
demo, nothing is tied to a named place. This is FAKE data: say so when showing it.
"""
import argparse
import random
import shutil
from datetime import datetime, timedelta

from recall import config
from recall.models import ObjectState, Transition, clock_s
from recall.store import Store

SPOTS = {  # where things rest in the room, metres
    "hall table": (0.35, 0.80, 0.40),
    "kitchen counter": (0.70, 0.92, 2.60),
    "dining table": (2.50, 0.76, 2.05),
    "armchair": (3.30, 0.48, 2.70),
    "sofa": (3.70, 0.45, 0.90),
    "window sill": (1.80, 1.05, 3.10),
}
NAMES = {"keys": "keys", "glasses": "glasses", "wallet": "wallet", "bag": "bag", "remote": "remote"}

# (days ago, "HH:MM", object, kind, from spot, to spot, container, confidence)
STORY = [
    (6, "08:05", "keys", "appeared", None, "hall table", None, 0.9),
    (6, "08:10", "glasses", "appeared", None, "dining table", None, 0.9),
    (6, "08:12", "wallet", "appeared", None, "hall table", None, 0.9),
    (6, "08:30", "bag", "appeared", None, "armchair", None, 0.9),
    (6, "10:15", "keys", "moved", "hall table", "kitchen counter", None, 0.9),
    (6, "19:40", "remote", "appeared", None, "sofa", None, 0.9),
    (5, "09:20", "glasses", "inferred_contained", "dining table", "armchair", "bag", 0.82),
    (5, "13:05", "glasses", "reappeared", None, "dining table", None, 0.9),
    (5, "17:30", "keys", "moved", "kitchen counter", "hall table", None, 0.9),
    (4, "08:40", "wallet", "inferred_contained", "hall table", "armchair", "bag", 0.78),
    (4, "11:00", "keys", "moved", "hall table", "kitchen counter", None, 0.9),
    (4, "21:15", "remote", "disappeared", "sofa", None, None, 0.5),
    (3, "07:55", "remote", "reappeared", None, "sofa", None, 0.9),
    (3, "10:30", "glasses", "inferred_contained", "dining table", "armchair", "bag", 0.86),
    (3, "15:45", "glasses", "reappeared", None, "dining table", None, 0.9),
    (3, "16:10", "keys", "moved", "kitchen counter", "window sill", None, 0.9),
    (2, "09:05", "glasses", "inferred_contained", "dining table", "armchair", "bag", 0.84),
    (2, "09:50", "bag", "moved", "armchair", "hall table", None, 0.9),
    (2, "09:50", "glasses", "moved", "armchair", "hall table", "bag", 0.84),
    (2, "12:20", "glasses", "reappeared", None, "dining table", None, 0.9),
    (2, "18:00", "wallet", "reappeared", None, "hall table", None, 0.9),
    (1, "08:30", "keys", "moved", "window sill", "hall table", None, 0.9),
    (1, "20:10", "remote", "moved", "sofa", "armchair", None, 0.9),
    (0, "08:20", "glasses", "inferred_contained", "dining table", "hall table", "bag", 0.88),
]
STATUS = {"appeared": "observed", "reappeared": "observed", "moved": "observed",
          "inferred_contained": "inferred", "disappeared": "unknown"}


def at(spot, rng):
    """A spot's position, a few centimetres off each time, like real detections."""
    if spot is None:
        return None
    return [round(v + rng.uniform(-0.03, 0.03), 3) for v in SPOTS[spot]]


def week(now=None, seed=13):
    """(events, final states) of the story, every time before now; the same week for the same day."""
    now = now or datetime.now()
    rng = random.Random(seed)
    events, state = [], {}
    for days_ago, hhmm, oid, kind, frm, to, container, conf in STORY:
        h, m = map(int, hhmm.split(":"))
        when = (now - timedelta(days=days_ago)).replace(hour=h, minute=m, second=0, microsecond=0)
        if when > now:
            continue
        ts = when.timestamp()
        status = "inferred" if container else STATUS[kind]
        t = Transition(oid, NAMES[oid], kind, None, None, at(frm, rng), at(to, rng), status, container,
                       conf, ts, clock_s(ts), None)
        events.append(t)
        o = state.setdefault(oid, ObjectState(oid, NAMES[oid]))
        if kind != "disappeared":  # last seen: where it was put down, or going into the bag
            o.xyz, o.last_observed = t.to_xyz, ts
        o.status, o.container, o.confidence, o.since = status, container, conf, ts
    for o in state.values():  # the camera keeps seeing what sits in plain view
        if o.status == "observed":
            o.last_observed = now.timestamp() - 30
    return events, state


def main():
    ap = argparse.ArgumentParser(description="Write a made-up demo week into this laptop's memory.")
    config.add_args(ap)
    cfg = config.from_args(ap.parse_args())
    store = Store(cfg.data_dir)
    backup = cfg.data_dir / f"before-demo-{datetime.now():%Y%m%d-%H%M%S}"
    moved = []
    for name in (Store.STATE, Store.EVENTS, Store.HEARTBEAT, Store.PHOTOS):
        if (cfg.data_dir / name).exists():
            backup.mkdir(parents=True, exist_ok=True)
            shutil.move(str(cfg.data_dir / name), str(backup / name))
            moved.append(name)
    events, state = week()
    for t in events:
        store.append(t)
    store.save(state, datetime.now().timestamp())
    if moved:
        print(f"moved the old memory ({', '.join(moved)}) to {backup}")
    print(f"wrote {len(events)} events and {len(state)} things to {cfg.data_dir}")


if __name__ == "__main__":
    main()
