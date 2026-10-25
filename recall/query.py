"""Questions over the memory files — templated answers, no LLM (PLAN §8).

Every call reads the files fresh, so it works from another process (REPL, API) while live runs.
When the camera isn't live (heartbeat stale or missing) every answer is stamped "As of <time>:"
— never answer from a frozen world as if it were live.

    where_is(oid)      -> Answer                       (CAMERA.md §6 templates)
    history(oid)       -> [Transition]
    changes_since(t)   -> [Transition]
    state_at(t)        -> {oid: {surface, status, container}}   replayed from the log
    diff(t)            -> [{object, before, now}]
"""
import re
import time
from dataclasses import replace
from datetime import datetime, timedelta

from .models import Answer, Transition
from .places import Places
from .store import Store
from .surfaces import load_surfaces
from .zones import load_zones


def clock(ts) -> str:
    return datetime.fromtimestamp(ts).strftime("%I:%M %p").lstrip("0")


def ago(ts, now) -> str:
    s = max(0, int(now - ts))
    if s < 10:
        return "just now"
    if s < 60:
        return f"{s} seconds ago"
    if s < 3600:
        m = s // 60
        return f"{m} minute{'s' if m != 1 else ''} ago"
    h = s // 3600
    return f"{h} hour{'s' if h != 1 else ''} ago"


def stamp(a: Answer, as_of: float | None) -> Answer:
    """Mark an answer as coming from a frozen world."""
    if as_of is None:
        return a
    sep = "\n" if "\n" in a.text else " "
    return replace(a, text=f"As of {clock(as_of)}:{sep}{a.text}", as_of=as_of)


class Query:
    def __init__(self, cfg, items, store: Store | None = None):
        self.cfg, self.items = cfg, items
        self.store = store or Store(cfg.data_dir)
        self.moss = None  # a MossSearch, when the caller turns semantic item matching on

    def places(self) -> Places:
        """Read fresh, only to word answers: the image zones, and the surfaces once the room is
        registered (as live.setup_places picks them), whose prepositions then come first."""
        registered = (self.cfg.room_dir / "camera.json").exists()
        surfaces = load_surfaces(self.cfg.surfaces_path) if registered else []
        return Places(load_zones(self.cfg.zones_path), surfaces)

    def name(self, oid) -> str:
        return self.items[oid].name if oid in self.items else oid

    def as_of(self, now=None) -> float | None:
        """None while the camera is live; otherwise when the world froze (its last real frame)."""
        now = time.time() if now is None else now
        hb = self.store.heartbeat()
        if hb and hb.get("last_frame_ts") is not None:
            age = now - hb["last_frame_ts"]
            live = hb.get("camera_ok") and -1.0 <= age <= self.cfg.stale_after_s  # -1: clock jitter
            return None if live else hb["last_frame_ts"]
        seen = [o.last_observed for o in self.store.load_state().values() if o.last_observed]
        return max(seen) if seen else None  # no heartbeat: as fresh as the newest sighting

    def where_is(self, oid, now=None, stamped=True) -> Answer:
        now = time.time() if now is None else now
        a = self._where_is(oid, now)
        return stamp(a, self.as_of(now)) if stamped else a

    def _where_is(self, oid, now) -> Answer:
        state, places = self.store.load_state(), self.places()
        o = state.get(oid)
        if not o or o.last_observed is None:
            return Answer("I don't have a record of that.", "none")
        plural = oid in self.items and self.items[oid].plural
        name, be = self.name(oid), "are" if plural else "is"
        they, them = ("They", "them") if plural else ("It", "it")
        seen = o.last_observed

        if o.status == "observed":
            if o.surface:
                text = f"Your {name} {be} {places.phrase(o.surface)} — seen {ago(seen, now)}."
            else:
                text = f"Your {name} {be} in view, but not in a named area — seen {ago(seen, now)}."
        elif o.status == "inferred" and o.container in state:
            c = state[o.container]
            if c.status == "observed":
                where = f", {places.phrase(c.surface)}" if c.surface else ""
                text = (f"Your {name} {be} probably in the {c.name}{where} — inferred, "
                        f"last seen going in {ago(seen, now)}.")
            else:
                text = (f"Your {name} {be} probably in the {c.name}, but I've lost track of the "
                        f"{c.name} too. {they} went in at {clock(seen)}.")
        else:
            text = f"Last seen {places.phrase(o.surface)} at {clock(seen)}. I haven't seen {them} since."
        return Answer(text, o.status, o.confidence, o.photo_path)

    def history(self, oid) -> list[Transition]:
        return [e for e in self.store.events() if e.object == oid]

    def changes_since(self, t) -> list[Transition]:
        return [e for e in self.store.events() if e.ts >= t]

    def state_at(self, t) -> dict[str, dict]:
        """Replay the log up to t. A disappeared object keeps its last-known place (CAMERA.md §5)."""
        snap = {}
        for e in self.store.events():
            if e.ts > t:
                break
            surface = e.from_surface if e.kind == "disappeared" else e.to_surface
            snap[e.object] = {"surface": surface, "status": e.status, "container": e.container}
        return snap

    def diff(self, t) -> list[dict]:
        before, now = self.state_at(t), self.state_at(float("inf"))
        return [{"object": oid, "before": before.get(oid), "now": now.get(oid)}
                for oid in sorted(set(before) | set(now)) if before.get(oid) != now.get(oid)]

    def describe(self, e: Transition, places: Places | None = None) -> str:
        n, k, ph = e.name, e.kind, (places or self.places()).phrase
        cname = self.name(e.container) if e.container else None
        if k in ("appeared", "reappeared"):
            return f"{n} {k} {ph(e.to_surface)}"
        if k == "moved":
            via = f" (inside the {cname}, inferred)" if cname else ""
            return f"{n} moved from {ph(e.from_surface)} to {ph(e.to_surface)}{via}"
        if k == "inferred_contained":
            return f"{n} went into the {cname} — inferred ({e.confidence:.2f})"
        if k == "disappeared":
            return f"{n} disappeared from {ph(e.from_surface)}"
        return f"{n} {k}"


def parse_since(s, now=None) -> float:
    """'10m' '2h' '30s' 'today' '14:30' '2:40pm' (optionally '... ago') -> epoch seconds."""
    now = time.time() if now is None else now
    s = re.sub(r"\s+ago$", "", (s or "today").strip().lower())
    if s == "today":
        return datetime.fromtimestamp(now).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    m = re.fullmatch(r"(\d+)\s*(s|sec|m|min|h|hr|hours?|minutes?|seconds?)", s)
    if m:
        return now - int(m.group(1)) * {"s": 1, "m": 60, "h": 3600}[m.group(2)[0]]
    m = re.fullmatch(r"(\d{1,2}):(\d{2})\s*(am|pm)?", s)
    if m:
        h, mi, ap = int(m.group(1)), int(m.group(2)), m.group(3)
        if ap == "pm" and h < 12:
            h += 12
        if ap == "am" and h == 12:
            h = 0
        d = datetime.fromtimestamp(now).replace(hour=h, minute=mi, second=0, microsecond=0)
        if d.timestamp() > now:
            d -= timedelta(days=1)
        return d.timestamp()
    raise ValueError(f"can't parse time '{s}' (try 10m, 2h, today, 14:30, 2:40pm)")
