"""Change detection (DESIGN §6, CAMERA.md §5): one detection pass in, Transitions out.

    first sighting                  -> appeared
    seen in a different zone        -> moved, after MOVE_CONFIRM_N looks in a row (+1 if the new
                                       zone is None: objects in mid-air while carried)
    missing MISSING_AFTER_K looks   -> gone. If its last box was >= CONTAIN_OVERLAP inside an
                                       enrolled container's box in that same frame:
                                       inferred_contained (status inferred, zone = container's,
                                       confidence clipped 0.5-0.9); else disappeared (unknown,
                                       zone kept as last known)
    container moves                 -> everything inferred inside it moves too (still inferred)
    seen again after gone           -> reappeared, observed, container cleared
    always                          -> never promote inferred -> observed except by seeing it
"""
import logging

from .geometry import base_point, overlap_frac
from .models import ObjectState, Transition, clock_s
from .overlay import still
from .zones import zone_at

log = logging.getLogger(__name__)


class Memory:
    def __init__(self, items, zones, store, cfg, pixel_to_world=None, on_event=None):
        """pixel_to_world: optional (u, v) -> (x, y, z) | None, the Part E hook (room.Room)."""
        self.items, self.zones, self.store, self.cfg = items, zones, store, cfg
        self.pixel_to_world, self.on_event = pixel_to_world, on_event
        self.state: dict[str, ObjectState] = store.load_state()
        for oid, it in items.items():
            o = self.state.setdefault(oid, ObjectState(oid, it.name))
            o.name, o.misses = it.name, 0
        self._last_frame = {}  # oid -> image it was last seen in (RAM only; evidence for stills)

    def update(self, sightings, frame, ts, held=()) -> list[Transition]:
        """Feed one detection pass. `held`: items re-id couldn't decide on — neither seen nor missed."""
        seen = {s.object_id: s for s in sightings if s.object_id in self.items}
        context = [{"object_id": s.object_id, "bbox": s.bbox} for s in seen.values()]
        events = []
        for oid in self.items:
            o = self.state[oid]
            if oid in seen:
                events += self._seen(o, seen[oid], context, frame, ts)
            elif oid not in held:
                events += self._missing(o, ts)
        events += self._follow_containers(frame, ts)
        self.store.save(self.state, ts)
        return events

    def zone_of(self, bbox) -> str | None:
        return zone_at(self.zones, bbox, self.cfg.zone_margin_px)

    def _seen(self, o, s, context, frame, ts):
        zone = self.zone_of(s.bbox)
        xyz = self.pixel_to_world(*base_point(s.bbox)) if self.pixel_to_world else None
        first_time = o.last_observed is None
        was_gone = o.status != "observed"
        o.misses, o.confidence, o.last_observed = 0, s.score, ts
        o.last_cam, o.last_bbox = s.cam, s.bbox
        o.context = [c for c in context if c["object_id"] != o.object_id]
        if self.pixel_to_world:
            o.xyz = None if xyz is None else [round(float(v), 3) for v in xyz]
        self._last_frame[o.object_id] = frame

        if first_time or was_gone:
            frm = None if first_time else o.zone
            o.status, o.zone, o.container, o.since = "observed", zone, None, ts
            o.pending_zone, o.pending_count = None, 0
            return [self._commit(o, "appeared" if first_time else "reappeared", frm, zone, ts, frame)]

        if zone == o.zone:
            o.pending_zone, o.pending_count = None, 0
            return []
        if o.pending_zone == zone:
            o.pending_count += 1
        else:
            o.pending_zone, o.pending_count = zone, 1
        if o.pending_count < self.cfg.move_confirm_n + (1 if zone is None else 0):
            return []
        frm = o.zone
        o.zone, o.since, o.pending_zone, o.pending_count = zone, ts, None, 0
        return [self._commit(o, "moved", frm, zone, ts, frame)]

    def _missing(self, o, ts):
        if o.last_observed is None or o.status != "observed":
            return []
        o.misses += 1
        if o.misses < self.cfg.missing_after_k:
            return []  # not gone, just blinked
        # Gone. Look at the last picture it was in and ask what it went into.
        frame = self._last_frame.get(o.object_id)
        cid, frac = self._most_overlapping_container(o)
        frm = o.zone
        if cid:
            c = self.state[cid]
            o.status, o.container, o.since = "inferred", cid, o.last_observed
            o.confidence, o.zone, o.xyz = round(min(0.9, max(0.5, frac)), 2), c.zone, c.xyz
            return [self._commit(o, "inferred_contained", frm, c.zone, ts, frame)]
        o.status, o.container, o.since = "unknown", None, o.last_observed
        return [self._commit(o, "disappeared", frm, None, ts, frame)]

    def _most_overlapping_container(self, o):
        """Only enrolled containers qualify: people are never items, so a hand can't 'contain' keys."""
        best, best_frac = None, 0.0
        for c in o.context:
            cid = c["object_id"]
            if cid == o.object_id or cid not in self.items or not self.items[cid].container:
                continue
            f = overlap_frac(o.last_bbox, c["bbox"])
            if f > best_frac:
                best, best_frac = cid, f
        return (best, best_frac) if best_frac >= self.cfg.contain_overlap else (None, 0.0)

    def _follow_containers(self, frame, ts):
        """Glasses in the bag; the bag moves -> the glasses move with it (still inferred)."""
        events = []
        for o in self.state.values():
            if o.status != "inferred" or o.container not in self.state:
                continue
            c = self.state[o.container]
            if c.status != "observed":
                continue
            o.xyz = c.xyz
            if c.zone != o.zone:
                frm, o.zone = o.zone, c.zone
                evidence = frame if c.last_observed == ts else None
                events.append(self._commit(o, "moved", frm, c.zone, ts, evidence,
                                           c.last_bbox, f"{o.name} (in {c.name}) moved"))
        return events

    def _commit(self, o, kind, frm, to, ts, frame, bbox=None, label=None) -> Transition:
        still_path = None
        if frame is not None:
            img = still(frame, bbox or o.last_bbox, label or f"{o.name} {kind}")
            still_path = o.still_path = self.store.save_still(img, f"{int(ts * 1000)}_{o.object_id}_{kind}")
        t = Transition(o.object_id, o.name, kind, frm, to, o.status, o.container, o.confidence,
                       ts, clock_s(ts), still_path)
        self.store.append(t)
        if self.on_event:
            try:
                self.on_event(t)
            except Exception:  # a display hook must never break the memory mid-update
                log.exception("on_event failed for %s %s", t.kind, t.object_id)
        return t
