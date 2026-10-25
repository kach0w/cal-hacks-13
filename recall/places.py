"""Places: where things are, in the household's words (DESIGN §6.1, CAMERA.md §4).

Once the room is registered and its surfaces are named (data/room/surfaces.json), an object's
place is the named surface its world point falls in. Before that, it is the named zone of the
camera image (data/zones.json) holding its box's base point. Only one kind is active at a time.
Names already in state.json stay as they are when the kind switches (zones, then the room is
registered): live.py warns, and each object logs a 'moved' to its new name when next seen, unless
the memory is reset first (CAMERA.md §4).
"""
from .geometry import area, base_point, contains
from .models import Surface, Zone

SAME_LEVEL_M = 0.10    # surfaces whose tops are this close are at one level (a tray on a table)


def zone_at(zones: list[Zone], bbox, margin=10.0) -> str | None:
    """The image zone holding the box's base point; the smallest wins (a tray on a counter)."""
    pt = base_point(bbox)
    inside = [z for z in zones if contains(z.box, pt, margin)]
    return min(inside, key=lambda z: area(z.box)).name if inside else None


def surface_at(surfaces: list[Surface], xyz, margin=0.05) -> str | None:
    """The surface box holding the world point. Where boxes overlap, the highest surface wins (a
    sofa seat over the floor box drawn up to it), then, of those at its level (tops within
    SAME_LEVEL_M), the smallest (a tray on a table)."""
    if xyz is None:
        return None
    inside = [s for s in surfaces
              if all(lo - margin <= v <= hi + margin for v, lo, hi in zip(xyz, s.box_min, s.box_max))]
    if not inside:
        return None
    level = max(s.top for s in inside)
    footprint = lambda s: (s.box_max[0] - s.box_min[0]) * (s.box_max[2] - s.box_min[2])
    return min((s for s in inside if s.top >= level - SAME_LEVEL_M), key=footprint).name


class Places:
    def __init__(self, zones=(), surfaces=()):
        self.zones, self.surfaces = list(zones), list(surfaces)

    @property
    def in_3d(self) -> bool:
        return bool(self.surfaces)

    def at(self, bbox, xyz, zone_margin=10.0, surface_margin=0.05) -> str | None:
        """The place of an object seen at bbox (pixels) whose base is at xyz (world, or None).
        The margins are memory settings (ZONE_MARGIN_PX, SURFACE_MARGIN_M): Memory passes them."""
        if self.in_3d:
            return surface_at(self.surfaces, xyz, surface_margin)
        return zone_at(self.zones, bbox, zone_margin)

    def names(self) -> list[str]:
        return [p.name for p in (self.surfaces if self.in_3d else self.zones)]

    def phrase(self, name: str | None) -> str:
        """'on the kitchen counter' / 'in the sink' (the surfaces' preposition first, then the
        zones'); None -> 'outside the named areas'."""
        if name is None:
            return "outside the named areas"
        prep = next((p.prep for p in [*self.surfaces, *self.zones] if p.name == name), "on")
        return f"{prep} the {name}"
