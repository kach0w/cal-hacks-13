"""The seams (DESIGN §7, CAMERA.md §2): typed records whose to_dict() is exactly the on-disk JSON."""
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime

Box = list[float]  # [x1, y1, x2, y2] in pixels


def clock_s(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")


# Keys of memory files written before the rename to DESIGN §7's seams, read as the new ones.
RENAMED = {"object_id": "object", "zone": "surface", "pending_zone": "pending_surface",
           "from_zone": "from_surface", "to_zone": "to_surface", "still_path": "photo_path"}


def renamed(d: dict) -> dict:
    return {RENAMED.get(k, k): v for k, v in d.items()}


class _Record:
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in renamed(d).items() if k in names})


@dataclass
class Item(_Record):
    """One entry of items.json: a thing worth tracking. Only these are ever committed to memory."""
    id: str
    name: str
    prompt: str                        # what the detector is told to look for
    aliases: list[str] = field(default_factory=list)   # other words a question can use
    plural: bool = False               # "are"/"them" in answers
    container: bool = False            # things can go inside it
    min_score: float | None = None     # set by calibration; None -> DEFAULT_MIN_SCORE
    reference: str | None = None       # background-removed photo, relative to data/
    extra: dict = field(default_factory=dict)   # calibration bookkeeping, kept as-is

    @classmethod
    def from_dict(cls, id: str, d: dict) -> "Item":
        names = {f.name for f in fields(cls)} - {"id", "extra"}
        known = {k: v for k, v in d.items() if k in names}
        known.setdefault("name", id)
        known.setdefault("prompt", known["name"])
        return cls(id=id, **known, extra={k: v for k, v in d.items() if k not in names})

    def to_dict(self) -> dict:
        d = asdict(self)
        del d["id"]
        extra = d.pop("extra")
        return {**d, **extra}


@dataclass(frozen=True)
class Sighting(_Record):
    """The best box for one enrolled item in one detection pass (DESIGN §7: object = its id)."""
    object: str
    bbox: Box
    score: float
    ts: float
    cam: int = 0


@dataclass
class Zone(_Record):
    """A named box of the camera image ("kitchen counter"): the places before the room is registered."""
    name: str
    box: Box
    prep: str = "on"                   # on | in | under | next to ...
    source: str = "manual"             # detected | manual
    cam: int = 0


@dataclass
class Surface(_Record):
    """A named region of the scanned room (DESIGN §6.1): an axis-aligned box in the world frame."""
    name: str
    box_min: list[float]
    box_max: list[float]
    prep: str = "on"
    top: float | None = None           # the surface's own height (y); None -> box_min's (a hand-made box)

    def __post_init__(self):
        if self.top is None:
            self.top = self.box_min[1]

    @classmethod
    def from_dict(cls, d: dict) -> "Surface":
        d = {**d}
        for old, new in (("min", "box_min"), ("max", "box_max")):  # handoffAB.md's zones3d.json
            if old in d:
                d.setdefault(new, d.pop(old))
        return super().from_dict(d)


@dataclass
class ObjectState(_Record):
    """One WorldState entry (DESIGN §7). status: observed | inferred | unknown; never seen =
    last_observed None. surface: the named place (a scanned surface, or an image zone before Part E)."""
    object: str
    name: str
    xyz: list[float] | None = None     # world point at the box's base; None until the room is registered
    surface: str | None = None
    status: str = "unknown"
    confidence: float = 0.0
    since: float | None = None
    last_observed: float | None = None
    container: str | None = None
    last_cam: int | None = None
    last_bbox: Box | None = None
    photo_path: str | None = None
    misses: int = 0
    pending_surface: str | None = None  # a new place waiting for MOVE_CONFIRM_N looks
    pending_count: int = 0
    context: list[dict] = field(default_factory=list)  # [{object, bbox}] seen with it last time

    @classmethod
    def from_dict(cls, d: dict) -> "ObjectState":
        o = super().from_dict(d)
        o.context = [renamed(c) for c in o.context]
        return o


@dataclass(frozen=True)
class Transition(_Record):
    """One line of events.jsonl (DESIGN §7). kind: appeared | reappeared | moved | disappeared |
    inferred_contained"""
    object: str
    name: str
    kind: str
    from_surface: str | None
    to_surface: str | None
    from_xyz: list[float] | None
    to_xyz: list[float] | None
    status: str
    container: str | None
    confidence: float
    ts: float
    time: str                          # clock_s(ts), for people reading the log
    photo_path: str | None

    @classmethod
    def from_dict(cls, d: dict) -> "Transition":
        return super().from_dict({"from_xyz": None, "to_xyz": None, **d})  # lines from before xyz


@dataclass(frozen=True)
class Answer(_Record):
    """status: observed | inferred | unknown | none (no record) | info (lists, histories)."""
    text: str
    status: str
    confidence: float = 0.0
    photo_path: str | None = None
    as_of: float | None = None         # set when the camera isn't live: the world froze then
