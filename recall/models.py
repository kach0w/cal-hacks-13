"""The seams (CAMERA.md §2): typed records whose to_dict() is exactly the on-disk JSON."""
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime

Box = list[float]  # [x1, y1, x2, y2] in pixels


def clock_s(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")


class _Record:
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


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
    """The best box for one enrolled item in one detection pass."""
    object_id: str
    bbox: Box
    score: float
    ts: float
    cam: int = 0


@dataclass
class Zone(_Record):
    """A named region of the camera image ("kitchen counter")."""
    name: str
    box: Box
    prep: str = "on"                   # on | in | under | next to ...
    source: str = "manual"             # detected | manual
    cam: int = 0


@dataclass
class ObjectState(_Record):
    """One WorldState entry. status: observed | inferred | unknown; never seen = last_observed None."""
    object_id: str
    name: str
    zone: str | None = None
    status: str = "unknown"
    confidence: float = 0.0
    since: float | None = None
    last_observed: float | None = None
    container: str | None = None
    last_cam: int | None = None
    last_bbox: Box | None = None
    xyz: list[float] | None = None     # world point (Part E), None until the room is registered
    still_path: str | None = None
    misses: int = 0
    pending_zone: str | None = None    # a new zone waiting for MOVE_CONFIRM_N looks
    pending_count: int = 0
    context: list[dict] = field(default_factory=list)  # [{object_id, bbox}] seen with it last time


@dataclass(frozen=True)
class Transition(_Record):
    """One line of events.jsonl. kind: appeared | reappeared | moved | disappeared | inferred_contained"""
    object_id: str
    name: str
    kind: str
    from_zone: str | None
    to_zone: str | None
    status: str
    container: str | None
    confidence: float
    ts: float
    time: str                          # clock_s(ts), for people reading the log
    still_path: str | None


@dataclass(frozen=True)
class Answer(_Record):
    """status: observed | inferred | unknown | none (no record) | info (lists, histories)."""
    text: str
    status: str
    confidence: float = 0.0
    still_path: str | None = None
    as_of: float | None = None         # set when the camera isn't live: the world froze then
