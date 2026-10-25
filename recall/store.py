"""The Store (DESIGN §6): the memory's files. The live process writes; anything can read.

    state.json       current WorldState (atomic rewrite)
    events.jsonl     append-only Transitions — history, diffs and "what changed" replay this
    photos/          the annotated picture behind each transition (photo_path)
    heartbeat.json   {last_frame_ts, camera_ok}: how fresh the memory is
"""
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path

from .models import ObjectState, Transition

log = logging.getLogger(__name__)


def read_json(path, default=None):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else default


def write_json_atomic(path, obj):
    """Write to a temp file, then rename: a crash mid-write never corrupts the file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, p)


class Store:
    STATE, EVENTS, PHOTOS, HEARTBEAT = "state.json", "events.jsonl", "photos", "heartbeat.json"
    OLD_PHOTOS = "stills"  # where photos went before the DESIGN §7 rename (models.RENAMED)

    def __init__(self, root):
        self.root = Path(root)

    # --- write side (the live process) ----------------------------------------
    def save(self, state: dict[str, ObjectState], frame_ts: float):
        write_json_atomic(self.root / self.STATE, {k: o.to_dict() for k, o in state.items()})
        self.beat(frame_ts)

    def beat(self, frame_ts: float | None, camera_ok=True):
        write_json_atomic(self.root / self.HEARTBEAT,
                          {"last_frame_ts": frame_ts, "camera_ok": camera_ok})

    def append(self, t: Transition):
        self.root.mkdir(parents=True, exist_ok=True)
        p = self.root / self.EVENTS
        torn = False
        if p.exists() and p.stat().st_size:
            with open(p, "rb") as f:
                f.seek(-1, os.SEEK_END)
                torn = f.read(1) != b"\n"  # a crash mid-append left half a line
        with open(p, "a") as f:
            f.write(("\n" if torn else "") + json.dumps(t.to_dict()) + "\n")

    def save_photo(self, image, name: str) -> str:
        import cv2
        rel = f"{self.PHOTOS}/{name}.jpg"
        (self.root / self.PHOTOS).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.root / rel), image)
        return rel

    def clear(self):
        """Forget everything (replays start from an empty memory)."""
        for name in (self.STATE, self.EVENTS, self.HEARTBEAT):
            (self.root / name).unlink(missing_ok=True)
        for name in (self.PHOTOS, self.OLD_PHOTOS):
            shutil.rmtree(self.root / name, ignore_errors=True)

    # --- read side (anyone) ---------------------------------------------------
    def load_state(self) -> dict[str, ObjectState]:
        raw = read_json(self.root / self.STATE, {})
        return {k: ObjectState.from_dict(v) for k, v in raw.items()}

    def events(self) -> list[Transition]:
        p = self.root / self.EVENTS
        if not p.exists():
            return []
        out = []
        for line in p.read_text().splitlines():
            try:
                out.append(Transition.from_dict(json.loads(line)))
            except (ValueError, TypeError):
                if line.strip():  # a torn last line from a crash mid-append
                    log.warning("skipping bad line in %s: %.60s", p, line)
        return out

    def heartbeat(self) -> dict | None:
        return read_json(self.root / self.HEARTBEAT, None)

    def photo_file(self, rel: str) -> Path | None:
        """Absolute path of a photo from a state/event photo_path; None if missing or outside photos/
        (or the stills/ of older memory files)."""
        p = (self.root / rel).resolve()
        dirs = [(self.root / d).resolve() for d in (self.PHOTOS, self.OLD_PHOTOS)]
        return p if any(p.is_relative_to(d) for d in dirs) and p.is_file() else None
