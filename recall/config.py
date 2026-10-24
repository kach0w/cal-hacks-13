"""One config object. Defaults are starting points — calibrate on the rig (CAMERA.md §9).

Override any field with an environment variable RECALL_<NAME> or on the command line with
--set NAME=VALUE (names are case-insensitive), e.g.

    RECALL_CHANGE_FRACTION=0.01 python -m recall.live --src demo.mp4 --set MISSING_AFTER_K=4
"""
import argparse
import logging
import os
import warnings
from dataclasses import dataclass, fields, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    # --- files ---------------------------------------------------------------
    data_dir: Path = ROOT / "data"
    items_path: Path = ROOT / "items.json"    # the dictionary of things we track

    # --- capture -------------------------------------------------------------
    src: str = "0"                     # camera index, or a video file to replay
    capture_w: int = 1280
    capture_h: int = 720
    loop_s: float = 1.0                # one look per second
    stale_after_s: float = 5.0         # no new frame this long -> memory frozen, answers "as of"
    replay_speed: float = 0.0          # video replays: 0 = as fast as detection allows, 1 = real time

    # --- change gate (skip detection when nothing changed) -------------------
    change_width: int = 160            # diff on a tiny grayscale copy
    change_pixel_delta: int = 25       # per-pixel intensity change that counts
    change_fraction: float = 0.004     # fraction of pixels that must change (~0.4%)
    settle_s: float = 5.0              # keep detecting this long after the last change
    heartbeat_s: float = 20.0          # detect anyway every N seconds

    # --- perception ----------------------------------------------------------
    detector: str = "yolo-world"       # yolo-world | yoloe | yolo (fine-tuned; class names = item ids)
    detect_model: str = ""             # "" = the backend's default (perception.DEFAULT_MODEL); downloads once
    detect_imgsz: int = 640
    detect_device: str = ""            # "" = mps on Apple silicon, else ultralytics' choice; or cpu
    default_min_score: float = 0.15    # YOLO-World scores run low; calibration sets per item

    # --- memory (CAMERA.md §5) -----------------------------------------------
    missing_after_k: int = 3           # misses before "gone" (a passing hand blanks objects briefly)
    move_confirm_n: int = 2            # same new zone N looks in a row before committing a move
    contain_overlap: float = 0.25      # fraction of the object's last box inside the container's box
    zone_margin_px: float = 10.0       # base point may sit this far outside a zone box

    # --- re-id (task 6; only used when an identity check is plugged in) ------
    reid_accept: float = 0.80          # cosine to the item's references: commit
    reid_hold: float = 0.60            # between hold and accept: commit nothing either way

    @property
    def zones_path(self) -> Path:
        return self.data_dir / "zones.json"

    @property
    def enroll_dir(self) -> Path:
        return self.data_dir / "enroll"

    @property
    def room_dir(self) -> Path:
        return self.data_dir / "room"

    @property
    def replay_dir(self) -> Path:
        return self.data_dir / "replay"


_FIELDS = {f.name: f for f in fields(Settings)}


def _coerce(name, value):
    default = _FIELDS[name].default
    try:
        return type(default)(value)
    except ValueError:
        raise ValueError(f"{name.upper()}={value!r}: expected {type(default).__name__}") from None


def load_settings(overrides=(), env=None) -> Settings:
    """Defaults <- RECALL_* environment <- overrides ('NAME=VALUE' strings, e.g. from --set)."""
    env = os.environ if env is None else env
    values = {}
    for k, v in env.items():
        if k.startswith("RECALL_"):
            if k[7:].lower() in _FIELDS:
                values[k[7:].lower()] = v
            else:
                logging.getLogger(__name__).warning("ignoring %s: not a setting", k)
    for pair in overrides:
        name, sep, value = pair.partition("=")
        name = name.strip().lower()
        if not sep or name not in _FIELDS:
            known = ", ".join(n.upper() for n in _FIELDS)
            raise ValueError(f"bad setting {pair!r}; use NAME=VALUE with one of: {known}")
        values[name] = value.strip()
    return replace(Settings(), **{n: _coerce(n, v) for n, v in values.items()})


# --- CLI glue -----------------------------------------------------------------
def add_args(ap: argparse.ArgumentParser):
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                    help="override a setting from recall/config.py (repeatable)")
    ap.add_argument("-v", "--verbose", action="store_true", help="debug logging")


def from_args(a, **flags) -> Settings:
    """Settings for a CLI run: env <- --set <- explicit flags (None = not given)."""
    setup_logging(a.verbose)
    extra = [f"{k}={v}" for k, v in flags.items() if v is not None]
    try:
        return load_settings([*a.set, *extra])
    except ValueError as e:
        raise SystemExit(str(e))


def setup_logging(verbose=False):
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname).1s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("ultralytics", "PIL", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # ultralytics loads YOLOE's text encoder with torch.jit; the deprecation notice is not ours to fix
    warnings.filterwarnings("ignore", message=".*torch.jit.load.*is deprecated", category=FutureWarning)
