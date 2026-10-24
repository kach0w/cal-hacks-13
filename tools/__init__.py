"""Setup tools: camera probe, range test, recording, simulation, lens calibration, registration."""
from pathlib import Path

import cv2

from recall.store import read_json, write_json_atomic as write_json  # noqa: F401  (shared by the tools)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def read_images(folder, limit=None):
    """Images in a folder, sorted by name."""
    paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS)[:limit]
    if not paths:
        raise SystemExit(f"no images in {folder}")
    return [cv2.imread(str(p)) for p in paths]
