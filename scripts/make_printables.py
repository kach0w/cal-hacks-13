#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["opencv-python>=4.8", "pillow"]
# ///
"""PDFs to print for setup (DESIGN.md §1, handoffAB.md), at exact physical size on US Letter.

    uv run scripts/make_printables.py

Print at 100% / "Actual size", never "Fit to page". Then measure each printed black square
with a ruler and use the measured size, not the nominal one.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from printed import (  # noqa: E402  the single source of what's printed
    BOARD_DICT, BOARD_MARKER_CM, BOARD_SQUARE_CM, BOARD_SQUARES, TAG_CM, TAG_DICT, TAG_IDS,
)

DPI = 300
LETTER_CM = (21.59, 27.94)
OUT = ROOT / "printables"


def px(cm):
    return round(cm / 2.54 * DPI)


def page(landscape=False):
    w, h = LETTER_CM[::-1] if landscape else LETTER_CM
    return np.full((px(h), px(w)), 255, np.uint8)


def paste_centered(canvas, img, y=None):
    h, w = img.shape
    y = (canvas.shape[0] - h) // 2 if y is None else y
    x = (canvas.shape[1] - w) // 2
    canvas[y:y + h, x:x + w] = img
    return y + h


def label(canvas, lines, y):
    for line in lines:
        cv2.putText(canvas, line, (px(1.5), y), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 0, 3, cv2.LINE_AA)
        y += px(0.8)


def save(canvases, name):
    path = OUT / name
    first, *rest = (Image.fromarray(c) for c in canvases)
    first.save(path, resolution=DPI, save_all=True, append_images=rest)
    print(f"wrote {path} ({len(canvases)} page{'s' if len(canvases) > 1 else ''})")


def markers():
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, TAG_DICT))
    pages = []
    for tag_id in TAG_IDS:
        canvas = page()
        bottom = paste_centered(canvas, cv2.aruco.generateImageMarker(dictionary, tag_id, px(TAG_CM)), y=px(2.0))
        label(canvas, [
            f"ROOM MARKER {tag_id} - ArUco {TAG_DICT}, id {tag_id}",
            f"Black square should measure {TAG_CM:.1f} cm. Measure it and use the real number.",
            "Print at 100% / Actual size. Matte paper. Keep it flat (tape or mount on card).",
            "Floor AND wall, near/middle/far from the camera. Stays up through the demo.",
        ], bottom + px(1.5))
        pages.append(canvas)
    save(pages, "markers.pdf")


def calibration_board():
    canvas = page(landscape=True)
    board = cv2.aruco.CharucoBoard(
        BOARD_SQUARES, BOARD_SQUARE_CM / 100, BOARD_MARKER_CM / 100,
        cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, BOARD_DICT)),
    )
    w, h = (px(n * BOARD_SQUARE_CM) for n in BOARD_SQUARES)
    bottom = paste_centered(canvas, board.generateImage((w, h)), y=px(1.2))
    label(canvas, [
        f"LENS CALIBRATION - ChArUco {BOARD_SQUARES[0]}x{BOARD_SQUARES[1]}, {BOARD_DICT}, "
        f"square {BOARD_SQUARE_CM} cm, marker {BOARD_MARKER_CM} cm",
        "Print at 100%. Mount flat on card. Show it to each webcam ~20 times: near, far, tilted.",
    ], bottom + px(1.0))
    save([canvas], "calibration_board.pdf")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    markers()
    calibration_board()
