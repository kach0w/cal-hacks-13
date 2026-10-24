#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["opencv-python>=4.8", "pillow"]
# ///
"""PDFs to print for setup (DESIGN.md §1), at exact physical size on US Letter.

    uv run scripts/make_printables.py

Print both at 100% / "Actual size", never "Fit to page". Then measure the printed black
square with a ruler and use the measured size, not the nominal one.
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

DPI = 300
LETTER_CM = (21.59, 27.94)
OUT = Path(__file__).resolve().parent.parent / "printables"

# The setup tag. Must match the detector in the setup script.
TAG_DICT = cv2.aruco.DICT_4X4_50
TAG_ID = 0
TAG_CM = 18.0

# Lens calibration board. A different dictionary, so it can never be mistaken for the tag.
BOARD_DICT = cv2.aruco.DICT_5X5_100
BOARD_SQUARES = (7, 5)
BOARD_SQUARE_CM = 3.5
BOARD_MARKER_CM = 2.6


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


def save(canvas, name):
    path = OUT / name
    Image.fromarray(canvas).save(path, resolution=DPI)
    print(f"wrote {path}")


def setup_tag():
    canvas = page()
    marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(TAG_DICT), TAG_ID, px(TAG_CM))
    bottom = paste_centered(canvas, marker, y=px(2.0))
    label(canvas, [
        f"SETUP TAG - ArUco DICT_4X4_50, id {TAG_ID}",
        f"Black square should measure {TAG_CM:.1f} cm. Measure it and use the real number.",
        "Print at 100% / Actual size. Matte paper. Keep it flat (tape or mount on card).",
        "Middle of the room, on the floor, seen by both cameras. Tape its outline.",
    ], bottom + px(1.5))
    save(canvas, "setup_tag.pdf")


def calibration_board():
    canvas = page(landscape=True)
    board = cv2.aruco.CharucoBoard(
        BOARD_SQUARES, BOARD_SQUARE_CM / 100, BOARD_MARKER_CM / 100,
        cv2.aruco.getPredefinedDictionary(BOARD_DICT),
    )
    w, h = (px(n * BOARD_SQUARE_CM) for n in BOARD_SQUARES)
    bottom = paste_centered(canvas, board.generateImage((w, h)), y=px(1.2))
    label(canvas, [
        f"LENS CALIBRATION - ChArUco {BOARD_SQUARES[0]}x{BOARD_SQUARES[1]}, DICT_5X5_100, "
        f"square {BOARD_SQUARE_CM} cm, marker {BOARD_MARKER_CM} cm",
        "Print at 100%. Mount flat on card. Show it to each webcam ~20 times: near, far, tilted.",
    ], bottom + px(1.0))
    save(canvas, "calibration_board.pdf")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    setup_tag()
    calibration_board()
