"""What gets printed, in one place. scripts/make_printables.py draws these; the tools detect them.

Change a value here and reprint, or detection silently fails. No imports, so anything can use it.
"""

# Room markers (handoffAB.md: >= 3, floor and wall, stay up through the demo).
TAG_DICT = "DICT_4X4_50"
TAG_IDS = range(6)
TAG_CM = 18.0  # nominal; markers.json uses the size measured off the print

# Lens calibration board. A different dictionary, so it can never be mistaken for a marker.
BOARD_DICT = "DICT_5X5_100"
BOARD_SQUARES = (7, 5)
BOARD_SQUARE_CM = 3.5
BOARD_MARKER_CM = 2.6
