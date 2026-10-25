"""tools/range_test.py: the recommendation and confusion checks (no camera, no model)."""
from recall.models import Sighting
from tools.range_test import overlaps, recommend


def row(det, size, ms, misses=(), confused=()):
    return {"detector": det, "imgsz": size, "ms": ms, "misses": list(misses), "confused": list(confused)}


def test_recommend_fewest_misses_within_budget():
    rows = [row("yolo-world", 640, [20, 22], ["keys@far", "glasses@far"]),
            row("yoloe", 640, [55, 60], ["wallet@far"]),
            row("yoloe", 1280, [200, 210]),                        # perfect but over budget
            row("yolo-world", 1280, [50, 51], ["keys@far"], ["keys/glasses@mid"])]
    assert recommend(rows) is rows[1]


def test_recommend_prefers_smaller_imgsz_on_a_tie():
    rows = [row("yoloe", 1280, [120]), row("yoloe", 640, [60])]
    assert recommend(rows) is rows[1]
    assert recommend([row("yoloe", 1280, [400])]) is None


def test_overlapping_labels_are_flagged():
    s = lambda oid, box: Sighting(oid, box, 0.5, 0)
    last = {"keys": s("keys", [100, 100, 200, 160]), "glasses": s("glasses", [105, 102, 195, 158]),
            "remote": s("remote", [400, 100, 500, 140])}
    assert overlaps(last) == [("glasses", "keys")]
