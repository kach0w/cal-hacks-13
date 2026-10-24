"""Detector interface (no model download: a fake ultralytics model), re-id hook, config."""
import json
from dataclasses import replace

import numpy as np
import pytest

from fakes import FRAME, ITEMS, KEYS_TABLE, world
from recall.config import ROOT, Settings, load_settings
from recall.enroll import load_items
from recall.perception import (PERSON, UltralyticsDetector, class_labels, make_detector,
                               reference_images)


class _T:  # what ultralytics tensors look like after .cpu().numpy()
    def __init__(self, a):
        self.a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class FakeModel:
    def __init__(self, rows):           # rows: (box, cls, score)
        self.rows, self.calls = rows, []

    def predict(self, image, **kw):
        self.calls.append(kw)
        rows = [r for r in self.rows if r[2] >= kw["conf"]]
        boxes = type("Boxes", (), {"xyxy": _T([r[0] for r in rows] or np.zeros((0, 4))),
                                   "cls": _T([r[1] for r in rows]), "conf": _T([r[2] for r in rows])})
        return [type("Result", (), {"boxes": boxes})]


LABELS = list(ITEMS) + [PERSON]   # yolo-world: class i = item i, then person
P = LABELS.index(PERSON)


def detector(rows, items=ITEMS, **cfg):
    return UltralyticsDetector(items, replace(Settings(), **cfg), FakeModel(rows), LABELS)


def test_best_box_per_item_and_persons():
    d = detector([([0, 0, 10, 10], 0, 0.30), ([5, 5, 20, 20], 0, 0.60),     # two keys boxes
                  ([1, 1, 2, 2], 2, 0.05),                                    # glasses below threshold
                  ([0, 0, 500, 700], P, 0.80),                                # a person
                  ([9, 9, 9, 9], 42, 0.99)])                                  # class we don't know
    dets = d.detect(FRAME, 7.0)
    assert [(s.object_id, s.bbox, s.score, s.ts) for s in dets.sightings] == [("keys", [5, 5, 20, 20], 0.6, 7.0)]
    assert dets.persons == [[0, 0, 500, 700]] and dets.held == set()


def test_per_item_thresholds_and_raw_mode():
    items = {k: replace(v, min_score=0.5 if k == "keys" else None) for k, v in ITEMS.items()}
    rows = [([0, 0, 10, 10], 0, 0.40), ([0, 0, 10, 10], 2, 0.20)]
    d = detector(rows, items, default_min_score=0.15)
    assert [s.object_id for s in d.detect(FRAME, 0).sightings] == ["glasses"]   # keys 0.40 < 0.5
    assert d.model.calls[-1]["conf"] == 0.15                                     # lowest threshold
    assert {s.object_id for s in d.detect(FRAME, 0, raw=True).sightings} == {"keys", "glasses"}
    assert d.model.calls[-1]["conf"] == 0.01


def test_model_settings_are_passed_through():
    d = detector([], detect_imgsz=416, detect_device="mps")
    d.detect(FRAME, 0)
    assert d.model.calls[-1]["imgsz"] == 416 and d.model.calls[-1]["device"] == "mps"
    d = detector([])
    d.detect(FRAME, 0)
    assert d.model.calls[-1]["device"] in (None, "mps")             # "" -> auto


def test_fine_tuned_class_names_map_to_items(caplog):
    names = {0: "keys", 1: "eyeglasses", 2: "Blue Bag", 3: "person", 4: "cup"}
    assert class_labels(names, ITEMS) == ["keys", "glasses", "bag", PERSON, None]
    assert "wallet" in caplog.text and "phone" in caplog.text      # warns about items it can't see


def test_unknown_backend():
    with pytest.raises(SystemExit):
        make_detector(ITEMS, replace(Settings(), detector="resnet"))


class Sims:
    def __init__(self, sims):
        self.sims, self.crops = sims, []

    def similarity(self, oid, crop):
        self.crops.append(crop.shape)
        return self.sims[oid]


def test_reid_accepts_holds_and_rejects():
    rows = [([0, 0, 10, 10], 0, 0.5), ([20, 20, 40, 30], 1, 0.5), ([50, 50, 60, 60], 2, 0.5)]
    ident = Sims({"keys": 0.9, "wallet": 0.7, "glasses": 0.3})
    d = UltralyticsDetector(ITEMS, Settings(), FakeModel(rows), LABELS, ident)
    dets = d.detect(FRAME, 0)
    assert [s.object_id for s in dets.sightings] == ["keys"]
    assert dets.held == {"wallet"}
    assert ident.crops[1] == (10, 20, 3)                         # the box crop, rows x cols
    assert {s.object_id for s in d.detect(FRAME, 0, raw=True).sightings} == {"keys", "wallet", "glasses"}


def test_held_items_flow_into_memory(tmp_path):
    from fakes import S
    _, _, mem, _ = world(tmp_path)
    mem.update([S("wallet", KEYS_TABLE)], FRAME, 0)
    for i in range(4):
        mem.update([], FRAME, i + 1, held={"wallet"})
    assert mem.state["wallet"].status == "observed"


def test_reference_images_prefer_camera_crops(tmp_path):
    (tmp_path / "enroll" / "keys").mkdir(parents=True)
    (tmp_path / "reference").mkdir()
    for p in ("enroll/keys/1.jpg", "enroll/keys/0.jpg", "reference/keys.png"):
        (tmp_path / p).write_bytes(b"x")
    item = replace(ITEMS["keys"], reference="reference/keys.png")
    assert [p.relative_to(tmp_path).as_posix() for p in reference_images(item, tmp_path)] == \
        ["enroll/keys/0.jpg", "enroll/keys/1.jpg", "reference/keys.png"]


# --- config -------------------------------------------------------------------------------------
def test_settings_precedence_and_types():
    s = load_settings(["change_fraction=0.01", "SRC=demo.mp4"],
                      env={"RECALL_CHANGE_FRACTION": "0.5", "RECALL_MISSING_AFTER_K": "4", "HOME": "/x"})
    assert (s.change_fraction, s.missing_after_k, s.src) == (0.01, 4, "demo.mp4")
    assert load_settings(["data_dir=/tmp/x"], env={}).zones_path.as_posix() == "/tmp/x/zones.json"
    assert load_settings(env={}) == Settings()


@pytest.mark.parametrize("bad", ["NOPE=1", "MISSING_AFTER_K=three", "LOOP_S"])
def test_bad_settings_fail_loudly(bad):
    with pytest.raises(ValueError):
        load_settings([bad], env={})


def test_items_json_round_trips_exactly():
    raw = json.loads((ROOT / "items.json").read_text())
    items = load_items(ROOT / "items.json")
    assert {k: v.to_dict() for k, v in items.items()} == raw
    assert all(list(items[k].to_dict()) == list(raw[k]) for k in raw)
    assert sum(it.container for it in items.values()) == 1 and items["bag"].container


def test_env_typos_warn(caplog):
    s = load_settings(env={"RECALL_CHANGE_FRACTON": "0.5"})
    assert s == Settings() and "RECALL_CHANGE_FRACTON" in caplog.text


def test_enroll_add_updates_only_what_was_given(tmp_path, monkeypatch):
    import sys
    from recall import enroll
    path = tmp_path / "items.json"
    path.write_text((ROOT / "items.json").read_text())
    run = lambda *args: (monkeypatch.setattr(sys, "argv", ["enroll", *args, "--set", f"ITEMS_PATH={path}"]),
                         enroll.main())
    before = load_items(path)["bag"]
    run("add", "bag", "--prompt", "brown paper bag")
    after = load_items(path)["bag"]
    assert after.prompt == "brown paper bag"
    assert (after.aliases, after.container, after.name) == (before.aliases, True, "bag")
    run("add", "phone", "--alias", "cell")
    assert load_items(path)["phone"].to_dict() == {"name": "phone", "prompt": "phone", "aliases": ["cell"],
                                                   "plural": False, "container": False,
                                                   "min_score": None, "reference": None}
    with pytest.raises(SystemExit):
        run("add", "--set")                                 # not an id
    with pytest.raises(SystemExit):
        run("remove", "nope")
