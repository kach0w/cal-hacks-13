"""The original slow-lane scenarios, ported: fake sightings, no camera, no model.

    python -m pytest tests/
"""
import numpy as np

from fakes import (AIR, BAG_COUNTER, BAG_TABLE, FRAME, GLASSES_COUNTER, GLASSES_OVER_BAG, ITEMS,
                   KEYS_COUNTER, KEYS_TABLE, ZONES, S, kinds, world)
from recall import ask
from recall.config import Settings
from recall.gate import ChangeGate
from recall.memory import Memory
from recall.models import Zone
from recall.query import parse_since
from recall.zones import zone_at

T = 1_000_000.0
STEPS = [
    [S("keys", KEYS_COUNTER), S("bag", BAG_TABLE), S("glasses", GLASSES_COUNTER)],  # t0 appear
    [S("keys", KEYS_TABLE), S("bag", BAG_TABLE), S("glasses", GLASSES_COUNTER)],    # t1 pending
    [S("keys", KEYS_TABLE), S("bag", BAG_TABLE), S("glasses", GLASSES_COUNTER)],    # t2 moved
    [S("keys", KEYS_TABLE), S("bag", BAG_TABLE), S("glasses", GLASSES_OVER_BAG)],   # t3 over bag
    [S("keys", KEYS_TABLE), S("bag", BAG_TABLE)],                                   # t4 miss 1
    [S("keys", KEYS_TABLE), S("bag", BAG_TABLE)],                                   # t5 miss 2
    [S("keys", KEYS_TABLE), S("bag", BAG_TABLE)],                                   # t6 miss 3 -> inferred
    [S("keys", KEYS_TABLE), S("bag", BAG_COUNTER)],                                 # t7 bag pending
    [S("keys", KEYS_TABLE), S("bag", BAG_COUNTER)],                                 # t8 bag moved, glasses follow
    [S("bag", BAG_COUNTER)], [S("bag", BAG_COUNTER)], [S("bag", BAG_COUNTER)],      # t9-11 keys gone
]


def run_scenario(tmp):
    cfg, store, mem, q = world(tmp)
    log = [mem.update(sightings, FRAME, T + i) for i, sightings in enumerate(STEPS)]
    return cfg, store, mem, q, log


def test_scenario(tmp_path):
    cfg, store, mem, q, log = run_scenario(tmp_path)
    k = [kinds(step) for step in log]
    assert sorted(k[0]) == ["appeared:bag", "appeared:glasses", "appeared:keys"], k[0]
    assert k[1] == [], k[1]                                    # one look isn't a move
    assert k[2] == ["moved:keys"], k[2]
    assert k[4] == [] and k[5] == [], k[4:6]                   # blinking, not gone
    assert k[6] == ["inferred_contained:glasses"], k[6]
    assert sorted(k[8]) == ["moved:bag", "moved:glasses"], k[8]
    assert k[11] == ["disappeared:keys"], k[11]

    st = mem.state
    assert st["keys"].status == "unknown" and st["keys"].zone == "table"
    assert st["glasses"].status == "inferred" and st["glasses"].container == "bag"
    assert st["glasses"].zone == "kitchen counter"             # followed the bag
    assert st["wallet"].last_observed is None
    assert (tmp_path / "stills").exists() and any((tmp_path / "stills").iterdir())

    now = T + 12
    a = q.where_is("glasses", now=now)
    assert "probably in the blue bag, on the kitchen counter" in a.text, a
    assert a.status == "inferred"
    assert "haven't seen them since" in q.where_is("keys", now=now).text
    assert q.where_is("wallet", now=now).text == "I don't have a record of that."
    assert "on the kitchen counter" in q.where_is("bag", now=now).text

    assert len(q.changes_since(T + 6)) == 4                    # inferred, bag, glasses, keys
    df = {x["object_id"]: x for x in q.diff(T + 0.5)}
    assert set(df) == {"keys", "bag", "glasses"}
    assert df["keys"]["before"]["zone"] == "kitchen counter" and df["keys"]["now"]["status"] == "unknown"

    # REPL routing
    assert "probably in the blue bag" in ask.handle("where are my glasses?", q, now).text
    assert "moved from" in ask.handle("what changed today", q, now).text
    assert "keys" in ask.handle("diff 30m", q, now).text
    assert "went into the blue bag" in ask.handle("history glasses", q, now).text
    assert ask.handle("where is my umbrella", q, now).text == "I don't have a record of that."

    # restart: state survives, no duplicate commits on the first look after reload
    mem2 = Memory(ITEMS, ZONES, store, cfg)
    assert mem2.update([S("bag", BAG_COUNTER)], FRAME, T + 20) == []


def test_reappear_clears_container(tmp_path):
    _, _, mem, _, _ = run_scenario(tmp_path)
    ev = mem.update([S("bag", BAG_COUNTER), S("glasses", GLASSES_COUNTER)], FRAME, T + 30)
    assert [e.kind for e in ev] == ["reappeared"]
    assert mem.state["glasses"].status == "observed" and mem.state["glasses"].container is None


def test_mid_air_needs_extra_confirm(tmp_path):
    _, _, mem, _ = world(tmp_path)
    mem.update([S("phone", [100, 400, 160, 440])], FRAME, 0)
    assert mem.update([S("phone", AIR)], FRAME, 1) == []
    assert mem.update([S("phone", AIR)], FRAME, 2) == []        # 2 isn't enough for "nowhere"
    assert [e.kind for e in mem.update([S("phone", AIR)], FRAME, 3)] == ["moved"]


def test_zone_at_smallest_wins():
    zones = ZONES + [Zone("tray", [50, 380, 200, 460])]
    assert zone_at(zones, KEYS_COUNTER) == "tray"
    assert zone_at(zones, KEYS_TABLE) == "table"


def test_change_gate():
    g = ChangeGate(Settings(settle_s=5, heartbeat_s=20))
    a = np.zeros((720, 1280, 3), np.uint8)
    b = a.copy()
    b[300:400, 500:700] = 255
    assert g.check(a, 0)[0]
    g.mark_detected(a, 0)
    assert g.check(a, 1) == (False, "static")
    assert g.check(b, 2) == (True, "change")
    g.mark_detected(b, 2)
    assert g.check(b, 3) == (True, "settling")
    g.mark_detected(b, 3)
    assert g.check(b, 8) == (False, "static")
    assert g.check(b, 24) == (True, "heartbeat")


def test_parse_since():
    now = 1_000_000.0
    assert parse_since("10m", now) == now - 600
    assert parse_since("2h", now) == now - 7200
    assert parse_since("today", now) <= now
