"""Answer templates (CAMERA.md §6), the frozen-world stamp, routing and time parsing."""
from datetime import datetime

import pytest

from fakes import ITEMS, ZONES, _it, world
from recall import ask
from recall.models import ObjectState, Transition, clock_s
from recall.query import clock, parse_since

NOW = 1_700_000_000.0
DEMO_ITEMS = {**ITEMS, "bag": _it("bag", "bag", "brown paper bag", container=True)}


def put(store, *objs, beat=NOW - 1, camera_ok=True):
    store.save({o.object: o for o in objs}, beat)
    store.beat(beat, camera_ok)


def obs(oid, name, zone, seen, **kw):
    return ObjectState(oid, name, surface=zone, status="observed", confidence=0.6,
                       since=seen, last_observed=seen, **kw)


# --- the six templates, verbatim -------------------------------------------------------------
def test_templates(tmp_path):
    _, store, _, q = world(tmp_path, items=DEMO_ITEMS)
    went_in = NOW - 240
    put(store,
        obs("keys", "keys", "kitchen counter", NOW - 3),
        obs("wallet", "wallet", None, NOW - 120),
        ObjectState("glasses", "glasses", surface="table", status="inferred", confidence=0.9,
                    since=went_in, last_observed=went_in, container="bag"),
        obs("bag", "bag", "table", NOW - 1),
        ObjectState("phone", "phone", surface="table", status="unknown", confidence=0.4,
                    since=NOW - 900, last_observed=NOW - 900))
    t = lambda oid: q.where_is(oid, now=NOW).text
    assert t("keys") == "Your keys are on the kitchen counter — seen just now."
    assert t("wallet") == "Your wallet is in view, but not in a named area — seen 2 minutes ago."
    assert t("glasses") == ("Your glasses are probably in the bag, on the table — inferred, "
                            "last seen going in 4 minutes ago.")
    assert t("phone") == f"Last seen on the table at {clock(NOW - 900)}. I haven't seen it since."
    assert t("remote") == "I don't have a record of that."

    a = q.where_is("glasses", now=NOW)
    assert (a.status, a.confidence, a.as_of) == ("inferred", 0.9, None)

    # the bag is lost too
    bag = obs("bag", "bag", "table", NOW - 60)
    bag.status = "unknown"
    put(store, store.load_state()["glasses"], bag)
    assert t("glasses") == (f"Your glasses are probably in the bag, but I've lost track of the bag "
                            f"too. They went in at {clock(went_in)}.")


def test_unknown_outside_named_areas(tmp_path):
    _, store, _, q = world(tmp_path)
    put(store, ObjectState("keys", "keys", status="unknown", since=NOW - 60, last_observed=NOW - 60))
    assert q.where_is("keys", now=NOW).text == \
        f"Last seen outside the named areas at {clock(NOW - 60)}. I haven't seen them since."


def test_zone_preposition(tmp_path):
    from recall.models import Zone
    _, store, _, q = world(tmp_path, zones=ZONES + [Zone("sink", [0, 0, 100, 100], "in")])
    put(store, obs("phone", "phone", "sink", NOW - 30))
    assert q.where_is("phone", now=NOW).text == "Your phone is in the sink — seen 30 seconds ago."


# --- never answer from a frozen world as if it were live ------------------------------------
def test_live_answers_are_not_stamped(tmp_path):
    _, store, _, q = world(tmp_path)
    put(store, obs("keys", "keys", "table", NOW - 2), beat=NOW - 4.9)
    assert q.as_of(NOW) is None
    assert q.where_is("keys", now=NOW).text.startswith("Your keys")


@pytest.mark.parametrize("beat_age, camera_ok", [(60, True), (2, False)])
def test_frozen_world_stamps_every_answer(tmp_path, beat_age, camera_ok):
    """Heartbeat too old (live.py died) or camera_ok=false (camera stalled): stamp "As of"."""
    _, store, _, q = world(tmp_path)
    put(store, obs("keys", "keys", "table", NOW - 70), beat=NOW - beat_age, camera_ok=camera_ok)
    frozen = NOW - beat_age
    a = q.where_is("keys", now=NOW)
    assert a.as_of == frozen
    assert a.text == f"As of {clock(frozen)}: Your keys are on the table — seen 1 minute ago."
    for text in ("where are my keys", "where is my umbrella", "what changed today", "diff 10m",
                 "history keys", "state"):
        r = ask.handle(text, q, NOW)
        assert r.as_of == frozen, (text, r)
        assert r.text.startswith(f"As of {clock(frozen)}:" + ("\n" if "\n" in r.text else " ")), (text, r)


def test_no_heartbeat_means_as_of_the_newest_sighting(tmp_path):
    _, store, _, q = world(tmp_path)
    store.save({"keys": obs("keys", "keys", "table", NOW - 70)}, NOW - 70)
    (tmp_path / "heartbeat.json").unlink()
    assert q.as_of(NOW) == NOW - 70
    assert q.where_is("keys", now=NOW).text.startswith(f"As of {clock(NOW - 70)}: ")


def test_stall_before_any_frame(tmp_path):
    _, store, _, q = world(tmp_path)
    store.beat(None, camera_ok=False)
    assert q.as_of(NOW) is None                              # nothing known: nothing to be stale about
    assert q.where_is("keys", now=NOW).text == "I don't have a record of that."


# --- history, describe, routing ------------------------------------------------------------
def ev(kind, oid="glasses", frm="table", to="table", status="inferred", container="bag", ts=NOW - 10):
    return Transition(oid, oid, kind, frm, to, None, None, status, container, 0.9, ts, clock_s(ts), None)


def test_describe_uses_names(tmp_path):
    _, _, _, q = world(tmp_path)
    assert q.describe(ev("inferred_contained")) == "glasses went into the blue bag — inferred (0.90)"
    assert q.describe(ev("moved", to="kitchen counter")) == \
        "glasses moved from on the table to on the kitchen counter (inside the blue bag, inferred)"
    assert q.describe(ev("disappeared", to=None, container=None)) == "glasses disappeared from on the table"
    assert q.describe(ev("appeared", frm=None, container=None)) == "glasses appeared on the table"


def test_match_item():
    items = {**ITEMS, "car_keys": _it("car_keys", "car keys", "car keys")}
    assert ask.match_item(items, "Where are my CAR KEYS?") == "car_keys"   # longest wins
    assert ask.match_item(items, "where's my key") == "keys"               # alias
    assert ask.match_item(items, "where is the bag") == "bag"
    assert ask.match_item(items, "where's my keyboard") is None            # whole words only


def test_answer_routing(tmp_path):
    _, store, _, q = world(tmp_path)
    put(store, obs("keys", "keys", "table", NOW - 3))
    for e in (ev("appeared", "keys", None, "table", "observed", None, NOW - 30),):
        store.append(e)
    assert ask.handle("", q, NOW).text == ""
    assert ask.handle("history", q, NOW).text.startswith("History of what?")
    assert "Nothing has changed since" in ask.handle("what changed since 10s", q, NOW).text
    assert "keys appeared on the table" in ask.handle("what changed since 1m", q, NOW).text
    assert ask.handle("where are my keys?", q, NOW).status == "observed"
    with pytest.raises(ValueError):
        ask.handle("what changed since banana", q, NOW)


def test_state_at_and_diff(tmp_path):
    _, store, _, q = world(tmp_path)
    store.append(ev("appeared", "keys", None, "table", "observed", None, NOW - 100))
    store.append(ev("moved", "keys", "table", "kitchen counter", "observed", None, NOW - 50))
    assert q.state_at(NOW - 75) == {"keys": {"surface": "table", "status": "observed", "container": None}}
    assert q.diff(NOW - 75) == [{"object": "keys",
                                 "before": {"surface": "table", "status": "observed", "container": None},
                                 "now": {"surface": "kitchen counter", "status": "observed", "container": None}}]
    assert q.diff(NOW - 10) == []


# --- time parsing -----------------------------------------------------------------------------
def test_parse_since_forms():
    now = datetime(2026, 10, 24, 15, 0).timestamp()
    assert parse_since("30s", now) == now - 30
    assert parse_since("10 minutes", now) == now - 600
    assert parse_since("10 minutes ago", now) == now - 600
    assert parse_since("2 hr", now) == now - 7200
    assert parse_since("today", now) == datetime(2026, 10, 24).timestamp()
    assert parse_since("14:30", now) == datetime(2026, 10, 24, 14, 30).timestamp()
    assert parse_since("2:40pm", now) == datetime(2026, 10, 24, 14, 40).timestamp()
    assert parse_since("4:00pm", now) == datetime(2026, 10, 23, 16, 0).timestamp()
    assert parse_since("12:15am", now) == datetime(2026, 10, 24, 0, 15).timestamp()
    with pytest.raises(ValueError):
        parse_since("yesterday-ish", now)


# --- review follow-ups ------------------------------------------------------------------------------
def test_template_edges(tmp_path):
    from recall.query import ago
    _, store, _, q = world(tmp_path, items=DEMO_ITEMS)
    went_in = NOW - 5
    bag = obs("bag", "bag", None, NOW - 1)                 # visible, but in no named area
    put(store, ObjectState("glasses", "glasses", status="inferred", confidence=0.9, since=went_in,
                           last_observed=went_in, container="bag"), bag,
        ObjectState("phone", "phone", status="inferred", confidence=0.6, since=NOW - 60,
                    last_observed=NOW - 60, container="bag"),
        obs("keys", "keys", "table", NOW - 3, ))
    assert q.where_is("glasses", now=NOW).text == \
        "Your glasses are probably in the bag — inferred, last seen going in just now."
    # "seen ..." counts from the last sighting, not from when it arrived in the zone
    keys = store.load_state()["keys"]
    keys.since = NOW - 7200
    bag.status = "unknown"
    put(store, keys, bag, store.load_state()["phone"])
    assert q.where_is("keys", now=NOW).text == "Your keys are on the table — seen just now."
    assert q.where_is("phone", now=NOW).text == (f"Your phone is probably in the bag, but I've lost "
                                                f"track of the bag too. It went in at {clock(NOW - 60)}.")
    assert (ago(0, 9), ago(0, 10), ago(0, 59), ago(0, 60), ago(0, 119), ago(0, 3600)) == \
        ("just now", "10 seconds ago", "59 seconds ago", "1 minute ago", "1 minute ago", "1 hour ago")


def test_heartbeat_from_the_future_is_not_live(tmp_path):
    _, store, _, q = world(tmp_path)
    put(store, obs("keys", "keys", "table", NOW), beat=NOW + 0.5)
    assert q.as_of(NOW) is None                            # clock jitter
    store.beat(NOW + 30)
    assert q.as_of(NOW) == NOW + 30                         # a clock that's wrong is not "live"


def test_state_names_never_seen_items(tmp_path):
    _, store, _, q = world(tmp_path)
    put(store, obs("keys", "keys", "table", NOW - 3))
    lines = ask.handle("state", q, NOW).text.splitlines()
    assert "- Your keys are on the table — seen just now." in lines and "- wallet: never seen" in lines
