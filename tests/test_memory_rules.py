"""The CAMERA.md §5 rules table, one rule at a time."""
import json

from fakes import (BAG_COUNTER, BAG_TABLE, FRAME, GLASSES_COUNTER, GLASSES_OVER_BAG, ITEMS,
                   KEYS_COUNTER, KEYS_TABLE, ZONES, S, kinds, world)
from recall.memory import Memory
from recall.places import Places


def feed(mem, steps, t0=0):
    return [mem.update(s, FRAME, t0 + i) for i, s in enumerate(steps)]


def contain(tmp, glasses_box, **overrides):
    """Glasses seen once at glasses_box with the bag, then gone K looks. -> (memory, gone events)"""
    _, _, mem, _ = world(tmp, **overrides)
    k = mem.cfg.missing_after_k
    log = feed(mem, [[S("bag", BAG_TABLE), S("glasses", glasses_box)]] + [[S("bag", BAG_TABLE)]] * k)
    return mem, log[-1]


# --- first sighting / moves -------------------------------------------------------
def test_first_sighting_commits_immediately(tmp_path):
    _, _, mem, _ = world(tmp_path)
    [ev] = mem.update([S("keys", KEYS_COUNTER, 0.41)], FRAME, 5)
    assert (ev.kind, ev.from_surface, ev.to_surface, ev.status, ev.confidence) == \
        ("appeared", None, "kitchen counter", "observed", 0.41)
    assert mem.state["keys"].since == 5 and mem.state["keys"].photo_path == ev.photo_path


def test_move_back_cancels_pending(tmp_path):
    _, _, mem, _ = world(tmp_path)
    log = feed(mem, [[S("keys", KEYS_COUNTER)], [S("keys", KEYS_TABLE)], [S("keys", KEYS_COUNTER)],
                     [S("keys", KEYS_TABLE)], [S("keys", KEYS_TABLE)]])
    assert [kinds(x) for x in log[1:]] == [[], [], [], ["moved:keys"]]


def test_pending_restarts_on_a_third_zone(tmp_path):
    zones = ZONES + [type(ZONES[0])("shelf", [0, 0, 600, 250])]
    _, _, mem, _ = world(tmp_path, zones=zones)
    shelf = [100, 150, 160, 200]
    log = feed(mem, [[S("keys", KEYS_COUNTER)], [S("keys", KEYS_TABLE)], [S("keys", shelf)],
                     [S("keys", shelf)]])
    assert [kinds(x) for x in log[1:]] == [[], [], ["moved:keys"]]
    assert log[3][0].from_surface == "kitchen counter" and log[3][0].to_surface == "shelf"


def test_move_confirm_n_is_configurable(tmp_path):
    _, _, mem, _ = world(tmp_path, move_confirm_n=3)
    log = feed(mem, [[S("keys", KEYS_COUNTER)]] + [[S("keys", KEYS_TABLE)]] * 3)
    assert [kinds(x) for x in log[1:]] == [[], [], ["moved:keys"]]


# --- missing / blinking -------------------------------------------------------------
def test_hand_passing_over_is_not_a_disappearance(tmp_path):
    """K-1 blank looks (a hand over the counter), then seen again: nothing is committed."""
    _, _, mem, _ = world(tmp_path)
    log = feed(mem, [[S("keys", KEYS_COUNTER)], [], [], [S("keys", KEYS_COUNTER)], [], []])
    assert [kinds(x) for x in log[1:]] == [[], [], [], [], []]
    assert mem.state["keys"].status == "observed" and mem.state["keys"].misses == 2


def test_misses_must_be_consecutive_then_gone(tmp_path):
    _, _, mem, _ = world(tmp_path, missing_after_k=4)
    log = feed(mem, [[S("keys", KEYS_TABLE)], [], [], [], []])
    assert [kinds(x) for x in log[1:]] == [[], [], [], ["disappeared:keys"]]
    ev = log[-1][0]
    assert (ev.from_surface, ev.to_surface, ev.status) == ("table", None, "unknown")
    o = mem.state["keys"]
    assert o.surface == "table" and o.since == 0 and o.container is None   # zone kept as last known


def test_gone_objects_stay_quiet(tmp_path):
    _, _, mem, _ = world(tmp_path)
    log = feed(mem, [[S("keys", KEYS_TABLE)]] + [[]] * 10)
    assert sum(len(x) for x in log) == 2                    # appeared + one disappeared
    assert kinds(log[3]) == ["disappeared:keys"]


def test_never_seen_items_never_go_missing(tmp_path):
    _, _, mem, _ = world(tmp_path)
    assert feed(mem, [[]] * 5) == [[]] * 5
    assert mem.state["wallet"].misses == 0 and mem.state["wallet"].status == "unknown"


# --- containment ----------------------------------------------------------------------
def test_containment_is_fraction_inside_not_iou(tmp_path):
    """Small glasses fully inside a big bag: IoU 0.04, fraction inside 1.0 -> inferred."""
    mem, ev = contain(tmp_path, GLASSES_OVER_BAG)
    assert kinds(ev) == ["inferred_contained:glasses"]
    assert (ev[0].from_surface, ev[0].to_surface, ev[0].container) == ("table", "table", "bag")
    o = mem.state["glasses"]
    assert (o.status, o.container, o.surface, o.confidence) == ("inferred", "bag", "table", 0.9)
    assert o.since == 0                                        # when it was last actually seen


def test_containment_confidence_is_clipped(tmp_path):
    half_in = [960, 400, 1060, 430]                           # 40% inside the bag box
    mem, ev = contain(tmp_path / "a", half_in)
    assert ev[0].kind == "inferred_contained" and ev[0].confidence == 0.5
    seventy = [930, 400, 1030, 430]                           # 70% inside
    (tmp_path / "b").mkdir()
    mem, ev = contain(tmp_path / "b", seventy)
    assert ev[0].confidence == 0.7


def test_below_contain_overlap_is_a_disappearance(tmp_path):
    barely = [980, 400, 1080, 430]                            # 20% inside < CONTAIN_OVERLAP 0.25
    mem, ev = contain(tmp_path, barely)
    assert kinds(ev) == ["disappeared:glasses"]
    assert mem.state["glasses"].container is None


def test_contain_overlap_is_configurable(tmp_path):
    barely = [980, 400, 1080, 430]
    _, ev = contain(tmp_path, barely, contain_overlap=0.15)
    assert kinds(ev) == ["inferred_contained:glasses"]


def test_only_enrolled_containers_contain(tmp_path):
    """Keys vanish on top of the wallet (not a container): disappeared, not 'in the wallet'."""
    _, _, mem, _ = world(tmp_path)
    wallet = [880, 580, 990, 660]
    log = feed(mem, [[S("wallet", wallet), S("keys", KEYS_TABLE)]] + [[S("wallet", wallet)]] * 3)
    assert kinds(log[-1]) == ["disappeared:keys"]


def test_people_are_never_containers(tmp_path):
    """A 'person' box is not an item: it can't enter memory, so a hand can't contain anything."""
    _, _, mem, _ = world(tmp_path)
    hand = [800, 500, 1100, 700]
    log = feed(mem, [[S("keys", KEYS_TABLE), S("person", hand)]] + [[S("person", hand)]] * 3)
    assert kinds(log[0]) == ["appeared:keys"] and kinds(log[-1]) == ["disappeared:keys"]
    assert "person" not in mem.state


def test_containment_uses_the_last_frame_the_object_was_in(tmp_path):
    """The bag moves away after the glasses vanish: the guess still uses the frame before."""
    _, _, mem, _ = world(tmp_path)
    log = feed(mem, [[S("bag", BAG_TABLE), S("glasses", GLASSES_OVER_BAG)],
                     [S("bag", BAG_TABLE)], [S("bag", BAG_COUNTER)], [S("bag", BAG_COUNTER)]])
    assert "inferred_contained:glasses" in kinds(log[-1])


def test_container_absent_from_last_frame_means_no_containment(tmp_path):
    _, _, mem, _ = world(tmp_path)
    log = feed(mem, [[S("glasses", GLASSES_OVER_BAG)], [S("bag", BAG_TABLE)],
                     [S("bag", BAG_TABLE)], [S("bag", BAG_TABLE)]])
    assert kinds(log[-1]) == ["disappeared:glasses"]


# --- inferred items ---------------------------------------------------------------------
def test_items_follow_their_container_and_stay_inferred(tmp_path):
    mem, _ = contain(tmp_path, GLASSES_OVER_BAG)
    log = feed(mem, [[S("bag", BAG_COUNTER)], [S("bag", BAG_COUNTER)]], t0=10)
    assert sorted(kinds(log[1])) == ["moved:bag", "moved:glasses"]
    g = next(e for e in log[1] if e.object == "glasses")
    assert (g.from_surface, g.to_surface, g.status, g.container) == ("table", "kitchen counter", "inferred", "bag")
    assert g.photo_path                                         # evidence: the bag in its new place
    assert mem.state["glasses"].status == "inferred"


def test_inferred_is_never_promoted_without_a_sighting(tmp_path):
    mem, _ = contain(tmp_path, GLASSES_OVER_BAG)
    # the bag vanishes too; the glasses stay inferred-in-the-bag, never "observed"
    for i in range(6):
        mem.update([], FRAME, 10 + i)
    g = mem.state["glasses"]
    assert (g.status, g.container) == ("inferred", "bag") and mem.state["bag"].status == "unknown"
    # the bag comes back somewhere else: glasses follow, still inferred
    feed(mem, [[S("bag", BAG_COUNTER)]] * 3, t0=20)
    assert (g.status, g.surface) == ("inferred", "kitchen counter")
    # only seeing the glasses makes them observed again — and clears the container
    [ev] = mem.update([S("bag", BAG_COUNTER), S("glasses", GLASSES_COUNTER)], FRAME, 30)
    assert (ev.kind, ev.from_surface, ev.status, ev.container) == ("reappeared", "kitchen counter", "observed", None)
    assert (g.status, g.container) == ("observed", None)


def test_reappeared_after_disappearing(tmp_path):
    _, _, mem, _ = world(tmp_path)
    log = feed(mem, [[S("keys", KEYS_TABLE)], [], [], [], [S("keys", KEYS_COUNTER)]])
    ev = log[-1][0]
    assert (ev.kind, ev.from_surface, ev.to_surface) == ("reappeared", "table", "kitchen counter")


def test_contained_items_take_the_container_position(tmp_path):
    p2w = lambda u, v: (u / 1000, 0.0, v / 1000)
    _, _, mem, _ = world(tmp_path, pixel_to_world=p2w)
    feed(mem, [[S("bag", BAG_TABLE), S("glasses", GLASSES_OVER_BAG)]] + [[S("bag", BAG_TABLE)]] * 3)
    assert mem.state["glasses"].xyz == mem.state["bag"].xyz == [0.9, 0.0, 0.56]
    feed(mem, [[S("bag", BAG_COUNTER)]] * 2, t0=10)
    assert mem.state["glasses"].xyz == mem.state["bag"].xyz == [0.2, 0.0, 0.56]


def test_held_items_are_neither_seen_nor_missed(tmp_path):
    _, _, mem, _ = world(tmp_path)
    mem.update([S("keys", KEYS_TABLE)], FRAME, 0)
    for i in range(5):
        assert mem.update([], FRAME, 1 + i, held={"keys"}) == []
    assert mem.state["keys"].misses == 0 and mem.state["keys"].status == "observed"


# --- persistence -------------------------------------------------------------------------
STATE_KEYS = ["object", "name", "xyz", "surface", "status", "confidence", "since", "last_observed",
              "container", "last_cam", "last_bbox", "photo_path", "misses", "pending_surface",
              "pending_count", "context"]
EVENT_KEYS = ["object", "name", "kind", "from_surface", "to_surface", "from_xyz", "to_xyz", "status",
              "container", "confidence", "ts", "time", "photo_path"]


def test_files_keep_the_documented_format(tmp_path):
    _, _, mem, _ = world(tmp_path)
    feed(mem, [[S("keys", KEYS_COUNTER), S("bag", BAG_TABLE)],
               [S("keys", KEYS_TABLE), S("bag", BAG_TABLE)]])
    state = json.loads((tmp_path / "state.json").read_text())
    assert set(state) == set(ITEMS) and all(list(o) == STATE_KEYS for o in state.values())
    assert state["keys"]["context"] == [{"object": "bag", "bbox": BAG_TABLE}]
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert all(list(e) == EVENT_KEYS for e in events) and len(events) == 2
    zones = json.loads((tmp_path / "zones.json").read_text())
    assert list(zones[0]) == ["name", "box", "prep", "source", "cam"]


def test_restart_reloads_state_and_resets_misses(tmp_path):
    cfg, store, mem, _ = world(tmp_path)
    feed(mem, [[S("keys", KEYS_TABLE)], [], []])
    assert mem.state["keys"].misses == 2
    mem2 = Memory(ITEMS, Places(ZONES), store, cfg)
    assert mem2.state["keys"].misses == 0 and mem2.state["keys"].status == "observed"
    assert feed(mem2, [[], []], t0=5) == [[], []]              # needs K fresh misses again


def test_torn_log_line_is_skipped(tmp_path):
    _, store, mem, q = world(tmp_path)
    mem.update([S("keys", KEYS_TABLE)], FRAME, 0)
    with open(tmp_path / "events.jsonl", "a") as f:
        f.write('{"object": "keys", "na')
    assert len(store.events()) == 1 and len(q.history("keys")) == 1


def test_memory_files_from_before_the_rename_still_load(tmp_path):
    """state.json / events.jsonl / stills/ written before the DESIGN §7 rename (object_id, zone,
    still_path...) load as today's records: a restart carries on and keeps the history."""
    cfg, store, _, q = world(tmp_path)
    old = lambda oid, box, context: {
        "object_id": oid, "name": ITEMS[oid].name, "zone": "table", "status": "observed", "confidence": 0.6,
        "since": 1.0, "last_observed": 1.0, "container": None, "last_cam": 0, "last_bbox": box, "xyz": None,
        "still_path": f"stills/1000_{oid}_appeared.jpg", "misses": 0, "pending_zone": None,
        "pending_count": 0, "context": context}
    (tmp_path / "state.json").write_text(json.dumps({
        "glasses": old("glasses", GLASSES_OVER_BAG, [{"object_id": "bag", "bbox": BAG_TABLE}]),
        "bag": old("bag", BAG_TABLE, [{"object_id": "glasses", "bbox": GLASSES_OVER_BAG}])}))
    (tmp_path / "events.jsonl").write_text("".join(json.dumps({
        "object_id": oid, "name": oid, "kind": "appeared", "from_zone": None, "to_zone": "table",
        "status": "observed", "container": None, "confidence": 0.6, "ts": 1.0, "time": "00:00:01",
        "still_path": f"stills/1000_{oid}_appeared.jpg"}) + "\n" for oid in ("glasses", "bag")))
    (tmp_path / "stills").mkdir()
    (tmp_path / "stills" / "1000_glasses_appeared.jpg").write_bytes(b"jpg")
    mem = Memory(ITEMS, Places(ZONES), store, cfg)
    g = mem.state["glasses"]
    assert (g.object, g.surface, g.photo_path) == ("glasses", "table", "stills/1000_glasses_appeared.jpg")
    assert q.where_is("glasses", now=2, stamped=False).text.startswith("Your glasses are on the table")
    [e] = q.history("glasses")
    assert (e.object, e.from_surface, e.to_surface, e.from_xyz, e.to_xyz) == ("glasses", None, "table", None, None)
    assert store.photo_file(e.photo_path) == (tmp_path / "stills" / "1000_glasses_appeared.jpg").resolve()
    # The old context (who it was seen with) still drives containment.
    log = feed(mem, [[S("bag", BAG_TABLE)]] * cfg.missing_after_k, t0=5)
    assert kinds(log[-1]) == ["inferred_contained:glasses"] and len(q.history("glasses")) == 2
    store.clear()
    assert not (tmp_path / "stills").exists()


# --- review follow-ups: rules a mutation could otherwise break silently ---------------------------
def test_zone_is_where_the_base_point_sits():
    """A tall bag seen from the corner: centre above the table zone, bottom edge on it -> table."""
    from recall.zones import zone_at
    tall_bag = [800, 150, 1000, 350]                       # centre y 250, base y 350; table starts at 300
    assert zone_at(ZONES, tall_bag) == "table"
    assert zone_at(ZONES, [800, 150, 1000, 295]) == "table"        # base 5 px above: inside the margin
    assert zone_at(ZONES, [800, 150, 1000, 285]) is None           # 15 px above: outside
    assert zone_at(ZONES, [800, 150, 1000, 285], margin=20) == "table"


def test_zone_margin_is_configurable(tmp_path):
    _, _, mem, _ = world(tmp_path, zone_margin_px=20)
    [ev] = mem.update([S("bag", [800, 150, 1000, 285])], FRAME, 0)
    assert ev.to_surface == "table"


def test_contain_overlap_boundary_is_inclusive(tmp_path):
    exactly_quarter = [975, 400, 1075, 430]                # 25 of 100 px wide inside the bag box
    mem, ev = contain(tmp_path, exactly_quarter)
    assert kinds(ev) == ["inferred_contained:glasses"] and ev[0].confidence == 0.5


def test_inferred_zone_is_the_containers_zone(tmp_path):
    """Glasses last seen with their base point on the counter, but mostly inside the bag on the table."""
    _, _, mem, q = world(tmp_path)
    glasses = [560, 400, 760, 430]                         # base (660, 430): no zone; 30% inside the bag
    bag = [700, 350, 1000, 560]
    log = feed(mem, [[S("bag", bag), S("glasses", glasses)]] + [[S("bag", bag)]] * 3)
    [ev] = log[-1]
    assert (ev.kind, ev.from_surface, ev.to_surface) == ("inferred_contained", None, "table")
    assert mem.state["glasses"].surface == "table"
    assert q.state_at(ev.ts)["glasses"] == {"surface": "table", "status": "inferred", "container": "bag"}


def test_state_at_keeps_the_last_known_zone_of_a_disappeared_object(tmp_path):
    _, _, mem, q = world(tmp_path)
    log = feed(mem, [[S("keys", KEYS_TABLE)], [], [], []])
    assert q.state_at(10)["keys"] == {"surface": "table", "status": "unknown", "container": None}
    assert mem.state["keys"].surface == "table"               # same as the live state


def test_a_failing_display_hook_does_not_break_memory(tmp_path, caplog):
    cfg, store, _, _ = world(tmp_path)
    def boom(t):
        raise RuntimeError("bad hook")
    mem = Memory(ITEMS, Places(ZONES), store, cfg, on_event=boom)
    [ev] = mem.update([S("keys", KEYS_TABLE)], FRAME, 0)
    assert ev.kind == "appeared" and len(store.events()) == 1 and "on_event failed" in caplog.text
    assert store.load_state()["keys"].status == "observed"


def test_a_failing_room_lookup_leaves_the_object_untouched(tmp_path):
    def p2w(u, v):
        raise ValueError("mesh not loaded")
    _, _, mem, _ = world(tmp_path, pixel_to_world=p2w)
    import pytest
    with pytest.raises(ValueError):
        mem.update([S("keys", KEYS_TABLE)], FRAME, 0)
    assert mem.state["keys"].last_observed is None


def test_append_after_a_torn_line_keeps_the_new_event(tmp_path):
    _, store, mem, q = world(tmp_path)
    mem.update([S("keys", KEYS_TABLE)], FRAME, 0)
    with open(tmp_path / "events.jsonl", "a") as f:
        f.write('{"object": "keys", "na')           # crash mid-append
    feed(mem, [[], [], []], t0=1)                          # -> disappeared, appended after the tear
    assert kinds(store.events()) == ["appeared:keys", "disappeared:keys"]
