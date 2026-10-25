"""The database copy of the memory (recall/db.py): every write gets there, the live loop never waits
on it, an outage is caught up from events.jsonl without duplicates, photos never leave the laptop.
The page's "This week" comes from the database when it answers in time, else from the laptop.
Over an in-memory fake; with RECALL_TEST_DATABASE_URL set, also over a real Postgres/TimescaleDB."""
import json
import os
import threading
import time
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from fakes import BAG_TABLE, FRAME, GLASSES_OVER_BAG, KEYS_COUNTER, KEYS_TABLE, S, world
from recall import db
from recall.config import load_settings
from recall.db import (Database, change_text, database_url, event_row, object_row, spots_text, week,
                       week_text, where_text)


@pytest.fixture(autouse=True)
def quick(monkeypatch):
    monkeypatch.setattr(db, "FLUSH_EVERY_S", 0.02)
    monkeypatch.setattr(db, "RETRY_MIN_S", 0.02)
    monkeypatch.setattr(db, "RETRY_MAX_S", 0.1)


def until(cond, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


class FakeService:
    """The database, in memory. down=True: unreachable (connecting and writing both fail)."""

    def __init__(self):
        self.events, self.keys, self.objects, self.beat = [], set(), {}, None
        self.down, self.connects = False, 0
        self.gate = None  # a threading.Event to hold every write until it's set

    def connect(self, url):
        if self.down:
            raise ConnectionError("connection refused")
        self.connects += 1
        return FakeTables(self)


class FakeTables:
    def __init__(self, service):
        self.s = service

    def setup(self):
        return True

    def newest(self, home):
        if self.s.down:
            raise ConnectionError("server closed the connection")
        ts = [r[1] for r in self.s.events if r[0] == home]
        return max(ts) if ts else None

    def write(self, events, objects, beat):
        if self.s.gate:
            self.s.gate.wait()
        if self.s.down:
            raise ConnectionError("server closed the connection")
        new = 0
        for r in events:  # the unique key on (home, object, kind, ts)
            key = (r[0], r[2], r[4], r[1])
            if key not in self.s.keys:
                self.s.keys.add(key)
                self.s.events.append(r)
                new += 1
        for o in objects:
            self.s.objects[(o[0], o[1])] = o
        if beat:
            self.s.beat = beat
        return new

    def close(self):
        pass


def memory_with_db(tmp_path, service, home="home"):
    cfg, store, memory, query = world(tmp_path)
    store.db = Database("postgres://fake", home, store.events, connect=service.connect)
    return store, memory


def keys_then_glasses_into_bag(memory, t=0):
    """keys appear on the counter, move to the table; the bag appears and the glasses go into it."""
    for frame in [[S("keys", KEYS_COUNTER)]] * 3 + [[S("keys", KEYS_TABLE)]] * 3 \
            + [[S("keys", KEYS_TABLE), S("bag", BAG_TABLE), S("glasses", GLASSES_OVER_BAG)]] * 3 \
            + [[S("keys", KEYS_TABLE), S("bag", BAG_TABLE)]] * 10:
        t += 1
        memory.update(frame, FRAME, t)
    return t


def sent_kinds(service):
    return sorted((r[2], r[4], r[1]) for r in service.events)


def file_kinds(store, home="home"):
    return sorted((r[2], r[4], r[1]) for r in (event_row(home, e) for e in store.events()))


def test_every_write_reaches_the_database(tmp_path):
    service = FakeService()
    store, memory = memory_with_db(tmp_path, service)
    keys_then_glasses_into_bag(memory)
    kinds = [e.kind for e in store.events()]
    assert {"appeared", "moved", "inferred_contained"} <= set(kinds)
    assert until(lambda: sent_kinds(service) == file_kinds(store))
    assert until(lambda: service.objects.get(("home", "glasses")) is not None
                 and service.objects[("home", "glasses")][3] == "inferred")
    glasses = service.objects[("home", "glasses")]
    assert glasses[5] == "bag" and glasses[4] == "table"  # container, surface
    assert service.beat[0] == "home" and service.beat[2] is True
    store.db.close()


def test_photos_never_reach_the_database(tmp_path):
    service = FakeService()
    store, memory = memory_with_db(tmp_path, service)
    keys_then_glasses_into_bag(memory)
    photos = {e.photo_path for e in store.events() if e.photo_path}
    assert photos and all((tmp_path / p).exists() for p in photos)
    assert until(lambda: len(service.events) == len(store.events()))
    sent = [v for r in [*service.events, *service.objects.values()] for v in r]
    assert not any(isinstance(v, str) and ("photo" in v or v.endswith(".jpg")) for v in sent)
    store.db.close()


def test_the_live_loop_never_waits_for_the_database(tmp_path):
    service = FakeService()
    service.gate = threading.Event()  # every write hangs, like a dead connection mid-request
    store, memory = memory_with_db(tmp_path, service)
    start = time.monotonic()
    keys_then_glasses_into_bag(memory)
    assert time.monotonic() - start < 2.0 and not service.events  # memory and files kept going
    assert file_kinds(store)
    service.gate.set()
    assert until(lambda: sent_kinds(service) == file_kinds(store))
    store.db.close()


def test_an_outage_is_caught_up_from_the_file_without_duplicates(tmp_path):
    service = FakeService()
    store, memory = memory_with_db(tmp_path, service)
    t = 0
    for _ in range(3):
        t += 1
        memory.update([S("keys", KEYS_COUNTER)], FRAME, t)
    assert until(lambda: len(service.events) == 1)  # keys appeared
    service.down = True                              # wifi drops
    t = keys_then_glasses_into_bag(memory, t)        # keys move, glasses go into the bag meanwhile
    time.sleep(0.2)
    assert len(service.events) == 1
    service.down = False                             # back
    assert until(lambda: sent_kinds(service) == file_kinds(store))
    assert len(service.keys) == len(service.events) == len(store.events())
    assert service.connects >= 2
    store.db.close()


def test_a_restart_sends_only_what_the_database_is_missing(tmp_path):
    service = FakeService()
    store, memory = memory_with_db(tmp_path, service)
    keys_then_glasses_into_bag(memory)
    assert until(lambda: len(service.events) == len(store.events()))
    store.db.close()
    sent_before = service.events[:]
    store.db = Database("postgres://fake", "home", store.events, connect=service.connect)  # live again
    assert until(lambda: not store.db._pending())
    assert service.events == sent_before and store.db.sent == 0
    store.db.close()


def test_close_sends_the_last_heartbeat(tmp_path):
    service = FakeService()
    store, memory = memory_with_db(tmp_path, service)
    keys_then_glasses_into_bag(memory)
    store.beat(99.0, camera_ok=False)  # what the pipeline writes when it stops
    store.db.close()
    assert service.beat == ("home", datetime.fromtimestamp(99.0, timezone.utc), False)


def test_households_stay_apart(tmp_path):
    service = FakeService()
    a, ma = memory_with_db(tmp_path / "a", service, home="grandmas-kitchen")
    b, mb = memory_with_db(tmp_path / "b", service, home="lab-bench")
    keys_then_glasses_into_bag(ma)
    for t in (1, 2, 3):
        mb.update([S("keys", KEYS_TABLE)], FRAME, t)
    assert until(lambda: len(service.events) == len(a.events()) + len(b.events()))
    # the state can land a send after the events (memory logs a change before saving the state)
    assert until(lambda: {k[0] for k in service.objects} == {"grandmas-kitchen", "lab-bench"})
    assert sorted(r[2] for r in service.events if r[0] == "lab-bench") == ["keys"]
    a.db.close()
    b.db.close()


def test_no_database_url_no_database(tmp_path):
    cfg, store, memory, query = world(tmp_path)
    assert database_url({}) is None
    assert Database.from_env(cfg, store, env={}) is None
    assert database_url({"DATABASE_URL": "postgres://x"}) == "postgres://x"
    assert load_settings().db_home == "home"
    assert load_settings(["DB_HOME=grandmas-kitchen"]).db_home == "grandmas-kitchen"


# --- what the command line prints -----------------------------------------------------------
def at(ts):
    return datetime.fromtimestamp(ts, timezone.utc)


def test_where_text_says_how_sure_and_whether_the_camera_is_live():
    now = 10_000.0
    objects = [
        {"object": "bag", "name": "blue bag", "status": "observed", "surface": "table", "container": None,
         "confidence": 0.7, "since": at(now - 300), "last_observed": at(now - 30)},
        {"object": "glasses", "name": "glasses", "status": "inferred", "surface": "table", "container": "bag",
         "confidence": 0.8, "since": at(now - 200), "last_observed": at(now - 200)},
        {"object": "phone", "name": "phone", "status": "unknown", "surface": None, "container": None,
         "confidence": 0.0, "since": None, "last_observed": None},
    ]
    live = where_text(objects, {"last_frame": at(now - 1), "camera_ok": True, "updated_at": at(now)}, now, 5)
    assert live.splitlines()[0] == "Camera live."
    assert "blue bag     on the table, seen" in live
    assert "glasses      probably in the blue bag (0.80)" in live
    assert "phone        not seen yet" in live
    paused = where_text(objects, {"last_frame": at(now - 60), "camera_ok": True, "updated_at": at(now)}, now, 5)
    assert paused.startswith("Camera paused since")
    assert where_text([], None, now, 5) == "The camera has never reported."


def test_change_and_week_text():
    e = {"ts": at(10_000), "object": "glasses", "name": "glasses", "kind": "inferred_contained",
         "from_surface": "table", "to_surface": "table", "container": "bag", "confidence": 0.8}
    assert change_text(e, {"bag": "blue bag"}).endswith("glasses went into the blue bag (inferred, 0.80)")
    moved = {**e, "kind": "moved", "from_surface": "kitchen counter", "to_surface": None}
    assert change_text(moved, {}).endswith("glasses moved from the kitchen counter to outside the named places")
    nowhere = {**e, "kind": "reappeared", "to_surface": None}
    assert change_text(nowhere, {}).endswith("glasses reappeared outside the named places")
    assert change_text({**e, "kind": "appeared"}, {}).endswith("glasses appeared on the table")
    assert change_text({**e, "kind": "disappeared"}, {}).endswith("glasses disappeared from the table")
    day = datetime(2026, 10, 24, tzinfo=timezone.utc).astimezone()
    rows = [{"day": day, "name": "glasses", "moved": 0, "gone": 3, "events": 5},
            {"day": day, "name": "keys", "moved": 2, "gone": 0, "events": 3}]
    assert week_text(rows).splitlines()[1:] == ["  glasses      went out of sight 3x", "  keys         moved 2x"]
    assert week_text([]) == "Nothing happened in the last 7 days."


# --- the page's "This week" ------------------------------------------------------------------
NAMES = {"keys": "keys", "glasses": "glasses", "bag": "blue bag", "wallet": "wallet", "phone": "phone"}


def a_week(memory, t):
    """keys and glasses as in keys_then_glasses_into_bag, then the keys vanish from the table (not
    into anything): glasses went into the blue bag on the table, keys disappeared from the table."""
    t = keys_then_glasses_into_bag(memory, t)
    for _ in range(5):
        t += 1
        memory.update([S("bag", BAG_TABLE)], FRAME, t)
    return t


def test_this_week_from_the_laptop(tmp_path):
    cfg, store, memory, query = world(tmp_path)
    now = time.time()
    a_week(memory, now - 3600)
    w = week(store, NAMES, now=now)
    assert w["source"] == "this laptop" and len(w["days"]) == 7
    assert w["days"][-1] == datetime.fromtimestamp(now).date().isoformat()
    items = {it["name"]: it for it in w["items"]}
    assert sum(items["glasses"]["per_day"]) == 1 and sum(items["keys"]["per_day"]) == 1
    assert sum(items["keys"]["moved"]) == 1 and sum(items["blue bag"]["per_day"]) == 0
    assert {"name": "glasses", "place": "table", "container": "blue bag", "times": 1} in w["spots"]
    assert {"name": "keys", "place": "table", "container": None, "times": 1} in w["spots"]


def test_this_week_leaves_out_older_days(tmp_path):
    cfg, store, memory, query = world(tmp_path)
    now = time.time()
    a_week(memory, now - 10 * 86400)
    w = week(store, NAMES, now=now)
    assert w["items"] == [] and w["spots"] == []


class FakeWeekTables:
    def __init__(self, delay=0.0, fail=False):
        self.delay, self.fail, self.closed = delay, fail, False

    def connect(self, url):
        time.sleep(self.delay)
        if self.fail:
            raise ConnectionError("could not connect to server")
        return self

    def per_day(self, home, days, tz):
        today = datetime.now().astimezone().replace(hour=0, minute=0, second=0, microsecond=0)
        return [{"day": today, "name": "glasses", "moved": 0, "gone": 3, "events": 4},
                {"day": today - timedelta(days=2), "name": "keys", "moved": 2, "gone": 1, "events": 3}]

    def lost_spots(self, home, days):
        return [{"object": "glasses", "name": "glasses", "place": "table", "container": "blue bag",
                 "times": 3, "last": None}]

    def close(self):
        self.closed = True


def store_with_database(tmp_path):
    cfg, store, memory, query = world(tmp_path)
    store.db = Database("postgres://fake", "home", store.events, connect=FakeService().connect)
    return store, memory


def test_this_week_from_the_database(tmp_path):
    store, memory = store_with_database(tmp_path)
    fake = FakeWeekTables()
    w = week(store, NAMES, connect=fake.connect)
    assert w["source"] == "database" and fake.closed
    items = {it["name"]: it for it in w["items"]}
    assert items["glasses"]["per_day"] == [0, 0, 0, 0, 0, 0, 3]
    assert items["keys"]["per_day"][4] == 1 and items["keys"]["moved"][4] == 2
    assert w["spots"] == [{"name": "glasses", "place": "table", "container": "blue bag", "times": 3}]
    store.db.close()


def test_this_week_falls_back_to_the_laptop_when_the_database_fails_or_hangs(tmp_path, monkeypatch):
    store, memory = store_with_database(tmp_path)
    a_week(memory, time.time() - 3600)
    assert week(store, NAMES, connect=FakeWeekTables(fail=True).connect)["source"] == "this laptop"
    monkeypatch.setattr(db, "WEB_WAIT_S", 0.2)
    hung = FakeWeekTables(delay=1.0)
    start = time.monotonic()
    w = week(store, NAMES, connect=hung.connect)
    assert w["source"] == "this laptop" and time.monotonic() - start < 0.6
    assert sum(next(it for it in w["items"] if it["name"] == "glasses")["per_day"]) == 1
    # while that one still hangs, the next asks nobody and answers at once
    start = time.monotonic()
    assert week(store, NAMES, connect=FakeWeekTables().connect)["source"] == "this laptop"
    assert time.monotonic() - start < 0.1
    assert until(lambda: db._asking.acquire(blocking=False))  # the hung one finished and let go
    db._asking.release()
    store.db.close()


def test_this_week_over_http(tmp_path):
    from recall.web import Web, lan_ip
    from test_web import StubPipeline
    cfg, store, memory, query = world(tmp_path)
    a_week(memory, time.time() - 3600)
    web = Web(cfg, memory.items, memory, StubPipeline(), memory.places)
    url = web.start(port=0).replace(lan_ip(), "127.0.0.1")
    w = json.loads(urllib.request.urlopen(url + "/week", timeout=5).read())
    assert w["source"] == "this laptop" and len(w["days"]) == 7
    assert {it["name"] for it in w["items"]} == {"blue bag", "glasses", "keys"}
    page = urllib.request.urlopen(url + "/", timeout=5).read().decode()
    assert 'id="weekBox"' in page and "/week" in page


def test_spots_text():
    rows = [{"name": "glasses", "place": "table", "container": "blue bag", "times": 3},
            {"name": "glasses", "place": "kitchen counter", "container": None, "times": 1},
            {"name": "keys", "place": None, "container": None, "times": 2}]
    assert spots_text(rows).splitlines() == ["glasses", "    3x  into the blue bag on the table",
                                             "    1x  on the kitchen counter", "keys",
                                             "    2x  outside the named places"]
    assert spots_text([]) == "Nothing went out of sight in the last 30 days."


# --- a real database: RECALL_TEST_DATABASE_URL=postgres://... (Postgres, or TimescaleDB) --------
REAL = os.environ.get("RECALL_TEST_DATABASE_URL")
real = pytest.mark.skipif(not REAL, reason="set RECALL_TEST_DATABASE_URL to test against a real database")


@pytest.fixture
def real_home():
    home = f"test-{uuid.uuid4().hex[:10]}"
    yield home
    tables = db.Tables.connect(REAL)
    for table in ("recall_events", "recall_objects", "recall_heartbeat"):
        tables.conn.execute(f"DELETE FROM {table} WHERE home = %s", (home,))
    tables.close()


@real
def test_real_setup_is_repeatable_and_makes_a_hypertable_on_timescaledb():
    tables = db.Tables.connect(REAL)
    first, again = tables.setup(), tables.setup()
    tsdb = tables.conn.execute("SELECT 1 FROM pg_extension WHERE extname = 'timescaledb'").fetchone()
    assert first == again == bool(tsdb)
    tables.close()


@real
def test_real_writes_are_safe_to_repeat(tmp_path, real_home):
    cfg, store, memory, query = world(tmp_path)
    keys_then_glasses_into_bag(memory)
    tables = db.Tables.connect(REAL)
    tables.setup()
    rows = [event_row(real_home, e) for e in store.events()]
    assert tables.write(rows) == len(rows)
    assert tables.write(rows) == 0  # the same events again: nothing new
    assert tables.newest(real_home) == max(r[1] for r in rows)
    tables.close()


@real
def test_real_live_copy_then_the_care_partners_queries(tmp_path, real_home):
    cfg, store, memory, query = world(tmp_path)
    store.db = Database(REAL, real_home, store.events)
    t0 = time.time() - 3600  # an hour ago, so "today" and "the last 7 days" include it
    keys_then_glasses_into_bag(memory, t0)
    store.db.close()
    tables = db.Tables.connect(REAL)
    assert len(tables.changes(real_home, at(t0 - 1))) == len(store.events())
    objs = {o["object"]: o for o in tables.objects(real_home)}
    assert objs["glasses"]["status"] == "inferred" and objs["glasses"]["container"] == "bag"
    assert tables.heartbeat(real_home)["last_frame"] is not None
    week = {r["name"]: r for r in tables.per_day(real_home, 7, db.local_tz())}
    assert week["keys"]["moved"] == 1 and week["glasses"]["gone"] == 1
    store.db = Database(REAL, real_home, store.events)  # a restart: nothing sent twice
    assert until(lambda: not store.db._pending())
    store.db.close()
    assert len(tables.changes(real_home, at(t0 - 1))) == len(store.events())
    tables.close()


@real
def test_real_daily_summary_matches_the_events_and_says_where_things_go_missing(tmp_path, real_home):
    cfg, store, memory, query = world(tmp_path)
    now = time.time()
    a_week(memory, now - 2 * 86400)
    tables = db.Tables.connect(REAL)
    hyper = tables.setup()
    tz = db.local_tz()
    assert tables.daily_in(tz) == hyper  # recall_daily on TimescaleDB, none on plain Postgres
    tables.write([event_row(real_home, e) for e in store.events()],
                 [object_row(real_home, o) for o in memory.state.values()])
    from_events = tables.per_day(real_home, 7, tz, daily=False)
    # Days already counted (an earlier run, the policy) show late events once recounted; live's
    # catch-up and sync do this after sending.
    tables.refresh_daily(at(now - 3 * 86400))
    assert tables.per_day(real_home, 7, tz) == from_events
    week_rows = {r["name"]: r for r in from_events}
    assert week_rows["glasses"]["gone"] == 1 and week_rows["keys"]["moved"] == 1
    spots = [(r["name"], r["place"], r["container"], r["times"]) for r in tables.lost_spots(real_home, 7)]
    assert ("glasses", "table", "blue bag", 1) in spots and ("keys", "table", None, 1) in spots
    tables.close()


@real
def test_real_daily_counts_late_events_from_the_middle_of_an_already_counted_day(tmp_path, real_home):
    """Live's catch-up and sync refresh from their earliest event. That event's own day must be
    recounted too, even though the window starts partway through it."""
    from zoneinfo import ZoneInfo
    cfg, store, memory, query = world(tmp_path)
    tz = db.local_tz()
    midday = (datetime.now(ZoneInfo(tz)) - timedelta(days=2)).replace(hour=13, minute=0, second=0,
                                                                       microsecond=0).timestamp()
    a_week(memory, midday)
    tables = db.Tables.connect(REAL)
    if not tables.setup():
        tables.close()
        pytest.skip("plain Postgres: no recall_daily")
    assert tables.daily_in(tz)
    # Hold the policy so it can't recount behind the test's back (it runs as soon as it's made).
    job = tables.conn.execute(
        "SELECT job_id FROM timescaledb_information.jobs WHERE proc_name = 'policy_refresh_continuous_aggregate'"
        " AND hypertable_name = 'recall_daily' AND hypertable_schema = current_schema()").fetchone()[0]
    tables.conn.execute("SELECT alter_job(%s, scheduled => false)", (job,))
    try:
        # the policy has already counted every whole day up to an hour ago, before these events arrive
        tables.conn.execute("CALL refresh_continuous_aggregate('recall_daily', NULL, %s)",
                            (datetime.now(timezone.utc) - timedelta(hours=1),))
        events = store.events()
        tables.write([event_row(real_home, e) for e in events],
                     [object_row(real_home, o) for o in memory.state.values()])
        tables.refresh_daily(min(at(e.ts) for e in events))
        from_events = tables.per_day(real_home, 7, tz, daily=False)
        assert any(r["gone"] or r["moved"] for r in from_events)
        assert tables.per_day(real_home, 7, tz) == from_events
    finally:
        tables.conn.execute("SELECT alter_job(%s, scheduled => true)", (job,))
        tables.close()


@real
def test_real_this_week_comes_from_the_database(tmp_path, real_home):
    cfg, store, memory, query = world(tmp_path)
    store.db = Database(REAL, real_home, store.events)
    a_week(memory, time.time() - 3600)
    assert until(lambda: not store.db._pending() and store.db.sent == len(store.events()))
    w = week(store, NAMES)
    assert w["source"] == "database"
    assert w == {**week(Store_without_db(store), NAMES), "source": "database"}  # the same numbers
    store.db.close()


def Store_without_db(store):
    from recall.store import Store
    return Store(store.root)
