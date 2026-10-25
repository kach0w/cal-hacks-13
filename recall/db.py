"""The database: the memory's history, what's where now, and whether the camera is live, in
Postgres (a Tiger Cloud service: Postgres with TimescaleDB). Questions about time (what changed
today, how often the glasses went missing this week) are asked of it from any computer, so the
care partner doesn't need the laptop.

    export DATABASE_URL=postgres://...      # the service's connection string (or a line in .env)
    python -m recall.db setup               # make the tables (live does this itself)
    python -m recall.db sync                # send what's in data/ now (live catches up by itself)
    python -m recall.db where               # what's where, and whether the camera is live
    python -m recall.db changes [SINCE]     # what changed: today (default), 2h, 14:30, 2:40pm
    python -m recall.db week                # per day, what each thing did, the last 7 days
    python -m recall.db spots               # where each thing usually goes missing, the last 30 days
    --set DB_HOME=NAME                      # which household (default "home"): one service, many rooms

data/ stays the memory's own copy: live writes its files first and never waits on the network
(store.Store hands each write to Database, which returns at once). A writer thread sends what
changed, at most once per FLUSH_EVERY_S. When the connection drops it backs off and, once back,
resends events.jsonl from the newest event the database has; the unique key on
(home, object, kind, ts) makes every send safe to repeat. Photos never leave the laptop:
photo_path isn't sent.

Events go in a TimescaleDB hypertable on ts, segmented by home and object (moved to the
columnstore after a chunk, 7 days). A continuous aggregate, recall_daily, keeps per local day,
home, thing and kind of event a count, refreshed by a policy and real-time for the newest hour,
so the week's summary reads a few rows a day instead of every event. On plain Postgres (tests, a
local server) events are an ordinary table, there is no recall_daily, and the same questions are
asked of the events themselves.

The web page's "This week" panel asks through week(): the database when live has one, within
WEB_WAIT_S, else the same numbers from events.jsonl on this laptop.
"""
import argparse
import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from . import config
from .moss_search import ENV_FILE, read_env

log = logging.getLogger("recall")

FLUSH_EVERY_S = 1.0    # live saves the state on every look; the database needn't hear that often
RETRY_MIN_S, RETRY_MAX_S = 1.0, 60.0  # backoff while the database can't be reached
CONNECT_TIMEOUT_S = 10
CLOSE_WAIT_S = 5.0     # on stop, how long the last send may take
WEB_CONNECT_S = 2      # the page's "This week": connecting may take this long
WEB_WAIT_S = 3.0       # and the whole answer this long, else it comes from this laptop's files
GONE = ("disappeared", "inferred_contained")  # the kinds of event that mean "went out of sight"

EVENTS = """CREATE TABLE IF NOT EXISTS recall_events (
    ts timestamptz NOT NULL,
    home text NOT NULL,
    object text NOT NULL,
    name text NOT NULL,
    kind text NOT NULL,
    from_surface text,
    to_surface text,
    from_xyz double precision[],
    to_xyz double precision[],
    status text,
    container text,
    confidence double precision,
    UNIQUE (home, object, kind, ts)
)"""
# TimescaleDB 2.20+; the unique key's columns must all be in segmentby or orderby for the columnstore.
HYPERTABLE = """ WITH (tsdb.hypertable, tsdb.partition_column = 'ts',
    tsdb.segmentby = 'home, object', tsdb.orderby = 'ts DESC, kind')"""
TABLES = [
    """CREATE TABLE IF NOT EXISTS recall_objects (
        home text NOT NULL,
        object text NOT NULL,
        name text NOT NULL,
        status text NOT NULL,
        surface text,
        container text,
        confidence double precision,
        xyz double precision[],
        since timestamptz,
        last_observed timestamptz,
        updated_at timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY (home, object)
    )""",
    """CREATE TABLE IF NOT EXISTS recall_heartbeat (
        home text PRIMARY KEY,
        last_frame timestamptz,
        camera_ok boolean NOT NULL,
        updated_at timestamptz NOT NULL DEFAULT now()
    )""",
    "CREATE INDEX IF NOT EXISTS recall_events_home_ts ON recall_events (home, ts DESC)",
]
INSERT_EVENT = """INSERT INTO recall_events (home, ts, object, name, kind, from_surface, to_surface,
    from_xyz, to_xyz, status, container, confidence)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING"""
UPSERT_OBJECT = """INSERT INTO recall_objects (home, object, name, status, surface, container,
    confidence, xyz, since, last_observed)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (home, object) DO UPDATE SET name = EXCLUDED.name, status = EXCLUDED.status,
        surface = EXCLUDED.surface, container = EXCLUDED.container, confidence = EXCLUDED.confidence,
        xyz = EXCLUDED.xyz, since = EXCLUDED.since, last_observed = EXCLUDED.last_observed,
        updated_at = now()"""
# Per local day (the home's time zone when it was made), home, thing and kind: a count. Real-time
# (materialized_only = false): the newest hour, not yet materialized, comes from recall_events.
DAILY = """CREATE MATERIALIZED VIEW IF NOT EXISTS recall_daily
    WITH (timescaledb.continuous, timescaledb.materialized_only = false) AS
    SELECT time_bucket('1 day', ts, {tz}) AS day, home, object, name, kind, count(*) AS n
    FROM recall_events GROUP BY day, home, object, name, kind WITH NO DATA"""
DAILY_POLICY = """SELECT add_continuous_aggregate_policy('recall_daily', start_offset => INTERVAL '35 days',
    end_offset => INTERVAL '1 hour', schedule_interval => INTERVAL '10 minutes', if_not_exists => true)"""
UPSERT_BEAT = """INSERT INTO recall_heartbeat (home, last_frame, camera_ok) VALUES (%s, %s, %s)
    ON CONFLICT (home) DO UPDATE SET last_frame = EXCLUDED.last_frame,
        camera_ok = EXCLUDED.camera_ok, updated_at = now()"""


def utc(ts) -> datetime | None:
    return None if ts is None else datetime.fromtimestamp(ts, timezone.utc)


def event_row(home, t) -> tuple:
    """A Transition as an INSERT_EVENT row. No photo_path: photos stay on the laptop."""
    return (home, utc(t.ts), t.object, t.name, t.kind, t.from_surface, t.to_surface,
            t.from_xyz, t.to_xyz, t.status, t.container, t.confidence)


def object_row(home, o) -> tuple:
    """An ObjectState as an UPSERT_OBJECT row."""
    return (home, o.object, o.name, o.status, o.surface, o.container, o.confidence,
            o.xyz, utc(o.since), utc(o.last_observed))


def database_url(env=None) -> str | None:
    env = {**read_env(ENV_FILE), **os.environ} if env is None else env
    return env.get("DATABASE_URL") or None


class Tables:
    """The SQL, on one open connection (psycopg 3, autocommit; writes in explicit transactions)."""

    def __init__(self, conn):
        self.conn = conn

    @classmethod
    def connect(cls, url, timeout=CONNECT_TIMEOUT_S) -> "Tables":
        import psycopg
        # Keepalives: a link that died quietly (the wifi went) is noticed in ~20 s, not minutes.
        return cls(psycopg.connect(url, autocommit=True, connect_timeout=timeout,
                                   application_name="recall", keepalives=1, keepalives_idle=5,
                                   keepalives_interval=5, keepalives_count=3))

    def close(self):
        self.conn.close()

    def setup(self, tz=None) -> bool:
        """Makes the tables that are missing. True when recall_events is a hypertable; then also
        recall_daily, its days in tz (default: this computer's time zone)."""
        tsdb = self.conn.execute("SELECT 1 FROM pg_extension WHERE extname = 'timescaledb'").fetchone()
        with self.conn.transaction():
            if tsdb:
                try:
                    with self.conn.transaction():
                        self.conn.execute(EVENTS + HYPERTABLE)
                except Exception as e:  # TimescaleDB before 2.20: make it a hypertable the older way
                    log.info("database: %s; using create_hypertable", str(e).splitlines()[0])
                    self.conn.execute(EVENTS)
                    self.conn.execute("SELECT create_hypertable('recall_events', 'ts', if_not_exists => TRUE)")
            else:
                self.conn.execute(EVENTS)
            for sql in TABLES:
                self.conn.execute(sql)
        hyper = self.is_hypertable()
        if hyper:
            self.setup_daily(tz or local_tz())
        return hyper

    def setup_daily(self, tz):
        """recall_daily and its refresh policy. Without them every question still works, from the events."""
        from psycopg import sql
        try:
            with self.conn.transaction():
                self.conn.execute(sql.SQL(DAILY).format(tz=sql.Literal(tz)))
                self.conn.execute(DAILY_POLICY)
        except Exception as e:
            log.info("database: no daily summary (%s); asking the events instead", str(e).splitlines()[0])

    def is_hypertable(self) -> bool:
        try:
            return self.conn.execute("SELECT 1 FROM timescaledb_information.hypertables WHERE "
                                     "hypertable_name = 'recall_events' AND hypertable_schema = current_schema()"
                                     ).fetchone() is not None
        except Exception:  # no TimescaleDB
            return False

    def refresh_daily(self, since: datetime):
        """Recounts recall_daily's whole days from since. Events that arrive late (a catch-up after
        an outage, a sync) land in days it has already counted, which it would otherwise show
        unchanged until its policy's next run; today's events never need this (real-time).
        TimescaleDB only recounts days wholly inside the window, so it starts a day before since:
        that covers the start of since's local day in any time zone."""
        try:
            self.conn.execute("CALL refresh_continuous_aggregate('recall_daily', %s, %s)",
                              (since - timedelta(days=1), datetime.now(timezone.utc) - timedelta(hours=1)))
        except Exception as e:  # no recall_daily, or nothing whole to recount
            log.debug("database: recall_daily not refreshed (%s)", str(e).splitlines()[0])

    def daily_in(self, tz) -> bool:
        """Whether recall_daily is there with its days in tz (days in another zone don't add up)."""
        try:
            r = self.conn.execute("SELECT view_definition FROM timescaledb_information.continuous_aggregates "
                                  "WHERE view_name = 'recall_daily' AND view_schema = current_schema()").fetchone()
        except Exception:  # no TimescaleDB
            return False
        return r is not None and f"'{tz}'" in r[0]

    def newest(self, home) -> datetime | None:
        return self.conn.execute("SELECT max(ts) FROM recall_events WHERE home = %s", (home,)).fetchone()[0]

    def write(self, events=(), objects=(), beat=None) -> int:
        """Rows from event_row / object_row and (home, last_frame, camera_ok), in one transaction.
        Returns how many events were new."""
        new = 0
        with self.conn.transaction(), self.conn.cursor() as cur:
            if events:
                cur.executemany(INSERT_EVENT, list(events))
                new = max(cur.rowcount, 0)
            if objects:
                cur.executemany(UPSERT_OBJECT, list(objects))
            if beat:
                cur.execute(UPSERT_BEAT, beat)
        return new

    # --- what the command line reads -------------------------------------------------------
    def objects(self, home) -> list[dict]:
        cur = self.conn.execute("SELECT object, name, status, surface, container, confidence, since, "
                                "last_observed FROM recall_objects WHERE home = %s ORDER BY name", (home,))
        return [dict(zip([c.name for c in cur.description], r)) for r in cur.fetchall()]

    def heartbeat(self, home) -> dict | None:
        r = self.conn.execute("SELECT last_frame, camera_ok, updated_at FROM recall_heartbeat "
                              "WHERE home = %s", (home,)).fetchone()
        return None if r is None else {"last_frame": r[0], "camera_ok": r[1], "updated_at": r[2]}

    def changes(self, home, since: datetime) -> list[dict]:
        cur = self.conn.execute("SELECT ts, object, name, kind, from_surface, to_surface, container, "
                                "confidence FROM recall_events WHERE home = %s AND ts >= %s ORDER BY ts",
                                (home, since))
        return [dict(zip([c.name for c in cur.description], r)) for r in cur.fetchall()]

    def per_day(self, home, days: int, tz: str, daily=None) -> list[dict]:
        """Per local day and thing, today and the days-1 before it: moves, times it went out of sight,
        and all events. From recall_daily when it's there (daily=False: from the events anyway)."""
        start = "date_trunc('day', now() - make_interval(days => %(days)s - 1), %(tz)s)"
        args = {"home": home, "days": days, "tz": tz}
        if daily is None:
            daily = self.daily_in(tz)
        if daily:
            sql = f"""SELECT day, name,
                    coalesce(sum(n) FILTER (WHERE kind = 'moved'), 0)::bigint AS moved,
                    coalesce(sum(n) FILTER (WHERE kind IN ('disappeared', 'inferred_contained')), 0)::bigint AS gone,
                    sum(n)::bigint AS events
                FROM recall_daily WHERE home = %(home)s AND day >= {start}
                GROUP BY day, name ORDER BY day, name"""
        else:
            bucket = ("time_bucket('1 day', ts, %(tz)s)" if self.is_hypertable()
                      else "date_trunc('day', ts, %(tz)s)")
            sql = f"""SELECT {bucket} AS day, name,
                    count(*) FILTER (WHERE kind = 'moved') AS moved,
                    count(*) FILTER (WHERE kind IN ('disappeared', 'inferred_contained')) AS gone,
                    count(*) AS events
                FROM recall_events WHERE home = %(home)s AND ts >= {start}
                GROUP BY day, name ORDER BY day, name"""
        cur = self.conn.execute(sql, args)
        return [dict(zip([c.name for c in cur.description], r)) for r in cur.fetchall()]

    def lost_spots(self, home, days: int) -> list[dict]:
        """Where each thing usually goes missing: per thing, the place it was last seen on and the
        container it went into, with how many times in the last days, most often first."""
        cur = self.conn.execute(
            """SELECT e.object, e.name, e.from_surface AS place, coalesce(c.name, e.container) AS container,
                    count(*) AS times, max(e.ts) AS last
                FROM recall_events e
                LEFT JOIN recall_objects c ON c.home = e.home AND c.object = e.container
                WHERE e.home = %(home)s AND e.kind IN ('disappeared', 'inferred_contained')
                    AND e.ts >= now() - make_interval(days => %(days)s)
                GROUP BY e.object, e.name, e.from_surface, coalesce(c.name, e.container)
                ORDER BY e.name, times DESC, last DESC""", {"home": home, "days": days})
        return [dict(zip([c.name for c in cur.description], r)) for r in cur.fetchall()]


class Database:
    """Copies the memory to the database on its own thread. store.Store calls event / state /
    beat after writing its own files; each returns at once, whatever the network is doing."""

    def __init__(self, url, home, local_events, connect=Tables.connect):
        self.url, self.home = url, home
        self.local_events = local_events  # () -> [Transition]: events.jsonl, for catching up
        self._connect = connect
        self._lock = threading.Lock()
        self._wake, self._stop = threading.Event(), threading.Event()
        self._events, self._objects, self._beat = [], None, None
        self._tables = None
        self._catch_up = True   # on (re)connecting: resend events.jsonl from the database's newest
        self._failing = False   # logged once per outage, not every retry
        self.sent = 0           # events the database didn't have yet, so far
        self.hypertable = None  # known once connected
        self._thread = threading.Thread(target=self._run, name="database", daemon=True)
        self._thread.start()

    @classmethod
    def from_env(cls, cfg, store, env=None) -> "Database | None":
        """The live memory's database when DATABASE_URL is set (environment or .env), else None."""
        url = database_url(env)
        if not url:
            return None
        try:
            import psycopg  # noqa: F401
        except ImportError:
            log.warning("database: DATABASE_URL is set but psycopg isn't installed "
                        "(pip install -r requirements.txt); the memory stays in %s only", store.root)
            return None
        log.info("database: copying the memory there as home %r", cfg.db_home)
        return cls(url, cfg.db_home, store.events)

    # --- the store's side: never blocks --------------------------------------------------
    def event(self, t):
        with self._lock:
            self._events.append(event_row(self.home, t))
        self._wake.set()

    def state(self, state: dict):
        rows = [object_row(self.home, o) for o in state.values()]  # a snapshot: memory keeps mutating
        with self._lock:
            self._objects = rows
        self._wake.set()

    def beat(self, frame_ts, camera_ok=True):
        with self._lock:
            self._beat = (self.home, utc(frame_ts), bool(camera_ok))
        self._wake.set()

    def close(self, wait=CLOSE_WAIT_S):
        """Send what's waiting (the last heartbeat says the camera stopped), then stop."""
        self._stop.set()
        self._wake.set()
        self._thread.join(wait)

    # --- the writer thread ---------------------------------------------------------------
    def _pending(self) -> bool:
        with self._lock:
            return bool(self._catch_up or self._events or self._objects or self._beat)

    def _run(self):
        backoff = 0.0
        while True:
            if not self._pending():
                if self._stop.is_set():
                    return
                self._wake.wait()
            self._wake.clear()
            stopping = self._stop.is_set()
            try:
                self._flush()
            except Exception as e:
                self._lost(e)
                if stopping:
                    return
                backoff = min(max(RETRY_MIN_S, backoff * 2), RETRY_MAX_S)
                self._stop.wait(backoff)  # close() cuts the wait short for one last try
                continue
            backoff = 0.0
            if stopping:
                return
            self._stop.wait(FLUSH_EVERY_S)

    def _flush(self):
        if self._tables is None:
            self._tables = self._connect(self.url)
            self.hypertable = self._tables.setup()
            self._catch_up = True
            log.info("database: connected%s", "" if self.hypertable else " (plain Postgres: no hypertable)")
            self._failing = False
        with self._lock:
            events, objects, beat = self._events, self._objects, self._beat
            self._events, self._objects, self._beat = [], None, None
            catch_up = self._catch_up
        if catch_up:
            # Everything queued is in events.jsonl already (the store writes its file first), so
            # the file alone says what the database is missing.
            newest = self._tables.newest(self.home)
            events = [event_row(self.home, t) for t in self.local_events()
                      if newest is None or utc(t.ts) >= newest]
        try:
            new = self._tables.write(events, objects or (), beat)
        except BaseException:
            with self._lock:  # newer state and heartbeat win; events come back from the file
                self._objects = self._objects or objects
                self._beat = self._beat or beat
            raise
        if catch_up:
            if new and self.hypertable:
                self._tables.refresh_daily(min(r[1] for r in events))
            with self._lock:
                self._catch_up = False
        if new:
            log.debug("database: %d new event(s)", new)
        self.sent += new

    def _lost(self, e):
        if not self._failing:
            log.warning("database: can't reach it (%s); the memory stays in data/ and catches up "
                        "when it's back", str(e).splitlines()[0] if str(e) else type(e).__name__)
            self._failing = True
        with self._lock:
            self._catch_up = True
        if self._tables is not None:
            try:
                self._tables.close()
            except Exception:
                pass
            self._tables = None


# --- command line ------------------------------------------------------------------------
def local_tz() -> str:
    """This computer's IANA time zone, for grouping by local day; UTC if it can't tell."""
    if os.environ.get("TZ"):
        return os.environ["TZ"]
    try:
        return os.path.realpath("/etc/localtime").split("zoneinfo/", 1)[1]
    except (OSError, IndexError):
        return "UTC"


def where_text(objects, beat, now, stale_after_s) -> str:
    from .query import ago, clock
    lines = []
    if beat is None:
        lines.append("The camera has never reported.")
    elif beat["camera_ok"] and beat["last_frame"] and now - beat["last_frame"].timestamp() <= stale_after_s:
        lines.append("Camera live.")
    elif beat["last_frame"]:
        lines.append(f"Camera paused since {clock(beat['last_frame'].timestamp())}: "
                     "things may have moved since then.")
    else:
        lines.append("Camera paused.")
    names = {o["object"]: o["name"] for o in objects}
    for o in objects:
        seen = o["last_observed"]
        where = f"in the {names.get(o['container'], o['container'])}" if o["container"] else (
            f"on the {o['surface']}" if o["surface"] else "outside the named places")
        if seen is None:
            lines.append(f"  {o['name']:<12} not seen yet")
        elif o["status"] == "observed":
            lines.append(f"  {o['name']:<12} {where}, seen {ago(seen.timestamp(), now)}")
        elif o["status"] == "inferred":
            lines.append(f"  {o['name']:<12} probably {where} ({o['confidence']:.2f}), "
                         f"went in at {clock(seen.timestamp())}")
        else:
            lines.append(f"  {o['name']:<12} not sure: last seen {where} at {clock(seen.timestamp())}")
    return "\n".join(lines)


def change_text(e, names) -> str:
    from .query import clock
    the = lambda s: f"the {s}" if s else "outside the named places"
    n, k = e["name"], e["kind"]
    if k in ("appeared", "reappeared"):
        what = f"{n} {k} " + (f"on the {e['to_surface']}" if e["to_surface"] else "outside the named places")
    elif k == "moved":
        what = f"{n} moved from {the(e['from_surface'])} to {the(e['to_surface'])}"
    elif k == "inferred_contained":
        what = f"{n} went into the {names.get(e['container'], e['container'])} (inferred, {e['confidence']:.2f})"
    elif k == "disappeared":
        what = f"{n} disappeared from {the(e['from_surface'])}"
    else:
        what = f"{n} {k}"
    return f"{clock(e['ts'].timestamp()):>8}  {what}"


def week_text(rows) -> str:
    if not rows:
        return "Nothing happened in the last 7 days."
    out, day = [], None
    for r in rows:
        d = r["day"].astimezone().strftime("%a %b %d")
        if d != day:
            out.append(d)
            day = d
        bits = [f"moved {r['moved']}x" if r["moved"] else "", f"went out of sight {r['gone']}x" if r["gone"] else ""]
        out.append(f"  {r['name']:<12} " + (", ".join(b for b in bits if b) or f"{r['events']} event(s)"))
    return "\n".join(out)


def spots_text(rows) -> str:
    if not rows:
        return "Nothing went out of sight in the last 30 days."
    out, name = [], None
    for r in rows:
        if r["name"] != name:
            out.append(r["name"])
            name = r["name"]
        where = (f"into the {r['container']}" + (f" on the {r['place']}" if r["place"] else "")
                 if r["container"] else f"on the {r['place']}" if r["place"] else "outside the named places")
        out.append(f"  {r['times']:>3}x  {where}")
    return "\n".join(out)


# --- the web page's "This week" ------------------------------------------------------------
_asking = threading.Lock()  # one question to the database at a time; a hung one doesn't pile up more


def week(store, names=None, now=None, days=7, connect=None) -> dict:
    """The "This week" panel: for each thing, per day, how many times it went out of sight, and
    where it usually goes missing. From the database when the store has one and it answers within
    WEB_WAIT_S; otherwise (no database, an error, too slow) from events.jsonl on this laptop.

        {"source": "database" | "this laptop", "days": ["2026-10-19", ...],
         "items": [{"name", "per_day": [gone, ...], "moved": [moved, ...]}],
         "spots": [{"name", "place", "container", "times"}]}"""
    now = time.time() if now is None else now
    today = datetime.fromtimestamp(now).date()
    dates = [today - timedelta(days=days - 1 - i) for i in range(days)]
    d = getattr(store, "db", None)
    if isinstance(d, Database):
        rows = _ask_database(d.url, d.home, days, local_tz(),
                             connect or (lambda url: Tables.connect(url, timeout=WEB_CONNECT_S)))
        if rows is not None:
            per_day, spots = rows
            tz = ZoneInfo(local_tz())
            for r in per_day:
                r["day"] = r["day"].astimezone(tz).date()
            return week_shape("database", dates, per_day, spots)
    per_day, spots = local_week(store.events(), dates, names or {})
    return week_shape("this laptop", dates, per_day, spots)


def _ask_database(url, home, days, tz, connect):
    """(per_day rows, lost_spots rows), or None when the database can't say within WEB_WAIT_S."""
    if not _asking.acquire(blocking=False):
        return None  # still waiting on the last one
    out = {}

    def ask():
        try:
            tables = connect(url)
            try:
                out["rows"] = tables.per_day(home, days, tz), tables.lost_spots(home, days)
            finally:
                tables.close()
        except Exception as e:
            out["error"] = e
        finally:
            _asking.release()

    t = threading.Thread(target=ask, name="week", daemon=True)
    t.start()
    t.join(WEB_WAIT_S)
    if "rows" not in out:
        e = out.get("error")
        log.info("database: this week comes from this laptop (%s)",
                 (str(e).splitlines()[0] if str(e) else type(e).__name__) if e else "too slow")
    return out.get("rows")


def local_week(events, dates, names) -> tuple[list[dict], list[dict]]:
    """per_day and lost_spots rows, from events.jsonl instead of the database."""
    names = {**names, **{e.object: e.name for e in events}}
    first, per_day, spots = dates[0], {}, {}
    for e in events:
        day = datetime.fromtimestamp(e.ts).date()
        if day < first:
            continue
        r = per_day.setdefault((day, e.name), {"day": day, "name": e.name, "moved": 0, "gone": 0, "events": 0})
        r["events"] += 1
        r["moved"] += e.kind == "moved"
        if e.kind in GONE:
            r["gone"] += 1
            key = (e.name, e.from_surface, names.get(e.container, e.container) if e.container else None)
            s = spots.setdefault(key, {"object": e.object, "name": key[0], "place": key[1],
                                       "container": key[2], "times": 0, "last": 0})
            s["times"] += 1
            s["last"] = max(s["last"], e.ts)
    spots = sorted(spots.values(), key=lambda s: (s["name"], -s["times"], -s["last"]))
    return sorted(per_day.values(), key=lambda r: (r["day"], r["name"])), spots


def week_shape(source, dates, per_day, spots) -> dict:
    index = {d: i for i, d in enumerate(dates)}
    items = {}
    for r in per_day:
        if r["day"] not in index:
            continue
        it = items.setdefault(r["name"], {"name": r["name"], "per_day": [0] * len(dates), "moved": [0] * len(dates)})
        it["per_day"][index[r["day"]]] += int(r["gone"])
        it["moved"][index[r["day"]]] += int(r["moved"])
    return {"source": source, "days": [d.isoformat() for d in dates],
            "items": sorted(items.values(), key=lambda it: it["name"]),
            "spots": [{"name": s["name"], "place": s["place"], "container": s["container"], "times": int(s["times"])}
                      for s in spots]}


def main(argv=None):
    from .enroll import load_items
    from .query import parse_since
    from .store import Store
    ap = argparse.ArgumentParser(description="The memory's database: set it up, fill it, ask it.")
    ap.add_argument("command", choices=["setup", "sync", "where", "changes", "week", "spots"])
    ap.add_argument("since", nargs="*", help="changes: today (default), 10m, 2h, 14:30, 2:40pm")
    config.add_args(ap)
    a = ap.parse_args(argv)
    cfg = config.from_args(a)
    url = database_url()
    if not url:
        raise SystemExit("set DATABASE_URL (the service's connection string), or put it in .env")
    try:
        tables = Tables.connect(url)
    except Exception as e:
        raise SystemExit(f"can't reach the database: {e}")
    home, now = cfg.db_home, time.time()
    try:
        if a.command == "setup":
            print("recall_events:", "hypertable" if tables.setup() else "table (no TimescaleDB)")
        elif a.command == "sync":
            tables.setup()
            store = Store(cfg.data_dir)
            events, state, beat = store.events(), store.load_state(), store.heartbeat() or {}
            new = tables.write([event_row(home, t) for t in events],
                               [object_row(home, o) for o in state.values()],
                               (home, utc(beat.get("last_frame_ts")), bool(beat.get("camera_ok"))) if beat else None)
            if new:
                tables.refresh_daily(min(utc(t.ts) for t in events))
            print(f"sent {len(events)} event(s) from {store.root} ({new} new) and {len(state)} thing(s) "
                  f"as home {home!r}")
        elif a.command == "where":
            print(where_text(tables.objects(home), tables.heartbeat(home), now, cfg.stale_after_s))
        elif a.command == "changes":
            since = parse_since(" ".join(a.since) or "today", now)
            names = {oid: it.name for oid, it in load_items(cfg.items_path).items()}
            rows = tables.changes(home, utc(since))
            print("\n".join(change_text(e, names) for e in rows) or "Nothing changed.")
        elif a.command == "spots":
            print(spots_text(tables.lost_spots(home, 30)))
        else:
            print(week_text(tables.per_day(home, 7, local_tz())))
    finally:
        tables.close()


if __name__ == "__main__":
    main()
