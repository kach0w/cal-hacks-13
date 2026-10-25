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
    --set DB_HOME=NAME                      # which household (default "home"): one service, many rooms

data/ stays the memory's own copy: live writes its files first and never waits on the network
(store.Store hands each write to Database, which returns at once). A writer thread sends what
changed, at most once per FLUSH_EVERY_S. When the connection drops it backs off and, once back,
resends events.jsonl from the newest event the database has; the unique key on
(home, object, kind, ts) makes every send safe to repeat. Photos never leave the laptop:
photo_path isn't sent.

Events go in a TimescaleDB hypertable on ts, segmented by home and object (moved to the
columnstore after a chunk, 7 days); on plain Postgres (tests, a local server) they're an ordinary
table and everything else works the same.
"""
import argparse
import logging
import os
import threading
import time
from datetime import datetime, timezone

from . import config
from .moss_search import ENV_FILE, read_env

log = logging.getLogger("recall")

FLUSH_EVERY_S = 1.0    # live saves the state on every look; the database needn't hear that often
RETRY_MIN_S, RETRY_MAX_S = 1.0, 60.0  # backoff while the database can't be reached
CONNECT_TIMEOUT_S = 10
CLOSE_WAIT_S = 5.0     # on stop, how long the last send may take

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
    def connect(cls, url) -> "Tables":
        import psycopg
        # Keepalives: a link that died quietly (the wifi went) is noticed in ~20 s, not minutes.
        return cls(psycopg.connect(url, autocommit=True, connect_timeout=CONNECT_TIMEOUT_S,
                                   application_name="recall", keepalives=1, keepalives_idle=5,
                                   keepalives_interval=5, keepalives_count=3))

    def close(self):
        self.conn.close()

    def setup(self) -> bool:
        """Makes the tables that are missing. True when recall_events is a hypertable."""
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
        return self.is_hypertable()

    def is_hypertable(self) -> bool:
        try:
            return self.conn.execute("SELECT 1 FROM timescaledb_information.hypertables "
                                     "WHERE hypertable_name = 'recall_events'").fetchone() is not None
        except Exception:  # no TimescaleDB
            return False

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

    def per_day(self, home, days: int, tz: str) -> list[dict]:
        """Per local day and thing: moves, times it went out of sight, and all events."""
        bucket = ("time_bucket('1 day', ts, %(tz)s)" if self.is_hypertable()
                  else "date_trunc('day', ts, %(tz)s)")
        cur = self.conn.execute(
            f"""SELECT {bucket} AS day, name,
                    count(*) FILTER (WHERE kind = 'moved') AS moved,
                    count(*) FILTER (WHERE kind IN ('disappeared', 'inferred_contained')) AS gone,
                    count(*) AS events
                FROM recall_events
                WHERE home = %(home)s AND ts >= now() - make_interval(days => %(days)s)
                GROUP BY day, name ORDER BY day, name""", {"home": home, "days": days, "tz": tz})
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


def main(argv=None):
    from .enroll import load_items
    from .query import parse_since
    from .store import Store
    ap = argparse.ArgumentParser(description="The memory's database: set it up, fill it, ask it.")
    ap.add_argument("command", choices=["setup", "sync", "where", "changes", "week"])
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
            print(f"sent {len(events)} event(s) from {store.root} ({new} new) and {len(state)} thing(s) "
                  f"as home {home!r}")
        elif a.command == "where":
            print(where_text(tables.objects(home), tables.heartbeat(home), now, cfg.stale_after_s))
        elif a.command == "changes":
            since = parse_since(" ".join(a.since) or "today", now)
            names = {oid: it.name for oid, it in load_items(cfg.items_path).items()}
            rows = tables.changes(home, utc(since))
            print("\n".join(change_text(e, names) for e in rows) or "Nothing changed.")
        else:
            print(week_text(tables.per_day(home, 7, local_tz())))
    finally:
        tables.close()


if __name__ == "__main__":
    main()
