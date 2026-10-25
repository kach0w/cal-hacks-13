"""Benchmark the memory's database on a real TimescaleDB: synthetic households, then how big the
events are before and after the columnstore, and how long the care partner's questions take from
the events themselves vs from recall_daily (the continuous aggregate).

    docker run -d --rm --name recall-tsdb-bench -e POSTGRES_PASSWORD=test \\
        -p 127.0.0.1:55442:5432 timescale/timescaledb:latest-pg17
    .venv/bin/python -m tools.db_bench --url postgresql://postgres:test@127.0.0.1:55442/postgres
    --homes 200 --days 30 --per-day 150     # the default: about 900k events
    --runs 20                               # timed runs per question (the median is reported)

Everything goes in its own schema (recall_bench, dropped and made again each run), made by the
same Tables.setup() live uses, so the tables, hypertable, columnstore settings and recall_daily
are the app's own. Don't point it at a service whose recall_bench schema you want to keep.
"""
import argparse
import random
import statistics
import time
from datetime import datetime, timedelta, timezone

from recall.db import Tables, local_tz

OBJECTS = {"keys": "keys", "wallet": "wallet", "glasses": "glasses", "phone": "phone",
           "remote": "TV remote", "bag": "blue bag"}
SURFACES = ["kitchen counter", "table", "sofa", "nightstand", "desk", "shelf"]
CONTAINERS = ["bag", "drawer", "coat"]
KINDS = {"moved": 35, "appeared": 15, "reappeared": 20, "disappeared": 20, "inferred_contained": 10}

# Per local day and kind of event, every home: what a dashboard over the whole service asks.
FLEET_RAW = """SELECT time_bucket('1 day', ts, %(tz)s) AS day, kind, count(*) FROM recall_events
    WHERE ts >= now() - make_interval(days => %(days)s) GROUP BY 1, 2"""
FLEET_DAILY = """SELECT day, kind, sum(n) FROM recall_daily
    WHERE day >= now() - make_interval(days => %(days)s) GROUP BY 1, 2"""


def events(homes, days, per_day, seed):
    """Rows for COPY, oldest day first. Each home has its own favourite spots, so where things go
    missing isn't uniform."""
    rnd = random.Random(seed)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    habits = {h: {o: rnd.sample(SURFACES, 3) for o in OBJECTS} for h in range(homes)}
    kinds, weights = list(KINDS), list(KINDS.values())
    for d in range(days):
        day0 = start + timedelta(days=d)
        for h in range(homes):
            home = f"home-{h:04d}"
            n = rnd.randint(per_day // 2, per_day * 3 // 2)
            for us in sorted(rnd.sample(range(86_400_000_000), n)):  # unique per home: the unique key
                ts = day0 + timedelta(microseconds=us)
                if ts > end:
                    break
                oid = rnd.choice(list(OBJECTS))
                kind = rnd.choices(kinds, weights)[0]
                frm = rnd.choices(habits[h][oid] + SURFACES, [6, 3, 1] + [1] * len(SURFACES))[0]
                to = None if kind in ("disappeared",) else rnd.choice(SURFACES)
                container = rnd.choice(CONTAINERS) if kind == "inferred_contained" else None
                xyz = [round(rnd.uniform(0, 5), 3), round(rnd.uniform(0, 4), 3), round(rnd.uniform(0, 1.2), 3)]
                yield (home, ts, oid, OBJECTS[oid], kind, frm, to, xyz, None if to is None else xyz,
                       "inferred" if container else "unknown" if kind == "disappeared" else "observed",
                       container, round(rnd.uniform(0.5, 0.9), 2) if container else 1.0)


def timed(fn, runs):
    """Median milliseconds of runs calls (after one untimed warm-up)."""
    fn(0)
    out = []
    for i in range(runs):
        t = time.perf_counter()
        fn(i)
        out.append((time.perf_counter() - t) * 1000)
    return statistics.median(out)


def size(conn):
    return conn.execute("SELECT hypertable_size('recall_events')").fetchone()[0]


def mb(b):
    return f"{b / 1e6:,.1f} MB"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Benchmark recall_events and recall_daily on TimescaleDB.")
    ap.add_argument("--url", required=True, help="a TimescaleDB to fill (e.g. a local Docker container)")
    ap.add_argument("--homes", type=int, default=200)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--per-day", type=int, default=150, help="events per home per day, on average")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tz", default=local_tz())
    a = ap.parse_args(argv)

    import psycopg
    with psycopg.connect(a.url, autocommit=True) as c:
        c.execute("DROP SCHEMA IF EXISTS recall_bench CASCADE")
        c.execute("CREATE SCHEMA recall_bench")
    conn = psycopg.connect(a.url, autocommit=True, options="-c search_path=recall_bench,public")
    tables = Tables(conn)
    version = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'").fetchone()
    pg = conn.execute("SHOW server_version").fetchone()[0]
    if not tables.setup(a.tz):
        raise SystemExit("recall_events isn't a hypertable: this needs TimescaleDB")
    assert tables.daily_in(a.tz)
    print(f"TimescaleDB {version[0]} on Postgres {pg}; {a.homes} homes x {a.days} days x ~{a.per_day} events/day")

    t = time.perf_counter()
    rows = 0
    with conn.cursor() as cur, cur.copy("COPY recall_events (home, ts, object, name, kind, from_surface, "
                                         "to_surface, from_xyz, to_xyz, status, container, confidence) "
                                         "FROM STDIN") as copy:
        for r in events(a.homes, a.days, a.per_day, a.seed):
            copy.write_row(r)
            rows += 1
    load_s = time.perf_counter() - t
    conn.execute("ANALYZE recall_events")

    t = time.perf_counter()
    conn.execute("CALL refresh_continuous_aggregate('recall_daily', NULL, NULL)")
    refresh_s = time.perf_counter() - t
    daily_rows = conn.execute("SELECT count(*) FROM recall_daily").fetchone()[0]
    chunks = conn.execute("SELECT count(*) FROM show_chunks('recall_events')").fetchone()[0]

    homes = [f"home-{h:04d}" for h in range(a.homes)]
    pick = lambda i: homes[(i * 7919) % len(homes)]  # a different home each run
    args = {"tz": a.tz, "days": a.days}

    def questions():
        return {
            "This week, one home, from the events": timed(lambda i: tables.per_day(pick(i), 7, a.tz, daily=False), a.runs),
            "This week, one home, from recall_daily": timed(lambda i: tables.per_day(pick(i), 7, a.tz, daily=True), a.runs),
            f"Every home, per day and kind, {a.days} days, from the events":
                timed(lambda i: conn.execute(FLEET_RAW, args).fetchall(), max(3, a.runs // 4)),
            f"Every home, per day and kind, {a.days} days, from recall_daily":
                timed(lambda i: conn.execute(FLEET_DAILY, args).fetchall(), max(3, a.runs // 4)),
            f"Where things go missing, one home, {a.days} days": timed(lambda i: tables.lost_spots(pick(i), a.days), a.runs),
        }

    before = size(conn)
    rowstore = questions()
    t = time.perf_counter()
    conn.execute("SELECT count(compress_chunk(c, if_not_compressed => true)) FROM show_chunks('recall_events') c")
    compress_s = time.perf_counter() - t
    conn.execute("ANALYZE recall_events")
    after = size(conn)
    columnstore = questions()
    example = tables.lost_spots(homes[0], a.days)[:3]

    print(f"\nLoaded {rows:,} events in {load_s:.1f} s ({rows / load_s:,.0f}/s, COPY) into {chunks} chunks")
    print(f"recall_daily: {daily_rows:,} rows ({rows / daily_rows:.0f}x fewer than events), "
          f"first full refresh {refresh_s:.1f} s")
    print(f"recall_events size: {mb(before)} before the columnstore, {mb(after)} after "
          f"({(1 - after / before) * 100:.1f}% smaller, {before / after:.1f}x); compress_chunk took {compress_s:.1f} s")
    print(f"\n| Question (median of {a.runs} runs; fleet-wide: {max(3, a.runs // 4)}) | Row store | Columnstore |")
    print("|---|---:|---:|")
    for q in rowstore:
        print(f"| {q} | {rowstore[q]:.1f} ms | {columnstore[q]:.1f} ms |")
    print(f"\nWhere things go missing in {homes[0]}, most often first:")
    for r in example:
        print(f"  {r['name']}: {r['times']}x on the {r['place']}" + (f", into the {r['container']}" if r["container"] else ""))
    conn.close()


if __name__ == "__main__":
    main()
