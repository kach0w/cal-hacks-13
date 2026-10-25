# Tiger Data in Recall

Recall's memory of where things are is a time series: an append-only stream of events like
"the keys moved from the counter to the table at 2:40 PM". The questions a care partner asks are
about time too: what changed today, how often the glasses went missing this week, where they
usually disappear. Recall copies the memory to Postgres with TimescaleDB (a Tiger Cloud service)
and asks those questions in SQL. Code: `recall/db.py`. Tests: `tests/test_db.py`. Benchmark:
`tools/db_bench.py`.

## What is stored

| Table | What | Kind |
|---|---|---|
| `recall_events` | every event: when, which household, which thing, what happened (appeared, moved, disappeared, went into a container), from where to where, how sure | hypertable on `ts` |
| `recall_daily` | per local day, household, thing and kind of event: how many | continuous aggregate over `recall_events` |
| `recall_objects` | one row per thing: where it is now, seen or inferred, how sure, last seen | plain table, upserted |
| `recall_heartbeat` | per household: the camera's last frame, so anyone can tell if it's live | plain table, upserted |

Photos never leave the laptop: the evidence picture's path isn't sent. Live writes its own files
(`data/`) first and a background thread copies them to the database, so the camera loop never
waits on the network. After an outage it resends from the newest event the database has. A unique
key on (home, object, kind, ts) makes resends safe.

## TimescaleDB choices

- **Hypertable**, partitioned on `ts` (7-day chunks), made in `CREATE TABLE ... WITH
  (tsdb.hypertable, ...)` on TimescaleDB 2.20+, `create_hypertable` on older versions.
- **Columnstore** (compression), `segmentby = 'home, object'`, `orderby = 'ts DESC, kind'`.
  Every question is per household and usually per thing, so a segment is exactly what a query
  reads. Chunks move to the columnstore after 7 days (the default policy for `tsdb.hypertable`).
- **Continuous aggregate `recall_daily`**, made by `Tables.setup()`:

  ```sql
  CREATE MATERIALIZED VIEW IF NOT EXISTS recall_daily
      WITH (timescaledb.continuous, timescaledb.materialized_only = false) AS
      SELECT time_bucket('1 day', ts, 'America/Los_Angeles') AS day, home, object, name, kind, count(*) AS n
      FROM recall_events GROUP BY day, home, object, name, kind WITH NO DATA;
  SELECT add_continuous_aggregate_policy('recall_daily', start_offset => INTERVAL '35 days',
      end_offset => INTERVAL '1 hour', schedule_interval => INTERVAL '10 minutes', if_not_exists => true);
  ```

  - Days are local days: the bucket uses the time zone of the computer that ran setup (the home's
    laptop). A summary asked in another time zone is computed from the events instead, because
    days in different zones don't add up.
  - Real-time (`materialized_only = false`): what the policy hasn't materialized yet (today) is
    read from `recall_events`, so the panel is never behind the camera.
  - Events that arrive late (live catching up after the wifi drops, `recall.db sync` of an old
    memory) land in days already materialized. The writer then calls
    `refresh_continuous_aggregate` from the oldest late event, so those days are recounted right
    away instead of on the policy's next run.
- **Plain Postgres still works.** No TimescaleDB: `recall_events` is a normal table, there is no
  `recall_daily`, and every query falls back to `date_trunc` over the events.

## The queries

- **This week** (`Tables.per_day`, CLI `python -m recall.db week`, web `/week`): per local day and
  thing, how many times it moved and went out of sight. Reads `recall_daily` when it's there.
- **Where do things usually go missing?** (`Tables.lost_spots`, CLI `python -m recall.db spots`):
  one query. Per thing, the place it was last seen on and the container it went into, counted over
  the last N days, most common first, with the container's name joined from `recall_objects`.

  ```sql
  SELECT e.object, e.name, e.from_surface AS place, coalesce(c.name, e.container) AS container,
         count(*) AS times, max(e.ts) AS last
  FROM recall_events e
  LEFT JOIN recall_objects c ON c.home = e.home AND c.object = e.container
  WHERE e.home = $1 AND e.kind IN ('disappeared', 'inferred_contained')
    AND e.ts >= now() - make_interval(days => $2)
  GROUP BY e.object, e.name, e.from_surface, coalesce(c.name, e.container)
  ORDER BY e.name, times DESC, last DESC;
  ```

## On the care partner's page

The right column has a **This week** panel under "What moved": a grid with a row per thing and a
column per day for the last 7 days. Each cell is how many times that thing went out of sight that
day, tinted by count. Below the grid, a line per thing: "Glasses: usually go missing into the blue
bag on the table (7 times), on the sofa (4 times)". A caption says where the numbers came from,
**From the database** or **From this laptop**. The panel refreshes every 30 seconds.

The page asks `/week`. When live has a database, the web server opens a short connection
(2 s connect timeout) on a separate thread and waits at most 3 s. If the database is down, slow
or has nothing, the panel shows the same numbers computed from `data/events.jsonl` on the laptop.
Only one database request runs at a time, so a hung connection can't pile up more.

## Benchmark (measured on TimescaleDB in Docker, not Tiger Cloud)

Command, from the repo root (Docker Desktop on an Apple Silicon (arm64) Mac, Python client on the
same machine):

```
docker run -d --rm --name recall-tsdb-bench -e POSTGRES_PASSWORD=test \
    -p 127.0.0.1:55442:5432 timescale/timescaledb:latest-pg17
.venv/bin/python -m tools.db_bench --url postgresql://postgres:test@127.0.0.1:55442/postgres
docker stop recall-tsdb-bench
```

Defaults: 200 households x 30 days x about 150 events per household per day (random, seeded).
The tables are the app's own, made by the same `Tables.setup()` live uses, in a separate schema.
Output of the run on 2026-10-25 (TimescaleDB 2.31.0 on Postgres 17.11):

```
Loaded 901,232 events in 7.5 s (120,772/s, COPY) into 5 chunks
recall_daily: 177,937 rows (5x fewer than events), first full refresh 1.8 s
recall_events size: 333.4 MB before the columnstore, 56.1 MB after (83.2% smaller, 5.9x); compress_chunk took 4.0 s
```

| Question (median of 20 runs; fleet-wide: 5) | Row store | Columnstore |
|---|---:|---:|
| This week, one home, from the events | 2.1 ms | 2.3 ms |
| This week, one home, from recall_daily | 0.7 ms | 1.1 ms |
| Every home, per day and kind, 30 days, from the events | 356.9 ms | 312.9 ms |
| Every home, per day and kind, 30 days, from recall_daily | 29.2 ms | 37.6 ms |
| Where things go missing, one home, 30 days | 3.0 ms | 1.5 ms |

What this shows:

- The columnstore made the events 83% smaller (5.9x), counting indexes. That's short of the 90%
  Tiger quotes; the synthetic positions are random floats, which compress worse than a real room
  where things sit at the same few spots. We didn't measure real data at this scale.
- A dashboard over every household for a month is about 10x faster from `recall_daily`
  (29 ms vs 357 ms). For one household's week, everything is a few milliseconds either way:
  the per-household index and segments already make it cheap, and the aggregate mostly saves work
  as history grows.
- "Where do things go missing" is one query of 1.5 to 3 ms over a month of one household's events.

Times are wall clock from the Python client to a local container, warm cache, and include the
round trip. Numbers on Tiger Cloud will differ (network latency, different hardware).

## Setup

1. Create a service in Tiger Console (free trial; we didn't use one for this work) and copy its
   connection string.
2. Put it in `.env` (gitignored) or the environment:

   ```
   DATABASE_URL=postgres://tsdbadmin:<password>@<host>:<port>/tsdb?sslmode=require
   ```

3. Run live as usual. On first connect it makes the tables, the hypertable and `recall_daily`.
   `python -m recall.db setup` does the same by hand.

Without `DATABASE_URL`, Recall runs exactly as before: the panel says "From this laptop".

## Demo

1. `python -m recall.live --web` with `DATABASE_URL` set; open the printed link.
2. Move the glasses into the bag, take the keys off the counter. "What moved" shows the events;
   "This week" shows today's counts and where each thing went missing, captioned "From the
   database".
3. From any other computer with the repo: `python -m recall.db week` and
   `python -m recall.db spots` print the same numbers from the service.
4. Unplug the wifi: the panel switches to "From this laptop" within a few seconds and keeps the
   same numbers; when the wifi is back live catches up and the caption returns to "From the
   database".

## What was verified, and against what

| | Verified how |
|---|---|
| Hypertable, columnstore, `recall_daily` creation and policy | Real TimescaleDB 2.31.0 (Docker, `timescale/timescaledb:latest-pg17`): `RECALL_TEST_DATABASE_URL=postgresql://postgres:test@127.0.0.1:55443/postgres pytest tests/test_db.py` |
| `per_day` from `recall_daily` equals `per_day` from the events, real-time and after a refresh, including late events | Same real TimescaleDB tests |
| `lost_spots` (container names joined) | Same real TimescaleDB tests |
| `/week` from the database through live's own `Database` | Same real TimescaleDB tests |
| Falling back to the laptop on errors and hangs, one request at a time, `/week` over HTTP | In-memory fakes (always run) |
| Compression and query timings | `tools/db_bench.py` on Docker TimescaleDB, numbers above |
| Plain Postgres fallback (no `recall_daily`) | The same real-database tests against plain Postgres 17 (Docker `postgres:17`) |
| Tiger Cloud | **Not yet verified.** No Tiger Cloud credentials were available on this machine. The SQL is standard TimescaleDB 2.20+ that Tiger Cloud runs, but none of it has been run against a Tiger Cloud service. |
