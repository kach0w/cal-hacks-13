# Final report: Cal Hacks 13.0 sponsor tracks for Recall

Written 5:45 AM PDT, Oct 25 2026. Submission deadline: **9:00 AM PDT today**.
Everything below is on the local branch `sponsor/integration` (head `76f2f18`). **Nothing has been pushed.**

## Executive summary

- **Selected:** one new primary build, **[Anthropic] Claude for Public Benefit**, and three supporting sponsor builds:
  **[MLH] Best Use of Tiger Data**, **[Cartesia] Best Use of Cartesia** and **[Moss] Best Use of Moss**. Recall also
  enters the Cal Hacks categories it fits with no extra build: People-First Social Impact, Most Technically Complex,
  Overall, Most Creative, Best Hardware and Hacker's Choice.
- **Built:** all four integrations are implemented, reviewed adversarially, merged and QA'd:
  - 265 tests pass and 9 skip. The skips are the tests that need real credentials.
  - The database tests also pass against real TimescaleDB 2.31 and Postgres 17 in Docker.
- **Readiness:** the code is ready to demo, but three things only a human can do stand between it and a strong
  submission:
  1. **Push** `sponsor/integration` to GitHub and merge it to `main` before 9:00. Judges read the repo, and `main` also
     has a guidance crash that this branch fixes.
  2. **Add API keys** so the sponsor features run live. None existed on the build machine, so every sponsor feature was
     verified only against fakes or local stand-ins (the database against real TimescaleDB in Docker).
  3. **Update the Devpost entry:** tick the prize tracks, add the Built With tags, and re-record or extend the video so
     each integration appears on screen.

## Track coverage

- 47 tracks were found and researched: 46 on the official Devpost page (https://cal-hacks-13-0.devpost.com/) plus
  "[MLH] Best Use of Gen AI", which appears only on the live site (https://hive.hackberkeley.org/prizes) and is
  unverified on Devpost.
- Each track is in [sponsor_tracks.md](sponsor_tracks.md) with its prize, judging criteria, required technology,
  eligibility, documentation, cost, effort, fit, confidence and source URL.
- Rankings are in [track_rankings.md](track_rankings.md) and [track_rankings.csv](track_rankings.csv). The weighted score
  out of 100 is the mean of three independent scorers (sponsor judge, engineering lead, strategist).

**Strongest tracks (weighted score):**

| Rank | Track | Score | Decision |
|---|---|---|---|
| 1 | Cal Hacks Best People-First Social Impact Hack | 88.2 | Enter (no build) |
| 2 | Cal Hacks Most Technically Complex Hack | 85.2 | Enter (no build) |
| 3 | [MLH] Best Use of Tiger Data | 82.2 | **Built** |
| 4 | Cal Hacks First / Second / Third Overall | 81.8 | Automatic |
| 7 | Cal Hacks Most Creative Hack | 80.8 | Enter (no build) |
| 8 | Cal Hacks Best Hardware Hack | 80.5 | Enter (no build) |
| 9 | [Anthropic] Claude for Public Benefit | 76.8 | **Built (primary)** |
| 10 | [Moss] Best Use of Moss | 73.5 | **Built** |
| 11 | [Cartesia] Best Use of Cartesia | 73.2 | **Built** |
| 12 | [Microsoft] Built for the Real World / Customer Impact | 72.8 | Framing-only opt-in |
| 14, 15 | [MLH] Commit Fellowship; [Meta] Bringing People Closer Together | 66.2, 64.2 | Framing-only opt-ins |

- **Why Claude is the primary build despite ranking 9th:** the tracks above it are Cal Hacks' own categories, which need
  no build and are entered anyway. Among new builds, Claude is the largest prize Recall can credibly win. Anthropic's four
  criteria are a named community, a real problem, that it works, and thoughtful use of Claude. Recall met the first three
  already; the build supplies the fourth.
- **Rejected:** 27 tracks, each with a reason in [track_rankings.md](track_rankings.md). Most would need a different
  product (finance, shopping, crypto, web scraping, agents platforms, space data) or a second LLM or backend alongside
  the ones we use.
- A high score is not a probability of winning.

## Implementation results

All branches are local. The scaffold `d1de0ad` gave each integration its own route registration and page placeholders,
so the four parallel builds merged with no conflicts.

### [Anthropic] Claude for Public Benefit: "Ask about the day"
- **What:** a card for the care partner that answers "Did the glasses go missing today?" and writes "Today's note".
  - Claude reads the memory only through five text-only tools: where_is, everything_now, changes, history and
    missing_counts. They wrap the existing Query code.
  - It answers only from what the tools return, cites times, keeps *seen* apart from *probably*, says when it doesn't
    know, and gives no medical advice.
  - Each answer lists what Claude looked at.
  - The person with dementia's spoken answers stay templated and local.
- **Technology:** Anthropic Messages API with tool use (anthropic SDK 2.3.0), model `claude-sonnet-5-5`.
- **Criteria:** all four of Anthropic's. For criterion 4, the failure modes and mitigations are written up in
  `docs/sponsors/claude.md`.
- **Branch:** `sponsor/claude`, commits `dfdd5d3` and `2faf434`.
- **Files:** `recall/care.py`, `tests/test_care.py`, `docs/sponsors/claude.md`, and the `[care]` placeholders in
  `recall/web.py`.
- **Tests:** 12 pass against a fake client, 1 live test skips without a key. The real SDK was also checked against a
  local fake endpoint.
- **Not verified:** the live Anthropic API (no key).
- **Demo:** put `ANTHROPIC_API_KEY` in `.env`, run `python -m recall.live --web`, then in "Ask about the day" press
  "Today's note". Ask "Did the glasses go missing today?", then "Where's the phone?" (it should say it has never seen it),
  then "Is her memory getting worse?" (it should decline and point to a doctor).
- **Limitations:**
  - No server-side refusal fallback.
  - Nothing checks Claude's wording against the facts beyond the prompt and the "Looked at" line.
  - The /care routes are open to anyone on the wifi.

### [MLH] Best Use of Tiger Data
- **What:**
  - **Earlier work (`bc45557`):** a background copy of every event, the current state and the heartbeat. It never blocks
    the camera loop, and after an outage it resends from the database's newest event, safe to repeat.
  - **Added:** a real-time continuous aggregate `recall_daily` with a refresh policy, which feeds a "This week" panel
    (per thing, per day, how often it went out of sight).
  - **Added:** a one-query "where do things usually go missing" (`python -m recall.db spots`).
  - **Added:** `tools/db_bench.py`.
- **Technology:** Tiger Data (Postgres + TimescaleDB): a hypertable, the columnstore, and the continuous aggregate.
- **Criteria:** "innovative, impactful, performance-driven". The benchmark ran on TimescaleDB 2.31 in Docker on a laptop,
  not Tiger Cloud:
  - 901,232 synthetic events across 200 homes and 30 days.
  - The columnstore made them 83% smaller.
  - A month-long summary across all homes took 29 ms from the aggregate versus 357 ms from raw events.
- **Branches:** `tiger-data` (`bc45557`) and `sponsor/tiger` (`9a5efc7`, `47dc44c`). The QA fix `67714d9` makes the
  aggregate recount late events.
- **Files:** `recall/db.py`, `tests/test_db.py`, `tools/db_bench.py`, `docs/sponsors/tiger.md`, and the `[week]`
  placeholders in `recall/web.py`.
- **Tests:** all database tests pass against real TimescaleDB and Postgres 17 in Docker. QA's end-to-end check showed
  /week fell back to the laptop within 2 s when the database was paused.
- **Not verified:** a Tiger Cloud service (no `DATABASE_URL`).
- **Demo:** set `DATABASE_URL` to a Tiger Cloud service and run live; "This week" shows "From the database". Then run
  `python -m recall.db week` and `spots` from another computer.
- **Limitation:** in a short demo the week grid has only today's column filled.

### [Cartesia] Best Use of Cartesia
- **What:** every answer and direction is spoken in a calm Sonic voice (`sonic-3.6`, voice "Jacqueline", emotion calm,
  speed 0.9), on the laptop (the band button and the terminal) and on the page.
  - Audio is cached; the direction phrases are made at startup so they play instantly.
  - It times out at 4 s, and the page waits at most 1.5 s.
  - It falls back to macOS `say` or the browser voice.
- **Branch:** `sponsor/cartesia`, commits `f6da29a` and `c8d04dc`.
- **Files:** `recall/tts.py`, `voice.speak()`, `/tts`, the page's `say()`, `tests/test_tts.py`,
  `docs/sponsors/cartesia.md`.
- **Tests:** 10 pass against a fake client, 1 live test skips. The real SDK was checked against a local server, and the
  page played a real WAV in headless Chrome.
- **Not verified:** the live Cartesia API, the voice ID and real latency (no key).
- **Demo:** set `CARTESIA_API_KEY`, run `python -m recall.tts "Your keys are on the table."`, then ask on the page and
  listen.

### [Moss] Best Use of Moss
- **What:**
  - When Moss, not the exact words, works out the item, the answer card says
    'Understood "the thing I open the front door with" as keys (Moss, N ms)'.
  - A second Moss index over the event history answers "when did the glasses go into the bag?".
- **Branch:** `sponsor/moss`, commits `ea62d73`, `5943000`, `82899f4` and `1183f37`.
- **Files:** `recall/moss_search.py`, `recall/ask.py`, `Web.ask`, `tests/test_moss_search.py`, `docs/sponsors/moss.md`.
- **Tests:** 22 pass, 1 live test skips.
- **Not verified:** the new parts have only run on the fake client. Item matching was tuned against live Moss before the
  hackathon, in commit `e761de6`.
- **Demo:** with Moss keys, ask "Where is the thing I lock the front door with?"

### Also on the branch
- The page redesign (`76c0ebd`).
- QA fixes:
  - `/guide` crashed on NumPy float32 (`e7ff34a`). **This bug is on `main` today**, so guidance never shows in the
    no-room camera mode.
  - Layout (`5b0b2f6`).
  - Honest README Privacy section (`273810f`).
  - README and DEVPOST cover the new integrations (`0b059af`).
  - Polish (`76f2f18`): plain timeline wording, paused-camera directions, the family card's wording and layout, and no
    sideways scroll at 360 px.
- Moss 1.21 is now actually installed in the venv; its 3 tests used to fail only because the package was missing.

## Budget report

- **Initial budget:** $200.
- **Actual spending:** **$0.** No paid API was called and nothing was bought.
- **Outstanding commitments:** none.
- **Remaining:** $200.
- **Projected live-key verification:** about $10 at most. Claude is a few cents per question (cap $5). Cartesia, Tiger
  Cloud and Moss free tiers are $0.
- **Not metered:** the agents' own model usage. See [budget.md](budget.md).

## Final recommendations

- **Emphasize in judging:** People-First Social Impact and Claude for Public Benefit, told as one story. A named community
  (people living with early dementia and their care partners), a real problem, a working system, and Claude used where it
  helps the family without ever speaking to the person with dementia.
- **Best secondary:** [MLH] Tiger Data. It is the most complete and best-verified integration, with real TimescaleDB
  numbers.
- **Three most important remaining tasks:**
  1. Push and merge to `main` before 9:00.
  2. Put the four keys in `.env`, run `pytest` (the 4 live tests then run) and fix anything they reveal.
  3. Re-record a video under 3 minutes showing each integration live, tick the prize tracks on Devpost, and add Built With
     tags: claude, anthropic, cartesia, moss, timescaledb.
- **Likely judge questions:**
  - *"Does Claude ever talk to the person with dementia?"* No. Their answers stay templated and local; Claude only serves
    the family page. Show the two answers side by side.
  - *"What if Claude is wrong?"* It reads only tool results, cites times, says when it doesn't know, and the board and
    timeline sit right next to it. Show "Looked at".
  - *"Why a time-series database?"* The questions are about time. Show the aggregate, `recall.db spots`, and the
    benchmark table in `docs/sponsors/tiger.md`.
  - *"What leaves the house?"* Only text. Camera frames, photos and the scan stay on the laptop (README Privacy).
- **Open risks:**
  - None of the four sponsor APIs has been called live.
  - The Devpost prize-track checkboxes could not be seen without login.
  - The Hive Anthropic entry still says "Coming soon!".
  - `/care` and `/tts` are unauthenticated on the LAN.
