# Implementation plan: shortlisted builds

Scope: the four builds from `track_rankings.md`.
- PRIMARY: Claude.
- SECONDARY: Tiger Data, Cartesia, Moss.

The no-build Cal Hacks categories and the framing-only opt-ins need only pitch, write-up and demo reliability (see the end of this file).

**Credentials: none are available on the build machine.** ANTHROPIC_API_KEY, CARTESIA_API_KEY, DATABASE_URL (Tiger Cloud) and MOSS_PROJECT_ID / MOSS_PROJECT_KEY are all missing. So every build:
- is developed and tested against fakes. Tiger uses a local Docker TimescaleDB.
- must do nothing (or fall back to today's behavior) when its key is missing.
- needs a live check with real keys on the demo laptop. Until that check, the feature is unproven against the real service. Mark it as such in the write-up until it has run live.

## Shared scaffold (commit d1de0ad, base of every sponsor/* branch)

`recall/web.py` has a route registry plus a placeholder for each integration. Each agent edits only its own placeholders, so the branches merge without conflicts.

| Where in recall/web.py | Placeholders | What goes there |
|---|---|---|
| `Web.__init__` | `self.routes = {}`, then `# [care] routes`, `# [week] routes`, `# [tts] routes`, `# [moss] routes` | `self.routes["/api/<x>"] = fn`. `fn(params)` gets the `parse_qs` dict and returns a dict (sent as JSON) or `(body_bytes, content_type)` |
| `do_GET` | `elif url.path in web.routes:` | Already wired. Don't touch |
| `PAGE` CSS | `/* [care] styles */`, `/* [week] styles */`, `/* [moss] styles */` | CSS for each panel |
| `PAGE` HTML | `<!-- [care] panel -->` (before the directions card), `<!-- [week] panel -->` (after "What moved") | Each integration's card |
| `PAGE` JS | `// [care] script`, `// [week] script`, `// [moss] script` | Fetch and render code. Cartesia has no JS placeholder; it changes the existing `say()` in place, and is the only build that edits shared lines |

`requirements.txt` already lists `anthropic>=2.3` (care.py) and `cartesia>=4.2` (tts.py). Both are imported lazily, only when their key is set.

**Merge order** into `sponsor/integration`:
1. tiger
2. claude (uses tiger's queries if present)
3. moss
4. cartesia (it edits `say()`)

Run `pytest` after each merge. Then integration goes to `main` before 9:00.

---

## 1. Claude care-partner assistant: [Anthropic] Claude for Public Benefit (PRIMARY, essential)

| | |
|---|---|
| User-facing feature | An "Ask about today" box on the care-partner page. The care partner asks open questions that the templates can't handle, e.g. "Did Mom take her keys and wallet when she went out?" or "What has she misplaced most this week, and where did it turn up?". The answer is short and cites times. Plus an optional end-of-day care note |
| Sponsor tech role | Claude runs a tool-use loop over Recall's own memory functions: `Query` (where_is, history, changes since t, state at t) and, when Tiger is merged, the week/changes and lost-spots queries. Claude decides which tools to call and writes the answer **only from tool results** |
| Criteria addressed | Anthropic: (1) named community: care partners of a parent with early Alzheimer's; (2) a real problem; (3) it works; (4) thoughtful use plus known failure modes. Overall: Application, Technical Complexity |
| Failure modes (state on Devpost and in the pitch) | It never invents a location; when the tools return nothing it says it doesn't know. Only text events are sent: no frames, no audio. The patient's spoken path stays templated and local. No medical advice. On API failure or timeout it gives the deterministic answer instead |
| Files | `recall/care.py` (new: the tool definitions, the loop, a timeout, the fallback); `tests/test_care.py` (new: a fake Anthropic client); `recall/web.py` placeholders `[care] routes/styles/panel/script` (`/api/care?q=`) |
| Dependencies / credentials | `anthropic` (in requirements.txt; install it into `.venv`). `ANTHROPIC_API_KEY`: **not available**. Get event credits (Slack #0-ask-directors, the Anthropic table) or approved prepaid usage. Model: research suggests a Haiku-class model for cost and latency |
| Time | 2-3 h (about 1 h for a minimal version) |
| Acceptance tests | (a) A fake client asks for `where_is("keys")`, and the answer contains the tool's place and time. (b) The tool returns nothing, so the answer says it doesn't know, with no place named. (c) No key, the client raises, or it times out: the endpoint returns the templated `ask.handle` answer and marks itself as the fallback. (d) No image or audio is ever passed to the client. (e) `/api/care` returns JSON through the route registry. (f) Existing `pytest` stays green |
| Demo sequence | Move the keys and wallet, then "leave". On the page, ask "Did she take her keys and wallet?". Claude's answer cites times, and the page shows which tools it used. Then ask about something never seen: it says it doesn't know |
| Fallback | No key or no network: the box answers with the deterministic template, so the demo still works. **Plan B if no key by about 5:15:** the same tool layer behind Gemini (free key), entered as [MLH] Gemini instead. Never ship both |
| Cost | $0 with credits. Otherwise a few cents per question; cap $5 (see budget.md) |
| Stretch | The end-of-day care note. An optional calm one-sentence rewrite of the patient answer, with a strict length limit and timeout and the template as the hard fallback |

## 2. Tiger Data: continuous aggregate, "This week" panel, lost-spots query, benchmark ([MLH] Best Use of Tiger Data, SECONDARY)

| | |
|---|---|
| User-facing feature | A "This week" panel on the care-partner page: moves and missing episodes per item per day (heatmap or sparkline), plus where each item usually turns up after going missing ("lost spots") |
| Sponsor tech role | Built on the tiger-data branch (hypertable of events, relational object state, heartbeat): (1) a continuous aggregate (hourly or daily per home/object, with counts of moves and missing episodes), `materialized_only=false` (real-time aggregates have been off by default since 2.13, per Tiger docs), and a refresh policy; (2) `week` reads the continuous aggregate instead of a query-time `time_bucket`; (3) a lost-spots query (where items reappeared after being missing); (4) a benchmark that backfills synthetic events under a separate home id, compresses chunks, and prints compression stats and raw-vs-aggregate timings |
| Criteria addressed | The track text: "performance-driven", "Instant Dashboards ... Continuous Aggregates", "90%+ Data Compression". Also Technical Complexity, and the Microsoft deployment story |
| Files | `recall/db.py` (the aggregate DDL in `setup()`, week/lost-spots queries); `tools/db_bench.py` (new); `tests/test_db.py`; `recall/web.py` placeholders `[week] routes/styles/panel/script` (`/api/week`) |
| Dependencies / credentials | `psycopg[binary]`. `DATABASE_URL`: **Tiger Cloud is not available on the build machine**. Develop against **local Docker TimescaleDB**. The live check needs the team's Tiger Cloud service (free service or the $1000 no-card trial) |
| Time | 15 min to merge (already done on sponsor/integration) + 1.5-2 h |
| Acceptance tests | (a) `setup()` is idempotent and creates the aggregate on TimescaleDB. (b) Inserted events show up in the week query immediately (real-time). (c) The lost-spots query ranks the place where an item reappeared most. (d) `/api/week` returns JSON; without DATABASE_URL it returns an empty or "not connected" result and the page hides the panel. (e) `tools/db_bench.py` against Docker prints before/after size and both timings. (f) Writes never block the live loop |
| Demo sequence | Show the "This week" panel with the glasses' row, point at "usually turns up: blue bag", then cut to the benchmark printout (x% smaller, y ms vs z ms). Name Tiger Data on screen |
| Fallback | DB unreachable at the venue: the panel hides and the core demo is unaffected. The video uses recorded numbers. If a free service won't run the policy jobs, call refresh and compress manually |
| Cost | $0 |
| Stretch | A multi-home backfill story ("many households, one care partner"). Leave out retention and tiering |

## 3. Cartesia voice ([Cartesia] Best Use of Cartesia, SECONDARY)

| | |
|---|---|
| User-facing feature | Every spoken answer and direction (band button, mic, typed ask, page directions) comes in a calm, slow, warm voice instead of macOS `say` and browser speechSynthesis |
| Sponsor tech role | Cartesia Sonic TTS (model "sonic-3.6" per research) via the Python SDK. The laptop plays the audio; the page fetches WAV from `/api/tts` |
| Criteria addressed | Cartesia (criteria unpublished; voice quality is the obvious target). Overall Intentionality (dementia-friendly speech). People-First; Hacker's Choice |
| Files | `recall/tts.py` (new: synthesize with timeout and cache, play); `recall/voice.py` `speak()` (try tts first, then `say`); `recall/web.py` `[tts] routes` (`/api/tts?text=`), plus the existing JS `say()` (play `/api/tts` audio, fall back to speechSynthesis); tests with a fake client |
| Dependencies / credentials | `cartesia` (in requirements.txt). `CARTESIA_API_KEY`: **not available**; free signup at play.cartesia.ai. A voice id from the library |
| Time | 1-2 h |
| Acceptance tests | (a) No key: `speak()` calls `say` exactly as before. (b) A fake client returns audio, which is played and cached by text. (c) Client error or timeout: falls back to `say` within the timeout. (d) `/api/tts` returns `audio/wav` bytes through the registry. (e) Page `say()` keeps its "answer is spoken whole" behavior (the `answering` guard) for both paths |
| Demo sequence | Press the band button, ask "where are my glasses?", and the answer plays in the Cartesia voice. Then play the same line through `say` for contrast (in the video). Name Cartesia on screen |
| Fallback | `say` and speechSynthesis are automatic. Pre-cache the scripted demo phrases |
| Cost | $0 on the free tier (20K credits/month, about 27 min of TTS). **No voice cloning**: it needs the $5 Pro plan and explicit approval |
| Stretch | Streaming over WebSocket for lower latency. Show latency on the page |

## 4. Moss visibility and optional event search ([Moss] Best Use of Moss, SECONDARY)

| | |
|---|---|
| User-facing feature | Next to each answer the page shows "matched by Moss in N ms, on-device" when Moss resolved the wording (e.g. "the thing I open the door with" → keys). Optional: "what did I put in the bag after lunch" is answered through Moss over memory events |
| Sponsor tech role | Existing `recall/moss_search.py` (session index of item names and aliases, 1 s timeout, the fallback after exact-word match), now with per-query latency recorded and surfaced. Optional: add memory transitions to the session index with metadata {object, container, ts} and filter by time and container |
| Criteria addressed | Moss's own pitch (low-latency on-device search for voice; criteria unpublished). Intentionality (vague speech from a person with dementia) |
| Files | `recall/moss_search.py` (time each search, return the latency); `recall/ask.py` (carry "matched by Moss" and the latency in the answer); `recall/web.py` `ask()` response + `[moss] routes/styles/script`; `tests/test_moss_search.py` |
| Dependencies / credentials | `moss` (already used). `MOSS_PROJECT_ID` / `MOSS_PROJECT_KEY`: **not available on the build machine** (they are in the team's `.env` on the demo laptop). Check session-minute usage in portal.usemoss.dev: the free plan has 60/month |
| Time | 1-1.5 h (+1 h for optional event search) |
| Acceptance tests | (a) A fake Moss client: a vague phrase resolves to the item, and the response includes `moss_ms`. (b) An exact-word match does not claim Moss. (c) No credentials: behaves exactly as today. (d) Timeout: the answer still comes back, with no Moss badge |
| Demo sequence | Ask "where's the thing I open the front door with?". The page shows the keys answer plus the "Moss, N ms, on-device" badge. Name Moss on screen |
| Fallback | Exact-word matching (already in place). If the session minutes are spent, use `load_index` local queries (unmetered) or pay-as-you-go (cents, needs approval) |
| Cost | $0 |
| Stretch | Memory-event search with metadata filters; hybrid search for garbled transcripts. **Not planned:** browser-side type-ahead (it needs a custom authenticator) |

Also add "moss" to Devpost Built With now, and "claude/anthropic" and "cartesia" once those ship.

---

## Essential vs stretch

| Essential | Stretch |
|---|---|
| Merge tiger-data + sponsor/ui to main before 9:00 | Claude end-of-day care note, calm rewrite |
| Claude `/api/care` with tool use, fallback and tests | Tiger multi-home backfill |
| Tiger continuous aggregate + week panel + lost-spots + bench numbers | Cartesia streaming |
| Cartesia `speak()` / `say()` with fallback | Moss event search, hybrid search |
| Moss latency badge + Built With | GoDaddy domain (non-engineer, 20 min) |
| Re-cut the video showing and naming every integration; tick every opt-in | Gen AI tick if selectable |

## No-build work (Cal Hacks categories and framing opt-ins)

- **Pitch.** Open with the person and the care partner (People-First). Contrast with AirTag/Tile (Creative). Keep an architecture diagram ready and show live ray-cast-to-3D (Technically Complex).
- **Hardware.** One person on band reliability: BLE reconnect, a charged battery, servo calibration. Keep a backup clip ready (Hardware).
- **Write-up.** Add a "How a family would deploy this" section (Microsoft) and a "closer together" framing (Meta). Commit Fellowship: the form, only if the team wants it. Do not tick Beginner, Solo or Planet-First.
- **Gallery.** Keep the demo running during the 1-2 PM gallery (Hacker's Choice).

## Timeline (PDT, Oct 25)

| Time | What |
|---|---|
| now to 6:35 | Builds in parallel in their worktrees (claude, tiger, cartesia, moss). Key and credit hunt for Claude by about 5:15, or switch to plan B |
| 6:35 to ~7:45 | Integrate into sponsor/integration in the merge order above, `pytest` after each merge, QA on the demo laptop with real keys, merge to main. **Cut any integration that isn't demo-ready** |
| ~7:45 to 8:45 | Devpost write-up (failure modes, privacy, deployment section, Built With), opt-in checkboxes, Commit form, re-record or re-cut the video showing and naming each integration |
| 8:50 | Final check: main has everything; the video plays; every intended prize is ticked; Beginner/Solo/Planet are not ticked; the repo link and image are present |
| **9:00** | **Deadline. No commits after 9:00** |
