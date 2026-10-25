# Moss in Recall

[Moss](https://usemoss.dev) is the semantic search that lets a person ask for something in
their own words. People living with dementia often lose the word before they lose the thing:
"the thing I open the front door with", "my reading things", "what I change the channel with".
Moss turns that into *keys*, *glasses*, *remote*, so Recall can answer and point the band.

Code: `recall/moss_search.py` (the indexes), `recall/ask.py` `find_item_info()` (when Moss is
asked), `recall/web.py` (`Web.ask`, `/moss`, `/moss/events`, the line under the answer).

## What it does

**1. Which item a question means.** At start-up Recall builds a Moss session index
(`recall-items`, model `moss-minilm`) from `items.json`: one document per way of naming each
item: its name, its detector prompt ("keys on a red lanyard"), each alias, and an `about`
line the family writes ("what you lock and unlock the front door and start the car with").
A question that doesn't name an item exactly is a Moss query; the best hit counts if it scores
at least `MIN_SCORE` (0.4).

**2. Showing its work.** When Moss (not the exact words) found the item, the answer card says
so in one quiet line, so a care partner can tell when it heard wrong:

> Understood "the thing I open the front door with" as keys (Moss, 12 ms)

`/ask` and `/listen` (the "Ask out loud" button and the band's button) carry it as JSON:

    {"answer": "Your keys are on the table — seen 3 minutes ago.", "status": "observed",
     "understood": {"item": "keys", "by": "moss", "ms": 12.3, "score": 0.71,
                    "name": "keys", "words": "the thing I open the front door with"}}

Exact matches ("where are my keys") carry nothing extra. `ms` is the Moss query alone (the
time the person waited on Moss), `score` is Moss's similarity for the best document.
`MossSearch.match_info(text) -> (item, score, ms) | None` gives the same from Python;
`match(text)` is unchanged for the other callers (voice REPL, `python -m recall.ask`).

**3. Searching what happened.** A second session index (`recall-events`) holds one document per
event in the memory's log, in the words the page's "What moved" list uses ("glasses probably
went into the blue bag"). New events are added when a history search runs: each event's id is
its time, object and kind (the same unique key the Tiger Data copy uses), so re-reading the log
adds only what's new and events from a memory that was started over are ignored.

- `GET /moss/events?q=when did the glasses go into the bag` returns the best matches with
  their times: `{"on": true, "ms": 2.1, "matches": [{"t": ..., "time": "5:01 AM",
  "item": "glasses", "text": "glasses probably went into the blue bag", "score": 0.57}]}`
- A question starting with "when" that is about an item (by exact words or by Moss) also gets
  the top match about that item (or something going into it) on the answer card:
  *From the history: 5:01 AM, glasses probably went into the blue bag (Moss)*.
  Matches about unrelated items are left off, so "when is my daughter coming?" never shows an
  unrelated moment from the log.

`GET /moss` says whether Moss is on: `{"on": true, "ready": true, "model": "moss-minilm",
"min_score": 0.4}` or `{"on": false}`.

## Why exact words go first

`ask.match_item` runs before Moss on every question. It is instant, needs nothing, and can't be
wrong about "where are my keys". Moss only sees the questions exact words can't place, so a
Moss outage or a low score can never make a plain question worse. The same holds for history
commands ("history keys") and for the band's button, which goes through the same `Web.ask`.

## Speed and never hanging

- The session indexes live in this process; Moss documents session queries as in-memory,
  about 1 to 10 ms. The Moss SDK is async; Recall runs it on its own event loop on a daemon
  thread, so the threaded web server calls it as a plain function.
- Every item query has a 1 s timeout (`TIMEOUT_S`): past that the answer is
  "I don't have a record of that", never a spinner. Only the very first question waits (up to
  10 s) for the index to finish building.
- The handler and the page both look up the item for one question; the last question's hit is
  remembered, so Moss is queried once per question.
- A history search waits at most 3 s (`EVENTS_TIMEOUT_S`); the first one also embeds the whole
  log, and if that takes longer the indexing finishes in the background and the next search
  uses it. The live camera loop never calls Moss.
- No credentials, no `moss` package, rejected credentials or a failed query: Moss reports itself
  off (`/moss` -> `{"on": false}`), no `understood` or `events` fields appear, and asking works
  exactly as without Moss.

## Privacy

Both indexes are Moss **session** indexes: documents are embedded and searched locally on the
laptop, and Recall never calls `push_index`, so item names and the event log are not uploaded
as an index. The SDK does need the project credentials (it checks them, and reports usage in
the background). Photos are never indexed; only the words of the log are. Questions themselves
are embedded locally.

## Setup

1. Make a project at portal.usemoss.dev.
2. Put the credentials in the environment or in the repo's gitignored `.env`:

       MOSS_PROJECT_ID=...
       MOSS_PROJECT_KEY=...
       # optional: MOSS_MIN_SCORE=0.4

3. `uv pip install moss` (already in `requirements.txt`). Then run Recall as usual
   (`python -m recall.live --web`), or check from the terminal:

       .venv/bin/python -m recall.moss_search "the thing I open the front door with"

Write an `about` line for each item in `items.json` in the words the person actually uses.

## Demo questions

With the room running, type or say (suggested phrasings; the exact Moss scores for these
were not re-checked live today):

- "Where is the thing I lock the front door with?" -> keys, with the "Understood ... as keys
  (Moss, N ms)" line.
- "Where are my reading things?" -> glasses.
- "Where's the thing I change the channel with?" -> remote. (A question with "change" and no
  "where" is read as "what changed?", so keep the "where".)
- "Where are my keys?" -> answered by exact words, no Moss line.
- "When did the glasses go into the bag?" -> the answer, plus "From the history: <time>,
  glasses probably went into the bag (Moss)".
- Open `/moss/events?q=when did the keys move` for the raw matches.

## What was verified, and how

| What | Real Moss | Fake Moss client (tests) |
|---|---|---|
| Item matching from items.json, `MIN_SCORE` 0.4 | Yes, before this hackathon morning (commit e761de6: tracked items scored 0.52 to 1.0, untracked things at most 0.24) | Yes |
| `match_info` score and timing, one query per question | No (no credentials on the machine this was built on) | Yes |
| `understood` on `/ask` and `/listen`, only for Moss matches; the page line | No | Yes, plus screenshots of the page with the fake client |
| Events index, incremental adds, `/moss/events`, "when" questions | No | Yes |
| `EVENTS_MIN_SCORE` 0.3 | Not tuned on real Moss yet | Only on the fake's scores |
| Failures: rejected credentials, failed adds, no credentials | No | Yes |

The fake (`tests/test_moss_search.py`) implements the same `session`/`add_docs`/`query` calls
and scores a document by the share of the question's words it contains: enough to test the
wiring and the contracts, not Moss's ranking. `test_real_moss_understands_vague_wording` runs
against real Moss when `MOSS_PROJECT_ID` and `MOSS_PROJECT_KEY` are set, and is skipped
otherwise. The milliseconds in the screenshots come from the fake, so they say nothing about
real Moss latency.
