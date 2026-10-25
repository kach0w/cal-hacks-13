# Ask about the day: Claude for the care partner

Track: Anthropic, Claude for Public Benefit. Code: `recall/care.py`, the "Ask about the day" card
in `recall/web.py`, tests in `tests/test_care.py`.

## Who it's for

The care partner: usually an adult son or daughter of a parent living with early dementia at
home. They are often not in the room. They check in by phone between work and their own family,
and they want to know two things without interrogating their parent: *is everything where it
should be*, and *how did today go*.

## The problem

Recall's camera already keeps an honest memory: every move, every "went into the blue bag", with
times and with "seen" kept apart from "probably". But a timeline of fifty lines is not something a
worried family member reads on a bus. They ask questions in their own words: "Did Mum lose her
glasses again today?", "Has the phone turned up?", "Is this happening more this week?". The
templated, local question answering in `recall/ask.py` is right for the person with dementia
(instant, spoken, predictable) but can't answer those.

## What Claude does

Claude answers the care partner's questions, and writes a short "Today's note", by reading the
memory through five tools. It does not see the room, the camera, or any picture. It decides which
parts of the memory to look at, puts the facts together, and words them warmly and plainly, with
clock times and with the memory's certainty kept intact.

The person with dementia's own path is unchanged: their questions are still answered by the
templated, local code and spoken aloud. Claude's answers are shown on the page only, never spoken.

| Tool | Wraps (existing code only) | Returns |
|---|---|---|
| `where_is(item)` | `Query.where_is` | The same sentence the spoken answer uses, plus `[certainty: seen / probably (inferred) / not sure; last seen at 2:40 PM]` |
| `everything_now()` | `Query.where_is` over all items | One line per tracked thing |
| `changes(since)` | `Query.changes_since` + `Query.describe`, `parse_since` (`today`, `2h`, `14:30`, `2:40pm`) | One line per event with its clock time and certainty |
| `history(item)` | `Query.history` + `Query.describe` | Every event for one thing, dated when not today |
| `missing_counts(days)` | counts over `Store.events()` | How often each thing went out of sight or into a container, and from where |

Item names go through the same matcher the spoken path uses (`ask.find_item`: exact words, then
Moss if it's on), so "specs" or "Mum's glasses" find the glasses. The page lists what Claude looked
at under each answer ("Looked at: what changed since today; where the glasses are now"), so the
family can see what the answer rests on.

Request: Anthropic Messages API, `anthropic` SDK, `client.messages.create` with `tools`, looping
while `stop_reason == "tool_use"` (at most 6 turns), `output_config.effort = "low"` (these are
short lookups). Default model `claude-sonnet-5-5`; `RECALL_CARE_MODEL` overrides it.

## Where it could go wrong, and what we did about it

| Risk for the family | Mitigation |
|---|---|
| Claude says something is somewhere when the memory only guesses | Every tool result carries the memory's own certainty words; the system prompt says never to turn a best guess into a certainty and explains what "seen", "probably" and "not sure" mean. |
| Claude fills a gap with a plausible story ("she probably left them in the bedroom") | The system prompt: use only tool results; if the tools don't show it, say you don't know. The tools only ever return what the memory recorded. The "Looked at" line makes it checkable. |
| The memory is stale (camera off) but the answer sounds current | `where_is` answers are stamped "As of 2:40 PM" by the existing code, and `changes` / `everything_now` start with "The camera is paused since ..."; the prompt tells Claude to say so. |
| Turning a log of lost things into a judgement about the person ("her memory is getting worse") | The prompt forbids medical advice, diagnosis, and comment on the person's memory, health or behaviour; it says those are questions for their doctor. `missing_counts` returns counts, not trends or interpretations. |
| Privacy: pictures of a home leaving the house | Tools return text only. No photo, no photo path, no camera frame is ever sent; a test checks every tool result for photo paths. Only short facts like "keys moved to the table at 2:40 PM" go to the API. |
| Claude or the network is slow or down, and the page hangs | One question has a 25 s budget across all turns (the per-request SDK timeout is what's left); the page stops waiting at 35 s. Any API error becomes a plain message ("Couldn't reach Claude. Check the laptop's internet connection."). The note falls back to one built from today's log, clearly labelled "not written by Claude". |
| A refusal or an empty answer | Shown as a plain "Claude couldn't answer that one" message, never as a blank. |
| No key on the laptop | The card says Claude isn't set up; the rest of Recall (board, timeline, spoken answers, guidance) works exactly as before. |

Known limits: answers are only as good as the camera's memory (things it never saw, it can't
report). Claude's wording is not checked against the facts by code; the "Looked at" line and the
board beside it are how a family member checks.

## Privacy

Text only. What goes to Anthropic: the care partner's question, the current time, the names of the
tracked things and of the places, and the tool results above. No images, no photo paths, no video,
no names of people (Recall has no concept of who moved something). The photos stay in
`data/photos/` on the laptop as before.

## Setup

Put a key in `.env` at the repo root (gitignored) or in the environment:

    ANTHROPIC_API_KEY=sk-ant-...
    RECALL_CARE_MODEL=claude-sonnet-5-5     # optional

Then run Recall as usual (`.venv/bin/python -m recall.live --web ...`); the card is on the page,
in the right-hand column under "This week" (last on a phone). From a terminal:

    .venv/bin/python -m recall.care "did the glasses go missing today?"
    .venv/bin/python -m recall.care --note

## Demo script

(Steps 2 to 5 describe what the prompt and tools are designed to produce; they have not yet been run
against the live API, see the last section.)

1. Run Recall live (`.venv/bin/python -m recall.live --web`) and open the printed link.
2. Under "Ask about the day", press **Today's note**: a few lines on what moved today, with times,
   and a "Looked at" line, for example "what changed since today; where everything is now".
3. Ask "Did the glasses go missing today?": Claude says they *probably* went into the blue bag at
   the time shown, and that it's a best guess.
4. Ask "Where's the phone?": Claude says the camera has never seen it, instead of guessing.
5. Ask "Is her memory getting worse?": Claude declines to judge and points to their doctor, then
   offers what the memory does show.
6. Unplug the network and ask again: a plain error appears within the time limit; "Today's note"
   still shows the note built from the log, labelled as not from Claude.

## What was verified, and how

- **Fakes (automated, `tests/test_care.py`, 10 tests):** a fake client whose `messages.create`
  returns `tool_use` then `end_turn`: the right tools run and their facts reach Claude; tool results
  never contain photo paths; status "off" without a key; plain errors on API failure; the overall
  timeout stops the loop; too many turns and refusals; the summary fallback; `/care` and
  `/care/summary` over HTTP on a real `Web` server.
- **Real SDK, fake server (manual):** the installed `anthropic` 2.3.0 SDK pointed at a local fake
  Messages endpoint: the request body (`model`, `system`, `tools`, `messages`, `output_config`) is
  accepted by the SDK, and Claude's assistant turn (thinking + tool_use blocks) goes back unchanged
  with the tool results on the next request.
- **Not verified against the live Claude API:** no `ANTHROPIC_API_KEY` was available while this was
  built, so the answers' quality, the model ID and the request shape have not been run against
  Anthropic's servers. `test_real_claude_answers_from_the_memory` runs that check once a key is set.
- **Page:** checked in the preview harness, light and dark, desktop and phone: off, loading and
  answer states.
