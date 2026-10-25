"""Ask about the day: Claude answers the care partner's questions from the memory, through tools.

For the family, not for the person living with dementia: their spoken answers stay templated and
local (recall/ask.py). Here Claude reads the memory only through five tools that wrap Query, and
answers in a few warm, plain lines, citing clock times and keeping "seen" apart from "probably".
Tool results are text only: no photos, no photo paths, no images leave the laptop.

    export ANTHROPIC_API_KEY=...          # or put it in .env; RECALL_CARE_MODEL picks another model
    python -m recall.care "did the glasses go missing today?"
    python -m recall.care --note          # today's care note

No key -> status "off" and the rest of Recall works as before. Any API error or the 25 s limit ->
status "error" with a plain message; the care note then falls back to one built from today's log,
labelled as not written by Claude.
"""
import argparse
import logging
import os
import time
from collections import Counter, defaultdict
from datetime import datetime

from .ask import find_item
from .moss_search import ENV_FILE, MossSearch, read_env
from .query import Query, clock, parse_since
from .store import Store

log = logging.getLogger("recall")

MODEL = "claude-sonnet-5-5"
TIMEOUT_S = 25.0     # the whole question, every turn included; the page waits a little longer
MAX_TURNS = 6
MAX_LINES = 60       # events per tool result, newest kept
CERTAINTY = {"observed": "seen", "inferred": "probably (inferred)", "unknown": "not sure"}

SYSTEM = """You help the family care partner of a person living with dementia at home. A corner camera keeps \
a memory of where everyday things are (keys, wallet, glasses, phone, and so on). You can read that memory \
only through the tools.

How to answer:
- Use only what the tools return. If the tools don't show something, say you don't know. Never guess \
where a thing is, and never fill gaps with what usually happens.
- Cite clock times (for example "at 2:40 PM") when the tools give them.
- Keep the memory's certainty exactly as the tools word it. "Seen" means the camera saw it there. \
"Probably", "inferred" or "went into" means a best guess: the camera saw it go into something, not inside it. \
"Not sure" or "haven't seen since" means its place is unknown now. Never turn a best guess into a certainty.
- If a tool says the camera is paused, say the answer is only as of that time.
- Do not give medical advice, and do not diagnose or comment on the person's memory, health or behaviour. \
If asked, say that is a question for their doctor, and stick to where things are and what moved.
- Write for a family member: short, warm, plain English, no jargon, no markdown headings or bold. \
A few short sentences, or a short list with one line per thing."""

NOTE_PROMPT = ("Write today's care note for the family: what moved today and when, anything that went out "
               "of sight or into something, and where the important things are now. At most 6 short lines.")


class Care:
    def __init__(self, query: Query, store: Store | None = None, client=None, model=None,
                 timeout=TIMEOUT_S):
        env = {**read_env(ENV_FILE), **os.environ}
        self.query, self.store = query, store or query.store
        self.model = model or env.get("RECALL_CARE_MODEL") or MODEL
        self.timeout = timeout
        self.client = client if client is not None else self._client(env.get("ANTHROPIC_API_KEY"))

    def _client(self, key):
        if not key:
            log.info("care: no ANTHROPIC_API_KEY; 'Ask about the day' is off")
            return None
        try:
            import anthropic
        except ImportError:
            log.warning("care: the anthropic package isn't installed; 'Ask about the day' is off")
            return None
        # No SDK retries: each retry gets the full per-request timeout again, which would break the
        # 25 s budget for the whole question. The page has a "try again" path instead.
        return anthropic.Anthropic(api_key=key, timeout=self.timeout, max_retries=0)

    @property
    def on(self) -> bool:
        return self.client is not None

    # --- what the page calls -------------------------------------------------------------

    def ask(self, question: str) -> dict:
        question = (question or "").strip()
        if not question:
            return self._out("Type a question about the day first.", "error", [], time.time())
        return self._run(question)

    def summary(self) -> dict:
        r = self._run(NOTE_PROMPT)
        if r["status"] != "ok":
            r["answer"] = f"{r['answer']}\n\n{self.fallback_note()}"
        return r

    def fallback_note(self, now=None) -> str:
        """Today's changes and where things are, straight from the memory: no Claude involved."""
        now = time.time() if now is None else now
        t = parse_since("today", now)
        lines = self._paused(now) + self._lines(self.query.changes_since(t), now)
        if lines.endswith("Nothing."):
            lines = lines[:-len("Nothing.")] + "Nothing moved today."
        return "Today, from the memory's log (not written by Claude):\n" + lines.replace(" \u2014 ", ", ")

    # --- the conversation ----------------------------------------------------------------

    def _out(self, answer, status, looked, t0) -> dict:
        return {"answer": answer, "status": status, "looked_at": looked, "model": self.model,
                "ms": int((time.time() - t0) * 1000)}

    def _run(self, question: str) -> dict:
        t0, looked = time.time(), []
        if not self.on:
            return self._out("Claude isn't set up for this home yet, so this part is off. Everything else "
                             "on the page works as usual.",
                             "off", looked, t0)
        now = datetime.now().strftime("%A %d %B, %I:%M %p").replace(" 0", " ")
        messages = [{"role": "user", "content": f"(It is now {now}.)\n\n{question}"}]
        try:
            for _ in range(MAX_TURNS):
                left = self.timeout - (time.time() - t0)
                if left < 1:
                    return self._out("Claude took too long to answer. Try again in a moment.", "error", looked, t0)
                r = self.client.messages.create(
                    model=self.model, max_tokens=4000, system=SYSTEM, tools=self.tools(),
                    messages=messages, output_config={"effort": "low"}, timeout=left)
                if r.stop_reason == "refusal":
                    return self._out("Claude couldn't answer that one. Try asking it another way.", "error", looked, t0)
                if r.stop_reason != "tool_use":
                    text = "\n".join(b.text for b in r.content if b.type == "text").strip()
                    if not text:
                        return self._out("Claude didn't give an answer. Try again.", "error", looked, t0)
                    return self._out(text, "ok", looked, t0)
                messages.append({"role": "assistant", "content": r.content})
                results = []
                for b in r.content:
                    if b.type != "tool_use":
                        continue
                    text, label, bad = self.run_tool(b.name, b.input or {})
                    looked.append(label)
                    results.append({"type": "tool_result", "tool_use_id": b.id, "content": text,
                                    **({"is_error": True} if bad else {})})
                messages.append({"role": "user", "content": results})
            return self._out("Claude needed too many steps for that one. Try a simpler question.", "error", looked, t0)
        except Exception as e:  # any API or network failure: say it plainly, never hang the page
            log.warning("care: %s: %s", type(e).__name__, e)
            return self._out(plain_error(e), "error", looked, t0)

    # --- the tools: text views over Query, nothing new about the memory ------------------

    def tools(self) -> list[dict]:
        names = ", ".join(it.name for it in self.query.items.values())
        item = {"type": "string", "description": f"The thing, in plain words. Tracked things: {names}."}
        obj = lambda props: {"type": "object", "properties": props, "required": list(props)}
        return [
            {"name": "where_is", "description": "Where one thing is now, as the memory says it: seen there, "
             "probably inside something, or not seen since a time.", "input_schema": obj({"item": item})},
            {"name": "everything_now", "description": "Where every tracked thing is now, one line each.",
             "input_schema": obj({})},
            {"name": "changes", "description": "Everything that moved, appeared or went out of sight since a time, "
             "oldest first, with clock times.", "input_schema": obj({"since": {
                 "type": "string", "description": "'today', '2h', '30m', '14:30' or '2:40pm'."}})},
            {"name": "history", "description": "Everything that happened to one thing, oldest first, with times.",
             "input_schema": obj({"item": item})},
            {"name": "missing_counts", "description": "For the last N days: how often each thing went out of sight "
             "or into a container, and from where. For patterns like 'does she keep losing her glasses?'.",
             "input_schema": obj({"days": {"type": "integer", "description": "1 to 30; 7 for a week."}})},
        ]

    def run_tool(self, name, args) -> tuple[str, str, bool]:
        """(result text, what the page lists under "Looked at", whether it failed)."""
        try:
            if name == "where_is":
                item = args.get("item", "")
                oid = self._oid(item)
                be = "are" if oid and self.query.items[oid].plural else "is"
                return self.where_is(item), f"where the {item} {be} now", False
            if name == "everything_now":
                return self.everything_now(), "where everything is now", False
            if name == "changes":
                since = str(args.get("since") or "today")
                return self.changes(since), f"what changed since {since}", False
            if name == "history":
                return self.history(args.get("item", "")), f"the history of the {args.get('item', '')}", False
            if name == "missing_counts":
                days = max(1, min(30, int(args.get("days") or 7)))
                return self.missing_counts(days), f"how often things went missing ({days} days)", False
            return f"There is no tool called {name}.", name, True
        except (ValueError, TypeError) as e:
            return str(e), f"{name} (failed)", True

    def _oid(self, item: str):
        if item in self.query.items:  # already an id (everything_now): no need to ask Moss
            return item
        return find_item(self.query, item or "")

    def _not_tracked(self, item):
        names = ", ".join(it.name for it in self.query.items.values())
        return f"'{item}' isn't one of the tracked things. Tracked things: {names}."

    def _paused(self, now=None) -> str:
        as_of = self.query.as_of(now)
        return f"The camera is paused since {clock(as_of)}; nothing after that is known.\n" if as_of else ""

    def where_is(self, item, now=None) -> str:
        oid = self._oid(item)
        if not oid:
            return self._not_tracked(item)
        a = self.query.where_is(oid, now)
        o = self.store.load_state().get(oid)
        if a.status == "none":
            return f"No record: the camera has never seen the {self.query.name(oid)}."
        word = CERTAINTY.get(a.status, a.status)
        return f"{a.text} [certainty: {word}; last seen at {clock(o.last_observed)}]"

    def everything_now(self, now=None) -> str:
        lines = [f"- {self.query.name(oid)}: {self.where_is(oid, now)}" for oid in self.query.items]
        return self._paused(now) + "\n".join(lines)

    def _lines(self, events, now) -> str:
        if not events:
            return "Nothing."
        today = datetime.fromtimestamp(now).date()
        places, out = self.query.places(), []
        for e in events[-MAX_LINES:]:
            d = datetime.fromtimestamp(e.ts)
            when = clock(e.ts) if d.date() == today else d.strftime("%a %d %b ") + clock(e.ts)
            out.append(f"{when}: {self.query.describe(e, places)} [{CERTAINTY.get(e.status, e.status)}]")
        skipped = len(events) - MAX_LINES
        return (f"({skipped} earlier ones not shown)\n" if skipped > 0 else "") + "\n".join(out)

    def changes(self, since, now=None) -> str:
        now = time.time() if now is None else now
        t = parse_since(since, now)
        return self._paused(now) + f"Since {clock(t)}:\n" + self._lines(self.query.changes_since(t), now)

    def history(self, item, now=None) -> str:
        now = time.time() if now is None else now
        oid = self._oid(item)
        if not oid:
            return self._not_tracked(item)
        return f"{self.query.name(oid)}:\n" + self._lines(self.query.history(oid), now)

    def missing_counts(self, days, now=None) -> str:
        now = time.time() if now is None else now
        since = now - days * 86400
        places = self.query.places()
        counts = defaultdict(Counter)  # name -> Counter of "went out of sight from the table" etc.
        for e in self.store.events():
            if e.ts < since:
                continue
            if e.kind == "disappeared":
                counts[e.name][f"went out of sight {places.phrase(e.from_surface)}"] += 1
            elif e.kind == "inferred_contained":
                counts[e.name][f"went into the {self.query.name(e.container)}, "
                               f"{places.phrase(e.from_surface)}"] += 1
        if not counts:
            return f"In the last {days} days nothing went out of sight or into anything."
        lines = [f"Last {days} days (counts of events in the memory):"]
        for name, c in sorted(counts.items()):
            lines.append(f"- {name}: " + "; ".join(f"{what}: {n} time{'s' if n != 1 else ''}"
                                                   for what, n in c.most_common()))
        return "\n".join(lines)


def plain_error(e) -> str:
    try:
        import anthropic
    except ImportError:
        return "Claude couldn't answer just now. Try again in a moment."
    if isinstance(e, anthropic.APITimeoutError):
        return "Claude took too long to answer. Try again in a moment."
    if isinstance(e, anthropic.APIConnectionError):
        return "Couldn't reach Claude. Check the laptop's internet connection."
    if isinstance(e, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
        return "Claude didn't accept this laptop's key. Check ANTHROPIC_API_KEY in the .env file."
    if isinstance(e, anthropic.RateLimitError):
        return "Claude is busy right now. Try again in a minute."
    return "Claude couldn't answer just now. Try again in a moment."


def main():
    from . import config
    from .enroll import load_items

    ap = argparse.ArgumentParser(description="Ask Claude about the day, from the memory.")
    ap.add_argument("question", nargs="*")
    ap.add_argument("--note", action="store_true", help="today's care note")
    config.add_args(ap)
    a = ap.parse_args()
    cfg = config.from_args(a)
    q = Query(cfg, load_items(cfg.items_path))
    q.moss = MossSearch.from_env(q.items)
    care = Care(q)
    r = care.summary() if a.note or not a.question else care.ask(" ".join(a.question))
    print(r["answer"])
    if r["looked_at"]:
        print("\nLooked at: " + "; ".join(r["looked_at"]))


if __name__ == "__main__":
    main()
