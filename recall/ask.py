"""Ask the memory questions. Typed here; the voice layer and POST /ask call handle().

    python -m recall.ask                         # REPL
    python -m recall.ask "where are my glasses"
    python -m recall.ask "what changed since 10m"     # also: today, 2h, 14:30, 2:40pm
    python -m recall.ask "diff 14:30"
    python -m recall.ask "history keys"
    python -m recall.ask state
    python -m recall.ask --replay "history glasses"   # ask about the last --src replay
"""
import argparse
import re
import time

from . import config
from .enroll import load_items
from .models import Answer
from .query import Query, clock, parse_since, stamp
from .store import Store
from .zones import load_zones, zone_phrase


def match_item(items, text) -> str | None:
    """The item id named in free text ('where are my specs?'); the longest match wins."""
    t = " " + re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower())) + " "
    best, best_len = None, 0
    for oid, it in items.items():
        for term in (oid, it.name, *it.aliases):
            term = term.lower().strip()
            if term and f" {term} " in t and len(term) > best_len:
                best, best_len = oid, len(term)
    return best


def _since_arg(t):
    if "since" in t:
        return t.split("since", 1)[1].strip(" ?") or "today"
    if t.startswith("diff"):
        return t[4:].strip(" ?") or "today"
    return "today"


def handle(text, q: Query, now=None) -> Answer:
    """Route one question to a query. Raises ValueError for an unparseable time."""
    now = time.time() if now is None else now
    t = text.strip().lower()
    if not t:
        return Answer("", "info")
    items, as_of = q.items, q.as_of(now)

    if t in ("state", "status", "all"):
        state = q.store.load_state()
        lines = [f"- {q.where_is(oid, now, stamped=False).text}" if oid in state and state[oid].last_observed
                 else f"- {q.name(oid)}: never seen" for oid in items]
        return stamp(Answer("\n".join(lines) if state else "No memory yet.", "info"), as_of)

    zones = load_zones(q.cfg.zones_path)
    if t.startswith("diff"):
        since = parse_since(_since_arg(t), now)
        d = q.diff(since)
        fmt = lambda s: "not seen" if s is None else (
            f"{s['status']} {zone_phrase(zones, s['zone'])}"
            + (f" (in {q.name(s['container'])})" if s.get("container") else ""))
        text = "\n".join(f"- {q.name(x['object_id'])}: {fmt(x['before'])}  →  {fmt(x['now'])}"
                         for x in d) or f"Nothing is different from {clock(since)}."
        return stamp(Answer(text, "info"), as_of)

    if t.startswith("history"):
        oid = match_item(items, t)
        if not oid:
            return Answer("History of what? e.g. 'history keys'", "info")
        h = q.history(oid)
        return stamp(Answer("\n".join(f"{clock(e.ts)}  {q.describe(e, zones)}" for e in h)
                            or "No history.", "info"), as_of)

    if ("change" in t or "moved" in t) and "where" not in t:
        since = parse_since(_since_arg(t), now)
        ev = q.changes_since(since)
        text = ("\n".join(f"{clock(e.ts)}  {q.describe(e, zones)}" for e in ev)
                or f"Nothing has changed since {clock(since)}.")
        return stamp(Answer(text, "info"), as_of)

    oid = match_item(items, t)
    a = q.where_is(oid, now, stamped=False) if oid else Answer("I don't have a record of that.", "none")
    return stamp(a, as_of)


def main():
    ap = argparse.ArgumentParser(description="Ask the memory a question.")
    ap.add_argument("text", nargs="*")
    ap.add_argument("--replay", action="store_true", help="ask about the last video replay")
    config.add_args(ap)
    a = ap.parse_args()
    cfg = config.from_args(a)
    q = Query(cfg, load_items(cfg.items_path), Store(cfg.replay_dir) if a.replay else None)

    def answer(text):
        r = handle(text, q)
        extra = f"\n  [{r.status} {r.confidence:.2f}] still: {r.still_path}" if r.still_path else ""
        return r.text + extra

    if a.text:
        try:
            print(answer(" ".join(a.text)))
        except ValueError as e:
            raise SystemExit(str(e))
        return
    print("ask: 'where are my keys', 'what changed since 10m', 'diff 14:30', "
          "'history glasses', 'state'. Ctrl-C to quit.")
    try:
        while True:
            try:
                print(answer(input("> ")))
            except ValueError as e:
                print(e)
    except (KeyboardInterrupt, EOFError):
        print()


if __name__ == "__main__":
    main()
