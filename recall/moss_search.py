"""Moss semantic search over the tracked items: which item a spoken question means.

The exact-word match in ask.match_item stays first (instant, offline); this catches the rest —
"the thing I open the door with", "my reading things", a garbled transcript. One Moss session
index lives in memory, built from items.json at start-up: one document per way of naming an
item (name, detector prompt, each alias, and the optional "about" line the family writes).

    export MOSS_PROJECT_ID=... MOSS_PROJECT_KEY=...     # from portal.usemoss.dev, or put them in .env
    python -m recall.moss_search "the thing I open the front door with"

No credentials, no `moss` package, or Moss down -> match() returns None and asking works as
before. match_info() also says how sure Moss was and how long the query took, so the page can show
'Understood "the thing I open the front door with" as keys (Moss, 12 ms)'. The SDK is async; it runs on a private event loop on a daemon thread, so the threaded
web server and the REPL call match() as a plain function.
"""
import asyncio
import logging
import os
import re
import threading
import time
from pathlib import Path

log = logging.getLogger("recall")

INDEX = "recall-items"
MODEL = "moss-minilm"
MIN_SCORE = 0.4      # tuned on real questions: our items 0.52-1.0, things we don't track <= 0.24
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"   # gitignored KEY=value lines
TIMEOUT_S = 1.0      # a slow answer is worse than "I don't have a record of that"
READY_WAIT_S = 10.0  # a question asked while the index is still building waits this long for it


def read_env(path) -> dict:
    """KEY=value lines (export prefixes, quotes and comments allowed); {} if there's no file."""
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return {}
    env = {}
    for line in lines:
        line = line.strip().removeprefix("export ").strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip("\"'")
    return env


def documents(items) -> list[tuple[str, str]]:
    """(doc id, text) per way of naming each item; the doc id starts with the item id."""
    docs = []
    for oid, it in items.items():
        about = it.extra.get("about")
        terms = dict.fromkeys(t.strip() for t in (it.name, it.prompt, *it.aliases, about) if t and t.strip())
        docs += [(f"{oid}#{n}", t) for n, t in enumerate(terms)]
    return docs


def wording(text) -> str:
    """The part of a question that names the thing: 'where is the thing I open the door with?'
    -> 'the thing I open the door with'."""
    t = re.sub(r"\s+", " ", text).strip(" ?.!")
    lead = r"^(?:(?:where(?:'s| is| are)?|have you seen|did you see|find|history(?: of)?)\s+)+(?:my |our )?"
    return re.sub(lead, "", t, flags=re.I).strip() or t


class MossSearch:
    def __init__(self, items, project_id, project_key, min_score=MIN_SCORE, client=None):
        self.items, self.min_score = items, min_score
        self._last = (None, None)  # (text, best hit) of the last question: handle() and the page both ask
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True, name="moss").start()
        self._ready = asyncio.run_coroutine_threadsafe(self._build(project_id, project_key, client), self._loop)

    @classmethod
    def from_env(cls, items) -> "MossSearch | None":
        env = {**read_env(ENV_FILE), **os.environ}
        pid, key = env.get("MOSS_PROJECT_ID"), env.get("MOSS_PROJECT_KEY")
        if not (pid and key):
            log.info("moss: no MOSS_PROJECT_ID/MOSS_PROJECT_KEY; questions use exact words only")
            return None
        try:
            import moss  # noqa: F401
        except ImportError:
            log.warning("moss: package not installed (pip install moss); questions use exact words only")
            return None
        return cls(items, pid, key, float(env.get("MOSS_MIN_SCORE", MIN_SCORE)))

    async def _build(self, project_id, project_key, client):
        from moss import DocumentInfo, MossClient
        client = client or MossClient(project_id, project_key)
        session = await client.session(INDEX, MODEL, auto_load_on_init=False)
        await session.add_docs([DocumentInfo(id=i, text=t) for i, t in documents(self.items)])
        log.info("moss: indexed %d item names", len(documents(self.items)))
        return session

    def ready(self, timeout=None) -> bool:
        try:
            self._ready.result(timeout)
            return True
        except Exception:
            return False

    def search(self, text, top_k=3) -> list[tuple[str, float]]:
        """[(item id, score)], best first; [] if Moss isn't ready or fails."""
        return self._search(text, top_k)[0]

    def _search(self, text, top_k):
        """(search results, milliseconds the query itself took; None if it never ran)."""
        if not text.strip() or not self.ready(READY_WAIT_S):  # only the first question waits
            return [], None

        async def q():
            from moss import QueryOptions
            return await self._ready.result().query(text, QueryOptions(top_k=top_k))
        t0 = time.perf_counter()
        try:
            r = asyncio.run_coroutine_threadsafe(q(), self._loop).result(TIMEOUT_S)
        except Exception:
            log.warning("moss: query failed", exc_info=True)
            return [], None
        return [(d.id.split("#")[0], d.score) for d in r.docs], (time.perf_counter() - t0) * 1000

    def match_info(self, text) -> tuple[str, float, float] | None:
        """(item id, score, query milliseconds) for the item a question means, or None when
        nothing clears min_score."""
        last_text, best = self._last
        if text != last_text:
            best, ms = self._search(text, top_k=1)
            best = (*best[0], ms) if best else None
            if ms is not None:  # a failed query isn't remembered: the next ask tries again
                self._last = (text, best)
        if best and best[1] >= self.min_score and best[0] in self.items:
            return best
        return None

    def match(self, text) -> str | None:
        """The item a question means, or None when nothing clears min_score."""
        info = self.match_info(text)
        return info[0] if info else None


def main():
    import argparse

    from . import config
    from .enroll import load_items
    ap = argparse.ArgumentParser(description="Which tracked item does a question mean (Moss)?")
    ap.add_argument("text", nargs="+")
    config.add_args(ap)
    a = ap.parse_args()
    cfg = config.from_args(a)
    m = MossSearch.from_env(load_items(cfg.items_path))
    if m is None or not m.ready(30):
        raise SystemExit("moss isn't available: set MOSS_PROJECT_ID and MOSS_PROJECT_KEY")
    text = " ".join(a.text)
    for oid, score in m.search(text):
        print(f"{score:.3f}  {oid}")
    info = m.match_info(text)
    print(f"-> {info[0]} ({info[2]:.1f} ms)" if info else "-> no match")


if __name__ == "__main__":
    main()
