"""Moss semantic search over the tracked items: which item a spoken question means.

The exact-word match in ask.match_item stays first (instant, offline); this catches the rest —
"the thing I open the door with", "my reading things", a garbled transcript. One Moss session
index lives in memory, built from items.json at start-up: one document per way of naming an
item (name, detector prompt, each alias, and the optional "about" line the family writes).

    export MOSS_PROJECT_ID=... MOSS_PROJECT_KEY=...     # from portal.usemoss.dev
    python -m recall.moss_search "the thing I open the front door with"

No credentials, no `moss` package, or Moss down -> match() returns None and asking works as
before. The SDK is async; it runs on a private event loop on a daemon thread, so the threaded
web server and the REPL call match() as a plain function.
"""
import asyncio
import logging
import os
import threading

log = logging.getLogger("recall")

INDEX = "recall-items"
MODEL = "moss-minilm"
MIN_SCORE = 0.5      # below this the question names nothing we track; tune on real questions
TIMEOUT_S = 1.0      # a slow answer is worse than "I don't have a record of that"


def documents(items) -> list[tuple[str, str]]:
    """(doc id, text) per way of naming each item; the doc id starts with the item id."""
    docs = []
    for oid, it in items.items():
        about = it.extra.get("about")
        terms = dict.fromkeys(t.strip() for t in (it.name, it.prompt, *it.aliases, about) if t and t.strip())
        docs += [(f"{oid}#{n}", t) for n, t in enumerate(terms)]
    return docs


class MossSearch:
    def __init__(self, items, project_id, project_key, min_score=MIN_SCORE, client=None):
        self.items, self.min_score = items, min_score
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True, name="moss").start()
        self._ready = asyncio.run_coroutine_threadsafe(self._build(project_id, project_key, client), self._loop)

    @classmethod
    def from_env(cls, items) -> "MossSearch | None":
        pid, key = os.environ.get("MOSS_PROJECT_ID"), os.environ.get("MOSS_PROJECT_KEY")
        if not (pid and key):
            log.info("moss: no MOSS_PROJECT_ID/MOSS_PROJECT_KEY; questions use exact words only")
            return None
        try:
            import moss  # noqa: F401
        except ImportError:
            log.warning("moss: package not installed (pip install moss); questions use exact words only")
            return None
        return cls(items, pid, key, float(os.environ.get("MOSS_MIN_SCORE", MIN_SCORE)))

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
        if not text.strip() or not self._ready.done() or self._ready.exception():
            return []

        async def q():
            from moss import QueryOptions
            return await self._ready.result().query(text, QueryOptions(top_k=top_k))
        try:
            r = asyncio.run_coroutine_threadsafe(q(), self._loop).result(TIMEOUT_S)
        except Exception:
            log.warning("moss: query failed", exc_info=True)
            return []
        return [(d.id.split("#")[0], d.score) for d in r.docs]

    def match(self, text) -> str | None:
        """The item a question means, or None when nothing clears min_score."""
        best = self.search(text, top_k=1)
        if best and best[0][1] >= self.min_score and best[0][0] in self.items:
            return best[0][0]
        return None


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
    print("->", m.match(text) or "no match")


if __name__ == "__main__":
    main()
