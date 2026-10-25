"""Moss item matching over a fake Moss client: documents, thresholds, fallbacks, and the wiring
into ask.handle and the web page. No network, no credentials."""
import json
import os
import shutil
import subprocess
from dataclasses import replace
from types import SimpleNamespace

import pytest

from fakes import BAG_TABLE, FRAME, GLASSES_OVER_BAG, ITEMS, KEYS_COUNTER, KEYS_TABLE, S, world
from recall import ask, moss_search
from recall.moss_search import MossSearch, documents
from recall.web import Web

ABOUT = {**ITEMS, "keys": replace(ITEMS["keys"], extra={"about": "what you open the front door with"})}


class FakeSession:
    """Scores a doc by the share of the question's words it contains — enough to rank."""
    def __init__(self):
        self.docs = []

    async def add_docs(self, docs):
        self.docs += docs

    async def query(self, text, options=None):
        words = set(text.lower().split())
        scored = sorted(((len(words & set(d.text.lower().split())) / len(words), d) for d in self.docs),
                        key=lambda x: -x[0])
        top = scored[: getattr(options, "top_k", 3)]
        return SimpleNamespace(docs=[SimpleNamespace(id=d.id, text=d.text, score=s) for s, d in top])


class FakeClient:
    def __init__(self, fail=False):
        self.fail, self.session_args = fail, None

    async def session(self, name, model=None, **kw):
        if self.fail:
            raise RuntimeError("The project credentials were rejected.")
        self.session_args = (name, model, kw)
        return FakeSession()


def search(items=ABOUT, **kw):
    m = MossSearch(items, "pid", "key", client=FakeClient(**kw))
    m.ready(5)
    return m


def test_documents_cover_every_way_of_naming_an_item():
    docs = dict(documents(ABOUT))
    assert docs["keys#0"] == "keys"
    assert "what you open the front door with" in docs.values()
    assert sum(i.startswith("keys#") for i in docs) == 3   # name/prompt deduped, alias, about
    assert all(i.split("#")[0] in ABOUT for i in docs)


def test_match_resolves_vague_wording_to_an_item():
    m = search()
    assert m.ready()
    assert m.match("where is the thing I open the front door with") == "keys"


def test_match_below_min_score_is_none():
    m = search()
    m.min_score = 0.9
    assert m.match("where is the thing I open the front door with") is None
    assert m.match("") is None


def test_rejected_credentials_mean_no_match_not_a_crash():
    m = search(fail=True)
    assert not m.ready()
    assert m.match("where are my keys") is None and m.search("keys") == []


def test_from_env_needs_credentials(monkeypatch, tmp_path):
    monkeypatch.delenv("MOSS_PROJECT_ID", raising=False)
    monkeypatch.delenv("MOSS_PROJECT_KEY", raising=False)
    monkeypatch.setattr(moss_search, "ENV_FILE", tmp_path / ".env")
    assert MossSearch.from_env(ITEMS) is None


def test_read_env(tmp_path):
    (tmp_path / ".env").write_text('# moss\nexport MOSS_PROJECT_ID="abc"\nMOSS_PROJECT_KEY=k=1\n\n')
    assert moss_search.read_env(tmp_path / ".env") == {"MOSS_PROJECT_ID": "abc", "MOSS_PROJECT_KEY": "k=1"}
    assert moss_search.read_env(tmp_path / "missing") == {}


def test_handle_uses_moss_only_when_exact_words_miss(tmp_path):
    cfg, store, memory, q = world(tmp_path, items=ABOUT)
    for t in (1, 2, 3):
        memory.update([S("keys", KEYS_TABLE)], FRAME, t)
    question = "where is the thing I open the front door with"
    assert ask.handle(question, q, 3).text == "I don't have a record of that."
    q.moss = search()
    assert "table" in ask.handle(question, q, 3).text
    q.moss.min_score = 2  # exact words never go through Moss
    assert "table" in ask.handle("where are my keys", q, 3).text


def test_web_ask_guides_to_the_item_moss_found(tmp_path, monkeypatch):
    monkeypatch.setattr(moss_search.MossSearch, "from_env", classmethod(lambda cls, items: search(items)))
    cfg, store, memory, _ = world(tmp_path, items=ABOUT)
    for t in (1, 2, 3):
        memory.update([S("keys", KEYS_TABLE)], FRAME, t)
    web = Web(cfg, memory.items, memory, SimpleNamespace(latest=lambda: None), memory.places)
    r = web.ask("where is the thing I open the front door with")
    assert "table" in r["answer"] and web.guiding["item"] == "keys"


def test_match_info_gives_item_score_and_query_time():
    m = search()
    oid, score, ms = m.match_info("where is the thing I open the front door with")
    assert oid == "keys" and m.min_score <= score <= 1 and 0 <= ms < 1000
    assert m.match_info("nothing like any item") is None
    m.min_score = 2  # the threshold applies to a remembered answer too
    assert m.match_info("where is the thing I open the front door with") is None


def test_match_info_asks_moss_once_for_the_same_words():
    m = search()
    session = m._ready.result()
    calls = []
    real = session.query
    session.query = lambda *a, **kw: calls.append(a) or real(*a, **kw)
    for _ in range(3):
        assert m.match("where is the thing I open the front door with") == "keys"
    assert len(calls) == 1


def test_wording_keeps_the_part_that_names_the_thing():
    assert moss_search.wording("Where is the thing I open the front door with?") == "the thing I open the front door with"
    assert moss_search.wording("where are my reading things") == "reading things"
    assert moss_search.wording("history of the door thing") == "the door thing"
    assert moss_search.wording("keyz") == "keyz"


def test_find_item_info_says_how_moss_understood(tmp_path):
    cfg, store, memory, q = world(tmp_path, items=ABOUT)
    assert ask.find_item_info(q, "where is the thing I open the front door with") == (None, None)
    q.moss = search()
    assert ask.find_item_info(q, "where are my keys") == ("keys", None)  # exact words: no Moss
    oid, u = ask.find_item_info(q, "where is the thing I open the front door with")
    assert oid == "keys" and u["item"] == "keys" and u["by"] == "moss"
    assert isinstance(u["ms"], float) and 0 < u["score"] <= 1


def moss_web(tmp_path, monkeypatch, moss=search):
    monkeypatch.setattr(moss_search.MossSearch, "from_env", classmethod(lambda cls, items: moss and moss(items)))
    cfg, store, memory, _ = world(tmp_path, items=ABOUT)
    for t in (1, 2, 3):
        memory.update([S("keys", KEYS_TABLE)], FRAME, t)
    return Web(cfg, memory.items, memory, SimpleNamespace(latest=lambda: None), memory.places)


def test_web_ask_carries_understood_only_for_moss_matches(tmp_path, monkeypatch):
    web = moss_web(tmp_path, monkeypatch)
    r = web.ask("Where is the thing I open the front door with?")
    u = r["understood"]
    assert u["item"] == "keys" and u["name"] == "keys" and u["by"] == "moss"
    assert u["words"] == "the thing I open the front door with"
    assert isinstance(u["ms"], float) and isinstance(u["score"], float)
    assert json.loads(json.dumps(r)) == r   # the page gets it as JSON
    assert "understood" not in web.ask("where are my keys")
    assert "understood" not in web.ask("what changed today")
    assert "understood" not in web.ask("where is the moon")
    assert web.ask("history of the thing I open the front door with")["understood"]["item"] == "keys"


def test_web_listen_carries_understood(tmp_path, monkeypatch):
    web = moss_web(tmp_path, monkeypatch)
    web.mic = SimpleNamespace(listen=lambda stop: b"audio")
    web.stt = lambda audio: "where is the thing I open the front door with"
    r = web.listen()
    assert r["heard"] and r["understood"]["item"] == "keys" and "table" in r["answer"]


def test_moss_status_route(tmp_path, monkeypatch):
    web = moss_web(tmp_path, monkeypatch)
    assert web.routes["/moss"]({}) == {"on": True, "ready": True, "model": moss_search.MODEL,
                                       "min_score": moss_search.MIN_SCORE}
    web = moss_web(tmp_path / "off", monkeypatch, moss=None)
    assert web.routes["/moss"]({}) == {"on": False}
    assert "understood" not in web.ask("where is the thing I open the front door with")


def test_page_shows_the_moss_line():
    from recall.web import PAGE
    assert 'id="answerCard"' in PAGE and "function understood(u, events)" in PAGE
    assert "understood(r.understood, r.events)" in PAGE and "(Moss, " in PAGE and "From the history" in PAGE


def bag_world(tmp_path, monkeypatch):
    """Keys on the table, then the glasses going into the blue bag."""
    web = moss_web(tmp_path, monkeypatch)
    memory = web.memory
    for t in (4, 5, 6):
        memory.update([S("keys", KEYS_TABLE), S("bag", BAG_TABLE), S("glasses", GLASSES_OVER_BAG)], FRAME, t)
    for t in (7, 8, 9, 10):
        memory.update([S("keys", KEYS_TABLE), S("bag", BAG_TABLE)], FRAME, t)
    assert any(e.kind == "inferred_contained" for e in memory.store.events())
    return web


def test_search_events_finds_the_moment_and_indexes_only_new_events(tmp_path, monkeypatch):
    web = bag_world(tmp_path, monkeypatch)
    m, store = web.query.moss, web.memory.store
    describe = web.query.describe
    hits, ms = m.search_events("when did the glasses go into the bag", store.events(), describe)
    e, words, score = hits[0]
    assert e.kind == "inferred_contained" and e.object == "glasses" and "went into the blue bag" in words
    assert 0 <= ms < 1000 and score >= moss_search.EVENTS_MIN_SCORE
    indexed = len(m._events.docs)
    assert indexed == len(store.events())
    m.search_events("glasses", store.events(), describe)
    assert len(m._events.docs) == indexed  # nothing new, nothing added
    web.memory.update([S("keys", KEYS_COUNTER), S("bag", BAG_TABLE)], FRAME, 11)
    web.memory.update([S("keys", KEYS_COUNTER), S("bag", BAG_TABLE)], FRAME, 12)
    m.search_events("keys", store.events(), describe)
    assert len(m._events.docs) == len(store.events()) > indexed
    assert m.search_events("", store.events(), describe) == ([], None)


def test_search_events_drops_events_no_longer_in_the_log(tmp_path, monkeypatch):
    web = bag_world(tmp_path, monkeypatch)
    m, events = web.query.moss, web.memory.store.events()
    m.search_events("glasses into the bag", events, web.query.describe)
    kept = [e for e in events if e.kind != "inferred_contained"]  # as if the memory started over
    hits, _ = m.search_events("glasses into the bag", kept, web.query.describe)
    assert all(e.kind != "inferred_contained" for e, _, _ in hits)


def test_search_events_survives_moss_failing(tmp_path, monkeypatch):
    web = bag_world(tmp_path, monkeypatch)
    m = web.query.moss

    async def broken(docs):
        raise RuntimeError("index full")
    m._events.add_docs = broken
    assert m.search_events("glasses", web.memory.store.events(), web.query.describe) == ([], None)
    assert m._event_ids == set()  # tried again next time
    assert search(fail=True).search_events("glasses", web.memory.store.events(), web.query.describe) == ([], None)


def test_moss_events_route_and_when_questions(tmp_path, monkeypatch):
    web = bag_world(tmp_path, monkeypatch)
    r = web.routes["/moss/events"]({"q": ["when did the glasses go into the bag"]})
    assert r["on"] and isinstance(r["ms"], float)
    top = r["matches"][0]
    assert top["text"] == "glasses probably went into the blue bag" and top["time"] and top["t"] == 9
    assert json.loads(json.dumps(r)) == r
    assert web.routes["/moss/events"]({}) == {"on": True, "ms": None, "matches": []}
    a = web.ask("When did the glasses go into the bag?")
    assert "blue bag" in a["answer"] and [(e["t"], e["text"]) for e in a["events"]] == [(top["t"], top["text"])]
    assert "events" not in web.ask("where are my glasses")
    assert top["item"] == "glasses" and all(e["item"] == "glasses" for e in a["events"])
    assert not web.ask("When is my daughter coming?").get("events")  # no item, no moment from the log
    assert not web.ask("When were my keys probably put into the bag?").get("events")  # only the keys' own moments
    a = web.ask("When did the glasses go into the blue bag?")  # names the bag: things going into it count
    assert [e["text"] for e in a["events"]] == ["glasses probably went into the blue bag"]
    off = moss_web(tmp_path / "off", monkeypatch, moss=None)
    assert off.routes["/moss/events"]({"q": ["glasses"]}) == {"on": False, "matches": []}
    assert "events" not in off.ask("when did the glasses go into the bag")


@pytest.mark.skipif(not shutil.which("node"), reason="needs node to parse the page's script")
def test_page_script_parses(tmp_path):
    from recall.web import PAGE
    js = tmp_path / "page.js"
    js.write_text(PAGE.split("<script>")[1].split("</script>")[0])
    assert subprocess.run(["node", "--check", str(js)], capture_output=True).returncode == 0


@pytest.mark.skipif(not (os.environ.get("MOSS_PROJECT_ID") and os.environ.get("MOSS_PROJECT_KEY")),
                    reason="needs MOSS_PROJECT_ID and MOSS_PROJECT_KEY (real Moss)")
def test_real_moss_understands_vague_wording():
    m = MossSearch(ABOUT, os.environ["MOSS_PROJECT_ID"], os.environ["MOSS_PROJECT_KEY"])
    assert m.ready(60)
    oid, score, ms = m.match_info("where is the thing I open the front door with")
    assert oid == "keys" and score >= m.min_score and ms < moss_search.TIMEOUT_S * 1000


def test_item_matching_works_without_the_history_index(tmp_path):
    class NoHistory(FakeClient):
        async def session(self, name, model=None, **kw):
            if name == moss_search.EVENTS_INDEX:
                raise RuntimeError("index limit reached")
            return await super().session(name, model, **kw)
    m = MossSearch(ABOUT, "pid", "key", client=NoHistory())
    assert m.ready(5) and m.match("what you open the front door with") == "keys"
    assert m.search_events("glasses", [], str) == ([], None)
