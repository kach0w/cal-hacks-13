"""Moss item matching over a fake Moss client: documents, thresholds, fallbacks, and the wiring
into ask.handle and the web page. No network, no credentials."""
import json
import os
from dataclasses import replace
from types import SimpleNamespace

import pytest

from fakes import FRAME, ITEMS, KEYS_TABLE, S, world
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
    assert 'id="answerCard"' in PAGE and "function understood(u)" in PAGE
    assert "understood(r.understood)" in PAGE and "(Moss, " in PAGE


@pytest.mark.skipif(not (os.environ.get("MOSS_PROJECT_ID") and os.environ.get("MOSS_PROJECT_KEY")),
                    reason="needs MOSS_PROJECT_ID and MOSS_PROJECT_KEY (real Moss)")
def test_real_moss_understands_vague_wording():
    m = MossSearch(ABOUT, os.environ["MOSS_PROJECT_ID"], os.environ["MOSS_PROJECT_KEY"])
    assert m.ready(60)
    oid, score, ms = m.match_info("where is the thing I open the front door with")
    assert oid == "keys" and score >= m.min_score and ms < moss_search.TIMEOUT_S * 1000
