"""Moss item matching over a fake Moss client: documents, thresholds, fallbacks, and the wiring
into ask.handle and the web page. No network, no credentials."""
from dataclasses import replace
from types import SimpleNamespace

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


def test_from_env_needs_credentials(monkeypatch):
    monkeypatch.delenv("MOSS_PROJECT_ID", raising=False)
    monkeypatch.delenv("MOSS_PROJECT_KEY", raising=False)
    assert MossSearch.from_env(ITEMS) is None


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
