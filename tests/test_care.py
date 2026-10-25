"""Ask about the day over a fake Anthropic client: the tool loop, grounded answers, no photo paths in
what Claude reads, off without a key, errors and timeouts, the fallback care note, and the /care
routes over HTTP. One real-API test, skipped unless ANTHROPIC_API_KEY is set."""
import json
import os
import time
import urllib.request
from types import SimpleNamespace as NS

import pytest

from fakes import BAG_TABLE, FRAME, GLASSES_OVER_BAG, KEYS_COUNTER, KEYS_TABLE, S, world
from recall import care as care_mod
from recall.care import Care
from recall.web import Web

KEY = os.environ.get("ANTHROPIC_API_KEY")  # read before the fixture below hides it


def text(t):
    return NS(type="text", text=t)


def tool(name, args, id="t1"):
    return NS(type="tool_use", name=name, input=args, id=id)


class FakeClient:
    """messages.create replays scripted responses and keeps every request it was sent."""
    def __init__(self, *responses, error=None, delay=0.0):
        self.responses, self.error, self.delay, self.calls = list(responses), error, delay, []
        self.messages = self

    def create(self, **kw):
        self.calls.append(json.loads(json.dumps(kw, default=lambda o: vars(o))))
        time.sleep(self.delay)
        if self.error:
            raise self.error
        return self.responses.pop(0)


def scene(tmp_path):
    """Keys on the counter then moved to the table; glasses went into the blue bag on the table."""
    cfg, store, memory, query = world(tmp_path)
    t = time.time() - 600
    for _ in range(3):
        t += 1
        memory.update([S("keys", KEYS_COUNTER)], FRAME, t)
    for _ in range(3):
        t += 1
        memory.update([S("keys", KEYS_TABLE), S("bag", BAG_TABLE), S("glasses", GLASSES_OVER_BAG)], FRAME, t)
    for _ in range(10):
        t += 1
        memory.update([S("keys", KEYS_TABLE), S("bag", BAG_TABLE)], FRAME, t)
    store.beat(t)  # the camera is live
    assert memory.state["glasses"].status == "inferred"
    return query, store, memory


@pytest.fixture(autouse=True)
def no_key(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("RECALL_CARE_MODEL", raising=False)
    monkeypatch.setattr(care_mod, "ENV_FILE", tmp_path / "no.env")


def test_grounded_answer_uses_the_tools_and_their_facts(tmp_path):
    query, store, _ = scene(tmp_path)
    client = FakeClient(
        NS(stop_reason="tool_use", content=[tool("where_is", {"item": "glasses"}, "a"),
                                            tool("changes", {"since": "today"}, "b")]),
        NS(stop_reason="end_turn", content=[text("Her glasses are probably in the blue bag.")]))
    r = Care(query, store, client=client).ask("where did mum's glasses go?")
    assert r["status"] == "ok" and r["answer"] == "Her glasses are probably in the blue bag."
    assert r["looked_at"] == ["where the glasses are now", "what changed since today"]
    assert r["model"] == "claude-sonnet-5-5" and isinstance(r["ms"], int)

    first, second = client.calls
    assert first["model"] == "claude-sonnet-5-5" and first["timeout"] <= 25
    assert {t["name"] for t in first["tools"]} == {"where_is", "everything_now", "changes", "history",
                                                   "missing_counts"}
    assert "only what the tools return" in first["system"]
    results = {b["tool_use_id"]: b["content"] for b in second["messages"][-1]["content"]}
    assert "probably in the blue bag" in results["a"] and "inferred" in results["a"]
    assert "keys moved from on the kitchen counter to on the table" in results["b"]
    assert "glasses went into the blue bag" in results["b"]
    assert second["messages"][1]["role"] == "assistant"  # Claude's own turn goes back unchanged


def test_tool_results_never_carry_photos(tmp_path):
    query, store, memory = scene(tmp_path)
    assert memory.state["keys"].photo_path  # the memory has photos; Claude must never see their paths
    c = Care(query, store, client=FakeClient())
    outs = [c.run_tool(n, a)[0] for n, a in [
        ("where_is", {"item": "keys"}), ("where_is", {"item": "glasses"}), ("everything_now", {}),
        ("changes", {"since": "today"}), ("history", {"item": "keys"}), ("missing_counts", {"days": 7})]]
    for out in outs:
        assert "photos/" not in out and ".jpg" not in out and "photo" not in out


def test_tools_wrap_the_memory(tmp_path):
    query, store, _ = scene(tmp_path)
    c = Care(query, store, client=FakeClient())
    assert "on the table" in c.where_is("keys") and "certainty: seen" in c.where_is("keys")
    assert "never seen" in c.where_is("phone")
    assert "isn't one of the tracked things" in c.where_is("umbrella")
    now = c.everything_now()
    assert "- keys:" in now and "- glasses:" in now and "probably in the blue bag" in now
    assert "went into the blue bag, on the table: 1 time" in c.missing_counts(7)
    assert "keys appeared" in c.history("key")
    text_, label, bad = c.run_tool("changes", {"since": "yesterday-ish"})
    assert bad and "can't parse" in text_


def test_paused_camera_is_said(tmp_path):
    query, store, _ = scene(tmp_path)
    store.beat(time.time() - 3600, camera_ok=False)
    c = Care(query, store, client=FakeClient())
    assert "camera is paused" in c.changes("today") and "As of" in c.where_is("keys")


def test_off_without_a_key(tmp_path):
    query, store, _ = scene(tmp_path)
    c = Care(query, store)
    assert not c.on
    r = c.ask("what happened today?")
    assert r["status"] == "off" and "isn't set up" in r["answer"]
    note = c.summary()
    assert note["status"] == "off" and "not written by Claude" in note["answer"] and "keys" in note["answer"]


def test_api_error_is_plain_and_the_note_falls_back(tmp_path):
    import anthropic
    query, store, _ = scene(tmp_path)
    err = anthropic.APIConnectionError(request=None)
    c = Care(query, store, client=FakeClient(error=err))
    r = c.ask("anything new?")
    assert r["status"] == "error" and "Couldn't reach Claude" in r["answer"]
    note = c.summary()
    assert note["status"] == "error"
    assert "from the memory's log (not written by Claude)" in note["answer"]
    assert "\u2014" not in note["answer"]
    assert "glasses went into the blue bag" in note["answer"]


def test_timeout_stops_the_loop(tmp_path):
    query, store, _ = scene(tmp_path)
    loop = NS(stop_reason="tool_use", content=[tool("everything_now", {})])
    client = FakeClient(*[loop] * 10, delay=0.3)
    r = Care(query, store, client=client, timeout=1.2).ask("where is everything?")
    assert r["status"] == "error" and "too long" in r["answer"]
    assert len(client.calls) < 6


def test_too_many_turns_and_refusals(tmp_path):
    query, store, _ = scene(tmp_path)
    loop = NS(stop_reason="tool_use", content=[tool("everything_now", {})])
    r = Care(query, store, client=FakeClient(*[loop] * 10)).ask("where is everything?")
    assert r["status"] == "error" and len(r["looked_at"]) == care_mod.MAX_TURNS
    r = Care(query, store, client=FakeClient(NS(stop_reason="refusal", content=[]))).ask("?")
    assert r["status"] == "error"
    assert Care(query, store, client=FakeClient()).ask("  ")["status"] == "error"


def test_model_from_env(tmp_path, monkeypatch):
    query, store, _ = scene(tmp_path)
    monkeypatch.setenv("RECALL_CARE_MODEL", "claude-haiku-5-5")
    assert Care(query, store, client=FakeClient()).model == "claude-haiku-5-5"


def test_one_question_keeps_to_its_budget(tmp_path, monkeypatch):
    query, store, _ = scene(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-real")
    c = Care(query, store)  # builds the real SDK client; nothing is sent
    assert c.on and c.client.max_retries == 0  # a retry would get the whole timeout again


def test_everything_now_does_not_search_by_id(tmp_path):
    query, store, _ = scene(tmp_path)

    class NoMoss:
        def match(self, text):
            raise AssertionError("everything_now should use the ids it already has")

    from recall.models import Item
    query.items["tv_remote"] = Item(id="tv_remote", name="clicker", prompt="tv remote")  # id not in its words
    query.moss = NoMoss()
    now = Care(query, store, client=FakeClient()).everything_now()
    assert "- keys:" in now and "- clicker: No record" in now

def test_care_routes_over_http(tmp_path):
    query, store, memory = scene(tmp_path)

    class P:
        source = NS(live=False)

        def latest(self):
            return None

    web = Web(query.cfg, memory.items, memory, P(), memory.places)
    url = web.start(port=0).replace(web.start.__globals__["lan_ip"](), "127.0.0.1")
    get = lambda p: json.loads(urllib.request.urlopen(url + p, timeout=10).read())
    assert get("/care?q=what%20happened")["status"] == "off"
    web.care.client = FakeClient(NS(stop_reason="end_turn", content=[text("The keys moved to the table.")]),
                                 NS(stop_reason="end_turn", content=[text("A calm day.")]))
    r = get("/care?q=what%20happened")
    assert r["status"] == "ok" and r["answer"] == "The keys moved to the table."
    assert get("/care/summary")["answer"] == "A calm day."
    page = urllib.request.urlopen(url + "/", timeout=10).read().decode()
    assert "Ask about the day" in page and "/care/summary" in page


@pytest.mark.skipif(not KEY, reason="needs ANTHROPIC_API_KEY")
def test_real_claude_answers_from_the_memory(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    query, store, _ = scene(tmp_path)
    r = Care(query, store).ask("Where are the glasses, and how sure are you?")
    assert r["status"] == "ok", r["answer"]
    assert r["looked_at"] and "bag" in r["answer"].lower()
