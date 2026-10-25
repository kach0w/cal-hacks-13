"""Cartesia text-to-speech over a fake client: what is asked for, the cache, failures falling back to
the plain voice (speak -> macOS say, /tts -> the browser's), and the /tts route over HTTP. The last
test calls the real Cartesia API and only runs with CARTESIA_API_KEY set."""
import json
import os
import subprocess
import urllib.request
from types import SimpleNamespace

import pytest

from recall import tts, voice
from recall.tts import MAX_CHARS, Voice

WAV = b"RIFF\x00\x00\x00\x00WAVEfmt fake audio"


class FakeCartesia:
    """client.tts.generate(**kw).read(), like the cartesia SDK; fail=True raises instead."""
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail
        self.tts = SimpleNamespace(generate=self.generate)

    def generate(self, **kw):
        self.calls.append(kw)
        if self.fail:
            raise TimeoutError("Cartesia took too long")
        return SimpleNamespace(read=lambda: WAV + kw["transcript"].encode())


def test_asks_sonic_for_a_calm_slower_wav_and_caches_by_text():
    fake = FakeCartesia()
    v = Voice(client=fake)
    audio = v.synthesize("Your keys are on the table.")
    assert audio.startswith(WAV)
    kw = fake.calls[0]
    assert kw["model_id"] == tts.MODEL and kw["voice"] == tts.VOICE
    assert kw["output_format"]["container"] == "wav"
    assert kw["generation_config"] == {"speed": tts.SPEED, "emotion": "calm"}
    assert kw["timeout"] == tts.TIMEOUT_S
    assert v.synthesize("  Your keys are   on the table. ") == audio  # same words: from the cache
    assert len(fake.calls) == 1


def test_cache_forgets_the_least_recently_used(monkeypatch):
    monkeypatch.setattr(tts, "CACHE", 2)
    fake = FakeCartesia()
    v = Voice(client=fake)
    for text in ("To your left", "To your right", "To your left", "Behind you"):
        v.synthesize(text)
    assert [c["transcript"] for c in fake.calls] == ["To your left", "To your right", "Behind you"]
    v.synthesize("To your left")   # kept: used more recently than "To your right"
    v.synthesize("To your right")  # forgotten
    assert len(fake.calls) == 4


def test_failure_returns_none_and_rests_cartesia_for_a_while():
    fake = FakeCartesia(fail=True)
    v = Voice(client=fake)
    assert v.synthesize("To your left") is None
    assert v.synthesize("To your right") is None
    assert len(fake.calls) == 1   # down: no second 4 s wait
    v._down_until = 0.0           # DOWN_S later
    fake.fail = False
    assert v.synthesize("To your right")


def test_no_key_means_no_voice(monkeypatch, tmp_path):
    monkeypatch.delenv("CARTESIA_API_KEY", raising=False)
    monkeypatch.setattr(tts, "ENV_FILE", tmp_path / ".env")
    v = Voice.from_env()
    assert not v.on and v.synthesize("To your left") is None
    (tmp_path / ".env").write_text("CARTESIA_API_KEY=sk_test\n")
    assert Voice.from_env().on


def test_text_limit():
    fake = FakeCartesia()
    v = Voice(client=fake)
    assert v.synthesize("x" * (MAX_CHARS + 1)) is None
    assert v.synthesize("") is None and v.synthesize(None) is None
    assert fake.calls == []


@pytest.fixture
def ran(monkeypatch):
    """subprocess.run recorded instead of run; `say` exists."""
    calls = []

    def run(cmd, check=False):
        calls.append(cmd)
        if cmd[0] == "afplay":
            with open(cmd[1], "rb") as f:
                assert f.read().startswith(WAV)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(voice.shutil, "which", lambda name: "/usr/bin/" + name)
    return calls


def test_speak_plays_the_cartesia_voice(monkeypatch, ran):
    monkeypatch.setattr(tts, "default", lambda: Voice(client=FakeCartesia()))
    voice.speak("Your keys are on the table.")
    assert [c[0] for c in ran] == ["afplay"]
    assert not os.path.exists(ran[0][1])  # the temp file is cleaned up


def test_speak_falls_back_to_say(monkeypatch, ran):
    monkeypatch.setattr(tts, "default", lambda: Voice(client=FakeCartesia(fail=True)))
    voice.speak("Your keys are on the table.")
    assert ran == [["say", "Your keys are on the table."]]
    ran.clear()
    monkeypatch.setattr(tts, "default", lambda: Voice())  # no key
    voice.speak("Your keys are on the table.")
    assert ran == [["say", "Your keys are on the table."]]


def test_route():
    route = tts.route(Voice(client=FakeCartesia()))
    audio, ctype = route({"text": ["To your left"]})
    assert audio.startswith(WAV) and ctype == "audio/wav"
    assert route({"text": ["x" * (MAX_CHARS + 1)]})["ok"] is False
    assert route({})["ok"] is False
    assert route({"text": ["To your left"]}) == (audio, ctype)
    assert tts.route(Voice(client=FakeCartesia(fail=True)))({"text": ["hi"]}) == {"ok": False}
    assert tts.route(Voice())({"text": ["hi"]}) == {"ok": False, "off": True}


def test_tts_over_http(monkeypatch, tmp_path):
    from test_web import make
    monkeypatch.setattr(tts, "default", lambda: Voice(client=FakeCartesia()))
    web, _ = make(tmp_path)
    url = web.start(port=0).replace(web.start.__globals__["lan_ip"](), "127.0.0.1")
    r = urllib.request.urlopen(url + "/tts?text=To%20your%20left", timeout=5)
    assert r.headers["Content-Type"] == "audio/wav" and r.read() == WAV + b"To your left"
    monkeypatch.setattr(tts, "default", lambda: Voice())
    web, _ = make(tmp_path / "off")
    url = web.start(port=0).replace(web.start.__globals__["lan_ip"](), "127.0.0.1")
    r = urllib.request.urlopen(url + "/tts?text=To%20your%20left", timeout=5)
    assert r.headers["Content-Type"] == "application/json" and json.loads(r.read()) == {"ok": False, "off": True}


@pytest.mark.skipif(not os.environ.get("CARTESIA_API_KEY"), reason="needs CARTESIA_API_KEY (real Cartesia API)")
def test_real_cartesia():
    v = Voice(api_key=os.environ["CARTESIA_API_KEY"])
    audio = v.synthesize("Your keys are on the table.")
    assert audio and audio[:4] == b"RIFF" and audio[8:12] == b"WAVE"


def test_the_real_sdk_sends_what_the_docs_ask_for():
    """The cartesia SDK itself, against a local stand-in for api.cartesia.ai: the request
    POST /tts/bytes carries the documented fields, and the audio comes back as bytes."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from cartesia import Cartesia
    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["path"], seen["version"] = self.path, self.headers["Cartesia-Version"]
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.end_headers()
            self.wfile.write(WAV)

        def log_message(self, *args):
            pass
    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.handle_request, daemon=True).start()
    client = Cartesia(api_key="test", base_url=f"http://127.0.0.1:{server.server_address[1]}", max_retries=0)
    assert Voice(client=client).synthesize("To your left") == WAV
    server.server_close()
    body = seen["body"]
    assert seen["path"] == "/tts/bytes" and seen["version"]
    assert body["model_id"] == tts.MODEL and body["voice"] == tts.VOICE and body["transcript"] == "To your left"
    assert body["output_format"] == tts.FORMAT and body["language"] == "en"
    assert body["generation_config"] == {"speed": tts.SPEED, "emotion": "calm"}
