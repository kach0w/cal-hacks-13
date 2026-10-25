"""Cartesia text-to-speech: one calm, clear voice for everything Recall says out loud (answers and
"To your left"), on the laptop (voice.speak) and on the page (the /tts route, web.py).

    export CARTESIA_API_KEY=...                # from play.cartesia.ai, or put it in .env
    python -m recall.tts "Your keys are on the table."      # plays it on this laptop

Sonic (MODEL) over the bytes endpoint, a WAV back in one request. Answers repeat and directions
repeat a lot, so audio is kept by text (LRU) and the direction phrases are made in the background at
start-up: those then play instantly. A request gets TIMEOUT_S, and after a failure Cartesia is left
alone for DOWN_S. No key, no `cartesia` package, Cartesia slow or down -> synthesize() returns None
and the caller speaks as before (macOS `say`, the browser's own voice).
"""
import logging
import os
import subprocess
import tempfile
import threading
import time
from collections import OrderedDict

from .moss_search import ENV_FILE, read_env

log = logging.getLogger("recall")

# https://docs.cartesia.ai/build-with-cartesia/tts-models/latest : the recommended Sonic model, and
# its recommended English voices for voice agents. CARTESIA_VOICE_ID overrides.
MODEL = "sonic-3.6"
VOICE = "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"  # Jacqueline (en-US)
# https://docs.cartesia.ai/build-with-cartesia/sonic-3/volume-speed-emotion : speed 0.6..1.5,
# emotion one of a fixed list. A little slower than normal and calm is easier to follow.
SPEED = 0.9
EMOTION = "calm"
FORMAT = {"container": "wav", "encoding": "pcm_s16le", "sample_rate": 24000}
CONTENT_TYPE = "audio/wav"
TIMEOUT_S = 4.0     # a late voice is worse than the plain one
DOWN_S = 30.0       # after a failure, the plain voice for this long (no 4 s wait per sentence)
CACHE = 128         # sentences kept, ~100 KB each
MAX_CHARS = 400     # longer text isn't sent (the page speaks it with the browser's voice)
PHRASES = ("Right here", "Straight ahead", "Behind you", "To your right", "To your left",
           "Sorry, I didn't catch that.")  # guide.words and voice.ask_once: made ahead of time


class Voice:
    """synthesize(text) -> WAV bytes, or None when there's no voice (no key) or it failed."""
    def __init__(self, api_key=None, client=None, voice_id=VOICE, model=MODEL):
        self.voice_id, self.model = voice_id, model
        self._client, self._key = client, api_key
        self.on = client is not None or bool(api_key)
        self._cache = OrderedDict()
        self._lock = threading.Lock()
        self._down_until = 0.0

    @classmethod
    def from_env(cls) -> "Voice":
        env = {**read_env(ENV_FILE), **os.environ}
        key = env.get("CARTESIA_API_KEY")
        if not key:
            log.info("tts: no CARTESIA_API_KEY; speaking with the built-in voices")
            return cls()
        try:
            import cartesia  # noqa: F401
        except ImportError:
            log.warning("tts: cartesia package not installed; speaking with the built-in voices")
            return cls()
        return cls(api_key=key, voice_id=env.get("CARTESIA_VOICE_ID") or VOICE)

    def client(self):
        if self._client is None:
            from cartesia import Cartesia
            self._client = Cartesia(api_key=self._key, timeout=TIMEOUT_S, max_retries=0)
        return self._client

    def synthesize(self, text) -> bytes | None:
        text = " ".join((text or "").split())
        if not (self.on and text) or len(text) > MAX_CHARS:
            return None
        with self._lock:
            if text in self._cache:
                self._cache.move_to_end(text)
                return self._cache[text]
            if time.monotonic() < self._down_until:
                return None
        try:
            t0 = time.monotonic()
            audio = self.client().tts.generate(
                model_id=self.model, transcript=text, voice=self.voice_id, output_format=FORMAT,
                language="en", generation_config={"speed": SPEED, "emotion": EMOTION},
                timeout=TIMEOUT_S).read()
            log.debug("tts: %d bytes in %.2f s for %r", len(audio), time.monotonic() - t0, text)
        except Exception as e:
            log.warning("tts: Cartesia failed (%s); the built-in voice for %.0f s", type(e).__name__, DOWN_S)
            with self._lock:
                self._down_until = time.monotonic() + DOWN_S
            return None
        if not audio:
            return None
        with self._lock:
            self._cache[text] = audio
            while len(self._cache) > CACHE:
                self._cache.popitem(last=False)
        return audio

    def warm(self, phrases=PHRASES):
        """Makes the phrases said most often in the background, so they play instantly."""
        if self.on:
            threading.Thread(target=lambda: [self.synthesize(p) for p in phrases], name="tts-warm",
                             daemon=True).start()


_default, _default_lock = None, threading.Lock()


def default() -> Voice:
    """The one Voice the laptop and the page share (one cache), from CARTESIA_API_KEY."""
    global _default
    with _default_lock:
        if _default is None:
            _default = Voice.from_env()
            _default.warm()
        return _default


def play(audio) -> bool:
    """Plays WAV bytes through the speakers (macOS afplay) and waits; False if it couldn't."""
    if not audio:
        return False
    path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio)
            path = f.name
        return subprocess.run(["afplay", path], check=False).returncode == 0
    except OSError:  # no afplay (not a Mac)
        return False
    finally:
        if path:
            os.unlink(path)


def route(voice):
    """GET /tts?text=... for the page: (WAV, "audio/wav"), or {"ok": false} to use its own voice
    ("off": true when there's no Cartesia voice at all, so it stops asking)."""
    def tts(params):
        if not voice.on:
            return {"ok": False, "off": True}
        text = " ".join(params.get("text", [""])[0].split())
        if not text or len(text) > MAX_CHARS:
            return {"ok": False, "error": f"text must be 1 to {MAX_CHARS} characters"}
        audio = voice.synthesize(text)
        return (audio, CONTENT_TYPE) if audio else {"ok": False}
    return tts


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    v = Voice.from_env()
    if not v.on:
        raise SystemExit("Cartesia isn't available: set CARTESIA_API_KEY")
    text = " ".join(sys.argv[1:]) or "Your keys are on the table."
    t0 = time.monotonic()
    clip = v.synthesize(text)
    if clip is None:
        raise SystemExit("Cartesia didn't answer (see the log above)")
    print(f"{len(clip)} bytes in {time.monotonic() - t0:.2f} s")
    t0 = time.monotonic()
    v.synthesize(text)
    print(f"again, from the cache: {time.monotonic() - t0:.3f} s")
    play(clip)
