"""Ask out loud on this laptop: press Enter, ask, hear the answer. Step 1 of the voice plan: the
band's button takes Enter's place later; everything after it stays.

    python -m recall.voice                            # ask the live memory (data/)
    python -m recall.voice --memory data/sim/real     # ask another memory, e.g. the simulated table
    python -m recall.voice --file question.wav        # a recording instead of the mic, then quit
    python -m recall.voice --mics                     # list microphones; pick one with --mic N
    python -m recall.voice --model small.en           # slower, better in a noisy room
    python -m recall.live --web --voice               # inside live: an answer also starts guidance (+ --band)

The web page's Listen button (recall/web.py) uses the same mic and Whisper: the press is the Enter.

Enter starts listening. Listening stops by itself SILENCE_S after you stop talking (or press Enter
again), and never runs past MAX_S: the question is cut off there and sent. Silero VAD (it ships with
faster-whisper) hears where the speech ends; Whisper turns the clip into text on this laptop (the
model downloads once, then no network); ask.handle answers it exactly as if it were typed; macOS
`say` speaks the answer. Nothing legible (Whisper unsure, or only the phrases it makes up from
noise) is never sent as a question.

A crowded room talks too. Between questions the mic keeps hearing the room, and speech only counts
when it's LOUDER times the room's usual level: the person asking is close to the mic, the people
chatting nearby are not. Speak toward the laptop. -v logs both levels, for tuning LOUDER.
"""
import argparse
import logging
import queue
import re
import select
import shutil
import subprocess
import sys
import threading
from collections import deque

import numpy as np

log = logging.getLogger("recall")

RATE = 16000           # what Whisper and the VAD both want
WINDOW = 512           # the VAD scores 32 ms windows
BLOCK = 1600           # the mic hands over 100 ms at a time
SPEECH_PROB = 0.5      # a window this likely to be speech counts as speech
QUIET_PROB = 0.35      # ...and below this as quiet (between them: whatever came before)
MIN_SPEECH_S = 0.15    # less speech than this is a click or a cough, not a question
SILENCE_S = 0.8        # this much quiet after speech ends the question
MAX_S = 5.0            # listening stops here: the question is cut off and sent (or, if none, dropped)
ROOM_S = 2.0           # how much of the room, just before Enter, sets its usual level
LOUDER = 3.0           # speech counts only this many times above the room's level (~10 dB)
MIN_LEVEL = 0.002      # ...and never below this (a silent room would let any whisper in)
PAD_S = 0.3            # kept before the first loud speech, so the first word isn't clipped
MODEL = "base.en"      # ~0.4 s per question on an M-series laptop; small.en ~1.2 s, more robust
# Legible = Whisper is sure of it. Mean log-probability per segment, measured: clear questions
# -0.02..-0.36, garbled chatter down to -0.9. Repeating itself (compression) and "is this silence?"
# use Whisper's own limits.
LEGIBLE_LOGPROB = -0.8
MAX_COMPRESSION = 2.4
NO_SPEECH = 0.6
NOISE_PHRASES = {"thank you", "thanks for watching", "thank you for watching", "you", "bye"}  # Whisper's inventions


def endpoint(probs, window_s=WINDOW / RATE) -> str | None:
    """From the speech probability of every window so far: 'done' (a question, then quiet, or cut
    off at MAX_S), 'nothing' (no speech by MAX_S), or None to keep listening."""
    heard = sum(p >= SPEECH_PROB for p in probs) * window_s
    if heard < MIN_SPEECH_S:
        return "nothing" if len(probs) * window_s >= MAX_S else None
    last = max(i for i, p in enumerate(probs) if p >= QUIET_PROB)
    if (len(probs) - 1 - last) * window_s >= SILENCE_S or len(probs) * window_s >= MAX_S:
        return "done"
    return None


def silero(audio) -> list[float]:
    """Speech probability of each whole WINDOW of audio (the whole clip at once: the model carries
    context from window to window, so it's steadier than scoring windows one at a time)."""
    from faster_whisper.vad import get_vad_model
    n = len(audio) // WINDOW * WINDOW
    return get_vad_model()(audio[:n]).ravel().tolist() if n else []


def loudness(audio) -> np.ndarray:
    """RMS of each whole WINDOW."""
    n = len(audio) // WINDOW
    return np.sqrt((audio[: n * WINDOW].reshape(n, WINDOW) ** 2).mean(1))


def listen_blocks(blocks, vad=silero, stop=lambda: False, gate=0.0) -> np.ndarray | None:
    """Gather mic blocks until the question is over: the audio from just before the first speech
    at least `gate` loud, or None if nobody spoke that loud."""
    got = []
    for block in blocks:
        got.append(block)
        audio = np.concatenate(got)
        if stop():
            return audio
        probs = [p if level >= gate else 0.0 for p, level in zip(vad(audio), loudness(audio))]
        result = endpoint(probs)
        if result == "nothing":
            return None
        if result == "done":
            first = next(i for i, p in enumerate(probs) if p >= SPEECH_PROB)
            return audio[max(0, first * WINDOW - int(PAD_S * RATE)):]
    return np.concatenate(got) if got else None


def enter_pressed() -> bool:
    """True once Enter was pressed (consumes the line); never blocks."""
    try:
        if sys.stdin.isatty() and select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.readline()
            return True
    except (OSError, ValueError):
        pass
    return False


class Mic:
    """The laptop mic. Open from the first question on: between questions its audio only goes to
    `room`, so when Enter is pressed it knows how loud the room usually is. The terminal and the
    web page can share one: their questions take turns."""
    def __init__(self, device=None, louder=LOUDER):
        self.device, self.louder = device, louder
        self.room = deque(maxlen=round(ROOM_S * RATE / BLOCK))
        self.asking = None  # a queue while someone is asking
        self.stream = None
        self._opening, self._turn = threading.Lock(), threading.Lock()

    def _on_audio(self, data, *_):
        block = data[:, 0].copy()
        q = self.asking
        (q.put if q is not None else self.room.append)(block)

    def open(self):
        with self._opening:
            if self.stream is None:
                import sounddevice as sd
                self.stream = sd.InputStream(samplerate=RATE, channels=1, dtype="float32", blocksize=BLOCK,
                                             device=self.device, callback=self._on_audio)
                self.stream.start()

    def listen(self, stop=enter_pressed) -> np.ndarray | None:
        """Record one question from the mic; stop() true ends it early."""
        self.open()
        with self._turn:
            return self._listen(stop)

    def _listen(self, stop):
        before = list(self.room)  # a copy: the audio thread keeps appending
        room = float(np.median(loudness(np.concatenate(before)))) if before else 0.0
        gate = max(MIN_LEVEL, self.louder * room)
        q, heard = queue.Queue(), []
        self.asking = q

        def blocks():
            while True:
                try:
                    heard.append(q.get(timeout=2))
                except queue.Empty:
                    raise RuntimeError("the microphone stopped sending audio") from None
                yield heard[-1]
        try:
            print("Listening…", flush=True)
            audio = listen_blocks(blocks(), stop=stop, gate=gate)
        finally:
            self.asking = None
        loudest = float(loudness(np.concatenate(heard)).max(initial=0.0)) if heard else 0.0
        log.debug("voice: room level %.4f, speech counts from %.4f, loudest heard %.4f%s", room, gate, loudest,
                  "" if audio is not None else " (not loud enough: lower --louder, or speak closer)")
        return audio


class FileMic:
    """A recording standing in for the mic (--file): Whisper reads any audio file itself."""
    def __init__(self, path):
        self.path = path

    def listen(self):
        return self.path


class Transcriber:
    """Speech to text with Whisper on this laptop. The item names go in as the prompt, so
    'specs' and 'wallet' win over words that merely sound like them."""
    def __init__(self, items, model=MODEL, whisper=None):
        names = dict.fromkeys(t for it in items.values() for t in (it.name, *it.aliases))
        self.prompt = "Where are my " + ", ".join(names) + "?"
        self.model_name, self._model = model, whisper
        self._loading = threading.Lock()

    def load(self):
        with self._loading:
            if self._model is None:
                from faster_whisper import WhisperModel
                try:  # the cached copy first: no network needed at the venue
                    self._model = WhisperModel(self.model_name, device="cpu", compute_type="int8",
                                               local_files_only=True)
                except Exception:
                    log.info("voice: downloading Whisper %s (once)", self.model_name)
                    self._model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
        return self._model

    def __call__(self, audio) -> str:
        """What was said, or "" when nothing in it is legible. audio: 16 kHz mono float32, or a
        path to any audio file."""
        segments, _ = self.load().transcribe(audio, language="en", vad_filter=True,
                                             initial_prompt=self.prompt)
        kept = []
        for seg in segments:
            log.debug("voice: whisper %r  logprob %.2f  no_speech %.2f  compression %.2f",
                      seg.text, seg.avg_logprob, seg.no_speech_prob, seg.compression_ratio)
            if legible(seg):
                kept.append(seg.text.strip())
        text = " ".join(kept).strip()
        words = re.sub(r"[^a-z ]+", "", text.lower()).strip()
        return text if re.search(r"[a-z]{2}", words) and words not in NOISE_PHRASES else ""


def legible(seg) -> bool:
    return (seg.avg_logprob >= LEGIBLE_LOGPROB and seg.compression_ratio <= MAX_COMPRESSION
            and seg.no_speech_prob < NO_SPEECH)


def speak(text):
    """Say it out loud (macOS `say`); elsewhere the printed answer has to do."""
    if text and shutil.which("say"):
        subprocess.run(["say", text], check=False)


def ask_once(answer, mic, stt, say=speak) -> str | None:
    """One question: listen, transcribe, answer, speak. What was heard, or None."""
    audio = mic.listen()
    if audio is None:
        print("(didn't hear anything: speak toward the laptop)")
        return None
    text = stt(audio)
    if not text:
        print("(couldn't make that out)")
        say("Sorry, I didn't catch that.")
        return None
    print(f"you: {text}")
    reply = answer(text)
    print(reply)
    say(reply)
    return text


def loop(answer, mic, stt, say=speak):
    """Enter, ask, repeat, until Ctrl-C or end of input."""
    mic.open()  # starts learning how loud the room is
    stt.load()  # the first question shouldn't wait for the model
    try:
        while True:
            input("\nPress Enter and ask a question (Ctrl-C quits) ")
            ask_once(answer, mic, stt, say)
    except (KeyboardInterrupt, EOFError):
        print()


def answer_with(q):
    """ask.handle as a text -> text function; an unparseable time is the answer, not a crash."""
    from .ask import handle

    def answer(text):
        try:
            return handle(text, q).text or "Ask me where something is."
        except ValueError as e:
            return str(e)
    return answer


def main():
    from pathlib import Path

    from . import config
    from .enroll import load_items
    from .moss_search import MossSearch
    from .query import Query
    from .store import Store
    ap = argparse.ArgumentParser(description="Ask the memory out loud.")
    ap.add_argument("--memory", type=Path, metavar="DIR",
                    help="memory folder to ask (default: the live one, data/)")
    ap.add_argument("--file", help="answer the question in this recording instead of listening")
    ap.add_argument("--mic", type=int, metavar="N", help="microphone number (see --mics)")
    ap.add_argument("--mics", action="store_true", help="list microphones and quit")
    ap.add_argument("--model", default=MODEL, help=f"Whisper model (default {MODEL})")
    ap.add_argument("--louder", type=float, default=LOUDER,
                    help=f"speech must be this many times the room's level (default {LOUDER}; -v shows levels)")
    ap.add_argument("--quiet", action="store_true", help="print answers without speaking them")
    config.add_args(ap)
    a = ap.parse_args()
    if a.mics:
        import sounddevice as sd
        print(sd.query_devices())
        return
    cfg = config.from_args(a)
    items = load_items(cfg.items_path)
    store = Store(a.memory or cfg.data_dir)
    if not (store.root / store.EVENTS).exists():
        log.warning("voice: no %s yet, so every answer will be 'no record'", store.root / store.EVENTS)
    q = Query(cfg, items, store)
    q.moss = MossSearch.from_env(items)
    answer, stt = answer_with(q), Transcriber(items, a.model)
    say = (lambda text: None) if a.quiet else speak
    if a.file:
        ask_once(answer, FileMic(a.file), stt, say)
        return
    print(f"asking {store.root}")
    loop(answer, Mic(a.mic, a.louder), stt, say)


if __name__ == "__main__":
    main()
