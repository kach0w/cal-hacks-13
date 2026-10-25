"""Voice: when a spoken question is over (at most MAX_S), the listen -> transcribe -> answer -> speak
turn, the prompt that biases Whisper toward item names, and dropping what isn't legible. No mic, no
model, no speaker."""
import math
from types import SimpleNamespace

import numpy as np

from fakes import FRAME, ITEMS, KEYS_TABLE, S, world
from recall import voice
from recall.voice import MAX_S, RATE, SILENCE_S, WINDOW, endpoint, listen_blocks

W = WINDOW / RATE


def windows(seconds, p):
    return [p] * math.ceil(seconds / W)


def test_endpoint_waits_for_the_question_to_finish():
    assert endpoint(windows(1, 0.0)) is None                                    # not started yet
    assert endpoint(windows(0.3, 0.0) + windows(1, 0.9)) is None                # still talking
    pause = windows(0.3, 0.0) + windows(1, 0.9) + windows(0.4, 0.1) + windows(0.5, 0.9)
    assert endpoint(pause) is None                                              # a pause between words
    assert endpoint(pause + windows(SILENCE_S, 0.1)) == "done"


def test_endpoint_gives_up_on_silence_and_ignores_a_click():
    assert endpoint(windows(MAX_S, 0.0)) == "nothing"
    assert endpoint(windows(0.5, 0.0) + windows(0.05, 0.99) + windows(MAX_S, 0.0)) == "nothing"


def test_endpoint_cuts_the_question_off_at_max_s():
    assert MAX_S == 5.0
    assert endpoint(windows(MAX_S - 0.1, 0.9)) is None       # still talking, still listening
    assert endpoint(windows(MAX_S, 0.9)) == "done"           # cut off and sent
    assert endpoint(windows(3, 0.0) + windows(MAX_S - 3, 0.9)) == "done"   # started late: still cut at MAX_S


def loud_blocks(pattern):
    """100 ms mic blocks, 1 = speech, 0 = quiet; the fake VAD scores a window by its loudness."""
    return [np.full(voice.BLOCK, float(x), np.float32) for x in pattern]


def fake_vad(audio):
    n = len(audio) // WINDOW
    return [float(audio[i * WINDOW:(i + 1) * WINDOW].mean()) for i in range(n)]


def test_listen_returns_the_audio_once_the_speaker_stops():
    blocks = loud_blocks([0, 0] + [1] * 10 + [0] * 30)
    audio = listen_blocks(iter(blocks), vad=fake_vad)
    assert audio is not None and 12 * voice.BLOCK <= len(audio) < 30 * voice.BLOCK


def test_listen_returns_none_when_nobody_speaks():
    assert listen_blocks(iter(loud_blocks([0] * 60)), vad=fake_vad) is None


def test_chatter_quieter_than_the_gate_is_not_a_question():
    # 0.6 = people talking across the room: speech to the VAD, but under the gate
    assert listen_blocks(iter(loud_blocks([0.6] * 60)), vad=fake_vad, gate=0.8) is None
    blocks = loud_blocks([0.6] * 20 + [1] * 10 + [0.6] * 30)
    audio = listen_blocks(iter(blocks), vad=fake_vad, gate=0.8)
    assert audio is not None
    # the 2 s of chatter before the question is cut down to PAD_S
    assert abs(int(np.argmax(audio > 0.9)) - voice.PAD_S * RATE) <= WINDOW


def test_enter_stops_listening_early():
    presses = iter([False, False, True])
    audio = listen_blocks(iter(loud_blocks([1] * 40)), vad=fake_vad, stop=lambda: next(presses))
    assert len(audio) == 3 * voice.BLOCK


def seg(text, logprob=-0.1, no_speech=0.01, compression=0.8):
    return SimpleNamespace(text=text, avg_logprob=logprob, no_speech_prob=no_speech, compression_ratio=compression)


class Whisper:
    def __init__(self, *segments):
        self.segments, self.calls = segments, []

    def transcribe(self, audio, **kw):
        self.calls.append(kw)
        return iter(self.segments), None


def heard(*segments):
    return voice.Transcriber(ITEMS, whisper=Whisper(*segments))(np.zeros(RATE, np.float32))


def test_transcriber_prompt_names_the_items_and_joins_segments():
    whisper = Whisper(seg(" Where are my "), seg("specs? "))
    assert voice.Transcriber(ITEMS, whisper=whisper)(np.zeros(RATE, np.float32)) == "Where are my specs?"
    assert "glasses" in whisper.calls[0]["initial_prompt"] and "key" in whisper.calls[0]["initial_prompt"]
    assert whisper.calls[0]["language"] == "en"


def test_nothing_legible_is_empty():
    assert heard() == ""                                                  # no speech at all
    assert heard(seg("Look at the buttons, look at the buttons.", logprob=-0.9)) == ""   # unsure
    assert heard(seg("keys keys keys keys keys keys", compression=3.1)) == ""             # looping
    assert heard(seg("Where are my keys?", no_speech=0.9)) == ""         # it thinks it was silence
    assert heard(seg(" Thank you.")) == "" and heard(seg("you")) == ""   # made up from noise
    assert heard(seg("...")) == "" and heard(seg("♪")) == ""
    # the unsure part is dropped, the clear part kept
    assert heard(seg("mumble mumble", logprob=-1.4), seg("Where are my glasses?")) == "Where are my glasses?"


def turn(tmp_path, heard, audio=np.zeros(RATE, np.float32)):
    cfg, store, memory, q = world(tmp_path)
    for t in (1, 2, 3):
        memory.update([S("keys", KEYS_TABLE)], FRAME, t)
    said = []
    mic = SimpleNamespace(listen=lambda: audio)
    text = voice.ask_once(voice.answer_with(q), mic, lambda a: heard, said.append)
    return text, said


def test_a_spoken_question_is_answered_out_loud(tmp_path):
    text, said = turn(tmp_path, "Where are my keys?")
    assert text == "Where are my keys?"
    assert len(said) == 1 and "keys" in said[0] and "table" in said[0]


def test_silence_and_mumbles_are_not_questions(tmp_path):
    assert turn(tmp_path, "anything", audio=None) == (None, [])
    text, said = turn(tmp_path, "")
    assert text is None and said == ["Sorry, I didn't catch that."]


def test_a_bad_time_is_an_answer_not_a_crash(tmp_path):
    _, said = turn(tmp_path, "what changed since banana")
    assert len(said) == 1 and said[0]
