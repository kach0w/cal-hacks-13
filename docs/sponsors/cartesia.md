# Cartesia: the voice Recall speaks with

Everything Recall tells a person living with dementia is spoken: the answer ("Your keys are on the
table, seen 3 minutes ago") and the turn-by-turn words while the band points ("To your left",
"Straight ahead", "Right here"). For someone who may not hold onto a sentence they read, and who can
be unsettled by a flat or rushed robotic voice, the voice is part of the product. It should be calm,
a little slower than normal, the same voice every time, and it has to start quickly: a direction that
arrives two seconds late points at where the person used to be facing.

With `CARTESIA_API_KEY` set, Recall speaks with Cartesia's Sonic text-to-speech, on the laptop (the
band's button, `python -m recall.voice`) and on the care-partner page. Without it, nothing changes:
macOS `say` on the laptop, the browser's own voice on the page.

## What we use

| | Value | Source |
|---|---|---|
| Model | `sonic-3.6` (`tts.MODEL`), the recommended model that tracks the latest stable snapshot | [TTS models](https://docs.cartesia.ai/build-with-cartesia/tts-models/latest) |
| Voice | Jacqueline, en-US, `9626c31c-bec5-4cca-baa8-f8ba9e84c8bc` (`tts.VOICE`), one of the English voices Cartesia recommends for voice agents (stable, realistic). `CARTESIA_VOICE_ID` swaps it | [TTS models](https://docs.cartesia.ai/build-with-cartesia/tts-models/latest), [Voice Library](https://play.cartesia.ai/voices) |
| Pace and tone | `generation_config = {"speed": 0.9, "emotion": "calm"}` (speed range 0.6 to 1.5; `calm` is one of the primary emotions) | [Volume, speed, and emotion](https://docs.cartesia.ai/build-with-cartesia/sonic-3/volume-speed-emotion) |
| Endpoint | `POST /tts/bytes` through the official Python SDK (`cartesia` 4.2.0, `client.tts.generate(...).read()`), WAV `pcm_s16le` at 24 kHz | [TTS bytes API](https://docs.cartesia.ai/api-reference/tts/bytes) |

The voice was picked from the documented recommended voices; it has not been auditioned against
the alternatives with a live key (see "What was verified"). Changing it is one environment variable.

## How it fits together

- `recall/tts.py`: `Voice.synthesize(text) -> bytes | None`. One shared `Voice` (`tts.default()`)
  serves the laptop and the page, so they share one cache.
- `recall/voice.py` `speak(text)`: Cartesia audio played with `afplay`; if there's no audio, macOS
  `say` exactly as before. This is what the band's button and the terminal voice loop use.
- `recall/web.py`: `GET /tts?text=...` returns `audio/wav`, or `{"ok": false}` JSON (with
  `"off": true` when there is no key). The page's `say()` asks `/tts` first and plays the WAV with an
  `Audio` element; after an `"off"` answer it stops asking and uses the browser's voice for the rest
  of the visit.

## Latency strategy: never wait on the network for long

- **Cache by text (LRU, 128 sentences).** Directions are five phrases said over and over, and
  answers repeat too ("Your keys are on the table..."). A repeat costs nothing.
- **Warm-up.** At start-up a background thread makes the five direction phrases and "Sorry, I didn't
  catch that.", so the first "To your left" is already instant.
- **Timeouts.** The SDK request gets 4 s and no retries. The page gives `/tts` 1.5 s; if the audio
  isn't there by then, that sentence is spoken by the browser instead. The server keeps the late
  audio, so next time it's instant.
- **Back-off.** After a failed request, Cartesia is left alone for 30 s, so a dropped wifi doesn't
  add a 4 s pause before every sentence.
- **Nothing in the live loop calls Cartesia.** Speaking happens on the web server's request threads
  and the band/voice threads; the camera loop never waits on it.
- **Answers are never cut off by directions.** The page's existing rule is kept: while an answer is
  being spoken (by either voice), new directions wait. Pressing "Ask out loud" silences both voices.

## Setup

```
export CARTESIA_API_KEY=...          # or a CARTESIA_API_KEY=... line in the gitignored .env
python -m recall.tts "Your keys are on the table."    # prints size and time, then plays it
python -m recall.live --web --band --voice            # the page and the band now use the voice
```

Optional: `CARTESIA_VOICE_ID=<voice id>` to try another voice from the Voice Library.

## Demo

1. Start live with the key set and open the page. Ask "where are my keys?": the answer is spoken in
   the Cartesia voice.
2. Stand in front of the camera and turn away from the keys: the arrow guides you, and "To your left"
   plays instantly (made at start-up).
3. Press the band's button and ask out loud: the laptop answers in the same voice.
4. Unset the key and restart: everything still talks, with the built-in voices.

## What was verified, and how

Verified **without** a Cartesia key (none was available while building this):

- `tests/test_tts.py`, with a fake client: the request fields (model, voice, WAV, speed 0.9, calm,
  4 s timeout), cache hits, LRU eviction, failure returns `None` and starts the back-off, no key means
  off, the 400-character limit, `speak()` plays through `afplay` and falls back to `say` (subprocess
  mocked), and `/tts` over real HTTP returning `audio/wav` or the off JSON.
- The real `cartesia` 4.2.0 SDK against a local stand-in server: it sends `POST /tts/bytes` with a
  `Cartesia-Version` header and exactly the documented body, and our code reads the audio bytes back.
- In headless Chrome with a stand-in voice serving a real WAV: the page plays the audio from `/tts`,
  pressing stop (speechSynthesis.cancel) silences it, an answer holds back directions until the audio
  ends, a voice that takes 3 s falls back to the browser's voice at 1.5 s and plays from the cache the
  next time, and a server without a key makes the page use the browser's voice from then on.
- `tts.play` plays a WAV through `afplay` on macOS.

**Not** verified: a request to the real Cartesia API, the real latency, and how Jacqueline actually
sounds. `test_real_cartesia` does that and is skipped until `CARTESIA_API_KEY` is set:

```
CARTESIA_API_KEY=... python -m pytest tests/test_tts.py -q
```

Browsers only play audio after the person has interacted with the page (autoplay rules). Asking a
question is that interaction; before it, for example guidance started from the band while the page
was only open, the page falls back to the browser's voice for that sentence.
