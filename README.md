# Recall

**A room that remembers where you put things, and points you to them.**

A webcam in the corner of a room keeps a running memory of where the things that matter are:
keys, wallet, glasses, phone, the TV remote. Ask *"where are my glasses?"*, out loud or typed, and
Recall answers in a second. Then a small band worn on the chest points an arrow toward them, so
the answer doesn't have to be remembered to be followed.

It's built for people living with Alzheimer's or another dementia, for whom "I don't remember
putting it anywhere" is a symptom, not an annoyance. It's also built for their care partner, who
gets one page showing what's where and what moved today.

Product write-up: [`DEVPOST.md`](./DEVPOST.md). Design: [`docs/DESIGN.md`](docs/DESIGN.md).
Camera pipeline: [`docs/CAMERA.md`](docs/CAMERA.md).

---

## What it does

- **Remembers places, not footage.** Every second the camera looks at the room and updates a
  memory of each object: which surface it's on ("the kitchen counter", "the sofa"), where that is
  in the room in metres, and when it was last seen. No video is kept.
- **Answers when the thing is out of sight.** Things vanish into bags and drawers exactly when
  nobody is watching. When an object disappears, Recall looks at the last picture it was in and
  works out what it went into: *"Your glasses are probably in the blue bag, on the table."* If
  the bag moves, the glasses move with it.
- **Never confidently lies.** Every answer says whether it was **seen** or is a **best guess**,
  and an inferred answer is never promoted to a seen one until the camera actually sees it.
- **Points the way.** After a question, Recall finds the person with a pose model, works out
  where they are standing and which way their chest faces, and sends the band a direction. The
  arrow turns toward the item (or the bag it's in), and the web page shows the same direction on
  a dial, with "To your left" spoken aloud as it changes.
- **Gives the care partner a view.** A web page, open on any phone or laptop on the same wifi,
  shows where each thing is, how sure Recall is, whether the camera is live, and a timeline of
  everything that moved.

## How it works

```
 SETUP (once, by the care partner)
   printed markers on the floor + a wall
   LiDAR phone scan (Polycam)  ──▶  3D map of the room's fixed surfaces
   webcam sees the markers     ──▶  camera registered inside that map
   surfaces named on a top-down plan ("kitchen counter", "sofa")

 LIVE LOOP (once a second)
   newest frame ─▶ change gate ─▶ open-vocabulary detector ─▶ one box per enrolled item
               ─▶ box base → ray into the scanned room → (x, y, z) → which surface
               ─▶ memory rules: appeared / moved / disappeared / went into a container
               ─▶ events + current state written to the database, evidence photo saved locally

 ASKING
   typed on the page │ "Ask out loud" on the page │ the band's button │ Enter in the terminal
        └──▶ Whisper on the laptop ─▶ exact-word match, else semantic search (Moss)
        └──▶ templated answer with seen / probably / not sure ─▶ spoken back

 GUIDING
   pose model → shoulders placed at shoulder height in the room → position + facing
   direction to the item (or its container) ─▶ "dir,dist" over Bluetooth LE ─▶ servo arrow
```

### Perception

The detector is open-vocabulary: each item is enrolled with a plain-language prompt ("keys on a
red lanyard", "light blue wallet"), so new things can be tracked without training. It runs
YOLO-World by default, with YOLOE or a fine-tuned YOLO as drop-in alternatives
(`--set DETECTOR=yoloe`). A change gate skips detection when nothing in the picture changed, and
people are detected separately so a hand passing over the counter never becomes an object.

### The memory rules

The rules that keep answers honest live in `recall/memory.py`:

| What happened | What Recall records |
|---|---|
| First sighting | **appeared** |
| Seen somewhere new for 2 looks in a row | **moved** (one stray frame can't move anything) |
| Missing for 3 looks, last box mostly inside a container's box | **went into** the container, status *inferred*, confidence 0.5 to 0.9 |
| Missing for 3 looks otherwise | **disappeared**, last known place kept |
| A container moves | everything inferred inside it moves too |
| Camera stops sending frames | memory **frozen**: nothing counts as missing, answers say "as of 2:40 PM" |

### The room, in 3D

One camera can't see depth, but a point at a known height can be placed: its pixel is a ray from
the camera, and the ray crosses that height at exactly one spot. Objects are placed where their
box's base ray meets the scanned surfaces (open3d ray casting against the scan). The person is
placed by their shoulders at their known shoulder height, and their facing comes from the line
between the shoulders, with the face deciding front from back.

### The band

An ESP32 with one servo and a button, advertising over Bluetooth LE as `sixsevendemons`. The
laptop sends it text packets, `dir,dist`: degrees to turn (+ is right) and closeness from 0 to
100. The band decides nothing about the world. It points the arrow, wiggles when you've arrived,
and centres itself when packets stop, because a frozen arrow is a lie. The servo only reaches ±90°,
so for things behind you the arrow waves at that side's edge: *turn around, this way*. Pressing
the button asks a question through the laptop's mic, and pressing it again stops listening.

### Voice

Speech-to-text runs on the laptop with faster-whisper (`base.en`, about 0.4 s a question on an
M-series Mac), with Silero VAD to hear when you've stopped talking. In a crowded room, speech only
counts when it's clearly louder than the room's usual level, and the phrases Whisper invents from
noise are never sent as questions. Vague wording ("the thing I open the front door with") is
resolved to an item by Moss semantic search.

## Built with our sponsors

- **Tiger Data:** Recall's memory is time-series data, so every change (moved, disappeared,
  went into the bag) is an event in a TimescaleDB hypertable on Tiger Data. A real-time continuous
  aggregate keeps per-day counts behind the care partner's "This week" panel, and "where do the
  glasses usually go missing?" is one SQL query. In our local benchmark over 900,000 synthetic
  events, compression made the data 83% smaller and a month-long summary ran in 29 ms from the
  aggregate instead of 357 ms from raw events.
- **Moss:** people don't name things the way a database does. Exact item names are matched
  instantly, and vague wording like "the thing I open the front door with" goes to Moss semantic
  search, which works out it means the keys. The page shows how Moss understood the question,
  so a care partner can tell if it heard wrong.
- **Claude (Anthropic):** "Ask about the day" is for the care partner, not the person with
  dementia. Claude answers questions like "did the glasses go missing today?" and writes a short
  daily note. It reads the memory only through tools, cites times, keeps "seen" apart from
  "probably", and says when it doesn't know. The person living with dementia still gets the same
  short, templated, local answers.
- **Cartesia:** everything Recall tells the person is spoken, so the voice matters. Answers and
  directions like "To your left" are spoken in a calm Cartesia Sonic voice. Directions are made
  ahead of time so they play instantly, and if Cartesia is slow or unreachable Recall falls back
  to the built-in voice.

## The database: Tiger Data

Recall uses [Tiger Data](https://www.tigerdata.com) (Tiger Cloud: managed Postgres with
TimescaleDB) as its database. The memory is naturally time-series: an append-only stream of
"the keys moved from the counter to the table at 2:40 PM", and the questions people ask are
about time ("what changed today?", "how often did the glasses go missing this week?").

- **Events** go into a hypertable partitioned on time, segmented by object, so "what changed
  since 2 PM" and per-day summaries stay fast as the history grows, and old chunks compress
  automatically into the columnstore.
- **Current state** (one row per object: place, status, confidence, last seen) and the camera's
  **heartbeat** are upserted alongside, so anyone with access can tell where things are and
  whether the memory is live without reaching the laptop.
- **Photos never leave the laptop.** Only the facts go to the database. The evidence pictures
  stay in `data/photos/`.
- **The laptop keeps working offline.** Writes go through a local copy first and are sent to
  Tiger Data by a background thread that retries and catches up, so a dropped connection never
  stalls the camera loop.

Point Recall at a Tiger Cloud service with its connection string:

    export DATABASE_URL="postgres://tsdbadmin:<password>@<service>.tsdb.cloud.timescale.com:<port>/tsdb?sslmode=require"

## Running it

### From a fresh clone

A clone has no room, no lens calibration and no memory: each setup makes its own (all under
`data/`, which git ignores).

1. **Install** (Python 3.11; open3d has no wheels past 3.12):

       uv venv --python 3.11 --seed .venv
       uv pip install --python .venv/bin/python -r requirements.txt

2. **Markers:** print `printables/markers.pdf` at 100% and put at least 3 up where the webcam will
   see them, on the floor and on walls. Print `printables/calibration_board.pdf` too and mount it
   flat.
3. **Webcam:** mount it high in a corner at its final spot (1280×720).
4. **Scan** with Polycam (Settings → Developer mode on, then LiDAR mode). Pass within about 1 m of
   each marker, facing it. Export **Raw data** and **glTF** and drop both zips into `data/room/`.
5. **Set the room up:** `uv run tools/setup_room.py --camera <webcam name>`. The first time for a
   webcam it opens the lens calibration: hold the board up until the grid fills.
6. **Name the surfaces:** `.venv/bin/python -m recall.surfaces name`.
7. **Run:**

       .venv/bin/python -m recall.live --web --band --set SHOULDER_HEIGHT_M=1.45

   `SHOULDER_HEIGHT_M` is the guided person's shoulder height, about 0.82 × their height. Open the
   printed link on a phone on the same wifi. Add `--voice` to also ask by pressing Enter in that
   terminal. Models download on the first run.

### The band

Flash `firmware/band/band.ino` to an ESP32 with Bluetooth (not the S2), with the ESP32Servo
library. Check it from the laptop with `.venv/bin/python tools/band_test.py sixsevendemons sweep`.
macOS asks to let the terminal use Bluetooth the first time; don't pair the band in System
Settings. A Raspberry Pi band over wifi also works: `--band raspberrypi.local`
(`firmware/pi_band.py`).

### Without any hardware

- **Simulated room:** `.venv/bin/python -m tools.simulate --show` plays a scripted 32-second scene
  (keys moved, a hand over the remote, glasses into the bag, the bag carried away) through the
  real pipeline and memory. `--mode real --set DETECTOR=yoloe` runs the actual model on it.
- **Ask from the terminal:** `.venv/bin/python -m recall.ask "where are my glasses"`, or
  `"what changed since 10m"`, `"history keys"`, `state`.
- **Voice only:** `.venv/bin/python -m recall.voice --memory data/sim/real`.
- **Tests:** `.venv/bin/python -m pytest tests/ -q` (no camera, band or network needed).

### Housekeeping

If the webcam is moved, re-run `uv run tools/register_camera.py --camera <webcam name>` with the
markers still up. To start the memory over, stop live and delete `data/state.json`,
`data/events.jsonl`, `data/heartbeat.json` and `data/photos/`.

## Project layout

| Path | What's in it |
|---|---|
| `recall/live.py`, `pipeline.py`, `camera.py`, `gate.py` | The live loop: capture thread, detection thread, change gate, frozen-camera guard |
| `recall/perception.py`, `enroll.py` | Open-vocabulary detection and the enrolled items (`items.json`) |
| `recall/memory.py`, `store.py`, `query.py`, `ask.py` | The memory rules, storage, and answering questions |
| `recall/room.py`, `surfaces.py`, `places.py` | The scanned room: pixels to world points, named surfaces |
| `recall/body.py`, `guide.py` | Finding the person and which way they face; the direction to the item |
| `recall/web.py` | The web page and its HTTP routes |
| `recall/voice.py`, `moss_search.py` | Asking out loud; semantic matching of vague questions |
| `recall/band.py`, `firmware/` | The band: Bluetooth LE link and ESP32 firmware |
| `tools/` | Room setup (scan, lens, registration), simulation, tuning, range tests |
| `docs/` | Product plan, system design, camera pipeline, setup handoff |

## Privacy

The camera's subject is a surface, not a person. Recall has no concept of *who* moved something,
no face recognition and no identification: the pose model only finds shoulders to point the
arrow. The room scan and the evidence photos stay on the laptop, and the web page never shows or
serves photos. Recall is not a medical device; it diagnoses nothing and promises no safety
outcome.
