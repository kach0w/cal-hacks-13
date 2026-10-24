# Camera session runbook (CAMERA.md §8, tasks 0–3)

For whoever runs the camera tasks. CAMERA.md is the source of truth; this is the order to do
things in, what to do with your hands, and what to send back. Run every command from the repo
root (model weights live there). Times are rough.

## Setup (once, ~15 min, on good wifi)

1. Python **3.11** (open3d has no wheels for 3.13+):
   `python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt`
2. `export SSL_CERT_FILE=$(.venv/bin/python -m certifi)` — python.org's Python can't verify
   downloads without it (`CERTIFICATE_VERIFY_FAILED`). Do this in every new terminal that downloads.
3. Download both detectors and their text encoders (~1 GB) and check they run, on a simulated table:
   `.venv/bin/python -m tools.simulate --mode real --set DETECTOR=yoloe`
   `.venv/bin/python -m tools.simulate --mode real --set DETECTOR=yolo-world`
4. `.venv/bin/python -m pytest tests/ -q` → everything passes (~35 s).
5. `.venv/bin/python -m tools.simulate --show` → a window plays a 32 s scripted scene with boxes.
   If no window appears, check behind the terminal.
6. macOS: System Settings → Privacy & Security → Camera → allow your terminal app.

Below, `python` means `.venv/bin/python` (or `source .venv/bin/activate` first).

## The room (before task 0)

1. Demo lighting on. A plain cloth over the glass table. No other lanyards, badges, magenta or
   pink things in view.
2. Camera **high in a corner**, tilted down 30–45°, aimed at the demo surface. If it has a settings
   app, turn off auto exposure, auto white balance and autofocus.
3. Props: keys on the red lanyard, black glasses, black remote, light-blue wallet, the bag
   (opening toward the camera).

## Task 0 — range test (~15 min)

1. `python -m tools.cameras` → a window with one tile per camera. Note the `--src N` of the corner
   camera's view; any key closes it.
2. Tape-measure from the lens to three spots: **near** (the closest edge of the demo area),
   **mid** (its centre), **far** (the farthest spot a prop will ever be).
3. `python -m tools.range_test capture --src N --stations near=1.2,mid=2.0,far=2.8` (your numbers).
4. At each station: lay out all props ~15 cm apart, not touching (keys loosely coiled, glasses
   folded, remote and wallet face up, bag upright). Step out of frame, press **SPACE**, keep still
   ~2 s. (n skips a station, q quits.)
5. It then compares both detectors at 640 and 1280 (~1 min) and prints a table and a
   `recommend: --set DETECTOR=… --set DETECT_IMGSZ=…` line.

**Done when** every prop is seen in ≥ 90% of each burst at every station, with the right label
(check the pictures in `data/range_test/report/`), under 150 ms. **Send back:** the printed table and
`data/range_test/report/`.

From now on, in every terminal (with your index and the recommendation):
```bash
export RECALL_SRC=N RECALL_DETECTOR=yoloe RECALL_DETECT_IMGSZ=640
```

## Task 1 — calibrate, zones, live (~20 min)

1. Tell the detector what the bag is: `python -m recall.enroll add bag --prompt "brown paper bag"`
   (describe the real bag; this keeps its other settings).
2. `python -m recall.enroll calibrate` — one item at a time, in the middle of the surface. When the
   green box sits on the item press **SPACE**; move or turn it a little; SPACE again — 3 times per
   item. n = next item, q = quit (accepted samples are kept).
3. `python -m recall.zones onboard` — a numbered picture opens in Preview; answer in the terminal:
   ENTER keeps a box, typing a name renames it ("kitchen counter"), `d` deletes it; then its
   preposition (on/in). Answer `y` to draw more: drag a box, **SPACE** to keep it, ESC when done,
   then name each. Check with `python -m recall.zones list`.
4. `python -m recall.live --show` — boxes are green when seen; the bottom-left panel is what the
   memory believes. Try it: carry the keys to another zone and wait 3 s (terminal: `MOVED`); hide the
   glasses and wait 4 s (`DISAPPEARED`); put them back (`REAPPEARED`). q quits.
5. In a second terminal: `python -m recall.ask "where are my keys"`.

**Done when** all five props are detected with the right label in demo lighting.
**Send back:** the terminal log, `python -m recall.enroll list`, `python -m recall.zones list`.

## Task 2 — record (~15 min)

Follow **"Recording script for tasks 2–3"** in CAMERA.md §8 — the steps, the waits and the expected
events are there. Record 3 normal takes and 1 fast take (glasses scooped straight into the bag):
```bash
python -m tools.record data/demo1.mp4      # q stops; refuses to overwrite (add --force to redo)
```
Check a take: `python -m recall.live --src data/demo1.mp4 --show --set REPLAY_SPEED=1`.
Copy the expectation template from CAMERA.md into `data/demo.expect`, with your zone names from
`python -m recall.zones list`.

## Task 3 — tune (~10 min, no camera needed)

Run the two `tools.tune` commands in CAMERA.md §8 ("Tuning"). **Done when** the best row scores
0 missing / 0 phantom on every take, fast take included. **Send back:** the tune output (table,
the best row's diff, the `recommend:` lines) and the four videos.

## If something goes wrong

| You see | Do |
|---|---|
| `camera N did not open` | wrong index (`tools.cameras`) or no camera permission (setup step 6) |
| `frames are 640x480, not ... 1280x720` | `export RECALL_CAPTURE_W=640 RECALL_CAPTURE_H=480` (use what it says), then redo zones |
| a window looks frozen | it's waiting for an answer in the terminal |
| `CERTIFICATE_VERIFY_FAILED` | setup step 2 |
| a prop never gets a box in calibrate | n to skip it, finish the rest, report the prop |
| anything else | send the whole terminal output |
