# Recall — camera pipeline (CAMERA.md)

The build plan for everything the corner camera does: enroll items, name places, run the
live loop, keep the memory, and use the scanned room so positions are 3D. Product context:
[PLAN.md](./PLAN.md). System design: [DESIGN.md](./DESIGN.md) — one corner camera, a phone scan
of the room, positions in the map. File contract between setup and the live loop:
[handoffAB.md](./handoffAB.md).

**Where this follows DESIGN.md and where it adds to it:** the JSON seams are DESIGN §7's
(`object`, `surface`, `from/to_xyz`, `photo_path`); places are DESIGN §6.1's named surfaces of the
scan once the room is registered, with image zones as the fallback before that. The memory rules are
this file's §5 (a move is a *confirmed change of place*, containment is *fraction inside*), which
DESIGN §8 sketches more loosely.

---

## 0. Status

| Part | What | Status |
|---|---|---|
| A | Enrollment — the dictionary of tracked items + per-item calibration | **Built** (`recall/enroll.py`, `items.json`) |
| B | Places — surfaces of the scan (`recall/surfaces.py`, named on its plan) once the room is registered; image zones (`recall/zones.py`) before that | **Built** (`recall/places.py`) |
| C | Live loop — capture thread, detection thread (gate → detect → memory), frozen-camera guard | **Built** (`recall/live.py`, `pipeline.py`, `camera.py`, `gate.py`, `perception.py`) |
| D | Memory + queries — state, append-only log, photos, where/changes/diff, "as of" stamping | **Built** (`recall/memory.py`, `store.py`, `query.py`, `ask.py`) |
| E | Scan → map + markers, lens, registration (Person A's `tools/setup_room.py`); the runtime bridge `pixel_to_world` (`recall/room.py`) | **Delivered** (`data/room/`: room.glb, markers.json, plan.png, camera.json; lens in `data/cameras/`) |
| F | HTTP surface (DESIGN §7) | **To build** (task 4) |

Tests (fake sightings, fake detector, the real scan with a known pose — no camera):
`python -m pytest tests/ -v` (includes Person A's `tests/test_room_tools.py`).
**See it without a camera:** `python -m tools.simulate --show` plays a scripted 32 s scene built
from the reference cutouts (keys moved, a hand over the remote, glasses gone and back, glasses
into the bag, the bag carried away) through the real pipeline and memory, in an OpenCV window;
`--mode real --set DETECTOR=yoloe` runs the actual model on it instead. Screenshots of every pass
+ a contact sheet land in `data/sim/<mode>/`.
The real models have run end to end on composites of the reference photos (§2b), **not yet on the
real camera** — that is tasks 0–1 in §8.

---

## 1. Pipeline

```
SETUP (once)
  items.json  ── enroll calibrate ──▶ per-item min_score + reference crops
  phone scan ──── Part E tools ─────▶ data/room/: room.glb, markers.json, plan.png,
                                       intrinsics.json, camera.json   (DESIGN §1, §7 below)
  plan.png ───── surfaces name ─────▶ data/room/surfaces.json  (named boxes in the room)
  camera still ── zones onboard ────▶ data/zones.json  (image boxes: the fallback places)

LIVE LOOP
  capture thread    grabs continuously, keeps only the newest frame
  detection thread  every LOOP_S (1 s), on the newest frame:
   → no new frame for STALE_AFTER_S → memory FROZEN (no misses counted), heartbeat camera_ok=false
   → change gate vs last-detected frame
        static            → skip (heartbeat still written)
        change/settling/heartbeat → detector on enrolled prompts (+ "person")
   → one best sighting per item
   → xyz = pixel_to_world(box base)   place = surface_at(xyz), or zone_at(box base) before E
   → memory.update: appeared / moved / disappeared / inferred_contained / reappeared
   → log events; photos to data/photos/; state.json + heartbeat.json
  main thread       the --show window (macOS needs it) and Ctrl-C → stop, join, release

QUERIES (separate process, reads files)
  "where is X" · "what changed since T" · "diff T" · "history X"
  heartbeat stale or camera_ok=false → every answer starts "As of <time>:"
```

---

## 2. Conventions and seams

**Files**
```
items.json            tracked-item dictionary (committed)
data/zones.json       image zones: named boxes in the camera image (places before Part E)
data/state.json       WorldState, atomic writes
data/events.jsonl     append-only Transitions — the source of truth for history/diff
data/photos/          one annotated JPEG per transition (photo_path)
data/heartbeat.json   {last_frame_ts, camera_ok} — how fresh the memory is ("as of" stamping)
data/replay/          state/events/photos of the last --src video replay (wiped each replay)
data/range_test/      task 0 bursts per station + report/ sheets
data/enroll/<id>/     crops from calibration (from the corner camera)
data/reference/<id>.png  background-removed photos of each prop (committed) — re-id references
data/cameras/         (E) one lens file per webcam (calibrate_intrinsics), reused in every room
data/room/            (E) committed; formats in handoffAB.md. From Person A's setup_room.py:
                      room.glb, markers.json, plan.png, scan_report.json (scan_to_room),
                      camera.json + camera_ref.jpg + camera_overlay.jpg (register_camera).
                      From recall: surfaces.json. Git-ignored: scan_raw*/ (the export, with the
                      home's GPS position), xyz_map.npy (H×W×3
                      world point per pixel, NaN = no hit; a cache) and the surfaces previews
                      (surfaces_drawn.jpg, surfaces.png, surfaces_camera.jpg)
*.pt, mobileclip*.ts  model weights in the repo root (git-ignored; downloaded once)
```

**World frame (Part E):** meters, **Y-up**, **floor at y = 0**, axes along the walls, origin at the
corner of the scanned floor's outline (found from the scan; `plan.png` shows it). Every 3D number in
the system is in this frame.

**Data shapes** — DESIGN §7's seams, as dataclasses in `recall/models.py`; `to_dict()` is exactly
the JSON on disk. Fields beyond DESIGN §7 are kept for the rules and replay (name, status, …).
```python
Sighting   = {object, bbox[x1,y1,x2,y2], score, ts, cam}           # object = the item id
Surface    = {name, box_min[x,y,z], box_max[x,y,z], prep, top}     # data/room/surfaces.json
             # top: the surface's height (y); a hand-made box without it: box_min's y
Zone       = {name, box[x1,y1,x2,y2], prep, source, cam}           # data/zones.json (fallback)
WorldState = {object: {object, name, xyz, surface, status, confidence, since, last_observed,
                       container, last_cam, last_bbox, photo_path,
                       misses, pending_surface, pending_count, context}}   # ObjectState
             # status: observed | inferred | unknown   (never seen = last_observed None)
             # surface: the place's name (a scanned surface, or an image zone before Part E)
Transition = {object, name, kind, from_surface, to_surface, from_xyz, to_xyz, status,
              container, confidence, ts, time, photo_path}
             # kind: appeared | reappeared | moved | disappeared | inferred_contained
Answer     = {text, status, confidence, photo_path, as_of}
             # as_of: null while the camera is live; else when the world froze
```

---

## 2a. Props, surface and camera placement

| Item | Prop | Prompt | Notes |
|---|---|---|---|
| keys | keys on the **red LOWELL lanyard** | `keys on a red lanyard` | strongest detection; pitch opens with keys |
| glasses | **black-frame** glasses | `black eyeglasses` | goes into the bag in the demo |
| remote | black Samsung TV remote | `black tv remote control` | different shape from glasses, so no swaps |
| wallet | light-blue quilted wallet | `light blue wallet` | **backup item** — weakest contrast on cream lace |
| bag (container) | **not blue** — brown paper bag or yellow/green tote, wide opening | set when chosen | the only container |

- Background-removed photos: `data/reference/{keys,glasses,remote,wallet}.png`. Bag photo still needed.
- **No magenta/pink props** — magenta is the band marker colour (DESIGN §4.2, `MARKER_HSV` in §9).
- **No two props with the same colour or silhouette.** No other lanyards/badges on the demo surface.
- **Surface:** the home table is glass with lace placemats. Lace is fine for contrast but busy;
  bare glass reflects and scans badly. Prefer a plain white/solid cloth over the demo area.
- **Camera:** high in the corner, tilted down 30–45°, wide FOV, aimed at the demo surface, not
  the whole room. Bag opening faces the camera. Lock exposure, white balance and focus.

## 2b. Model choice

| Option | Verdict |
|---|---|
| **YOLO-World `yolov8s-worldv2`** (`DETECTOR=yolo-world`, default) | Open-vocabulary, no training, prompts in `items.json`. **On composites of the reference photos it only finds the remote** (see below) |
| **YOLOE `yoloe-26l-seg`** (`DETECTOR=yoloe`) | Newer open-vocabulary model, same prompts. **The only one that labels all four props on the composites**, 56 ms at 640 on the M4 GPU. Front-runner for task 0 |
| **YOLO11n fine-tuned on these props** (`DETECTOR=yolo DETECT_MODEL=best.pt`) | Fallback if both score low: ~50–100 corner-camera photos per item on the demo surface, auto-label with YOLOE, fix boxes in Roboflow, train ~20 min. Class names = item ids; nothing else changes |
| COCO YOLO11n as-is | Knows remote/handbag/backpack, **not** wallet/keys/glasses |
| ResNet | Classifier only — no boxes, so it can't say *where* |
| SAM / SAM 3 | Overkill: heavy, and masks aren't needed for zones |

Swapping is config only: `--set DETECTOR=yoloe` (`DETECT_MODEL` empty = that backend's default weights).

**Composite check** (reference cutouts pasted on a cream cloth at corner-camera sizes, prompts from
`items.json`, M4 MPS; raw best score on the right box, `→x` = labelled as x). Synthetic — task 0
on the real camera decides.

| Model @ imgsz | ms | keys | glasses | remote | wallet |
|---|---|---|---|---|---|
| yolov8s-worldv2 @640 | 20–33 | 0.03 →person | — | 0.49 | 0.05 →remote |
| yolov8x-worldv2 @640 | 80 | 0.03 | 0.58 | 0.14 | — |
| yoloe-26s-seg @1280 | 79 | 0.70 | 0.51 →keys | 0.30 | 0.03 |
| **yoloe-26l-seg @640** | **56** | **0.79** | **0.42** | **0.48** | **0.17** |
| yoloe-26l-seg @1280 | 208 | 0.79 | 0.34 | 0.47 | 0.24 |

---

## 3. Part A — Enrollment (built)

`items.json` is the dictionary of things worth tracking. Only these are ever committed to
memory; everything else the detector sees is ignored.

```json
"glasses": {"name": "glasses", "prompt": "eyeglasses", "aliases": ["specs"],
            "plural": true, "container": false, "min_score": null,
            "reference": "reference/glasses.png"}
```

- `prompt` — the text YOLO-World looks for. Tune it ("eyeglasses" beats "glasses").
- `aliases` — words a question can use ("where are my specs").
- `plural` — grammar in answers ("are" / "them").
- `container` — things can go inside it. **Containment only ever resolves to containers.**
- `min_score` — set by calibration; `null` falls back to `DEFAULT_MIN_SCORE`.

```bash
python -m recall.enroll list
python -m recall.enroll add bag --name "bag" --prompt "brown paper bag" --container
python -m recall.enroll calibrate            # all items; SPACE accept ×3, n skip, q quit
python -m recall.enroll calibrate glasses --set DETECTOR=yoloe   # one item, another backend
```
Calibration sets `min_score = 0.6 × lowest accepted score` and saves 3 crops per item.
**Re-run in demo lighting at the demo camera position.**

**Identity rule:** identity = which prompt fired, one best box per item per frame. Holds
while every tracked item is one of a kind and visually distinct. If two props swap
identity, change the props first (PLAN §13); embedding re-id against `data/enroll/` + `data/reference/`
is the fallback (§8, task 6).

---

## 4. Part B — Places: surfaces and zones (built)

The household's words for where things are. **One kind is active at a time** (`recall/places.py`);
`live.py` logs which at startup. **Switching kinds** (running on zones, then registering the room and
naming surfaces) leaves the zone names in `state.json`: `live.py` warns, and each object logs a
`moved` to its surface name the next time it's seen. To start clean, reset the memory before the
first run on surfaces: delete `data/state.json`, `data/events.jsonl`, `data/heartbeat.json` and
`data/photos/` (§10).

**Surfaces (DESIGN §6.1)** — once the room is registered (`data/room/camera.json`) and named:
```bash
python -m recall.surfaces name   # drag boxes on data/room/plan.png, name them in the terminal
python -m recall.surfaces show   # surfaces.png (the plan) + surfaces_camera.jpg (camera_ref.jpg)
python -m recall.surfaces list
```
- A surface is a box in the world frame: the dragged rectangle on the plan, from 10 cm under the
  scan's height there to 60 cm over it. The height (`top`, saved with the box) is the busiest level
  under the rectangle, looking down as the plan is drawn; a raised level holding ≥ 25% of it beats
  the floor, so a rectangle dragged a little loose around a table still gets the table's height.
  The terminal shows each top ("top at 0.35 m", "(the floor)") before you name it.
- An object's place = the surface holding `xyz` (the map point under its box's base), 5 cm margin.
  Where boxes overlap, the **highest surface wins** (the sofa seat, not the floor box dragged into
  it), then, among surfaces within 10 cm of that height, the **smallest** (a tray on a counter).
  No map point, or in no box → `None`.
- Check the boxes with `show`: on `surfaces_camera.jpg` each top outline must sit on its furniture,
  with its name inside it. Outlines are drawn only where the lens model holds (no further off-axis
  than the image corners), so a surface out of view never shows up at the image edge.

**Image zones** — before Part E, or with no `surfaces.json`:
```bash
python -m recall.zones onboard   # detect furniture, confirm/rename/delete, optional manual boxes
python -m recall.zones manual    # draw boxes only (fallback that always works)
python -m recall.zones show      # overlay on live feed
python -m recall.zones list
```
- An object's zone = the zone containing its **base point** (bottom-centre of its box),
  10 px margin, **smallest zone wins**.

No place → `None`, rendered as "not in a named area". Names must be unique (answers find them by
name); the preposition ("on", "in") words the answers.

---

## 5. Part C — Live loop (built)

```bash
python -m recall.live --show                      # camera 0 + overlay window (q quits)
python -m recall.live --src 1 --show              # another camera index
python -m recall.live --src demo.mp4 --show       # replay a recording into data/replay/ (wiped first)
python -m recall.live --src demo.mp4 --show --set REPLAY_SPEED=1   # …at real speed, to watch it
python -m recall.live --src demo.mp4 --snapshots data/shots        # every pass's overlay as a jpg
python -m recall.live --src demo.mp4 -v --set CONTAIN_OVERLAP=0.2 --set MISSING_AFTER_K=4
                                                  # tune: -v logs every pass with its change fraction
```
**Threads** (`pipeline.py`, DESIGN §9.1 rule 1): capture keeps only the newest frame; detection
runs on its own thread once per `LOOP_S` and never blocks capture; the main thread only draws
the window, which shows the boxes coloured by memory status (green observed, amber inferred,
red unknown) and a panel of what the memory believes per item. Ctrl-C (or SIGTERM) stops detection after its current pass, joins the capture
thread, then releases the camera. A replay runs as fast as detection allows; its frames carry
the time they were shot (the file's mtime minus its length), so runs are repeatable.

**Frozen camera:** no new frame for `STALE_AFTER_S` → no detection and **no misses counted**
(an unplugged camera must not make everything "disappear"), the window shows CAMERA STALLED,
`heartbeat.json` says `camera_ok: false`, and every answer is stamped "As of <last frame>:".
It unfreezes on the next new frame.

**Change gate** (`gate.py`): 160-px-wide blurred grayscale diff against the **last frame
detection ran on**.
- ≥ `CHANGE_FRACTION` of pixels changed by ≥ `CHANGE_PIXEL_DELTA` → detect, and stay "hot"
  for `SETTLE_S`, detecting every loop. Without the settle window a vanished object gets one
  miss and never reaches `MISSING_AFTER_K`.
- Heartbeat detection every `HEARTBEAT_S` regardless.

**Detection** (`perception.py`): the `DETECTOR` backend (§2b) prompted with the item prompts +
`"person"`. Person boxes are reported, never tracked. `DETECT_DEVICE` empty = the Apple GPU (MPS)
when present — ultralytics never picks it by itself (20 vs 69 ms for YOLO-World @640 on the M4).

**Memory rules** (`memory.py`; DESIGN §8, with these specifics):

| Situation | Rule |
|---|---|
| First sighting | commit `appeared` immediately |
| Seen in a different place | commit `moved` only after `MOVE_CONFIRM_N` (2) consecutive looks; **+1** if the new place is `None` (objects in mid-air while carried) |
| Not seen | `misses += 1`; at `MISSING_AFTER_K` (3) → gone |
| Gone, and its last box was ≥ `CONTAIN_OVERLAP` inside an enrolled **container's** box in that same frame | commit `inferred_contained`, status `inferred`, place and `xyz` = the container's, confidence clipped 0.5–0.9 |
| Gone, no container | commit `disappeared`, status `unknown`, place and `xyz` kept as last-known |
| Container moves | every item inferred inside it moves too (`moved`, still inferred; takes the container's `xyz`) |
| Seen again after gone | commit `reappeared`, status `observed`, container cleared |
| Always | never promote `inferred` → `observed` except by actually seeing it |

`CONTAIN_OVERLAP` is **fraction of the object's box inside the container box**, not IoU (DESIGN §9
still says IoU) — IoU of small glasses vs a big bag is tiny even when they're fully inside.
Containment stays a 2D question in the pixels of the last frame, as DESIGN §8 says.

**Positions:** `xyz` is the map point under the box's base (`pixel_to_world`, DESIGN §4.1). It follows
the committed belief: while a move waits for confirmation the stored `xyz` stays put, and a
transition carries both ends (`from_xyz`, `to_xyz`).

**Output:** each transition logs an event line (and gets a photo); `-v` also logs every
detection pass (`[change 0.012] seen: keys@kitchen counter 0.41, ...`). The state table is
logged on exit.

---

## 6. Part D — Memory and queries (built)

```bash
python -m recall.ask                          # REPL
python -m recall.ask "where are my glasses"
python -m recall.ask "what changed since 10m"     # also: today, 2h, 14:30, 2:40pm
python -m recall.ask "diff 14:30"
python -m recall.ask "history keys"
python -m recall.ask state
python -m recall.ask --replay "history glasses"  # about the last --src replay (data/replay/)
```
Queries read `state.json` / `events.jsonl` fresh, so they run alongside `live.py`.
`state_at(T)` replays the log; `diff(T)` compares that to now. `ask.handle(text, query)` →
`Answer` is what the voice layer and `POST /ask` call.

**Never answer from a frozen world as if it were live:** if `heartbeat.json` is older than
`STALE_AFTER_S` or says `camera_ok: false` (camera stalled, or `live.py` isn't running), every
answer — where/changes/diff/history/state — starts "As of 2:41 PM:" and `Answer.as_of` is set.

Answer templates (`Query.where_is`):

| Status | Answer |
|---|---|
| observed, a place | "Your keys are on the kitchen counter — seen just now." |
| observed, no place | "Your wallet is in view, but not in a named area — seen 2 minutes ago." |
| inferred, container visible | "Your glasses are probably in the bag, on the table — inferred, last seen going in 4 minutes ago." |
| inferred, container gone | "…probably in the bag, but I've lost track of the bag too. They went in at 2:12 PM." |
| unknown | "Last seen on the table at 2:40 PM. I haven't seen them since." |
| never seen | "I don't have a record of that." |

---

## 7. Part E — The scanned room (Person A's setup tools + the runtime bridge)

The camera "lives" inside the scan. Setup (DESIGN §1, done once by Person A / the care partner)
writes `data/room/`; at runtime `recall/room.py` reads it and gives the live loop one function,
`pixel_to_world(u, v)`, so every sighting gets an `xyz` and a surface. File formats: handoffAB.md.
The setup tools are `uv` scripts with their own environment (Python ≤ 3.12, open3d 0.19, trimesh);
the shared geometry and conventions live in `tools/roomlib.py`, and what's printed in
`tools/printed.py` (markers: ArUco `DICT_4X4_50` ids 0–5, 18 cm nominal; lens board: ChArUco
`DICT_5X5_100`, 7×5 squares of 3.5 cm).

**One command** (DESIGN §1): markers up, the Polycam export zip(s) dropped into `data/room/`, then
`uv run tools/setup_room.py --camera <name>`. It runs E1–E3 below and prints its checks (E4). The
three tools still run on their own: re-run `register_camera.py` alone after the camera is bumped.

**State of `data/room/` (3ddesign):**
- `room.glb` (56k triangles), `markers.json` (markers 0, 3, 4, 5 on the floor and two walls,
  16.8 cm read from the scan), `plan.png`, `scan_report.json`. World frame found from the scan
  (no clicks): floor at y = 0 (floor markers 3 and 4 sit within 1.3 cm of it), X and Z along the
  walls, origin at the corner of the floor's outline. `scan_report.json` still notes that markers
  0, 3 and 4's corners spread 20, 24 and 36 mm across sightings (Polycam's drift between passes).
- Lens: `data/cameras/brio-101.json` — fx 1348, fy 1354, centre (619, 311), 0.85 px over 49
  views (the first calibration's `cy = −249` is gone). Reused in every room.
- `camera.json` + `camera_ref.jpg` + `camera_overlay.jpg`: registered from markers 0, 4, 5
  (marker 3 is under the glass table) at 1.26 px; the webcam is at (−2.52, 2.58, 4.44), past the
  scanned floor. With 3 markers in view, floor positions are good to roughly 5–8 cm; another
  marker or two in its view (then a rescan) tightens it.

### E1. Scan → map + markers (`tools/scan_to_room.py`)
The export zip(s) in `data/room/` are unpacked into `scan_raw/` (git-ignored: it holds the home's
GPS position). The tool checks the export against its own LiDAR depth, finds the markers in the
keyframes and casts their corners onto the mesh, sets the world frame from the scan alone, and
writes `room.glb`, `markers.json`, `plan.png` (top-down, 100 px/m, origin and axes drawn) and
`scan_report.json`.

**Fallback when it stops at the depth check** (`FAIL: no reading lines the mesh up with LiDAR
depth`): the markers can't be found in the keyframes, so their corners would have to be clicked on
the mesh by hand, in OpenCV's order (top-left, top-right, bottom-right, bottom-left, looking at the
marker upright), into handoffAB.md §2's format. No tool does this; it hasn't been needed.

### E2. Lens (`tools/calibrate_intrinsics.py`)
Once per webcam, into `data/cameras/<webcam>.json`. `setup_room.py` runs it the first time it
meets a webcam: hold `printables/calibration_board.pdf` (ChArUco) 40–60 cm away until the
on-screen 3×3 grid has filled and a few shots are tilted; it finishes by itself. Redo
(`--recalibrate`) if resolution, focus or zoom change.

### E3. Pose (`tools/register_camera.py`)
Watches the markers ~2 s, matches their corners to `markers.json`, `solvePnPRansac` +
`solvePnPRefineLM` → `camera.json` (`{width, height, K, dist, R, t, camera_center, rms_px,
markers_used, registered_at}`), `camera_ref.jpg` and `camera_overlay.jpg` (the markers and the mesh
drawn where the pose puts them: they must sit on the real ones). **Camera bumped:** re-run it —
the markers are still up.

### E4. Checks (automatic)
No tape measurements. `setup_room.py` reports: the mesh against the phone's LiDAR depth (< 2 cm),
floor markers at y = 0, the camera's fit to the markers, and how many markers it sees. The
overlay is the eyeball check.

| If positions look off | Likely cause |
|---|---|
| Same offset everywhere | mesh origin / axes |
| Error grows with distance | scale, or the focal length in K |
| Error grows toward the image edges | distortion not applied |
| Mirrored, or the camera behind a wall | marker corner order, or the OpenCV vs Y-up flip |
| Good in the middle, bad in a corner | too few markers on that side |

### E5. Surfaces — `python -m recall.surfaces name` (§4)

### E6. `recall/room.py` — the runtime bridge (handoffAB.md "What B does")
```python
room = Room.load("data/room")             # camera.json (needs R, t) + room.glb (optional)
room.pixel_to_world(u, v) -> (x, y, z) | None   # exact first hit along the pixel's ray (Open3D)
room.floor_hit(u, v)      -> (x, 0, z) | None   # analytic ray ∩ plane y = 0
room.world_to_pixel(x, y, z) -> (u, v)          # cv2.projectPoints, for overlays
room.xyz_map                                    # H×W×3, cached to data/room/xyz_map.npy (+ .sha1)
opencv_to_gl(R, t) / gl_to_opencv(T)            # ARKit/three.js camera axes = OpenCV · diag(1,−1,−1)
```
Reads the mesh the way the setup tools do (trimesh, nodes merged, transforms baked) and uses
`roomlib`'s conventions; `tests/test_room.py` checks it against `roomlib.project`, `floor_hit` and
`solve_pose` on the real scan. Pixels are undistorted with a tight criteria (the default 5
iterations are off at the image edges). Without `room.glb` (or without trimesh to read it)
everything lands on the floor plane: `xyz` stays on (`source=floor` in the log) but places fall
back to image zones, since every surface would be wrong. A bad `data/room/` never stops the demo —
`live.py` warns and runs with image zones and no `xyz`. So does a `camera.json` registered at
another size than `CAPTURE_W`×`CAPTURE_H` (logged as an error): the lens calibration only holds at
its own size (DESIGN §9), so capture at that size, or calibrate and register again (E2, E3).

### E7. Viewer (optional, care-partner page)
three.js / React Three Fiber: load `room.glb`, place a virtual camera at R, t with FOV from
K (OpenCV → three.js: multiply camera axes by `diag(1, −1, −1)`), blend the live frame over
the render, pin each object at `xyz`, flash the last-seen spot. Projected video:
`three-projected-material`.

---

## 8. Build tasks, in order

Nothing starts before the task above it passes its check.

| # | Task | Done when |
|---|---|---|
| 0 | **Range test** (`python -m tools.range_test capture`): camera at its demo position, all props + bag at near / mid / far stations; compares `yolo-world` vs `yoloe` × imgsz 640 / 1280 on the same frames (re-run offline with `eval`) | Every prop seen in ≥ 90% of each burst at every station, right label (check `data/range_test/report/`), within budget → `DETECTOR` + `DETECT_IMGSZ` chosen |
| 1 | **Run A–D on the real camera.** `pip install -r requirements.txt`; detector ms/frame (range test prints it); `enroll calibrate`; places: `surfaces name` once the room is registered, else `zones onboard`; `live --show` | All props (keys, glasses, remote, wallet, bag) detected with the right label in demo lighting; detector < 150 ms/frame |
| 2 | **Record a demo video** (`python -m tools.record data/demo1.mp4`, script below; refuses to overwrite without `--force`) and replay with `--src` | Event log matches what happened, no phantom moves from a hand passing over |
| 3 | **Tune** `LOOP_S`, `MISSING_AFTER_K`, `MOVE_CONFIRM_N`, `CHANGE_FRACTION`, `CONTAIN_OVERLAP`, per-item `min_score`, prompts on those videos (`python -m tools.tune`, below) | Glasses→bag inferred every run; hand-over-object produces no `disappeared` |
| 4 | **HTTP surface** (`recall/api.py`, FastAPI, **one uvicorn worker**), DESIGN §7: `GET /state`, `GET /changes?since=today`, `POST /ask {text}`, `GET /stream/{cam}` (MJPEG) — thin wrappers over `query.py`. (`GET /proximity` belongs to the wrist tracker, DESIGN §4.2. Open: DESIGN lists no route for the photos behind `photo_path`.) | care-partner page and voice layer can call it |
| 5 | **Part E** (Person A): `setup_room.py` (E1–E4, done); then E5 surfaces (reset the memory first, §4) | the overlay sits on the room; `state.json` shows `xyz` and surface names |
| 6 | **Only if identity swaps:** re-id — embed `data/enroll/` crops (camera view, preferred) and `data/reference/` cutouts (phone photos, starting point) with CLIP via `open_clip`, accept a detection only if cosine ≥ 0.80 to its item, hold 0.60–0.80 | no swaps on the demo video |
| 7 | **Only if time:** E7 viewer | pins on the mesh at the right spots |

Do **not** add: a database (stay on JSON/JSONL), an LLM query layer, a second camera,
evidence clips, alerts. (PLAN §8.)

### Recording script for tasks 2–3

Record **3 takes** of the same script (`python -m tools.record data/demo1.mp4`, `demo2`, `demo3`),
varying hand speed and where the glasses land in the bag — "every run" in task 3 means every take.
After each action take your hands fully out of frame and **wait the stated time** (the memory needs
`MOVE_CONFIRM_N` / `MISSING_AFTER_K` looks at one per second). Place names below are examples: use
yours from `python -m recall.surfaces list` (room registered) or `python -m recall.zones list`.

| Step | Action | Wait | Expected event |
|---|---|---|---|
| 0 | Empty surface, start recording | 3 s | — |
| 1 | Place keys, glasses, remote, wallet, bag on the **table**, one at a time | 2 s each | `appeared` ×5 |
| 2 | Carry the keys to the **counter** | 5 s | `moved keys counter` |
| 3 | Sweep a hand slowly over the remote and wallet (~1 s), withdraw | 5 s | **none** (the phantom check) |
| 4 | Hold the glasses **over the bag's opening for ~1 s**, then drop them in | 6 s | `inferred_contained glasses` |
| 5 | Carry the bag to the **counter** | 6 s | `moved bag counter`, then `moved glasses counter` |
| 6 | Take the keys out of frame | 6 s | `disappeared keys` |
| 7 | Stop recording (q) | — | — |

**Containment risk (measured on the simulated table, ground-truth boxes):** the guess uses the *last
frame the glasses were seen in*, and the loop looks once per `LOOP_S`. With the glasses over the bag
for only 0.3 s, `LOOP_S=1` logs `disappeared` instead of `inferred_contained` for **every** K / N /
SETTLE combination tried (0 of 32 pass); `LOOP_S=0.5` passed all 32 (but only because a look happened
to land in the 0.3 s window); `LOOP_S=0.25` passes once `MISSING_AFTER_K ≥ 6` — at 4 looks a second the
1 s hand pass is 4 looks, so K=3 turns it into a phantom `disappeared`. K and N count **looks**, not
seconds: change them together with `LOOP_S`, and keep `SETTLE_S > MISSING_AFTER_K × LOOP_S`. Record
at least one take with a fast drop; real detector flicker will move these numbers.

**Tuning (task 3)** — detect once per take, then replay the gate + memory under a grid of
settings in milliseconds, exactly as `recall.live --src` would (`tools/tune.py`):
```bash
for s in 1 0.5 0.25; do python -m tools.tune cache data/demo*.mp4 --set DETECTOR=yoloe --set LOOP_S=$s; done
python -m tools.tune run data/demo1.mp4 data/demo2.mp4 data/demo3.mp4 --expect data/demo.expect \
    --set DETECTOR=yoloe --grid LOOP_S=1,0.5,0.25 --grid MISSING_AFTER_K=3,4,6,8 \
    --grid MOVE_CONFIRM_N=2,3,4 --grid CONTAIN_OVERLAP=0.15,0.25,0.35 --grid MIN_SCORE.wallet=0.1,0.15,0.2
```
It prints one row per combination (missing/phantom events per take), the best row's diff against the
expectation, and the `--set` flags to use live. Without `--expect` it prints each take's events in
expect-file format, ready to check and save.

The matching expectation file (`data/demo.expect`, one file for every take; `#` = comment; place
names must match `surfaces.json` / `zones.json` exactly — the tool rejects unknown items and places):
```
appeared keys table
appeared glasses table
appeared remote table
appeared wallet table
appeared bag table
moved keys counter
# step 3: the hand pass must commit nothing
inferred_contained glasses
moved bag counter
moved glasses counter
disappeared keys
```

---

## 9. Parameters (`recall/config.py`)

One `Settings` object. Override any of these per run with `--set NAME=VALUE` (repeatable,
case-insensitive) or an environment variable `RECALL_NAME=VALUE`; precedence defaults < env <
`--set` < explicit flags like `--src`. A typo is an error, not a silent default.

| Name | Start | Notes |
|---|---|---|
| `DATA_DIR` / `ITEMS_PATH` | `data/` / `items.json` | point at a scratch dir to experiment |
| `SRC` | 0 | camera index or video path (`--src`) |
| `CAPTURE_W` / `CAPTURE_H` | 1280 / 720 | |
| `LOOP_S` | 1.0 | one look per second |
| `STALE_AFTER_S` | 5 | no new frame this long → memory frozen, answers "As of …" |
| `REPLAY_SPEED` | 0 | `--src video` replays: 0 = as fast as detection allows, 1 = real time (to watch with `--show`) |
| `CHANGE_WIDTH` / `CHANGE_PIXEL_DELTA` / `CHANGE_FRACTION` | 160 / 25 / 0.004 | raise FRACTION if auto-exposure flicker triggers detection |
| `SETTLE_S` / `HEARTBEAT_S` | 5 / 20 s | SETTLE must exceed `MISSING_AFTER_K × LOOP_S` |
| `DETECTOR` | yolo-world | yolo-world \| yoloe \| yolo (fine-tuned) — §2b |
| `DETECT_MODEL` | (empty) | empty = the backend's default weights; required for `yolo` |
| `DETECT_IMGSZ` | 1280 | DESIGN §9: 640 loses small objects past ~2 m; task 0 confirms (YOLOE @1280 is ~210 ms) |
| `DETECT_DEVICE` | (empty) | empty = `mps` on Apple silicon; or `cpu` |
| `DEFAULT_MIN_SCORE` | 0.15 | open-vocabulary scores run low; calibration overrides per item |
| `MISSING_AFTER_K` | 3 | looks before "gone" |
| `MOVE_CONFIRM_N` | 2 | +1 for the no-place case |
| `CONTAIN_OVERLAP` | 0.25 | fraction of object box inside container box |
| `ZONE_MARGIN_PX` | 10 | base point may sit this far outside a zone box (image zones) |
| `SURFACE_MARGIN_M` | 0.05 | `xyz` may sit this far outside a surface box (scanned surfaces) |
| `REID_ACCEPT` / `REID_HOLD` | 0.80 / 0.60 | task 6 only, when an identity check is plugged in |

---

## 10. Gotchas

- **First run downloads** the detector weights and their text encoder (CLIP ~340 MB for
  YOLO-World, MobileCLIP ~250 MB for YOLOE) — do it on good wifi before the demo, not at the table.
  Weights land in the repo root (git-ignored); copy them to the demo machine.
- **python.org Python on macOS has no CA bundle**: downloads fail with
  `CERTIFICATE_VERIFY_FAILED`. Run `export SSL_CERT_FILE=$(.venv/bin/python -m certifi)` first (or
  "Install Certificates.command" in `/Applications/Python 3.11/`).
- **Python 3.11 venv** (`python3.11 -m venv .venv`): open3d (Part E) has no wheels for 3.13+/3.14.
- **Two environments:** `recall` and `python -m tools.*` run in `.venv`; Person A's setup tools
  (`setup_room`, `scan_to_room`, `calibrate_intrinsics`, `register_camera`) are `uv run` scripts that
  build their own (Python ≤ 3.12, open3d 0.19, trimesh). Their test also runs under the venv's pytest.
- **One OpenCV wheel only** — `opencv-python` (ArUco is in core since 4.7). Having
  `opencv-contrib-python` installed too makes them overwrite each other's `cv2`.
- **macOS:** grant the terminal camera permission; OpenCV windows must run on the main thread
  (they do).
- **Lock exposure and white balance** on the webcam, or the change gate fires constantly and
  detection scores drift.
- **Mount the bag opening toward the camera** so "glasses going in" is actually visible.
- **A hand on an object** is excluded from containment by construction — only enrolled
  containers count, and people are never enrolled.
- **Interactive tools on macOS:** OpenCV windows only repaint inside `waitKey`, so while the terminal
  asks for zone or surface names the numbered picture opens in Preview instead
  (`data/zones_candidates.jpg`, `data/room/surfaces_drawn.jpg`).
  q or ESC quits every window; Ctrl-C always closes windows and releases the camera; a quit or
  Ctrl-C mid-calibration keeps earlier items' calibration and the current item's accepted samples.
- **Restart-safe:** state reloads from `state.json`; the log is append-only. Files written before
  the rename to DESIGN §7's names (`object_id`, `zone`, `still_path`, `stills/`) are read as the new
  ones. Delete `data/state.json`, `data/events.jsonl`, `data/photos/` and `data/heartbeat.json` to
  reset a demo (keep `zones.json`, `enroll/`, `room/`); also before the first run on surfaces (§4).

## 11. Layout

```
items.json
requirements.txt
recall/                      DESIGN §6 component in brackets
  config.py      Settings: every tunable; env RECALL_* and --set NAME=VALUE
  models.py      DESIGN §7 seams: Item, Sighting, Surface, Zone, ObjectState, Transition, Answer
  geometry.py    box math: overlap_frac (fraction inside), base_point, contains
  camera.py      CameraSource (capture thread, newest frame) · VideoSource (exact replay)  [Camera threads]
  gate.py        change gate
  perception.py  Detector interface: yolo-world | yoloe | yolo; re-id hook       [Perception]
  places.py      Places: surface_at(xyz) once registered, else zone_at(bbox); phrases   [Surfaces]
  surfaces.py    surfaces.json + naming them on the scan's plan.png (CLI)
  zones.py       image zones + onboarding CLI (the fallback places)
  memory.py      the §5 rules: sightings in, Transitions out                      [Change detection]
  store.py       state.json / events.jsonl / photos / heartbeat.json             [Store]
  query.py       Query: where_is / history / changes_since / state_at / diff; "as of" stamping
  ask.py         text → query routing + REPL (voice and POST /ask call handle())
  pipeline.py    the detection thread: stall check → gate → detect → memory
  overlay.py     all drawing: window, photos, /stream, enroll, zones, surfaces
  live.py        CLI: wires it together; window on the main thread; clean Ctrl-C
  enroll.py      items.json CLI + calibration
  room.py        (E) data/room/ → pixel_to_world / floor_hit / world_to_pixel   [Localizer]
  api.py         (task 4) FastAPI — to build
tools/           Part E setup (Person A; `uv run tools/<name>.py`):
  setup_room.py           (E1-E4) the one command: scan, lens, pose, checks
  scan_to_room.py         (E1) Polycam export zip(s) -> room.glb, markers.json, plan.png
  calibrate_intrinsics.py (E2) ChArUco board -> data/cameras/<webcam>.json
  register_camera.py      (E3) markers -> camera.json, camera_ref.jpg, camera_overlay.jpg
  roomlib.py, printed.py, webcam.py   shared geometry, what's printed, opening a camera by name
                 Session tools (`python -m tools.<name>`):
  cameras.py              which index is the corner camera (one frame from each)
  range_test.py           task 0: detectors × imgsz at near/mid/far
  record.py               tasks 2–3: record the camera at a fixed frame rate for --src replays
  simulate.py             a scripted table from the reference cutouts → window, screenshots, test
  tune.py                 task 3: cache detections per take, replay gate + memory over a settings grid
scripts/         make_printables.py (markers + ChArUco board PDFs), range_test.py (walk-back hit rate)
printables/      markers.pdf, calibration_board.pdf
tests/           python -m pytest tests/   (no camera, no model download; test_room_tools.py is Person A's)
```
