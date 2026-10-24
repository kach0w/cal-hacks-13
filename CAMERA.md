# Recall — camera pipeline (CAMERA.md)

The build plan for everything the corner camera does: enroll items, name zones, run the
live loop, keep the memory, and register the camera to the room mesh so positions become
3D. Product context: [PLAN.md](./PLAN.md). System design: [DESIGN.md](./DESIGN.md).

**Scope changes vs DESIGN.md:** **one camera** (no second camera, no cross-camera merge).
The 3D mesh/registration is **added back** (Part E). Positions stay in zone/pixel space until
the mesh handoff lands, then `xyz` fills in through one hook.

---

## 0. Status

| Part | What | Status |
|---|---|---|
| A | Enrollment — the dictionary of tracked items + per-item calibration | **Built** (`recall/enroll.py`, `items.json`) |
| B | Zones — onboarding + manual boxes + `zone_at` | **Built** (`recall/zones.py`) |
| C | Live loop — capture thread, detection thread (gate → detect → memory), frozen-camera guard | **Built** (`recall/live.py`, `pipeline.py`, `camera.py`, `gate.py`, `perception.py`) |
| D | Memory + queries — state, append-only log, stills, where/changes/diff, "as of" stamping | **Built** (`recall/memory.py`, `store.py`, `query.py`, `ask.py`) |
| E | Mesh + camera registration → `pixel_to_world` | **Built against a synthetic room** (`recall/room.py`, `tools/`); waiting on Person A's `data/room/` |
| F | HTTP surface (DESIGN §5) | **To build** (task 4) |

Tests (fake sightings, fake detector, synthetic room — no camera): `python -m pytest tests/ -v`.
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
  camera still ── zones onboard ────▶ data/zones.json  (named boxes in image space)
  phone scan ──── Part E ───────────▶ room.glb + data/camera.json + data/xyz_map.npy

LIVE LOOP
  capture thread    grabs continuously, keeps only the newest frame
  detection thread  every LOOP_S (1 s), on the newest frame:
   → no new frame for STALE_AFTER_S → memory FROZEN (no misses counted), heartbeat camera_ok=false
   → change gate vs last-detected frame
        static            → skip (heartbeat still written)
        change/settling/heartbeat → detector on enrolled prompts (+ "person")
   → one best sighting per item
   → zone_at(base point)                       (+ pixel_to_world(base point) once E lands)
   → memory.update: appeared / moved / disappeared / inferred_contained / reappeared
   → log events; stills to data/stills/; state.json + heartbeat.json
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
data/zones.json       zones, image-space boxes
data/state.json       WorldState, atomic writes
data/events.jsonl     append-only Transitions — the source of truth for history/diff
data/stills/          one annotated JPEG per transition
data/heartbeat.json   {last_frame_ts, camera_ok} — how fresh the memory is ("as of" stamping)
data/replay/          state/events/stills of the last --src video replay (wiped each replay)
data/range_test/      task 0 bursts per station + report/ sheets
data/enroll/<id>/     crops from calibration (from the corner camera)
data/reference/<id>.png  background-removed photos of each prop (committed) — re-id references
data/room/            (E) Person A's delivery, formats in handoffAB.md: room.glb, markers.json,
                      camera.json (K, dist, R, t, …), camera_ref.jpg, checks.json, zones3d.json
                      + xyz_map.npy (H×W×3 world point per pixel, NaN = no hit; built cache)
*.pt, mobileclip*.ts  model weights in the repo root (git-ignored; downloaded once)
```

**World frame (Part E):** meters, **Y-up**, **floor at y = 0**, origin at a taped room
corner, axes along the walls. Every 3D number in the system is in this frame.

**Data shapes** — dataclasses in `recall/models.py`; `to_dict()` is exactly the JSON on disk.
```python
Sighting   = {cam, object_id, bbox[x1,y1,x2,y2], score, ts}
Zone       = {cam, name, box[x1,y1,x2,y2], prep, source}          # prep: "on" | "in" | ...
WorldState = {object_id: {name, zone, status, confidence, since, last_observed,
                          container, last_cam, last_bbox, xyz, still_path,
                          misses, pending_zone, pending_count, context}}   # ObjectState
             # status: observed | inferred | unknown   (never seen = last_observed None)
Transition = {object_id, name, kind, from_zone, to_zone, status, container,
              confidence, ts, time, still_path}
             # kind: appeared | reappeared | moved | disappeared | inferred_contained
Answer     = {text, status, confidence, still_path, as_of}
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
- **No magenta/pink props** — magenta is the band marker colour (DESIGN §7).
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

## 4. Part B — Zones (built)

```bash
python -m recall.zones onboard   # detect furniture, confirm/rename/delete, optional manual boxes
python -m recall.zones manual    # draw boxes only (fallback that always works)
python -m recall.zones show      # overlay on live feed
python -m recall.zones list
```
- An object's zone = the zone containing its **base point** (bottom-centre of its box),
  10 px margin, **smallest zone wins** (a tray on a counter).
- No zone → `None`, rendered as "not in a named area".
- When Part E lands, zones can optionally become 3D boxes on the mesh (point-in-box on
  `xyz`). Keep the image-space version as the fallback.

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
**Threads** (`pipeline.py`, DESIGN §7.1 rule 1): capture keeps only the newest frame; detection
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

**Memory rules** (`memory.py`, DESIGN §6 with these specifics):

| Situation | Rule |
|---|---|
| First sighting | commit `appeared` immediately |
| Seen in a different zone | commit `moved` only after `MOVE_CONFIRM_N` (2) consecutive looks; **+1** if the new zone is `None` (objects in mid-air while carried) |
| Not seen | `misses += 1`; at `MISSING_AFTER_K` (3) → gone |
| Gone, and its last box was ≥ `CONTAIN_OVERLAP` inside an enrolled **container's** box in that same frame | commit `inferred_contained`, status `inferred`, zone = container's zone, confidence clipped 0.5–0.9 |
| Gone, no container | commit `disappeared`, status `unknown`, zone kept as last-known |
| Container moves | every item inferred inside it moves too (`moved`, still inferred; takes the container's `xyz`) |
| Seen again after gone | commit `reappeared`, status `observed`, container cleared |
| Always | never promote `inferred` → `observed` except by actually seeing it |

`CONTAIN_OVERLAP` is **fraction of the object's box inside the container box**, not IoU —
IoU of small glasses vs a big bag is tiny even when they're fully inside.

**Output:** each transition logs an event line (and gets a still); `-v` also logs every
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
| observed, zone | "Your keys are on the kitchen counter — seen just now." |
| observed, no zone | "Your wallet is in view, but not in a named area — seen 2 minutes ago." |
| inferred, container visible | "Your glasses are probably in the bag, on the table — inferred, last seen going in 4 minutes ago." |
| inferred, container gone | "…probably in the bag, but I've lost track of the bag too. They went in at 2:12 PM." |
| unknown | "Last seen on the table at 2:40 PM. I haven't seen them since." |
| never seen | "I don't have a record of that." |

---

## 7. Part E — Mesh + camera registration (built; waiting on Person A's `data/room/`)

Goal: the camera "lives" inside the room model. One function, `pixel_to_world(u, v)`, is
the whole handoff — `live.py` passes it to `Memory(..., pixel_to_world=fn)` as soon as
`data/room/camera.json` exists, and `xyz` fills in. File formats: `handoffAB.md` (on `main`).

**Try the whole chain on a synthetic room** (known ground truth; also what the tests use):
```bash
python -m tools.synthetic_room --out data/room_synthetic       # box room + table, 4 markers, renders
python -m tools.calibrate_intrinsics --room data/room_synthetic --images data/room_synthetic/boards
python -m tools.register_camera --room data/room_synthetic --images data/room_synthetic/frames \
    --test-pixel far_corner_floor=625.4,246.4 --test-pixel mid_floor=725.3,445.7
open data/room_synthetic/check_overlay.jpg                      # mesh edges + markers on the frame
```
Synthetic result: 19/19 checks pass with the *calibrated* K — centre 2.8 cm from the "tape",
rms 0.21 px, floor hits ≤ 0.3 cm, jitter 0.005°.

### E1. Scan → mesh
- **Prep:** demo lighting; clear clutter; cover glass tabletops (LiDAR scans glass badly); tape **3–4 printed ArUco markers** (DICT_4X4_50,
  ~15 cm) on floor/wall where the corner camera sees them; tape-mark the origin corner.
  (handoffAB.md says the markers stay up through the demo; if they come down, re-registration
  after a bump needs them taped back in the same spots.)
- **Scan:** iPhone Pro LiDAR in Polycam or Scaniverse (LiDAR mode) → export `.glb`.
  No LiDAR: phone video → MapAnything (metric) → point cloud → Open3D Poisson → mesh.
- **Clean (Blender/MeshLab):** crop, decimate to ~200k tris, translate/rotate so floor
  y = 0, origin at the taped corner, axes along walls.
- **Accept:** a tape-measured distance (wall to wall) matches the mesh within ~2%.

### E2. Intrinsics — `tools/calibrate_intrinsics.py`
- Lock camera resolution (1280×720), autofocus, exposure, white balance.
- Checkerboard (9×6 inner corners), ~15 views → `cv2.calibrateCamera` → K, dist.
- Emergency fallback: `fx = fy = (W/2)/tan(HFOV/2)`, centre principal point, zero dist.

### E3. Pose — `tools/register_camera.py`
1. **ArUco (primary):** one-time, click each marker's 4 corners on the mesh (Open3D
   `VisualizerWithEditing`) → 3D corners in `data/room/markers.json`. Then detect markers in the
   camera frame (`cv2.aruco`) → `solvePnPRansac` → `solvePnPRefineLM` → R, t.
2. **Manual (fallback):** click 8–12 matching points (room/counter/door corners), spread
   across depth and height, image ↔ mesh, solve the same way.
3. **Tape check:** measure the lens position from the origin corner; camera centre
   `−Rᵀt` must be within ~5 cm.
- **Verify:** project subsampled mesh vertices with `cv2.projectPoints` onto the live frame;
  edges must line up. Mean reprojection error **< 3–5 px**.
- **Output:** `data/room/camera.json` = `{width, height, K, dist, R, t, camera_center, rms_px,
  markers_used, registered_at}` + `camera_ref.jpg` (median frame) + `check_overlay.jpg`.
- Re-registration after a bump = re-run step 1 (no clicking).

```bash
python -m tools.calibrate_intrinsics --room data/room --src 0      # SPACE per board view, ESC solves (≥10)
python -m tools.calibrate_intrinsics --room data/room --hfov 75    # emergency: no checkerboard
python -m tools.register_camera --room data/room --src 0           # ~30 frames averaged; click the 2 test objects
python -m tools.register_camera --room data/room --check-only      # re-run the acceptance table
```
New intrinsics drop the old pose from `camera.json` (it's invalid with a new K). Registration
pools each marker's corners over all frames, solves `solvePnPRansac` → `solvePnPRefineLM`, and
reports per-frame jitter so a bumped or vibrating camera shows up. It then prints the handoff
acceptance table (PASS/FAIL + likely cause): scale (with `--wall-to-wall-x/-z`), rms < 5 px,
centre vs tape < 5 cm, floor_hit / pixel_to_world on `mid_floor` < 5 cm and `far_corner_floor`
< 10 cm, projected test points < 5 px, plus sanity checks (det R = +1, markers in front and
on their printed side, camera upright and inside the walls, mesh Y-up with floor at y = 0,
markers.json corners on the mesh, and the ARKit/GL ↔ OpenCV round trip).

### E4. `recall/room.py` — the bridge
```python
room = Room.load("data/room")             # camera.json (needs R, t) + room.glb (optional)
room.pixel_to_world(u, v) -> (x, y, z) | None   # exact first hit along the pixel's ray (Open3D)
room.floor_hit(u, v)      -> (x, 0, z) | None   # analytic ray ∩ plane y = 0
room.world_to_pixel(x, y, z) -> (u, v)          # cv2.projectPoints, for overlays
room.xyz_map                                    # H×W×3, cached to data/room/xyz_map.npy (+ .sha1)
opencv_to_gl(R, t) / gl_to_opencv(T)            # ARKit/three.js camera axes = OpenCV · diag(1,−1,−1)
```
Pixels are undistorted with a tight criteria (the default 5 iterations are off by pixels at the
image edges) and pixels where the fitted lens model has no inverse return `None` (warned at load).
Without `room.glb` every pixel lands on the floor plane; if Open3D can't read the mesh, the cached
`xyz_map.npy` is used. Rays start 20 cm out, past the webcam's own mount if it was scanned.
A bad `data/room/` never stops the demo: `live.py` warns and runs without `xyz`.

### E5. Viewer (optional, care-partner page)
three.js / React Three Fiber: load `room.glb`, place a virtual camera at R, t with FOV from
K (OpenCV → three.js: multiply camera axes by `diag(1, −1, −1)`), blend the live frame over
the render, pin each object at `xyz`, flash the last-seen spot. Projected video:
`three-projected-material`.

### E6. Sanity check (acceptance for Part E)
- Object **A** on the floor in the **far** corner the camera sees; object **B** mid-floor.
- Tape-measure both (x, z) from the origin corner.
- `floor_hit` on each object's clicked base pixel → B within **~5 cm**, A within **~10 cm**.
- `world_to_pixel` of the tape points lands on the objects within a few px.

| Symptom | Cause |
|---|---|
| Same offset everywhere | mesh origin/axes (E1 clean-up) |
| Error grows with distance | scale or focal length |
| Error grows toward image edges | distortion not applied |
| Mirrored / behind camera | OpenCV vs Y-up axis flip, or a mis-clicked pair |
| Good in middle, bad in corner | too few correspondences on that side |

---

## 8. Build tasks, in order

Nothing starts before the task above it passes its check.

| # | Task | Done when |
|---|---|---|
| 0 | **Range test** (`python -m tools.range_test capture`): camera at its demo position, all props + bag at near / mid / far stations; compares `yolo-world` vs `yoloe` × imgsz 640 / 1280 on the same frames (re-run offline with `eval`) | Every prop seen in ≥ 90% of each burst at every station, right label (check `data/range_test/report/`), within budget → `DETECTOR` + `DETECT_IMGSZ` chosen |
| 1 | **Run A–D on the real camera.** `pip install -r requirements.txt`; detector ms/frame (range test prints it); `enroll calibrate`; `zones onboard`; `live --show` | All props (keys, glasses, remote, wallet, bag) detected with the right label in demo lighting; detector < 150 ms/frame |
| 2 | **Record a demo video** (`python -m tools.record data/demo1.mp4`, script below; refuses to overwrite without `--force`) and replay with `--src` | Event log matches what happened, no phantom moves from a hand passing over |
| 3 | **Tune** `LOOP_S`, `MISSING_AFTER_K`, `MOVE_CONFIRM_N`, `CHANGE_FRACTION`, `CONTAIN_OVERLAP`, per-item `min_score`, prompts on those videos (`python -m tools.tune`, below) | Glasses→bag inferred every run; hand-over-object produces no `disappeared` |
| 4 | **HTTP surface** (`recall/api.py`, FastAPI, **one uvicorn worker**): `GET /state`, `GET /objects/{id}/history`, `GET /changes?since=`, `POST /ask {text}`, `GET /stills/{path}`, `GET /stream` (MJPEG) — thin wrappers over `query.py` | care-partner page and voice layer can call it |
| 5 | **Part E** E1→E4, then E6 | sanity check passes; `state.json` shows `xyz` |
| 6 | **Only if identity swaps:** re-id — embed `data/enroll/` crops (camera view, preferred) and `data/reference/` cutouts (phone photos, starting point) with CLIP via `open_clip`, accept a detection only if cosine ≥ 0.80 to its item, hold 0.60–0.80 | no swaps on the demo video |
| 7 | **Only if time:** E5 viewer | pins on the mesh at the right spots |

Do **not** add: a database (stay on JSON/JSONL), an LLM query layer, a second camera,
evidence clips, alerts. (PLAN §8.)

### Recording script for tasks 2–3

Record **3 takes** of the same script (`python -m tools.record data/demo1.mp4`, `demo2`, `demo3`),
varying hand speed and where the glasses land in the bag — "every run" in task 3 means every take.
After each action take your hands fully out of frame and **wait the stated time** (the memory needs
`MOVE_CONFIRM_N` / `MISSING_AFTER_K` looks at one per second). Zone names below are examples: use
yours from `python -m recall.zones list`.

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

The matching expectation file (`data/demo.expect`, one file for every take; `#` = comment; zone
names must match `zones.json` exactly — the tool rejects unknown items and zones):
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
| `DETECT_IMGSZ` | 640 | task 0 decides 640 vs 1280 |
| `DETECT_DEVICE` | (empty) | empty = `mps` on Apple silicon; or `cpu` |
| `DEFAULT_MIN_SCORE` | 0.15 | open-vocabulary scores run low; calibration overrides per item |
| `MISSING_AFTER_K` | 3 | looks before "gone" |
| `MOVE_CONFIRM_N` | 2 | +1 for the no-zone case |
| `CONTAIN_OVERLAP` | 0.25 | fraction of object box inside container box |
| `ZONE_MARGIN_PX` | 10 | base point may sit this far outside a zone box |
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
  asks for zone names the numbered picture opens in Preview instead (`data/zones_candidates.jpg`).
  q or ESC quits every window; Ctrl-C always closes windows and releases the camera; a quit or
  Ctrl-C mid-calibration keeps earlier items' calibration and the current item's accepted samples.
- **Restart-safe:** state reloads from `state.json`; the log is append-only. Delete
  `data/state.json`, `data/events.jsonl`, `data/stills/` and `data/heartbeat.json` to reset a
  demo (keep `zones.json`, `enroll/`, `room/`).

## 11. Layout

```
items.json
requirements.txt
recall/                      DESIGN §4 component in brackets
  config.py      Settings: every tunable; env RECALL_* and --set NAME=VALUE
  models.py      Item, Sighting, Zone, ObjectState, Transition, Answer (on-disk JSON shapes)
  geometry.py    box math: overlap_frac (fraction inside), base_point, contains
  camera.py      CameraSource (capture thread, newest frame) · VideoSource (exact replay)  [Camera threads]
  gate.py        change gate
  perception.py  Detector interface: yolo-world | yoloe | yolo; re-id hook       [Perception]
  zones.py       zone_at / zone_phrase + onboarding CLI                           [Zone map]
  memory.py      the §5 rules: sightings in, Transitions out                      [Change detection]
  store.py       state.json / events.jsonl / stills / heartbeat.json             [Store]
  query.py       Query: where_is / history / changes_since / state_at / diff; "as of" stamping
  ask.py         text → query routing + REPL (voice and POST /ask call handle())
  pipeline.py    the detection thread: stall check → gate → detect → memory
  overlay.py     all drawing: window, stills, /stream, enroll, zones
  live.py        CLI: wires it together; window on the main thread; clean Ctrl-C
  enroll.py      items.json CLI + calibration
  room.py        (E) camera.json + room.glb → pixel_to_world / floor_hit / world_to_pixel
  api.py         (task 4) FastAPI — to build
tools/
  cameras.py              which index is the corner camera (one frame from each)
  range_test.py           task 0: detectors × imgsz at near/mid/far
  record.py               tasks 2–3: record the camera at a fixed frame rate for --src replays
  simulate.py             a scripted table from the reference cutouts → window, screenshots, test
  tune.py                 task 3: cache detections per take, replay gate + memory over a settings grid
  synthetic_room.py       (E) fake data/room/ from known ground truth (tests)
  calibrate_intrinsics.py (E2) checkerboard → K, dist
  register_camera.py      (E3) ArUco → R, t + the handoff acceptance checks
tests/           python -m pytest tests/   (no camera, no model download)
```
