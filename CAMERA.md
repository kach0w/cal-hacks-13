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
| C | Live loop — change gate → detect → zone → match → memory → print | **Built** (`recall/live.py`, `change.py`, `perception.py`) |
| D | Memory + queries — state, append-only log, stills, where/changes/diff | **Built** (`recall/memory.py`, `query.py`, `ask.py`) |
| E | Mesh + camera registration → `pixel_to_world` | **To build** |
| F | HTTP surface (DESIGN §5) | **To build** |

Logic tests for A–D pass with fake sightings: `python tests/test_slow_lane.py`.
**Not yet run against a real camera or the real model** — that is task 1 in §8.

---

## 1. Pipeline

```
SETUP (once)
  items.json  ── enroll calibrate ──▶ per-item min_score + reference crops
  camera still ── zones onboard ────▶ data/zones.json  (named boxes in image space)
  phone scan ──── Part E ───────────▶ room.glb + data/camera.json + data/xyz_map.npy

LIVE LOOP (every 1 s)
  frame
   → change gate vs last-detected frame
        static            → skip
        change/settling/heartbeat → YOLO-World on enrolled prompts (+ "person")
   → one best sighting per item
   → zone_at(base point)                       (+ pixel_to_world(base point) once E lands)
   → memory.update: appeared / moved / disappeared / inferred_contained / reappeared
   → print events + state table; stills to data/stills/

QUERIES (separate process, reads files)
  "where is X" · "what changed since T" · "diff T" · "history X"
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
data/enroll/<id>/     crops from calibration (from the corner camera)
data/reference/<id>.png  background-removed photos of each prop (committed) — re-id references
data/camera.json      (E) K, dist, R, t, width, height
data/xyz_map.npy      (E) H×W×3 world point per pixel, NaN = no hit
room.glb              (E) the mesh
```

**World frame (Part E):** meters, **Y-up**, **floor at y = 0**, origin at a taped room
corner, axes along the walls. Every 3D number in the system is in this frame.

**Data shapes**
```python
Sighting   = {cam, object_id, bbox[x1,y1,x2,y2], score, ts}
Zone       = {cam, name, box[x1,y1,x2,y2], prep, source}          # prep: "on" | "in" | ...
WorldState = {object_id: {name, zone, status, confidence, since, last_observed,
                          container, last_cam, last_bbox, xyz, still_path,
                          misses, pending_zone, pending_count, context}}
             # status: observed | inferred | unknown   (never seen = last_observed None)
Transition = {object_id, name, kind, from_zone, to_zone, status, container,
              confidence, ts, time, still_path}
             # kind: appeared | reappeared | moved | disappeared | inferred_contained
Answer     = {text, status, confidence, still_path}
```

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
python -m recall.enroll add bag --name "blue bag" --prompt "blue bag" --container
python -m recall.enroll calibrate            # all items; SPACE accept ×3, n skip, q quit
```
Calibration sets `min_score = 0.6 × lowest accepted score` and saves 3 crops per item.
**Re-run in demo lighting at the demo camera position.**

**Identity rule:** identity = which prompt fired, one best box per item per frame. Holds
while every tracked item is one of a kind and visually distinct. If two props swap
identity, change the props first (PLAN §13); embedding re-id against `data/enroll/` crops
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
python -m recall.live --show          # camera 0 + overlay window
python -m recall.live --src demo.mp4  # replay a recording (best way to iterate)
```

**Change gate** (`change.py`): 160-px-wide blurred grayscale diff against the **last frame
detection ran on**.
- ≥ `CHANGE_FRACTION` of pixels changed by ≥ `CHANGE_PIXEL_DELTA` → detect, and stay "hot"
  for `SETTLE_S`, detecting every loop. Without the settle window a vanished object gets one
  miss and never reaches `MISSING_AFTER_K`.
- Heartbeat detection every `HEARTBEAT_S` regardless.

**Detection** (`perception.py`): YOLO-World (`yolov8s-worldv2.pt`) with
`set_classes(item prompts + ["person"])`. Person boxes are reported, never tracked.

**Memory rules** (`memory.py`, DESIGN §6 with these specifics):

| Situation | Rule |
|---|---|
| First sighting | commit `appeared` immediately |
| Seen in a different zone | commit `moved` only after `MOVE_CONFIRM_N` (2) consecutive looks; **+1** if the new zone is `None` (objects in mid-air while carried) |
| Not seen | `misses += 1`; at `MISSING_AFTER_K` (3) → gone |
| Gone, and its last box was ≥ `CONTAIN_OVERLAP` inside an enrolled **container's** box in that same frame | commit `inferred_contained`, status `inferred`, zone = container's zone, confidence clipped 0.5–0.9 |
| Gone, no container | commit `disappeared`, status `unknown`, zone kept as last-known |
| Container moves | every item inferred inside it moves too (`moved`, still inferred) |
| Seen again after gone | commit `reappeared`, status `observed`, container cleared |
| Always | never promote `inferred` → `observed` except by actually seeing it |

`CONTAIN_OVERLAP` is **fraction of the object's box inside the container box**, not IoU —
IoU of small glasses vs a big bag is tiny even when they're fully inside.

**Output:** each detection pass prints one line (`seen: keys@kitchen counter 0.41, ...`);
each transition prints an event line + the state table.

---

## 6. Part D — Memory and queries (built)

```bash
python -m recall.ask                          # REPL
python -m recall.ask "where are my glasses"
python -m recall.ask "what changed since 10m"     # also: today, 2h, 14:30, 2:40pm
python -m recall.ask "diff 14:30"
python -m recall.ask "history keys"
python -m recall.ask state
```
Queries read `state.json` / `events.jsonl` fresh, so they run alongside `live.py`.
`state_at(T)` replays the log; `diff(T)` compares that to now.

Answer templates (`query.where_is`):

| Status | Answer |
|---|---|
| observed, zone | "Your keys are on the kitchen counter — seen just now." |
| observed, no zone | "Your phone is in view, but not in a named area — seen 2 minutes ago." |
| inferred, container visible | "Your glasses are probably in the blue bag, on the table — inferred, last seen going in 4 minutes ago." |
| inferred, container gone | "…probably in the blue bag, but I've lost track of the blue bag too. They went in at 2:12 PM." |
| unknown | "Last seen on the table at 2:40 PM. I haven't seen them since." |
| never seen | "I don't have a record of that." |

---

## 7. Part E — Mesh + camera registration (to build)

Goal: the camera "lives" inside the room model. One function, `pixel_to_world(u, v)`, is
the whole handoff — pass it to `Memory(..., pixel_to_world=fn)` and `xyz` fills in.

### E1. Scan → mesh
- **Prep:** demo lighting; clear clutter; tape **3–4 printed ArUco markers** (DICT_4X4_50,
  ~15 cm) on floor/wall where the corner camera sees them; tape-mark the origin corner.
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
   `VisualizerWithEditing`) → 3D corners in `data/markers.json`. Then detect markers in the
   camera frame (`cv2.aruco`) → `solvePnPRansac` → `solvePnPRefineLM` → R, t.
2. **Manual (fallback):** click 8–12 matching points (room/counter/door corners), spread
   across depth and height, image ↔ mesh, solve the same way.
3. **Tape check:** measure the lens position from the origin corner; camera centre
   `−Rᵀt` must be within ~5 cm.
- **Verify:** project subsampled mesh vertices with `cv2.projectPoints` onto the live frame;
  edges must line up. Mean reprojection error **< 3–5 px**.
- **Output:** `data/camera.json` = `{width, height, K, dist, R, t, rms_px, registered_at}`.
- Re-registration after a bump = re-run step 1 (no clicking).

### E4. `recall/room.py` — the bridge
```python
load_room(mesh_path, camera_json)        # build once, cache xyz_map.npy
pixel_to_world(u, v) -> (x, y, z) | None # xyz_map lookup (Open3D RaycastingScene, one ray/pixel)
floor_hit(u, v)      -> (x, 0, z) | None # analytic ray ∩ plane y=0 — exact, use for floor objects
world_to_pixel(x, y, z) -> (u, v)        # cv2.projectPoints, for overlays
```
Undistort pixel coordinates (`cv2.undistortPoints`) before casting rays.
Wire it: `Memory(items, zones, pixel_to_world=room.pixel_to_world)` in `live.py` when
`data/camera.json` exists.

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
| 1 | **Run A–D on the real camera.** `pip install -r requirements.txt`; time the detector (DESIGN §7 snippet); `enroll calibrate`; `zones onboard`; `live --show` | All props (keys, glasses, remote, wallet, bag) detected with the right label in demo lighting; detector < 150 ms/frame |
| 2 | **Record a demo video** (props appear, keys move, glasses go into the bag, bag moves) and replay with `--src` | Event log matches what happened, no phantom moves from a hand passing over |
| 3 | **Tune** `CHANGE_FRACTION`, `MISSING_AFTER_K`, `CONTAIN_OVERLAP`, prompts on that video | Glasses→bag inferred every run; hand-over-object produces no `disappeared` |
| 4 | **HTTP surface** (`recall/api.py`, FastAPI, **one uvicorn worker**): `GET /state`, `GET /objects/{id}/history`, `GET /changes?since=`, `POST /ask {text}`, `GET /stills/{path}`, `GET /stream` (MJPEG) — thin wrappers over `query.py` | care-partner page and voice layer can call it |
| 5 | **Part E** E1→E4, then E6 | sanity check passes; `state.json` shows `xyz` |
| 6 | **Only if identity swaps:** re-id — embed `data/enroll/` crops (CLIP via `open_clip`), accept a detection only if cosine ≥ 0.80 to its item, hold 0.60–0.80 | no swaps on the demo video |
| 7 | **Only if time:** E5 viewer | pins on the mesh at the right spots |

Do **not** add: a database (stay on JSON/JSONL), an LLM query layer, a second camera,
evidence clips, alerts. (PLAN §8.)

---

## 9. Parameters (`recall/config.py`)

| Name | Start | Notes |
|---|---|---|
| `LOOP_S` | 1.0 | one look per second |
| `CHANGE_WIDTH` / `CHANGE_PIXEL_DELTA` / `CHANGE_FRACTION` | 160 / 25 / 0.004 | raise FRACTION if auto-exposure flicker triggers detection |
| `SETTLE_S` / `HEARTBEAT_S` | 5 / 20 s | SETTLE must exceed `MISSING_AFTER_K × LOOP_S` |
| `DETECT_MODEL` / `DETECT_IMGSZ` | yolov8s-worldv2 / 640 | drop to 416 if slow |
| `DEFAULT_MIN_SCORE` | 0.15 | YOLO-World scores run low; calibration overrides |
| `MISSING_AFTER_K` | 3 | looks before "gone" |
| `MOVE_CONFIRM_N` | 2 | +1 for the no-zone case |
| `CONTAIN_OVERLAP` | 0.25 | fraction of object box inside container box |

---

## 10. Gotchas

- **First run downloads** the YOLO-World weights and its CLIP text encoder — do it on good
  wifi before the demo, not at the table.
- **macOS:** grant the terminal camera permission; OpenCV windows must run on the main thread
  (they do).
- **Lock exposure and white balance** on the webcam, or the change gate fires constantly and
  detection scores drift.
- **Mount the bag opening toward the camera** so "glasses going in" is actually visible.
- **A hand on an object** is excluded from containment by construction — only enrolled
  containers count, and people are never enrolled.
- **Restart-safe:** state reloads from `state.json`; the log is append-only. Delete `data/`
  to reset a demo.

## 11. Layout

```
items.json
requirements.txt
recall/
  config.py      tunables
  util.py        geometry, atomic json, item matching, time formatting
  camera.py      threaded camera / video-file source
  change.py      change gate
  perception.py  YOLO-World detector (+ one-off prompt detection for zones)
  zones.py       zones + onboarding CLI
  enroll.py      item dictionary + calibration CLI
  memory.py      WorldState, transitions, stills
  query.py       where_is / changes_since / state_at / diff / describe
  ask.py         typed query REPL (voice calls handle())
  live.py        the loop
  room.py        (E) mesh + camera bridge — to build
  api.py         (task 4) FastAPI — to build
tools/           (E) calibrate_intrinsics.py, register_camera.py — to build
tests/test_slow_lane.py
```
