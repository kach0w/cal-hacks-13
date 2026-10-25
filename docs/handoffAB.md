# Handoff A → B: room model + camera registration

**Person A** delivers the scanned room, the ArUco markers and the corner camera's
registration. **Person B** turns it into `pixel_to_world(u, v)` for the live loop
(CAMERA.md, Part E).

All files go in **`data/room/`**.

---

## World frame — set once, used by everything

- **Units:** meters.
- **Axes:** **Y up**. Floor at **y = 0**. Origin at a **taped floor corner** of the room.
  **+X** along one wall, **+Z** along the other.
- **Transforms baked into the mesh** — no node rotations or scales in the export.

---

## Agree before starting

1. **Camera resolution and settings:** 1280×720, with autofocus, auto-exposure and white
   balance **locked**. Calibration is invalid if any of these change.
2. **The camera never moves after registration.** If it does, re-run the marker step (no new
   scan needed).
3. **The markers stay up** in the same spots through the demo.

---

## Deliverables

### 1. `room.glb` — the scanned mesh

- A **triangle mesh**, not a point cloud or Gaussian splat. Under ~300k triangles, merged
  into one mesh. Texture optional.
- In the world frame above.
- Cover glass tabletops before scanning (LiDAR scans glass badly).
- **Check:** a tape-measured distance (wall to wall) matches the mesh within **2%**.

### 2. `markers.json` — the ArUco markers in world coordinates

```json
{
  "dictionary": "DICT_4X4_50",
  "marker_size_m": 0.15,
  "markers": {
    "3": [[x, y, z], [x, y, z], [x, y, z], [x, y, z]],
    "7": [[x, y, z], [x, y, z], [x, y, z], [x, y, z]]
  }
}
```

- `marker_size_m` is the edge length of the **black square only**, measured on the printout.
- Corners in OpenCV's order: **top-left, top-right, bottom-right, bottom-left**, looking at
  the marker upright. That's the order `cv2.aruco` returns them in; the wrong order gives a
  mirrored camera pose.
- **At least 3 markers**, spread near / middle / far from the camera, and **not all on one
  plane** (e.g. floor + wall).

### 3. `camera.json` — lens calibration and camera position

```json
{
  "width": 1280,
  "height": 720,
  "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "dist": [k1, k2, p1, p2, k3],
  "R": [[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]],
  "t": [tx, ty, tz],
  "camera_center": [x, y, z],
  "rms_px": 2.4,
  "markers_used": [3, 7, 12],
  "registered_at": "2026-10-24T14:30:00-07:00"
}
```

- `K`, `dist`: ChArUco board calibration (`tools/calibrate_intrinsics.py` → `intrinsics.json`) at
  **1280×720**, autofocus off.
- `R`, `t`: **OpenCV world→camera convention**, `X_cam = R · X_world + t`; camera axes
  x right, y down, z forward. This is `cv2.solvePnP` output after `cv2.Rodrigues`.
- `camera_center` = `−Rᵀ t`, the lens position in the room.
- **Checks:** `rms_px` < **5**; `camera_center` within **~5 cm** of the tape-measured lens
  position.

### 4. `camera_ref.jpg` — reference frame

One 1280×720 frame from the corner camera, taken at registration time with the markers
visible. Used to check the overlay and to detect a bumped camera.

### 5. `checks.json` — ground truth for the sanity test

```json
{
  "lens_measured": [x, y, z],
  "test_points": {
    "far_corner_floor": [x, 0, z],
    "mid_floor": [x, 0, z]
  }
}
```

Tape-measured from the origin corner. Place a small object at each test point when
`camera_ref.jpg` is taken (or take a second frame with them).

### 6. `surfaces.json` — the named surfaces (DESIGN §6.1)

```json
[
  {"name": "kitchen counter", "box_min": [x, y, z], "box_max": [x, y, z], "prep": "on", "top": y}
]
```

Axis-aligned boxes in the world frame. They replace the image-space zones once positions are 3D.
Written by `python -m recall.surfaces name` (drag boxes on `plan.png`; heights come from the scan).
`top` is the surface's height; without it, `box_min`'s y. A hand-made `zones3d.json` with `min` /
`max` instead of `box_min` / `box_max` is read too.

---

## What B does with it

- Builds `data/room/xyz_map.npy` by ray-casting every camera pixel into `room.glb`
  (Open3D `RaycastingScene`), after undistorting with `K` + `dist`.
- `pixel_to_world(u, v)` → lookup in `xyz_map`; `floor_hit(u, v)` → ray ∩ plane y = 0.
- Plugs it in: `Memory(..., pixel_to_world=room.pixel_to_world)` fills `xyz` in the state, and
  each object's surface is the `surfaces.json` box its `xyz` falls in.
- Re-registers after a bump from `markers.json` + a fresh frame.

## Acceptance (run together)

| Check | Pass |
|---|---|
| Scale: tape distance vs mesh | within 2% |
| Reprojection error (`rms_px`) | < 5 px |
| `camera_center` vs tape-measured lens | within ~5 cm |
| `floor_hit` on `mid_floor` object | within ~5 cm of `checks.json` |
| `floor_hit` on `far_corner_floor` object | within ~10 cm |
| Project `checks.json` points into `camera_ref.jpg` | land on the objects within a few px |

| If it fails | Likely cause |
|---|---|
| Same offset everywhere | mesh origin / axes |
| Error grows with distance | scale, or focal length in `K` |
| Error grows toward image edges | `dist` not applied |
| Mirrored, or camera behind a wall | marker corner order, or OpenCV vs Y-up flip |
| Good in the middle, bad in a corner | too few markers on that side |
