# Handoff A → B: room model + camera registration

**Person A** delivers the scanned room, the ArUco markers and the corner camera's
registration. **Person B** turns it into `pixel_to_world(u, v)` for the live loop
(CAMERA.md, Part E).

All files go in **`data/room/`**, written by one command, `uv run tools/setup_room.py --camera brio`
(scan → map, lens, camera position; see DESIGN.md §1). The webcam's lens file lives in
`data/cameras/` and is reused in every room.

---

## World frame — set once, used by everything

- **Units:** meters.
- **Axes:** **Y up**. Floor at **y = 0**. **+X** and **+Z** along the walls. Origin at the corner
  of the floor's outline, so the room is at x, z ≥ 0. All of it is found from the scan; no tape
  or clicks. `plan.png` shows the origin and axes.
- **Transforms baked into the mesh** — no node rotations or scales in the export.

---

## Agree before starting

1. **Camera resolution and settings:** 1280×720 with focus fixed. Calibration is invalid if the
   resolution, focus or zoom changes. Lock exposure and white balance too, for the change gate.
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
- **Checked automatically:** the mesh matches the phone's LiDAR depth to < 2 cm.

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

- `marker_size_m` is the edge length of the **black square only**, as measured in the scan.
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

- `K`, `dist`: the webcam's ChArUco calibration at **1280×720** (`data/cameras/<webcam>.json`).
- `R`, `t`: **OpenCV world→camera convention**, `X_cam = R · X_world + t`; camera axes
  x right, y down, z forward. This is `cv2.solvePnP` output after `cv2.Rodrigues`.
- `camera_center` = `−Rᵀ t`, the lens position in the room.
- **Checked automatically:** `rms_px` < **5**; `camera_overlay.jpg` draws the markers and the
  mesh where this pose puts them, on top of `camera_ref.jpg`.

### 4. `camera_ref.jpg` — reference frame

One 1280×720 frame from the corner camera, taken at registration time with the markers
visible. Used to check the overlay and to detect a bumped camera.

### 5. `zones3d.json` — optional

```json
[
  {"name": "kitchen counter", "min": [x, y, z], "max": [x, y, z], "prep": "on"}
]
```

Axis-aligned boxes in the world frame. Replace the image-space zones once positions are 3D.

---

## What B does with it

- Builds `data/room/xyz_map.npy` by ray-casting every camera pixel into `room.glb`
  (Open3D `RaycastingScene`), after undistorting with `K` + `dist`.
- `pixel_to_world(u, v)` → lookup in `xyz_map`; `floor_hit(u, v)` → ray ∩ plane y = 0.
- Plugs it in: `Memory(..., pixel_to_world=room.pixel_to_world)` fills `xyz` in the state.
- Re-registers after a bump from `markers.json` + a fresh frame.

## Checks

`setup_room.py` checks everything it can and prints notes: the scan lines up with its own LiDAR,
floor markers sit at y = 0, the camera's fit to the markers. No tape measurements. With 3
markers in view, floor positions are good to roughly 5–8 cm; more markers in view tighten it.

| If positions look off | Likely cause |
|---|---|
| Same offset everywhere | mesh origin / axes |
| Error grows with distance | scale, or focal length in `K` |
| Error grows toward image edges | `dist` not applied |
| Mirrored, or camera behind a wall | marker corner order, or OpenCV vs Y-up flip |
| Good in the middle, bad in a corner | too few markers on that side |
