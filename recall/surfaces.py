"""Surfaces (DESIGN §6.1): named regions of the scanned room — "coffee table", "sofa" — as boxes
in the world frame, in data/room/surfaces.json. Once the room is registered they are the places
an object can be (recall/places.py); image zones are the fallback before that.

    python -m recall.surfaces name    # drag boxes on the scan's top-down plan, name them
    python -m recall.surfaces show    # the surfaces on the plan and on camera_ref.jpg
    python -m recall.surfaces list

Drawn on plan.png, which tools/scan_to_room.py renders looking straight down from 1.1 m at
PLAN_PPM pixels per metre, with PLAN_MARGIN_M around the mesh. A box's height comes from the scan:
its top is the busiest level under the box (seen the same way), and a raised level holding at least
TOP_SHARE of it beats the floor, so the floor showing around a loosely drawn table doesn't pull the
table down. The box runs from BELOW_M under that top to ABOVE_M over it, so whatever stands on it
is inside; the top is saved with it.
"""
import argparse
import logging
from pathlib import Path

import numpy as np

from . import config
from .models import Surface
from .store import read_json, write_json_atomic

log = logging.getLogger(__name__)
PLAN_PPM, PLAN_MARGIN_M, PLAN_EYE_Y = 100, 0.3, 1.1   # must match tools/scan_to_room.py plan_png()
BELOW_M, ABOVE_M = 0.10, 0.60
GRID_M = 0.05          # spacing of the height samples under a box
LEVEL_M = 0.05         # heights within this of a level are on it
RAISED_M = 0.10        # levels at least this high are furniture, not the floor
TOP_SHARE = 0.25       # a raised level with this share of the samples under a box beats the floor
FLOOR_M = 0.05         # a top this close to y = 0 is the floor: said so when naming


def load_surfaces(path) -> list[Surface]:
    """surfaces.json; without it, handoffAB.md's zones3d.json next to it (same boxes, min/max)."""
    path = Path(path)
    if not path.exists() and (path.parent / "zones3d.json").exists():
        path = path.parent / "zones3d.json"
    return [Surface.from_dict(s) for s in read_json(path, [])]


def save_surfaces(surfaces: list[Surface], path):
    write_json_atomic(path, [s.to_dict() for s in surfaces])


def plan_origin(vertices, plan_shape=None):
    """World (x, z) of plan.png's top-left corner. With plan_shape, checks plan.png came from this mesh."""
    xz = np.asarray(vertices)[:, [0, 2]]
    lo, hi = xz.min(axis=0) - PLAN_MARGIN_M, xz.max(axis=0) + PLAN_MARGIN_M
    if plan_shape is not None:
        want = tuple(((hi - lo) * PLAN_PPM).astype(int)[::-1])
        if tuple(plan_shape[:2]) != want:
            raise SystemExit(f"plan.png is {plan_shape[1]}x{plan_shape[0]} px but room.glb makes "
                             f"{want[1]}x{want[0]}: they're from different scans; re-run tools/scan_to_room.py")
    return lo


def to_plan(lo, x, z):
    return int((x - lo[0]) * PLAN_PPM), int((z - lo[1]) * PLAN_PPM)


def top_height(scene, x0, x1, z0, z1) -> float | None:
    """Height of the surface under a box, looking down from PLAN_EYE_Y (as plan.png does): the median
    of the busiest level (the heights within LEVEL_M of it). Raised levels holding at least TOP_SHARE
    of the samples go first: a table drawn loosely shows the floor around it, often more of it."""
    import open3d as o3d
    xs, zs = np.arange(x0, x1, GRID_M) + GRID_M / 2, np.arange(z0, z1, GRID_M) + GRID_M / 2
    X, Z = np.meshgrid(xs, zs)
    rays = np.column_stack([X.ravel(), np.full(X.size, PLAN_EYE_Y), Z.ravel(),
                            np.zeros(X.size), -np.ones(X.size), np.zeros(X.size)]).astype(np.float32)
    t = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    h = PLAN_EYE_Y - t[np.isfinite(t)]
    if not len(h):
        return None
    levels = np.unique(np.round(h / LEVEL_M)) * LEVEL_M
    share = np.array([(np.abs(h - lv) <= LEVEL_M).mean() for lv in levels])
    raised = (levels >= RAISED_M) & (share >= TOP_SHARE)
    best = levels[raised][np.argmax(share[raised])] if raised.any() else levels[np.argmax(share)]
    return float(np.median(h[np.abs(h - best) <= LEVEL_M]))


def draw_plan(plan, surfaces, lo, numbered=True):
    import cv2
    from .overlay import AMBER
    out = plan.copy()
    for i, s in enumerate(surfaces):
        p0, p1 = to_plan(lo, s.box_min[0], s.box_min[2]), to_plan(lo, s.box_max[0], s.box_max[2])
        cv2.rectangle(out, p0, p1, AMBER, 2)
        label = f"[{i}] {s.name}" if numbered else s.name
        cv2.putText(out, label, (p0[0] + 4, p0[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, AMBER, 2)
    return out


def _room_inputs(cfg):
    import cv2
    from .room import read_mesh
    glb, plan_png = cfg.room_dir / "room.glb", cfg.room_dir / "plan.png"
    if not glb.exists() or not plan_png.exists():
        raise SystemExit(f"need {glb} and {plan_png}: run tools/scan_to_room.py first")
    vertices, triangles = read_mesh(glb)
    plan = cv2.imread(str(plan_png))
    return vertices, triangles, plan, plan_origin(vertices, plan.shape)


def name(cfg):
    import cv2
    import open3d as o3d
    from .overlay import close_windows, preview
    from .zones import _ask_name, _ask_prep, _boxes
    vertices, triangles, plan, lo = _room_inputs(cfg)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(vertices.astype(np.float32)), o3d.core.Tensor(triangles.astype(np.uint32)))
    surfaces = load_surfaces(cfg.surfaces_path)
    print("Drag a box over each surface on the plan (table, counter, sofa...): SPACE or ENTER keeps it,\n"
          "c cancels it, ESC when done. Then name each one here.")
    try:
        rois = cv2.selectROIs("name surfaces", draw_plan(plan, surfaces, lo), showCrosshair=False)
    finally:
        close_windows()
    new = []
    for x0, y0, x1, y1 in _boxes(rois, plan.shape[1], plan.shape[0]):
        (wx0, wz0), (wx1, wz1) = lo + np.array([x0, y0]) / PLAN_PPM, lo + np.array([x1, y1]) / PLAN_PPM
        top = top_height(scene, wx0, wx1, wz0, wz1)
        if top is None:
            print(f"  nothing scanned under the box at ({x0:.0f},{y0:.0f}): dropped")
            continue
        new.append(Surface("?", [round(wx0, 3), round(top - BELOW_M, 3), round(wz0, 3)],
                           [round(wx1, 3), round(top + ABOVE_M, 3), round(wz1, 3)], top=round(top, 3)))
    if not new:
        print("no boxes drawn")
        return
    n, path = len(surfaces), cfg.room_dir / "surfaces_drawn.jpg"
    cv2.imwrite(str(path), draw_plan(plan, surfaces + new, lo))
    preview(path)  # as zones does: Preview stays responsive while the terminal asks for names
    print(f"Name the new boxes [{n}..{n + len(new) - 1}] (numbered in {path}) here:")
    for i, s in enumerate(new, n):
        floor = " (the floor)" if abs(s.top) < FLOOR_M else ""
        s.name = _ask_name(f"[{i}] top at {s.top:.2f} m{floor} — name (ENTER drops it): ", surfaces) or ""
        if s.name:
            s.prep = _ask_prep(s.name)
            surfaces.append(s)
    save_surfaces(surfaces, cfg.surfaces_path)
    print(f"saved {len(surfaces)} surfaces -> {cfg.surfaces_path}")


def show(cfg):
    import cv2
    from .overlay import AMBER, draw_surfaces, preview
    vertices, _, plan, lo = _room_inputs(cfg)
    surfaces = load_surfaces(cfg.surfaces_path)
    out = [cfg.room_dir / "surfaces.png"]
    cv2.imwrite(str(out[0]), draw_plan(plan, surfaces, lo, numbered=False))
    ref = cv2.imread(str(cfg.room_dir / "camera_ref.jpg"))
    if ref is not None and (cfg.room_dir / "camera.json").exists():
        from .room import Room
        out.append(cfg.room_dir / "surfaces_camera.jpg")
        cv2.imwrite(str(out[1]), draw_surfaces(ref.copy(), surfaces, Room.load(cfg.room_dir), AMBER))
    for p in out:
        print(f"wrote {p}")
        preview(p)


def main():
    ap = argparse.ArgumentParser(description="Name the surfaces of the scanned room.")
    ap.add_argument("cmd", choices=["name", "show", "list"])
    config.add_args(ap)
    a = ap.parse_args()
    cfg = config.from_args(a)
    if a.cmd == "list":
        surfaces = load_surfaces(cfg.surfaces_path)
        for s in surfaces:
            print(f"{s.name:<20} {s.prep:<8} x {s.box_min[0]:.2f}..{s.box_max[0]:.2f}  "
                  f"top {s.top:.2f}  z {s.box_min[2]:.2f}..{s.box_max[2]:.2f}")
        if not surfaces:
            print(f"no surfaces in {cfg.surfaces_path} yet: python -m recall.surfaces name")
    else:
        try:
            name(cfg) if a.cmd == "name" else show(cfg)
        except (KeyboardInterrupt, EOFError):
            raise SystemExit(f"\nstopped: {cfg.surfaces_path} not changed")


if __name__ == "__main__":
    main()
