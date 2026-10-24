"""Register the corner camera to the room with ArUco markers (CAMERA.md E3), then run the Part E
acceptance checks (handoffAB.md) and write check_overlay.jpg for the eyeball check.

    python -m tools.register_camera --room data/room --src 0                  # live camera
    python -m tools.register_camera --room data/room --images DIR --test-pixel mid_floor=612,455
    python -m tools.register_camera --room data/room --check-only

The camera is static, so each marker corner is pooled over many frames before solvePnPRansac ->
solvePnPRefineLM. Writes R, t, camera_center, rms_px, markers_used, registered_at into camera.json
(keeping K/dist) and the median frame as camera_ref.jpg. After a bump, re-run the same command.
"""
import argparse
import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from recall.room import (ORTHO_TOL, Room, gl_to_opencv, mesh_frame_problems, opencv_to_gl, orthonormalize, planes,
                         project, undistort)
from tools import read_images, read_json, write_json

log = logging.getLogger(__name__)

FRAMES = 30
WARMUP = 15  # live frames discarded while the camera settles
RANSAC_PX = 8.0
MIN_MARKERS = 3
# Not all on one plane: two markers face > PLANE_DEG apart, or parallel ones are > PLANE_M apart along
# their normal (clicked corners can tilt a marker's normal a few degrees, so small offsets don't count).
PLANE_DEG, PLANE_M = 30.0, 0.25
POOL_PX = 1.0  # observations this far from a corner's median are dropped before averaging
JITTER_WARN_M, JITTER_WARN_DEG = 0.01, 0.1  # per-frame pose spread that means a moving camera
RMS_PX, LENS_M, PROJ_PX, SCALE_FRAC = 5.0, 0.05, 5.0, 0.02  # acceptance limits (handoffAB.md)
FLOOR_M, FLOOR_FAR_M = 0.05, {"far_corner_floor": 0.10}  # floor_hit tolerance; per test point overrides
MESH_M = 0.05  # markers.json corners must lie this close to the mesh surface
WALL_MARGIN_M = 0.1
MIN_ROOM_M = 1.0  # opposite walls are at least this far apart
ARKIT_PX = 1e-3  # the ARKit/GL round trip is exact up to float error
MIRROR = "mirrored, or camera behind a wall: marker corner order, or OpenCV vs Y-up flip"
ORIGIN = "mesh origin / axes (E1 clean-up)"
MESH_HINT = (f"{ORIGIN}: room.glb and markers.json are in different frames; or something in the scan "
             "sits in front of the object")
STATUS = {True: "PASS", False: "FAIL", None: "  - "}
FLOOR = ("same offset everywhere: mesh origin / axes; grows with distance: scale or focal length in K; "
         "grows toward image edges: dist not applied; bad in one corner only: too few markers on that side")


# --- registration -----------------------------------------------------------
def read_markers(path):
    """markers.json -> (dictionary name, {id: (4, 3) world corners TL, TR, BR, BL})."""
    m = read_json(path)
    if m is None:
        raise SystemExit(f"{path} is missing")
    return m["dictionary"], {int(k): np.array(v, np.float64) for k, v in m["markers"].items()}


def detector(dictionary):
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.relativeCornerRefinmentWinSize = 0.8  # default 0.3 module is ~1.5 px on a 30 px marker
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary)), params)


def detect(frames, dictionary, known):
    """Per frame: {marker id: (4, 2) image corners} for the markers listed in markers.json."""
    det = detector(dictionary)
    out = []
    for frame in frames:
        corners, ids, _ = det.detectMarkers(frame)
        ids = [] if ids is None else ids.ravel()
        out.append({int(i): c.reshape(4, 2).astype(np.float64) for i, c in zip(ids, corners) if int(i) in known})
    return out


def pool(per_frame):
    """Robust image corners over frames: per corner, the mean of observations within POOL_PX of the median.
    Markers seen in under half the frames are dropped as likely false detections."""
    pooled = {}
    for i in sorted(set().union(*per_frame)):
        obs = np.array([d[i] for d in per_frame if i in d])  # (n, 4, 2)
        if 2 * len(obs) < len(per_frame):
            log.warning("marker %d seen in only %d of %d frames: ignored", i, len(obs), len(per_frame))
            continue
        med = np.median(obs, 0)
        near = np.linalg.norm(obs - med, axis=2) <= POOL_PX
        pooled[i] = np.array([obs[near[:, c], c].mean(0) if near[:, c].any() else med[c] for c in range(4)])
    return pooled


def solve_pose(markers, image_corners, K, dist):
    """World/image corners -> (R, t, rms_px, {marker id used: (4,) RANSAC inlier mask of its corners}).

    The pose is refined on RANSAC inliers; rms_px covers every corner of the markers used, so a
    marker with some rejected corners (wrong corner order, bad click) still shows up in it.
    """
    ids = sorted(image_corners)
    obj = np.vstack([markers[i] for i in ids])
    img = np.vstack([image_corners[i] for i in ids])
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(obj, img, K, dist, iterationsCount=1000,
                                                 reprojectionError=RANSAC_PX, flags=cv2.SOLVEPNP_SQPNP)
    if not ok or inliers is None:
        raise RuntimeError(f"solvePnPRansac found no pose; likely cause: {MIRROR}, or wrong marker ids/size")
    keep = np.zeros(len(obj), bool)
    keep[inliers.ravel()] = True
    per_marker = keep.reshape(-1, 4).sum(1)
    if (per_marker < 4).any():
        log.warning("RANSAC rejected corners of markers %s (moved marker, wrong corner order or bad position?)",
                    [i for i, n in zip(ids, per_marker) if n < 4])
    rvec, tvec = cv2.solvePnPRefineLM(obj[keep], img[keep], K, dist, rvec, tvec)
    R, t = cv2.Rodrigues(rvec)[0], tvec.ravel()
    sel = np.repeat(per_marker > 0, 4)
    rms = float(np.sqrt((np.linalg.norm(project(obj[sel], K, dist, R, t) - img[sel], axis=1) ** 2).mean()))
    return R, t, rms, {i: m for i, m in zip(ids, keep.reshape(-1, 4)) if m.any()}


def frame_poses(per_frame, markers, inliers, K, dist, R, t):
    """Pose from each frame alone on the pooled pose's own corners (RANSAC inliers), seeded with it:
    [(R, camera centre)]. Frames that miss one of those markers are skipped: another corner set moves
    the pose by markers.json's click error, which is not the camera moving."""
    obj = np.vstack([markers[i][m] for i, m in inliers.items()])
    out = []
    for d in per_frame:
        if len(obj) < 6 or not all(i in d for i in inliers):
            continue
        img = np.vstack([d[i][m] for i, m in inliers.items()])
        ok, rv, tv = cv2.solvePnP(obj, img, K, dist, cv2.Rodrigues(R)[0], t.reshape(3, 1).copy(),
                                  useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        if ok:
            Ri = cv2.Rodrigues(rv)[0]
            out.append((Ri, -Ri.T @ tv.ravel()))
    return out


def rotation_deg(Ra, Rb):
    return float(np.degrees(np.linalg.norm(cv2.Rodrigues(Ra @ Rb.T)[0])))


def pose_spread(poses):
    """RMS spread of [(R, camera centre)] about their mean: (metres, degrees)."""
    if len(poses) < 2:
        return 0.0, 0.0
    Rs, cs = (np.array(a) for a in zip(*poses))
    mean_R = orthonormalize(Rs.sum(0))  # chordal mean rotation
    return (float(np.sqrt(((cs - cs.mean(0)) ** 2).sum(1).mean())),
            float(np.sqrt(np.mean([rotation_deg(Ri, mean_R) ** 2 for Ri in Rs]))))


def register_frames(frames, dictionary, markers, K, dist):
    """Pose of a static camera from many frames: R, t, camera_center, rms_px, markers_used, and the spread
    of per-frame poses: jitter_m (camera centre) and jitter_deg (rotation: a sagging mount barely moves
    the centre)."""
    per_frame = detect(frames, dictionary, markers)
    pooled = pool(per_frame)
    if len(pooled) < 2:
        raise RuntimeError(f"found markers {sorted(pooled)} from markers.json; need 2 to solve, "
                           f"{MIN_MARKERS}+ on floor and wall for a reliable pose")
    if len(pooled) < MIN_MARKERS:
        log.warning("only markers %s in view; 3+ on floor and wall give a reliable pose", sorted(pooled))
    R, t, rms, inliers = solve_pose(markers, pooled, K, dist)
    jitter_m, jitter_deg = pose_spread(frame_poses(per_frame, markers, inliers, K, dist, R, t))
    if jitter_m > JITTER_WARN_M or jitter_deg > JITTER_WARN_DEG:
        log.warning("camera pose jitters %.1f cm / %.2f° across frames: was it bumped, or is it vibrating? "
                    "Fix the mount and re-run", jitter_m * 100, jitter_deg)
    return {"R": R, "t": t, "camera_center": -R.T @ t, "rms_px": rms, "markers_used": sorted(inliers),
            "jitter_m": jitter_m, "jitter_deg": jitter_deg, "frames": len(frames)}


def register(room_dir, frames):
    """Register from frames; write the pose into camera.json and the median frame as camera_ref.jpg."""
    room_dir = Path(room_dir)
    cam = read_json(room_dir / "camera.json")
    if not cam or "K" not in cam:
        raise SystemExit(f"{room_dir}/camera.json has no K/dist: run python -m tools.calibrate_intrinsics first")
    size = (cam["width"], cam["height"])
    if any(f.shape[1::-1] != size for f in frames):
        raise SystemExit(f"frames must be {size[0]}x{size[1]}, the size K was calibrated at")
    dictionary, markers = read_markers(room_dir / "markers.json")
    dist = np.array(cam.get("dist") or [0.0] * 5, float)
    reg = register_frames(frames, dictionary, markers, np.array(cam["K"], float), dist)
    cam.update(R=reg["R"].tolist(), t=reg["t"].tolist(), camera_center=reg["camera_center"].tolist(),
               rms_px=reg["rms_px"], markers_used=reg["markers_used"],
               registered_at=datetime.now().astimezone().isoformat(timespec="seconds"))
    write_json(room_dir / "camera.json", cam)
    cv2.imwrite(str(room_dir / "camera_ref.jpg"), np.median(np.stack(frames), 0).round().astype(np.uint8))
    return reg


# --- acceptance checks ------------------------------------------------------
def wall_planes(vertices, triangles, axis):
    """Offsets of the walls facing ±axis: the largest plane and the largest one MIN_ROOM_M+ away from it,
    or None. Unlike the mesh's bounding box, scan spill past a wall (window recess, open door) can't move them."""
    offset, area = planes(vertices, triangles, axis)
    if not len(area):
        return None
    first = offset[area.argmax()]
    far = np.abs(offset - first) >= MIN_ROOM_M
    return sorted((first, offset[far][area[far].argmax()])) if far.any() else None


def plane_spread(corners):
    """Markers' (4, 3) corner sets -> (not all on one plane, largest angle between two markers' normals
    in degrees, largest offset of a marker's centre from another marker's plane in m)."""
    c = np.asarray(corners, np.float64)
    n = np.cross(c[:, 3] - c[:, 0], c[:, 1] - c[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    centre = c.mean(1)
    angle = np.degrees(np.arccos(np.clip(np.abs(n @ n.T), 0, 1))).max()
    offset = np.abs(np.einsum("ijk,ik->ij", centre[None] - centre[:, None], n)).max()  # [i, j]: (c_j - c_i)·n_i
    return bool(angle > PLANE_DEG or offset > PLANE_M), float(angle), float(offset)


def run_checks(room_dir, test_pixels=None, tape=None, room=None):
    """handoffAB.md acceptance + sanity checks on the files in room_dir.

    test_pixels: {test point name: (u, v)} base pixel of each object in camera_ref.jpg.
    tape: {"x": m, "z": m} wall-to-wall tape distances. Returns rows
    {check, ok (True / False / None = not checked), value, hint}.
    """
    room_dir = Path(room_dir)
    room = room or Room.load(room_dir)
    cam = read_json(room_dir / "camera.json")
    dictionary, markers = read_markers(room_dir / "markers.json")
    checks = read_json(room_dir / "checks.json")
    test_pixels, tape = test_pixels or {}, tape or {}
    used = [int(i) for i in cam.get("markers_used", sorted(markers)) if int(i) in markers]
    rows = []

    def add(check, ok, value, hint=""):
        rows.append({"check": check, "ok": None if ok is None else bool(ok), "value": value, "hint": hint})

    if not (room_dir / "room.glb").exists():
        add("room.glb loads as a triangle mesh", None, "not checked: no room.glb (floor plane only)")
    else:
        add("room.glb loads as a triangle mesh", room.source == "mesh",
            "ok" if room.source == "mesh" else f"{room.mesh_error}; using {room.source}",
            "re-export room.glb as a plain triangle mesh (no Draco/meshopt compression), or install open3d")
    name = f"markers.json corners on the mesh within {MESH_M * 100:.0f} cm"
    if room.source != "mesh":
        add(name, None, "not checked: no mesh")
    else:
        off = {i: float(room.surface_distance(c).max()) for i, c in sorted(markers.items())}
        worst = max(off, key=off.get)
        add(name, off[worst] <= MESH_M, f"farthest: marker {worst}, {off[worst] * 100:.1f} cm", MESH_HINT)
    for axis, a in (("x", 0), ("z", 2)):
        name = f"scale: wall-to-wall {axis} vs mesh within {SCALE_FRAC:.0%}"
        if tape.get(axis) is None or room.vertices is None:
            add(name, None, f"not checked ({'no mesh' if room.vertices is None else f'no --wall-to-wall-{axis}'})")
            continue
        walls = wall_planes(room.vertices, room.triangles, a)
        if walls is None:
            add(name, None, f"not checked: no two walls facing ±{axis} in the mesh")
            continue
        err = abs(walls[1] - walls[0] - tape[axis]) / tape[axis]
        add(name, err <= SCALE_FRAC, f"tape {tape[axis]:.3f} m, mesh walls at {axis} = {walls[0]:.3f} and "
            f"{walls[1]:.3f} m ({err:.1%})", "mesh scale: export in meters, or the wrong walls were measured")
    rms = cam.get("rms_px")
    add(f"reprojection rms_px < {RMS_PX:g}", None if rms is None else rms < RMS_PX,
        "not registered" if rms is None else f"{rms:.2f} px",
        f"markers.json doesn't match the printed markers; {MIRROR}")
    # RANSAC can quietly drop bad markers; every marker visible in camera_ref.jpg must agree with the pose.
    ref = cv2.imread(str(room_dir / "camera_ref.jpg"))
    name = f"every marker in camera_ref.jpg reprojects < {RMS_PX:g} px"
    if ref is None:
        add(name, None, "not checked: no camera_ref.jpg")
    else:
        err = {}
        for i, c in sorted(detect([ref], dictionary, markers)[0].items()):
            e = project(markers[i], room.K, room.dist, room.R, room.t) - c
            err[i] = float(np.sqrt((e ** 2).sum(1).mean()))
        add(name, bool(err) and max(err.values()) < RMS_PX,
            ", ".join(f"{i}: {e:.1f} px" for i, e in err.items()) or "no markers detected",
            f"{MIRROR}; or a marker moved since markers.json was made")
    spread, angle, offset = plane_spread([markers[i] for i in used]) if used else (False, 0.0, 0.0)
    add(f"{MIN_MARKERS}+ markers used, not all on one plane", len(used) >= MIN_MARKERS and spread,
        f"markers {used}: normals up to {angle:.0f}° apart, centres up to {offset:.2f} m off each other's plane",
        "too few markers: put 3+ where the camera sees them, on the floor and a wall")

    if checks is None:
        add("tape checks (checks.json)", None, "not checked: no checks.json")
    else:
        name = f"camera_center vs lens_measured within {LENS_M * 100:.0f} cm"
        if "lens_measured" not in checks:
            add(name, None, "not checked: no lens_measured in checks.json")
        else:
            d = float(np.linalg.norm(room.center - checks["lens_measured"]))
            add(name, d <= LENS_M, f"{d * 100:.1f} cm (solved {np.round(room.center, 3).tolist()})",
                f"{ORIGIN}; or scale / focal length in K")
        # floor_hit tests the pose; pixel_to_world (the live loop's function) also tests the mesh against it.
        lookups = [("floor_hit", room.floor_hit, FLOOR)]
        if room.source != "floor":
            lookups.append(("pixel_to_world", room.pixel_to_world, MESH_HINT))
        for name, p in checks.get("test_points", {}).items():
            tol = FLOOR_FAR_M.get(name, FLOOR_M)
            names = [f"{label} {name} within {tol * 100:.0f} cm" for label, _, _ in lookups]
            names.append(f"projected {name} within {PROJ_PX:g} px")
            if name not in test_pixels:
                for check in names:
                    add(check, None, "not checked: no --test-pixel")
                continue
            uv = test_pixels[name]
            for check, (_, lookup, hint) in zip(names, lookups):
                hit = lookup(*uv)
                d = np.inf if hit is None else float(np.linalg.norm(np.subtract(hit, p)))
                add(check, d <= tol, "no hit" if hit is None else f"{d * 100:.1f} cm", hint)
            e = float(np.hypot(*np.subtract(room.world_to_pixel(*p), uv)))
            add(names[-1], e <= PROJ_PX, f"{e:.1f} px", FLOOR)

    R_file = np.array(cam["R"], float)  # Room uses the nearest orthonormal matrix; judge the file's own R
    d, off = np.linalg.det(R_file), np.abs(R_file.T @ R_file - np.eye(3)).max()
    add("sanity: det(R) = +1, orthonormal", d > 0 and off < ORTHO_TOL, f"det {d:+.6f}, |RᵀR - I| {off:.0e}",
        MIRROR if d <= 0 else "camera.json R is not a rotation (edited by hand?): re-run register_camera")
    R, t, c = room.R, room.t, room.center
    depth = min(((markers[i] @ R.T + t)[:, 2].min() for i in used), default=np.nan)
    add("sanity: markers in front of the camera (z > 0)", depth > 0, f"nearest corner depth {depth:.2f} m", MIRROR)
    # A marker's printed face points along (BL - TL) × (TR - TL); a mirrored corner order flips it.
    away = [i for i in used
            if (c - markers[i].mean(0)) @ np.cross(markers[i][3] - markers[i][0], markers[i][1] - markers[i][0]) <= 0]
    add("sanity: camera on the printed side of every marker", not away,
        f"behind marker(s) {away}" if away else f"markers {used}", MIRROR)
    up = R.T @ [0.0, -1.0, 0.0]  # image up, in the world
    add("sanity: camera upright (image up within 60° of world +Y)", up[1] > 0.5,
        f"image up = {np.round(up, 2).tolist()}", "markers.json is not Y-up (OpenCV vs Y-up flip), or the camera "
        "is mounted sideways")
    if room.vertices is not None:  # no upper y bound: cleaned scans often crop the ceiling
        lo, hi = room.vertices.min(0), room.vertices.max(0)
        xz = [0, 2]
        inside = (lo[xz] - WALL_MARGIN_M <= c[xz]).all() and (c[xz] <= hi[xz] + WALL_MARGIN_M).all() and c[1] > lo[1]
        add("sanity: camera inside the walls and above the floor", inside,
            f"centre {np.round(c, 2).tolist()}, mesh {np.round(lo, 2).tolist()}..{np.round(hi, 2).tolist()}", MIRROR)
        problems = mesh_frame_problems(room.vertices, room.triangles)
        add("sanity: mesh Y-up with floor at y = 0", not problems, "; ".join(problems) or "ok", ORIGIN)
    else:
        add("sanity: camera above the floor (no mesh)", c[1] > 0, f"centre y = {c[1]:.2f} m", MIRROR)

    if used:  # a known marker corner through the ARKit/GL pose and back lands on the same pixel, in front
        X = markers[used[0]][0]
        T = opencv_to_gl(R, t)
        R2, t2 = gl_to_opencv(T)
        du = float(np.abs(project([X], room.K, room.dist, R, t) - project([X], room.K, room.dist, R2, t2)).max())
        z_gl = float((np.linalg.inv(T) @ np.r_[X, 1.0])[2])
        add("sanity: OpenCV -> ARKit/GL -> OpenCV keeps a marker corner", du < ARKIT_PX and z_gl < 0,
            f"Δ {du:.1e} px, marker {used[0]} at GL z = {z_gl:+.2f} m",
            "camera axes flip diag(1, -1, -1) misapplied")
    return rows


def print_table(rows):
    w = max(len(r["check"]) for r in rows)
    for r in rows:
        print(f"{STATUS[r['ok']]}  {r['check']:<{w}}  {r['value']}")
        if r["ok"] is False and r["hint"]:
            print(f"      {'':<{w}}  likely cause: {r['hint']}")
    failed = sum(r["ok"] is False for r in rows)
    print(f"\n{'FAIL' if failed else 'PASS'}: {sum(r['ok'] is True for r in rows)} passed, {failed} failed, "
          f"{sum(r['ok'] is None for r in rows)} not checked")


# --- overlay ----------------------------------------------------------------
def visible_pixels(room, points):
    """Project world points; keep those in front of the camera that land in the frame on their own ray
    (the distortion polynomial can fold far off-axis points back into the frame)."""
    points = np.asarray(points, np.float64)
    cam = points @ room.R.T + room.t
    front = cam[:, 2] > 0.05
    uv, xy = project(points[front], room.K, room.dist, room.R, room.t), cam[front, :2] / cam[front, 2:]
    ok = (uv >= 0).all(1) & (uv[:, 0] < room.width) & (uv[:, 1] < room.height)
    ok[ok] = np.linalg.norm(undistort(uv[ok], room.K, room.dist) - xy[ok], axis=1) < 1e-3
    return uv[ok]


def overlay(room, markers, test_points, test_pixels, image, max_tris=4000):
    """camera_ref.jpg + mesh edges (green), marker outlines (yellow, dot at TL), test points (red), clicks (cyan)."""
    img = image.copy()
    if room.vertices is not None:
        tris = room.triangles
        if len(tris) > max_tris:
            tris = tris[np.random.default_rng(0).choice(len(tris), max_tris, replace=False)]
        a = room.vertices[tris]
        n = int(np.clip(200_000 // (3 * len(tris)), 8, 400))  # samples per edge
        s = np.linspace(0, 1, n)[:, None, None, None]
        uv = visible_pixels(room, (a + s * (np.roll(a, -1, axis=1) - a)).reshape(-1, 3)).astype(int)
        img[uv[:, 1], uv[:, 0]] = (0, 255, 0)
    for i, c in markers.items():
        uv = project(c, room.K, room.dist, room.R, room.t)
        cv2.polylines(img, [np.rint(uv).astype(np.int32)], True, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(img, tuple(np.rint(uv[0]).astype(int)), 3, (0, 255, 255), -1)
        cv2.putText(img, str(i), tuple(np.rint(uv[0]).astype(int) + (4, -4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1)
    for name, p in test_points.items():
        u, v = np.rint(room.world_to_pixel(*p)).astype(int)
        cv2.drawMarker(img, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 16, 1)
        cv2.putText(img, name, (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    for u, v in test_pixels.values():
        cv2.circle(img, (int(round(u)), int(round(v))), 6, (255, 255, 0), 1)
    return img


# --- CLI ----------------------------------------------------------------------
def read_frames(src, n, size):
    """n frames from a camera index (warmed up, set to size) or a video file."""
    live = str(src).isdigit()
    cap = cv2.VideoCapture(int(src) if live else src)
    if not cap.isOpened():
        raise SystemExit(f"could not open {src}")
    if live:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        for _ in range(WARMUP):
            cap.read()
    frames = []
    while len(frames) < n:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise SystemExit(f"no frames from {src}")
    return frames


def click_pixels(image, names):
    """Click each named object's base on camera_ref.jpg (ESC skips one). Returns {name: (u, v)}."""
    win, picked, state = "camera_ref: click the base of each test object", {}, {}

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["pt"] = (x, y)

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    for name in names:
        state.clear()
        show = image.copy()
        cv2.putText(show, f"click: {name}   (ESC skips)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        while "pt" not in state:
            cv2.imshow(win, show)
            if cv2.waitKey(20) & 0xFF == 27:
                break
        if "pt" in state:
            picked[name] = state["pt"]
    cv2.destroyWindow(win)
    return picked


def parse_test_pixel(s):
    name, uv = s.split("=")
    u, v = (float(x) for x in uv.split(","))
    return name, (u, v)


def main():
    ap = argparse.ArgumentParser(description="Register the corner camera to the room and run the Part E checks.")
    ap.add_argument("--room", default="data/room")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--src", default="0", help="camera index or video file")
    src.add_argument("--images", help="folder of frames from the corner camera")
    src.add_argument("--check-only", action="store_true", help="re-run the checks on the existing files")
    ap.add_argument("--frames", type=int, default=FRAMES, help="frames to average")
    ap.add_argument("--test-pixel", type=parse_test_pixel, action="append", default=[], metavar="NAME=U,V",
                    help="base pixel of a checks.json test object in camera_ref.jpg")
    ap.add_argument("--no-click", action="store_true", help="skip test objects without --test-pixel")
    ap.add_argument("--wall-to-wall-x", type=float, metavar="M", help="tape distance between the X walls")
    ap.add_argument("--wall-to-wall-z", type=float, metavar="M", help="tape distance between the Z walls")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    room_dir = Path(args.room)

    if not args.check_only:
        cam = read_json(room_dir / "camera.json", {})
        size = (cam.get("width", 1280), cam.get("height", 720))
        frames = read_images(args.images, args.frames) if args.images else read_frames(args.src, args.frames, size)
        try:
            reg = register(room_dir, frames)
        except RuntimeError as e:
            raise SystemExit(f"registration failed: {e}")
        print(f"registered from {reg['frames']} frames: markers {reg['markers_used']}, rms {reg['rms_px']:.2f} px, "
              f"camera at {np.round(reg['camera_center'], 3).tolist()} m, "
              f"jitter {reg['jitter_m'] * 100:.2f} cm / {reg['jitter_deg']:.3f}°\n")

    ref = cv2.imread(str(room_dir / "camera_ref.jpg"))
    checks = read_json(room_dir / "checks.json")
    test_pixels = dict(args.test_pixel)
    missing = [k for k in (checks or {}).get("test_points", {}) if k not in test_pixels]
    if missing and not args.no_click and ref is not None:
        test_pixels.update(click_pixels(ref, missing))
    try:
        room = Room.load(room_dir)
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(str(e))
    rows = run_checks(room_dir, test_pixels, {"x": args.wall_to_wall_x, "z": args.wall_to_wall_z}, room)
    print_table(rows)
    if ref is not None:
        _, markers = read_markers(room_dir / "markers.json")
        img = overlay(room, markers, (checks or {}).get("test_points", {}), test_pixels, ref)
        cv2.imwrite(str(room_dir / "check_overlay.jpg"), img)
        print(f"overlay: {room_dir / 'check_overlay.jpg'}")
    raise SystemExit(1 if any(r["ok"] is False for r in rows) else 0)


if __name__ == "__main__":
    main()
