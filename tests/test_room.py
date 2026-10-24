"""Part E end to end on a synthetic room: render -> calibrate / register -> Room bridge -> acceptance checks."""
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from recall.room import (Room, gl_to_opencv, mesh_frame_problems, opencv_to_gl, pixel_rays, project, read_mesh,
                         undistort)
from tools import register_camera
from tools import synthetic_room as syn
from tools.calibrate_intrinsics import calibrate, find_board, hfov_intrinsics, lens_problems, write_intrinsics
from tools.register_camera import (JITTER_WARN_DEG, JITTER_WARN_M, detect, parse_test_pixel, plane_spread, pool,
                                   read_markers, register, register_frames, run_checks, solve_pose)

REPO = Path(__file__).resolve().parents[1]
N_FRAMES = 8
TAPE = {"x": 4.0, "z": 3.5}
RMS = "reprojection rms_px < 5"
REPROJECTS = "every marker in camera_ref.jpg reprojects < 5 px"
LENS = "camera_center vs lens_measured within 5 cm"
ON_MESH = "markers.json corners on the mesh within 5 cm"
MESH_LOADS = "room.glb loads as a triangle mesh"
COPLANAR = "3+ markers used, not all on one plane"
PRINTED_SIDE = "sanity: camera on the printed side of every marker"
IN_FRONT = "sanity: markers in front of the camera (z > 0)"
INSIDE = "sanity: camera inside the walls and above the floor"
MESH_FRAME = "sanity: mesh Y-up with floor at y = 0"
UPRIGHT = "sanity: camera upright (image up within 60° of world +Y)"
DET = "sanity: det(R) = +1, orthonormal"
ARKIT = "sanity: OpenCV -> ARKit/GL -> OpenCV keeps a marker corner"
MESH_MID, MESH_FAR = "pixel_to_world mid_floor within 5 cm", "pixel_to_world far_corner_floor within 10 cm"
SCALE_X = "scale: wall-to-wall x vs mesh within 2%"
FOLDING_DIST = np.array([-0.38, 0.17, 0.0, 0.0, -0.04])  # k3 bends r_d(r) back down before the image corners
IMAGE_CORNERS = np.array([[0, 0], [1279, 0], [0, 719], [1279, 719]], float)


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    """Synthetic room registered from N noisy frames; camera.json starts with K/dist only."""
    d = tmp_path_factory.mktemp("room")
    truth, clean = syn.generate(d, frames=0)
    shutil.copy(d / "camera_intrinsics.json", d / "camera.json")
    frames = syn.noisy_frames(clean, N_FRAMES, np.random.default_rng(1))
    truth["clean"] = clean
    truth["clean_frame"] = np.clip(np.rint(clean), 0, 255).astype(np.uint8)
    return d, truth, frames, register(d, frames)


@pytest.fixture(scope="module")
def room(synthetic):
    return Room.load(synthetic[0])


@pytest.fixture(scope="module")
def truth_camera(synthetic):
    _, truth, _, _ = synthetic
    return {"width": 1280, "height": 720, "K": syn.K.tolist(), "dist": syn.DIST.tolist(), "R": truth["R"],
            "t": truth["t"]}


@pytest.fixture(scope="module")
def truth_room(synthetic, truth_camera):
    """Room at the true pose, not the registered one: for comparisons with ground truth."""
    return Room(truth_camera, *read_mesh(synthetic[0] / "room.glb"))


@pytest.fixture
def room_copy(synthetic, tmp_path):
    shutil.copytree(synthetic[0], tmp_path / "room")
    for name in ("xyz_map.npy", "xyz_map.sha1"):
        (tmp_path / "room" / name).unlink(missing_ok=True)
    return tmp_path / "room"


def rotation_deg(Ra, Rb):
    return np.degrees(np.arccos(np.clip((np.trace(Ra @ np.asarray(Rb).T) - 1) / 2, -1, 1)))


def centre_error(reg, truth):
    return np.linalg.norm(reg["camera_center"] - truth["camera_center"])


def edit_json(path, change):
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))


def nudge(path, *keys, by):
    """Add `by` to the number at data[keys[0]][keys[1]]... in a JSON file."""
    def change(data):
        for k in keys[:-1]:
            data = data[k]
        data[keys[-1]] += by
    edit_json(path, change)


def move_camera(room_dir, R=None, centre=None):
    """Rewrite camera.json with another rotation and/or camera centre."""
    def change(cam):
        R2 = np.array(cam["R"]) if R is None else R(np.array(cam["R"]))
        c = np.array(cam["camera_center"]) if centre is None else centre(np.array(cam["camera_center"]))
        cam.update(R=R2.tolist(), t=(-R2 @ c).tolist(), camera_center=c.tolist())
    edit_json(room_dir / "camera.json", change)


def shift_mesh(room_dir, offset):
    v, f = read_mesh(room_dir / "room.glb")
    syn.write_mesh(room_dir / "room.glb", v + offset, f)


def no_ceiling(v, f):
    """Drop the triangles at the top of the mesh, as cleaned-up scans often do."""
    return v, f[~np.isclose(v[f][..., 1], v[:, 1].max()).all(1)]


def crop_ceiling(room_dir):
    syn.write_mesh(room_dir / "room.glb", *no_ceiling(*read_mesh(room_dir / "room.glb")))


def failed_checks(room_dir, truth, tape=TAPE):
    return {r["check"] for r in run_checks(room_dir, truth["test_pixels"], tape) if r["ok"] is False}


def box_mesh(size, lo=(0.0, 0.0, 0.0)):
    import open3d as o3d

    m = o3d.geometry.TriangleMesh.create_box(*size).translate(lo)
    return np.asarray(m.vertices), np.asarray(m.triangles)


def merge(*meshes):
    v = np.vstack([m[0] for m in meshes])
    offsets = np.cumsum([0] + [len(m[0]) for m in meshes[:-1]])
    return v, np.vstack([m[1] + o for m, o in zip(meshes, offsets)])


def count_builds(monkeypatch):
    builds, real = [], Room._build_xyz_map
    monkeypatch.setattr(Room, "_build_xyz_map", lambda self: builds.append(1) or real(self))
    return builds


# --- registration -------------------------------------------------------------
def test_registration_recovers_truth(synthetic):
    d, truth, frames, reg = synthetic
    assert centre_error(reg, truth) < 0.02
    assert rotation_deg(reg["R"], truth["R"]) < 0.2
    assert reg["rms_px"] < 1.0
    assert reg["markers_used"] == sorted(syn.MARKERS)
    assert reg["jitter_m"] < 0.001 and reg["jitter_deg"] < 0.02  # a static camera
    cam = json.loads((d / "camera.json").read_text())
    assert np.allclose(cam["camera_center"], -np.array(cam["R"]).T @ cam["t"])
    assert np.allclose(cam["K"], syn.K) and cam["registered_at"]
    ref = cv2.imread(str(d / "camera_ref.jpg"))
    assert ref.shape == (720, 1280, 3)
    # camera_ref.jpg is the median frame: less noisy than any single frame
    assert np.abs(ref - truth["clean"]).mean() < 0.6 * np.abs(frames[0] - truth["clean"]).mean()


def test_averaging_frames_beats_a_typical_single_frame(synthetic):
    # The ~3 mm left vs truth is ArUco corner-refinement bias, the same in every frame; averaging only
    # removes per-frame noise, so measure against the pose solved from the noise-free render.
    d, truth, frames, reg = synthetic
    dictionary, markers = read_markers(d / "markers.json")
    solve = lambda fs: register_frames(fs, dictionary, markers, syn.K, syn.DIST)  # noqa: E731
    noise_free = solve([truth["clean_frame"]])
    single = [centre_error(solve([f]), noise_free) for f in frames]
    assert centre_error(reg, noise_free) <= np.median(single)
    # Half the single frames pass that too: also pin that the pose comes from every frame, pooled.
    R, t, _, _ = solve_pose(markers, pool(detect(frames, dictionary, markers)), syn.K, syn.DIST)
    assert np.allclose(reg["R"], R, rtol=0, atol=1e-12) and np.allclose(reg["t"], t, rtol=0, atol=1e-12)


def test_pool_averages_noise_and_drops_outliers(synthetic):
    d, truth, frames, _ = synthetic
    dictionary, markers = read_markers(d / "markers.json")
    per_frame = detect(frames, dictionary, markers)
    clean = detect([truth["clean_frame"]], dictionary, markers)[0]

    def rms(corners):  # vs the noise-free detections, so ArUco's own bias cancels
        return np.sqrt(np.mean([((corners[i] - clean[i]) ** 2).sum(1) for i in clean]))

    pooled = pool(per_frame)
    assert rms(pooled) < 0.6 * np.sqrt(np.mean([rms(f) ** 2 for f in per_frame]))  # ~1/sqrt(N)
    occluded = [dict(f) for f in per_frame]
    for f in occluded[:2]:  # e.g. a hand passing in front of marker 12
        f[12] = f[12] + (6.0, -4.0)
    assert np.abs(pool(occluded)[12] - pooled[12]).max() < 0.05  # a plain mean moves 1.8 px


@pytest.fixture(scope="module")
def steady_view():
    return syn.render_room(*gl_to_opencv(syn.camera_gl()), ss=1)


@pytest.mark.parametrize("bump, over, under", [({"pitch": syn.PITCH - 1.0}, "jitter_deg", "jitter_m"),
                                               ({"eye": np.add(syn.EYE, (0.03, 0, 0))}, "jitter_m", "jitter_deg")],
                         ids=["sagged 1 deg", "moved 3 cm"])
def test_bumped_camera_warns(synthetic, steady_view, bump, over, under, caplog):
    # A mount that tilts barely moves the camera centre; the rotation spread has to show it.
    limit = {"jitter_m": JITTER_WARN_M, "jitter_deg": JITTER_WARN_DEG}
    dictionary, markers = read_markers(synthetic[0] / "markers.json")
    rng = np.random.default_rng(3)
    bumped = syn.render_room(*gl_to_opencv(syn.camera_gl(**bump)), ss=1)
    frames = syn.noisy_frames(steady_view, 4, rng) + syn.noisy_frames(bumped, 4, rng)
    with caplog.at_level(logging.WARNING):
        reg = register_frames(frames, dictionary, markers, syn.K, syn.DIST)
    assert reg[over] > limit[over] and reg[under] < limit[under] and "bumped" in caplog.text


# --- Room bridge --------------------------------------------------------------
def surface_points(rng, n):
    return {"floor": np.c_[rng.uniform(0, 4, n), np.zeros(n), rng.uniform(0, 3.5, n)],
            "wall x=4": np.c_[np.full(n, 4.0), rng.uniform(0, 2.6, n), rng.uniform(0, 3.5, n)],
            "wall z=3.5": np.c_[rng.uniform(0, 4, n), rng.uniform(0, 2.6, n), np.full(n, 3.5)],
            "table top": np.c_[rng.uniform(2.3, 3.3, n), np.full(n, 0.75), rng.uniform(0.6, 1.4, n)]}


def grid_pixels(nu=33, nv=19):
    """A pixel grid that includes the image corners, where distortion is largest."""
    u, v = np.meshgrid(np.linspace(0, 1279, nu), np.linspace(0, 719, nv))
    return np.c_[u.ravel(), v.ravel()]


def test_world_to_pixel_to_world_round_trip(room):
    assert room.source == "mesh"
    for name, pts in surface_points(np.random.default_rng(1), 400).items():
        pts = pts[syn.visible(pts, room.R, room.t)]
        assert len(pts) > 40, name
        back = np.array([room.pixel_to_world(*room.world_to_pixel(*p)) for p in pts])
        assert np.linalg.norm(back - pts, axis=1).max() < 1e-4, name  # the exact (u, v) ray, not a rounded pixel


def test_room_matches_ground_truth(truth_room):
    # Against the analytic scene and cv2.projectPoints with the true lens and pose, so errors that cancel
    # in Room's own round trips (a misread dist, say) show up, largest at the image corners.
    R, t = truth_room.R, truth_room.t
    px = grid_pixels()
    origin, dirs = pixel_rays(px, syn.K, syn.DIST, R, t)
    s, surf, _ = syn.first_hit(origin, dirs)
    truth = origin + s[:, None] * dirs
    uv = cv2.projectPoints(truth, cv2.Rodrigues(R)[0], t, syn.K, syn.DIST)[0].reshape(-1, 2)
    assert np.abs(uv - px).max() < 1e-3
    for (u, v), p, floor in zip(px, truth, surf == 2):
        assert np.linalg.norm(np.subtract(truth_room.pixel_to_world(u, v), p)) < 1e-3
        assert np.hypot(*np.subtract(truth_room.world_to_pixel(*p), (u, v))) < 0.01
        if floor:
            assert np.linalg.norm(np.subtract(truth_room.floor_hit(u, v), p)) < 1e-3


def test_undistort_is_exact_at_the_corners_of_a_wide_lens():
    dist = np.array([-0.35, 0.15, 5e-4, -4e-4, -0.03])  # webcam-like; OpenCV's default criteria are 1.6 px off
    px = np.array([[0, 0], [1279, 0], [0, 719], [1279, 719], [640, 0], [0, 360]], float)
    R, t = gl_to_opencv(syn.camera_gl())
    origin, dirs = pixel_rays(px, syn.K, dist, R, t)
    assert np.abs(project(origin + 2.0 * dirs, syn.K, dist, R, t) - px).max() < 0.01


def test_floor_hit_reprojects_to_the_same_pixel(room):
    for u, v in grid_pixels():
        p = room.floor_hit(u, v)
        assert p is not None and p[1] == 0.0
        assert np.hypot(*np.subtract(room.world_to_pixel(*p), (u, v))) < 0.1


def test_mesh_and_floor_agree_on_floor_pixels(room):
    px = grid_pixels()
    _, surf, _ = syn.first_hit(*pixel_rays(px, room.K, room.dist, room.R, room.t))
    floor = px[surf == 2]
    to_corner = np.linalg.norm(floor[:, None] - [[0, 0], [1279, 0], [0, 719], [1279, 719]], axis=2).min(1)
    assert len(floor) > 200 and (to_corner < 60).sum() >= 2
    for u, v in floor:
        assert np.linalg.norm(np.subtract(room.pixel_to_world(u, v), room.floor_hit(u, v))) < 0.01


def test_floor_hit_misses_above_the_horizon(room):
    up = room.world_to_pixel(*(room.center + [0.0, 0.5, 0.0] + 3 * (room.R.T @ [0, 0, 1]) * [1, 0, 1]))
    assert room.floor_hit(*up) is None


def test_ray_through_a_hole_in_the_mesh_is_none(truth_room, truth_camera):
    # An unscanned wall or an open doorway: rays through it hit nothing. NaN in xyz_map, never inf.
    v, f = truth_room.vertices, truth_room.triangles
    holed = Room(truth_camera, v, f[~np.isclose(v[f][..., 2], 3.5).all(1)])
    u, v = np.rint(holed.world_to_pixel(2.0, 1.5, 3.5)).astype(int)  # the far wall, above marker 21
    assert truth_room.pixel_to_world(u, v) is not None and holed.pixel_to_world(u, v) is None
    xyz = holed.xyz_map
    assert np.isnan(xyz[v, u]).all() and np.isnan(xyz[..., 0]).sum() > 100_000 and not np.isinf(xyz).any()


def test_lens_without_an_inverse_maps_to_none(truth_room, truth_camera, caplog):
    # Calibrated from views that missed the corners, dist can fold over: those pixels have no ray at all.
    assert np.isnan(undistort(IMAGE_CORNERS, syn.K, FOLDING_DIST)).all()
    camera = {**truth_camera, "dist": FOLDING_DIST.tolist()}
    with caplog.at_level(logging.WARNING):
        rooms = [Room(camera), Room(camera, truth_room.vertices, truth_room.triangles)]
    assert "no inverse" in caplog.text
    for room in rooms:
        assert room.pixel_to_world(0, 719) is None and room.floor_hit(0, 719) is None
        assert room.pixel_to_world(640, 600) is not None
    assert np.isnan(rooms[0].xyz_map[719, 0]).all() and not np.isnan(rooms[0].xyz_map[600, 640]).any()


def test_scanned_webcam_does_not_block_the_rays(truth_room, truth_camera):
    # The scan may include the webcam and its mount; the solved lens centre can sit inside them.
    webcam = box_mesh((0.06, 0.06, 0.06), truth_room.center - [0.03, 0.03, 0.02])
    with_camera = Room(truth_camera, *merge((truth_room.vertices, truth_room.triangles), webcam))
    for u, v in grid_pixels(9, 5):
        assert np.allclose(with_camera.pixel_to_world(u, v), truth_room.pixel_to_world(u, v), atol=1e-4)


def test_reads_are_thread_safe(room, room_copy, monkeypatch):
    px = grid_pixels(9, 5)
    expected = [room.pixel_to_world(u, v) for u, v in px]
    builds = count_builds(monkeypatch)
    fresh = Room.load(room_copy)  # no xyz_map.npy yet: every thread asks for it at once
    results, errors = [], []

    def work():
        try:
            results.append((fresh.xyz_map, [room.pixel_to_world(u, v) for u, v in px] == expected))
        except Exception as e:  # noqa: BLE001 - reported below
            errors.append(e)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert not errors and len(results) == 4 and len(builds) == 1
    assert all(xyz is results[0][0] and same for xyz, same in results)


def test_xyz_map_cache(room_copy, monkeypatch):
    builds = count_builds(monkeypatch)
    room = Room.load(room_copy)
    xyz = room.xyz_map
    assert room.xyz_map is xyz and len(builds) == 1
    assert xyz.shape == (720, 1280, 3) and xyz.dtype == np.float32
    for u, v in [(0, 719), (640, 360), (1279, 719), (1279, 0), (100, 50)]:
        assert np.allclose(xyz[v, u], room.pixel_to_world(u, v), atol=1e-4)

    assert np.array_equal(Room.load(room_copy).xyz_map, xyz, equal_nan=True) and len(builds) == 1  # from npy

    changes = {"camera.json": lambda: nudge(room_copy / "camera.json", "t", 0, by=0.05),
               "room.glb": lambda: shift_mesh(room_copy, [0.0, 0.0, 0.01])}
    for name, change in changes.items():
        n = len(builds)
        later = (room_copy / "xyz_map.npy").stat().st_mtime + 10
        os.utime(room_copy / name, (later, later))  # newer than the map
        Room.load(room_copy).xyz_map
        assert len(builds) == n + 1, name
        older = (room_copy / "xyz_map.npy").stat().st_mtime - 3600
        change()
        os.utime(room_copy / name, (older, older))  # changed, but delivered with an old mtime (unzip, cp -p)
        changed = Room.load(room_copy)
        assert np.allclose(changed.xyz_map[500, 900], changed.pixel_to_world(900, 500), atol=1e-4), name
        assert len(builds) == n + 2, name


def test_xyz_map_fallback_when_mesh_unreadable(room_copy, room, monkeypatch):
    xyz = Room.load(room_copy).xyz_map  # built from the mesh and cached

    def no_open3d(path):
        raise ImportError("No module named 'open3d'")

    monkeypatch.setattr("recall.room.read_mesh", no_open3d)
    fallback = Room.load(room_copy)
    assert fallback.source == "xyz_map" and "open3d" in fallback.mesh_error
    uv = np.random.default_rng(2).uniform([0, 0], [1279, 719], (300, 2))
    cell = np.floor(uv)[:, None] + [[0, 0], [1, 0], [0, 1], [1, 1]]  # the 4 map pixels around each sample
    _, surf, axis = syn.first_hit(*pixel_rays(cell.reshape(-1, 2), room.K, room.dist, room.R, room.t))
    face = (3 * surf + axis).reshape(-1, 4)  # the table is one surface with several faces
    planar = (face == face[:, :1]).all(1)
    assert planar.sum() > 250
    for u, v in uv[planar]:  # bilinear: the nearest pixel would be a few mm off
        assert np.linalg.norm(np.subtract(room.pixel_to_world(u, v), fallback.pixel_to_world(u, v))) < 1e-3
    # At a depth edge (table in front of the floor) the answer is on one of the surfaces, not in mid-air.
    cell = np.stack([xyz[:-1, :-1], xyz[:-1, 1:], xyz[1:, :-1], xyz[1:, 1:]])
    vs, us = np.nonzero(np.nan_to_num(np.ptp(cell, 0)).max(-1) > 0.3)
    assert len(us) > 100
    pick = np.random.default_rng(3).choice(len(us), 50, replace=False)
    got = [fallback.pixel_to_world(u + 0.5, v + 0.5) for u, v in zip(us[pick], vs[pick])]
    assert room.surface_distance(got).max() < 1e-3


def test_room_without_mesh_uses_floor_hit(room_copy):
    Room.load(room_copy).xyz_map  # leaves the mesh's xyz_map.npy behind
    (room_copy / "room.glb").unlink()
    room = Room.load(room_copy)
    assert room.source == "floor" and room.mesh_error is None
    table_top = room.world_to_pixel(2.8, 0.75, 1.0)
    for u, v in [(640, 600), (0, 719), (1200, 300), table_top]:
        assert room.pixel_to_world(u, v) == room.floor_hit(u, v)
    u, v = np.rint(table_top).astype(int)
    assert room.xyz_map[v, u, 1] == 0.0  # the floor map, not the stale mesh one


def test_room_needs_a_pose(room_copy):
    shutil.copy(room_copy / "camera_intrinsics.json", room_copy / "camera.json")
    with pytest.raises(ValueError, match="register_camera"):
        Room.load(room_copy)


def test_mesh_frame_sanity(synthetic):
    v, f = read_mesh(synthetic[0] / "room.glb")
    assert mesh_frame_problems(v, f) == [] and mesh_frame_problems(*no_ceiling(v, f)) == []
    for size in [(6.0, 2.6, 2.5), (4.0, 2.6, 2.0), (5.0, 2.7, 4.0)]:  # walls as big as the floor or bigger
        for mesh in (box_mesh(size), no_ceiling(*box_mesh(size))):
            assert mesh_frame_problems(*mesh) == [], size
    bv, bf = box_mesh((6.0, 2.6, 6.0))
    assert any("Y-up" in p for p in mesh_frame_problems(bv[:, [0, 2, 1]], bf))  # Z-up export of a big room
    assert any("floor" in p for p in mesh_frame_problems(v - [0, 1.4, 0], f))  # origin at the phone


# --- ARKit / GL ------------------------------------------------------------------
def test_arkit_gl_conversion():
    R0, t0 = cv2.Rodrigues(np.array([0.3, -1.2, 0.4]))[0], np.array([0.2, -0.5, 3.0])
    R1, t1 = gl_to_opencv(opencv_to_gl(R0, t0))
    assert np.allclose(R1, R0) and np.allclose(t1, t0)

    level = np.eye(4)
    level[:3, 3] = (1.0, 1.5, 2.0)  # ARKit identity orientation: looks down world -Z, +Y up
    for T in (level, syn.camera_gl()):
        R, t = gl_to_opencv(T)
        assert np.allclose(opencv_to_gl(R, t), T)
        ahead = T[:3, 3] - 2.0 * T[:3, 2]
        u0, v0 = project([ahead], syn.K, syn.DIST, R, t)[0]
        assert np.allclose((u0, v0), syn.K[:2, 2])
        assert project([ahead + 0.3 * T[:3, 0]], syn.K, syn.DIST, R, t)[0, 0] > syn.K[0, 2]
        assert project([ahead + [0, 0.3, 0]], syn.K, syn.DIST, R, t)[0, 1] < syn.K[1, 2]


# --- markers.json convention ------------------------------------------------------
def test_marker_corner_order_is_opencvs(synthetic):
    d, truth, _, _ = synthetic
    _, markers = read_markers(d / "markers.json")
    # TL, TR, BR, BL seen upright from inside the room: on the far wall z = 3.5, image right is world -X.
    tl_tr_br_bl = [[2.075, 0.975, 3.5], [1.925, 0.975, 3.5], [1.925, 0.825, 3.5], [2.075, 0.825, 3.5]]
    assert np.allclose(markers[21], tl_tr_br_bl)
    seen = detect([truth["clean_frame"]], "DICT_4X4_50", markers)[0]
    assert sorted(seen) == sorted(markers)
    for i, corners in seen.items():  # aruco's corner k is markers.json's corner k, no reordering
        assert np.abs(corners - project(markers[i], syn.K, syn.DIST, truth["R"], truth["t"])).max() < 1.0, i


def test_marker_layouts_on_one_plane_or_not():
    floor, wall = (0, 1, 0), (-1, 0, 0)
    floor_wall = [((1.5, 0, 1.9), floor, (0, 0, 1)), ((2.5, 0, 3.0), floor, (1, 0, 0)),
                  ((4.0, 0.9, 1.5), wall, (0, 1, 0))]
    for size in (0.15, 0.10):  # 3 markers, floor + wall: the handoff's minimum layout
        assert plane_spread([syn.marker_corners(*m, size=size) for m in floor_wall])[0]
    floor_table = [((1.5, 0, 1.9), floor, (0, 0, 1)), ((2.5, 0, 3.0), floor, (1, 0, 0)),
                   ((2.8, 0.75, 1.0), floor, (0, 0, 1))]
    assert plane_spread([syn.marker_corners(*m) for m in floor_table])[0]
    rng = np.random.default_rng(0)  # all on the floor, corners clicked with ~5 mm error
    all_floor = [syn.marker_corners((x, 0, z), floor, (0, 0, 1)) + rng.normal(0, 0.005, (4, 3))
                 for x, z in [(0.8, 1.0), (2.0, 3.0), (3.5, 1.5)]]
    assert not plane_spread(all_floor)[0]


# --- acceptance checks ------------------------------------------------------------
def test_acceptance_checks_pass(synthetic):
    d, truth, _, _ = synthetic
    rows = run_checks(d, truth["test_pixels"], TAPE)
    assert [r for r in rows if r["ok"] is not True] == []
    assert {MESH_LOADS, ON_MESH, UPRIGHT, "pixel_to_world mid_floor within 5 cm"} <= {r["check"] for r in rows}


def mirror_corner_order(c):
    return [c[1], c[0], c[3], c[2]]  # TR, TL, BL, BR: each marker left-right mirrored


def flip_y(c):
    return [[x, -y, z] for x, y, z in c]


@pytest.mark.parametrize("corrupt, expected, one_of", [
    (mirror_corner_order, {PRINTED_SIDE, RMS}, None),
    # The wall markers land under the floor; the floor markers don't change, so depending on the noise
    # RANSAC keeps those two alone (coplanar) or the flipped ones (rms, upright).
    (flip_y, {REPROJECTS, ON_MESH}, {COPLANAR, RMS, UPRIGHT})])
def test_mirrored_markers_fail_with_mirror_hint(synthetic, room_copy, corrupt, expected, one_of):
    _, truth, frames, _ = synthetic
    edit_json(room_copy / "markers.json",
              lambda m: m.update(markers={k: corrupt(c) for k, c in m["markers"].items()}))
    register(room_copy, frames)
    rows = [r for r in run_checks(room_copy, truth["test_pixels"], TAPE) if r["ok"] is False]
    failed = {r["check"] for r in rows}
    assert expected <= failed and (one_of is None or one_of & failed), failed
    assert any("corner order" in r["hint"] for r in rows)


def test_bad_marker_clicks_fail_rms_not_jitter(synthetic, room_copy, caplog):
    # A static camera with markers.json off: one corner by 20 cm (RANSAC drops it from the pose, but
    # rms_px still has to show it), the rest by ~1 cm, and marker 21 hidden in some frames by a hand.
    # None of it may be blamed on the mount.
    _, truth, frames, _ = synthetic
    rng = np.random.default_rng(5)
    edit_json(room_copy / "markers.json", lambda m: m.update(
        markers={k: (np.array(c) + rng.normal(0, 0.01, (4, 3))).tolist() for k, c in m["markers"].items()}))
    nudge(room_copy / "markers.json", "markers", "12", 0, 1, by=0.2)
    quad = np.rint(project(syn.scene_markers()[21], syn.K, syn.DIST, truth["R"], truth["t"])).astype(np.int32)
    hidden = [cv2.fillConvexPoly(f.copy(), quad, (180, 180, 180)) for f in frames[:3]]
    with caplog.at_level(logging.WARNING):
        reg = register(room_copy, hidden + frames[3:])
    assert "RANSAC rejected corners" in caplog.text and "bumped" not in caplog.text
    assert reg["jitter_m"] < JITTER_WARN_M / 10 and reg["jitter_deg"] < JITTER_WARN_DEG / 3
    assert RMS in failed_checks(room_copy, truth)


ROLL_90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
CASES = {  # name: (change the room dir, failed checks: exactly (==) or at least (<=))
    "ceiling cropped from the scan (fine)": (crop_ceiling, "==", set()),
    "lens tape 10 cm off": (lambda d: nudge(d / "checks.json", "lens_measured", 0, by=0.1), "==", {LENS}),
    "mesh shifted 20 cm": (lambda d: shift_mesh(d, [0.2, 0, 0]), "==", {ON_MESH}),
    # floor_hit doesn't use the mesh, so its rows pass; pixel_to_world's show the mesh is off
    "mesh origin at the phone": (lambda d: shift_mesh(d, [0, -1.4, 0]), "==",
                                 {MESH_FRAME, ON_MESH, MESH_MID, MESH_FAR}),
    "one marker corner clicked 15 cm off the wall": (
        lambda d: nudge(d / "markers.json", "markers", "12", 0, 0, by=-0.15), "==", {ON_MESH, REPROJECTS}),
    "room.glb unreadable": (lambda d: (d / "room.glb").write_bytes(b"glTF, Draco-compressed"), "==", {MESH_LOADS}),
    "camera behind the wall": (lambda d: move_camera(d, centre=lambda c: c * [-1, 1, 1] + [8.0, 0, 0]), "<=",
                               {IN_FRONT, INSIDE}),
    "camera below the floor": (lambda d: move_camera(d, centre=lambda c: c * [1, 0, 1] - [0, 0.5, 0]), "<=",
                               {INSIDE}),
    "R is a reflection": (lambda d: move_camera(d, R=lambda R: R @ np.diag([1.0, 1.0, -1.0])), "<=", {DET}),
    "R is not a rotation": (lambda d: move_camera(d, R=lambda R: 1.01 * R), "<=", {DET}),
    "camera on its side": (lambda d: move_camera(d, R=lambda R: ROLL_90 @ R), "<=", {UPRIGHT}),
    "markers all on the floor": (lambda d: (
        edit_json(d / "markers.json", lambda m: m["markers"].update(
            {"30": syn.marker_corners((1.0, 0.0, 1.0), (0, 1, 0), (0, 0, 1)).tolist()})),
        edit_json(d / "camera.json", lambda c: c.update(markers_used=["3", "7", "30"]))), "==", {COPLANAR}),
}


@pytest.mark.parametrize("case", CASES)
def test_each_check_catches_its_failure(synthetic, room_copy, case):
    _, truth, _, _ = synthetic
    corrupt, how, expected = CASES[case]
    corrupt(room_copy)
    failed = failed_checks(room_copy, truth)
    assert failed == expected if how == "==" else expected <= failed, failed


def test_wrong_tape_scale_fails(synthetic):
    d, truth, _, _ = synthetic
    assert failed_checks(d, truth, {"x": 4.2, "z": 3.5}) == {SCALE_X}


def test_scale_check_ignores_scan_past_the_walls(synthetic, room_copy):
    # A window recess in wall x = 4 and a strip of hallway floor through a door: the mesh's bounding box
    # grows to 4.6 m, but the walls are still 4.0 m apart.
    _, truth, _, _ = synthetic
    recess, hallway = box_mesh((0.12, 1.0, 1.0), (4.0, 1.0, 0.5)), box_mesh((0.6, 0.01, 0.8), (4.0, -0.01, 2.6))
    syn.write_mesh(room_copy / "room.glb", *merge(read_mesh(room_copy / "room.glb"), recess, hallway))
    assert failed_checks(room_copy, truth) == set()
    assert failed_checks(room_copy, truth, {"x": 4.2, "z": 3.5}) == {SCALE_X}
    assert register_camera.wall_planes(*box_mesh((0.5, 2.0, 3.0)), 0) is None  # a cupboard, not a room


@pytest.mark.parametrize("precision", [4, 6, np.float32])
def test_rounded_pose_passes(synthetic, room_copy, precision):
    # camera.json written by other tools: R/t rounded, or float32 as from ARKit's simd_float4x4.
    _, truth, _, _ = synthetic
    def rnd(a):
        return np.asarray(a).astype(precision) if precision is np.float32 else np.round(a, precision)
    edit_json(room_copy / "camera.json", lambda c: c.update(R=rnd(c["R"]).astype(float).tolist(),
                                                           t=rnd(c["t"]).astype(float).tolist()))
    assert failed_checks(room_copy, truth) == set()


def gl_pose_no_flip(R, t):
    return np.r_[np.c_[R.T, -R.T @ t], [[0, 0, 0, 1]]]


def opencv_pose_no_flip(T):
    return T[:3, :3].T, -T[:3, :3].T @ T[:3, 3]


@pytest.mark.parametrize("to_gl, to_cv, why", [
    (gl_pose_no_flip, None, "GL z = +"),  # round trip is exact, but the marker is behind the GL camera
    (gl_pose_no_flip, opencv_pose_no_flip, "GL z = +"),
    (None, opencv_pose_no_flip, "at GL z = -"),  # the marker is in front; the round trip moves it
], ids=["opencv_to_gl", "both, consistently", "gl_to_opencv"])
def test_arkit_check_catches_a_missing_axis_flip(synthetic, monkeypatch, to_gl, to_cv, why):
    d, truth, _, _ = synthetic
    for name, fn in (("opencv_to_gl", to_gl), ("gl_to_opencv", to_cv)):
        if fn:
            monkeypatch.setattr(register_camera, name, fn)
    rows = [r for r in run_checks(d, truth["test_pixels"], TAPE) if r["ok"] is False]
    assert [r["check"] for r in rows] == [ARKIT] and why in rows[0]["value"]


@pytest.mark.parametrize("name, dx, expected", [
    ("mid_floor", 0.20, {"floor_hit mid_floor within 5 cm", "pixel_to_world mid_floor within 5 cm",
                         "projected mid_floor within 5 px"}),
    ("far_corner_floor", 0.07, {"projected far_corner_floor within 5 px"}),  # inside the far corner's 10 cm
    ("far_corner_floor", 0.12, {"floor_hit far_corner_floor within 10 cm",
                                "pixel_to_world far_corner_floor within 10 cm",
                                "projected far_corner_floor within 5 px"}),
])
def test_bad_test_point_fails_floor_hit(synthetic, room_copy, name, dx, expected):
    _, truth, _, _ = synthetic
    nudge(room_copy / "checks.json", "test_points", name, 0, by=dx)
    assert failed_checks(room_copy, truth) == expected


def test_checks_skip_without_checks_json_or_pixels(synthetic, room_copy):
    rows = run_checks(room_copy)
    assert all(r["ok"] is None for r in rows if r["check"].startswith(("floor_hit", "pixel_to_world", "projected")))
    (room_copy / "checks.json").unlink()
    rows = run_checks(room_copy)
    assert not any(r["ok"] is False for r in rows)
    assert any(r["ok"] is None and "checks.json" in r["value"] for r in rows)


def test_register_cli_runs_without_clicks(synthetic, room_copy):
    # Re-registration after a bump: same command, no clicking, test pixels on the command line.
    _, truth, frames, _ = synthetic
    (room_copy / "frames").mkdir()
    for k, f in enumerate(frames):
        cv2.imwrite(str(room_copy / "frames" / f"{k:02d}.png"), f)
    base = [sys.executable, "-m", "tools.register_camera", "--room", str(room_copy), "--no-click"]
    pixels = [a for k, (u, v) in truth["test_pixels"].items() for a in ("--test-pixel", f"{k}={u:.2f},{v:.2f}")]
    run = subprocess.run(base + ["--images", str(room_copy / "frames"), "--wall-to-wall-x", "4", "--wall-to-wall-z",
                                 "3.5", *pixels], cwd=REPO, capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "FAIL" not in run.stdout and "0 failed, 0 not checked" in run.stdout
    ref, overlay = (cv2.imread(str(room_copy / n)) for n in ("camera_ref.jpg", "check_overlay.jpg"))
    assert (np.abs(ref.astype(int) - overlay).max(2) > 40).sum() > 5000  # drawn on, not just re-encoded
    check_only = subprocess.run(base + ["--check-only"], cwd=REPO, capture_output=True, text=True, timeout=120)
    assert check_only.returncode == 0 and "not checked: no --test-pixel" in check_only.stdout
    assert parse_test_pixel("mid_floor=612.5,455") == ("mid_floor", (612.5, 455.0))


# --- intrinsics -------------------------------------------------------------------
@pytest.fixture(scope="module")
def boards(tmp_path_factory):
    d = tmp_path_factory.mktemp("boards")
    images = [img for img, _, _ in syn.board_views(15, np.random.default_rng(0))]
    for k, img in enumerate(images):
        cv2.imwrite(str(d / f"{k:02d}.png"), img)
    return d, images


def assert_lens_recovered(K, dist):
    assert abs(K[0, 0] / syn.K[0, 0] - 1) < 0.01 and abs(K[1, 1] / syn.K[1, 1] - 1) < 0.01
    assert abs(K[0, 2] - syn.K[0, 2]) < 3 and abs(K[1, 2] - syn.K[1, 2]) < 3
    # The true rays through the image corners land within a few px (dist = 0 would be ~45 px off).
    rays = np.c_[undistort(IMAGE_CORNERS, syn.K, syn.DIST), np.ones(4)]
    assert np.abs(project(rays, K, dist, np.eye(3), np.zeros(3)) - IMAGE_CORNERS).max() < 5


def test_calibrate_intrinsics_from_rendered_boards(boards):
    views = [find_board(img) for img in boards[1]]
    assert all(v is not None for v in views)
    K, dist, rms = calibrate(views)
    assert_lens_recovered(K, dist)
    assert rms < 0.5 and lens_problems(views, K, dist) == []


def test_lens_problems():
    rng = np.random.default_rng(0)
    centred = [rng.uniform([400, 200], [880, 520], (54, 2)) for _ in range(10)]
    assert lens_problems(centred, syn.K, syn.DIST) == [
        "no view puts the board in the top-left, top-right, bottom-left, bottom-right corner(s) of the image"]
    spread = [IMAGE_CORNERS + rng.uniform(-60, 60, (4, 2)) * [-1, 1] for _ in range(3)]
    assert lens_problems([*centred, np.abs(np.vstack(spread))], syn.K, syn.DIST) == []
    assert any("folds" in p for p in lens_problems(centred, syn.K, FOLDING_DIST))


def test_calibrate_cli(boards, tmp_path):
    def run(room, *args):
        cmd = [sys.executable, "-m", "tools.calibrate_intrinsics", "--room", str(room), *args]
        done = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=120)
        assert done.returncode == 0, done.stdout + done.stderr
        return done.stdout, json.loads((room / "camera.json").read_text())

    out, cam = run(tmp_path / "board", "--images", str(boards[0]))
    assert_lens_recovered(np.array(cam["K"]), np.array(cam["dist"]))
    assert (cam["width"], cam["height"]) == (1280, 720) and cam["intrinsics_rms_px"] < 0.5 and "recapture" not in out
    _, cam = run(tmp_path / "hfov", "--hfov", "75")
    assert np.allclose(cam["K"], hfov_intrinsics(75.0)[0]) and cam["dist"] == [0.0] * 5


def test_hfov_fallback_and_merge(tmp_path):
    K, dist = hfov_intrinsics(75.0)
    assert K[0, 0] == K[1, 1] and np.isclose(K[0, 0], 640 / np.tan(np.radians(37.5)))
    assert np.allclose(K[:2, 2], (639.5, 359.5)) and not dist.any()
    (tmp_path / "camera.json").write_text(json.dumps({"width": 1280, "height": 720, "K": np.eye(3).tolist(),
                                                      "dist": [0.1] * 5, "R": np.eye(3).tolist(), "t": [0, 0, 1],
                                                      "camera_center": [0, 0, -1], "rms_px": 1.0, "note": "kept"}))
    write_intrinsics(tmp_path, (1280, 720), K, dist)
    cam = json.loads((tmp_path / "camera.json").read_text())
    assert np.allclose(cam["K"], K) and cam["dist"] == [0.0] * 5 and cam["note"] == "kept"
    assert not {"R", "t", "camera_center", "rms_px"} & set(cam)


def test_the_live_loop_fills_xyz_from_data_room(synthetic, tmp_path):
    """The handoff end to end: data/room/ -> live.load_room -> Memory -> state.json xyz."""
    from dataclasses import replace

    from fakes import FRAME, ITEMS, ZONES
    from recall.config import Settings
    from recall.live import load_room
    from recall.memory import Memory
    from recall.models import Sighting
    from recall.store import Store

    d, truth, _, _ = synthetic
    shutil.copytree(d, tmp_path / "room")
    cfg = replace(Settings(data_dir=tmp_path))
    p2w = load_room(cfg)
    assert p2w is not None
    u, v = truth["test_pixels"]["mid_floor"]
    keys_box = [u - 40, v - 30, u + 40, v]                    # base point = the mid_floor pixel
    mem = Memory(ITEMS, ZONES, Store(tmp_path), cfg, pixel_to_world=p2w)
    mem.update([Sighting("keys", keys_box, 0.6, 0)], FRAME, 0)
    xyz = json.loads((tmp_path / "state.json").read_text())["keys"]["xyz"]
    want = json.loads((d / "checks.json").read_text())["test_points"]["mid_floor"]
    assert np.linalg.norm(np.subtract(xyz, want)) < 0.05      # handoff: mid_floor within ~5 cm


def test_no_room_means_no_xyz(tmp_path):
    from recall.config import Settings
    from recall.live import load_room
    assert load_room(Settings(data_dir=tmp_path)) is None
    (tmp_path / "room").mkdir()
    (tmp_path / "room" / "camera.json").write_text('{"width": 1280}')   # no pose: warned, demo goes on
    assert load_room(Settings(data_dir=tmp_path)) is None
