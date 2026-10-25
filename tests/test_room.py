"""recall/room.py, the runtime half of handoffAB.md, on Person A's real scan (data/room/) with a known
camera pose, checked against the setup tools' own geometry (tools/roomlib.py)."""
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from recall.config import ROOT, Settings
from recall.room import CV_TO_GL, Room, gl_to_opencv, opencv_to_gl, read_mesh

sys.path.insert(0, str(ROOT / "tools"))
import roomlib  # noqa: E402  (their tools import each other as scripts)

REAL = ROOT / "data" / "room"
K = np.array([[900.0, 0, 640], [0, 900, 360], [0, 0, 1]])
DIST = np.array([-0.05, 0.02, 0.0, 0.0, 0.0])


def look_at(eye, target):
    """OpenCV world->camera (R, t) for a camera at eye looking at target, image up = world +Y."""
    z = np.subtract(target, eye) / np.linalg.norm(np.subtract(target, eye))
    x = np.cross(z, [0.0, 1.0, 0.0])
    x /= np.linalg.norm(x)
    R = np.vstack([x, np.cross(z, x), z])
    return R, -R @ np.asarray(eye, float)


EYE, TARGET = (4.9, 2.4, 3.8), (1.5, 0.3, 1.0)   # high in the far corner, looking across the room
R, T = look_at(EYE, TARGET)


def camera_json(R=R, t=T):
    """camera.json exactly as tools/register_camera.py writes it (rounded R and t)."""
    return {"width": 1280, "height": 720, "K": K.tolist(), "dist": DIST.tolist(),
            "R": np.round(R, 7).tolist(), "t": np.round(t, 5).tolist(),
            "camera_center": np.round(roomlib.camera_center(R, t), 4).tolist(), "rms_px": 0.4,
            "markers_used": [0, 3, 4, 5], "registered_at": "2026-10-24T17:00:00-07:00"}


@pytest.fixture(scope="module")
def room_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("room")
    for name in ("room.glb", "markers.json", "plan.png"):
        shutil.copy(REAL / name, d / name)
    (d / "camera.json").write_text(json.dumps(camera_json()))
    return d


@pytest.fixture(scope="module")
def room(room_dir):
    return Room.load(room_dir)


@pytest.fixture(scope="module")
def markers():
    m = json.loads((REAL / "markers.json").read_text())
    return {int(i): np.array(c) for i, c in m["markers"].items()}


def test_reads_their_mesh_like_their_tools(room_dir):
    v, f = read_mesh(room_dir / "room.glb")
    theirs = roomlib.Mesh.load(room_dir / "room.glb")
    assert np.array_equal(v, theirs.vertices) and np.array_equal(f, theirs.faces)


def test_their_scan_is_in_the_world_frame(room, caplog):
    assert room.source == "mesh" and "room mesh:" not in caplog.text   # Y up, floor at y = 0


def test_pixel_to_world_lands_on_their_markers(room, room_dir, markers):
    """A marker corner seen from the camera maps back to itself: room.glb and markers.json agree."""
    mesh = roomlib.Mesh.load(room_dir / "room.glb")
    checked = 0
    for marker_id, corners in markers.items():
        centre = corners.mean(axis=0)
        if not mesh.visible(roomlib.camera_center(R, T), centre[None])[0]:
            continue
        u, v = room.world_to_pixel(*centre)
        if 0 <= u < 1280 and 0 <= v < 720:
            assert np.linalg.norm(np.subtract(room.pixel_to_world(u, v), centre)) < 0.01, marker_id
            checked += 1
    assert checked >= 3


def test_conventions_match_roomlib(room, markers):
    pts = np.concatenate(list(markers.values()))
    assert np.allclose([room.world_to_pixel(*p) for p in pts], roomlib.project(K, DIST, R, T, pts), atol=1e-3)
    uv = np.array([[u, v] for u in np.linspace(100, 1180, 12) for v in np.linspace(80, 640, 8)])
    theirs = roomlib.floor_hit(K, DIST, R, T, uv)
    ours = np.array([room.floor_hit(u, v) or (np.nan,) * 3 for u, v in uv], float)
    ok = np.isfinite(theirs[:, 0])
    assert ok.sum() > 50 and np.array_equal(ok, np.isfinite(ours[:, 0]))
    assert np.nanmax(np.linalg.norm(ours[ok] - theirs[ok], axis=1)) < 0.002   # theirs undistorts less tightly
    assert np.array_equal(CV_TO_GL, roomlib.FLIP_YZ)


def test_their_solver_recovers_the_pose_from_our_projections(room, markers):
    """What register_camera.py does, minus the camera: the same pose comes back."""
    obj = np.concatenate(list(markers.values()))
    img = np.array([room.world_to_pixel(*p) for p in obj])
    R2, t2, rms, _ = roomlib.solve_pose(obj, img, K, DIST)
    assert rms < 0.01 and np.allclose(R2, R, atol=1e-5) and np.allclose(t2, T, atol=1e-4)


def test_round_trip_within_a_centimetre(room):
    """Points of the scan the camera sees: world -> pixel -> world."""
    xyz = room.xyz_map[::40, ::40].reshape(-1, 3)
    xyz = xyz[np.isfinite(xyz[:, 0])]
    assert len(xyz) > 300
    back = np.array([room.pixel_to_world(*room.world_to_pixel(*p)) for p in xyz], float)
    assert np.nanmax(np.linalg.norm(back - xyz, axis=1)) < 0.01


def test_xyz_map_floor_and_cache(room, room_dir):
    m = room.xyz_map
    assert m.shape == (720, 1280, 3) and (room_dir / "xyz_map.npy").exists()
    floor = np.abs(m[..., 1]) < 0.02
    assert floor.mean() > 0.1                                 # floor between the furniture
    again = Room.load(room_dir)
    assert again._cached_map() is not None                   # reused, not rebuilt


def test_rounded_rotation_from_register_camera_loads_quietly(room_dir, caplog):
    with caplog.at_level(logging.WARNING):
        Room.load(room_dir)
    assert "not a rotation" not in caplog.text


def test_gl_round_trip():
    T4 = opencv_to_gl(R, T)
    R2, t2 = gl_to_opencv(T4)
    assert np.allclose(R2, R) and np.allclose(t2, T)
    ahead = np.asarray(TARGET) - EYE
    cam = np.linalg.inv(T4)[:3, :3] @ ahead
    assert cam[2] < 0                                         # GL cameras look down -z


def test_without_a_mesh_everything_lands_on_the_floor(tmp_path):
    (tmp_path / "camera.json").write_text(json.dumps(camera_json()))
    room = Room.load(tmp_path)
    assert room.source == "floor"
    x, y, z = room.pixel_to_world(640, 600)
    assert abs(y) < 1e-9


def test_the_live_loop_fills_xyz_from_data_room(room_dir, tmp_path, markers):
    """The handoff end to end: data/room/ -> live.load_room -> Memory -> state.json xyz."""
    from fakes import FRAME, ITEMS
    from recall.live import load_room
    from recall.memory import Memory
    from recall.models import Sighting
    from recall.places import Places
    from recall.store import Store
    shutil.copytree(room_dir, tmp_path / "room")
    cfg = Settings(data_dir=tmp_path)
    room = load_room(cfg)
    floor_marker = markers[4].mean(axis=0)                   # marker 4 lies on the floor
    u, v = room.world_to_pixel(*floor_marker)
    mem = Memory(ITEMS, Places(), Store(tmp_path), cfg, pixel_to_world=room.pixel_to_world)
    mem.update([Sighting("keys", [u - 40, v - 30, u + 40, v], 0.6, 0)], FRAME, 0)
    xyz = json.loads((tmp_path / "state.json").read_text())["keys"]["xyz"]
    assert np.linalg.norm(np.subtract(xyz, floor_marker)) < 0.01


def test_no_room_or_a_bad_one_means_no_xyz(tmp_path):
    from recall.live import load_room
    assert load_room(Settings(data_dir=tmp_path)) is None
    (tmp_path / "room").mkdir()
    (tmp_path / "room" / "camera.json").write_text('{"width": 1280}')   # no pose: warned, demo goes on
    assert load_room(Settings(data_dir=tmp_path)) is None


def test_a_room_registered_at_another_size_is_not_used(room_dir, tmp_path, caplog):
    """camera.json is 1280x720; frames at 640x480 (RUNBOOK's CAPTURE fix for such a camera) would put
    every xyz somewhere else, so the room stays off rather than answer wrong."""
    from dataclasses import replace
    from recall.live import load_room
    shutil.copytree(room_dir, tmp_path / "room")
    cfg = Settings(data_dir=tmp_path)
    with caplog.at_level(logging.INFO):
        assert load_room(cfg).source == "mesh"
    assert "xyz on (source=mesh)" in caplog.text
    with caplog.at_level(logging.ERROR):
        assert load_room(replace(cfg, capture_w=640, capture_h=480)) is None
    assert "registered at 1280x720 but CAPTURE is 640x480" in caplog.text
