"""recall/body.py on simulated people seen through the real corner camera (data/room/camera.json):
a skeleton stands at a known spot facing a known way, its keypoints are projected into the picture,
and the person is placed and turned from them alone."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from recall import body as B
from recall.room import Room

CAMERA = json.loads((Path(__file__).resolve().parent.parent / "data/room/camera.json").read_text())
# (x, z) on the floor: across the scanned room, and (0, 3.5) near the camera, where the legs are out of view
SPOTS = [(2.0, 1.5), (3.0, 2.5), (1.5, 3.0), (4.0, 0.5), (0.0, 3.5)]
PAIRS = [(B.L_EYE, B.R_EYE), (B.L_SHOULDER, B.R_SHOULDER)]
SHOULDER_M = 1.40
LOOK_S = 0.2  # between looks, as live guidance runs


@pytest.fixture(scope="module")
def room():
    return Room(CAMERA)


def toward_camera(room, spot, yaw):
    """cos between the way someone at spot faces and the way to the camera."""
    to_cam = room.center[[0, 2]] - np.array(spot)
    return (math.sin(yaw) * to_cam[0] + math.cos(yaw) * to_cam[1]) / np.linalg.norm(to_cam)


def keypoints(room, spot, yaw_deg, size=1.0, swap=False):
    """COCO keypoints (17, 2) and confidences of someone at spot (x, z) facing yaw (forward =
    (sin yaw, cos yaw)), size times SHOULDER_M tall at the shoulder. The face shows only when
    turned toward the camera. swap: the pose model mixes up left and right."""
    yaw = math.radians(yaw_deg)
    f = np.array([math.sin(yaw), 0, math.cos(yaw)])
    r = np.array([-f[2], 0, f[0]])
    up, base = np.array([0, 1.0, 0]), np.array([spot[0], 0, spot[1]])
    world = {
        B.NOSE: base + (SHOULDER_M + 0.18) * size * up + 0.10 * f,
        B.L_EYE: base + (SHOULDER_M + 0.22) * size * up + 0.08 * f - 0.03 * r,
        B.R_EYE: base + (SHOULDER_M + 0.22) * size * up + 0.08 * f + 0.03 * r,
        B.L_SHOULDER: base + SHOULDER_M * size * up - 0.20 * r,
        B.R_SHOULDER: base + SHOULDER_M * size * up + 0.20 * r,
    }
    kp, conf = np.zeros((17, 2)), np.zeros(17)
    for i, p in world.items():
        kp[i] = room.world_to_pixel(*p)
        conf[i] = 0.9
    assert ((kp[list(world)] >= 0) & (kp[list(world)] < [room.width, room.height])).all(), "out of view"
    if toward_camera(room, spot, yaw) < 0.3:
        conf[[B.NOSE, B.L_EYE, B.R_EYE]] = 0.1
    if swap:
        for a, b in PAIRS:
            kp[[a, b]], conf[[a, b]] = kp[[b, a]], conf[[b, a]]
    return kp, conf


def seen(room, spot, yaw_deg, frames=5, **kw):
    """A Body after `frames` looks at them, ending at now = (frames - 1) * LOOK_S."""
    body = B.Body(room, SHOULDER_M)
    for i in range(frames):
        assert body.update(*keypoints(room, spot, yaw_deg, **kw), now=i * LOOK_S)
    return body


def yaw_error(body, yaw_deg):
    return abs(math.degrees(B.wrap(body.yaw - math.radians(yaw_deg))))


def facing_off_camera(room, spot, deg):
    """The yaw that faces deg degrees to the right of straight at the camera."""
    return math.degrees(math.atan2(*(room.center[[0, 2]] - spot))) + deg


def test_shoulders_place_the_person(room):
    for spot in SPOTS:
        assert np.linalg.norm(seen(room, spot, 30).pos - spot) < 0.02


def test_a_wrong_shoulder_height_is_off_in_proportion(room):
    """7% shorter than SHOULDER_HEIGHT_M (10 cm): the shoulders slide along their rays, about 8%
    of the way to the camera; set the guided person's height."""
    for spot in SPOTS:
        body = seen(room, spot, 30, size=0.93)
        assert np.linalg.norm(body.pos - spot) < 0.1 * np.linalg.norm(room.center[[0, 2]] - spot)


def test_facing_all_the_way_round(room):
    for spot in SPOTS:
        for yaw in range(0, 360, 15):
            assert yaw_error(seen(room, spot, yaw), yaw) < 10, (spot, yaw)


def test_the_face_corrects_swapped_labels(room):
    """From behind, the pose model's left and right are often swapped; the face settles it."""
    spot, checked = SPOTS[0], set()
    for yaw in range(0, 360, 15):
        if abs(toward_camera(room, spot, math.radians(yaw))) < B.SIDE_ON:
            continue
        body = seen(room, spot, yaw, swap=True)
        assert yaw_error(body, yaw) < 10, yaw
        checked.add(body.why)
    assert checked == {"face", "back"}


def test_a_face_60_degrees_off_shows_one_eye(room):
    """Turned 60° from the camera the far eye hides behind the nose; that's still a face (live log:
    calling it 'back' turned the person round while they stood still)."""
    spot = SPOTS[0]
    for off in (-58, 58):
        yaw = facing_off_camera(room, spot, off)
        kp, conf = keypoints(room, spot, yaw)
        conf[B.R_EYE if off > 0 else B.L_EYE] = 0.1
        body = B.Body(room, SHOULDER_M)
        assert body.update(kp, conf)
        assert body.why == "face" and yaw_error(body, yaw) < 5


def test_turning_round_follows(room):
    spot = SPOTS[0]
    yaw = facing_off_camera(room, spot, 0)
    body, t = seen(room, spot, yaw), 1.0
    for step in range(1, 13):  # turning left at 75°/s, a look every 0.2 s
        body.update(*keypoints(room, spot, yaw + 15 * step), now=t)
        t += LOOK_S
    for _ in range(5):
        body.update(*keypoints(room, spot, yaw + 180), now=t)
        t += LOOK_S
    assert yaw_error(body, yaw + 180) < 5


def test_turn_to_the_item(room):
    spot, yaw = SPOTS[0], 40
    body = seen(room, spot, yaw)
    f = np.array([math.sin(math.radians(yaw)), math.cos(math.radians(yaw))])
    r = np.array([-f[1], f[0]])

    def at(v):
        x, z = np.array(spot) + v
        return [x, 0.45, z]  # on a table

    deg, dist, metres = body.turn(at(1.5 * r))
    assert abs(deg - 90) < 2 and abs(metres - 1.5) < 0.01
    assert abs(body.turn(at(-1.5 * r))[0] + 90) < 2
    deg, dist, _ = body.turn(at(2.0 * f))
    assert abs(deg) < 2 and dist == round(2.0 / B.REACH_M * B.FOUND_DIST)
    assert abs(body.turn(at(-2.0 * f))[0]) > 135
    assert body.turn(at(0.3 * f))[1] <= B.FOUND_DIST  # within arm's reach: the band's found


def test_no_shoulders_means_lost(room):
    body = seen(room, SPOTS[0], 30)
    pos = body.pos.copy()
    kp, conf = keypoints(room, SPOTS[0], 90)
    conf[[B.L_SHOULDER, B.R_SHOULDER]] = 0.1
    assert not body.update(kp, conf)
    assert (body.pos == pos).all()
