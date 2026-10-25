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
SPOTS = [(2.0, 1.5), (3.0, 2.5), (1.5, 3.0), (4.0, 0.5)]  # (x, z) where a whole person is in view
PAIRS = [(B.L_EYE, B.R_EYE), (B.L_SHOULDER, B.R_SHOULDER), (B.L_HIP, B.R_HIP), (B.L_ANKLE, B.R_ANKLE)]


@pytest.fixture(scope="module")
def room():
    return Room(CAMERA)  # no mesh: the floor shows behind every ankle


def toward_camera(room, spot, yaw):
    """cos between the way someone at spot faces and the way to the camera."""
    to_cam = room.center[[0, 2]] - np.array(spot)
    return (math.sin(yaw) * to_cam[0] + math.cos(yaw) * to_cam[1]) / np.linalg.norm(to_cam)


def keypoints(room, spot, yaw_deg, size=1.0, feet=True, swap=False):
    """COCO keypoints (17, 2) and confidences of someone at spot (x, z) facing yaw (forward =
    (sin yaw, cos yaw)), size times an adult's heights. The face shows only when turned toward the
    camera. swap: the pose model mixes up left and right."""
    yaw = math.radians(yaw_deg)
    f = np.array([math.sin(yaw), 0, math.cos(yaw)])
    r = np.array([-f[2], 0, f[0]])
    up, base = np.array([0, 1.0, 0]), np.array([spot[0], 0, spot[1]])
    world = {
        B.NOSE: base + 1.58 * size * up + 0.10 * f,
        B.L_EYE: base + 1.62 * size * up + 0.08 * f - 0.03 * r,
        B.R_EYE: base + 1.62 * size * up + 0.08 * f + 0.03 * r,
        B.L_SHOULDER: base + B.SHOULDER_Y * size * up - 0.20 * r,
        B.R_SHOULDER: base + B.SHOULDER_Y * size * up + 0.20 * r,
        B.L_HIP: base + B.HIP_Y * size * up - 0.15 * r,
        B.R_HIP: base + B.HIP_Y * size * up + 0.15 * r,
        B.L_ANKLE: base + B.ANKLE_Y * up - 0.10 * r,
        B.R_ANKLE: base + B.ANKLE_Y * up + 0.10 * r,
    }
    kp, conf = np.zeros((17, 2)), np.zeros(17)
    for i, p in world.items():
        kp[i] = room.world_to_pixel(*p)
        conf[i] = 0.9
    assert ((kp[list(world)] >= 0) & (kp[list(world)] < [room.width, room.height])).all(), "out of view"
    if toward_camera(room, spot, yaw) < 0.3:
        conf[[B.NOSE, B.L_EYE, B.R_EYE]] = 0.1
    if not feet:
        conf[[B.L_ANKLE, B.R_ANKLE]] = 0.1
    if swap:
        for a, b in PAIRS:
            kp[[a, b]], conf[[a, b]] = kp[[b, a]], conf[[b, a]]
    return kp, conf


def seen(room, spot, yaw_deg, frames=5, **kw):
    body = B.Body(room)
    for _ in range(frames):
        assert body.update(*keypoints(room, spot, yaw_deg, **kw))
    return body


def yaw_error(body, yaw_deg):
    return abs(math.degrees(B.wrap(body.yaw - math.radians(yaw_deg))))


def test_feet_place_the_person(room):
    for spot in SPOTS:
        body = seen(room, spot, 30)
        assert body.why.startswith("feet")
        assert np.linalg.norm(body.pos - spot) < 0.05


def test_hips_place_them_when_the_feet_are_hidden(room):
    for spot in SPOTS:
        body = seen(room, spot, 30, feet=False, size=0.93)  # a shorter person: the usual heights are off
        assert body.why.startswith("hips")
        # 6 cm too low a hip slides along a ray that drops 1.7 m over the whole range: ~4% of it
        assert np.linalg.norm(body.pos - spot) < 0.05 * np.linalg.norm(room.center[[0, 2]] - spot)


def test_facing_all_the_way_round(room):
    for spot in SPOTS:
        for yaw in range(0, 360, 15):
            assert yaw_error(seen(room, spot, yaw), yaw) < 15, (spot, yaw)


def test_hidden_face_corrects_swapped_labels(room):
    """From behind, the pose model's left and right are often swapped; the hidden face settles it."""
    spot, checked = SPOTS[0], set()
    for yaw in range(0, 360, 15):
        c = toward_camera(room, spot, math.radians(yaw))
        if abs(c) < B.SIDE_ON:
            continue
        body = seen(room, spot, yaw, swap=True)
        assert yaw_error(body, yaw) < 15, yaw
        checked.add(body.why.split(", ")[1])
    assert checked == {"face", "back"}


def test_turning_around_follows(room):
    """Facing the camera, then the back to it: the smoothed facing must swing round, not stick."""
    spot = SPOTS[0]
    yaw = math.degrees(math.atan2(*(room.center[[0, 2]] - spot)))  # straight at the camera
    body = seen(room, spot, yaw)
    for _ in range(15):
        body.update(*keypoints(room, spot, yaw + 180))
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


def box(center, along, half):
    """Vertices and triangles of a box on the floor: center (x, z), axes along (x, z) unit and
    square to it, half-sizes (along, across, height)."""
    a, b = np.array([along[0], 0, along[1]]), np.array([-along[1], 0, along[0]])
    c = np.array([center[0], 0, center[1]])
    v = np.array([c + sa * half[0] * a + sb * half[1] * b + [0, h, 0]
                  for h in (0, half[2]) for sa in (-1, 1) for sb in (-1, 1)])
    quads = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
    return v, np.array([t for q in quads for t in ((q[0], q[1], q[2]), (q[0], q[2], q[3]))])


def test_a_table_hiding_the_legs_means_hips(room):
    spot = np.array(SPOTS[0])
    to_cam = room.center[[0, 2]] - spot
    to_cam /= np.linalg.norm(to_cam)
    v, t = box(spot + 0.45 * to_cam, to_cam, (0.15, 0.5, 0.45))  # a coffee table between the camera and the legs
    body = B.Body(Room(CAMERA, v, t))
    assert body.update(*keypoints(room, tuple(spot), 30))
    assert body.why.startswith("hips")
    assert np.linalg.norm(body.pos - spot) < 0.05


def test_no_shoulders_means_lost(room):
    body = seen(room, SPOTS[0], 30)
    pos = body.pos.copy()
    kp, conf = keypoints(room, SPOTS[0], 90)
    conf[[B.L_SHOULDER, B.R_SHOULDER, B.L_HIP, B.R_HIP]] = 0.1
    assert not body.update(kp, conf)
    assert (body.pos == pos).all()
