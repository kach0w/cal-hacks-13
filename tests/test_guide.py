"""Checks the facing estimate and its words against simulated people.

A simulated person stands in front of the camera and turns, holds still, and walks
closer or farther. We build the COCO keypoints a pose model would report, feed them
to Guide, and compare its facing and its words with the truth.

Picture coordinates match guide.py: x right, y down = toward the camera.
Turn angle phi: 0 = facing the camera, +90 = facing picture-right, 180 = back to camera.

    .venv/bin/python tests/test_guide.py
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from recall import guide as G  # noqa: E402

BODY = np.array([640.0, 300.0])
WIDTH = 200.0   # shoulder width squarely facing the camera, at scale 1
TORSO = 260.0   # shoulder-to-hip height at scale 1; doesn't change when turning


def facing_vec(phi):
    a = math.radians(phi)
    return np.array([math.sin(a), math.cos(a)])


def person_right(phi):
    f = facing_vec(phi)
    return np.array([-f[1], f[0]])  # facing the camera, their right is picture-left


def keypoints(phi, scale=1.0, hips=True):
    f = facing_vec(phi)
    left_dir = -person_right(phi)
    w, t = WIDTH * scale, TORSO * scale
    kp, conf = np.zeros((17, 2)), np.zeros(17)
    kp[G.L_SHOULDER] = [BODY[0] + w / 2 * left_dir[0], BODY[1]]
    kp[G.R_SHOULDER] = [BODY[0] - w / 2 * left_dir[0], BODY[1]]
    conf[[G.L_SHOULDER, G.R_SHOULDER]] = 0.9
    if hips:
        kp[G.L_HIP] = [BODY[0] + w / 3 * left_dir[0], BODY[1] + t]
        kp[G.R_HIP] = [BODY[0] - w / 3 * left_dir[0], BODY[1] + t]
        conf[[G.L_HIP, G.R_HIP]] = 0.9
    if f[1] > 0.15:  # face visible only when turned toward the camera
        kp[G.NOSE] = [BODY[0] + 0.35 * w * f[0], BODY[1] - 0.4 * w]
        conf[G.NOSE] = 0.9
    return kp, conf


def angle_of(v):
    return math.degrees(math.atan2(v[0], v[1]))


def wrap(d):
    return (d + 180) % 360 - 180


def expected_words(rel):
    if abs(rel) < 25:
        return "Straight ahead"
    if abs(rel) > 135:
        return "Behind you"
    return "To your right" if rel > 0 else "To your left"


def check(name, poses, frames_per_pose=8):
    """poses: list of (phi, scale, hips). Returns (worst error by zone, word failures)."""
    g = G.Guide.__new__(G.Guide)  # skip loading the pose model
    g.facing = g.body = g.ref_width = g.ref_ratio = None
    g.turn_side = 1.0
    worst = {"front <=60": 0.0, "side 60-120": 0.0, "back >120": 0.0}
    fails, checks = [], 0
    for phi, scale, hips in poses:
        for _ in range(frames_per_pose):
            g.from_keypoints(*keypoints(phi, scale, hips))
        err = abs(wrap(angle_of(g.facing) - phi))
        zone = "front <=60" if abs(phi) <= 60 else "side 60-120" if abs(phi) <= 120 else "back >120"
        worst[zone] = max(worst[zone], err)
        for rel in (-90, -45, 0, 45, 90):
            a = math.radians(rel)
            target = BODY + 250 * scale * (math.cos(a) * facing_vec(phi) + math.sin(a) * person_right(phi))
            deg, dist = g.toward(target, 1468)
            checks += 1
            # Words within 10 degrees of a boundary can go either way; don't count those.
            if any(abs(abs(rel) - b) < 10 for b in (25, 135)):
                continue
            got = G.words(deg, dist)
            if got != expected_words(rel):
                fails.append((phi, rel, round(deg), got, expected_words(rel)))
    print(f"{name:42s} worst error {{{', '.join(f'{k}: {round(v)}' for k, v in worst.items())}}}  "
          f"words wrong {len(fails)}/{checks}")
    for f in fails[:5]:
        print("    phi %4d  target %+3d  got %+4d deg %-15s want %s" % f)
    return worst, fails


def sweep(limit, scale=1.0, hips=True):
    up = list(range(0, limit + 1, 5))
    return [(p, scale, hips) for p in [0] * 3 + up + up[::-1] + [-x for x in up] + [-x for x in up[::-1]]]


if __name__ == "__main__":
    for limit in (90, 135, 170):
        check(f"turn to +-{limit} and back", sweep(limit))
    check("hold turned 45 for a long time", [(0, 1, True)] * 3 + [(45, 1, True)] * 200)
    check("hold turned 45, no hips in view", [(0, 1, False)] * 3 + [(45, 1, False)] * 200)
    walk = [(0, 1.0, True)] * 3 + [(60, s, True) for s in np.linspace(1.0, 0.5, 40)] + \
           [(-60, s, True) for s in np.linspace(0.5, 1.4, 40)]
    check("turned 60, walking away then closer", walk)
    check("same walk, no hips in view (expect worse)", [(p, s, False) for p, s, _ in walk])
