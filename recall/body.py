"""Where the person is and which way their chest faces, in the scanned room (world frame: metres,
Y up, floor at y = 0; tools/roomlib.py). The band then points from there to the item.

One camera can't see depth, but a body point at a known height can be placed: its pixel is a ray
from the camera, and the ray crosses that height at one spot (Room.plane_hit).

  where   the ankles, on the floor, when the scan says the floor shows there (a table in front of
          the legs would catch the ray first); else the hips, else the shoulders, at an adult's
          usual heights
  facing  the shoulders (and hips) placed in 3D: left to right is the person's right, and forward
          is square to that along the floor. Which way along it: a face the camera sees means
          toward the camera, no face means away. Side-on, where half a face shows either way, the
          pose model's left/right labels decide.

Floor directions are (x, z). Angles are clockwise seen from above, so + = to the person's right.
"""
import math

import numpy as np

NOSE, L_EYE, R_EYE = 0, 1, 2  # COCO keypoints
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_ANKLE, R_ANKLE = 5, 6, 11, 12, 15, 16
POSE_MODEL, POSE_IMGSZ = "yolo11s-pose.pt", 640  # ankles and eyes across the room need more than n at 480
JOINT_CONF = 0.3       # shoulders and hips; a side-on far shoulder scores ~0.3-0.5. The width check catches bad ones
ANKLE_CONF = 0.5       # ankles place the person outright, so they must be sure
FACE_CONF = 0.5        # nose and both eyes this sure = the camera sees the face
ANKLE_Y, HIP_Y, SHOULDER_Y = 0.08, 0.92, 1.40  # an adult's, metres
FLOOR_SEEN_Y = 0.15    # the scan behind an ankle is lower than this: nothing hides the floor there
WIDTH_M = (0.15, 0.6)  # shoulders or hips this far apart in 3D; otherwise a keypoint is off
SIDE_ON = 0.5          # |cos| between forward and the way to the camera below this = side-on (±30°)
POS_SMOOTH, YAW_SMOOTH = 0.5, 0.35  # 0..1, higher = snappier, jitterier
JUMP_M = 1.0           # a position this far from the last is taken as is, not blended
REACH_M = 0.6          # arm's reach: what the band counts as found
FOUND_DIST = 8         # the band's found distance (firmware/band FOUND_DIST, guide.FOUND_DIST)
RING_M, ARROW_M = 0.25, 0.8  # overlay: the ring under the person, their facing arrow on the floor
GREEN, ORANGE, WHITE = (0, 255, 0), (0, 200, 255), (255, 255, 255)


def wrap(a):
    """Angle in radians -> -pi..pi."""
    return (a + math.pi) % (2 * math.pi) - math.pi


class PoseModel:
    """The pose model; loads (and downloads, once) when guidance first needs it."""

    def __init__(self, path=POSE_MODEL):
        from ultralytics import YOLO
        self.model = YOLO(path)

    def largest(self, frame):
        """(keypoints (17, 2) in pixels, their confidences (17,)) of the biggest person, or None."""
        r = self.model(frame, imgsz=POSE_IMGSZ, verbose=False)[0]
        if r.keypoints is None or r.keypoints.conf is None or not len(r.boxes):
            return None
        b = r.boxes.xyxy.cpu().numpy()
        i = int(np.argmax((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])))
        return r.keypoints.xy[i].cpu().numpy(), r.keypoints.conf[i].cpu().numpy()


class Body:
    """One person in the room: pos (x, z) and yaw (forward = (sin yaw, cos yaw)), both smoothed."""

    def __init__(self, room):
        self.room = room
        self.camera = room.center[[0, 2]]
        self.pos = None
        self.yaw = None
        self.why = ""  # how they were placed and turned, for the overlay

    @property
    def forward(self):
        return np.array([math.sin(self.yaw), math.cos(self.yaw)])

    @property
    def right(self):
        f = self.forward
        return np.array([-f[1], f[0]])  # forward × up

    def update(self, kp, conf) -> bool:
        """One person's COCO keypoints (pixels) and confidences -> position and facing. False, with
        nothing changed, when they can't be both placed and turned."""
        placed, right = self._place(kp, conf), self._right_axis(kp, conf)
        if placed is None or right is None:
            return False
        pos, where = placed
        forward = np.array([right[1], -right[0]])  # up × right: forward, if the labels are right
        to_camera = self.camera - pos
        c = forward @ to_camera / max(np.linalg.norm(to_camera), 1e-6)
        if abs(c) < SIDE_ON:
            how = "side-on labels"
        else:
            face = min(conf[NOSE], conf[L_EYE], conf[R_EYE]) >= FACE_CONF
            if (c > 0) != face:
                forward = -forward
            how = "face" if face else "back"
        yaw = math.atan2(forward[0], forward[1])
        if self.pos is None or np.linalg.norm(pos - self.pos) > JUMP_M:
            self.pos, self.yaw = pos, yaw
        else:  # the angle, not the vector: a vector blended with its opposite would never turn
            self.pos = self.pos + POS_SMOOTH * (pos - self.pos)
            self.yaw = wrap(self.yaw + YAW_SMOOTH * wrap(yaw - self.yaw))
        self.why = f"{where}, {how}"
        return True

    def turn(self, target):
        """(degrees, dist 0..100, metres) from the person to world point target: + = to their
        right; dist FOUND_DIST or less = within arm's reach."""
        v = np.array([target[0], target[2]], np.float64) - self.pos
        metres = float(np.linalg.norm(v))
        deg = math.degrees(math.atan2(v @ self.right, v @ self.forward))
        return deg, min(100, round(metres / REACH_M * FOUND_DIST)), metres

    def _pair(self, kp, conf, a, b, y, min_conf):
        """Keypoints a and b placed at height y: (2, 2) floor points (x, z), or None."""
        if conf[a] < min_conf or conf[b] < min_conf:
            return None
        p = self.room.plane_hit(np.asarray(kp)[[a, b]], y)
        return None if np.isnan(p).any() else p[:, [0, 2]]

    def _place(self, kp, conf):
        """((x, z), what placed them): the ankles if the floor shows behind them, else hips, else shoulders."""
        feet = self._pair(kp, conf, L_ANKLE, R_ANKLE, ANKLE_Y, ANKLE_CONF)
        if feet is not None and all(self._floor_shows(*kp[i]) for i in (L_ANKLE, R_ANKLE)):
            return feet.mean(0), "feet"
        for a, b, y, name in ((L_HIP, R_HIP, HIP_Y, "hips"), (L_SHOULDER, R_SHOULDER, SHOULDER_Y, "shoulders")):
            p = self._pair(kp, conf, a, b, y, JOINT_CONF)
            if p is not None:
                return p.mean(0), name
        return None

    def _floor_shows(self, u, v):
        """The scan behind pixel (u, v) is floor: nothing (a table, the sofa) hides an ankle there."""
        hit = self.room.pixel_to_world(float(u), float(v))
        return hit is not None and hit[1] < FLOOR_SEEN_Y

    def _right_axis(self, kp, conf):
        """The person's right along the floor, unit (x, z), from the shoulders and hips placed in 3D
        as the pose model labels them. None if neither pair is believable."""
        axes = []
        for a, b, y in ((L_SHOULDER, R_SHOULDER, SHOULDER_Y), (L_HIP, R_HIP, HIP_Y)):
            p = self._pair(kp, conf, a, b, y, JOINT_CONF)
            if p is not None:
                d = p[1] - p[0]
                w = np.linalg.norm(d)
                if WIDTH_M[0] <= w <= WIDTH_M[1]:
                    axes.append(d / w)
        if not axes:
            return None
        if len(axes) == 2 and axes[0] @ axes[1] < 0:  # the pairs disagree: trust the shoulders
            axes = axes[:1]
        r = sum(axes)
        return r / np.linalg.norm(r)


def draw(img, room, body, target, title, deg, metres):
    """The guidance on the camera picture: a ring on the floor under the person, their facing as an
    arrow on the floor, a line to the target, and the numbers. If the arrow on the floor points
    where their chest points, the facing is right."""
    import cv2
    from .room import project

    def pixels(points):
        pts = np.asarray(points, np.float64).reshape(-1, 3)
        if not room.in_lens(pts).all():
            return None
        return [tuple(int(c) for c in p) for p in project(pts, room.K, room.dist, room.R, room.t).round()]

    x, z = body.pos
    a = np.linspace(0, 2 * math.pi, 33)
    ring = pixels(np.c_[x + RING_M * np.sin(a), np.zeros_like(a), z + RING_M * np.cos(a)])
    if ring:
        cv2.polylines(img, [np.array(ring, np.int32)], True, GREEN, 2)
    f = body.forward
    arrow = pixels([[x, 0, z], [x + ARROW_M * f[0], 0, z + ARROW_M * f[1]]])
    if arrow:
        cv2.arrowedLine(img, arrow[0], arrow[1], GREEN, 3, tipLength=0.25)
    line = pixels([[x, 0, z], target])
    if line:
        cv2.line(img, line[0], line[1], ORANGE, 2)
        cv2.circle(img, line[1], 10, ORANGE, 2)
    lines = [f"you       x {x:+.2f}  z {z:+.2f} m  ({body.why})",
             f"turn      {abs(deg):.0f} deg {'right' if deg > 0 else 'left'}",
             f"distance  {metres:.1f} m"]
    for i, (text, scale, color) in enumerate([(title, 1.0, ORANGE)] + [(s, 0.6, WHITE) for s in lines]):
        at = (16, 40 if i == 0 else 54 + 26 * i)
        cv2.putText(img, text, at, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5)  # outline: readable on a bright window
        cv2.putText(img, text, at, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)
