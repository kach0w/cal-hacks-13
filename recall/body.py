"""Where the person is and which way their chest faces, in the scanned room (world frame: metres,
Y up, floor at y = 0; tools/roomlib.py). The band then points from there to the item.

One camera can't see depth, but a point at a known height can be placed: its pixel is a ray from
the camera, and the ray crosses that height at one spot (Room.plane_hit). Everything comes from
the two shoulders, placed at the person's shoulder height:

  where   midway between them
  facing  square to the line from the left shoulder to the right, along the floor, the way the
          pose model's left/right labels say. From behind the model can swap left and right, so
          when they face toward or away from the camera the face decides: seen = toward it.

Floor directions are (x, z). Angles are clockwise seen from above, so + = to the person's right.
"""
import math
import time

import numpy as np

NOSE, L_EYE, R_EYE, L_SHOULDER, R_SHOULDER = 0, 1, 2, 5, 6  # COCO keypoints
POSE_MODEL, POSE_IMGSZ = "yolo11s-pose.pt", 640  # shoulders and eyes across the room need more than n at 480
SHOULDER_CONF = 0.3    # a side-on far shoulder scores ~0.3-0.5; the width check catches bad ones
FACE_CONF = 0.5        # the nose and an eye this sure = the camera sees the face (60° off, the far eye hides)
WIDTH_M = (0.2, 0.6)   # shoulders this far apart once placed; otherwise a keypoint is off
SIDE_ON = 0.5          # |cos| between forward and the way to the camera below this = side-on (±30°)
LOST_S = 1.0           # unseen this long: the next look is taken as is
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
    """One person in the room: pos (x, z) and yaw (forward = (sin yaw, cos yaw)), both smoothed.
    shoulder_m: their shoulder height, about 0.82 x their height. smooth: 0..1, the share of each
    look taken (higher = turns show sooner, jitterier)."""

    def __init__(self, room, shoulder_m=1.40, smooth=0.5):
        self.room, self.shoulder_m, self.smooth = room, shoulder_m, smooth
        self.camera = room.center[[0, 2]]
        self.pos = self.yaw = None
        self.why = ""           # what decided the facing, for the overlay and the log
        self.face = (0.0, 0.0)  # the last look's nose and best eye confidence, for the log
        self.lost = ""          # why the last failed look failed, for the log
        self.seen_at = 0.0

    @property
    def forward(self):
        return np.array([math.sin(self.yaw), math.cos(self.yaw)])

    @property
    def right(self):
        f = self.forward
        return np.array([-f[1], f[0]])  # forward × up

    def update(self, kp, conf, now=None) -> bool:
        """One person's COCO keypoints (pixels) and confidences -> position and facing. False, with
        nothing changed, when their shoulders can't be placed. now: seconds (default: the clock)."""
        if conf[L_SHOULDER] < SHOULDER_CONF or conf[R_SHOULDER] < SHOULDER_CONF:
            self.lost = f"shoulder confidence {conf[L_SHOULDER]:.2f} / {conf[R_SHOULDER]:.2f}"
            return False
        left, right = self.room.plane_hit(np.asarray(kp)[[L_SHOULDER, R_SHOULDER]], self.shoulder_m)[:, [0, 2]]
        width = np.linalg.norm(right - left)
        if not WIDTH_M[0] <= width <= WIDTH_M[1]:  # NaN (a ray above the shoulders' height) fails too
            self.lost = f"shoulders {width:.2f} m apart"
            return False
        pos, r = (left + right) / 2, (right - left) / width
        forward = np.array([r[1], -r[0]])  # up × right
        to_camera = self.camera - pos
        c = forward @ to_camera / np.linalg.norm(to_camera)
        self.face = (float(conf[NOSE]), float(max(conf[L_EYE], conf[R_EYE])))
        if abs(c) < SIDE_ON:
            self.why = "side-on: labels"
        else:
            face = min(self.face) >= FACE_CONF
            if (c > 0) != face:
                forward = -forward
            self.why = "face" if face else "back"
        yaw = math.atan2(forward[0], forward[1])
        now = time.monotonic() if now is None else now
        if self.pos is None or now - self.seen_at > LOST_S:
            self.pos, self.yaw = pos, yaw
        else:  # the angle, not the vector: a vector blended with its opposite would never turn
            self.pos = self.pos + self.smooth * (pos - self.pos)
            self.yaw = wrap(self.yaw + self.smooth * wrap(yaw - self.yaw))
        self.seen_at = now
        return True

    def turn(self, target):
        """(degrees, dist 0..100, metres) from the person to world point target: + = to their
        right; dist FOUND_DIST or less = within arm's reach."""
        v = np.array([target[0], target[2]], np.float64) - self.pos
        metres = float(np.linalg.norm(v))
        deg = math.degrees(math.atan2(v @ self.right, v @ self.forward))
        return deg, min(100, round(metres / REACH_M * FOUND_DIST)), metres


def draw(img, room, body, target, title, deg, metres):
    """The guidance on the camera picture: a ring round the person at shoulder height, their facing
    as an arrow out of their chest, a line to the target, and the numbers. If the arrow points
    where their chest points, the facing is right. Drawn at the shoulders, not the floor: near the
    camera the floor under them is below the picture, and the shoulders are in it whenever they're seen."""
    import cv2
    from .room import project

    def pixels(points):
        pts = np.asarray(points, np.float64).reshape(-1, 3)
        if not room.in_lens(pts).all():
            return None
        return [tuple(int(c) for c in p) for p in project(pts, room.K, room.dist, room.R, room.t).round()]

    (x, z), y = body.pos, body.shoulder_m
    a = np.linspace(0, 2 * math.pi, 33)
    ring = pixels(np.c_[x + RING_M * np.sin(a), np.full_like(a, y), z + RING_M * np.cos(a)])
    if ring:
        cv2.polylines(img, [np.array(ring, np.int32)], True, GREEN, 2)
    f = body.forward
    arrow = pixels([[x, y, z], [x + ARROW_M * f[0], y, z + ARROW_M * f[1]]])
    if arrow:
        cv2.arrowedLine(img, arrow[0], arrow[1], GREEN, 3, tipLength=0.25)
    line = pixels([[x, y, z], target])
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
