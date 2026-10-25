"""Which way the person should go, from one camera in front of them.

Finds the person's shoulders, hips and nose with a pose model, works out which way
they're turned, and gives the direction from them to a target point in the picture.

The picture is used as a rough floor map: x is left/right and down counts as toward
the camera. How far someone is turned comes from how narrow their shoulders look
compared to their width when squarely facing the camera. Whether they face the camera
comes from seeing their nose, and which side they're turned to from where the nose
sits between the shoulders. With their back to the camera the face is hidden, so
they're assumed to still be turned to the side they last turned to.
"""

import math

import numpy as np
from ultralytics import YOLO

NOSE, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 0, 5, 6, 11, 12  # COCO keypoints
MIN_CONF = 0.5
SQUARE_OFFSET = 0.08  # nose this close to centre (x shoulder width) = squarely facing the camera
REF_LEARN = 0.2       # how fast the squarely-facing reference updates
SMOOTH = 0.25         # 0..1, higher = snappier facing
FOUND_DIST = 8        # dist (0..100) that counts as "right here"
SHOULDER_M = 0.40     # typical adult shoulder width, our ruler for metres


class Guide:
    def __init__(self):
        self.model = YOLO("yolo11n-pose.pt")
        self.facing = None      # unit vector out of the chest, picture coordinates
        self.body = None        # point between the shoulders
        self.ref_width = None   # shoulder width when squarely facing the camera
        self.ref_ratio = None   # that width divided by torso height, survives moving closer/farther
        self.turn_side = 1.0    # which side they last turned to, for when the face is hidden
        self.full_px = None     # current full shoulder width in pixels, for metres

    def update(self, frame):
        """Track the biggest person in frame. Returns False if nobody's there."""
        r = self.model(frame, imgsz=480, verbose=False)[0]
        if r.keypoints is None or r.keypoints.conf is None or len(r.boxes) == 0:
            self.facing = self.body = None
            return False
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in r.boxes.xyxy.tolist()]
        i = int(np.argmax(areas))
        kp = r.keypoints.xy[i].cpu().numpy()
        conf = r.keypoints.conf[i].cpu().numpy()
        return self.from_keypoints(kp, conf)

    def from_keypoints(self, kp, conf):
        """Update facing and position from one person's COCO keypoints."""
        if conf[L_SHOULDER] < MIN_CONF or conf[R_SHOULDER] < MIN_CONF:
            self.facing = self.body = None
            return False
        left, right = kp[L_SHOULDER], kp[R_SHOULDER]
        width = abs(right[0] - left[0])
        mid = (left + right) / 2
        torso = None
        if conf[L_HIP] >= MIN_CONF and conf[R_HIP] >= MIN_CONF:
            torso = abs((kp[L_HIP][1] + kp[R_HIP][1]) / 2 - mid[1])
        nose_seen = conf[NOSE] >= MIN_CONF
        offset = kp[NOSE][0] - mid[0] if nose_seen else 0.0

        # Learn "full width" only from frames where they squarely face the camera.
        if nose_seen and abs(offset) < SQUARE_OFFSET * max(width, 1.0):
            self.ref_width = width if self.ref_width is None else \
                self.ref_width + REF_LEARN * (width - self.ref_width)
            if torso and torso > 1:
                ratio = width / torso
                self.ref_ratio = ratio if self.ref_ratio is None else \
                    self.ref_ratio + REF_LEARN * (ratio - self.ref_ratio)
        full = self.full_width(torso)
        self.full_px = full
        if full is None:  # never seen them square on yet; can't judge the turn
            self.facing, self.body = None, mid
            return False

        c = min(1.0, width / full)                  # |cos(turn)|
        side = math.sqrt(max(0.0, 1 - c * c))       # |sin(turn)|
        if nose_seen:
            if abs(offset) < SQUARE_OFFSET * full:
                side = 0.0
            elif side > 0.2:
                self.turn_side = math.copysign(1.0, offset)
            f = np.array([math.copysign(side, offset) if side else 0.0, c])
        else:
            f = np.array([self.turn_side * side, -c])
        f /= max(np.linalg.norm(f), 1e-6)

        # Smooth the vector, not the angle, so it never jumps at +-180.
        self.facing = f if self.facing is None else self.facing + SMOOTH * (f - self.facing)
        self.facing /= max(np.linalg.norm(self.facing), 1e-6)
        self.body = mid
        return True

    def full_width(self, torso):
        if self.ref_ratio is not None and torso and torso > 1:
            return self.ref_ratio * torso
        return self.ref_width

    def toward(self, target, frame_diag):
        """(degrees, dist 0..100) from the person to target; + = to their right. None if lost."""
        if self.facing is None or self.body is None:
            return None
        v = np.asarray(target, dtype=float) - self.body
        d = np.linalg.norm(v)
        v /= max(d, 1e-6)
        cross = self.facing[0] * v[1] - self.facing[1] * v[0]
        deg = math.degrees(math.atan2(cross, self.facing @ v))
        return deg, min(100, int(d / (frame_diag * 0.5) * 100))


    def metres(self, target):
        """Rough distance to target in metres, measured in shoulder widths. Ignores depth."""
        if self.body is None or not self.full_px:
            return None
        # float() at the end: full_px can be numpy.float32, which json can't write (NumPy 2 keeps the float32)
        return float(np.linalg.norm(np.asarray(target) - self.body) / self.full_px * SHOULDER_M)

    def facing_degrees(self):
        """Which way they face in the picture's floor map: 0 = at the camera, +90 = picture-right."""
        return None if self.facing is None else math.degrees(math.atan2(self.facing[0], self.facing[1]))


def words(deg, dist):
    if dist <= FOUND_DIST:
        return "Right here"
    if abs(deg) < 25:
        return "Straight ahead"
    if abs(deg) > 135:
        return "Behind you"
    return "To your right" if deg > 0 else "To your left"
