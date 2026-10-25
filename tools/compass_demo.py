"""Camera -> necklace arrow, with a clicked fake target.

Tracks the person's shoulders with a pose model, works out which way their body faces,
and sends the direction to the clicked target to the necklace as "dir,dist" (recall/band.py:
Bluetooth to the ESP32 band, or wifi to a host such as the Raspberry Pi band).

    .venv/bin/python tools/compass_demo.py                       # phone only, camera in front
    .venv/bin/python tools/compass_demo.py --view top            # camera looking straight down
    .venv/bin/python tools/compass_demo.py --find "keys,glasses,bag"   # real objects, keys 1-9 pick one
    .venv/bin/python tools/compass_demo.py --find "keys,wallet" --rooms "kitchen:1,bedroom:0"   # + Clue board at /board
    .venv/bin/python tools/compass_demo.py sixsevendemons
    .venv/bin/python tools/compass_demo.py raspberrypi.local --cam 1

Click the video to set the target. Keys: c = clear target, f = flip facing,
m = mirror left/right (camera facing the person instead of looking down),
r = reverse only the servo (if it points opposite the yellow dial needle), q = quit.
Sends nothing when it loses the person or has no target, so the arrow rests.
"""

import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO

import phone_compass
from room_status import RoomStatus

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from recall.band import connect  # noqa: E402

SEND_HZ = 15
NOSE, L_EAR, R_EAR, L_SHOULDER, R_SHOULDER = 0, 3, 4, 5, 6  # COCO keypoint indexes
NOSE_MIN_OFFSET = 0.08  # nose must stick out this much (x shoulder width) to count
WIDTH_DECAY = 0.998  # forget the widest-shoulders reference slowly, in case you step back
MIN_KP_CONF = 0.5
FACING_SMOOTH = 0.2  # 0..1, higher = snappier
MIN_SHOULDER_FRAC = 0.04  # shoulders narrower than this (x frame width) are too noisy to trust
HOLD_S = 1.0  # keep the last good facing this long through bad frames
FLIP_CONFIRM = 5  # a >100 degree jump must repeat this many frames before we believe it
MAGENTA_HSV = ((140, 100, 80), (170, 255, 255))  # OpenCV hue is 0..179
CYAN_HSV = ((85, 100, 80), (100, 255, 255))
CARD_MIN_AREA = 150
ITEMS_FILE = os.path.join(os.path.dirname(__file__), "items.json")
DETECT_EVERY = 5  # run the object detector every Nth frame; objects sit still
DETECT_IMGSZ = 640
DETECT_CONF = 0.15
SERVO_FORWARD, SERVO_REST, SERVO_REVERSED = 90, 180, True  # keep in sync with band.ino

target = None
guide_name = None  # object we're guiding to, chosen with keys 1-9


def on_click(event, x, y, flags, param):
    global target, guide_name
    if event == cv2.EVENT_LBUTTONDOWN:
        target = np.array([x, y], dtype=float)
        guide_name = None  # a click means "point here", not at an object


def find_person(result):
    """Keypoints and confidences of the biggest person in frame, or None."""
    if result.keypoints is None or result.keypoints.conf is None or len(result.boxes) == 0:
        return None
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in result.boxes.xyxy.tolist()]
    i = int(np.argmax(areas))
    kp = result.keypoints.xy[i].cpu().numpy()
    conf = result.keypoints.conf[i].cpu().numpy()
    if conf[L_SHOULDER] < MIN_KP_CONF or conf[R_SHOULDER] < MIN_KP_CONF:
        return None
    return kp, conf


def facing_front_view(kp, conf, max_width, prev):
    """Facing for a camera in front of the person, at about chest height.

    Uses the image as a rough floor map: x = left/right, down = toward the camera.
    How much you're turned comes from how narrow your shoulders look compared to
    their widest; whether you face the camera comes from seeing your nose; which
    side you're turned to comes from where the nose sits between the shoulders.
    """
    left, right = kp[L_SHOULDER], kp[R_SHOULDER]
    width = abs(right[0] - left[0])
    if max_width <= 0:
        return None
    c = min(1.0, width / max_width)          # |cos(turn)|
    side = math.sqrt(max(0.0, 1 - c * c))    # |sin(turn)|
    mid_x = (left[0] + right[0]) / 2
    toward = conf[NOSE] >= MIN_KP_CONF

    if toward:
        offset = kp[NOSE][0] - mid_x
        sx = 1.0 if offset > 0 else -1.0
        if abs(offset) < NOSE_MIN_OFFSET * max_width:
            side = 0.0
        f = np.array([sx * side, c])
    else:
        sx = 1.0 if prev is not None and prev[0] >= 0 else -1.0
        f = np.array([sx * side, -c])
    n = np.linalg.norm(f)
    return f / n if n > 1e-6 else None


def facing_from_pose(kp, conf, prev):
    """Unit vector out of the chest, or None if we can't tell front from back.

    The shoulder line only gives the body's axis; front and back look the same.
    The nose sticks out in front of the head, so it picks the side. With no nose
    in view (turned away), keep whichever side is closest to last frame's facing.
    """
    left, right = kp[L_SHOULDER], kp[R_SHOULDER]
    s = right - left
    width = np.linalg.norm(s)
    if width < 1e-6:
        return None
    axis = np.array([s[1], -s[0]]) / width  # perpendicular, sign unknown yet

    if conf[NOSE] >= MIN_KP_CONF:
        if conf[L_EAR] >= MIN_KP_CONF and conf[R_EAR] >= MIN_KP_CONF:
            head = (kp[L_EAR] + kp[R_EAR]) / 2
        else:
            head = (left + right) / 2
        v = kp[NOSE] - head
        if np.linalg.norm(v) > NOSE_MIN_OFFSET * width:
            return axis if axis @ v > 0 else -axis

    if prev is not None:
        return axis if axis @ prev >= 0 else -axis
    return None


def find_dot(hsv, lo, hi):
    mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < CARD_MIN_AREA:
        return None
    m = cv2.moments(c)
    return np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]])


def find_card(frame):
    """Front (magenta) and back (cyan) dots of the desk-test card, or None."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    front = find_dot(hsv, *MAGENTA_HSV)
    back = find_dot(hsv, *CYAN_HSV)
    if front is None or back is None or np.linalg.norm(front - back) < 5:
        return None
    return front, back


def load_items(spec):
    """Items to track -> the phrasings YOLO-World is given for each.

    spec "" uses every item in items.json. Otherwise a comma list; names found in
    items.json get its phrasings, anything else is used as its own phrasing.
    """
    with open(ITEMS_FILE) as f:
        known = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    names = [n.strip() for n in spec.split(",") if n.strip()] or list(known)
    return {n: known.get(n, [n]) for n in names}


class ObjectFinder:
    """YOLO-World told several phrasings per item. Runs every few frames; remembers the
    best box per item and when it was last seen."""

    def __init__(self, items, model_path, conf=DETECT_CONF):
        self.names = list(items)
        self.conf = conf
        self.prompts, self.prompt_item = [], []
        for name, phrasings in items.items():
            for p in phrasings:
                self.prompts.append(p)
                self.prompt_item.append(name)
        self.model = YOLO(model_path)
        self.model.set_classes(self.prompts)
        self.seen = {}  # name -> (box xyxy, conf, time)

    def detect(self, frame):
        """Best box per listed name in this frame: {name: (box xyxy, conf)}."""
        r = self.model(frame, imgsz=DETECT_IMGSZ, conf=self.conf, verbose=False)[0]
        best = {}
        for box, cls, conf in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            name = self.prompt_item[int(cls)]
            if name not in best or conf > best[name][1]:
                best[name] = (box, conf)
        return best

    def update(self, frame):
        """Detect in the guidance camera's frame and remember it for drawing."""
        best = self.detect(frame)
        now = time.time()
        for name, (box, conf) in best.items():
            self.seen[name] = (box, conf, now)
        return best

    def where(self, name):
        """(centre point, seconds since seen) or None if never seen."""
        if name not in self.seen:
            return None
        box, _, t = self.seen[name]
        return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]), time.time() - t

    def draw(self, frame, selected):
        now = time.time()
        for name, (box, conf, t) in self.seen.items():
            if now - t > 2:
                continue
            colour = (0, 200, 255) if name == selected else (200, 200, 200)
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(frame, f"{name} {conf:.2f}", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)


def snapshot_jpeg(frame):
    """Small JPEG of a room for the board's tiles."""
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (480, int(480 * h / w)))
    return cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 60])[1].tobytes()


def draw_gauge(frame, direction):
    """Top-right dial showing what the necklace arrow should be doing.

    Drawn from the wearer's point of view: up = straight out of the chest.
    direction None = no target or lost, so the arrow is parked at rest.
    """
    h, w = frame.shape[:2]
    c = np.array([w - 110, 120])
    r = 80
    cv2.ellipse(frame, tuple(c), (r, r), 0, 180, 360, (255, 255, 255), 2)
    cv2.putText(frame, "L", (c[0] - r - 22, c[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "R", (c[0] + r + 8, c[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "ahead", (c[0] - 28, c[1] - r - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if direction is None:
        cv2.putText(frame, f"resting (servo {SERVO_REST})", (c[0] - 95, c[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
        return
    shown = max(-90, min(90, direction))  # the servo can't point behind
    a = math.radians(shown)
    tip = c + np.array([math.sin(a), -math.cos(a)]) * r
    cv2.arrowedLine(frame, tuple(c), tuple(tip.astype(int)), (0, 200, 255), 4)
    servo = int(max(0, min(180, SERVO_FORWARD + (-direction if SERVO_REVERSED else direction))))
    behind = "  (behind you)" if abs(direction) > 90 else ""
    cv2.putText(frame, f"servo {servo}{behind}", (c[0] - 60, c[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)


def signed_angle(facing, to_target):
    """Degrees from facing to to_target, positive = clockwise on screen."""
    cross = facing[0] * to_target[1] - facing[1] * to_target[0]
    dot = facing @ to_target
    return math.degrees(math.atan2(cross, dot))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("band", nargs="?", default=None,
                    help="the band's Bluetooth name (sixsevendemons) or a wifi host; leave out for phone only")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--model", default="yolo11n-pose.pt")
    ap.add_argument("--find", default="",
                    help='items to track, e.g. "keys,phone,bag"; leave out to track everything in items.json. '
                         'Keys 1-9 pick which one to guide to')
    ap.add_argument("--conf", type=float, default=DETECT_CONF,
                    help="detection confidence; lower finds more but guesses more (try 0.08-0.3)")
    ap.add_argument("--no-find", action="store_true", help="no object detection, click a target instead")
    ap.add_argument("--world-model", default="yolov8s-worldv2.pt",
                    help="yolov8m-worldv2.pt or yolov8l-worldv2.pt are more accurate but slower")
    ap.add_argument("--rooms", default="",
                    help='one camera per room, e.g. "kitchen:1,bedroom:0". The first room is where guidance happens')
    ap.add_argument("--view", choices=["front", "top"], default="front",
                    help="front = camera in front of the person (default), top = camera looking straight down")
    ap.add_argument("--card", action="store_true", help="desk test: track a paper card (magenta front, cyan back) instead of a person")
    ap.add_argument("--mirror", action="store_true", help="swap left/right, for a camera facing the person")
    args = ap.parse_args()

    link = connect(args.band) if args.band else None  # looks for the band in the background
    phone_url = phone_compass.start()
    model = None if args.card else YOLO(args.model)
    items = {} if args.no_find else load_items(args.find)
    names = list(items)
    finder = ObjectFinder(items, args.world_model, args.conf) if items else None
    if finder is not None:
        print(f"tracking {len(names)} items with {len(finder.prompts)} phrasings: {', '.join(names)}")

    rooms = []  # (name, camera index); the first one is the guidance camera
    for part in filter(None, (p.strip() for p in args.rooms.split(","))):
        room, _, idx = part.partition(":")
        rooms.append((room.strip(), int(idx)))
    if not rooms:
        rooms = [("room", args.cam)]
    guide_room = rooms[0][0]
    cap = cv2.VideoCapture(rooms[0][1])
    if not cap.isOpened():
        raise SystemExit(f"can't open camera {rooms[0][1]}; try another number, and allow camera access for Terminal")
    other_caps = []
    for room, idx in rooms[1:]:
        c = cv2.VideoCapture(idx)
        if not c.isOpened():
            raise SystemExit(f"can't open camera {idx} for the {room}")
        other_caps.append((room, c))

    room_status = RoomStatus(names) if finder is not None else None
    if room_status is not None:
        phone_compass.attach_board(room_status, [r for r, _ in rooms])

    cv2.namedWindow("compass")
    cv2.setMouseCallback("compass", on_click)
    global target, guide_name
    frame_i = 0
    facing = None
    flip = False
    mirror = args.mirror
    band_ok = False
    servo_flip = False  # r key: reverse only the servo, for when it mirrors the dial
    last_send = 0.0
    last_good = 0.0
    max_width = 0.0  # widest shoulders seen, i.e. squarely facing or backing the camera
    jump_frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        diag = math.hypot(w, h)

        for _, c in other_caps:
            c.grab()  # keep the other rooms' buffers fresh; decode only when we detect

        guide_note = None
        if finder is not None:
            if frame_i % DETECT_EVERY == 0:
                now = time.time()
                for name, (box, conf) in finder.update(frame).items():
                    room_status.report(guide_room, name, box, conf, now)
                phone_compass.set_snapshot(guide_room, snapshot_jpeg(frame))
                for room, c in other_caps:
                    ok_other, other = c.retrieve()
                    if not ok_other:
                        continue
                    for name, (box, conf) in finder.detect(other).items():
                        room_status.report(room, name, box, conf, now)
                    phone_compass.set_snapshot(room, snapshot_jpeg(other))
            if guide_name is not None:
                info = room_status.get(guide_name)
                if info is None:
                    target, guide_note = None, "not seen yet"
                elif info["room"] != guide_room:
                    target, guide_note = None, f"in the {info['room']}"
                else:
                    # Keep pointing at the last place we saw it in this room.
                    b = info["box"]
                    target = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
        frame_i += 1

        status = "no person"
        sent_dir = None
        if args.card:
            # Desk test: a paper card stands in for the person. Magenta dot = chest, cyan = back.
            person = find_card(frame)
            if person is not None:
                front, back = person
                body = (front + back) / 2
                f = (front - back) / np.linalg.norm(front - back)
                facing = -f if flip else f
                cv2.circle(frame, tuple(front.astype(int)), 8, (255, 0, 255), -1)
                cv2.circle(frame, tuple(back.astype(int)), 8, (255, 255, 0), -1)
            else:
                status = "no card: need the magenta + cyan dots in view"
        else:
            result = model(frame, imgsz=480, verbose=False)[0]
            person = find_person(result)
            if person is not None:
                kp, conf = person
                left, right = kp[L_SHOULDER], kp[R_SHOULDER]
                body = (left + right) / 2
                now = time.time()
                f = None
                if np.linalg.norm(right - left) >= MIN_SHOULDER_FRAC * w:
                    prev_raw = None if facing is None else (-facing if flip else facing)
                    if args.view == "front":
                        max_width = max(max_width * WIDTH_DECAY, abs(right[0] - left[0]))
                        f = facing_front_view(kp, conf, max_width, prev_raw)
                    else:
                        f = facing_from_pose(kp, conf, prev_raw)
                    if f is not None and flip:
                        f = -f
                if f is not None and facing is not None and f @ facing < math.cos(math.radians(100)):
                    # Big sudden jump: usually a bad keypoint frame. Believe it only if it sticks.
                    jump_frames += 1
                    if jump_frames < FLIP_CONFIRM:
                        f = None
                else:
                    jump_frames = 0
                if f is not None:
                    # Smooth the facing vector, not the angle, so it never jumps at +-180.
                    facing = f if facing is None else facing + FACING_SMOOTH * (f - facing)
                    facing /= np.linalg.norm(facing)
                    last_good = now
                elif facing is not None and now - last_good > HOLD_S:
                    facing = None
                if facing is None:
                    status = "can't tell front from back yet: face the camera once"

                cv2.circle(frame, tuple(left.astype(int)), 8, (255, 0, 255), -1)
                cv2.circle(frame, tuple(right.astype(int)), 8, (255, 255, 0), -1)
                if conf[NOSE] >= MIN_KP_CONF:
                    cv2.circle(frame, tuple(kp[NOSE].astype(int)), 6, (0, 0, 255), -1)


        if person is not None and facing is not None:
            tip = body + facing * 120
            cv2.arrowedLine(frame, tuple(body.astype(int)), tuple(tip.astype(int)), (0, 255, 0), 3)

            if target is None:
                status = "click to set a target"
            else:
                to_t = target - body
                d = np.linalg.norm(to_t)
                direction = signed_angle(facing, to_t / max(d, 1e-6))
                if mirror:
                    direction = -direction
                dist = min(100, int(d / (diag * 0.5) * 100))
                status = f"dir {direction:+.0f}  dist {dist}"
                sent_dir = direction
                cv2.line(frame, tuple(body.astype(int)), tuple(target.astype(int)), (0, 200, 255), 2)
                now = time.time()
                if now - last_send >= 1 / SEND_HZ:
                    servo_dir = -direction if servo_flip else direction
                    if link is not None:
                        link.send(servo_dir, dist)
                    band_ok = link is not None and link.connected  # off or out of range: phone only
                    last_send = now
        elif person is None:
            facing = None

        if finder is not None:
            finder.draw(frame, guide_name)
            menu = "  ".join(f"{i + 1}:{n}" for i, n in enumerate(names[:9]))
            cv2.putText(frame, f"guide to -> {menu}", (16, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if guide_name is not None:
                info = room_status.get(guide_name)
                if info is None:
                    note = "not seen yet"
                else:
                    age = time.time() - info["last_seen"]
                    note = f"{info['room']}, " + ("seen now" if age < 2 else f"last seen {age:.0f}s ago")
                cv2.putText(frame, f"guiding to: {guide_name} ({note})", (16, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        if target is not None:
            cv2.drawMarker(frame, tuple(target.astype(int)), (0, 200, 255), cv2.MARKER_CROSS, 30, 3)
        cv2.putText(frame, status, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"c clear  f flip  m mirror ({'on' if mirror else 'off'})  r servo reverse ({'on' if servo_flip else 'off'})  q quit", (16, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        phone_compass.update(sent_dir, dist if sent_dir is not None else 100, guide_name, guide_note)
        draw_gauge(frame, sent_dir)
        cv2.putText(frame, f"phone: {phone_url}", (16, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        if not band_ok:
            cv2.putText(frame, "band not connected (phone only)", (16, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("compass", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            target = None
            guide_name = None
        if finder is not None and ord("1") <= key <= ord("9") and key - ord("1") < len(names):
            guide_name = names[key - ord("1")]
        if key == ord("r"):
            servo_flip = not servo_flip
        if key == ord("m"):
            mirror = not mirror
        if key == ord("f"):
            flip = not flip
            if facing is not None:
                facing = -facing

    cap.release()
    for _, c in other_caps:
        c.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
