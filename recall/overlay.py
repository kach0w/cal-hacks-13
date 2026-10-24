"""All drawing: the live window, evidence stills, the /stream feed, the enroll and zone tools."""
import subprocess
import sys

import cv2

from .models import clock_s

GREEN, AMBER, RED, GREY, BLUE = (0, 220, 0), (0, 200, 255), (0, 0, 255), (160, 160, 160), (255, 160, 0)
STATUS_COLOR = {"observed": GREEN, "inferred": AMBER, "unknown": RED}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def label_box(img, bbox, text, color, thick=2, scale=0.6):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
    if text:
        cv2.putText(img, text, (x1, max(18, y1 - 6)), FONT, scale, color, max(1, thick))
    return img


def draw_zones(img, zones, color=BLUE, numbered=False):
    for i, z in enumerate(zones):
        x1, y1, x2, y2 = map(int, z.box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1 if not numbered else 2)
        name = f"[{i}] {z.name}" if numbered else z.name
        cv2.putText(img, name, (x1 + 4, y1 + 20), FONT, 0.6, color, 1 if not numbered else 2)
    return img


def banner(img, text, color=(255, 255, 255)):
    cv2.putText(img, text, (12, 32), FONT, 0.9, (0, 0, 0), 4)
    cv2.putText(img, text, (12, 32), FONT, 0.9, color, 2)
    return img


def still(frame, bbox, text):
    """An evidence still: the frame with the object's box and what happened."""
    out = frame.copy()
    return label_box(out, bbox, text, GREEN, 3, 0.8) if bbox else out


def status_panel(img, state, items):
    """What the memory believes, one line per item, bottom-left."""
    lines = []
    for oid, it in items.items():
        o = state.get(oid)
        if o is None or o.last_observed is None:
            lines.append((f"{it.name}: never seen", GREY))
            continue
        where = o.zone or "no named area"
        if o.status == "inferred" and o.container in state:
            where = f"in {state[o.container].name}, {where}"
        lines.append((f"{it.name}: {o.status} - {where}", STATUS_COLOR.get(o.status, GREY)))
    y = img.shape[0] - 12 - 24 * (len(lines) - 1)
    for text, color in lines:
        cv2.putText(img, text, (12, y), FONT, 0.6, (0, 0, 0), 4)
        cv2.putText(img, text, (12, y), FONT, 0.6, color, 2)
        y += 24
    return img


def live_view(image, result, zones, state, items, frozen_at=None):
    """The --show window, snapshots and /stream: zones, latest boxes coloured by memory status,
    and the memory's belief per item."""
    out = draw_zones(image.copy(), zones)
    if result and result.detections:
        for b in result.detections.persons:
            label_box(out, b, "person", GREY, 1)
        for s in result.detections.sightings:
            o = state.get(s.object_id)
            color = STATUS_COLOR.get(o.status if o else "", GREY)
            label_box(out, s.bbox, f"{items[s.object_id].name} {s.score:.2f}", color)
    status_panel(out, state, items)
    if frozen_at is not None:
        banner(out, f"CAMERA STALLED - memory frozen as of {clock_s(frozen_at)}", RED)
    elif result:
        banner(out, f"{clock_s(result.frame.ts)}  {result.reason}")
    return out


def jpeg(img, quality=80) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


# --- windows of the interactive tools. macOS repaints (and removes) them only inside waitKey ------
QUIT = ("q", "\x1b")   # q or ESC


def key(ms=1) -> str:
    """cv2.waitKey as a lowercase character: '' = no key, ' ' = SPACE, '\\x1b' = ESC."""
    k = cv2.waitKey(ms)
    return "" if k < 0 else chr(k & 0xFF).lower()


def close_windows():
    cv2.destroyAllWindows()
    for _ in range(4):
        cv2.waitKey(1)


def preview(path):
    """Open a saved picture in Preview on macOS without waiting: unlike an OpenCV window it stays
    responsive while the terminal waits for typing."""
    if sys.platform == "darwin":
        subprocess.Popen(["open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
