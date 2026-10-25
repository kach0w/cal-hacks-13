"""All drawing: the live window, evidence photos, the /stream feed, the enroll, zone and surface tools."""
import subprocess
import sys

import cv2
import numpy as np

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
        where = o.surface or "no named area"
        if o.status == "inferred" and o.container in state:
            where = f"in {state[o.container].name}, {where}"
        lines.append((f"{it.name}: {o.status} - {where}", STATUS_COLOR.get(o.status, GREY)))
    y = img.shape[0] - 12 - 24 * (len(lines) - 1)
    for text, color in lines:
        cv2.putText(img, text, (12, y), FONT, 0.6, (0, 0, 0), 4)
        cv2.putText(img, text, (12, y), FONT, 0.6, color, 2)
        y += 24
    return img


EDGE_STEP_M = 0.05   # a surface outline is drawn through points this far apart, each checked by room.in_lens


def surface_outline(s, room, size):
    """Surface s's top (at s.top) as the camera sees it: (runs of pixels, each an open polyline, and
    where its name goes or None). Only points room.in_lens keeps are drawn; the name goes at the
    middle of the outline's part inside the image (size = (width, height)), so it lies within it."""
    from .room import project
    (x0, _, z0), (x1, _, z1) = s.box_min, s.box_max
    corners = np.array([[x0, s.top, z0], [x1, s.top, z0], [x1, s.top, z1], [x0, s.top, z1], [x0, s.top, z0]])
    pts = np.concatenate([np.linspace(a, b, max(2, int(np.linalg.norm(b - a) / EDGE_STEP_M) + 1))[:-1]
                          for a, b in zip(corners, corners[1:])])
    ok = room.in_lens(pts)
    uv = np.full((len(pts), 2), np.nan)
    if ok.any():
        uv[ok] = project(pts[ok], room.K, room.dist, room.R, room.t)
    idx = np.arange(len(pts))
    if ok.all():
        runs = [np.r_[idx, 0]]                                   # the whole outline, closed
    else:
        idx = np.roll(idx, -int(np.argmin(ok)))                  # start where it's hidden
        runs = [r[ok[r]] for r in np.split(idx, np.flatnonzero(~ok[idx]))]
    w, h = size
    shown = ok & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    anchor = tuple(uv[shown].mean(axis=0)) if shown.any() else None
    return [uv[r].round().astype(np.int32) for r in runs if len(r) > 1], anchor


def draw_surfaces(img, surfaces, room, color=BLUE):
    """Each scanned surface's top outline where the registered camera sees it, named in its middle."""
    for s in surfaces:
        runs, anchor = surface_outline(s, room, (img.shape[1], img.shape[0]))
        if runs:
            cv2.polylines(img, runs, False, color, 2)
        if anchor is not None:
            (tw, th), _ = cv2.getTextSize(s.name, FONT, 0.6, 2)
            cv2.putText(img, s.name, (int(anchor[0] - tw / 2), int(anchor[1] + th / 2)), FONT, 0.6, color, 2)
    return img


def live_view(image, result, places, state, items, frozen_at=None, room=None):
    """The --show window, snapshots and /stream: the places, latest boxes coloured by memory
    status, and the memory's belief per item."""
    out = image.copy()
    if places.in_3d and room is not None:
        draw_surfaces(out, places.surfaces, room)
    else:
        draw_zones(out, places.zones)
    if result and result.detections:
        for b in result.detections.persons:
            label_box(out, b, "person", GREY, 1)
        for s in result.detections.sightings:
            o = state.get(s.object)
            color = STATUS_COLOR.get(o.status if o else "", GREY)
            label_box(out, s.bbox, f"{items[s.object].name} {s.score:.2f}", color)
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
