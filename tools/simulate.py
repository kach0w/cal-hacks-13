"""A simulated table to watch the whole pipeline without a camera: your reference cutouts on two
named surfaces, following a 32-second script, replayed through the real pipeline and memory.

    python -m tools.simulate --show                         # scripted boxes, real-time window
    python -m tools.simulate --mode real --set DETECTOR=yoloe --show   # what the model really sees
    python -m tools.simulate                                # headless: screenshots + contact sheet

Modes:
    scripted  boxes come from the script (ground truth), so every memory rule is exercised:
              a move via mid-air, a hand passing over the remote, glasses gone and back, glasses
              into the bag, the bag carried to the counter with the glasses inferred inside
    real      the configured detector on the rendered frames (the drawn bag isn't a real bag, so
              the detector won't find it and containment can't happen)

Writes data/sim/: sim.mp4 (the scene at 10 fps), zones.json, and <mode>/ with the memory
(state.json, events.jsonl, stills/), screens/ (overlay of every detection pass) and
contact_sheet.jpg (all of them on one page). Then asks the memory a few questions.
"""
import argparse
import logging
import math
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from recall import config
from recall.ask import handle
from recall.camera import VideoSource
from recall.enroll import load_items
from recall.live import run
from recall.models import Sighting, Zone
from recall.perception import Detections, make_detector
from recall.query import Query
from recall.store import Store
from recall.zones import save_zones

log = logging.getLogger("recall")
W, H, FPS, LENGTH_S = 1280, 720, 10, 32
ZONES = [Zone("kitchen counter", [0, 230, 600, 720]), Zone("table", [680, 230, 1280, 720])]

# Where each prop is (x, y, width) as a function of time; None = out of sight.
def keys(t):
    return (90, 380, 230) if t < 4 else (520, 60, 200) if t < 5 else (800, 440, 200)   # carried via mid-air


def glasses(t):
    if t < 12:
        return 360, 560, 170          # on the counter
    if t < 16:
        return None                   # picked up: gone
    if t < 19:
        return 880, 610, 150          # back, on the table
    if t < 20:
        return 1080, 360, 130         # over the bag's opening...
    return None                       # ...and into the bag


def bag(t):
    return (1040, 330) if t < 25 else (560, 20) if t < 26 else (330, 330)   # carried to the counter


HAND = (9, 10)                        # a hand passes over the remote for one look
REMOTE, WALLET, BAG_SIZE = (700, 300, 200), (720, 590, 130), (200, 240)


class Scene:
    def __init__(self, reference_dir: Path):
        rng = np.random.default_rng(0)
        bg = np.zeros((H, W, 3), np.uint8)
        bg[:] = (120, 130, 140)                            # wall
        bg[230:, :600] = (205, 200, 190)                   # counter: pale grey laminate
        bg[230:, 600:680] = (70, 80, 95)                   # gap between them
        bg[230:, 680:] = (200, 225, 235)                   # table: cream cloth
        self.bg = np.clip(bg + rng.normal(0, 3, bg.shape), 0, 255).astype(np.uint8)
        self.cutouts = {n: cv2.imread(str(reference_dir / f"{n}.png"), cv2.IMREAD_UNCHANGED)
                        for n in ("keys", "glasses", "remote", "wallet")}
        self.bag = paper_bag(*BAG_SIZE)

    def render(self, t):
        """-> (image, {item: box}, [person boxes]) at time t."""
        img, boxes, persons = self.bg.copy(), {}, []
        bx, by = bag(t)
        boxes["bag"] = paste(img, self.bag, bx, by)
        for name, where in (("remote", REMOTE), ("wallet", WALLET), ("keys", keys(t)), ("glasses", glasses(t))):
            if where:
                x, y, w = where
                boxes[name] = paste(img, resize(self.cutouts[name], w), x, y)
        if HAND[0] <= t < HAND[1]:
            persons.append(draw_hand(img, REMOTE[0] + 100, REMOTE[1] + 30))
            boxes.pop("remote")                            # covered by the hand
        return img, boxes, persons


def resize(img, width):
    return cv2.resize(img, (width, round(img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_AREA)


def paste(bg, img, x, y):
    """Alpha-blend img at (x, y); returns the tight box of what's visible."""
    h, w = img.shape[:2]
    x2, y2 = min(x + w, bg.shape[1]), min(y + h, bg.shape[0])
    src = img[:y2 - y, :x2 - x]
    a = src[..., 3:4].astype(float) / 255
    bg[y:y2, x:x2] = (a * src[..., :3] + (1 - a) * bg[y:y2, x:x2]).astype(np.uint8)
    ys, xs = np.nonzero(src[..., 3] > 32)
    return [float(x + xs.min()), float(y + ys.min()), float(x + xs.max() + 1), float(y + ys.max() + 1)]


def paper_bag(w, h):
    """A brown paper bag with its opening toward the camera (drawn: real detectors won't know it)."""
    img = np.zeros((h, w, 4), np.uint8)
    body = np.array((70, 120, 170), float)
    for x in range(w):
        img[:, x, :3] = np.clip(body * (0.75 + 0.25 * math.sin(math.pi * x / w)), 0, 255)
    img[..., 3] = 255
    rim = int(h * 0.12)
    img[:rim, :, :3] = np.clip(body * 1.15, 0, 255)
    cv2.ellipse(img, (w // 2, rim), (int(w * 0.42), int(rim * 0.8)), 0, 0, 360, (25, 40, 55, 255), -1)
    for x in (int(w * 0.18), int(w * 0.82)):
        cv2.line(img, (x, rim), (x, h - 1), (45, 85, 125, 255), 2)
    return img


def draw_hand(img, cx, cy):
    skin = (140, 170, 215)
    cv2.rectangle(img, (cx - 45, 0), (cx + 45, cy - 40), skin, -1)        # forearm from above
    cv2.ellipse(img, (cx, cy), (85, 60), 0, 0, 360, skin, -1)             # hand
    return [float(cx - 85), 0.0, float(cx + 85), float(cy + 60)]


def write_video(scene, path):
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(LENGTH_S * FPS):
        out.write(scene.render(i / FPS)[0])
    out.release()


class ScriptedDetector:
    """Ground-truth boxes from the script: exercises the memory without any model."""

    def __init__(self, scene, items, t0):
        self.scene, self.items, self.t0 = scene, items, t0

    def detect(self, image, ts, raw=False):
        _, boxes, persons = self.scene.render(ts - self.t0 + 1e-6)
        return Detections([Sighting(oid, b, 0.9, ts) for oid, b in boxes.items() if oid in self.items], persons)

    def warmup(self, h, w):
        pass


def contact_sheet(screens: Path, out: Path, cols=3, width=480):
    tiles = [cv2.resize(cv2.imread(str(p)), (width, width * H // W)) for p in sorted(screens.glob("*.jpg"))]
    if not tiles:
        return
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % cols)
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(str(out), np.vstack(rows))


def main():
    ap = argparse.ArgumentParser(description="Watch the pipeline on a simulated table.")
    ap.add_argument("--mode", choices=["scripted", "real"], default="scripted")
    ap.add_argument("--show", action="store_true", help="OpenCV window, real time (q quits)")
    config.add_args(ap)
    a = ap.parse_args()
    cfg = config.from_args(a)
    sim = cfg.data_dir / "sim"
    cfg = replace(cfg, data_dir=sim, replay_speed=cfg.replay_speed or (1.0 if a.show else 0.0))
    items = load_items(cfg.items_path)

    sim.mkdir(parents=True, exist_ok=True)
    scene = Scene(config.ROOT / "data" / "reference")
    video = sim / "sim.mp4"
    write_video(scene, video)
    save_zones(ZONES, cfg.zones_path)
    source = VideoSource(str(video), cfg.loop_s)
    if a.mode == "scripted":
        detector = ScriptedDetector(scene, items, source.t0)
    else:
        detector = make_detector(items, cfg)
        detector.warmup(H, W)
    out = sim / a.mode
    store = Store(out)
    store.clear()
    for old in (out / "screens").glob("*.jpg"):
        old.unlink()
    log.info("simulating %s mode: %s -> %s", a.mode, video, out)
    run(cfg, items, ZONES, source, detector, store, a.show, out / "screens")
    contact_sheet(out / "screens", out / "contact_sheet.jpg")

    q = Query(cfg, items, store)
    print()
    for question in ("where are my glasses", "where are my keys", "where is the bag", "what changed today"):
        print(f"> {question}\n{handle(question, q).text}\n")
    print(f"screenshots: {out / 'screens'}\ncontact sheet: {out / 'contact_sheet.jpg'}")


if __name__ == "__main__":
    main()
