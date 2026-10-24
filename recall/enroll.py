"""Enrollment (CAMERA.md Part A): the dictionary of things we track + per-item calibration.

    python -m recall.enroll list
    python -m recall.enroll add bag --name "bag" --prompt "brown paper bag" --container
    python -m recall.enroll remove remote
    python -m recall.enroll calibrate [ids...] [--src 0]

calibrate: put one item in view; the window shows the detector's best box for it. SPACE = this
box is right (saves a crop + its score), 3 samples per item (move/rotate it between them),
n = next item, q or ESC = quit; an item keeps the samples accepted before n, q, Ctrl-C or the end
of a video (none: unchanged). Sets min_score = 0.6 x the lowest accepted score. Re-run in demo
lighting at the demo camera position (DESIGN §8). The crops in data/enroll/<id>/ are the re-id
references; an item's crops are replaced as a set when it ends, never one by one.
"""
import argparse
import logging
import re
import time
from dataclasses import replace

from . import config
from .models import Item
from .store import read_json, write_json_atomic

log = logging.getLogger(__name__)
SAMPLES = 3
MIN_GAP_S = 0.5   # between accepted samples, so they aren't three copies of one frame


def load_items(path) -> dict[str, Item]:
    return {oid: Item.from_dict(oid, d) for oid, d in read_json(path, {}).items()}


def save_items(items: dict[str, Item], path):
    write_json_atomic(path, {oid: it.to_dict() for oid, it in items.items()})


def cmd_list(items):
    for oid, it in items.items():
        flags = " ".join(f for f, on in (("plural", it.plural), ("container", it.container)) if on)
        print(f"{oid:<10} name={it.name!r:<16} prompt={it.prompt!r:<26} min_score={it.min_score}  {flags}")


def calibrate(items, ids, cfg):
    from .camera import open_source
    from .overlay import close_windows
    from .perception import make_detector
    det, src = make_detector(items, cfg), open_source(cfg)
    try:
        for oid in ids or list(items):
            if oid not in items:
                print(f"skip unknown item {oid}")
                continue
            it = items[oid]
            print(f"\n== {it.name} (prompt {it.prompt!r}): put it in view. SPACE accept, n next item, q quit")
            samples = []
            try:
                more = _samples(it, det, src, cfg, samples)
            finally:  # Ctrl-C or a crash keeps the accepted samples too, as q does
                _keep(items, it, samples, cfg)
            if not more:
                break
    finally:
        src.close()
        close_windows()


def _samples(it, det, src, cfg, samples) -> bool:
    """Append (score, crop) for each accepted box to samples; -> whether to go on to the next item."""
    import cv2
    from .overlay import QUIT, banner, key, label_box
    from .zones import check_size
    last_accept = 0.0
    while len(samples) < SAMPLES:
        frame = src.read()
        if frame is None:
            if not src.live:
                print("  the video ended")
                return False
            time.sleep(0.05)
            continue
        check_size(frame.image, cfg)
        s = next((x for x in det.detect(frame.image, frame.ts, raw=True).sightings
                  if x.object_id == it.id), None)
        crop = _crop(frame.image, s.bbox) if s else None
        view = frame.image.copy()
        if s:
            label_box(view, s.bbox, f"{it.name} {s.score:.2f}", (0, 255, 0))
        cv2.imshow("enroll", banner(view, f"{it.name}: {len(samples)}/{SAMPLES}" + ("" if s else "  not seen")))
        k = key(1)
        if k == "n":
            return True
        if k in QUIT:
            return False
        if k == " " and crop is not None and time.time() - last_accept > MIN_GAP_S:
            samples.append((s.score, crop.copy()))
            last_accept = time.time()
            print(f"  sample {len(samples)}: score {s.score:.3f}")
    return True


def _keep(items, it, samples, cfg):
    """The accepted samples become the item's crops in data/enroll/<id>/ (an older calibration's are
    deleted: they must not linger as re-id references) and its min_score. None: nothing changes."""
    import cv2
    if not samples:
        print(f"  {it.name}: no samples, min_score stays {it.min_score}")
        return
    out = cfg.enroll_dir / it.id
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.jpg"):
        old.unlink()
    for i, (_, crop) in enumerate(samples):
        cv2.imwrite(str(out / f"{i}.jpg"), crop)
    it.min_score = round(max(0.03, 0.6 * min(score for score, _ in samples)), 3)
    it.extra.update(enrolled_at=time.time(), ref_crops=[f"enroll/{it.id}/{i}.jpg" for i in range(len(samples))])
    save_items(items, cfg.items_path)
    print(f"  {it.name}: min_score = {it.min_score}")


def _crop(image, bbox):
    """The box's pixels, clipped to the image; None if nothing is left."""
    h, w = image.shape[:2]
    x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
    x2, y2 = min(w, int(bbox[2])), min(h, int(bbox[3]))
    return image[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else None


def main():
    ap = argparse.ArgumentParser(description="Manage the tracked-item dictionary (items.json).")
    ap.add_argument("cmd", choices=["list", "add", "remove", "calibrate"])
    ap.add_argument("ids", nargs="*", help="add/remove: one item id; calibrate: some or all")
    ap.add_argument("--name")
    ap.add_argument("--prompt", help="what the detector is told to look for")
    ap.add_argument("--alias", action="append", help="another word a question can use (repeatable)")
    ap.add_argument("--plural", action="store_true", default=None)
    ap.add_argument("--container", action="store_true", default=None, help="things can go inside it")
    ap.add_argument("--src", help="calibrate: camera index or video file")
    config.add_args(ap)
    a = ap.parse_args()
    cfg = config.from_args(a, src=a.src)
    if a.cmd in ("add", "remove") and len(a.ids) != 1:
        ap.error(f"{a.cmd} takes exactly one item id")
    if a.cmd == "add" and not re.fullmatch(r"[a-z0-9_]+", a.ids[0]):
        ap.error(f"item ids are lowercase words like car_keys, not {a.ids[0]!r}")

    items = load_items(cfg.items_path)
    if a.cmd == "list":
        cmd_list(items)
    elif a.cmd == "add":
        oid = a.ids[0]
        given = {k: v for k, v in dict(name=a.name, prompt=a.prompt, aliases=a.alias,
                                       plural=a.plural, container=a.container).items() if v is not None}
        if oid in items:  # update only what was given; calibration and reference survive
            items[oid] = replace(items[oid], **given)
        else:
            name = given.pop("name", oid)
            items[oid] = Item(oid, name, given.pop("prompt", name), **given)
        save_items(items, cfg.items_path)
        print(f"saved {oid}. Now: python -m recall.enroll calibrate {oid}")
    elif a.cmd == "remove":
        if items.pop(a.ids[0], None) is None:
            raise SystemExit(f"no item {a.ids[0]!r} in {cfg.items_path}")
        save_items(items, cfg.items_path)
        print(f"removed {a.ids[0]}")
    else:
        try:
            calibrate(items, a.ids, cfg)
        except KeyboardInterrupt:
            raise SystemExit("\nstopped: samples accepted before Ctrl-C are saved")


if __name__ == "__main__":
    main()
