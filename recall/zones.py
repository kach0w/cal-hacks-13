"""Zones (DESIGN §4.1): named regions of the camera image ("kitchen counter").

Onboarding: detect furniture once, confirm/rename/delete, add manual boxes for anything the
detector misses, freeze to data/zones.json.

    python -m recall.zones onboard [--src 0]   # detect + confirm + optional manual boxes
    python -m recall.zones manual  [--src 0]   # just draw boxes (the fallback that always works)
    python -m recall.zones show    [--src 0]   # overlay current zones on the live feed (q quits)
    python -m recall.zones list

Names are typed in the terminal while the numbered boxes are open in Preview
(data/zones_candidates.jpg, data/zones_drawn.jpg); every zone needs its own name.
"""
import argparse
import functools
import logging
import time

from . import config
from .geometry import area, base_point, contains
from .models import Zone
from .store import read_json, write_json_atomic

log = logging.getLogger(__name__)

FURNITURE_PROMPTS = ["kitchen counter", "table", "desk", "shelf", "couch", "chair",
                     "bed", "sink", "cabinet", "nightstand"]
SETTLE_S = 1.0     # a live camera's first frames are dark while auto-exposure settles
DARK_WAIT_S = 5.0  # then this long at most for a frame that isn't black
MIN_BOX_PX = 10    # a drawn box smaller than this is a slip of the mouse


def load_zones(path) -> list[Zone]:
    return [Zone.from_dict(z) for z in read_json(path, [])]


def save_zones(zones: list[Zone], path):
    write_json_atomic(path, [z.to_dict() for z in zones])


def zone_at(zones: list[Zone], bbox, margin=10.0) -> str | None:
    """The zone holding the object's base point; the smallest wins (a tray on a counter)."""
    pt = base_point(bbox)
    inside = [z for z in zones if contains(z.box, pt, margin)]
    return min(inside, key=lambda z: area(z.box)).name if inside else None


def zone_phrase(zones: list[Zone], name: str | None) -> str:
    """'on the kitchen counter' / 'in the sink'; None -> 'outside the named areas'."""
    if name is None:
        return "outside the named areas"
    prep = next((z.prep for z in zones if z.name == name), "on")
    return f"{prep} the {name}"


def check_size(image, cfg):
    """Zones and calibration are pixels of one camera mode: warn (once) when frames come in another."""
    h, w = image.shape[:2]
    if (w, h) != (cfg.capture_w, cfg.capture_h):
        _warn_size(w, h, cfg.capture_w, cfg.capture_h)


@functools.cache
def _warn_size(w, h, want_w, want_h):
    log.warning("frames are %dx%d, not CAPTURE_W x CAPTURE_H = %dx%d: zones drawn at one size are in the wrong "
                "place at the other. Use one camera mode everywhere (set RECALL_CAPTURE_W/H to what the camera "
                "gives)", w, h, want_w, want_h)


# --- interactive tools (need a camera and a window) ----------------------------
def _grab(cfg):
    """One still. A live camera gets SETTLE_S, then up to DARK_WAIT_S for a frame that isn't black."""
    from .camera import open_source
    src = open_source(cfg)
    try:
        if src.live:
            time.sleep(SETTLE_S)  # let auto-exposure settle (lock it for the demo)
            deadline = time.time() + DARK_WAIT_S
            while ((frame := src.read()) is None or frame.image.mean() < 10) and time.time() < deadline:
                time.sleep(0.1)
        else:
            frame = src.read()
    finally:
        src.close()
    if frame is None:
        raise SystemExit("could not read a frame from the camera")
    check_size(frame.image, cfg)
    return frame.image


def _picture(image, zones, path):
    """The numbered boxes, saved and opened in Preview: an OpenCV window would freeze while the
    terminal asks for names (macOS repaints it only inside waitKey)."""
    import cv2
    from .overlay import draw_zones, preview
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), draw_zones(image.copy(), zones, numbered=True))
    preview(path)
    return path


def _ask_name(prompt, zones, default=None) -> str | None:
    """A zone name not used yet (zone_phrase finds zones by name). ENTER = default; None = drop it."""
    taken = {z.name.lower() for z in zones}
    while True:
        ans = " ".join(input(prompt).split())
        if ans.lower() == "d" or not (ans or default):
            return None
        name = ans or default
        if name.lower() not in taken:
            return name
        print(f"  there is already a zone called '{name}': type another name, or d to drop this one")


def _ask_prep(name):
    ans = input(f"  preposition for '{name}' [on] (on/in/under/next to): ")
    return " ".join(ans.lower().split()) or "on"


def _boxes(rois, w, h) -> list:
    """selectROIs' (x, y, w, h) rows -> boxes clipped to the image; slips of the mouse dropped."""
    boxes = []
    for x, y, bw, bh in rois:
        b = [max(0, int(x)), max(0, int(y)), min(w, int(x + bw)), min(h, int(y + bh))]
        if b[2] - b[0] >= MIN_BOX_PX and b[3] - b[1] >= MIN_BOX_PX:
            boxes.append([float(v) for v in b])
    if len(boxes) < len(rois):
        print(f"  dropped {len(rois) - len(boxes)} box(es) under {MIN_BOX_PX} px")
    return boxes


def _manual(image, zones, cfg):
    """Draw new boxes over the current zones, then name each one in the terminal."""
    import cv2
    from .overlay import close_windows, draw_zones
    print("Draw boxes in the window: drag, then SPACE or ENTER to keep each box (c cancels it).\n"
          "ESC when done; a box not kept with SPACE/ENTER first is dropped.")
    try:
        rois = cv2.selectROIs("draw zones", draw_zones(image.copy(), zones, numbered=True),
                              showCrosshair=False)
    finally:
        close_windows()
    boxes = _boxes(rois, image.shape[1], image.shape[0])
    if not boxes:
        print("no boxes drawn")
        return zones
    n = len(zones)
    path = _picture(image, zones + [Zone("?", b) for b in boxes], cfg.data_dir / "zones_drawn.jpg")
    print(f"Name the new boxes [{n}..{n + len(boxes) - 1}] (numbered in {path}) here:")
    for i, box in enumerate(boxes, n):
        name = _ask_name(f"[{i}] name (ENTER drops it): ", zones)
        if name:
            zones.append(Zone(name, box, _ask_prep(name), "manual"))
    return zones


def _save(zones, cfg):
    if not zones:
        print(f"no zones: {cfg.zones_path} left as it was")
        return
    save_zones(zones, cfg.zones_path)
    print(f"saved {len(zones)} zones -> {cfg.zones_path}")


def onboard(cfg):
    from .perception import detect_prompts
    image = _grab(cfg)
    h, w = image.shape[:2]
    dets = detect_prompts(image, FURNITURE_PROMPTS, cfg, conf=0.08)
    cands = [Zone(d["label"], d["bbox"], "on", "detected") for d in dets
             if area(d["bbox"]) > 0.02 * w * h]  # keep the big static stuff
    zones = []
    if cands:
        path = _picture(image, cands, cfg.data_dir / "zones_candidates.jpg")
        print(f"{len(cands)} candidate zones, numbered in {path}.\n"
              "For each, type here: ENTER keep, a new name to rename, 'd' delete.")
        for i, z in enumerate(cands):
            name = _ask_name(f"[{i}] {z.name}: ", zones, default=z.name)
            if name:
                zones.append(Zone(name, z.box, _ask_prep(name), "detected"))
    else:
        print("No furniture detected: draw the zones by hand.")
    if not cands or input("Draw manual zones too? [y/N]: ").strip().lower().startswith("y"):
        zones = _manual(image, zones, cfg)
    _save(zones, cfg)


def show(cfg):
    import cv2
    from .camera import open_source
    from .overlay import QUIT, close_windows, draw_zones, key
    zones, src = load_zones(cfg.zones_path), open_source(cfg)
    print("q or ESC quits")
    try:
        while True:
            frame = src.read()
            if frame is not None:
                check_size(frame.image, cfg)
                cv2.imshow("zones", draw_zones(frame.image.copy(), zones, numbered=True))
            if key(30) in QUIT:
                break
    except KeyboardInterrupt:
        pass
    finally:
        src.close()
        close_windows()


def main():
    ap = argparse.ArgumentParser(description="Name the regions of the camera image.")
    ap.add_argument("cmd", choices=["onboard", "manual", "show", "list"])
    ap.add_argument("--src", help="camera index or video file")
    config.add_args(ap)
    a = ap.parse_args()
    cfg = config.from_args(a, src=a.src)
    if a.cmd == "list":
        zones = load_zones(cfg.zones_path)
        for z in zones:
            print(f"{z.name:<20} {z.prep:<8} {[int(v) for v in z.box]}  ({z.source})")
        if not zones:
            print(f"no zones in {cfg.zones_path} yet: python -m recall.zones onboard")
    elif a.cmd == "show":
        show(cfg)
    else:
        try:
            if a.cmd == "onboard":
                onboard(cfg)
            else:
                _save(_manual(_grab(cfg), load_zones(cfg.zones_path), cfg), cfg)
        except (KeyboardInterrupt, EOFError):
            raise SystemExit(f"\nstopped: {cfg.zones_path} not changed")


if __name__ == "__main__":
    main()
