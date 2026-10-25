"""Task 0 (CAMERA.md §8): can the detector see every prop at the real distances?

    python -m tools.range_test capture [--src 0] [--stations near=1.0,mid=2.0,far=3.0] [--burst 8]
    python -m tools.range_test eval [--detectors yolo-world,yoloe] [--imgsz 640,1280]

capture: camera at its demo position. For each station put ALL the props + the bag at that
distance (spread out, not touching), press SPACE and keep still while a burst is grabbed;
n skips the station, q quits. Each completed burst replaces the station's frames in
data/range_test/<station>/ (stations.json keeps their distances), then eval runs.
eval: runs each detector x imgsz on the saved frames (so prompts and models can be compared later
without the camera), prints the table, writes annotated sheets to data/range_test/report/, and
recommends DETECTOR + DETECT_IMGSZ: fewest misses within the 150 ms budget, then the smaller size.
"""
import argparse
import statistics
import time
from dataclasses import replace

import cv2

from recall import config
from recall.enroll import load_items
from recall.geometry import area, intersection
from recall.overlay import AMBER, GREEN, QUIT, banner, close_windows, key, label_box
from recall.perception import make_detector
from recall.zones import check_size
from tools import read_json, write_json

BUDGET_MS = 150      # CAMERA.md §8 task 1
HIT_RATE = 0.9       # an item "works" at a station if seen in >= 90% of the burst
BURST_GAP_MS = 250   # between the frames of a burst
NEW_FRAME_S = 2.0    # no new frame this long during a burst: the camera stalled (or the video ended)


def capture(cfg, stations, burst, out):
    from recall.camera import open_source
    src = open_source(cfg)
    try:
        for name, dist in stations:
            msg = f"{name} ({dist} m): all props here, SPACE grab, n skip, q quit"
            print(msg)
            while True:
                f = src.read()
                if f is not None:
                    check_size(f.image, cfg)
                    cv2.imshow("range test", banner(f.image.copy(), msg))
                k = key(30)
                if k in QUIT:
                    return
                if k == "n":
                    break
                if k == " ":
                    grab(src, out / name, burst)
                    save_station(out, name, dist)
                    break
    finally:
        src.close()
        close_windows()


def grab(src, folder, burst):
    """`burst` different camera frames, BURST_GAP_MS apart, into folder. An earlier burst there is
    replaced only once this one is complete."""
    imgs, last = [], None
    for i in range(burst):
        deadline = time.time() + NEW_FRAME_S
        while (f := src.read()) is None or f.seq == last:
            if time.time() > deadline:
                raise SystemExit(f"no new frame for {NEW_FRAME_S:g} s: burst for {folder} abandoned")
            time.sleep(0.01)
        last = f.seq
        imgs.append(f.image)
        cv2.imshow("range test", banner(f.image.copy(), f"grabbing {i + 1}/{burst}: keep still"))
        cv2.waitKey(BURST_GAP_MS)  # also keeps the window painting
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.jpg"):
        old.unlink()
    for i, img in enumerate(imgs):
        cv2.imwrite(str(folder / f"{i:02d}.jpg"), img)
    print(f"  saved {burst} frames to {folder}")


def save_station(out, name, dist):
    """stations.json: each captured station's distance, nearest first (the order eval reports in)."""
    path = out / "stations.json"
    meta = {**read_json(path, {}), name: dist}
    write_json(path, dict(sorted(meta.items(), key=lambda kv: (kv[1] is None, kv[1] or 0))))


def run(det, items, frames):
    """-> ({item: [best raw score per frame or 0]}, {item: [box]}, ms per frame, last frame's sightings)"""
    det.detect(frames[0], 0, raw=True)  # warm up this imgsz
    scores, boxes, t0, last = {i: [] for i in items}, {i: [] for i in items}, time.time(), None
    for img in frames:
        last = {s.object: s for s in det.detect(img, 0, raw=True).sightings}
        for i in items:
            scores[i].append(last[i].score if i in last else 0.0)
            if i in last:
                boxes[i].append(last[i].bbox)
    return scores, boxes, (time.time() - t0) / len(frames) * 1000, last


def overlaps(last):
    """Pairs of different labels on (nearly) the same box: a sign two props are confused."""
    ids = sorted(last)
    return [(a, b) for n, a in enumerate(ids) for b in ids[n + 1:]
            if intersection(last[a].bbox, last[b].bbox) > 0.5 * min(area(last[a].bbox), area(last[b].bbox))]


def evaluate(cfg, out, detectors, sizes):
    items = load_items(cfg.items_path)
    order = list(read_json(out / "stations.json", {}))
    dirs = [d for d in out.iterdir() if d.is_dir() and d.name != "report"] if out.is_dir() else []
    stations = sorted(dirs, key=lambda d: (order.index(d.name) if d.name in order else len(order), d.name))
    frames = {}
    for s in stations:
        imgs = [img for p in sorted(s.glob("*.jpg")) if (img := cv2.imread(str(p))) is not None]
        if imgs:
            frames[s.name] = imgs
        else:
            print(f"skipping {s.name}: no images")
    if not frames:
        raise SystemExit(f"no captured stations in {out}; run: python -m tools.range_test capture")
    report = out / "report"
    report.mkdir(exist_ok=True)
    thr = lambda i: items[i].min_score if items[i].min_score is not None else cfg.default_min_score
    results = []
    for kind in detectors:
        det = make_detector(items, replace(cfg, detector=kind, detect_model=""))
        for size in sizes:
            det.cfg = replace(det.cfg, detect_imgsz=size)
            row = {"detector": kind, "imgsz": size, "ms": [], "misses": [], "confused": []}
            print(f"\n{kind} @ {size}")
            for st, imgs in frames.items():
                scores, boxes, ms, last = run(det, items, imgs)
                row["ms"].append(ms)
                cells = []
                for i in items:
                    rate = sum(s >= thr(i) for s in scores[i]) / len(imgs)
                    w = statistics.median(b[2] - b[0] for b in boxes[i]) if boxes[i] else 0
                    cells.append(f"{i} {rate:4.0%} ({statistics.median(scores[i]):.2f}, {w:.0f}px)")
                    if rate < HIT_RATE:
                        row["misses"].append(f"{i}@{st}")
                row["confused"] += [f"{a}/{b}@{st}" for a, b in overlaps(last)]
                print(f"  {st:<6} {ms:5.0f} ms  " + "  ".join(cells))
                sheet = imgs[-1].copy()
                for s in last.values():
                    label_box(sheet, s.bbox, f"{items[s.object].name} {s.score:.2f}",
                              GREEN if s.score >= thr(s.object) else AMBER)
                cv2.imwrite(str(report / f"{st}_{kind}_{size}.jpg"), banner(sheet, f"{st} {kind} {size}"))
            results.append(row)
    print(f"\nhit = seen at >= the item's threshold (min_score, else {cfg.default_min_score}); "
          f"(median raw score, median box width); sheets in {report}")
    best = recommend(results)
    for r in results:
        flag = "  <- recommended" if r is best else ""
        print(f"  {r['detector']:<10} {r['imgsz']:<5} {max(r['ms']):5.0f} ms  misses: {', '.join(r['misses']) or 'none'}"
              f"{'  CONFUSED: ' + ', '.join(r['confused']) if r['confused'] else ''}{flag}")
    if best:
        print(f"\nrecommend: --set DETECTOR={best['detector']} --set DETECT_IMGSZ={best['imgsz']}")
    return results, best


def recommend(results):
    """Within budget: fewest misses + confusions, then the smaller imgsz, then the faster one."""
    ok = [r for r in results if max(r["ms"]) < BUDGET_MS]
    return min(ok, key=lambda r: (len(r["misses"]) + len(r["confused"]), r["imgsz"], max(r["ms"])), default=None)


def main():
    ap = argparse.ArgumentParser(description="Detector range test at the real distances (task 0).")
    ap.add_argument("cmd", choices=["capture", "eval"])
    ap.add_argument("--src", help="camera index")
    ap.add_argument("--stations", default="near=1.0,mid=2.0,far=3.0", help="name=meters,...")
    ap.add_argument("--burst", type=int, default=8)
    ap.add_argument("--detectors", default="yolo-world,yoloe")
    ap.add_argument("--imgsz", default="640,1280")
    config.add_args(ap)
    a = ap.parse_args()
    cfg = config.from_args(a, src=a.src)
    out = cfg.data_dir / "range_test"
    if a.cmd == "capture":
        stations = [(s.split("=")[0], float(s.split("=")[1]) if "=" in s else None) for s in a.stations.split(",")]
        try:
            capture(cfg, stations, a.burst, out)
        except KeyboardInterrupt:
            raise SystemExit("\nstopped; captured stations are kept (python -m tools.range_test eval)")
    evaluate(cfg, out, a.detectors.split(","), [int(s) for s in a.imgsz.split(",")])


if __name__ == "__main__":
    main()
