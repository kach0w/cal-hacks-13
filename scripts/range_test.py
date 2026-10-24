#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#   "ultralytics>=8.3",
#   "clip @ git+https://github.com/ultralytics/CLIP.git",
# ]
# ///
"""How far away can the detector still see an object?

Hold the object up, walk back, and watch the hit rate. No calibration, no depth.

    uv run scripts/range_test.py --camera "C920" --objects keys,wallet,glasses

Keys while running:
    i      toggle inference size 640 <-> 1280 (the far-object question)
    1-9    log the last second at that many metres to range_log.csv
    q/esc  quit
"""

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO, YOLOWorld

WINDOW = 30  # frames behind the hit-rate readout, ~1s
SHOW_BELOW = 0.05  # draw near-misses down to here, in orange
DISPLAY_WIDTH = 1280
LOG = Path("range_log.csv")


def list_cameras():
    """Camera names in OpenCV's index order.

    OpenCV's AVFoundation backend sorts devices by uniqueID before indexing them,
    so sorting system_profiler's list the same way gives matching indices.
    """
    out = subprocess.run(
        ["system_profiler", "-json", "SPCameraDataType"], capture_output=True, text=True
    ).stdout
    cams = json.loads(out or "{}").get("SPCameraDataType", [])
    cams.sort(key=lambda c: c.get("spcamera_unique-id", ""))
    return [c["_name"] for c in cams]


def open_camera(camera, width, height):
    names = list_cameras()
    for i, name in enumerate(names):
        print(f"  [{i}] {name}")

    if camera.isdigit():
        index = int(camera)
    else:
        matches = [i for i, n in enumerate(names) if camera.lower() in n.lower()]
        if len(matches) != 1:
            sys.exit(f"'{camera}' matched {len(matches)} cameras; pass a name from the list or its index")
        index = matches[0]

    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        sys.exit(f"couldn't open camera {index} (check the terminal's camera permission)")
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"using [{index}] {names[index] if index < len(names) else '?'} at {w}x{h}")
    return cap


def load_model(name, objects):
    """Returns (model, class ids to keep). An open-vocabulary model learns the names."""
    if "world" in name:
        model = YOLOWorld(name)
        model.set_classes(objects)
        return model, None

    model = YOLO(name)
    ids = {label: i for i, label in model.names.items()}
    missing = [o for o in objects if o not in ids]
    if missing:
        sys.exit(f"{missing} aren't classes in {name}; use a -world model. Known: {sorted(ids)}")
    return model, [ids[o] for o in objects]


def detect(model, frame, imgsz, classes, device):
    """[(label, conf, (x1, y1, x2, y2))] in frame pixels, plus inference ms."""
    t = time.perf_counter()
    r = model.predict(
        frame, imgsz=imgsz, conf=SHOW_BELOW, classes=classes, device=device, verbose=False
    )[0]
    ms = (time.perf_counter() - t) * 1000
    dets = [
        (r.names[int(c)], conf, tuple(int(v) for v in box))
        for box, c, conf in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())
    ]
    return dets, ms


def summarize(history, accept):
    """Hit rate over the window, and median conf and box size over the hits."""
    hits = [h for h in history if h[0] >= accept]
    if not hits:
        return len(hits), len(history), 0.0, 0, 0
    return (
        len(hits),
        len(history),
        statistics.median(h[0] for h in hits),
        int(statistics.median(h[1] for h in hits)),
        int(statistics.median(h[2] for h in hits)),
    )


def draw(frame, dets, lines, accept):
    scale = DISPLAY_WIDTH / frame.shape[1]
    view = cv2.resize(frame, None, fx=scale, fy=scale)
    for label, conf, (x1, y1, x2, y2) in dets:
        color = (0, 220, 0) if conf >= accept else (0, 140, 255)
        p1, p2 = (int(x1 * scale), int(y1 * scale)), (int(x2 * scale), int(y2 * scale))
        cv2.rectangle(view, p1, p2, color, 2)
        text = f"{label} {conf:.2f}  {x2 - x1}x{y2 - y1}px"
        cv2.putText(view, text, (p1[0], max(p1[1] - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    for i, line in enumerate(lines):
        y = 26 + i * 24
        cv2.putText(view, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
        cv2.putText(view, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
    return view


def log(row):
    new = not LOG.exists()
    with LOG.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if new:
            writer.writeheader()
        writer.writerow(row)
    print(", ".join(f"{k}={v}" for k, v in row.items()))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", default="0", help="name (substring is fine) or index")
    p.add_argument("--objects", default="keys,wallet,glasses", help="comma-separated")
    p.add_argument("--model", default="yolov8s-worldv2.pt",
                   help="a -world model takes any object name; yolo11s.pt etc. only know COCO classes")
    p.add_argument("--imgsz", type=int, default=640, choices=(640, 1280))
    p.add_argument("--accept", type=float, default=0.25, help="conf that counts as a hit")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    args = p.parse_args()

    objects = [o.strip() for o in args.objects.split(",") if o.strip()]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cap = open_camera(args.camera, args.width, args.height)
    model, classes = load_model(args.model, objects)
    imgsz = args.imgsz
    history = {o: deque(maxlen=WINDOW) for o in objects}

    while True:
        ok, frame = cap.read()
        if not ok:
            sys.exit("camera stopped returning frames")

        dets, ms = detect(model, frame, imgsz, classes, device)
        for o in objects:
            best = max((d for d in dets if d[0] == o), key=lambda d: d[1], default=None)
            if best:
                x1, y1, x2, y2 = best[2]
                history[o].append((best[1], x2 - x1, y2 - y1))
            else:
                history[o].append((0.0, 0, 0))

        lines = [f"{args.model}  imgsz {imgsz}  {ms:.0f}ms  [{device}]   i: toggle size  1-9: log metres"]
        for o in objects:
            hits, n, conf, w, h = summarize(history[o], args.accept)
            lines.append(f"{o}: seen {hits}/{n}" + (f"  conf {conf:.2f}  {w}x{h}px" if hits else ""))
        cv2.imshow("range test", draw(frame, dets, lines, args.accept))

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("i"):
            imgsz = 1280 if imgsz == 640 else 640
            for h in history.values():
                h.clear()
        if ord("1") <= key <= ord("9"):
            for o in objects:
                hits, n, conf, w, h = summarize(history[o], args.accept)
                log({
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "camera": args.camera, "model": args.model, "imgsz": imgsz,
                    "distance_m": key - ord("0"), "object": o,
                    "hit_rate": f"{hits}/{n}", "conf": round(conf, 2), "box_w": w, "box_h": h,
                })

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
