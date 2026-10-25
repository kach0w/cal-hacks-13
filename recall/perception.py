"""Perception (DESIGN §6): one pass -> boxes that already know which enrolled item they are.

Identity = which enrolled label fired, one best box per item. That holds while every tracked item
is one of a kind and visually distinct (PLAN §13: pick distinct props).

Backends, chosen by the DETECTOR setting — nothing else changes when you swap:
    yolo-world   YOLO-World (yolov8s-worldv2) prompted with each item's `prompt`, plus "person"
    yoloe        YOLOE (yoloe-26l-seg), the newer open-vocabulary model, prompted the same way
    yolo         a fine-tuned YOLO (e.g. YOLO11n on these props) via DETECT_MODEL=path/best.pt;
                 its class names must be item ids (or names/prompts); "person" is optional
Person boxes are reported (hands get their own label instead of being forced onto an item) but
never become objects.

Re-id hook (task 6, only if two props swap identity): give make_detector an IdentityCheck that
scores a crop against the item's references (reference_images). >= REID_ACCEPT commits, between
REID_HOLD and REID_ACCEPT holds (neither seen nor missed), below that the box is ignored.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .models import Box, Item, Sighting

log = logging.getLogger(__name__)
PERSON = "person"
DEFAULT_MODEL = {"yolo-world": "yolov8s-worldv2.pt", "yoloe": "yoloe-26l-seg.pt"}  # DETECT_MODEL=""


@dataclass
class Detections:
    sightings: list[Sighting]                          # best box per enrolled item
    persons: list[Box] = field(default_factory=list)
    held: set[str] = field(default_factory=set)        # items re-id could not confirm or reject


class Detector(Protocol):
    def detect(self, image, ts: float, raw: bool = False) -> Detections: ...
    def warmup(self, h: int, w: int) -> None: ...


class IdentityCheck(Protocol):
    def similarity(self, object: str, crop) -> float: ...   # cosine to that item's references


class UltralyticsDetector:
    """Any ultralytics model whose class i means labels[i]: an item id, PERSON, or None (ignore)."""

    def __init__(self, items: dict[str, Item], cfg, model, labels, identity: IdentityCheck | None = None):
        self.items, self.cfg, self.model, self.labels = items, cfg, model, labels
        self.identity = identity

    def threshold(self, oid) -> float:
        s = self.items[oid].min_score
        return self.cfg.default_min_score if s is None else s

    def detect(self, image, ts, raw=False) -> Detections:
        """raw=True ignores per-item thresholds and re-id (calibration uses it to set them)."""
        floor = 0.01 if raw else min(self.threshold(i) for i in self.items)
        r = self.model.predict(image, imgsz=self.cfg.detect_imgsz, conf=floor, verbose=False,
                               device=device(self.cfg))[0]
        best, persons = {}, []
        for box, cls, score in _boxes(r):
            label = self.labels[cls] if cls < len(self.labels) else None
            if label == PERSON:
                persons.append([round(v, 1) for v in box])
            elif label in self.items and (raw or score >= self.threshold(label)):
                if label not in best or score > best[label].score:
                    best[label] = Sighting(label, [round(v, 1) for v in box], round(score, 3), ts)
        dets = Detections(list(best.values()), persons)
        return dets if raw or self.identity is None else self._reid(dets, image)

    def _reid(self, dets, image) -> Detections:
        keep, held = [], set()
        for s in dets.sightings:
            x1, y1, x2, y2 = (max(0, int(v)) for v in s.bbox)
            sim = self.identity.similarity(s.object, image[y1:y2, x1:x2])
            if sim >= self.cfg.reid_accept:
                keep.append(s)
            elif sim >= self.cfg.reid_hold:
                held.add(s.object)
            else:
                log.debug("re-id rejected %s (%.2f)", s.object, sim)
        return Detections(keep, dets.persons, held)

    def warmup(self, h, w):
        """The first predict is slow (graph build, MPS compile): pay it before the clock starts."""
        import numpy as np
        self.detect(np.zeros((h, w, 3), np.uint8), 0.0, raw=True)


def _boxes(result):
    """(xyxy, class index, score) per box of an ultralytics result."""
    b = result.boxes
    return zip(b.xyxy.cpu().numpy().tolist(), b.cls.cpu().numpy().astype(int).tolist(),
               b.conf.cpu().numpy().tolist())


def device(cfg) -> str | None:
    """DETECT_DEVICE, or the Apple GPU when there is one (ultralytics never picks it by itself)."""
    if cfg.detect_device:
        return cfg.detect_device
    import torch
    return "mps" if torch.backends.mps.is_available() else None


def make_detector(items: dict[str, Item], cfg, identity=None) -> UltralyticsDetector:
    ids = list(items)
    if cfg.detector in DEFAULT_MODEL:
        model = open_vocab(cfg, [items[i].prompt for i in ids] + [PERSON])
        labels = ids + [PERSON]
    elif cfg.detector == "yolo":
        from ultralytics import YOLO  # lazy: tests and queries never load torch
        if not cfg.detect_model:
            raise SystemExit("DETECTOR=yolo needs DETECT_MODEL=path/to/best.pt")
        model = YOLO(cfg.detect_model)
        labels = class_labels(model.names, items)
    else:
        raise SystemExit(f"unknown DETECTOR {cfg.detector!r} (yolo-world | yoloe | yolo)")
    log.info("detector %s (%s), imgsz %d", cfg.detector, cfg.detect_model or DEFAULT_MODEL[cfg.detector],
             cfg.detect_imgsz)
    return UltralyticsDetector(items, cfg, model, labels, identity)


def open_vocab(cfg, prompts):
    """An open-vocabulary model (YOLO-World or YOLOE) told to look for `prompts`."""
    kind = cfg.detector if cfg.detector in DEFAULT_MODEL else "yolo-world"
    path = cfg.detect_model if cfg.detector in DEFAULT_MODEL and cfg.detect_model else DEFAULT_MODEL[kind]
    if kind == "yoloe":
        from ultralytics import YOLOE
        model = YOLOE(path)
        model.set_classes(prompts, model.get_text_pe(prompts))
    else:
        from ultralytics import YOLOWorld
        model = YOLOWorld(path)
        model.set_classes(prompts)
    return model


def class_labels(names: dict[int, str], items: dict[str, Item]) -> list[str | None]:
    """Map a fine-tuned model's class names onto item ids (by id, name or prompt)."""
    lookup = {PERSON: PERSON}
    for oid, it in items.items():
        for key in (it.prompt, it.name, oid):
            lookup[key.lower()] = oid
    labels = [lookup.get(names[i].lower()) for i in range(len(names))]
    missing = set(items) - set(labels)
    if missing:
        log.warning("model has no class for %s — they will never be seen", ", ".join(sorted(missing)))
    return labels


def reference_images(item: Item, data_dir) -> list[Path]:
    """Re-id references, best first: calibration crops from the corner camera, then the cutout."""
    data_dir = Path(data_dir)
    refs = sorted((data_dir / "enroll" / item.id).glob("*.jpg"))
    if item.reference and (data_dir / item.reference).exists():
        refs.append(data_dir / item.reference)
    return refs


def detect_prompts(image, prompts, cfg, conf=0.1) -> list[dict]:
    """One-off open-vocabulary detection (zone onboarding): [{label, bbox, score}]."""
    r = open_vocab(cfg, prompts).predict(image, imgsz=cfg.detect_imgsz, conf=conf, verbose=False,
                                         device=device(cfg))[0]
    return [{"label": prompts[c], "bbox": [float(v) for v in box], "score": float(s)}
            for box, c, s in _boxes(r)]
