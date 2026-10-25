"""Shared fixtures for the logic tests: a fake item dictionary, zones and sightings."""
from dataclasses import replace

import numpy as np

from recall.config import Settings
from recall.memory import Memory
from recall.models import Item, Sighting, Zone
from recall.places import Places
from recall.query import Query
from recall.store import Store
from recall.zones import save_zones


def _it(oid, name, prompt, plural=False, container=False, aliases=()):
    return Item(oid, name, prompt, list(aliases), plural, container)


# Independent of items.json so prop changes don't break tests.
ITEMS = {
    "keys": _it("keys", "keys", "keys", plural=True, aliases=["key"]),
    "wallet": _it("wallet", "wallet", "wallet"),
    "glasses": _it("glasses", "glasses", "eyeglasses", plural=True),
    "phone": _it("phone", "phone", "cell phone"),
    "bag": _it("bag", "blue bag", "blue bag", container=True, aliases=["bag"]),
}
ZONES = [
    Zone("kitchen counter", [0, 300, 600, 720], "on", "manual"),
    Zone("table", [700, 300, 1280, 720], "on", "manual"),
]
FRAME = np.zeros((720, 1280, 3), np.uint8)

KEYS_COUNTER = [100, 400, 160, 440]
KEYS_TABLE = [900, 600, 960, 640]
GLASSES_COUNTER = [300, 420, 380, 460]
GLASSES_OVER_BAG = [850, 400, 900, 430]
BAG_TABLE = [800, 350, 1000, 560]
BAG_COUNTER = [100, 350, 300, 560]
AIR = [640, 100, 680, 150]          # base point in no zone


def S(oid, box, score=0.5):
    return Sighting(oid, box, score, 0)


def world(tmp, items=ITEMS, zones=ZONES, pixel_to_world=None, **overrides):
    """A fresh memory in tmp: (cfg, store, memory, query)."""
    cfg = replace(Settings(data_dir=tmp, items_path=tmp / "items.json"), **overrides)
    save_zones(zones, cfg.zones_path)
    store = Store(tmp)
    return cfg, store, Memory(items, Places(zones), store, cfg, pixel_to_world), Query(cfg, items, store)


def kinds(events):
    return [f"{e.kind}:{e.object}" for e in events]
