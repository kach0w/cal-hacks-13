"""The web page over a fake memory: board state, ask + guidance start, container targets, HTTP."""
import json
import urllib.request

from fakes import BAG_TABLE, FRAME, GLASSES_OVER_BAG, KEYS_TABLE, S, world
from recall.camera import Frame
from recall.pipeline import Result
from recall.web import Web


class StubPipeline:
    def __init__(self):
        self.result = Result(Frame(FRAME, 1.0, 1), "first", None)

    def latest(self):
        return self.result


def make(tmp_path):
    cfg, store, memory, query = world(tmp_path)
    web = Web(cfg, memory.items, memory, StubPipeline(), memory.places)
    return web, memory


def test_state_lists_items_by_place(tmp_path):
    web, memory = make(tmp_path)
    for t in (1, 2, 3):
        memory.update([S("keys", KEYS_TABLE)], FRAME, t)
    s = web.state()
    keys = next(i for i in s["items"] if i["id"] == "keys")
    assert keys["place"] == "table" and keys["status"] == "observed"
    assert next(i for i in s["items"] if i["id"] == "wallet")["status"] == "never"
    assert s["places"] == ["kitchen counter", "table"]
    assert any("keys" in e["text"] for e in s["log"])


def test_ask_answers_and_starts_guidance(tmp_path):
    web, memory = make(tmp_path)
    for t in (1, 2, 3):
        memory.update([S("keys", KEYS_TABLE)], FRAME, t)
    r = web.ask("where are my keys?")
    assert "table" in r["answer"]
    assert web.guiding["item"] == "keys"
    assert web.guidance()["item"] == "keys" and web.guidance()["dir"] is None  # nobody in frame yet


def test_ask_about_unknown_thing_does_not_guide(tmp_path):
    web, _ = make(tmp_path)
    web.ask("where is my umbrella")
    assert web.guiding["item"] is None


def test_guides_to_the_container_when_item_is_inside(tmp_path):
    web, memory = make(tmp_path)
    t = 0
    for _ in range(3):
        t += 1
        memory.update([S("bag", BAG_TABLE), S("glasses", GLASSES_OVER_BAG)], FRAME, t)
    for _ in range(10):  # glasses vanish over the bag
        t += 1
        memory.update([S("bag", BAG_TABLE)], FRAME, t)
    assert memory.state["glasses"].status == "inferred"
    box, name = web._target("glasses")
    assert name == "blue bag" and box == memory.state["bag"].last_bbox


def test_http_routes(tmp_path):
    web, memory = make(tmp_path)
    for t in (1, 2, 3):
        memory.update([S("keys", KEYS_TABLE)], FRAME, t)
    url = web.start(port=0).replace(web.start.__globals__["lan_ip"](), "127.0.0.1")
    get = lambda p: urllib.request.urlopen(url + p).read()
    assert b"<title>Recall</title>" in get("/")
    assert json.loads(get("/state"))["places"] == ["kitchen counter", "table"]
    assert "table" in json.loads(get("/ask?q=where%20are%20my%20keys"))["answer"]
    assert json.loads(get("/guide"))["item"] == "keys"


def test_guidance_goes_to_the_band(tmp_path):
    import socket
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(2)
    cfg, store, memory, query = world(tmp_path)
    web = Web(cfg, memory.items, memory, StubPipeline(), memory.places, band="127.0.0.1")
    web.band = ("127.0.0.1", rx.getsockname()[1])
    for t in (1, 2, 3):
        memory.update([S("keys", KEYS_TABLE)], FRAME, t)
    web.ask("where are my keys")

    class FakeGuide:  # a person facing the camera, standing left of the keys
        body, full_px = __import__("numpy").array([400.0, 300.0]), 120.0
        facing = __import__("numpy").array([0.0, 1.0])
        def update(self, image): return True
        def toward(self, target, diag): return -90.0, 40
        def metres(self, target): return 1.5
        def facing_degrees(self): return 0.0

    web.guide = FakeGuide()
    web._guide_once()
    assert rx.recv(64) == b"-90,40"
    assert web.guidance()["words"] == "To your left"
