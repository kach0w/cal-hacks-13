"""The web page over a fake memory: board state, ask + guidance start, container targets, HTTP."""
import json
import urllib.request

from fakes import BAG_TABLE, FRAME, GLASSES_OVER_BAG, KEYS_TABLE, S, world
from recall.camera import Frame
from recall.pipeline import Result
from recall.web import Web


class StubCamera:
    live = True

    def __init__(self):
        self.frame = Frame(FRAME.copy(), 1.5, 9)  # newer than the last detection's frame

    def read(self):
        return self.frame


class StubPipeline:
    def __init__(self, source=None):
        self.result = Result(Frame(FRAME, 1.0, 1), "first", None)
        self.source = source or type("Video", (), {"live": False})()

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


def test_guidance_looks_at_the_newest_camera_frame(tmp_path):
    """Detection looks once a second; guidance on its frame froze the arrow between looks."""
    cfg, store, memory, query = world(tmp_path)
    camera = StubCamera()
    web = Web(cfg, memory.items, memory, StubPipeline(camera), memory.places)
    for t in (1, 2, 3):
        memory.update([S("keys", KEYS_TABLE)], FRAME, t)
    web.ask("where are my keys")
    looked_at = []
    web._point = lambda oid, image, shown: looked_at.append(image)
    web._guide_once()
    assert looked_at[0] is camera.frame.image


def test_band_keeps_its_side_when_the_item_is_behind():
    """Straight behind, wobble flips the turn between +179 and -179: the band mustn't follow it
    edge to edge, but it must change sides once the item is clearly on the other one."""
    from recall.web import band_dir
    side, sent = 0, []
    for deg in (170, -178, 179, -175, -160, -140, -60, 30):
        d, side = band_dir(deg, side)
        sent.append(d)
    assert sent == [170, 178, 179, 175, 160, -140, -60, 30]


def test_band_waves_when_the_item_is_past_its_reach(tmp_path):
    """The servo reaches ±90: further round it would sit still at its edge, which on the left is
    also where it rests. It waves there instead, keeping to its side straight behind."""
    web, _ = make(tmp_path)
    sent = [web._send_band(deg, 50) for deg in (60, 120, 120, 170, -175, -100, -40)]
    assert sent == [60, 90, 60, 90, 60, -90, -40]  # -175 kept to the right; -100 is clearly left
