"""The web page over a fake memory: board state, ask + guidance start, container targets, HTTP,
and the 🎤 button over a fake mic."""
import json
import threading
import time
import urllib.error
import urllib.request

import pytest

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


def test_photos_never_reach_the_page(tmp_path):
    web, memory = make(tmp_path)
    for t in (1, 2, 3):
        memory.update([S("keys", KEYS_TABLE)], FRAME, t)
    photo = memory.state["keys"].photo_path
    assert photo and "photo" not in web.ask("where are my keys")
    url = web.start(port=0).replace(web.start.__globals__["lan_ip"](), "127.0.0.1")
    with pytest.raises(urllib.error.HTTPError, match="404"):
        urllib.request.urlopen(url + "/photo/" + photo)
    assert b"/photo/" not in urllib.request.urlopen(url + "/").read()


def test_guidance_goes_to_the_band(tmp_path):
    import socket
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(2)
    cfg, store, memory, query = world(tmp_path)
    web = Web(cfg, memory.items, memory, StubPipeline(), memory.places, band="127.0.0.1")
    import time
    from recall.band import Udp
    web.band = Udp("127.0.0.1", rx.getsockname()[1])  # what start() opens for a wifi host, on a test port
    while not web.band.connected:
        time.sleep(0.01)
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


def test_room_guidance_keeps_showing_the_last_estimate(tmp_path):
    """A look that misses the person (nobody found, a hidden shoulder) keeps the last arrow up."""
    import json
    import time
    from pathlib import Path

    import numpy as np

    from recall.body import GREEN, Body
    from recall.room import Room
    web, memory = make(tmp_path)
    for t in (1, 2, 3):
        memory.update([S("keys", KEYS_TABLE)], FRAME, t)
    memory.state["keys"].xyz = [2.5, 0.45, 2.0]
    web.room = Room(json.loads((Path(__file__).resolve().parent / "fixtures/room/camera.json").read_text()))
    web.pose = type("NobodyInView", (), {"largest": lambda self, image: None})()
    web.body = Body(web.room)
    web.body.pos, web.body.yaw, web.body.seen_at = np.array([2.0, 1.5]), 0.0, time.monotonic()
    shown = np.zeros((720, 1280, 3), np.uint8)
    web._point_in_room("keys", shown, shown)
    assert (shown == GREEN).all(axis=2).sum() > 100


# --- the 🎤 button: the laptop's mic (fake here) and Whisper (fake here) ------------------------
class FakeMic:
    """Hears `audio`; with hold=True it keeps listening until the page presses stop."""
    def __init__(self, audio="a clip", hold=False, broken=False):  # the fake Whisper ignores it
        self.audio, self.hold, self.broken = audio, hold, broken
        self.started = threading.Event()

    def listen(self, stop):
        if self.broken:
            raise RuntimeError("Error querying device -1")
        self.started.set()
        while self.hold and not stop():
            time.sleep(0.01)
        return self.audio


def keys_on_table(tmp_path, mic, heard="where are my keys"):
    web, memory = make(tmp_path)
    for t in (1, 2, 3):
        memory.update([S("keys", KEYS_TABLE)], FRAME, t)
    web.mic, web.stt = mic, lambda audio: heard
    return web


def test_mic_button_answers_what_the_laptop_heard_and_guides(tmp_path):
    web = keys_on_table(tmp_path, FakeMic())
    r = web.listen()
    assert r["heard"] == "where are my keys" and "table" in r["answer"]
    assert web.guiding["item"] == "keys"


def test_mic_button_when_nobody_spoke_or_nothing_was_made_out(tmp_path):
    r = keys_on_table(tmp_path, FakeMic(audio=None)).listen()
    assert r["heard"] is None and "didn't hear anything" in r["answer"]
    web = keys_on_table(tmp_path, FakeMic(), heard="")    # nothing legible: never sent as a question
    web.ask = lambda text: pytest.fail(f"asked {text!r}")
    r = web.listen()
    assert r["heard"] is None and r["answer"] == "Sorry, I didn't catch that."
    assert web.guiding["item"] is None
    r = keys_on_table(tmp_path, FakeMic(broken=True)).listen()
    assert "microphone isn't working" in r["answer"]


def test_mic_button_one_at_a_time_and_stop_over_http(tmp_path):
    mic = FakeMic(hold=True)
    web = keys_on_table(tmp_path, mic)
    url = web.start(port=0).replace(web.start.__globals__["lan_ip"](), "127.0.0.1")
    get = lambda p: json.loads(urllib.request.urlopen(url + p, timeout=5).read())
    first = {}
    t = threading.Thread(target=lambda: first.update(get("/listen")))
    t.start()
    assert mic.started.wait(5)
    assert get("/listen") == {"busy": True}             # a second press elsewhere
    assert get("/listen?stop=1") == {"ok": True}        # pressing 🎤 again stops it
    t.join(5)
    assert first["heard"] == "where are my keys" and "table" in first["answer"]


# --- the band's button: the 🎤 button's job, the answer spoken on the laptop ------------------
def test_band_button_asks_out_loud_and_the_laptop_speaks_the_answer(tmp_path):
    web = keys_on_table(tmp_path, FakeMic())
    said = []
    web.speak = said.append
    web.band_pressed()
    assert len(said) == 1 and "table" in said[0]
    assert web.guiding["item"] == "keys"


def test_band_button_pressed_again_stops_listening(tmp_path):
    mic = FakeMic(hold=True)
    web = keys_on_table(tmp_path, mic)
    said = []
    web.speak = said.append
    first = threading.Thread(target=web.band_pressed)
    first.start()
    assert mic.started.wait(5)
    web.band_pressed()  # the second press, while it listens
    first.join(5)
    assert not first.is_alive() and "table" in said[0]


def test_the_page_sees_the_band_button_listening(tmp_path):
    """The 🎤 goes red for a band press too: /guide says the laptop is listening."""
    mic = FakeMic(hold=True)
    web = keys_on_table(tmp_path, mic)
    web.speak = lambda text: None
    assert web.guidance()["listening"] is False
    pressed = threading.Thread(target=web.band_pressed)
    pressed.start()
    assert mic.started.wait(5)
    assert web.guidance()["listening"] is True
    web.stop_listening()
    pressed.join(5)
    assert web.guidance()["listening"] is False


def test_guidance_json_survives_numpy_numbers(tmp_path):
    """Under NumPy 2, float / np.float32 stays float32; json.dumps rejected it and /guide went empty."""
    import numpy as np
    from recall.guide import Guide
    g = Guide.__new__(Guide)  # skip loading the pose model
    g.body, g.full_px = np.array([400.0, 300.0]), np.float32(120.0)
    assert type(g.metres((520.0, 300.0))) is float

    web, memory = make(tmp_path)
    for t in (1, 2, 3):
        memory.update([S("keys", KEYS_TABLE)], FRAME, t)
    web.ask("where are my keys")
    with web._lock:
        web.guiding.update(dir=np.float32(-37.6), dist=40, metres=np.float32(1.234), t=time.time())
    out = web.guidance()
    json.dumps(out)
    assert out["dir"] == -38 and type(out["dir"]) is int
    assert out["metres"] == 1.2 and type(out["metres"]) is float
