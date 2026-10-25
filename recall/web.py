"""The web page on top of the live loop: board, ask box with spoken answers, and left/right
guidance to the item asked about.

    python -m recall.live --web            # then open the printed link on any device on the wifi

Runs inside live (it needs the newest frame and the memory), on its own threads:
  - an HTTP server (stdlib, no extra dependencies)
  - a guidance thread: while someone is being guided, finds them with a pose model and points
    them at the item, or at its container when the memory says it went into one. With the room
    registered this happens in the scanned room (recall/body.py), else in the picture (guide.py).

The "Ask out loud" button listens through the laptop's mic (recall/voice.py: Whisper on the laptop), so it
works from any device on the wifi; press it again to stop listening sooner. The answer is spoken
back by the page. The band's button (over Bluetooth) is the same press, and the laptop speaks that
answer itself. With MOSS_PROJECT_ID/MOSS_PROJECT_KEY set, Moss resolves vague wording to an item
(moss_search.py).

Routes: /  /state  /ask?q=  /listen  /listen?stop=1  /guide  /cam.jpg

With --band, every guidance update also goes to the band (recall/band.py: Bluetooth to the ESP32,
or wifi to a host) as "dir,dist": dir in degrees, + = to the person's right; dist 0..100. The servo reaches
±90, so further round the arrow waves at that side's edge: 'turn around, this way'. When guidance
stops, nothing is sent and the band parks its arrow by itself after a second.
"""
import json
import logging
import math
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2

from .ask import find_item_info, handle
from .band import connect as connect_band
from .body import Body, PoseModel, draw
from .moss_search import MODEL as MOSS_MODEL, MossSearch, wording
from .guide import SHOULDER_M, Guide, words
from .overlay import live_view
from .query import Query, clock
from .voice import Mic, Transcriber, speak

log = logging.getLogger("recall")

PORT = 8000
GUIDE_FOR_S = 60       # stop guiding this long after a question
GUIDE_EVERY_S = 0.1    # pause between the guidance thread's looks
BAND_HOLD_S = 1.5      # person lost for a moment: keep sending the last direction this long
BAND_BEHIND_DEG = 150  # further round than this either way, the band keeps the side it was on
BAND_REACH_DEG = 90    # the servo's travel each side of ahead (firmware/band)
BAND_WAVE_DEG = 30     # past its reach the arrow swings between the edge and this far in from it
CAM_WIDTH = 960        # the page's camera picture


def band_dir(deg, side):
    """(direction to send the band, the side it's on now: +1 right, -1 left). The band's servo
    reaches ±90, so anything further round sits at that side's edge. Straight behind, a few
    degrees of wobble flip the turn between +179 and -179, which would swing the arrow edge to edge
    each time: past BAND_BEHIND_DEG it keeps to the side it was on."""
    if side and abs(deg) > BAND_BEHIND_DEG and math.copysign(1, deg) != side:
        deg = -deg
    return deg, math.copysign(1, deg)


class Web:
    def __init__(self, cfg, items, memory, pipeline, places, room=None, band=None):
        self.cfg, self.items, self.memory, self.pipeline = cfg, items, memory, pipeline
        self.places, self.room = places, room
        self.query = Query(cfg, items, memory.store)
        self.query.moss = MossSearch.from_env(items)
        self.guide = None  # the pose model loads on the first question, not at start-up
        self.pose = self.body = None  # the same, for guidance in the room
        self._lock = threading.Lock()
        self.guiding = {"item": None, "since": 0.0, "dir": None, "dist": 100, "metres": None,
                        "target": None, "note": None, "t": 0.0}
        self.cam_jpg = None
        self.band_host = band
        self.band = None  # the link to the band (recall/band.py), opened by start() when --band was given
        self.band_side = 0  # band_dir's side, 0 before anything is sent
        self._waving = False  # which end of the wave the last packet past the servo's reach was
        self.mic, self.stt = Mic(), Transcriber(items)  # the "Ask out loud" button's; warm_voice() readies them
        self.speak = speak  # says the band button's answers on the laptop (macOS say)
        self._listening = threading.Lock()  # one question at a time from the page
        self._stop_listening = threading.Event()
        # More GET routes: path -> fn(query params, as parse_qs gives them) -> a dict (sent as JSON)
        # or (body bytes, content type). Each integration registers its own below.
        self.routes = {}
        # [care] Ask about the day: Claude answers the care partner from the memory (recall/care.py)
        from .care import Care
        self.care = Care(self.query, memory.store)
        self.routes["/care"] = lambda p: self.care.ask(p.get("q", [""])[0])
        self.routes["/care/summary"] = lambda p: self.care.summary()

        # [week] Tiger Data: this week, from the database (a continuous aggregate) or this laptop's files
        from .db import week
        self.routes["/week"] = lambda params: week(
            self.memory.store, {oid: it.name for oid, it in self.items.items()})

        # Cartesia: the page's spoken words in the same calm voice as the laptop's (recall/tts.py)
        from . import tts
        self.routes["/tts"] = tts.route(tts.default())

        # Moss semantic search: whether it's on, and what in the history matches a question
        self.routes["/moss"] = lambda params: self.moss_status()
        self.routes["/moss/events"] = lambda params: self.moss_events(params.get("q", [""])[0])

    def _send_band(self, deg, dist):
        """Sends a turn to the band; returns the direction it was sent."""
        deg, self.band_side = band_dir(deg, self.band_side)
        if abs(deg) > BAND_REACH_DEG:
            # The servo can't point there and would sit still at its edge (the left edge is also
            # where it rests): wave at that edge instead, a step per packet.
            deg = math.copysign(BAND_REACH_DEG - BAND_WAVE_DEG * self._waving, deg)
            self._waving = not self._waving
        if self.band:
            self.band.send(deg, dist)
        return deg

    def _hold(self):
        """The person was lost for a moment (a hidden shoulder, someone walking past): repeat the last
        direction for BAND_HOLD_S, so the band doesn't drop to rest and swing back."""
        with self._lock:
            deg, dist, t = self.guiding["dir"], self.guiding["dist"], self.guiding["t"]
        if deg is not None and time.time() - t < BAND_HOLD_S:
            self._send_band(deg, dist)

    # --- what the page reads -------------------------------------------------------------

    def state(self) -> dict:
        now = time.time()
        st = self.memory.state
        items = []
        for oid, it in self.items.items():
            o = st.get(oid)
            seen = o is not None and o.last_observed is not None
            items.append({
                "id": oid, "name": it.name,
                "place": o.surface if seen else None,
                "status": o.status if seen else "never",
                "container": st[o.container].name if seen and o.container in st else None,
                "age": now - o.last_observed if seen else None,
            })
        events = self.memory.store.events()[-50:]
        log_ = [{"t": e.ts, "text": self.query.describe(e, self.places)} for e in events]
        hb = self.query.as_of(now)
        return {"places": self.places.names(), "items": items, "log": log_, "now": now,
                "frozen_at": hb, "frozen_text": f"Camera paused since {clock(hb)}" if hb else None}

    def ask(self, text: str) -> dict:
        try:
            a = handle(text, self.query)
        except ValueError as e:
            return {"answer": str(e), "status": "info"}
        t = text.strip().lower()  # as handle() asks, so Moss answers both from one query
        oid, understood = find_item_info(self.query, t)
        if oid and a.status in ("observed", "inferred", "unknown"):
            with self._lock:
                self.guiding.update(item=oid, since=time.time(), dir=None, target=None, note=None)
        r = {"answer": a.text, "status": a.status}
        # Moss: when it (not the exact words) worked out the item, the page says how it understood
        if understood and (a.status != "info" or t.startswith("history")):
            r["understood"] = {**understood, "name": self.query.name(oid), "words": wording(text)}
        if t.startswith("when ") and oid and self.query.moss:  # and for "when did...", that item's moment in the history
            r["events"] = self.moss_events(text, about=oid)["matches"][:1]
        return r

    def moss_status(self) -> dict:
        """Never waits: an index still building reads as not ready."""
        m = self.query.moss
        if m is None:
            return {"on": False}
        return {"on": True, "ready": m.ready(0), "model": MOSS_MODEL, "min_score": m.min_score}

    def moss_events(self, q: str, about=None) -> dict:
        """GET /moss/events?q=: the events in the memory's log that best match a question about
        what happened ("when did the glasses go into the bag?"), found by Moss, best first.
        about: only the events of that item, or of things going into it."""
        if self.query.moss is None:
            return {"on": False, "matches": []}
        places = self.query.places()

        def told(e):  # the log's words, without the confidence number, and never surer than the memory
            text = self.query.describe(e, places)
            if e.kind == "inferred_contained":
                text = text.split(" (inferred")[0].replace(" went into ", " probably went into ", 1)
            return text
        hits, ms = self.query.moss.search_events(q, self.memory.store.events(), told)
        hits = [h for h in hits if about is None or about in (h[0].object, h[0].container)]
        return {"on": True, "ms": None if ms is None else round(ms, 1),
                "matches": [{"t": e.ts, "time": clock(e.ts), "item": e.object, "text": text, "score": round(score, 2)}
                            for e, text, score in hits]}

    def listen(self) -> dict:
        """The "Ask out loud" button: the laptop's mic hears one question, answered as ask() would, with what was
        heard. A second press while it's listening gets {"busy": true}."""
        if not self._listening.acquire(blocking=False):
            return {"busy": True}
        try:
            self._stop_listening.clear()
            audio = self.mic.listen(stop=self._stop_listening.is_set)
            heard = self.stt(audio) if audio is not None else None
        except Exception as e:  # no mic, no permission, no Whisper: say so on the page
            log.warning("voice: %s", e)
            return {"heard": None, "answer": "The laptop's microphone isn't working.", "status": "info"}
        finally:
            self._listening.release()
        if not heard:
            return {"heard": None, "status": "info",
                    "answer": "Sorry, I didn't catch that." if audio is not None
                    else "I didn't hear anything. Speak toward the laptop."}
        log.info("voice: heard %r", heard)
        return {"heard": heard, **self.ask(heard)}

    def band_pressed(self):
        """The band's button: the "Ask out loud" button's job, with the answer spoken on the laptop
        (nobody is looking at a page). A press while a question is being heard ends it, as a second
        page press does."""
        if self._listening.locked():
            self.stop_listening()
            return
        r = self.listen()
        if r.get("busy"):  # the page or the terminal started listening first
            return
        log.info("voice (band): heard %r -> %s", r.get("heard"), r["answer"])
        self.speak(r["answer"])

    def stop_listening(self):
        self._stop_listening.set()

    def warm_voice(self):
        """Open the mic and load Whisper before the first press: that question is then as quick as
        the rest, and the mic already knows how loud the room is (voice.Mic)."""
        try:
            self.mic.open()
            self.stt.load()
        except Exception as e:
            log.warning("voice: the 'Ask out loud' button won't work (%s)", e)

    def guidance(self) -> dict:
        with self._lock:
            g = dict(self.guiding)
        live = g["item"] and g["dir"] is not None and time.time() - g["t"] < 1
        name = self.items[g["item"]].name if g["item"] else None
        return {"item": name, "target": g["target"],
                "dir": int(round(float(g["dir"]))) if live else None,  # plain Python numbers: json can't write numpy
                "words": words(g["dir"], g["dist"]) if live else g["note"],
                "metres": round(float(g["metres"]), 1) if live and g["metres"] is not None else None,
                "listening": self._listening.locked()}  # the laptop mic is hearing a question, from any press

    # --- guidance thread -----------------------------------------------------------------

    def _target(self, oid):
        """(box, name) to point at: the item, or its container when it's inferred inside one."""
        st = self.memory.state
        o = st.get(oid)
        if o is None or o.last_bbox is None:
            return None
        if o.status == "inferred" and o.container in st and st[o.container].last_bbox is not None:
            c = st[o.container]
            return c.last_bbox, c.name
        return o.last_bbox, o.name

    def _target_xyz(self, oid):
        """(world point, name) to point at in the room, like _target; None without a room position."""
        st = self.memory.state
        o = st.get(oid)
        if o is None:
            return None
        if o.status == "inferred" and o.container in st and st[o.container].xyz is not None:
            o = st[o.container]
        return (o.xyz, o.name) if o.xyz is not None else None

    def _guide_loop(self):
        while True:
            time.sleep(GUIDE_EVERY_S)
            try:
                self._guide_once()
            except Exception:  # guidance must never take the memory down
                log.exception("guidance pass failed")

    def _guide_once(self):
        r = self.pipeline.latest()
        if r is None:
            return
        # The newest camera frame: r.frame is the one detection last looked at, once per LOOP_S,
        # and guiding on it froze the arrow for a second at a time.
        frame = (self.pipeline.source.read() if self.pipeline.source.live else None) or r.frame
        shown = live_view(frame.image, r, self.places, self.memory.state, self.items, room=self.room)
        with self._lock:
            oid = self.guiding["item"]
            if oid and time.time() - self.guiding["since"] > GUIDE_FOR_S:
                self.guiding["item"] = oid = None
        if oid:
            self._point(oid, frame.image, shown)
        h, w = shown.shape[:2]
        small = cv2.resize(shown, (CAM_WIDTH, int(CAM_WIDTH * h / w)))
        jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
        with self._lock:
            self.cam_jpg = jpg

    def _point_in_room(self, oid, image, shown):
        """Guidance in the scanned room: the person and the item as world points (recall/body.py)."""
        found = self._target_xyz(oid)
        with self._lock:
            self.guiding["note"] = None if found else f"I don't know where in the room the {self.items[oid].name} is"
        if found is None:
            return
        xyz, target_name = found
        if self.pose is None:
            self.pose, self.body = PoseModel(), Body(self.room, self.cfg.shoulder_height_m, self.cfg.guide_smooth)
        person = self.pose.largest(image)
        seen = person is not None and self.body.update(*person)
        if not seen:
            self._hold()
            log.debug("guide: lost this look: %s", "nobody found" if person is None else self.body.lost)
            if self.body.pos is None:
                cv2.putText(shown, "looking for you", (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)
                return
        deg, dist, metres = self.body.turn(xyz)
        title = f"{target_name}: {words(deg, dist)}"
        if seen:
            with self._lock:
                self.guiding.update(dir=deg, dist=dist, metres=metres, target=target_name, t=time.time())
            sent = self._send_band(deg, dist)
            log.debug("guide: at (%.2f, %.2f) facing %+4.0f (%s; nose %.2f eye %.2f)  turn %+4.0f  band %+4.0f  %.1f m",
                      *self.body.pos, math.degrees(self.body.yaw), self.body.why, *self.body.face, deg, sent, metres)
        else:  # keep showing the last estimate rather than nothing
            title += f"  (seen {time.monotonic() - self.body.seen_at:.1f} s ago)"
        draw(shown, self.room, self.body, xyz, title, deg, metres)

    def _point(self, oid, image, shown):
        if self.room is not None:
            return self._point_in_room(oid, image, shown)
        found = self._target(oid)
        if found is None:
            return
        box, target_name = found
        if self.guide is None:
            self.guide = Guide()
        if not self.guide.update(image):
            self._hold()
            cv2.putText(shown, "face the camera for a second", (16, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 200, 255), 2)
            return
        target = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        h, w = image.shape[:2]
        result = self.guide.toward(target, math.hypot(w, h))
        if result is None:
            self._hold()
            return
        deg, dist = result
        metres = self.guide.metres(target)
        with self._lock:
            self.guiding.update(dir=deg, dist=dist, metres=metres, target=target_name, t=time.time())
        self._send_band(deg, dist)

        body = tuple(int(v) for v in self.guide.body)
        tip = tuple(int(v) for v in self.guide.body + self.guide.facing * 100)
        cv2.arrowedLine(shown, body, tip, (0, 255, 0), 3)
        cv2.line(shown, body, tuple(int(v) for v in target), (0, 200, 255), 2)
        cv2.putText(shown, f"{target_name}: {words(deg, dist)}", (16, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 200, 255), 2)
        # The math, on screen: which way they face, where the item is, the turn, the distance.
        face = self.guide.facing_degrees()
        bearing = math.degrees(math.atan2(target[0] - self.guide.body[0], target[1] - self.guide.body[1]))
        lines = [f"facing     {face:+5.0f} deg  (0 = toward camera)",
                 f"item at    {bearing:+5.0f} deg",
                 f"turn       {abs(deg):5.0f} deg {'right' if deg > 0 else 'left'}",
                 f"distance  ~{metres:.1f} m  ({SHOULDER_M:.2f} m shoulders = {self.guide.full_px:.0f} px)"
                 if metres else "distance   ?"]
        for i, line in enumerate(lines):
            cv2.putText(shown, line, (16, 80 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # --- server --------------------------------------------------------------------------

    def start(self, port=PORT) -> str:
        web = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                url = urlparse(self.path)
                if url.path == "/":
                    self._send(PAGE.encode(), "text/html; charset=utf-8")
                elif url.path == "/state":
                    self._json(web.state())
                elif url.path == "/ask":
                    self._json(web.ask(parse_qs(url.query).get("q", [""])[0]))
                elif url.path == "/listen":
                    if parse_qs(url.query).get("stop"):
                        web.stop_listening()
                        self._json({"ok": True})
                    else:
                        self._json(web.listen())
                elif url.path == "/guide":
                    self._json(web.guidance())
                elif url.path == "/cam.jpg":
                    with web._lock:
                        jpg = web.cam_jpg
                    self._send(jpg, "image/jpeg") if jpg else self.send_error(404)
                elif url.path in web.routes:
                    out = web.routes[url.path](parse_qs(url.query))
                    self._json(out) if isinstance(out, dict) else self._send(*out)
                else:
                    self.send_error(404)

            def _json(self, obj):
                self._send(json.dumps(obj).encode(), "application/json")

            def _send(self, body, ctype):
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        threading.Thread(target=server.serve_forever, name="web", daemon=True).start()
        threading.Thread(target=self._guide_loop, name="guidance", daemon=True).start()
        if self.band_host:
            self.band = connect_band(self.band_host, on_press=self.band_pressed)
        url = f"http://{lan_ip()}:{server.server_address[1]}"  # port 0 = any free port (tests)
        log.info("web page: %s", url)
        return url


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # picks the wifi interface; sends nothing
        return s.getsockname()[0]
    except OSError:
        return "localhost"
    finally:
        s.close()


PAGE = """<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Recall</title>
<style>
  :root { --bg: #f4f5f7; --card: #fff; --fg: #1c1e24; --muted: #6b6f7b; --line: #dfe1e6; --accent: #2f6fd0;
          --seen: #1f8a5b; --guess: #b86a00; }
  @media (prefers-color-scheme: dark) { :root { --bg: #14161b; --card: #1e2128; --fg: #eceef3; --muted: #9a9eab;
          --line: #2e323b; --accent: #6fa3ff; --seen: #4fcf96; --guess: #f0a640; } }
  body { margin: 0; background: var(--bg); color: var(--fg); font: 16px/1.45 -apple-system, system-ui, sans-serif; }
  main { max-width: 1100px; margin: 0 auto; padding: 20px 16px 40px; display: grid; gap: 20px; }
  h1 { margin: 0; font-size: 1.8rem; } h2 { margin: 0 0 10px; font-size: 1rem; color: var(--muted); }
  .top { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 20px; align-items: start; }
  @media (max-width: 760px) { .top { grid-template-columns: 1fr; } }
  form { display: flex; gap: 8px; }
  input { flex: 1; min-width: 0; font: inherit; padding: 12px 14px; border: 1px solid var(--line); border-radius: 10px;
          background: var(--card); color: var(--fg); }
  button { font: inherit; font-weight: 600; padding: 12px 18px; border: 0; border-radius: 10px; background: var(--accent); color: #fff; }
  #mic { background: var(--card); border: 1px solid var(--line); padding: 12px 14px; }
  #mic.on { background: #d0342f; border-color: #d0342f; }
  #answer { font-size: 1.25rem; font-weight: 600; min-height: 1.5em; white-space: pre-line; margin: 12px 0; }
  #guide { display: flex; align-items: center; gap: 18px; background: var(--card); border: 1px solid var(--line);
           border-radius: 12px; padding: 14px 18px; margin-top: 12px; }
  #arrow { width: 72px; height: 72px; flex: none; transition: transform 150ms linear; }
  #guide.lost #arrow { opacity: 0.2; }
  #guideWords { font-size: 1.8rem; font-weight: 700; }
  #guideSub { color: var(--muted); }
  #cam { width: 100%; border-radius: 12px; display: block; background: var(--card); aspect-ratio: 16 / 9; object-fit: cover; }
  .frozen { color: var(--guess); font-weight: 600; }
  .places { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; }
  .place { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 14px; }
  .place h3 { margin: 0 0 10px; text-transform: capitalize; }
  .chip { display: inline-block; margin: 0 6px 6px 0; padding: 6px 10px; border-radius: 999px; background: var(--bg);
          border: 1px solid var(--line); border-left: 4px solid var(--seen); }
  .chip.inferred { border-left-color: var(--guess); }
  .chip small { color: var(--muted); }
  .log { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 12px; max-height: 300px;
         overflow-y: auto; display: flex; flex-direction: column; gap: 6px; }
  .log div { padding: 6px 10px; border-radius: 10px; background: var(--bg); }
  .log div:last-child { outline: 2px solid var(--accent); }
  .log time { color: var(--muted); font-size: 0.85em; margin-right: 8px; }
  [hidden] { display: none !important; }  /* #guide's display: flex used to override hidden */
  .vh { position: absolute !important; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; }
  .card { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 14px; }
  h2 small { font-weight: 400; margin-left: 6px; }
  /* Moss: how vague wording was understood, under the answer */
  .understood { color: var(--muted); font-size: 0.9rem; white-space: pre-line; margin: -4px 0 12px; overflow-wrap: anywhere; }
  /* Tiger Data: this week */
  .week-scroll { position: relative; overflow-x: auto; }
  .week-grid { border-collapse: separate; border-spacing: 4px; margin: 0 -4px; width: calc(100% + 8px); min-width: 24rem; table-layout: fixed; }
  .week-grid col.thing { width: 6.5rem; }
  .week-grid th { text-align: left; font-weight: 600; text-transform: capitalize; }
  .week-grid thead th { color: var(--muted); font-size: 0.8rem; text-align: center; text-transform: none; }
  .week-grid tbody th { position: sticky; left: 0; background: var(--card); }
  .week-grid td { height: 32px; text-align: center; font-weight: 600; border-radius: 8px; background: var(--bg); }
  .week-grid td[data-n="1"] { background: color-mix(in srgb, var(--accent) 22%, var(--card)); }
  .week-grid td[data-n="2"] { background: color-mix(in srgb, var(--accent) 40%, var(--card)); }
  .week-grid td[data-n="3"] { background: color-mix(in srgb, var(--accent) 60%, var(--card)); }
  .week-spots { list-style: none; margin: 10px 0 0; padding: 0; }
  .week-spots li { padding: 6px 0; } .week-spots li + li { border-top: 1px solid var(--line); }
  .week-empty, .week-key { color: var(--muted); margin: 0; } .week-key { font-size: 0.85rem; margin-top: 8px; }
  /* Claude: ask about the day */
  #careForm { flex-wrap: wrap; } #careForm input { flex-basis: 100%; } #careForm button { flex: 1; }
  #careNote { background: var(--card); color: var(--fg); border: 1px solid var(--line); }
  #careForm button:disabled { opacity: 0.6; }
  #careText { white-space: pre-line; margin: 12px 0 0; overflow-wrap: anywhere; }
  #careOut[data-state="busy"] #careText, #careOut[data-state="off"] #careText { color: var(--muted); }
  .care-looked { color: var(--muted); font-size: 0.85rem; margin: 8px 0 0; }
</style></head>
<body><main>
  <h1>Recall</h1>
  <div class="top">
    <section>
      <form id="ask"><input id="q" placeholder="Where are my keys?" autocomplete="off"><button type="button" id="mic" title="Ask out loud (the laptop listens)">🎤</button></form>
      <div id="answerCard"><p id="answer"></p><p class="understood" id="understood" hidden></p></div>
      <div id="guide" hidden>
        <svg id="arrow" viewBox="0 0 100 100"><polygon points="50,6 66,52 54,46 54,94 46,94 46,46 34,52" fill="var(--accent)"/></svg>
        <div><div id="guideWords">Looking for you…</div><div id="guideSub"></div></div>
      </div>
    </section>
    <section><img id="cam" alt="Live camera"><p id="frozen" class="frozen" hidden></p></section>
  </div>
  <section><h2>Where things are</h2><div class="places" id="places"></div></section>
  <section><h2>What moved (newest at the bottom)</h2><div class="log" id="log"><div>Nothing yet.</div></div></section>
  <section id="weekBox"><h2>This week <small id="weekSrc"></small></h2>
    <div class="card" id="weekCard" aria-live="polite" aria-busy="true">
      <p class="week-empty" id="weekMsg">Loading</p>
      <div class="week-scroll" id="weekScroll" role="region" aria-label="Times each thing went out of sight, by day" tabindex="0" hidden>
        <table class="week-grid" id="weekGrid"></table></div>
      <ul class="week-spots" id="weekSpots" hidden></ul>
      <p class="week-key" id="weekKey" hidden>Each number is how many times it went out of sight that day.</p>
    </div>
  </section>
  <section id="careBox"><h2>Ask about the day <small>for family: Claude reads the memory and answers (not spoken aloud)</small></h2>
    <div class="card">
      <form id="careForm"><input id="careQ" aria-label="Ask Claude about the day" placeholder="Lost anything today?" autocomplete="off"><button type="submit" id="careAsk">Ask Claude</button><button type="button" id="careNote">Today's note</button></form>
      <div id="careOut" data-state="empty" aria-live="polite" hidden><p id="careText"></p><p class="care-looked" id="careLooked" hidden></p></div>
    </div>
  </section>
</main>
<script>
  const $ = id => document.getElementById(id);
  const t = s => new Date(s * 1000).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  const ago = a => a < 60 ? "now" : a < 3600 ? Math.floor(a / 60) + " min ago" : "earlier";  // floor: 30 s isn't "0 min ago"
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const cap = s => s ? s[0].toUpperCase() + s.slice(1) : "";
  // An answer is spoken whole: directions that start meanwhile wait for it (they used to cut it off).
  // Cartesia: words are spoken in the laptop's Cartesia voice (/tts, recall/tts.py) when it has one,
  // else, or when the audio isn't ready within 1.5 s, in the browser's own voice.
  let answering = 0;
  let voiceOff = false;  // the laptop has no Cartesia voice: stop asking
  let playing = null, turn = 0;
  const synth = window.speechSynthesis;  // missing in some in-app browsers: the page must still load
  const browserCancel = synth ? synth.cancel.bind(synth) : () => {};
  const hush = () => {
    turn++; answering = 0;
    if (playing) { playing.pause(); playing = null; }
    browserCancel();
  };
  if (synth) synth.cancel = hush;  // so the mic button's cancel silences this voice too
  const say = (text, isAnswer) => {
    hush();
    if (!text) return;
    const mine = ++turn;
    let done = () => {};
    if (isAnswer) {
      const until = answering = Date.now() + 1500 + 90 * text.length;  // in case onend never comes
      done = () => { if (answering === until) answering = 0; };
    }
    const browser = () => {
      if (mine !== turn) return;
      if (!synth) return done();
      const u = new SpeechSynthesisUtterance(text);
      u.onend = u.onerror = done;
      speechSynthesis.speak(u);
    };
    if (voiceOff || text.length > 400) return browser();  // 400: tts.MAX_CHARS
    const ctl = new AbortController(), late = setTimeout(() => ctl.abort(), 1500);
    fetch("/tts?text=" + encodeURIComponent(text), { signal: ctl.signal }).then(async r => {
      if (!(r.headers.get("Content-Type") || "").startsWith("audio/")) {
        if ((await r.json()).off) voiceOff = true;
        throw new Error("no voice");
      }
      const src = URL.createObjectURL(await r.blob());
      clearTimeout(late);
      if (mine !== turn) return URL.revokeObjectURL(src);
      const a = playing = new Audio(src);
      a.onended = () => { URL.revokeObjectURL(src); if (playing === a) playing = null; done(); };
      try { await a.play(); } catch (e) { if (playing === a) playing = null; URL.revokeObjectURL(src); throw e; }
    }).catch(browser).finally(() => clearTimeout(late));
  };

  const show = r => {
    $("answer").textContent = r.answer;
    understood(r.understood, r.events);
    say(r.answer, true);
  };

  $("ask").onsubmit = async e => {
    e?.preventDefault();
    show(await (await fetch("/ask?q=" + encodeURIComponent($("q").value))).json());
  };

  // Ask out loud: the laptop's mic listens until you stop talking (Whisper on the laptop,
  // recall/voice.py), so this works from a phone too; press again to stop sooner. Moss on the
  // server works out which item vague wording means.
  let listening = false;        // this page's own question is being heard
  let laptopListening = false;  // the laptop is hearing a question from anywhere (/guide)
  $("mic").onclick = async () => {
    if (listening || laptopListening) return fetch("/listen?stop=1");
    listening = true;
    hush(); understood(null);
    $("mic").classList.add("on"); $("q").value = ""; $("q").placeholder = "Listening… ask the laptop";
    try {
      const r = await (await fetch("/listen")).json();
      if (r.busy) { $("answer").textContent = "The laptop is already listening."; return; }
      $("q").value = r.heard || "";
      show(r);
    } catch (e) {
      $("answer").textContent = "Couldn't reach the laptop.";
    } finally {
      listening = false;
      $("mic").classList.remove("on"); $("q").placeholder = "Where are my keys?";
    }
  };

  let frozen = false;
  async function tick() {
    try {
      const s = await (await fetch("/state")).json();
      frozen = !!s.frozen_text;
      $("frozen").hidden = !frozen; $("frozen").textContent = s.frozen_text || "";
      const cols = [...s.places, null];
      $("places").innerHTML = cols.map(place => {
        const here = s.items.filter(i => i.place === place && i.status !== "never");
        if (place === null && !here.length) return "";
        const chips = here.map(i => `<span class="chip ${i.status}">${esc(i.name)}` +
          (i.container ? ` <small>in ${esc(i.container)}</small>` : "") +
          ` <small>${ago(i.age)}</small></span>`).join("");
        return `<div class="place"><h3>${esc(place ?? "Not in a named place")}</h3>${chips || "<small>nothing here</small>"}</div>`;
      }).join("");
      const never = s.items.filter(i => i.status === "never");
      if (never.length) $("places").innerHTML += `<div class="place"><h3>Not seen yet</h3>` +
        never.map(i => `<span class="chip">${esc(i.name)}</span>`).join("") + `</div>`;
      const box = $("log");
      if (s.log.length) {
        const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
        box.innerHTML = s.log.map(e => `<div><time>${t(e.t)}</time>${esc(e.text)}</div>`).join("");
        if (atBottom) box.scrollTop = box.scrollHeight;
      }
    } catch (e) {}
    setTimeout(tick, 1000);
  }

  let lastWords = "";
  async function guideTick() {
    try {
      const g = await (await fetch("/guide")).json();
      // The band's button (or the terminal) can start the laptop listening too: show it on the 🎤.
      laptopListening = g.listening;
      if (!listening) {
        $("mic").classList.toggle("on", g.listening);
        $("q").placeholder = g.listening ? "Listening… ask the laptop" : "Where are my keys?";
      }
      $("guide").hidden = !g.item;
      if (g.item) {
        const to = g.target && g.target !== g.item ? `the ${g.target} (your ${g.item} is in it)` : `your ${g.item}`;
        const math = g.dir === null ? "" : ` · ${Math.abs(g.dir)}° ${g.dir > 0 ? "right" : "left"}` + (g.metres ? ` · about ${g.metres} m` : "");
        $("guideSub").textContent = "Guiding you to " + to + math;
        $("guide").classList.toggle("lost", g.dir === null);
        // A paused camera can't see anyone: don't ask the person to step in front of it.
        $("guideWords").textContent = g.words ?? (frozen ? "Camera paused" : "Step in front of the camera");
        if (g.dir !== null) $("arrow").style.transform = `rotate(${g.dir}deg)`;
        if (g.words && g.words !== lastWords && Date.now() > answering) { lastWords = g.words; say(g.words); }
      }
    } catch (e) {}
    setTimeout(guideTick, 200);
  }

  function camTick() { $("cam").src = "/cam.jpg?t=" + Date.now(); }
  $("cam").onload = () => setTimeout(camTick, 150);
  $("cam").onerror = () => setTimeout(camTick, 1000);
  // [care] Ask about the day: Claude answers the family from the memory (/care, /care/summary).
  // The server gives up after 25 s; the page stops waiting at 35 s either way.
  async function careAsk(path) {
    const out = $("careOut"), btns = [$("careAsk"), $("careNote")];
    out.hidden = false; out.dataset.state = "busy"; out.setAttribute("aria-busy", "true");
    $("careText").textContent = "Reading today's memory"; $("careLooked").hidden = true;
    btns.forEach(b => b.disabled = true);
    const stop = new AbortController(), timer = setTimeout(() => stop.abort(), 35000);
    try {
      const r = await (await fetch(path, { signal: stop.signal })).json();
      out.dataset.state = r.status || "error";
      $("careText").textContent = r.answer || "No answer came back. Try again.";
      $("careLooked").hidden = !(r.looked_at && r.looked_at.length);
      $("careLooked").textContent = "Looked at: " + (r.looked_at || []).join("; ") + ".";
    } catch (e) {
      out.dataset.state = "error";
      $("careText").textContent = e.name === "AbortError" ? "That took too long. Try again in a moment." : "Couldn't reach the laptop. Try again in a moment.";
    } finally {
      clearTimeout(timer); out.setAttribute("aria-busy", "false");
      btns.forEach(b => b.disabled = false);
    }
  }
  $("careForm").onsubmit = e => {
    e.preventDefault();
    const q = $("careQ").value.trim();
    q ? careAsk("/care?q=" + encodeURIComponent(q)) : $("careQ").focus();
  };
  $("careNote").onclick = () => careAsk("/care/summary");

  // [week] Tiger Data: this week, from the database when the laptop has one (recall/db.py week())
  const weekday = (iso, i, n) => i === n - 1 ? "Today"
    : new Date(iso + "T12:00:00").toLocaleDateString([], { weekday: "short" });
  const times = n => n === 1 ? "once" : n === 2 ? "twice" : n + " times";
  const spot = s => s.container ? "into the " + s.container + (s.place ? " on the " + s.place : "")
    : s.place ? "on the " + s.place : "outside the named places";
  let lastWeek = null;
  async function weekTick() {
    const ctl = new AbortController(), timer = setTimeout(() => ctl.abort(), 8000);
    try {
      const r = await fetch("/week", { signal: ctl.signal });
      if (!r.ok) throw new Error(r.status);
      const w = await r.json(), key = JSON.stringify(w);
      if (key !== lastWeek) {
        lastWeek = key;
        $("weekSrc").textContent = w.source === "database" ? "From the database" : "From this laptop";
        const any = w.items.some(it => it.per_day.some(n => n));
        $("weekMsg").hidden = any;
        $("weekMsg").textContent = "Nothing went out of sight this week.";
        $("weekScroll").hidden = $("weekKey").hidden = !any;
        const n = w.days.length;
        $("weekGrid").innerHTML = `<colgroup><col class="thing">${"<col>".repeat(n)}</colgroup>` +
          `<thead><tr><th scope="col"><span class="vh">Thing</span></th>` +
          w.days.map((d, i) => `<th scope="col"${i === n - 1 ? ' class="today"' : ""}>${esc(weekday(d, i, n))}</th>`).join("") + `</tr></thead><tbody>` +
          w.items.filter(it => it.per_day.some(n => n)).map(it => `<tr><th scope="row">${esc(cap(it.name))}</th>` +
            it.per_day.map(c => `<td data-n="${Math.min(c, 3)}">${c || '<span class="vh">none</span>'}</td>`).join("") + `</tr>`).join("") +
          `</tbody>`;
        $("weekScroll").scrollLeft = $("weekScroll").scrollWidth;  // a narrow screen opens on today
        const by = {};
        for (const s of w.spots) (by[s.name] = by[s.name] || []).push(s);
        // "Usually" once something has gone missing more than once; before that, just where it went.
        const lines = Object.keys(by).map(name => `<li><b>${esc(cap(name))}</b>: ` +
          (by[name].some(s => s.times > 1) ? `usually ${/s$/.test(name) ? "go" : "goes"} missing ` : "went missing ") +
          by[name].slice(0, 3).map(s => esc(spot(s)) + " (" + times(s.times) + ")").join(", ") + `</li>`);
        $("weekSpots").innerHTML = lines.join("");
        $("weekSpots").hidden = !lines.length;
      }
    } catch (e) {
      if (lastWeek === null) $("weekMsg").textContent = "Couldn't load this week yet. Trying again soon.";
    } finally {
      clearTimeout(timer);
      $("weekCard").setAttribute("aria-busy", "false");
    }
    setTimeout(weekTick, 30000);
  }
  weekTick();

  // Moss semantic search: when Moss (not the exact words) worked out which thing was meant, one
  // quiet line under the answer says how, so a care partner can tell if it heard wrong.
  // For "when did..." questions, the moment in the history Moss found goes on the line too.
  function understood(u, events) {
    const el = $("understood"), ms = m => Math.max(1, Math.round(m || 0)) + " ms";
    const lines = [];
    if (u) lines.push(`Understood “${u.words}” as ${u.name} (Moss, ${ms(u.ms)})`);
    if (events?.length) lines.push(`From the history: ${events[0].time}, ${events[0].text} (Moss)`);
    el.hidden = !lines.length;
    el.textContent = lines.join("\\n");
  }

  tick(); guideTick(); camTick();
</script>
</body></html>
"""
