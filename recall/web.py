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
        # [care] routes

        # [week] routes

        # [tts] routes

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
                text = text.split(" — inferred")[0].replace(" went into ", " probably went into ", 1)
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
                "dir": round(g["dir"]) if live else None,
                "words": words(g["dir"], g["dist"]) if live else g["note"],
                "metres": round(g["metres"], 1) if live and g["metres"] is not None else None,
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
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>Recall</title>
<style>
  :root {
    --bg: #eef0f3; --surface: #ffffff; --sunken: #f5f6f8; --fg: #14171c; --muted: #596170; --line: #dde1e6;
    --accent: #b93d0b; --accent-press: #9a3209; --on-accent: #ffffff; --accent-soft: #fbeae2;
    --seen: #17704a; --seen-bg: #e3f2ea; --prob: #2b58a8; --prob-bg: #e6edf9; --unsure: #596170; --unsure-bg: #eceef2;
    --inverse: #14171c; --on-inverse: #f4f5f7; --on-inverse-muted: #c3c8d0;
    --r-s: 8px; --r-m: 12px; --r-l: 20px;
    --sans: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
    --round: ui-rounded, "SF Pro Rounded", system-ui, -apple-system, "Segoe UI", sans-serif;
    --ease: cubic-bezier(.2, .7, .2, 1);
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #0e1013; --surface: #171a1f; --sunken: #1d2127; --fg: #eceef2; --muted: #a0a7b3; --line: #2a2f37;
      --accent: #ff8a50; --accent-press: #ff9f70; --on-accent: #1c0b02; --accent-soft: #3a2014;
      --seen: #5ccf96; --seen-bg: #15291f; --prob: #93b6ff; --prob-bg: #1a2338; --unsure: #a9b0bb; --unsure-bg: #23272e;
      --inverse: #eceef2; --on-inverse: #14171c; --on-inverse-muted: #444b57;
    }
  }
  [hidden] { display: none !important; }
  *, *::before, *::after { box-sizing: border-box; }
  html { -webkit-text-size-adjust: 100%; }
  body { margin: 0; background: var(--bg); color: var(--fg); font: 16px/1.5 var(--sans); -webkit-font-smoothing: antialiased; }
  h1, h2, h3, p, ol, ul, figure { margin: 0; }
  ol, ul { padding: 0; list-style: none; }
  svg { display: block; flex: none; }
  .vh { position: absolute !important; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; }
  .muted { color: var(--muted); }
  .wrap { max-width: 1240px; margin: 0 auto; padding: 0 24px; }

  /* header */
  .top { display: flex; align-items: center; justify-content: space-between; gap: 16px; padding-top: 22px; padding-bottom: 18px; }
  .brand { display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap; min-width: 0; }
  h1 { font: 700 1.6rem/1.1 var(--round); letter-spacing: -0.01em; }
  .tag { color: var(--muted); }
  .live { display: inline-flex; align-items: center; gap: 8px; font-weight: 600; font-size: .9375rem; white-space: nowrap; }
  .live i { width: 10px; height: 10px; border-radius: 50%; background: var(--seen); }
  .live[data-s="paused"] i, .live[data-s="off"] i { background: transparent; box-shadow: inset 0 0 0 2px var(--muted); }
  .live[data-s="wait"] i { background: var(--line); }

  /* paused / offline notice */
  .notice { display: flex; align-items: center; gap: 16px; background: var(--inverse); color: var(--on-inverse);
            border-radius: var(--r-l); padding: 16px 20px; margin-bottom: 20px; }
  .notice svg { width: 32px; height: 32px; }
  .notice strong { display: block; font: 700 1.25rem/1.3 var(--round); }
  .notice span { color: var(--on-inverse-muted); }

  /* layout */
  main.wrap { display: grid; grid-template-columns: minmax(0, 7fr) minmax(0, 5fr); gap: 24px; align-items: start; padding-bottom: 48px; }
  .col { display: flex; flex-direction: column; gap: 24px; min-width: 0; }
  .card { background: var(--surface); border: 1px solid var(--line); border-radius: var(--r-l); }
  .sec-head { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; flex-wrap: wrap; margin-bottom: 12px; padding: 0 4px; }
  h2 { font: 650 1.1875rem/1.3 var(--sans); }
  .sec-head p { font-size: .9375rem; }

  /* ask */
  form { display: flex; gap: 10px; }
  input { flex: 1; min-width: 0; height: 56px; font: 500 1.125rem var(--sans); color: var(--fg); padding: 0 18px;
          background: var(--surface); border: 1px solid var(--line); border-radius: var(--r-m); }
  input::placeholder { color: var(--muted); opacity: 1; }
  input:focus { border-color: var(--accent); outline: none; }
  input:focus-visible { outline: 3px solid var(--accent); outline-offset: 2px; }
  button { font: 600 1rem var(--sans); color: inherit; cursor: pointer; display: inline-flex; align-items: center; justify-content: center;
           gap: 8px; min-height: 56px; padding: 0 20px; border-radius: var(--r-m); border: 1px solid transparent;
           transition: background-color 120ms, transform 120ms var(--ease); -webkit-tap-highlight-color: transparent; }
  button svg { width: 20px; height: 20px; }
  button:focus-visible { outline: 3px solid var(--accent); outline-offset: 2px; }
  button:active { transform: scale(.97); }
  .primary { background: var(--accent); color: var(--on-accent); }
  .primary:active { background: var(--accent-press); }
  .secondary { background: var(--surface); border-color: var(--line); color: var(--fg); }
  .secondary:hover { background: var(--sunken); }
  .secondary:active { background: var(--line); }
  #mic.on { background: var(--inverse); color: var(--on-inverse); border-color: var(--inverse); }
  @media (prefers-reduced-motion: no-preference) {
    #mic.on { animation: ring 1.6s var(--ease) infinite; }
    @keyframes ring { 0% { box-shadow: 0 0 0 0 color-mix(in srgb, var(--inverse) 35%, transparent); }
                      100% { box-shadow: 0 0 0 12px transparent; } }
  }
  .try { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; align-items: center; }
  .try span { color: var(--muted); font-size: .9375rem; margin-right: 2px; }
  .guiding .try { display: none; }
  .try button { min-height: 48px; padding: 0 16px; font-weight: 500; background: transparent; border-color: var(--line); color: var(--fg); }
  .try button:hover { background: var(--surface); }
  .try button:active { background: var(--line); }

  .answer { padding: 22px 24px 24px; min-height: 148px; }
  .answer-meta { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; min-height: 28px; margin-bottom: 8px; }
  .badge { font-weight: 650; font-size: .875rem; padding: 3px 10px; border-radius: var(--r-s); }
  .s-observed { color: var(--seen); } .s-inferred { color: var(--prob); } .s-unknown, .s-none, .s-never { color: var(--unsure); }
  .badge.s-observed { background: var(--seen-bg); } .badge.s-inferred { background: var(--prob-bg); }
  .badge.s-unknown, .badge.s-none { background: var(--unsure-bg); }
  .heard { color: var(--muted); font-size: .9375rem; overflow-wrap: anywhere; }
  .answer-text { font: 650 1.875rem/1.25 var(--round); letter-spacing: -0.01em; white-space: pre-line; overflow-wrap: anywhere; }
  .answer.list .answer-text { font: 400 1.0625rem/1.6 var(--sans); letter-spacing: 0; }
  .answer[data-state="empty"] .answer-text, .answer[data-state="busy"] .answer-text,
  .answer[data-state="listening"] .answer-text { color: var(--muted); font-weight: 600; }
  @media (prefers-reduced-motion: no-preference) {
    .answer[data-state="answer"] .answer-text { animation: rise 260ms var(--ease); }
    @keyframes rise { from { opacity: 0; transform: translateY(4px); } }
  }

  /* guidance */
  .guide { display: grid; grid-template-columns: auto minmax(0, 1fr); align-items: center; gap: 32px; padding: 28px 32px;
           border-color: var(--accent); box-shadow: inset 0 0 0 1px var(--accent); }
  .dial { position: relative; width: 248px; aspect-ratio: 1; flex: none; }
  .dial > svg { position: absolute; inset: 0; width: 100%; height: 100%; }
  .dial .face { fill: var(--sunken); stroke: var(--line); stroke-width: 1.5; }
  .dial .tick { stroke: var(--muted); stroke-width: 1.5; stroke-linecap: round; opacity: .55; }
  .dial .tick.major { stroke-width: 3; opacity: 1; }
  .dial .tick.ahead { stroke: var(--accent); stroke-width: 4; opacity: 1; }
  .dial text { fill: var(--muted); font: 600 10.5px var(--sans); text-anchor: middle; dominant-baseline: middle; }
  .needle { position: absolute; inset: 31%; color: var(--accent); }
  .seek { position: absolute; inset: 36%; color: var(--muted); opacity: 0; }
  .seek svg { width: 100%; height: 100%; }
  .needle svg { width: 100%; height: 100%; }
  @media (prefers-reduced-motion: no-preference) { .needle { transition: transform 420ms var(--ease), opacity 200ms; } }
  .guide.lost .needle { opacity: 0; }
  .guide.lost .seek { opacity: 1; }
  .guide.lost .dial .face { stroke-dasharray: 6 6; }
  .guide-to { color: var(--muted); font-size: 1.0625rem; }
  .guide-to b { color: var(--fg); font-weight: 650; }
  .guide-words { font: 750 3.5rem/1.05 var(--round); letter-spacing: -0.02em; margin: 6px 0 10px; overflow-wrap: anywhere; }
  .guide.lost .guide-words { font-size: 2.25rem; }
  .guide-math { color: var(--muted); font-variant-numeric: tabular-nums; font-size: 1.0625rem; }

  /* camera */
  .cam { position: relative; overflow: hidden; border-radius: var(--r-l); background: var(--sunken); border: 1px solid var(--line); aspect-ratio: 16 / 9; }
  .cam img { display: block; width: 100%; height: 100%; object-fit: cover; }
  .cam:not([data-s="live"]) img { display: none; }
  .cam-empty { position: absolute; inset: 0; display: grid; place-content: center; justify-items: center; gap: 6px; padding: 20px; text-align: center; }
  .cam[data-s="live"] .cam-empty { display: none; }
  .cam-empty svg { width: 36px; height: 36px; color: var(--muted); margin-bottom: 4px; }
  .cam-empty strong { font-weight: 650; }
  .cam-empty p { color: var(--muted); font-size: .9375rem; max-width: 30ch; }
  .cam-flag { position: absolute; left: 12px; top: 12px; background: var(--inverse); color: var(--on-inverse); font-weight: 650;
              font-size: .875rem; padding: 4px 10px; border-radius: var(--r-s); }
  .cam-cap { color: var(--muted); font-size: .875rem; margin-top: 10px; padding: 0 4px; }

  /* board */
  .places > li { display: grid; align-items: start; grid-template-columns: minmax(0, 11rem) minmax(0, 1fr); gap: 4px 20px; padding: 16px 22px; }
  .places > li + li { border-top: 1px solid var(--line); }
  .places h3 { font: 650 1.0625rem/1.4 var(--sans); display: flex; align-items: center; min-height: 60px; }
  .things li { display: flex; align-items: center; justify-content: space-between; gap: 6px 16px; min-height: 48px; padding: 6px 0; flex-wrap: wrap; }
  .things li + li { border-top: 1px solid var(--line); }
  .thing { font-weight: 600; font-size: 1.0625rem; }
  .st { text-align: right; line-height: 1.35; margin-left: auto; }
  .st b { display: block; font-weight: 650; }
  .st small { display: block; color: var(--muted); font-size: .875rem; }
  .things .none { color: var(--muted); }
  .key { display: flex; flex-wrap: wrap; gap: 6px 20px; padding: 14px 22px; border-top: 1px solid var(--line); font-size: .875rem; color: var(--muted); }
  .key b { font-weight: 650; }

  /* timeline */
  .log { max-height: 440px; overflow-y: auto; padding: 8px 0; overscroll-behavior: contain; }
  .log li { display: grid; grid-template-columns: 5.5rem minmax(0, 1fr); gap: 16px; padding: 9px 22px; }
  .log time { color: var(--muted); font-variant-numeric: tabular-nums; text-align: right; font-size: .9375rem; padding-top: 1px; }
  .log span { border-left: 2px solid var(--line); padding-left: 14px; overflow-wrap: anywhere; }
  .log li:last-child span { border-left-color: var(--accent); font-weight: 600; }
  .log .empty { display: block; color: var(--muted); padding: 18px 22px; }

  @media (max-width: 900px) {
    main.wrap { grid-template-columns: minmax(0, 1fr); gap: 20px; }
    .col { display: contents; }
    #guide { order: 0; } #askBox { order: 1; } #boardBox { order: 3; } #camBoxWrap { order: 4; } #logBox { order: 5; }
  }
  @media (max-width: 640px) {
    .wrap { padding: 0 16px; }
    .top { padding-top: 16px; padding-bottom: 14px; align-items: flex-start; }
    .tag { display: none; }
    form { flex-wrap: wrap; }
    input { flex-basis: 100%; }
    form button { flex: 1; }
    .try { flex-wrap: nowrap; overflow-x: auto; margin: 12px -16px 0; padding: 0 16px; scrollbar-width: none; }
    .try button { flex: none; }
    .answer { padding: 18px; min-height: 0; }
    .answer-text { font-size: 1.5rem; }
    .guide { grid-template-columns: minmax(0, 1fr); justify-items: center; text-align: center; gap: 18px; padding: 24px 18px; }
    .dial { width: min(240px, 68vw); }
    .guide-words { font-size: 2.75rem; }
    .places > li { grid-template-columns: minmax(0, 1fr); padding: 14px 18px; }
    .places h3 { min-height: 0; }
    .key { padding: 14px 18px; }
    .log li { grid-template-columns: 4.75rem minmax(0, 1fr); gap: 12px; padding: 9px 18px 9px 12px; }
    .notice { padding: 14px 16px; gap: 14px; }
  }
  /* [care] styles */

  /* [week] styles */

  /* Moss semantic search: how vague wording was understood, under the answer */
  .understood { margin-top: 12px; color: var(--muted); font-size: .9375rem; white-space: pre-line; overflow-wrap: anywhere; }

</style></head>
<body>
<header class="wrap top">
  <div class="brand"><h1>Recall</h1><p class="tag">Where things are in the room, and what moved.</p></div>
  <p class="live" id="live" data-s="wait" role="status"><i aria-hidden="true"></i><span id="liveText">Connecting</span></p>
</header>
<div class="wrap"><div class="notice" id="notice" role="alert" hidden>
  <svg id="pauseIcon" viewBox="0 0 256 256" fill="currentColor" aria-hidden="true"><path d="M251.77,73a8,8,0,0,0-8.21.39L208,97.05V72a16,16,0,0,0-16-16H113.06a8,8,0,0,0,0,16H192v87.63a8,8,0,0,0,16,0V159l35.56,23.71A8,8,0,0,0,248,184a8,8,0,0,0,8-8V80A8,8,0,0,0,251.77,73ZM240,161.05l-32-21.33V116.28L240,95ZM53.92,34.62A8,8,0,1,0,42.08,45.38L51.73,56H32A16,16,0,0,0,16,72V184a16,16,0,0,0,16,16H182.64l19.44,21.38a8,8,0,1,0,11.84-10.76ZM32,184V72H66.28L168.1,184Z"/></svg>
  <svg id="wifiIcon" viewBox="0 0 256 256" fill="currentColor" aria-hidden="true" hidden><path d="M213.92,210.62a8,8,0,1,1-11.84,10.76l-52-57.15a60,60,0,0,0-57.41,7.24,8,8,0,1,1-9.42-12.93A75.43,75.43,0,0,1,128,144c1.28,0,2.55,0,3.82.1L104.9,114.49A108,108,0,0,0,61,135.31,8,8,0,0,1,49.73,134,8,8,0,0,1,51,122.77a124.27,124.27,0,0,1,41.71-21.66L69.37,75.4a155.43,155.43,0,0,0-40.29,24A8,8,0,0,1,18.92,87,171.87,171.87,0,0,1,58,62.86L42.08,45.38A8,8,0,1,1,53.92,34.62ZM128,192a12,12,0,1,0,12,12A12,12,0,0,0,128,192ZM237.08,87A172.3,172.3,0,0,0,106,49.4a8,8,0,1,0,2,15.87A158.33,158.33,0,0,1,128,64a156.25,156.25,0,0,1,98.92,35.37A8,8,0,0,0,237.08,87ZM195,135.31a8,8,0,0,0,11.24-1.3,8,8,0,0,0-1.3-11.24,124.25,124.25,0,0,0-51.73-24.2A8,8,0,1,0,150,114.24,108.12,108.12,0,0,1,195,135.31Z"/></svg>
  <div><strong id="noticeTitle"></strong><span id="noticeBody"></span></div>
</div></div>
<main class="wrap">
  <div class="col">
    <section id="askBox" aria-label="Ask">
      <form id="ask">
        <label for="q" class="vh">Ask where something is</label>
        <input id="q" name="q" placeholder="Where are my keys?" autocomplete="off" enterkeyhint="go">
        <button type="submit" class="primary">Ask<svg viewBox="0 0 256 256" fill="currentColor" aria-hidden="true"><path d="M224.49,136.49l-72,72a12,12,0,0,1-17-17L187,140H40a12,12,0,0,1,0-24H187L135.51,64.48a12,12,0,0,1,17-17l72,72A12,12,0,0,1,224.49,136.49Z"/></svg></button>
        <button type="button" id="mic" class="secondary" aria-pressed="false">
          <svg id="micIcon" viewBox="0 0 256 256" fill="currentColor" aria-hidden="true"><path d="M128,176a48.05,48.05,0,0,0,48-48V64a48,48,0,0,0-96,0v64A48.05,48.05,0,0,0,128,176ZM96,64a32,32,0,0,1,64,0v64a32,32,0,0,1-64,0Zm40,143.6V240a8,8,0,0,1-16,0V207.6A80.11,80.11,0,0,1,48,128a8,8,0,0,1,16,0,64,64,0,0,0,128,0,8,8,0,0,1,16,0A80.11,80.11,0,0,1,136,207.6Z"/></svg>
          <svg id="stopIcon" viewBox="0 0 256 256" fill="currentColor" aria-hidden="true" hidden><path d="M216,56V200a16,16,0,0,1-16,16H56a16,16,0,0,1-16-16V56A16,16,0,0,1,56,40H200A16,16,0,0,1,216,56Z"/></svg>
          <span id="micLabel">Ask out loud</span>
        </button>
      </form>
      <div class="try" id="try" hidden><span>Try</span></div>
      <div class="card answer" id="answerCard" data-state="empty" aria-live="polite" style="margin-top:16px">
        <div class="answer-meta"><span class="badge" id="badge" hidden></span><span class="heard" id="heard">The answer is also spoken out loud.</span></div>
        <p class="answer-text" id="answer">Ask where something is, by typing or out loud.</p>
      </div>
    </section>

    <!-- [care] panel -->

    <section class="card guide" id="guide" hidden aria-label="Directions">
      <div class="dial">
        <svg viewBox="0 0 200 200" aria-hidden="true" id="dialFace"><circle class="face" cx="100" cy="100" r="98"/>
          <text x="100" y="36">Ahead</text><text x="100" y="165">Behind</text><text x="37" y="101">Left</text><text x="163" y="101">Right</text></svg>
        <div class="needle" id="needle"><svg viewBox="0 0 256 256" fill="currentColor" aria-hidden="true"><path d="M231.39,123.06A8,8,0,0,1,224,128H184v80a16,16,0,0,1-16,16H88a16,16,0,0,1-16-16V128H32a8,8,0,0,1-5.66-13.66l96-96a8,8,0,0,1,11.32,0l96,96A8,8,0,0,1,231.39,123.06Z"/></svg></div>
        <div class="seek"><svg viewBox="0 0 256 256" fill="currentColor" aria-hidden="true"><path d="M224,40V76a8,8,0,0,1-16,0V48H180a8,8,0,0,1,0-16h36A8,8,0,0,1,224,40Zm-8,132a8,8,0,0,0-8,8v28H180a8,8,0,0,0,0,16h36a8,8,0,0,0,8-8V180A8,8,0,0,0,216,172ZM76,208H48V180a8,8,0,0,0-16,0v36a8,8,0,0,0,8,8H76a8,8,0,0,0,0-16ZM40,84a8,8,0,0,0,8-8V48H76a8,8,0,0,0,0-16H40a8,8,0,0,0-8,8V76A8,8,0,0,0,40,84Zm136,92a8,8,0,0,1-6.41-3.19,52,52,0,0,0-83.2,0,8,8,0,1,1-12.8-9.62A67.94,67.94,0,0,1,101,141.51a40,40,0,1,1,53.94,0,67.94,67.94,0,0,1,27.43,21.68A8,8,0,0,1,176,176Zm-48-40a24,24,0,1,0-24-24A24,24,0,0,0,128,136Z"/></svg></div>
      </div>
      <div>
        <p class="guide-to" id="guideTo"></p>
        <p class="guide-words" id="guideWords">Looking for you</p>
        <p class="guide-math" id="guideMath"></p>
      </div>
    </section>

    <section id="boardBox" aria-labelledby="boardH">
      <div class="sec-head"><h2 id="boardH">Where things are</h2><p class="muted" id="asOf"></p></div>
      <div class="card">
        <ul class="places" id="places"><li><h3 class="muted">Loading</h3></li></ul>
        <p class="key"><span><b class="s-observed">Seen</b>: the camera saw it there.</span>
          <span><b class="s-inferred">Probably</b>: it went into something and hasn't come out.</span>
          <span><b class="s-unknown">Not sure</b>: it was there, but hasn't been seen since.</span></p>
      </div>
    </section>
  </div>

  <div class="col">
    <section id="camBoxWrap" aria-label="Live camera">
      <figure class="cam" id="camBox" data-s="wait">
        <img id="cam" alt="Live camera view of the room">
        <div class="cam-empty">
          <svg viewBox="0 0 256 256" fill="currentColor" aria-hidden="true"><path d="M251.77,73a8,8,0,0,0-8.21.39L208,97.05V72a16,16,0,0,0-16-16H113.06a8,8,0,0,0,0,16H192v87.63a8,8,0,0,0,16,0V159l35.56,23.71A8,8,0,0,0,248,184a8,8,0,0,0,8-8V80A8,8,0,0,0,251.77,73ZM240,161.05l-32-21.33V116.28L240,95ZM53.92,34.62A8,8,0,1,0,42.08,45.38L51.73,56H32A16,16,0,0,0,16,72V184a16,16,0,0,0,16,16H182.64l19.44,21.38a8,8,0,1,0,11.84-10.76ZM32,184V72H66.28L168.1,184Z"/></svg>
          <strong id="camTitle">No picture yet</strong>
          <p id="camBody">The picture appears here once the laptop's camera sends its first frame.</p>
        </div>
        <span class="cam-flag" id="camFlag" hidden>Paused</span>
      </figure>
      <p class="cam-cap">The live camera. Boxes mark the things it knows.</p>
    </section>

    <section id="logBox" aria-labelledby="logH">
      <div class="sec-head"><h2 id="logH">What moved</h2><p class="muted">Newest at the bottom</p></div>
      <div class="card"><ol class="log" id="log"><li class="empty">Nothing has moved yet.</li></ol></div>
    </section>

    <!-- [week] panel -->

  </div>
</main>
<script>
  const $ = id => document.getElementById(id);
  const t = s => new Date(s * 1000).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  const ago = (a, now) => a < 60 ? "just now" : a < 3600 ? Math.floor(a / 60) + " min ago" : "at " + t(now - a);
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const cap = s => s ? s[0].toUpperCase() + s.slice(1) : "";
  // An answer is spoken whole: directions that start meanwhile wait for it (they used to cut it off).
  let answering = 0;
  const say = (text, isAnswer) => {
    speechSynthesis.cancel();
    if (!text) return;
    const u = new SpeechSynthesisUtterance(text);
    if (isAnswer) {
      const mine = answering = Date.now() + 1500 + 90 * text.length;  // in case onend never comes
      u.onend = u.onerror = () => { if (answering === mine) answering = 0; };
    }
    speechSynthesis.speak(u);
  };

  // The answer card: empty, busy, listening, or an answer with how sure it is.
  const LABEL = { observed: "Seen there", inferred: "Probably", unknown: "Not sure", none: "Not found" };
  const card = (state, text, meta, status) => {
    $("answerCard").dataset.state = state;
    $("answerCard").setAttribute("aria-busy", state === "busy" || state === "listening");
    $("answerCard").classList.toggle("list", state === "answer" && text.includes("\\n"));
    $("badge").hidden = !LABEL[status];
    $("badge").className = "badge s-" + status;
    $("badge").textContent = LABEL[status] || "";
    $("heard").textContent = meta || "";
    $("answer").textContent = text;
    understood(null);
  };
  const show = (r, meta) => {
    card("answer", r.answer || "Ask where something is.", meta, r.status);
    understood(r.understood, r.events);
    say(r.answer, true);
  };

  async function ask(q) {
    q = q.trim();
    if (!q) return $("q").focus();
    card("busy", "Checking the memory", "You asked: “" + q + "”");
    try {
      show(await (await fetch("/ask?q=" + encodeURIComponent(q))).json(), "You asked: “" + q + "”");
    } catch (e) {
      card("answer", "Couldn't reach the laptop. Try again in a moment.", "", "info");
    }
  }
  $("ask").onsubmit = e => { e?.preventDefault(); ask($("q").value); };

  // Ask out loud: the laptop's mic listens until you stop talking (Whisper on the laptop,
  // recall/voice.py), so this works from a phone too; press again to stop sooner. Moss on the
  // server works out which item vague wording means.
  let listening = false;        // this page's own question is being heard
  let laptopListening = false;  // the laptop is hearing a question from anywhere (/guide)
  const setListening = on => {
    $("mic").classList.toggle("on", on);
    $("mic").setAttribute("aria-pressed", on);
    $("micIcon").toggleAttribute("hidden", on); $("stopIcon").toggleAttribute("hidden", !on);  // svg has no .hidden
    $("micLabel").textContent = on ? "Stop" : "Ask out loud";
    $("q").placeholder = on ? "Listening" : "Where are my keys?";
  };
  $("mic").onclick = async () => {
    if (listening || laptopListening) { try { await fetch("/listen?stop=1"); } catch (e) {} return; }
    listening = true;
    speechSynthesis.cancel();
    setListening(true); $("q").value = "";
    card("listening", "Listening. Ask your question toward the laptop.", "Press Stop when you're done.");
    try {
      const r = await (await fetch("/listen")).json();
      if (r.busy) { card("answer", "The laptop is already listening.", "", "info"); return; }
      $("q").value = r.heard || "";
      show(r, r.heard ? "You said: “" + r.heard + "”" : "");
    } catch (e) {
      card("answer", "Couldn't reach the laptop.", "", "info");
    } finally {
      listening = false;
      setListening(false);
    }
  };

  // Suggested questions, from the things the camera knows about.
  let tried = false;
  const suggest = items => {
    if (tried || !items.length) return;
    tried = true;
    const known = items.filter(i => i.status !== "never").concat(items.filter(i => i.status === "never"));
    const qs = known.slice(0, 2).map(i => `Where ${/s$/.test(i.name) ? "are" : "is"} my ${i.name}?`).concat(["What changed today?"]);
    for (const q of qs) {
      const b = document.createElement("button");
      b.type = "button"; b.textContent = q;
      b.onclick = () => { $("q").value = q; ask(q); };
      $("try").append(b);
    }
    $("try").hidden = false;
  };

  // Board: one row per place, each thing with how sure the memory is, in words.
  const row = (i, now, loose) => {
    const when = i.age == null ? "" : ago(i.age, now);
    let label = "", detail = "";
    if (i.status === "observed") { label = "Seen"; detail = when; }
    else if (i.status === "inferred") { label = i.container ? "Probably in the " + i.container : "Probably inside something"; detail = "Last seen " + when; }
    else if (i.status === "unknown") { label = "Not sure"; detail = (loose ? "Last seen " : "Last seen here ") + when; }
    return `<li><span class="thing">${esc(cap(i.name))}</span>` + (label ? `<span class="st"><b class="s-${i.status}">${esc(label)}</b>` +
      (detail ? `<small>${esc(detail)}</small>` : "") + `</span>` : "") + `</li>`;
  };
  const group = (name, items, now, loose) => `<li><h3>${esc(cap(name))}</h3><ul class="things">` +
    (items.length ? items.map(i => row(i, now, loose)).join("") : `<li class="none">Nothing here</li>`) + `</ul></li>`;

  let offline = false, frozen = false, lastLog = null, lastOk = null;
  const notice = () => {
    $("notice").hidden = !offline && !frozen;
    $("wifiIcon").toggleAttribute("hidden", !offline); $("pauseIcon").toggleAttribute("hidden", offline);
    if (offline && lastOk) $("asOf").textContent = "Last updated at " + t(lastOk);
    if (offline) {
      $("noticeTitle").textContent = "Can't reach the laptop";
      $("noticeBody").textContent = "Check that it's on and on the same wifi. This page keeps trying and updates by itself.";
    }
    const live = $("live");
    live.dataset.s = offline ? "off" : frozen ? "paused" : "live";
    $("liveText").textContent = offline ? "Not connected" : frozen ? "Camera paused" : "Live";
    $("camFlag").hidden = !frozen || offline;
  };

  async function tick() {
    try {
      const r = await fetch("/state");
      if (!r.ok) throw new Error(r.status);
      const s = await r.json();
      offline = false; frozen = !!s.frozen_text; lastOk = s.now;
      if (frozen) {
        $("noticeTitle").textContent = s.frozen_text;
        $("noticeBody").textContent = "Things may have moved since then. Where things are below is from before the pause.";
        $("camFlag").textContent = "Paused at " + t(s.frozen_at);
      }
      $("asOf").textContent = frozen ? "As of " + t(s.frozen_at) : "";
      const shown = s.items.filter(i => i.status !== "never");
      let html = s.places.map(p => group(p, shown.filter(i => i.place === p), s.now)).join("");
      const loose = shown.filter(i => i.place === null);
      if (loose.length) html += group("Not in a named place", loose, s.now, true);
      const never = s.items.filter(i => i.status === "never");
      if (never.length) html += group("Not seen yet", never, s.now);
      if (!html) html = `<li><span class="muted">No places or things set up yet.</span></li>`;
      $("places").innerHTML = html;
      suggest(s.items);

      const box = $("log"), key = JSON.stringify(s.log);
      if (key !== lastLog) {
        lastLog = key;
        const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
        box.innerHTML = s.log.length ? s.log.map(e => `<li><time>${t(e.t)}</time><span>${esc(e.text)}</span></li>`).join("")
          : `<li class="empty">Nothing has moved yet.</li>`;
        if (atBottom) box.scrollTop = box.scrollHeight;
      }
    } catch (e) {
      offline = true;
    }
    notice();
    setTimeout(tick, 1000);
  }

  // Directions: the arrow turns the short way round (angle kept cumulative).
  let lastWords = "", angle = null;
  const point = dir => {
    angle = angle === null ? dir : angle + ((dir - angle) % 360 + 540) % 360 - 180;
    $("needle").style.transform = `rotate(${angle}deg)`;
  };
  async function guideTick() {
    try {
      const g = await (await fetch("/guide")).json();
      // The band's button (or the terminal) can start the laptop listening too: the mic button
      // shows it, and pressing it then stops that question.
      if (!listening && g.listening !== laptopListening) setListening(g.listening);
      laptopListening = g.listening;
      $("guide").hidden = !g.item;
      document.body.classList.toggle("guiding", !!g.item);
      if (g.item) {
        const lost = g.dir === null;
        $("guideTo").innerHTML = g.target && g.target !== g.item
          ? `Guiding you to <b>the ${esc(g.target)}</b>. Your ${esc(g.item)} ${/s$/.test(g.item) ? "are" : "is"} in it.`
          : `Guiding you to <b>your ${esc(g.item)}</b>`;
        $("guide").classList.toggle("lost", lost);
        $("guideWords").textContent = g.words ?? "Step in front of the camera";
        $("guideMath").textContent = lost ? "I can't see you right now."
          : (g.dir === 0 ? "Straight ahead" : `Turn ${Math.abs(g.dir)}° ${g.dir > 0 ? "right" : "left"}`) + (g.metres ? ` · about ${g.metres} m away` : "");
        if (!lost) point(g.dir);
        if (g.words && g.words !== lastWords && Date.now() > answering) { lastWords = g.words; say(g.words); }
      }
    } catch (e) {}
    setTimeout(guideTick, 200);
  }

  // Dial ticks, every 15 degrees; the long one at the top is straight ahead.
  $("dialFace").insertAdjacentHTML("beforeend", Array.from({ length: 24 }, (_, k) => {
    const a = k * Math.PI / 12, r1 = k % 6 ? 86 : 79, xy = r => [100 + Math.sin(a) * r, 100 - Math.cos(a) * r].map(v => v.toFixed(1));
    const [x1, y1] = xy(r1), [x2, y2] = xy(92);
    return `<line class="tick${k % 6 ? "" : " major"}${k ? "" : " ahead"}" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"/>`;
  }).join(""));

  // Camera: the next picture 150 ms after one arrives; a second after a miss.
  let camSeen = false;
  function camTick() { $("cam").src = "/cam.jpg?t=" + Date.now(); }
  $("cam").onload = () => { camSeen = true; $("camBox").dataset.s = "live"; setTimeout(camTick, 150); };
  $("cam").onerror = () => {
    $("camBox").dataset.s = camSeen ? "lost" : "wait";
    $("camTitle").textContent = frozen ? "Camera paused" : camSeen ? "The picture stopped" : "No picture yet";
    $("camBody").textContent = frozen ? "The picture comes back when the camera does."
      : camSeen ? "Trying again every second." : "The picture appears here once the laptop's camera sends its first frame.";
    setTimeout(camTick, 1000);
  };
  // [care] script

  // [week] script

  // Moss semantic search: when Moss (not the exact words) worked out which thing was meant, one
  // quiet line under the answer says how, so a care partner can tell if it heard wrong.
  $("answerCard").insertAdjacentHTML("beforeend", '<p class="understood" id="understood" hidden></p>');
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
