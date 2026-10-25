"""The web page on top of the live loop: board, ask box with spoken answers, and left/right
guidance to the item asked about.

    python -m recall.live --web            # then open the printed link on any device on the wifi

Runs inside live (it needs the newest frame and the memory), on its own threads:
  - an HTTP server (stdlib, no extra dependencies)
  - a guidance thread: while someone is being guided, finds them with a pose model and points
    them at the item, or at its container when the memory says it went into one.

Routes: /  /state  /ask?q=  /guide  /cam.jpg  /photo/<path>
"""
import json
import logging
import math
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

import cv2

from .ask import handle, match_item
from .guide import SHOULDER_M, Guide, words
from .overlay import live_view
from .query import Query, clock

log = logging.getLogger("recall")

PORT = 8000
GUIDE_FOR_S = 60       # stop guiding this long after a question
GUIDE_EVERY_S = 0.15   # how often the guidance thread looks
CAM_WIDTH = 960        # the page's camera picture


class Web:
    def __init__(self, cfg, items, memory, pipeline, places, room=None):
        self.cfg, self.items, self.memory, self.pipeline = cfg, items, memory, pipeline
        self.places, self.room = places, room
        self.query = Query(cfg, items, memory.store)
        self.guide = None  # the pose model loads on the first question, not at start-up
        self._lock = threading.Lock()
        self.guiding = {"item": None, "since": 0.0, "dir": None, "dist": 100, "metres": None,
                        "target": None, "t": 0.0}
        self.cam_jpg = None

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
            return {"answer": str(e), "status": "info", "photo": None}
        oid = match_item(self.items, text)
        if oid and a.status in ("observed", "inferred", "unknown"):
            with self._lock:
                self.guiding.update(item=oid, since=time.time(), dir=None, target=None)
        return {"answer": a.text, "status": a.status, "photo": a.photo_path}

    def guidance(self) -> dict:
        with self._lock:
            g = dict(self.guiding)
        live = g["item"] and g["dir"] is not None and time.time() - g["t"] < 1
        name = self.items[g["item"]].name if g["item"] else None
        return {"item": name, "target": g["target"],
                "dir": round(g["dir"]) if live else None,
                "words": words(g["dir"], g["dist"]) if live else None,
                "metres": round(g["metres"], 1) if live and g["metres"] is not None else None}

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
        shown = live_view(r.frame.image, r, self.places, self.memory.state, self.items, room=self.room)
        with self._lock:
            oid = self.guiding["item"]
            if oid and time.time() - self.guiding["since"] > GUIDE_FOR_S:
                self.guiding["item"] = oid = None
        if oid:
            self._point(oid, r.frame.image, shown)
        h, w = shown.shape[:2]
        small = cv2.resize(shown, (CAM_WIDTH, int(CAM_WIDTH * h / w)))
        jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
        with self._lock:
            self.cam_jpg = jpg

    def _point(self, oid, image, shown):
        found = self._target(oid)
        if found is None:
            return
        box, target_name = found
        if self.guide is None:
            self.guide = Guide()
        if not self.guide.update(image):
            cv2.putText(shown, "face the camera for a second", (16, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 200, 255), 2)
            return
        target = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        h, w = image.shape[:2]
        result = self.guide.toward(target, math.hypot(w, h))
        if result is None:
            return
        deg, dist = result
        metres = self.guide.metres(target)
        with self._lock:
            self.guiding.update(dir=deg, dist=dist, metres=metres, target=target_name, t=time.time())

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
                elif url.path == "/guide":
                    self._json(web.guidance())
                elif url.path == "/cam.jpg":
                    with web._lock:
                        jpg = web.cam_jpg
                    self._send(jpg, "image/jpeg") if jpg else self.send_error(404)
                elif url.path.startswith("/photo/"):
                    p = web.memory.store.photo_file(unquote(url.path[len("/photo/"):]))
                    self._send(p.read_bytes(), "image/jpeg") if p else self.send_error(404)
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
  #answer { font-size: 1.25rem; font-weight: 600; min-height: 1.5em; white-space: pre-line; margin: 12px 0; }
  #photo { max-width: 100%; border-radius: 10px; display: block; }
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
</style></head>
<body><main>
  <h1>Recall</h1>
  <div class="top">
    <section>
      <form id="ask"><input id="q" placeholder="Where are my keys?" autocomplete="off"><button>Ask</button></form>
      <p id="answer"></p>
      <img id="photo" hidden alt="Where it was last seen">
      <div id="guide" hidden>
        <svg id="arrow" viewBox="0 0 100 100"><polygon points="50,6 66,52 54,46 54,94 46,94 46,46 34,52" fill="var(--accent)"/></svg>
        <div><div id="guideWords">Looking for you…</div><div id="guideSub"></div></div>
      </div>
    </section>
    <section><img id="cam" alt="Live camera"><p id="frozen" class="frozen" hidden></p></section>
  </div>
  <section><h2>Where things are</h2><div class="places" id="places"></div></section>
  <section><h2>What moved (newest at the bottom)</h2><div class="log" id="log"><div>Nothing yet.</div></div></section>
</main>
<script>
  const $ = id => document.getElementById(id);
  const t = s => new Date(s * 1000).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  const ago = a => a < 10 ? "now" : a < 3600 ? Math.round(a / 60) + " min ago" : "earlier";
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const say = text => { speechSynthesis.cancel(); speechSynthesis.speak(new SpeechSynthesisUtterance(text)); };

  $("ask").onsubmit = async e => {
    e.preventDefault();
    const r = await (await fetch("/ask?q=" + encodeURIComponent($("q").value))).json();
    $("answer").textContent = r.answer;
    $("photo").hidden = !r.photo;
    if (r.photo) $("photo").src = "/photo/" + encodeURIComponent(r.photo);
    say(r.answer);
  };

  async function tick() {
    try {
      const s = await (await fetch("/state")).json();
      $("frozen").hidden = !s.frozen_text; $("frozen").textContent = s.frozen_text || "";
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
      $("guide").hidden = !g.item;
      if (g.item) {
        const to = g.target && g.target !== g.item ? `the ${g.target} (your ${g.item} is in it)` : `your ${g.item}`;
        const math = g.dir === null ? "" : ` · ${Math.abs(g.dir)}° ${g.dir > 0 ? "right" : "left"}` + (g.metres ? ` · about ${g.metres} m` : "");
        $("guideSub").textContent = "Guiding you to " + to + math;
        $("guide").classList.toggle("lost", g.dir === null);
        $("guideWords").textContent = g.words ?? "Step in front of the camera";
        if (g.dir !== null) $("arrow").style.transform = `rotate(${g.dir}deg)`;
        if (g.words && g.words !== lastWords) { lastWords = g.words; say(g.words); }
      }
    } catch (e) {}
    setTimeout(guideTick, 200);
  }

  function camTick() { $("cam").src = "/cam.jpg?t=" + Date.now(); }
  $("cam").onload = () => setTimeout(camTick, 150);
  $("cam").onerror = () => setTimeout(camTick, 1000);
  tick(); guideTick(); camTick();
</script>
</body></html>
"""
