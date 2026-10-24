"""Phone fallback for the necklace: a compass page served from the laptop.

compass_demo.py calls start() once and update() every frame. Any phone on the
same hotspot opens http://<laptop-ip>:8080 and sees the same arrow as the dial.
"""

import json
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

PORT = 8080

_state = {"dir": None, "dist": 100, "t": 0.0, "name": None, "note": None}
_board = {"status": None, "rooms": [], "seeking": None}  # set by attach_board()
_snaps = {}  # room -> latest jpeg bytes
BOARD_FILE = os.path.join(os.path.dirname(__file__), "board.html")
_lock = threading.Lock()

PAGE = """<!doctype html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Recall Compass</title>
<style>
  :root { --bg: #101218; --fg: #f2f2f5; --muted: #8a8d9c; --ring: #2a2d3a; --needle: #ffb020; }
  html, body { height: 100%; margin: 0; background: var(--bg); color: var(--fg);
    font: 600 20px -apple-system, system-ui, sans-serif; }
  body { display: grid; place-items: center; text-align: center;
    padding: env(safe-area-inset-top) 16px env(safe-area-inset-bottom); }
  .dial { width: min(80vw, 60vh); aspect-ratio: 1; border-radius: 50%;
    border: 6px solid var(--ring); position: relative; }
  .dial .ahead { position: absolute; top: 10px; left: 0; right: 0; font-size: 14px; color: var(--muted); }
  #needle { position: absolute; inset: 0; transition: transform 120ms linear; }
  #needle svg { width: 100%; height: 100%; }
  #label { font-size: 34px; margin-top: 24px; }
  #sub { color: var(--muted); margin-top: 6px; font-size: 16px; }
  .lost #needle { opacity: 0.15; }
  .found .dial { border-color: #3ccf7e; }
</style></head>
<body>
  <div>
    <div class="dial"><span class="ahead">AHEAD</span>
      <div id="needle"><svg viewBox="0 0 100 100">
        <polygon points="50,8 61,52 50,46 39,52" fill="var(--needle)"/>
        <circle cx="50" cy="50" r="5" fill="var(--fg)"/>
      </svg></div>
    </div>
    <div id="label">Connecting…</div>
    <div id="sub"></div>
  </div>
<script>
  const needle = document.getElementById("needle"), label = document.getElementById("label"),
        sub = document.getElementById("sub");
  function words(d) {
    const a = Math.abs(d);
    if (a < 20) return "Straight ahead";
    if (a > 135) return "Behind you";
    return d > 0 ? "To your right" : "To your left";
  }
  async function tick() {
    try {
      const s = await (await fetch("/state", { cache: "no-store" })).json();
      const lost = s.dir === null || s.age > 1;
      document.body.className = lost ? "lost" : (s.dist <= 8 ? "found" : "");
      if (lost) {
        const what = s.name ? s.name.charAt(0).toUpperCase() + s.name.slice(1) + ": " : "";
        label.textContent = s.note ? what + s.note : "Looking…"; sub.textContent = "";
      }
      else {
        needle.style.transform = `rotate(${s.dir}deg)`;
        const where = s.dist <= 8 ? "Right here" : words(s.dir);
        label.textContent = s.name
          ? s.name.charAt(0).toUpperCase() + s.name.slice(1) + ": " + where.toLowerCase()
          : where;
        sub.textContent = s.dist <= 20 ? "Very close" : s.dist <= 50 ? "Getting close" : "Keep going";
      }
    } catch (e) { document.body.className = "lost"; label.textContent = "No connection"; }
    setTimeout(tick, 100);
  }
  tick();
</script>
</body></html>
"""


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/state":
            with _lock:
                body = json.dumps({"dir": _state["dir"], "dist": _state["dist"], "name": _state["name"],
                                   "note": _state["note"],
                                   "age": time.time() - _state["t"]}).encode()
            ctype = "application/json"
        elif self.path in ("/", "/index.html"):
            body, ctype = PAGE.encode(), "text/html; charset=utf-8"
        elif self.path == "/board":
            with open(BOARD_FILE, "rb") as f:
                body, ctype = f.read(), "text/html; charset=utf-8"
        elif self.path == "/items" and _board["status"] is not None:
            snap = _board["status"].snapshot()
            snap["rooms"] = _board["rooms"]
            with _lock:
                snap["seeking"] = _state["name"]
            body, ctype = json.dumps(snap).encode(), "application/json"
        elif self.path.startswith("/snap/"):
            room = self.path[len("/snap/"):].split("?")[0].removesuffix(".jpg")
            with _lock:
                body = _snaps.get(unquote(room))
            if body is None:
                self.send_error(404)
                return
            ctype = "image/jpeg"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass  # keep the terminal quiet


def update(direction, dist, name=None, note=None):
    """direction in degrees (clockwise, 0 = ahead) or None when lost; name = what we're guiding to;
    note = what to say when we can't point, e.g. "in the bedroom"."""
    with _lock:
        _state["name"] = name
        _state["note"] = note
        _state["dir"] = None if direction is None else round(direction)
        _state["dist"] = dist
        if direction is not None:
            _state["t"] = time.time()


def attach_board(status, rooms):
    """Serve the Clue-style board at /board from a RoomStatus."""
    _board["status"], _board["rooms"] = status, list(rooms)


def set_snapshot(room, jpeg_bytes):
    with _lock:
        _snaps[room] = jpeg_bytes


def _lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # no packet is sent; just picks the outgoing interface
        return s.getsockname()[0]
    except OSError:
        return "localhost"
    finally:
        s.close()


def start():
    server = ThreadingHTTPServer(("0.0.0.0", PORT), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://{_lan_ip()}:{PORT}"
    print(f"phone compass: {url}   board: {url}/board   (same wifi)")
    return url
