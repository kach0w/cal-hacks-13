"""Recall, simplest version (app.py; the full pipeline is the recall/ package).

One camera per room. YOLO watches for a short list of items and remembers which room
each one was last seen in. A web page shows the board, a log of moves (newest at the
bottom), and answers "where is my phone?" out loud. If the item is in the first room,
the page then points the person to it: left, right, or straight ahead.

    .venv/bin/python app.py          # camera 0 = the living room
    then open the printed link on any device on the same wifi

Stop with Ctrl+C.
"""

import argparse
import json
import math
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

import cv2
from ultralytics import YOLO

from recall.guide import SHOULDER_M, Guide, words

# Item name -> the phrases YOLO-World looks for. Several phrasings per item match
# better than one. Edit this list to change what Recall watches for.
ITEMS = {
    "keys": ["keys", "key ring", "bunch of keys"],
    "wallet": ["wallet", "leather wallet"],
    "glasses": ["eyeglasses", "glasses", "sunglasses"],
    "phone": ["cell phone", "smartphone"],
    "remote": ["remote control", "tv remote"],
    "bag": ["handbag", "backpack", "tote bag"],
    "cup": ["cup", "coffee mug"],
    "book": ["book"],
    "bottle": ["water bottle", "bottle"],
}
# Furniture used to describe where an item is ("on the table, next to the TV").
# Found in the same YOLO-World pass; never tracked or logged themselves.
LANDMARKS = {
    "TV": ["television", "tv"],
    "couch": ["couch", "sofa"],
    "chair": ["chair", "armchair"],
    "table": ["table", "coffee table", "dining table"],
    "desk": ["desk"],
    "bed": ["bed"],
    "shelf": ["bookshelf", "shelf"],
    "lamp": ["lamp"],
    "sink": ["sink"],
    "fridge": ["refrigerator"],
    "plant": ["potted plant"],
}
SURFACES = {"couch", "chair", "table", "desk", "bed", "shelf"}  # things an item can be "on"
NEXT_TO = 0.15  # within this fraction of the frame width counts as "next to"

# Extra words people might say for an item.
SYNONYMS = {"cellphone": "phone", "mobile": "phone", "iphone": "phone", "purse": "bag",
            "backpack": "bag", "handbag": "bag", "mug": "cup", "clicker": "remote",
            "key": "keys", "specs": "glasses", "spectacles": "glasses", "sunglasses": "glasses"}
PLURAL = {"keys", "glasses"}
CONF = 0.20  # YOLO-World scores run lower than plain YOLO; 0.20 tested live
MOVE_CONFIRM = 2  # sightings in a new room before we believe the item moved
PORT = 8000
GUIDE_FOR_S = 60  # stop guiding this long after a question

NAMED = {**ITEMS, **LANDMARKS}
PROMPTS = [p for phrases in NAMED.values() for p in phrases]
PROMPT_TO_ITEM = [name for name, phrases in NAMED.items() for _ in phrases]

state = {}    # item -> {"room", "t", "box": [x1, y1, x2, y2], "where": "on the table, next to the TV"}
furniture = {}  # room -> {landmark: box}, last seen; furniture doesn't move
pending = {}  # item -> (room, count) while a move is unconfirmed
log = []      # {"item", "from", "to", "t"}, oldest first
frames = {}   # room -> latest camera picture with boxes, as JPEG
guiding = {"item": None, "since": 0.0, "dir": None, "dist": 100, "metres": None, "t": 0.0}
lock = threading.Lock()


def describe(room, box, frame_w):
    """Where a box is relative to the room's furniture, e.g. "on the table, next to the TV"."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    on, near, near_gap = None, None, NEXT_TO * frame_w
    for name, (x1, y1, x2, y2) in furniture.get(room, {}).items():
        if name in SURFACES and x1 <= cx <= x2 and y1 <= cy <= y2:
            on = name
            continue
        # Gap between the two boxes' edges (0 if they touch or overlap).
        gap = max(x1 - box[2], box[0] - x2, 0) + max(y1 - box[3], box[1] - y2, 0)
        if gap < near_gap:
            near, near_gap = name, gap
    parts = ([f"on the {on}"] if on else []) + ([f"next to the {near}"] if near else [])
    return ", ".join(parts)


def report(room, item, now, box, where=""):
    with lock:
        cur = state.get(item)
        if cur and cur["room"] != room:
            prev, count = pending.get(item, (room, 0))
            count = count + 1 if prev == room else 1
            pending[item] = (room, count)
            if count < MOVE_CONFIRM:
                return
        pending.pop(item, None)
        if not cur or cur["room"] != room:
            log.append({"item": item, "from": cur and cur["room"], "to": room, "t": now})
        state[item] = {"room": room, "t": now, "box": box, "where": where}


def ago(seconds):
    if seconds < 10:
        return "just now"
    if seconds < 90:
        return f"{int(seconds)} seconds ago"
    if seconds < 3600:
        return f"{round(seconds / 60)} minutes ago"
    return "at " + time.strftime("%-I:%M %p", time.localtime(time.time() - seconds))


def answer(question):
    said = question.lower().replace("?", " ").replace(",", " ").split()
    item = next((w for w in said if w in ITEMS), None) or \
        next((SYNONYMS[w] for w in said if w in SYNONYMS), None)
    if item is None:
        return f"I'm only watching for: {', '.join(ITEMS)}."
    with lock:
        cur = state.get(item)
        guiding.update(item=item, since=time.time(), dir=None)
    if cur is None:
        return f"I haven't seen your {item} yet."
    is_, it = ("are", "them") if item in PLURAL else ("is", "it")
    where = f", {cur['where']}" if cur.get("where") else ""
    return f"Your {item} {is_} in the {cur['room']}{where}. I saw {it} there {ago(time.time() - cur['t'])}."


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            self.send(PAGE.encode(), "text/html; charset=utf-8")
        elif url.path == "/state":
            now = time.time()
            with lock:
                body = {"rooms": ROOMS, "now": now, "log": log[-50:],
                        "items": [{"name": n, "room": state.get(n, {}).get("room"),
                                   "age": now - state[n]["t"] if n in state else None} for n in ITEMS]}
            self.send(json.dumps(body).encode(), "application/json")
        elif url.path.startswith("/cam/"):
            with lock:
                jpg = frames.get(unquote(url.path[len("/cam/"):]))
            if jpg is None:
                self.send_error(404)
            else:
                self.send(jpg, "image/jpeg")
        elif url.path == "/guide":
            with lock:
                g = dict(guiding)
            live = g["item"] and g["dir"] is not None and time.time() - g["t"] < 1
            body = {"item": g["item"], "dir": round(g["dir"]) if live else None,
                    "words": words(g["dir"], g["dist"]) if live else None,
                    "metres": round(g["metres"], 1) if live and g["metres"] is not None else None}
            self.send(json.dumps(body).encode(), "application/json")
        elif url.path == "/ask":
            q = parse_qs(url.query).get("q", [""])[0]
            self.send(json.dumps({"answer": answer(q)}).encode(), "application/json")
        else:
            self.send_error(404)

    def send(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def lan_ip():
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
  :root { --bg: #f4f5f7; --card: #fff; --fg: #1c1e24; --muted: #6b6f7b; --line: #dfe1e6; --accent: #2f6fd0; }
  @media (prefers-color-scheme: dark) { :root { --bg: #14161b; --card: #1e2128; --fg: #eceef3; --muted: #9a9eab; --line: #2e323b; --accent: #6fa3ff; } }
  body { margin: 0; background: var(--bg); color: var(--fg); font: 16px/1.45 -apple-system, system-ui, sans-serif; }
  main { max-width: 960px; margin: 0 auto; padding: 20px 16px 40px; display: grid; gap: 20px; }
  h1 { margin: 0; font-size: 1.8rem; } h2 { margin: 0 0 10px; font-size: 1rem; color: var(--muted); }
  form { display: flex; gap: 8px; }
  input { flex: 1; min-width: 0; font: inherit; padding: 12px 14px; border: 1px solid var(--line); border-radius: 10px; background: var(--card); color: var(--fg); }
  button { font: inherit; font-weight: 600; padding: 12px 18px; border: 0; border-radius: 10px; background: var(--accent); color: #fff; }
  #answer { font-size: 1.3rem; font-weight: 600; min-height: 1.5em; }
  #guide { display: flex; align-items: center; gap: 18px; background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 14px 18px; }
  #guide[hidden] { display: none; }
  #arrow { width: 72px; height: 72px; flex: none; transition: transform 150ms linear; }
  #guide.lost #arrow { opacity: 0.2; }
  #guideWords { font-size: 1.8rem; font-weight: 700; }
  #guideSub { color: var(--muted); }
  .rooms { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }
  .room { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 14px; }
  .room h3 { margin: 0 0 10px; text-transform: capitalize; }
  .room img { width: 100%; border-radius: 8px; margin-bottom: 10px; display: block; background: var(--bg); aspect-ratio: 16 / 9; object-fit: cover; }
  .chip { display: inline-block; margin: 0 6px 6px 0; padding: 6px 10px; border-radius: 999px; background: var(--bg); border: 1px solid var(--line); }
  .chip small { color: var(--muted); }
  .log { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 12px; max-height: 300px; overflow-y: auto; display: flex; flex-direction: column; gap: 6px; }
  .log div { padding: 6px 10px; border-radius: 10px; background: var(--bg); }
  .log div:last-child { outline: 2px solid var(--accent); }
  .log time { color: var(--muted); font-size: 0.85em; margin-right: 8px; }
</style></head>
<body><main>
  <h1>Recall</h1>
  <section>
    <form id="ask"><input id="q" placeholder="Where is my phone?" autocomplete="off"><button>Ask</button></form>
    <p id="answer"></p>
    <div id="guide" hidden>
      <svg id="arrow" viewBox="0 0 100 100"><polygon points="50,6 66,52 54,46 54,94 46,94 46,46 34,52" fill="var(--accent)"/></svg>
      <div><div id="guideWords">Looking for you…</div><div id="guideSub"></div></div>
    </div>
  </section>
  <section><h2>Where things are</h2><div class="rooms" id="rooms"></div></section>
  <section><h2>What moved (newest at the bottom)</h2><div class="log" id="log"><div>Nothing yet.</div></div></section>
</main>
<script>
  const t = s => new Date(s * 1000).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  const ago = a => a < 10 ? "now" : a < 3600 ? Math.round(a / 60) + " min ago" : "earlier";
  document.getElementById("ask").onsubmit = async e => {
    e.preventDefault();
    const q = document.getElementById("q").value;
    const { answer } = await (await fetch("/ask?q=" + encodeURIComponent(q))).json();
    document.getElementById("answer").textContent = answer;
    speechSynthesis.cancel(); speechSynthesis.speak(new SpeechSynthesisUtterance(answer));
  };
  async function tick() {
    try {
      const s = await (await fetch("/state")).json();
      const cols = [...s.rooms, null];
      document.getElementById("rooms").innerHTML = cols.map(room => {
        const here = s.items.filter(i => i.room === room);
        if (room === null && !here.length) return "";
        const chips = here.map(i => `<span class="chip">${i.name}${i.age === null ? "" : ` <small>${ago(i.age)}</small>`}</span>`).join("");
        const cam = room === null ? "" : `<img alt="${room} camera" src="/cam/${encodeURIComponent(room)}?t=${Date.now()}">`;
        return `<div class="room"><h3>${room ?? "Not seen yet"}</h3>${cam}${chips || "<small>nothing here</small>"}</div>`;
      }).join("");
      const box = document.getElementById("log");
      if (s.log.length) {
        const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
        box.innerHTML = s.log.map(c => `<div><time>${t(c.t)}</time>${c.from ? `${c.item} moved from the ${c.from} to the ${c.to}` : `${c.item} seen in the ${c.to}`}</div>`).join("");
        if (atBottom) box.scrollTop = box.scrollHeight;
      }
    } catch (e) {}
    setTimeout(tick, 1000);
  }
  tick();
  let lastWords = "";
  async function guideTick() {
    try {
      const g = await (await fetch("/guide")).json();
      const box = document.getElementById("guide");
      box.hidden = !g.item;
      if (g.item) {
        const math = g.dir === null ? "" : ` · ${Math.abs(g.dir)}° ${g.dir > 0 ? "right" : "left"}` + (g.metres ? ` · about ${g.metres} m` : "");
        document.getElementById("guideSub").textContent = "Guiding you to your " + g.item + math;
        box.classList.toggle("lost", g.dir === null);
        document.getElementById("guideWords").textContent = g.words ?? "Step in front of the camera";
        if (g.dir !== null) document.getElementById("arrow").style.transform = `rotate(${g.dir}deg)`;
        if (g.words && g.words !== lastWords) {
          lastWords = g.words;
          speechSynthesis.cancel(); speechSynthesis.speak(new SpeechSynthesisUtterance(g.words));
        }
      }
    } catch (e) {}
    setTimeout(guideTick, 200);
  }
  guideTick();
</script>
</body></html>
"""


def guide_step(guide, frame, shown, now):
    """Point the person at the asked-for item, if it's in this (the first) room."""
    with lock:
        item = guiding["item"]
        if item and now - guiding["since"] > GUIDE_FOR_S:
            guiding["item"] = item = None
        cur = state.get(item) if item else None
    if cur is None or cur["room"] != ROOMS[0] or not guide.update(frame):
        return
    b = cur["box"]
    target = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    h, w = frame.shape[:2]
    result = guide.toward(target, (w * w + h * h) ** 0.5)
    if result is None:
        cv2.putText(shown, "face the camera for a second", (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)
        return
    deg, dist = result
    metres = guide.metres(target)
    with lock:
        guiding.update(dir=deg, dist=dist, metres=metres, t=now)
    body = tuple(int(v) for v in guide.body)
    tip = tuple(int(v) for v in guide.body + guide.facing * 100)
    cv2.arrowedLine(shown, body, tip, (0, 255, 0), 3)
    cv2.line(shown, body, tuple(int(v) for v in target), (0, 200, 255), 2)
    cv2.putText(shown, f"{item}: {words(deg, dist)}", (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)
    # The math, on screen: which way they face, where the item is, the turn, the distance.
    face = guide.facing_degrees()
    bearing = math.degrees(math.atan2(target[0] - guide.body[0], target[1] - guide.body[1]))
    side = "right" if deg > 0 else "left"
    lines = [f"facing     {face:+5.0f} deg  (0 = toward camera)",
             f"item at    {bearing:+5.0f} deg",
             f"turn       {abs(deg):5.0f} deg {side}",
             f"distance  ~{metres:.1f} m  ({SHOULDER_M:.2f} m shoulders = {guide.full_px:.0f} px)" if metres else "distance   ?"]
    for i, line in enumerate(lines):
        cv2.putText(shown, line, (16, 80 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    mid = ((body[0] + int(target[0])) // 2, (body[1] + int(target[1])) // 2)
    if metres:
        cv2.putText(shown, f"{metres:.1f} m", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)


def main():
    global ROOMS
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", default="living room:0", help="one camera per room; default is camera 0 as the living room")
    args = ap.parse_args()

    cams = []
    for part in args.rooms.split(","):
        room, _, idx = part.partition(":")
        cap = cv2.VideoCapture(int(idx or 0))
        if not cap.isOpened():
            raise SystemExit(f"can't open camera {idx} for the {room}; check the number and camera permission")
        cams.append((room.strip(), cap))
    ROOMS = [room for room, _ in cams]

    model = YOLO("yolov8s-worldv2.pt")
    model.set_classes(PROMPTS)
    guide = Guide()  # guidance happens in the first room's camera
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Recall: open http://{lan_ip()}:{PORT}  (rooms: {', '.join(ROOMS)})")

    # Runs until Ctrl+C. The web page is the only screen.
    try:
        while True:
            for room, cap in cams:
                ok, frame = cap.read()
                if not ok:
                    continue
                result = model(frame, conf=CONF, verbose=False)[0]
                now = time.time()
                best = {}  # item -> (conf, box)
                for c, conf, box in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(),
                                        result.boxes.xyxy.tolist()):
                    item = PROMPT_TO_ITEM[int(c)]
                    if item not in best or conf > best[item][0]:
                        best[item] = (conf, box)
                with lock:
                    for name in LANDMARKS:
                        if name in best:
                            furniture.setdefault(room, {})[name] = best[name][1]
                for item, (_, box) in best.items():
                    if item in ITEMS:
                        report(room, item, now, box, describe(room, box, frame.shape[1]))
                with lock:
                    seeking = guiding["item"]
                if seeking and seeking in best:
                    # While guiding, show only the item we're looking for.
                    shown = frame.copy()
                    x1, y1, x2, y2 = map(int, best[seeking][1])
                    cv2.rectangle(shown, (x1, y1), (x2, y2), (0, 200, 255), 3)
                    cv2.putText(shown, seeking, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
                else:
                    shown = result.plot()
                if room == ROOMS[0]:
                    guide_step(guide, frame, shown, now)
                small = cv2.resize(shown, (640, int(640 * shown.shape[0] / shown.shape[1])))
                with lock:
                    frames[room] = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
    except KeyboardInterrupt:
        pass

    for _, cap in cams:
        cap.release()


if __name__ == "__main__":
    main()
