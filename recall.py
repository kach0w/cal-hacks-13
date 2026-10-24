"""Recall, simplest version.

One camera per room. YOLO watches for a short list of items and remembers which room
each one was last seen in. A web page shows the board, a log of moves (newest at the
bottom), and answers "where is my phone?" out loud.

    .venv/bin/python recall.py          # camera 0 = the living room
    then open the printed link on any device on the same wifi

Press q in a camera window to quit.
"""

import argparse
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

import cv2
from ultralytics import YOLO

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
# Extra words people might say for an item.
SYNONYMS = {"cellphone": "phone", "mobile": "phone", "iphone": "phone", "purse": "bag",
            "backpack": "bag", "handbag": "bag", "mug": "cup", "clicker": "remote",
            "key": "keys", "specs": "glasses", "spectacles": "glasses", "sunglasses": "glasses"}
CONF = 0.20  # YOLO-World scores run lower than plain YOLO; 0.20 tested live
MOVE_CONFIRM = 2  # sightings in a new room before we believe the item moved
PORT = 8000

PROMPTS = [p for phrases in ITEMS.values() for p in phrases]
PROMPT_TO_ITEM = [item for item, phrases in ITEMS.items() for _ in phrases]

state = {}    # item -> {"room": str, "t": seconds}
pending = {}  # item -> (room, count) while a move is unconfirmed
log = []      # {"item", "from", "to", "t"}, oldest first
frames = {}   # room -> latest camera picture with boxes, as JPEG
lock = threading.Lock()


def report(room, item, now):
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
        state[item] = {"room": room, "t": now}


def ago(seconds):
    if seconds < 10:
        return "just now"
    if seconds < 90:
        return f"{int(seconds)} seconds ago"
    if seconds < 3600:
        return f"{round(seconds / 60)} minutes ago"
    return "at " + time.strftime("%-I:%M %p", time.localtime(time.time() - seconds))


def answer(question):
    words = question.lower().replace("?", " ").replace(",", " ").split()
    item = next((w for w in words if w in ITEMS), None) or \
        next((SYNONYMS[w] for w in words if w in SYNONYMS), None)
    if item is None:
        return f"I'm only watching for: {', '.join(ITEMS)}."
    with lock:
        cur = state.get(item)
    if cur is None:
        return f"I haven't seen your {item} yet."
    return f"Your {item} is in the {cur['room']}. I saw it there {ago(time.time() - cur['t'])}."


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
</script>
</body></html>
"""


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
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Recall: open http://{lan_ip()}:{PORT}  (rooms: {', '.join(ROOMS)})")

    while True:
        for room, cap in cams:
            ok, frame = cap.read()
            if not ok:
                continue
            result = model(frame, conf=CONF, verbose=False)[0]
            now = time.time()
            for item in {PROMPT_TO_ITEM[int(c)] for c in result.boxes.cls.tolist()}:
                report(room, item, now)
            shown = result.plot()
            small = cv2.resize(shown, (640, int(640 * shown.shape[0] / shown.shape[1])))
            with lock:
                frames[room] = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
            cv2.imshow(room, shown)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    for _, cap in cams:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
