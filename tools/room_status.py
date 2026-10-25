"""Which room each listed item is in, and when it was last seen.

Every camera is a room. Detections from all rooms are reported here; an item's room
is the room it was most recently seen in. A move between rooms only counts once the
item has shown up in the new room twice in a row, so one bad detection can't move it.
"""

import threading
import time

MOVE_CONFIRM = 2  # detections in a new room before we believe the item moved
LOG_LIMIT = 50


class RoomStatus:
    def __init__(self, items):
        self.items = list(items)
        self._state = {}    # name -> {room, box, conf, last_seen}
        self._pending = {}  # name -> (room, count) for a not-yet-confirmed move
        self._log = []      # newest last: {item, from, to, t}
        self._lock = threading.Lock()

    def report(self, room, name, box, conf, now=None):
        now = now or time.time()
        with self._lock:
            cur = self._state.get(name)
            if cur is not None and cur["room"] != room:
                prev_room, count = self._pending.get(name, (room, 0))
                count = count + 1 if prev_room == room else 1
                self._pending[name] = (room, count)
                if count < MOVE_CONFIRM:
                    return
            self._pending.pop(name, None)
            if cur is None or cur["room"] != room:
                self._log.append({"item": name, "from": None if cur is None else cur["room"],
                                  "to": room, "t": now})
                del self._log[:-LOG_LIMIT]
            self._state[name] = {"room": room, "box": box, "conf": conf, "last_seen": now}

    def get(self, name):
        with self._lock:
            cur = self._state.get(name)
            return None if cur is None else dict(cur)

    def snapshot(self):
        """JSON-ready view for the board page."""
        now = time.time()
        with self._lock:
            items = []
            for name in self.items:
                cur = self._state.get(name)
                if cur is None:
                    items.append({"name": name, "room": None, "last_seen": None, "age": None})
                else:
                    items.append({"name": name, "room": cur["room"], "last_seen": cur["last_seen"],
                                  "age": now - cur["last_seen"]})
            log = list(reversed(self._log))
        return {"now": now, "items": items, "changes": log}
