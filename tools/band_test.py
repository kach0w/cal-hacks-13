"""Send fake "dir,dist" packets to the band so the servo can be tested without the camera.

    python tools/band_test.py sixsevendemons sweep        # Bluetooth: arrow swings, distance closes in
    python tools/band_test.py sixsevendemons manual       # type "dir,dist" lines yourself
    python tools/band_test.py sixsevendemons fixed -30 50 # hold one value
    python tools/band_test.py sixsevendemons found        # walk in to 0 and trigger the wiggle
    python tools/band_test.py sixsevendemons stop         # send nothing; band should centre in 1s
    python tools/band_test.py raspberrypi.local sweep     # wifi: a host name or IP (the Pi band)
"""

import logging
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from recall.band import connect  # noqa: E402

HZ = 15


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    target, mode = sys.argv[1], sys.argv[2]
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    link = connect(target)
    print(f"looking for {target}, mode {mode}, ctrl-c to stop")
    try:
        while not link.connected:
            time.sleep(0.1)
        t0 = time.time()
        if mode == "manual":
            for line in sys.stdin:
                direction, _, dist = line.strip().partition(",")
                link.send(int(direction), int(dist))
        elif mode == "fixed":
            direction, dist = int(sys.argv[3]), int(sys.argv[4])
            while True:
                link.send(direction, dist)
                time.sleep(1 / HZ)
        elif mode == "sweep":
            while True:
                t = time.time() - t0
                direction = 80 * math.sin(t * 0.8)
                dist = 50 + 50 * math.cos(t * 0.2)
                link.send(direction, dist)
                time.sleep(1 / HZ)
        elif mode == "found":
            for dist in range(100, -1, -2):
                link.send(10, dist)
                time.sleep(1 / HZ)
            while True:
                link.send(0, 0)
                time.sleep(1 / HZ)
        elif mode == "stop":
            print("sending nothing; the arrow should centre within 1s")
            while True:
                time.sleep(1)
        else:
            print(__doc__)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
