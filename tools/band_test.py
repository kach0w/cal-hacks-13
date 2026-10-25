"""Send fake "dir,dist" packets to the band so the servo can be tested without the camera.

    python tools/band_test.py recall-band.local sweep     # arrow swings, distance closes in
    python tools/band_test.py 172.20.10.5 manual          # type "dir,dist" lines yourself
    python tools/band_test.py 172.20.10.5 fixed -30 50    # hold one value
    python tools/band_test.py 172.20.10.5 found           # walk in to 0 and trigger the wiggle
    python tools/band_test.py 172.20.10.5 stop            # send nothing; band should centre in 1s
"""

import math
import socket
import sys
import time

PORT = 4210
HZ = 15


def send(sock, addr, direction, dist):
    sock.sendto(f"{int(direction)},{int(dist)}".encode(), addr)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    host, mode = sys.argv[1], sys.argv[2]
    addr = (socket.gethostbyname(host), PORT)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"sending to {addr[0]}:{PORT}, mode {mode}, ctrl-c to stop")

    t0 = time.time()
    try:
        if mode == "manual":
            for line in sys.stdin:
                sock.sendto(line.strip().encode(), addr)
        elif mode == "fixed":
            direction, dist = int(sys.argv[3]), int(sys.argv[4])
            while True:
                send(sock, addr, direction, dist)
                time.sleep(1 / HZ)
        elif mode == "sweep":
            while True:
                t = time.time() - t0
                direction = 80 * math.sin(t * 0.8)
                dist = 50 + 50 * math.cos(t * 0.2)
                send(sock, addr, direction, dist)
                time.sleep(1 / HZ)
        elif mode == "found":
            for dist in range(100, -1, -2):
                send(sock, addr, 10, dist)
                time.sleep(1 / HZ)
            while True:
                send(sock, addr, 0, 0)
                time.sleep(1 / HZ)
        elif mode == "stop":
            print("sending nothing; the arrow should centre within 1s")
        else:
            print(__doc__)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
