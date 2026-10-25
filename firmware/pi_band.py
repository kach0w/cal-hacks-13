"""Recall necklace on a Raspberry Pi: the same job as firmware/band/band.ino.

Listens for UDP text packets "dir,dist" from the laptop and points a servo.
  dir  = degrees from straight ahead to the target, -180..180, + = to the person's right
  dist = 0..100, 0 = on top of the target

Wiring (servo -> Pi header):
  brown/black (GND)    -> pin 6  (GND)
  red (+)              -> pin 2  (5V)
  orange/yellow (sig)  -> pin 12 (GPIO18)

On the Pi:
  sudo apt install -y python3-gpiozero python3-lgpio     # usually already there
  python3 pi_band.py
Then on the laptop:  python -m recall.live --web --band <pi-name>.local
Test from the laptop: echo "-30,50" | nc -u -w0 <pi-name>.local 4210
"""

import socket
import time

from gpiozero import AngularServo

SERVO_PIN = 18
UDP_PORT = 4210
SERVO_REVERSED = True   # SG90 angle grows counterclockwise from above; dir is clockwise
FORWARD = 90            # servo angle where the arrow points straight out from the chest
REST = 180              # where the arrow sits when idle or lost
TIMEOUT_S = 1.0         # no packet this long -> rest (a frozen arrow is a lie)
SMOOTH = 0.35           # 0..1, higher = snappier, jitterier
DEADBAND_DEG = 3        # ignore moves smaller than this
TICK_S = 0.02           # 50 Hz updates
FOUND_DIST = 8
FOUND_HOLD_S = 1.5

# SG90 pulse range: 0.5 ms .. 2.5 ms, symmetric around 1.5 ms so left and right swing equally.
servo = AngularServo(SERVO_PIN, min_angle=0, max_angle=180,
                     min_pulse_width=0.0005, max_pulse_width=0.0025)


def dir_to_servo(d):
    if SERVO_REVERSED:
        d = -d
    return max(0, min(180, FORWARD + d))  # directions it can't reach pin to the nearest edge


def parse(packet):
    """'dir,dist' -> (dir, dist), or None for garbage so a bad packet never moves the arrow."""
    try:
        d, k = (int(x) for x in packet.decode().strip().split(","))
    except ValueError:
        return None
    if not (-180 <= d <= 180 and 0 <= k <= 100):
        return None
    return d, k


def wiggle():
    for _ in range(3):
        servo.angle = FORWARD - 20
        time.sleep(0.12)
        servo.angle = FORWARD + 20
        time.sleep(0.12)


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", UDP_PORT))
    sock.setblocking(False)
    print(f"listening on udp {UDP_PORT}")

    # Boot self-test: a visible swing proves wiring and power before the network matters.
    for a in (45, 135, REST):
        servo.angle = a
        time.sleep(0.4)

    smoothed = written = target = REST
    last_dist, last_packet, close_since, found_done = 100, 0.0, None, False
    while True:
        while True:  # drain the queue; only the newest packet matters
            try:
                data, _ = sock.recvfrom(64)
            except BlockingIOError:
                break
            got = parse(data)
            if got:
                target, last_dist = dir_to_servo(got[0]), got[1]
                last_packet = time.monotonic()

        now = time.monotonic()
        stale = now - last_packet > TIMEOUT_S
        if stale:
            target, close_since, found_done = REST, None, False

        # Found: close enough for long enough -> wiggle once, then keep pointing.
        if not stale and last_dist <= FOUND_DIST:
            close_since = close_since or now
            if not found_done and now - close_since >= FOUND_HOLD_S:
                wiggle()
                servo.angle = written
                found_done = True
        else:
            close_since = None
            if last_dist > FOUND_DIST * 2:
                found_done = False  # moved away, allow another find

        smoothed += SMOOTH * (target - smoothed)
        angle = int(round(smoothed))
        if abs(angle - written) >= DEADBAND_DEG:
            servo.angle = angle
            written = angle
        time.sleep(TICK_S)


if __name__ == "__main__":
    main()
