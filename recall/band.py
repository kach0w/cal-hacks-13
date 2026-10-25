"""The link to the band: the text packet "dir,dist" (dir in degrees, + = to the person's right;
dist 0..100) to the ESP32 band over Bluetooth LE (firmware/band/band.ino), or over wifi as UDP to
a host such as the Raspberry Pi band (firmware/pi_band.py).

    link = connect("sixsevendemons")      # Bluetooth: the ESP32 band's name
    link = connect("raspberrypi.local")   # wifi: a host name or IP address (it has a dot)
    link.send(-30, 50)                    # never blocks; only the newest packet matters
    link = connect("sixsevendemons", on_press=f)   # f() runs on its own thread at each button press

Over Bluetooth the band's button comes back the other way: "press" on its UART TX characteristic.

Bluetooth writes to the band's Nordic UART service, the one phone apps such as nRF Connect can
also write to when testing. macOS asks for Bluetooth permission for the terminal the first time;
don't pair the band in System Settings, the link connects to it directly.
"""
import asyncio
import logging
import socket
import threading
import time

log = logging.getLogger("recall")

NAME = "sixsevendemons"                              # what the ESP32 band advertises
UART_RX = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"     # its Nordic UART RX characteristic: packets go here
UART_TX = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"     # and TX: it notifies button presses here
UDP_PORT = 4210                                      # firmware/pi_band.py listens here
RETRY_S = 5                                          # look for a missing band again this often


def packet(deg, dist) -> bytes:
    return f"{int(round(deg))},{int(dist)}".encode()


def is_host(target: str) -> bool:
    """A wifi host (a dotted name or an IP address) rather than a Bluetooth name."""
    return "." in target


def connect(target=NAME, on_press=None):
    """A link to the band, looking for it in the background until it's found. on_press: called at
    each press of the band's button (Bluetooth only; the Raspberry Pi band has none)."""
    return Udp(target) if is_host(target) else Ble(target, on_press)


class Ble:
    """Bluetooth LE to the band, on its own thread: finds it by name, connects, writes each packet
    without waiting for a reply, and reconnects when the link drops. Each button press runs
    on_press on a thread of its own, so a long question never holds up the link. scanner and client
    stand in for bleak's BleakScanner and BleakClient in tests."""

    def __init__(self, name=NAME, on_press=None, scanner=None, client=None):
        self.name, self.connected, self.on_press = name, False, on_press
        self._scanner, self._client = scanner, client
        self._packet = None
        self._loop = self._wake = None
        threading.Thread(target=lambda: asyncio.run(self._run()), name="band", daemon=True).start()

    def send(self, deg, dist):
        self._packet = packet(deg, dist)
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(self._wake.set)

    async def _run(self):
        if self._scanner is None:
            from bleak import BleakClient, BleakScanner
            self._scanner, self._client = BleakScanner, BleakClient
        self._wake = asyncio.Event()
        self._loop = asyncio.get_running_loop()
        warned = False
        while True:
            try:
                device = await self._scanner.find_device_by_name(self.name, timeout=RETRY_S)
            except Exception as e:  # Bluetooth off, or the terminal not allowed to use it yet
                if not warned:
                    log.warning("band: Bluetooth unavailable (%s); guidance stays on the page", e)
                    warned = True
                await asyncio.sleep(RETRY_S)
                continue
            if device is None:
                if not warned:
                    log.warning("band: can't find %s over Bluetooth yet; guidance stays on the page until it's on",
                                self.name)
                    warned = True
                continue
            try:
                async with self._client(device) as client:
                    if self.on_press is not None:
                        await client.start_notify(UART_TX, self._pressed)
                    self.connected, warned = True, False
                    log.info("band: connected to %s over Bluetooth", self.name)
                    while client.is_connected:
                        try:
                            await asyncio.wait_for(self._wake.wait(), timeout=1.0)
                        except asyncio.TimeoutError:
                            continue  # nothing to send: check the link is still up
                        self._wake.clear()
                        await client.write_gatt_char(UART_RX, self._packet, response=False)
            except Exception as e:  # out of range, switched off, a failed write
                log.warning("band: Bluetooth link to %s dropped (%s)", self.name, e)
            finally:
                self.connected = False
            log.info("band: looking for %s again", self.name)

    def _pressed(self, _characteristic, data):
        log.info("band: button pressed (%s)", bytes(data).decode(errors="replace"))
        threading.Thread(target=self.on_press, name="band-button", daemon=True).start()


class Udp:
    """UDP over wifi to a host, looked up until it answers: it may join the wifi after we start."""

    def __init__(self, host, port=UDP_PORT):
        self.host, self.port, self.addr = host, port, None
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        threading.Thread(target=self._find, name="band", daemon=True).start()

    @property
    def connected(self) -> bool:
        return self.addr is not None

    def _find(self):
        warned = False
        while self.addr is None:
            try:
                self.addr = (socket.gethostbyname(self.host), self.port)
                log.info("band: sending guidance to %s:%d over wifi", *self.addr)
            except OSError:
                if not warned:
                    log.warning("band: can't find %s yet; guidance stays on the page until it's on the wifi",
                                self.host)
                    warned = True
                time.sleep(RETRY_S)

    def send(self, deg, dist):
        if self.addr:
            try:
                self._sock.sendto(packet(deg, dist), self.addr)
            except OSError:  # band off or out of range: the page keeps guiding
                pass
