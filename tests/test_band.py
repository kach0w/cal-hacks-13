"""recall/band.py: which link a --band target gets, delivery over wifi, and the Bluetooth link's
find -> connect -> write -> reconnect cycle against a stand-in for bleak."""
import socket
import time

from recall import band


def wait(condition, timeout=3.0):
    end = time.time() + timeout
    while not condition():
        assert time.time() < end, "timed out"
        time.sleep(0.01)


def test_names_are_bluetooth_and_hosts_are_wifi():
    assert not band.is_host(band.NAME) and not band.is_host("another-band")
    assert band.is_host("raspberrypi.local") and band.is_host("192.168.4.64")


def test_wifi_delivers_the_packet():
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(2)
    link = band.Udp("127.0.0.1", rx.getsockname()[1])
    wait(lambda: link.connected)
    link.send(-30.4, 50)
    assert rx.recv(64) == b"-30,50"


class FakeScanner:
    looks = 0

    @classmethod
    async def find_device_by_name(cls, name, timeout):
        cls.looks += 1
        return None if cls.looks == 1 else f"device:{name}"  # not advertising yet on the first look


class FakeClient:
    sessions, writes = 0, []

    def __init__(self, device):
        self.device, self.is_connected = device, True

    async def __aenter__(self):
        FakeClient.sessions += 1
        return self

    async def __aexit__(self, *exc):
        self.is_connected = False

    async def write_gatt_char(self, char, data, response):
        FakeClient.writes.append((self.device, char, data, response))
        if data == b"0,0":
            self.is_connected = False  # walked out of range


def test_bluetooth_finds_connects_writes_and_reconnects():
    link = band.Ble(band.NAME, scanner=FakeScanner, client=FakeClient)
    wait(lambda: link.connected)
    assert FakeScanner.looks == 2  # kept looking until the band advertised
    link.send(-30, 50)
    wait(lambda: FakeClient.writes)
    assert FakeClient.writes[0] == (f"device:{band.NAME}", band.UART_RX, b"-30,50", False)
    link.send(0, 0)  # the stand-in drops the link after this write
    wait(lambda: FakeClient.sessions == 2 and link.connected)
    link.send(10, 20)
    wait(lambda: FakeClient.writes[-1][2] == b"10,20")


class NotifyingClient:
    """A band whose button can be pressed: start_notify keeps the handler."""
    handlers = {}

    def __init__(self, device):
        self.is_connected = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.is_connected = False

    async def start_notify(self, char, handler):
        NotifyingClient.handlers[char] = handler

    async def write_gatt_char(self, char, data, response):
        pass


class AlwaysThere:
    @staticmethod
    async def find_device_by_name(name, timeout):
        return f"device:{name}"


def test_the_band_button_calls_on_press():
    pressed = []
    link = band.Ble(band.NAME, on_press=lambda: pressed.append(1), scanner=AlwaysThere, client=NotifyingClient)
    wait(lambda: link.connected)
    NotifyingClient.handlers[band.UART_TX](None, bytearray(b"press"))
    wait(lambda: pressed == [1])
