// Recall compass band: ESP32 + one servo.
//
// Takes text packets "dir,dist" from the laptop over Bluetooth LE and points the arrow.
//   dir  = degrees from the band's forward axis (toward the fingers) to the target,
//          -180..180, positive = clockwise seen from above
//   dist = 0..100, 0 = on top of the target
//
// The band advertises as "sixsevendemons" with the Nordic UART service; the laptop
// (recall/band.py) writes each packet to its RX characteristic. Needs an ESP32 with
// Bluetooth (not the S2) and the ESP32Servo library; the BLE library comes with the
// ESP32 board package. If Arduino says the sketch is too big: Tools -> Partition Scheme
// -> Huge APP.
//
// The band decides nothing about the world. It only:
//   - points the servo at dir (clamped to what the servo can reach)
//   - wiggles once when dist stays under FOUND_DIST for FOUND_HOLD_MS
//   - centres the arrow when packets stop (a frozen arrow is a lie)
//
// Test from the laptop:  .venv/bin/python tools/band_test.py sixsevendemons sweep
// or from a phone: nRF Connect -> connect to sixsevendemons -> write the text "-30,50"
// to the RX characteristic (6E400002-...).

#include <BLEDevice.h>
#include <BLEServer.h>
#include <ESP32Servo.h>

const int SERVO_PIN = 13;
const char* BLE_NAME = "sixsevendemons";
const char* UART_SERVICE = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E";  // Nordic UART service
const char* UART_RX = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E";       // the laptop writes packets here
const bool SERVO_REVERSED = true;   // SG90 angle grows counterclockwise from above; dir is clockwise

// FORWARD: servo angle where the arrow points straight out from the chest.
// REST: where the arrow sits when idle or lost. Set both to match how the servo is mounted.
const int FORWARD = 90;
const int REST = 180;
const unsigned long TIMEOUT_MS = 1000;     // no packet this long -> centre
const float SMOOTH = 0.35;                 // 0..1, higher = snappier, jitterier
const int DEADBAND_DEG = 3;                // ignore moves smaller than this
const unsigned long SERVO_TICK_MS = 20;    // 50 Hz servo updates
const int FOUND_DIST = 8;
const unsigned long FOUND_HOLD_MS = 1500;

Servo servo;

float smoothed = REST;
int written = REST;
int targetAngle = REST;
int lastDist = 100;
unsigned long lastPacketAt = 0;
unsigned long closeSince = 0;
bool foundDone = false;
unsigned long lastTick = 0;

// The newest packet, written by the Bluetooth task and read by loop(). Only the newest matters.
portMUX_TYPE rxMux = portMUX_INITIALIZER_UNLOCKED;
char rxBuf[32];
volatile bool rxNew = false;

int dirToServo(int dir) {
  if (SERVO_REVERSED) dir = -dir;
  // Servo reaches 0..180. Directions it cannot reach pin to the nearest edge.
  return constrain(FORWARD + dir, 0, 180);
}

void writeServo(int angle) {
  if (abs(angle - written) < DEADBAND_DEG) return;
  servo.write(angle);
  written = angle;
}

void wiggle() {
  for (int i = 0; i < 3; i++) {
    servo.write(FORWARD - 20); delay(120);
    servo.write(FORWARD + 20); delay(120);
  }
  // Snap back to where we were pointing; the loop keeps following the target.
  servo.write(written);
}

class RxCallbacks : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic* c) override {
    size_t len = c->getLength();
    if (len > sizeof(rxBuf) - 1) len = sizeof(rxBuf) - 1;
    portENTER_CRITICAL(&rxMux);
    memcpy(rxBuf, c->getData(), len);
    rxBuf[len] = '\0';
    rxNew = true;
    portEXIT_CRITICAL(&rxMux);
  }
};

class ServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer* server) override {
    Serial.println("ble: laptop connected");
  }
  void onDisconnect(BLEServer* server) override {
    // Packets stop, so the arrow goes to rest by itself; be findable again.
    Serial.println("ble: disconnected, advertising again");
    BLEDevice::startAdvertising();
  }
};

void startBluetooth() {
  BLEDevice::init(BLE_NAME);
  BLEServer* server = BLEDevice::createServer();
  server->setCallbacks(new ServerCallbacks());
  BLEService* service = server->createService(UART_SERVICE);
  BLECharacteristic* rx = service->createCharacteristic(
      UART_RX, BLECharacteristic::PROPERTY_WRITE | BLECharacteristic::PROPERTY_WRITE_NR);
  rx->setCallbacks(new RxCallbacks());
  service->start();
  BLEDevice::startAdvertising();  // the name only: the laptop finds the band by name
  Serial.printf("ble: advertising as %s\n", BLE_NAME);
}

// Parses "dir,dist". Returns false on garbage so a bad packet never moves the arrow.
bool parsePacket(const char* s, int& dir, int& dist) {
  char* end;
  long d = strtol(s, &end, 10);
  if (end == s || *end != ',') return false;
  const char* rest = end + 1;
  long k = strtol(rest, &end, 10);
  if (end == rest) return false;
  if (d < -180 || d > 180 || k < 0 || k > 100) return false;
  dir = d;
  dist = k;
  return true;
}

void readPackets() {
  if (!rxNew) return;
  char buf[sizeof(rxBuf)];
  portENTER_CRITICAL(&rxMux);
  memcpy(buf, rxBuf, sizeof(buf));
  rxNew = false;
  portEXIT_CRITICAL(&rxMux);
  int dir, dist;
  if (!parsePacket(buf, dir, dist)) {
    Serial.printf("rx bad: %s\n", buf);
    return;
  }
  Serial.printf("rx dir %d dist %d -> servo %d\n", dir, dist, dirToServo(dir));
  targetAngle = dirToServo(dir);
  lastDist = dist;
  lastPacketAt = millis();
}

void setup() {
  Serial.begin(115200);
  ESP32PWM::allocateTimer(0);
  servo.setPeriodHertz(50);
  servo.attach(SERVO_PIN, 500, 2500);  // symmetric around 1500us so left and right swing equally
  // Boot self-test: a visible swing proves wiring and power before Bluetooth is involved.
  Serial.println("servo: self-test");
  servo.write(45); delay(400);
  servo.write(135); delay(400);
  servo.write(REST);
  startBluetooth();
}

void loop() {
  readPackets();

  unsigned long now = millis();
  if (now - lastTick < SERVO_TICK_MS) return;
  lastTick = now;

  bool stale = lastPacketAt == 0 || now - lastPacketAt > TIMEOUT_MS;
  if (stale) {
    targetAngle = REST;
    closeSince = 0;
    foundDone = false;  // next guidance session can find again
  }

  // Found: close enough for long enough -> wiggle once, then keep pointing.
  if (!stale && lastDist <= FOUND_DIST) {
    if (closeSince == 0) closeSince = now;
    if (!foundDone && now - closeSince >= FOUND_HOLD_MS) {
      wiggle();
      foundDone = true;
    }
  } else {
    closeSince = 0;
    if (lastDist > FOUND_DIST * 2) foundDone = false;  // moved away, allow another find
  }
  smoothed += SMOOTH * (targetAngle - smoothed);
  writeServo((int)(smoothed + 0.5));
}
