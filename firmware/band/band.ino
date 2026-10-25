// Recall compass band: ESP32 + one servo.
//
// Listens for UDP text packets "dir,dist" from the laptop and points the arrow.
//   dir  = degrees from the band's forward axis (toward the fingers) to the target,
//          -180..180, positive = clockwise seen from above
//   dist = 0..100, 0 = on top of the target
//
// The band decides nothing about the world. It only:
//   - points the servo at dir (clamped to what the servo can reach)
//   - wiggles once when dist stays under FOUND_DIST for FOUND_HOLD_MS
//   - centres the arrow when packets stop (a frozen arrow is a lie)
//
// Test from the laptop:  echo "-30,50" | nc -u -w0 <band-ip> 4210

#include <WiFi.h>
#include <WiFiUdp.h>
#include <ESPmDNS.h>
#include <ESP32Servo.h>
#include "secrets.h"  // WIFI_SSID, WIFI_PASS. Copy secrets.example.h

const int SERVO_PIN = 13;
const int UDP_PORT = 4210;
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
WiFiUDP udp;

float smoothed = REST;
int written = REST;
int targetAngle = REST;
int lastDist = 100;
unsigned long lastPacketAt = 0;
unsigned long closeSince = 0;
bool foundDone = false;
unsigned long lastTick = 0;

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

void connectWifi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("wifi: joining ");
  Serial.println(WIFI_SSID);
  while (WiFi.status() != WL_CONNECTED) {
    delay(250);
    Serial.print(".");
  }
  Serial.print("\nwifi: ip ");
  Serial.println(WiFi.localIP());
  if (MDNS.begin("recall-band")) Serial.println("mdns: recall-band.local");
  udp.begin(UDP_PORT);
  Serial.printf("udp: listening on %d\n", UDP_PORT);
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
  char buf[32];
  // Drain everything queued; only the newest packet matters.
  while (int n = udp.parsePacket()) {
    int len = udp.read(buf, sizeof(buf) - 1);
    if (len <= 0) continue;
    buf[len] = '\0';
    int dir, dist;
    if (!parsePacket(buf, dir, dist)) {
      Serial.printf("rx bad: %s\n", buf);
      continue;
    }
    Serial.printf("rx dir %d dist %d -> servo %d\n", dir, dist, dirToServo(dir));
    targetAngle = dirToServo(dir);
    lastDist = dist;
    lastPacketAt = millis();
  }
}

void setup() {
  Serial.begin(115200);
  ESP32PWM::allocateTimer(0);
  servo.setPeriodHertz(50);
  servo.attach(SERVO_PIN, 500, 2500);  // symmetric around 1500us so left and right swing equally
  // Boot self-test: a visible swing proves wiring and power before wifi is involved.
  Serial.println("servo: self-test");
  servo.write(45); delay(400);
  servo.write(135); delay(400);
  servo.write(REST);
  connectWifi();
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    servo.write(REST);
    written = REST;
    connectWifi();
  }

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
