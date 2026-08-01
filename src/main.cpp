// SortVision arm robot firmware — ESP32, Opsi A (no Jetson Nano).
//
// Subscribes to MQTT_BASE_TOPIC "/command" for ready-to-run joint angles
// computed by the Laravel backend (App\Services\ArmMqttService /
// App\Models\TargetZonePreset — the ESP32 does NOT compute inverse
// kinematics, it only executes). Publishes "/status" telemetry consumed by
// App\Console\Commands\MqttListen on the backend.
//
// Command payload (JSON), matches ArmMqttService::buildCommandPayload():
//   {
//     "category": "Food & Beverage",
//     "zone": "food-beverage",
//     "joint_angles": [10, 20, 30, 40, 50, 60],   // degrees, 6 elements:
//                                                   // [base, shoulder, elbow,
//                                                   //  wristRoll, wristPitch,
//                                                   //  gripperServo]
//     "issued_at": "2026-08-01T12:00:00+00:00"
//   }
//
// Status payload (JSON) this firmware publishes, matches what
// MqttListen::handleStatus() reads:
//   { "state": "idle"|"running"|"error", "detail": "...", "last_command": {...},
//     "telemetry": { "positions_deg": [...], "rssi": -55 } }

#include <Arduino.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <AccelStepper.h>
#include <ESP32Servo.h>
#include <ArduinoJson.h>

#include "config.h"

// ---------------------------------------------------------------------------
// Stepper + servo setup
// ---------------------------------------------------------------------------
struct AxisPins {
  uint8_t step;
  uint8_t dir;
  uint8_t enable;
};

static const AxisPins AXIS_PINS[AXIS_COUNT] = {
  {BASE_STEP_PIN, BASE_DIR_PIN, BASE_EN_PIN},
  {SHOULDER_STEP_PIN, SHOULDER_DIR_PIN, SHOULDER_EN_PIN},
  {ELBOW_STEP_PIN, ELBOW_DIR_PIN, ELBOW_EN_PIN},
  {WRIST_ROLL_STEP_PIN, WRIST_ROLL_DIR_PIN, WRIST_ROLL_EN_PIN},
  {WRIST_PITCH_STEP_PIN, WRIST_PITCH_DIR_PIN, WRIST_PITCH_EN_PIN},
};

// Steps-per-degree per axis. All axes default to the same value; override
// per-index here once real gearbox/pulley ratios are measured (e.g. the base
// axis often has extra reduction on Moveo-style arms).
static const float STEPS_PER_DEGREE[AXIS_COUNT] = {
  DEFAULT_STEPS_PER_DEGREE, // base
  DEFAULT_STEPS_PER_DEGREE, // shoulder
  DEFAULT_STEPS_PER_DEGREE, // elbow
  DEFAULT_STEPS_PER_DEGREE, // wrist roll
  DEFAULT_STEPS_PER_DEGREE, // wrist pitch
};

AccelStepper steppers[AXIS_COUNT] = {
  AccelStepper(AccelStepper::DRIVER, BASE_STEP_PIN, BASE_DIR_PIN),
  AccelStepper(AccelStepper::DRIVER, SHOULDER_STEP_PIN, SHOULDER_DIR_PIN),
  AccelStepper(AccelStepper::DRIVER, ELBOW_STEP_PIN, ELBOW_DIR_PIN),
  AccelStepper(AccelStepper::DRIVER, WRIST_ROLL_STEP_PIN, WRIST_ROLL_DIR_PIN),
  AccelStepper(AccelStepper::DRIVER, WRIST_PITCH_STEP_PIN, WRIST_PITCH_DIR_PIN),
};

Servo gripperServo;
float lastGripperAngle = 90.0f;

// ---------------------------------------------------------------------------
// WiFi + MQTT
// ---------------------------------------------------------------------------
WiFiClient wifiClient;
PubSubClient mqtt(wifiClient);

String commandTopic;
String statusTopic;

// Snapshot of the currently-executing command, echoed back in "last_command"
// on the next status publish so the backend/dashboard can show what the arm
// is actually doing.
String lastCommandJson = "null";

enum class ArmState { IDLE, RUNNING, ERROR };
ArmState currentState = ArmState::IDLE;
unsigned long lastHeartbeatAt = 0;
bool statusDirty = true; // force one publish right after boot

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
void connectWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.printf("Connecting to WiFi '%s'", WIFI_SSID);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print('.');
  }
  Serial.printf("\nWiFi connected, IP: %s\n", WiFi.localIP().toString().c_str());
}

void setAllEnabled(bool enabled) {
  // A4988/DRV8825 EN pin is active-LOW: LOW = driver enabled/energized.
  for (uint8_t i = 0; i < AXIS_COUNT; i++) {
    digitalWrite(AXIS_PINS[i].enable, enabled ? LOW : HIGH);
  }
}

bool allAxesIdle() {
  for (uint8_t i = 0; i < AXIS_COUNT; i++) {
    if (steppers[i].distanceToGo() != 0) return false;
  }
  return true;
}

void publishStatus(const char *state, const char *detail) {
  JsonDocument doc;
  doc["state"] = state;
  if (detail != nullptr) doc["detail"] = detail;

  JsonDocument lastCmdDoc;
  DeserializationError err = deserializeJson(lastCmdDoc, lastCommandJson);
  if (!err) doc["last_command"] = lastCmdDoc;

  JsonObject telemetry = doc["telemetry"].to<JsonObject>();
  JsonArray positions = telemetry["positions_deg"].to<JsonArray>();
  for (uint8_t i = 0; i < AXIS_COUNT; i++) {
    positions.add(steppers[i].currentPosition() / STEPS_PER_DEGREE[i]);
  }
  positions.add(lastGripperAngle);
  telemetry["rssi"] = WiFi.RSSI();

  char buffer[512];
  size_t n = serializeJson(doc, buffer);
  mqtt.publish(statusTopic.c_str(), buffer, n);
}

void setState(ArmState next, const char *detail = nullptr) {
  bool changed = (next != currentState);
  currentState = next;
  if (changed) statusDirty = true;

  const char *label = (next == ArmState::RUNNING) ? "running"
                     : (next == ArmState::ERROR)   ? "error"
                                                    : "idle";
  if (statusDirty) {
    publishStatus(label, detail);
    statusDirty = false;
  }
}

// Apply a parsed "joint_angles" array (6 elements) to the 5 steppers + 1
// servo. Returns false (and does not move anything) if the payload is
// malformed, so a bad command can never partially move the arm.
bool applyJointAngles(JsonArrayConst anglesArray) {
  if (anglesArray.size() != AXIS_COUNT + 1) {
    Serial.printf("Rejecting command: expected %d joint_angles, got %d\n",
                  AXIS_COUNT + 1, anglesArray.size());
    return false;
  }

  for (uint8_t i = 0; i < AXIS_COUNT; i++) {
    float degrees = anglesArray[i].as<float>();
    long targetSteps = lroundf(degrees * STEPS_PER_DEGREE[i]);
    steppers[i].moveTo(targetSteps);
  }

  lastGripperAngle = anglesArray[AXIS_COUNT].as<float>();
  gripperServo.write((int)constrain(lastGripperAngle, 0, 180));

  return true;
}

void handleCommand(char *payload, unsigned int length) {
  JsonDocument doc;
  DeserializationError err = deserializeJson(doc, payload, length);
  if (err) {
    Serial.printf("arm/command: invalid JSON (%s)\n", err.c_str());
    setState(ArmState::ERROR, "Payload JSON tidak valid");
    return;
  }

  JsonArrayConst angles = doc["joint_angles"].as<JsonArrayConst>();
  if (angles.isNull() || !applyJointAngles(angles)) {
    setState(ArmState::ERROR, "joint_angles hilang atau salah jumlah elemen");
    return;
  }

  // Echo the raw command back verbatim as "last_command" telemetry.
  lastCommandJson = String(payload, length);

  setAllEnabled(true);
  setState(ArmState::RUNNING, "Menjalankan command baru");
}

void mqttCallback(char *topic, byte *payload, unsigned int length) {
  if (String(topic) == commandTopic) {
    handleCommand(reinterpret_cast<char *>(payload), length);
  }
}

void reconnectMqtt() {
  while (!mqtt.connected()) {
    Serial.print("Connecting to MQTT broker...");
    bool ok;
    if (strlen(MQTT_USERNAME) > 0) {
      ok = mqtt.connect(MQTT_CLIENT_ID, MQTT_USERNAME, MQTT_PASSWORD);
    } else {
      ok = mqtt.connect(MQTT_CLIENT_ID);
    }

    if (ok) {
      Serial.println("connected.");
      mqtt.subscribe(commandTopic.c_str());
      statusDirty = true;
    } else {
      Serial.printf("failed, rc=%d, retrying in 2s\n", mqtt.state());
      delay(2000);
    }
  }
}

// ---------------------------------------------------------------------------
// Arduino entry points
// ---------------------------------------------------------------------------
void setup() {
  Serial.begin(115200);

  commandTopic = String(MQTT_BASE_TOPIC) + "/command";
  statusTopic = String(MQTT_BASE_TOPIC) + "/status";

  for (uint8_t i = 0; i < AXIS_COUNT; i++) {
    pinMode(AXIS_PINS[i].enable, OUTPUT);
    steppers[i].setMaxSpeed(MAX_SPEED_STEPS_PER_SEC);
    steppers[i].setAcceleration(ACCELERATION_STEPS_PER_SEC2);
  }
  setAllEnabled(false); // drivers stay de-energized (cool, no holding torque) until a command arrives

  gripperServo.setPeriodHertz(50);
  gripperServo.attach(GRIPPER_SERVO_PIN, 500, 2400);
  gripperServo.write((int)lastGripperAngle);

  connectWiFi();

  mqtt.setServer(MQTT_HOST, MQTT_PORT);
  mqtt.setCallback(mqttCallback);
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    connectWiFi();
  }
  if (!mqtt.connected()) {
    reconnectMqtt();
  }
  mqtt.loop();

  for (uint8_t i = 0; i < AXIS_COUNT; i++) {
    steppers[i].run();
  }

  if (currentState == ArmState::RUNNING && allAxesIdle()) {
    setAllEnabled(false); // de-energize once the move finishes
    setState(ArmState::IDLE, "Command selesai");
  }

  unsigned long now = millis();
  if (statusDirty || now - lastHeartbeatAt >= STATUS_HEARTBEAT_MS) {
    const char *label = (currentState == ArmState::RUNNING) ? "running"
                       : (currentState == ArmState::ERROR)   ? "error"
                                                              : "idle";
    publishStatus(label, nullptr);
    statusDirty = false;
    lastHeartbeatAt = now;
  }
}
