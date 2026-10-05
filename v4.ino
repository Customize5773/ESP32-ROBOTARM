#include <Arduino.h>
#include <AccelStepper.h>
#include <ESP32Servo.h>
#include <Preferences.h>
#include <math.h>

// =========================================================================
// DIMENSI LENGAN (mm) - dipakai hanya oleh mode XYZ (IK)
// =========================================================================
const float L1 = 235.0;
const float L2 = 225.0;
const float L3 = 220.0;

#define SERVO_PIN   16
#define AXIS_COUNT  5
#define CAL_POINTS  4

// Axis 0 = J1 | Axis 1 = J2/J3 (satu motor) | Axis 2 = J4 | Axis 3 = J5 | Axis 4 = J6
struct AxisPins { uint8_t stepPin; uint8_t dirPin; };

AxisPins pins[AXIS_COUNT] = {
  {13, 14},  // J1
  {27, 23},  // J2/J3
  {32, 33},  // J4
  {19, 18},  // J5
  {26, 25}   // J6
};

// step per derajat (sudah termasuk microstep + gearbox/pulley)
float stepsPerDegree[AXIS_COUNT] = { 92.44456, 58.5185, 228.9259, 8.89, 44.8889 };

// arah motor (+1 / -1)
int jointDirection[AXIS_COUNT] = { 1, -1, 1, 1, 1 };

// batas joint (derajat)
float jointMin[AXIS_COUNT] = { -180, -90, -150, -180, -180 };
float jointMax[AXIS_COUNT] = {  180,  90,  150,  180,  180 };

// Workspace mode XYZ (IK)
float MIN_X = -180, MAX_X = 180;
float MIN_Y =  370, MAX_Y = 500;   // CATATAN: jangkauan maksimum L2+L3 = 445 mm
float MIN_Z =  120, MAX_Z = 350;
float J1_SCALE   = 0.5;
float Z_MIN_J6   = 20;
float Z_MAX_J6   = 30;

// Kecepatan referensi untuk axis yang jaraknya paling jauh (step/s).
// Axis lain otomatis diperlambat supaya semua tiba bersamaan.
float globalMaxSpeed = 800;
float globalAccel    = 400;

AccelStepper axis[AXIS_COUNT] = {
  AccelStepper(AccelStepper::DRIVER, pins[0].stepPin, pins[0].dirPin),
  AccelStepper(AccelStepper::DRIVER, pins[1].stepPin, pins[1].dirPin),
  AccelStepper(AccelStepper::DRIVER, pins[2].stepPin, pins[2].dirPin),
  AccelStepper(AccelStepper::DRIVER, pins[3].stepPin, pins[3].dirPin),
  AccelStepper(AccelStepper::DRIVER, pins[4].stepPin, pins[4].dirPin)
};

Servo gripperServo;
Preferences preferences;

bool zeroCalibrated = false;
bool motionActive   = false;
int  servoAngle     = 0;
String serialBuffer = "";

// ---------------- Data kalibrasi 4 titik ----------------
// Titik 1 = (0,0)   kiri-dekat robot
// Titik 2 = (W,0)   kanan-dekat robot
// Titik 3 = (W,H)   kanan-jauh
// Titik 4 = (0,H)   kiri-jauh
long    calPos[CAL_POINTS][AXIS_COUNT];
uint8_t calMask = 0;          // bit n = titik n+1 sudah disimpan
float   areaW = 300.0;        // lebar area (mm), arah X
float   areaH = 200.0;        // panjang area (mm), arah Y


// =========================================================================
// KONVERSI
// =========================================================================
long degreeToSteps(int a, float degree) {
  return lround(degree * stepsPerDegree[a] * jointDirection[a]);
}

float stepsToDegree(int a, long steps) {
  return (float)steps / (stepsPerDegree[a] * jointDirection[a]);
}

float mapFloat(float x, float inMin, float inMax, float outMin, float outMax) {
  if (inMax == inMin) return outMin;
  return (x - inMin) * (outMax - outMin) / (inMax - inMin) + outMin;
}

int jointToAxis(int joint) {
  switch (joint) {
    case 1: return 0;
    case 2: return 1;
    case 3: return 1;
    case 4: return 2;
    case 5: return 3;
    case 6: return 4;
  }
  return -1;
}


// =========================================================================
// PENYIMPANAN (Preferences)
// =========================================================================
void savePosition() {
  preferences.begin("robotarm", false);
  for (int i = 0; i < AXIS_COUNT; i++) {
    String key = "p" + String(i);
    preferences.putLong(key.c_str(), axis[i].currentPosition());
  }
  preferences.putBool("zero", zeroCalibrated);
  preferences.putInt("servo", servoAngle);
  preferences.end();
}

void saveCalibration() {
  preferences.begin("robotarm", false);
  for (int p = 0; p < CAL_POINTS; p++) {
    for (int a = 0; a < AXIS_COUNT; a++) {
      String key = "c" + String(p) + "_" + String(a);
      preferences.putLong(key.c_str(), calPos[p][a]);
    }
  }
  preferences.putUChar("cmask", calMask);
  preferences.putFloat("areaW", areaW);
  preferences.putFloat("areaH", areaH);
  preferences.end();
}

void loadAll() {
  preferences.begin("robotarm", true);

  zeroCalibrated = preferences.getBool("zero", false);
  for (int i = 0; i < AXIS_COUNT; i++) {
    String key = "p" + String(i);
    axis[i].setCurrentPosition(preferences.getLong(key.c_str(), 0));
  }
  servoAngle = preferences.getInt("servo", 0);

  calMask = preferences.getUChar("cmask", 0);
  areaW   = preferences.getFloat("areaW", 300.0);
  areaH   = preferences.getFloat("areaH", 200.0);
  for (int p = 0; p < CAL_POINTS; p++) {
    for (int a = 0; a < AXIS_COUNT; a++) {
      String key = "c" + String(p) + "_" + String(a);
      calPos[p][a] = preferences.getLong(key.c_str(), 0);
    }
  }

  preferences.end();
}


// =========================================================================
// GERAKAN (sinkron: semua axis tiba bersamaan)
// =========================================================================
bool limitOK(int a, float deg) {
  if (deg < jointMin[a] || deg > jointMax[a]) {
    Serial.print("LIMIT axis ");
    Serial.print(a);
    Serial.print(" : ");
    Serial.println(deg);
    return false;
  }
  return true;
}

void getTargets(long t[AXIS_COUNT]) {
  for (int i = 0; i < AXIS_COUNT; i++) t[i] = axis[i].targetPosition();
}

// Cek SEMUA limit dulu, baru gerak. Kalau ada satu yang gagal, tidak ada axis yang bergerak.
bool commitMove(const long target[AXIS_COUNT]) {
  for (int i = 0; i < AXIS_COUNT; i++) {
    if (!limitOK(i, stepsToDegree(i, target[i]))) return false;
  }

  long dist[AXIS_COUNT];
  long dMax = 0;
  for (int i = 0; i < AXIS_COUNT; i++) {
    dist[i] = labs(target[i] - axis[i].currentPosition());
    if (dist[i] > dMax) dMax = dist[i];
  }

  if (dMax == 0) {
    Serial.println("Sudah di posisi target");
    return true;
  }

  for (int i = 0; i < AXIS_COUNT; i++) {
    if (dist[i] > 0) {
      float k = (float)dist[i] / (float)dMax;   // skala 0..1
      axis[i].setMaxSpeed(fmaxf(1.0f, globalMaxSpeed * k));
      axis[i].setAcceleration(fmaxf(1.0f, globalAccel * k));
    }
    axis[i].moveTo(target[i]);
  }

  motionActive = true;
  return true;
}

void setServo(int val) {
  val = constrain(val, 0, 180);
  servoAngle = val;
  gripperServo.write(val);
  Serial.print("Servo=");
  Serial.println(val);
}


// =========================================================================
// ZERO / HOME / STATUS
// =========================================================================
void setZero() {
  for (int i = 0; i < AXIS_COUNT; i++) axis[i].setCurrentPosition(0);
  zeroCalibrated = true;
  savePosition();
  Serial.println("ZERO OK");
  if (calMask != 0) {
    Serial.println("PERINGATAN: kalibrasi 4 titik hanya valid jika posisi ZERO fisik sama dengan saat kalibrasi.");
  }
}

void moveHome() {
  if (!zeroCalibrated) {
    Serial.println("ZERO belum dibuat");
    return;
  }
  long t[AXIS_COUNT] = {0, 0, 0, 0, 0};
  commitMove(t);
}

void printStatus() {
  const char* name[AXIS_COUNT] = {"J1", "J2/J3", "J4", "J5", "J6"};

  Serial.println();
  Serial.println("===== STATUS =====");
  for (int i = 0; i < AXIS_COUNT; i++) {
    Serial.print(name[i]);
    Serial.print(" : ");
    Serial.print(stepsToDegree(i, axis[i].currentPosition()));
    Serial.println(" deg");
  }
  Serial.print("Servo : ");
  Serial.println(servoAngle);
  Serial.print("Kalibrasi titik: ");
  for (int p = 0; p < CAL_POINTS; p++) Serial.print((calMask & (1 << p)) ? "[x]" : "[ ]");
  Serial.print("  Area: ");
  Serial.print(areaW);
  Serial.print(" x ");
  Serial.print(areaH);
  Serial.println(" mm");
}


// =========================================================================
// KALIBRASI 4 TITIK
// =========================================================================
void savePoint(int n) {
  if (n < 1 || n > CAL_POINTS) {
    Serial.println("Nomor titik harus 1-4");
    return;
  }
  if (motionActive) {
    Serial.println("Tunggu gerakan selesai dulu");
    return;
  }
  if (!zeroCalibrated) {
    Serial.println("ZERO belum dibuat. Buat ZERO dulu sebelum kalibrasi.");
    return;
  }

  for (int a = 0; a < AXIS_COUNT; a++) calPos[n - 1][a] = axis[a].currentPosition();
  calMask |= (1 << (n - 1));
  saveCalibration();

  Serial.print("Titik ");
  Serial.print(n);
  Serial.println(" disimpan:");
  for (int a = 0; a < AXIS_COUNT; a++) {
    Serial.print("  axis");
    Serial.print(a);
    Serial.print(" = ");
    Serial.print(calPos[n - 1][a]);
    Serial.print(" step (");
    Serial.print(stepsToDegree(a, calPos[n - 1][a]));
    Serial.println(" deg)");
  }
  if (calMask == 0x0F) Serial.println(">> 4 titik lengkap. Gunakan XY:x,y <<");
}

void printPoints() {
  const char* label[CAL_POINTS] = {"(0,0)", "(W,0)", "(W,H)", "(0,H)"};
  Serial.println();
  Serial.println("===== TITIK KALIBRASI =====");
  for (int p = 0; p < CAL_POINTS; p++) {
    Serial.print("Titik ");
    Serial.print(p + 1);
    Serial.print(" ");
    Serial.print(label[p]);
    if (!(calMask & (1 << p))) {
      Serial.println(" : belum disimpan");
      continue;
    }
    Serial.print(" : ");
    for (int a = 0; a < AXIS_COUNT; a++) {
      Serial.print(stepsToDegree(a, calPos[p][a]));
      Serial.print(a < AXIS_COUNT - 1 ? " | " : "");
    }
    Serial.println(" deg");
  }
  Serial.print("Area W x H = ");
  Serial.print(areaW);
  Serial.print(" x ");
  Serial.print(areaH);
  Serial.println(" mm");
}

void clearCalibration() {
  calMask = 0;
  for (int p = 0; p < CAL_POINTS; p++)
    for (int a = 0; a < AXIS_COUNT; a++) calPos[p][a] = 0;
  saveCalibration();
  Serial.println("Kalibrasi dihapus");
}

// Interpolasi bilinear posisi joint dari 4 titik sudut
void moveXY(float x, float y) {
  if (calMask != 0x0F) {
    Serial.println("Kalibrasi belum lengkap (butuh SAVE:1 sampai SAVE:4)");
    return;
  }
  if (!zeroCalibrated) {
    Serial.println("ZERO belum dibuat");
    return;
  }
  if (x < 0 || x > areaW || y < 0 || y > areaH) {
    Serial.print("XY di luar area (0..");
    Serial.print(areaW);
    Serial.print(", 0..");
    Serial.print(areaH);
    Serial.println(")");
    return;
  }

  float u = x / areaW;
  float v = y / areaH;

  float w1 = (1 - u) * (1 - v);   // titik 1 (0,0)
  float w2 = u * (1 - v);         // titik 2 (W,0)
  float w3 = u * v;               // titik 3 (W,H)
  float w4 = (1 - u) * v;         // titik 4 (0,H)

  long t[AXIS_COUNT];
  for (int a = 0; a < AXIS_COUNT; a++) {
    t[a] = lround(w1 * calPos[0][a] + w2 * calPos[1][a] +
                  w3 * calPos[2][a] + w4 * calPos[3][a]);
  }

  Serial.print("XY -> (");
  Serial.print(x);
  Serial.print(", ");
  Serial.print(y);
  Serial.println(")");

  if (commitMove(t)) {
    for (int a = 0; a < AXIS_COUNT; a++) {
      Serial.print("  axis");
      Serial.print(a);
      Serial.print(" -> ");
      Serial.print(stepsToDegree(a, t[a]));
      Serial.println(" deg");
    }
  }
}


// =========================================================================
// IK PLANAR Y-Z (mode XYZ lama)
// =========================================================================
bool calculatePlanarIK(float Y, float Z, float &J2, float &J4) {
  float z = Z - L1;
  float r = Y;
  float D = sqrt(r * r + z * z);

  if (D > (L2 + L3) || D < fabs(L2 - L3)) {
    Serial.println("IK OUT OF RANGE");
    return false;
  }

  float cosJ4 = (D * D - L2 * L2 - L3 * L3) / (2 * L2 * L3);
  cosJ4 = constrain(cosJ4, -1, 1);

  float elbow = acos(cosJ4) * RAD_TO_DEG;
  float alpha = atan2(z, r) * RAD_TO_DEG;
  float beta  = atan2(L3 * sin(elbow * DEG_TO_RAD),
                      L2 + L3 * cos(elbow * DEG_TO_RAD)) * RAD_TO_DEG;

  J2 = alpha - beta;
  J4 = elbow;
  return true;
}

void moveCartesianXYZ(float X, float Y, float Z) {
  if (X < MIN_X || X > MAX_X || Y < MIN_Y || Y > MAX_Y || Z < MIN_Z || Z > MAX_Z) {
    Serial.println("XYZ di luar workspace");
    return;
  }

  float J2, J4;
  if (!calculatePlanarIK(Y, Z, J2, J4)) return;   // hitung dulu, belum ada yang bergerak

  long t[AXIS_COUNT];
  getTargets(t);
  t[0] = degreeToSteps(0, X * J1_SCALE);
  t[1] = degreeToSteps(1, J2);
  t[2] = degreeToSteps(2, J4);
  t[4] = degreeToSteps(4, mapFloat(Z, MIN_Z, MAX_Z, Z_MIN_J6, Z_MAX_J6));

  if (commitMove(t)) {
    Serial.print("J2=");
    Serial.println(J2);
    Serial.print("J4=");
    Serial.println(J4);
  }
}


// =========================================================================
// PERINTAH JOINT MANUAL
//   J1:90        -> absolut 90 deg
//   J1:+10 /-10  -> relatif
//   J1:=-45      -> absolut negatif (pakai tanda '=')
//   Boleh banyak joint sekaligus: "J1:+5 J2:-3 J4:=20"
// =========================================================================
bool parseJointToken(String token, long target[AXIS_COUNT]) {
  token.trim();
  if (token.length() < 3) return false;
  if (token.charAt(0) != 'J' && token.charAt(0) != 'j') return false;

  int colon = token.indexOf(':');
  if (colon < 2) return false;

  int joint = token.substring(1, colon).toInt();
  int a = jointToAxis(joint);
  if (a < 0) return false;

  String v = token.substring(colon + 1);
  float deg;

  if (v.startsWith("="))                         deg = v.substring(1).toFloat();
  else if (v.startsWith("+") || v.startsWith("-")) deg = stepsToDegree(a, target[a]) + v.toFloat();
  else                                           deg = v.toFloat();

  target[a] = degreeToSteps(a, deg);

  Serial.print("J");
  Serial.print(joint);
  Serial.print(" -> ");
  Serial.print(deg);
  Serial.println(" deg");
  return true;
}

void handleJointLine(String cmd) {
  long t[AXIS_COUNT];
  getTargets(t);

  int start = 0;
  while (start < (int)cmd.length()) {
    int space = cmd.indexOf(' ', start);
    if (space == -1) space = cmd.length();
    parseJointToken(cmd.substring(start, space), t);
    start = space + 1;
  }
  commitMove(t);
}


// =========================================================================
// COMMAND HANDLER
// =========================================================================
void handleCommand(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;

  String up = cmd;
  up.toUpperCase();

  if (up == "ZERO")   { setZero();     return; }
  if (up == "HOME")   { moveHome();    return; }
  if (up == "STATUS") { printStatus(); return; }
  if (up == "POINTS") { printPoints(); return; }
  if (up == "CLEARCAL") { clearCalibration(); return; }

  if (up == "STOP") {
    for (int i = 0; i < AXIS_COUNT; i++) axis[i].stop();
    Serial.println("STOP");
    return;
  }

  if (up.startsWith("SAVE:")) {
    savePoint(up.substring(5).toInt());
    return;
  }

  if (up.startsWith("AREA:")) {
    float w, h;
    if (sscanf(up.c_str(), "AREA:%f,%f", &w, &h) == 2 && w > 0 && h > 0) {
      areaW = w;
      areaH = h;
      saveCalibration();
      Serial.print("Area = ");
      Serial.print(areaW);
      Serial.print(" x ");
      Serial.println(areaH);
    } else {
      Serial.println("Format: AREA:lebar,panjang  (mm)");
    }
    return;
  }

  // XY:x,y atau XY:x,y,servo
  if (up.startsWith("XY:")) {
    float x, y;
    int s = -1;
    int n = sscanf(up.c_str(), "XY:%f,%f,%d", &x, &y, &s);
    if (n >= 2) {
      if (n == 3) setServo(s);
      moveXY(x, y);
    } else {
      Serial.println("Format: XY:x,y  atau  XY:x,y,servo");
    }
    return;
  }

  if (up.startsWith("XYZ:")) {
    float x, y, z;
    if (sscanf(up.c_str(), "XYZ:%f,%f,%f", &x, &y, &z) == 3) moveCartesianXYZ(x, y, z);
    else Serial.println("Format XYZ salah");
    return;
  }

  if (up.startsWith("SERVO:")) {
    setServo(up.substring(6).toInt());
    return;
  }

  if (up.startsWith("SPEED:")) {
    float spd = up.substring(6).toFloat();
    if (spd > 0) {
      globalMaxSpeed = spd;
      Serial.print("Speed=");
      Serial.println(spd);
    }
    return;
  }

  if (up.startsWith("ACCEL:")) {
    float acc = up.substring(6).toFloat();
    if (acc > 0) {
      globalAccel = acc;
      Serial.print("Accel=");
      Serial.println(acc);
    }
    return;
  }

  if (up.charAt(0) == 'J') {
    handleJointLine(cmd);
    return;
  }

  Serial.println("Command tidak dikenal");
}


// =========================================================================
// SETUP & LOOP
// =========================================================================
void setup() {
  Serial.begin(115200);
  delay(300);

  for (int i = 0; i < AXIS_COUNT; i++) {
    axis[i].setMaxSpeed(globalMaxSpeed);
    axis[i].setAcceleration(globalAccel);
  }

  ESP32PWM::allocateTimer(0);
  gripperServo.setPeriodHertz(50);
  gripperServo.attach(SERVO_PIN, 500, 2400);

  loadAll();
  gripperServo.write(servoAngle);

  Serial.println();
  Serial.println("==============================================");
  Serial.println(" ESP32 ROBOT ARM - KALIBRASI 4 TITIK");
  Serial.println("==============================================");
  Serial.println("Jog manual : J1:+5  J2:-3  J4:=20  (boleh banyak joint)");
  Serial.println("Servo      : SERVO:90");
  Serial.println("Kalibrasi  : AREA:300,200  lalu jog -> SAVE:1 .. SAVE:4");
  Serial.println("             POINTS | CLEARCAL");
  Serial.println("Gerak XY   : XY:150,100  atau  XY:150,100,45 (dgn servo)");
  Serial.println("Lainnya    : ZERO | HOME | STATUS | STOP | SPEED:n | ACCEL:n");
  Serial.println("Mode IK    : XYZ:0,435,230");
  Serial.println("==============================================");
  printStatus();
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      serialBuffer.trim();
      if (serialBuffer.length() > 0) handleCommand(serialBuffer);
      serialBuffer = "";
    } else {
      serialBuffer += c;
    }
  }

  bool moving = false;
  for (int i = 0; i < AXIS_COUNT; i++) {
    axis[i].run();
    if (axis[i].distanceToGo() != 0) moving = true;
  }

  if (motionActive && !moving) {
    motionActive = false;
    savePosition();
    Serial.println("Motion selesai");
  }
}
