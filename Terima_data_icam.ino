#include <WiFi.h>
#include <ArduinoJson.h>
const char* ssid = "Lab Robotika AI";
const char* password = "labrobotm101";

const uint16_t TCP_PORT = 5000;

WiFiServer server(
  TCP_PORT);

// ============================================================
// TARGET DARI ICAM
//
// Unit:
// X = mm
// Y = mm
// ============================================================

float targetX = 0.0;

float targetY = 0.0;


// ============================================================
// STATUS
// ============================================================

bool targetAvailable =
  false;


// ============================================================
// PROCESS DATA ICAM
// ============================================================

void processICAMData(
  String jsonString,
  WiFiClient& client) {

  JsonDocument doc;


  // --------------------------------------------------------
  // PARSE JSON
  // --------------------------------------------------------

  DeserializationError error = deserializeJson(doc,jsonString);


  if (error) {

    Serial.print("[JSON ERROR] ");

    Serial.println(error.c_str());

    client.println( "ERROR");

    return;
  }


  // --------------------------------------------------------
  // WAJIB ADA X DAN Y
  // --------------------------------------------------------

  if (
    !doc["x"].is<float>()
    && !doc["x"].is<int>()
    && !doc["x"].is<long>()) {

    Serial.println(
      "[ERROR] Data X tidak ditemukan.");

    client.println(
      "ERROR");

    return;
  }


  if (
    !doc["y"].is<float>()
    && !doc["y"].is<int>()
    && !doc["y"].is<long>()) {

    Serial.println(
      "[ERROR] Data Y tidak ditemukan.");

    client.println(
      "ERROR");

    return;
  }


  // --------------------------------------------------------
  // AMBIL X,Y
  // --------------------------------------------------------

  targetX =
    doc["x"].as<float>();

  targetY =
    doc["y"].as<float>();


  targetAvailable =
    true;


  // --------------------------------------------------------
  // SERIAL MONITOR
  // --------------------------------------------------------

  Serial.println();

  Serial.println(
    "================================");

  Serial.println(
    "      TARGET DARI ICAM");

  Serial.println(
    "================================");


  Serial.print(
    "X : ");

  Serial.print(
    targetX,
    2);

  Serial.println(
    " mm");


  Serial.print(
    "Y : ");

  Serial.print(
    targetY,
    2);

  Serial.println(
    " mm");


  Serial.println(
    "================================");


  // --------------------------------------------------------
  // ACK KE ICAM
  // --------------------------------------------------------

  client.println(
    "OK");
}


// ============================================================
// SETUP
// ============================================================

void setup() {

  Serial.begin(
    115200);


  delay(
    1000);


  Serial.println();

  Serial.println(
    "================================");

  Serial.println(
    " ESP32 ICAM XY RECEIVER");

  Serial.println(
    "================================");


  // --------------------------------------------------------
  // WIFI
  // --------------------------------------------------------

  WiFi.mode(
    WIFI_STA);


  WiFi.begin(
    ssid,
    password);


  Serial.print(
    "Connecting WiFi");


  while (
    WiFi.status()
    != WL_CONNECTED) {

    delay(
      500);

    Serial.print(
      ".");
  }


  Serial.println();

  Serial.println(
    "WiFi connected.");


  Serial.print(
    "ESP32 IP : ");

  Serial.println(
    WiFi.localIP());


  // --------------------------------------------------------
  // TCP SERVER
  // --------------------------------------------------------

  server.begin();


  Serial.print(
    "TCP Port : ");

  Serial.println(
    TCP_PORT);


  Serial.println(
    "Waiting ICAM...");
}


// ============================================================
// LOOP
// ============================================================

void loop() {

  WiFiClient client =
    server.available();


  if (!client) {

    return;
  }


  Serial.println();

  Serial.println(
    "[TCP] ICAM CONNECTED");


  // --------------------------------------------------------
  // SELAMA CLIENT TERHUBUNG
  // --------------------------------------------------------

  while (
    client.connected()) {

    if (
      client.available()) {

      String line =
        client.readStringUntil(
          '\n');


      line.trim();


      if (line.length()== 0) {
        continue;
      }

      Serial.println();
      Serial.print("[RX] ");
      Serial.println(line);
      processICAMData(line,client);
    }
    delay(1);
  }

  // --------------------------------------------------------
  // DISCONNECT
  // --------------------------------------------------------
  client.stop();
  Serial.println(
    "[TCP] ICAM DISCONNECTED");
}