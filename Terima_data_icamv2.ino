#include <WiFi.h>
#include <ArduinoJson.h>

// ============================================================
// WIFI
// ============================================================
const char* ssid = "Lab Robotika AI";
const char* password = "labrobotm101";

// ============================================================
// TCP SERVER
// ============================================================
const uint16_t TCP_PORT = 5000;

WiFiServer server(TCP_PORT);

// ============================================================
// DATA DARI ICAM
// ============================================================
float targetX = 0.0;      // Koordinat target dalam mm
float targetY = 0.0;      // Koordinat target dalam mm

int targetG = 0;          // One-hot object/color
int targetR = 0;
int targetYellow = 0;

bool targetAvailable = false;  // Penanda data baru

// ============================================================
// HELPER: KIRIM ERROR KE CLIENT + SERIAL
// ============================================================
static void sendError(WiFiClient& client, const String& code, const String& message)
{
    Serial.println("[ERROR] " + message);
    client.println(code);
}

// ============================================================
// PROCESS DATA ICAM
//
// Payload dari ICAM (icam_xy_client.py), diakhiri '\n':
//   {"x":100.0,"y":50.0,"G":1,"R":0,"Y":0}
//
// x, y : float dalam mm
// G,R,Y: flag one-hot (huruf besar), tepat satu bernilai 1
//        GREEN =(1,0,0)
//        RED   =(0,1,0)
//        YELLOW=(0,0,1)
//
// Balasan: "OK" atau "ERROR_*"
// ============================================================
void processICAMData(const String& jsonString, WiFiClient& client)
{
    Serial.println();
    Serial.println(">>> DATA ICAM MASUK <<<");
    Serial.println("RAW JSON : " + jsonString);

    // --------------------------------------------------------
    // PARSE JSON
    // --------------------------------------------------------
    JsonDocument doc;

    DeserializationError error = deserializeJson(doc, jsonString);

    if (error)
    {
        Serial.println(String("[JSON ERROR] ") + error.c_str());
        client.println("ERROR_JSON");
        return;
    }

    // --------------------------------------------------------
    // CHECK FIELD WAJIB
    // --------------------------------------------------------
    if (!doc.containsKey("x"))
    {
        sendError(client, "ERROR_X", "Field X tidak ditemukan.");
        return;
    }

    if (!doc.containsKey("y"))
    {
        sendError(client, "ERROR_Y", "Field Y tidak ditemukan.");
        return;
    }

    if (!doc.containsKey("G"))
    {
        sendError(client, "ERROR_G", "Field G tidak ditemukan.");
        return;
    }

    if (!doc.containsKey("R"))
    {
        sendError(client, "ERROR_R", "Field R tidak ditemukan.");
        return;
    }

    if (!doc.containsKey("Y"))
    {
        sendError(client, "ERROR_YFLAG", "Field Y (flag warna) tidak ditemukan.");
        return;
    }

    // --------------------------------------------------------
    // READ DATA
    // --------------------------------------------------------
    targetX      = doc["x"].as<float>();
    targetY      = doc["y"].as<float>();
    targetG      = doc["G"].as<int>();
    targetR      = doc["R"].as<int>();
    targetYellow = doc["Y"].as<int>();

    // --------------------------------------------------------
    // VALIDATE ONE-HOT
    // --------------------------------------------------------
    if ((targetG != 0 && targetG != 1) ||
        (targetR != 0 && targetR != 1) ||
        (targetYellow != 0 && targetYellow != 1))
    {
        sendError(client, "ERROR_COLOR_VALUE", "Nilai G/R/YELLOW harus 0 atau 1.");
        return;
    }

    int activeObject = targetG + targetR + targetYellow;

    if (activeObject != 1)
    {
        sendError(client, "ERROR_ONE_HOT", "Hanya satu object boleh bernilai 1.");
        return;
    }

    // --------------------------------------------------------
    // DETERMINE OBJECT
    // --------------------------------------------------------
    String objectName = "UNKNOWN";

    if (targetG == 1)
    {
        objectName = "GREEN";
    }
    else if (targetR == 1)
    {
        objectName = "RED";
    }
    else if (targetYellow == 1)
    {
        objectName = "YELLOW";
    }

    targetAvailable = true;

    // --------------------------------------------------------
    // SERIAL MONITOR
    // --------------------------------------------------------
    Serial.println();
    Serial.println("================================");
    Serial.println("       TARGET DARI ICAM");
    Serial.println("================================");

    Serial.println("OBJECT : " + objectName);
    Serial.print("X      : ");
    Serial.print(targetX, 2);
    Serial.println(" mm");
    Serial.print("Y      : ");
    Serial.print(targetY, 2);
    Serial.println(" mm");
    Serial.println();

    Serial.println("G      : " + String(targetG));
    Serial.println("R      : " + String(targetR));
    Serial.println("YELLOW : " + String(targetYellow));
    Serial.println("================================");

    // --------------------------------------------------------
    // MOTOR / SMOOTHINGPITCH (BELUM DIJALANKAN OTOMATIS DI SINI)
    //
    // Nantinya:
    //
    // targetX + targetY
    //      -> transform koordinat robot
    //      -> SmoothingPitch / IK
    //
    // Object flag dipakai untuk menentukan tempat sorting:
    //   GREEN  -> tujuan hijau
    //   RED    -> tujuan merah
    //   YELLOW -> tujuan kuning
    // --------------------------------------------------------

    // --------------------------------------------------------
    // ACK
    // --------------------------------------------------------
    client.println("OK");
    Serial.println("[TX ACK] OK");
}

// ============================================================
// SETUP
// ============================================================
void setup()
{
    Serial.begin(115200);
    delay(1000);

    Serial.println();
    Serial.println("====================================");
    Serial.println(" ESP32 ICAM XY + OBJECT RECEIVER");
    Serial.println("====================================");

    // --------------------------------------------------------
    // WIFI
    // --------------------------------------------------------
    WiFi.mode(WIFI_STA);
    WiFi.begin(ssid, password);

    Serial.print("Connecting WiFi");

    while (WiFi.status() != WL_CONNECTED)
    {
        delay(500);
        Serial.print(".");
    }

    Serial.println();
    Serial.println("WiFi connected.");

    Serial.print("ESP32 IP : ");
    Serial.println(WiFi.localIP());

    Serial.print("Gateway  : ");
    Serial.println(WiFi.gatewayIP());

    Serial.print("Subnet   : ");
    Serial.println(WiFi.subnetMask());

    // --------------------------------------------------------
    // TCP SERVER
    // --------------------------------------------------------
    server.begin();
    server.setNoDelay(true);

    Serial.print("TCP Port : ");
    Serial.println(TCP_PORT);

    Serial.println("Waiting ICAM...");
}

// ============================================================
// LOOP
// ============================================================
void loop()
{
    WiFiClient client = server.available();

    if (!client)
    {
        return;
    }

    // --------------------------------------------------------
    // CLIENT CONNECTED
    // --------------------------------------------------------
    Serial.println();
    Serial.println("================================");
    Serial.println("[TCP] ICAM CONNECTED");

    Serial.print("Remote IP   : ");
    Serial.println(client.remoteIP());

    Serial.print("Remote Port : ");
    Serial.println(client.remotePort());
    Serial.println("================================");

    client.setNoDelay(true);

    // --------------------------------------------------------
    // READ DATA
    // --------------------------------------------------------
    while (client.connected())
    {
        if (client.available())
        {
            String line = client.readStringUntil('\n');
            line.trim();

            if (line.length() == 0)
            {
                continue;
            }

            Serial.println();
            Serial.println("[RX TCP] " + line);

            processICAMData(line, client);
        }

        delay(1);
    }

    // --------------------------------------------------------
    // DISCONNECTED
    // --------------------------------------------------------
    client.stop();

    Serial.println();
    Serial.println("[TCP] ICAM DISCONNECTED");
    Serial.println("Waiting new connection...");
}