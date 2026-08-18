// modbus_slave_test.ino
// Minimal Modbus RTU slave for ESP32 + MAX485
// Slave ID = 1, Function 03 (Read Holding Registers) supported

#define RXD2 16
#define TXD2 17
#define DE_RE_PIN 4      // tied DE+RE on MAX485
#define SLAVE_ID 1
#define BAUD 9600         // start slow & reliable; can raise to 115200 later

uint16_t holdingRegisters[10] = {1234, 5678, 100, 200, 300, 400, 500, 600, 700, 800};

uint8_t buf[256];
uint8_t bufLen = 0;
unsigned long lastByteTime = 0;

uint16_t modbusCRC16(uint8_t *data, uint8_t len) {
  uint16_t crc = 0xFFFF;
  for (uint8_t pos = 0; pos < len; pos++) {
    crc ^= (uint16_t)data[pos];
    for (uint8_t i = 8; i != 0; i--) {
      if (crc & 0x0001) {
        crc >>= 1;
        crc ^= 0xA001;
      } else {
        crc >>= 1;
      }
    }
  }
  return crc;
}

void setup() {
  Serial.begin(115200);          // USB debug monitor
  Serial2.begin(BAUD, SERIAL_8N1, RXD2, TXD2);
  pinMode(DE_RE_PIN, OUTPUT);
  digitalWrite(DE_RE_PIN, LOW);  // start in receive mode

  Serial.println("=== ESP32 Modbus RTU Slave Test ===");
  Serial.println("Slave ID: 1, Baud: 9600, Waiting for Modbus Poll...");
}

void loop() {
  // Collect incoming bytes
  while (Serial2.available()) {
    if (bufLen < sizeof(buf)) {
      buf[bufLen++] = Serial2.read();
    } else {
      Serial2.read(); // discard overflow
    }
    lastByteTime = millis();
  }

  // Frame considered complete after 10ms silence
  if (bufLen > 0 && (millis() - lastByteTime > 10)) {
    processFrame();
    bufLen = 0;
  }
}

void processFrame() {
  if (bufLen < 8) return; // too short for function 03 request

  uint8_t slaveId = buf[0];
  uint8_t function = buf[1];

  if (slaveId != SLAVE_ID) return; // not for us

  // Verify CRC
  uint16_t recvCRC = buf[bufLen - 2] | (buf[bufLen - 1] << 8);
  uint16_t calcCRC = modbusCRC16(buf, bufLen - 2);
  if (recvCRC != calcCRC) {
    Serial.println("CRC error, ignoring frame");
    return;
  }

  Serial.print("Received function: 0x");
  Serial.println(function, HEX);

  if (function == 0x03) {
    uint16_t startAddr = (buf[2] << 8) | buf[3];
    uint16_t qty = (buf[4] << 8) | buf[5];

    if (startAddr + qty > 10) {
      sendException(function, 0x02); // illegal data address
      return;
    }

    uint8_t response[5 + 20];
    response[0] = SLAVE_ID;
    response[1] = 0x03;
    response[2] = qty * 2; // byte count

    for (int i = 0; i < qty; i++) {
      response[3 + i*2] = holdingRegisters[startAddr + i] >> 8;
      response[4 + i*2] = holdingRegisters[startAddr + i] & 0xFF;
    }

    uint16_t crc = modbusCRC16(response, 3 + qty*2);
    response[3 + qty*2] = crc & 0xFF;
    response[4 + qty*2] = crc >> 8;

    sendResponse(response, 5 + qty*2);
    Serial.println("Sent holding registers response");
  }
}

void sendException(uint8_t function, uint8_t code) {
  uint8_t response[5];
  response[0] = SLAVE_ID;
  response[1] = function | 0x80;
  response[2] = code;
  uint16_t crc = modbusCRC16(response, 3);
  response[3] = crc & 0xFF;
  response[4] = crc >> 8;
  sendResponse(response, 5);
}

void sendResponse(uint8_t *data, uint8_t len) {
  digitalWrite(DE_RE_PIN, HIGH); // switch to transmit
  delayMicroseconds(50);
  Serial2.write(data, len);
  Serial2.flush();
  delayMicroseconds(50);
  digitalWrite(DE_RE_PIN, LOW);  // back to receive
}