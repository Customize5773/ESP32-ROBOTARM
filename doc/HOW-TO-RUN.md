# How to Run — ESP32 Arm Robot Firmware

Panduan singkat menjalankan firmware ini dari nol sampai lengan robot bergerak lewat MQTT.

## 1. Prasyarat

- Board ESP32 (esp32dev) + kabel USB.
- 5x driver stepper A4988/DRV8825 + 5x NEMA17, sesuai wiring [README.md](README.md#wiring).
- 1x servo gripper (SG90/MG90S).
- Power rail motor terpisah (jangan dari 5V/3V3 ESP32).
- Broker MQTT yang sudah jalan (Mosquitto/EMQX), dan tahu host/port/topic-nya.
- [PlatformIO](https://platformio.org/) — CLI atau VS Code extension.

## 2. Clone & buka project

```bash
git clone https://github.com/Customize5773/ESP32-ROBOTARM.git
cd ESP32-ROBOTARM
```

Buka folder ini di VS Code (dengan ekstensi PlatformIO terpasang) atau kerjakan lewat CLI `pio`.

## 3. Buat config.h

`src/config.h` di-gitignore karena berisi kredensial. Salin dari contoh:

```bash
cp src/config.h.example src/config.h
```

Edit `src/config.h`:

- `WIFI_SSID`, `WIFI_PASSWORD` — kredensial WiFi ESP32.
- `MQTT_HOST`, `MQTT_PORT`, `MQTT_BASE_TOPIC` — **harus sama** dengan setelan broker di backend Laravel (`Innowork-dashboard-app-v1/config/services.php`, blok `mqtt`).
- `MQTT_USERNAME` / `MQTT_PASSWORD` — isi kalau broker butuh auth, kosongkan kalau anonymous.
- Pin mapping (`BASE_STEP_PIN`, dst.) — ubah kalau wiring board Anda beda dari default.
- `STEPS_PER_DEGREE` / `DEFAULT_STEPS_PER_DEGREE` — sesuaikan kalau gearbox/pulley axis Anda punya rasio reduksi (default asumsi NEMA17 1.8°, microstepping 1/16, tanpa reduksi).

## 4. Wiring

Ikuti tabel pin di [README.md § Wiring](README.md#wiring). Poin penting:

- GND common wajib antara ESP32 dan semua driver board.
- Power motor (VMOT) dari rail terpisah, bukan dari ESP32.
- Servo gripper idealnya disuplai 5V eksternal, bukan pin 5V ESP32 langsung.

## 5. Build & flash

```bash
pio run                 # compile, cek error dulu
pio run -t upload       # flash ke ESP32 (sambungkan via USB)
pio device monitor      # lihat log serial: WiFi connect, MQTT connect, command masuk
```

Atau via VS Code: tombol **Upload** dan **Monitor** di PlatformIO toolbar.

## 6. Uji tanpa backend Laravel (opsional)

Publish langsung ke broker MQTT untuk tes gerakan tanpa backend:

```bash
mosquitto_pub -h <MQTT_HOST> -t "arm/command" -m '{
  "category": "Food & Beverage",
  "zone": "food-beverage",
  "joint_angles": [10, 20, 30, 40, 50, 60],
  "issued_at": "2026-08-01T12:00:00+00:00"
}'
```

Lalu subscribe ke status untuk lihat telemetrinya:

```bash
mosquitto_sub -h <MQTT_HOST> -t "arm/status"
```

Kontrak lengkap payload command/status ada di [README.md § Kontrak MQTT](README.md#kontrak-mqtt).

## 7. Integrasi dengan backend Laravel

Kalau menjalankan bersama `Innowork-dashboard-app-v1`, pastikan `.env` backend (`MQTT_HOST`, `MQTT_PORT`, `MQTT_BASE_TOPIC`) identik dengan `src/config.h` firmware ini, lalu jalankan `MqttListen` command di backend untuk menerima status ESP32.

## 8. Sebelum uji fisik penuh

- **Belum ada homing/limit switch** — firmware asumsi posisi start = 0 step semua axis saat boot. Pastikan posisi fisik arm memang di titik nol sebelum power on.
- Uji gerakan axis satu-satu dengan voltase driver rendah atau motor dilepas dari mekanik dulu, sebelum dipasang penuh ke lengan.
- Repo ini belum pernah di-compile/flash ke hardware fisik sebelumnya — jalankan `pio run` untuk menangkap error compile lebih dulu.

## Troubleshooting cepat

| Gejala | Kemungkinan penyebab |
| --- | --- |
| ESP32 tidak connect WiFi | `WIFI_SSID`/`WIFI_PASSWORD` salah, sinyal lemah |
| ESP32 connect WiFi tapi tidak connect MQTT | `MQTT_HOST`/`MQTT_PORT` salah, broker tidak reachable dari jaringan ESP32, auth salah |
| Command diterima tapi arm tidak bergerak | Payload `joint_angles` bukan 6 elemen (ditolak, state `error`), cek `pio device monitor` |
| Motor bergetar tapi tidak berputar | `STEPS_PER_DEGREE` atau arah DIR pin salah, cek wiring |
| Servo tidak bergerak | `GRIPPER_SERVO_PIN` salah, servo tidak dapat power terpisah |
