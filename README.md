# ESP32 Arm Robot Firmware — SortVision

Firmware ESP32 untuk lengan robot 6-axis (acuan BCN3D Moveo: 5x stepper
NEMA17 + 1x servo gripper), bagian dari arsitektur **Opsi A tanpa Jetson
Nano** yang didokumentasikan di `Opsi-a&b.md` (repo
`Innowork-dashboard-app-v1`):

```
Mobile app / Web dashboard --(REST+MQTT)--> Laravel backend + broker MQTT --(MQTT, WiFi)--> ESP32 (firmware ini) --> Stepper + servo motor
```

ESP32 **tidak** menghitung inverse kinematics — backend Laravel sudah
mengirim `joint_angles` (derajat) siap pakai lewat topic MQTT
`<base_topic>/command`. Firmware ini hanya menerjemahkan derajat → step
pulses dan menggerakkan motor.

> **Repo ini ditulis di sandbox tanpa akses hardware fisik** — belum
> di-compile/flash/uji ke ESP32 nyata. Source code sudah lengkap dan siap
> pakai, tapi Anda perlu membangun & mem-flash-nya sendiri, lalu verifikasi
> gerakan fisik di robot yang sebenarnya.

## Cara pakai / push ke repo Anda

Repo target: `Customize5773/ESP32-ROBOTARM` (baru dibuat, masih kosong).
Salin seluruh isi folder ini ke situ, misalnya:

```bash
git clone https://github.com/Customize5773/ESP32-ROBOTARM.git
cp -r esp32-armrobot-firmware/* esp32-armrobot-firmware/.gitignore ESP32-ROBOTARM/
cd ESP32-ROBOTARM
git add -A
git commit -m "feat: firmware ESP32 arm robot (stepper + servo, MQTT)"
git push
```

## Uji software sebelum hardware

Lihat [panduan integrasi dan simulator MQTT](INTEGRATION-QUICKSTART.md) untuk
menguji perintah web/mobile serta feedback robot tanpa motor terhubung.

## Wiring

| Axis          | Driver | STEP pin | DIR pin | ENABLE pin |
| ------------- | ------ | -------- | ------- | ---------- |
| Base          | A4988/DRV8825 | GPIO25 | GPIO26 | GPIO27 |
| Shoulder      | A4988/DRV8825 | GPIO14 | GPIO13 | GPIO16 |
| Elbow         | A4988/DRV8825 | GPIO17 | GPIO18 | GPIO19 |
| Wrist roll    | A4988/DRV8825 | GPIO21 | GPIO22 | GPIO23 |
| Wrist pitch   | A4988/DRV8825 | GPIO32 | GPIO33 | GPIO4  |
| Gripper servo | SG90/MG90S PWM | — | — | GPIO5 (signal) |

- Semua driver stepper: VMOT + GND ke rail power motor bersama (bukan dari
  ESP32 — jangan menyuplai motor dari 5V/3V3 board), logic VDD dari ESP32
  3.3V, GND common antara ESP32 dan driver board wajib disambung.
- Servo gripper: VCC ke 5V eksternal (jangan dari pin 5V ESP32 langsung kalau
  servo menarik arus besar saat stall), signal ke GPIO5, GND common.
- Pin dipilih menghindari strapping pin ESP32 (GPIO0, 2, 12, 15) dan pin
  input-only (GPIO34-39) supaya boot tidak terganggu.
- Ganti pin mapping di `src/config.h` kalau wiring board Anda berbeda.

## Build & flash

1. Install [PlatformIO](https://platformio.org/) (VS Code extension atau CLI).
2. `cp src/config.h.example src/config.h` lalu isi:
   - `WIFI_SSID` / `WIFI_PASSWORD`
   - `MQTT_HOST` / `MQTT_PORT` / `MQTT_BASE_TOPIC` — **harus sama** dengan
     `MQTT_HOST` / `MQTT_PORT` / `MQTT_BASE_TOPIC` di `.env` backend Laravel
     (`Innowork-dashboard-app-v1/config/services.php`, blok `mqtt`), supaya
     ESP32 dan backend bicara ke broker & topic yang sama.
   - Sesuaikan `STEPS_PER_DEGREE` per axis kalau gearbox/pulley Anda beda
     dari asumsi default (NEMA17 1.8°, microstepping 1/16, tanpa reduksi).
3. `pio run -t upload` (atau tombol Upload di VS Code) dengan ESP32
   tersambung via USB.
4. `pio device monitor` untuk lihat log serial (WiFi connect, MQTT connect,
   command diterima).

## Kontrak MQTT

**Subscribe** `<base_topic>/command` (default topic: `arm/command`), payload
dari `App\Services\ArmMqttService::buildCommandPayload()`:

```jsonc
{
  "category": "Food & Beverage",
  "zone": "food-beverage",
  "joint_angles": [10, 20, 30, 40, 50, 60], // derajat: [base, shoulder, elbow, wristRoll, wristPitch, gripperServo]
  "issued_at": "2026-08-01T12:00:00+00:00"
}
```

Command dengan `joint_angles` yang hilang atau bukan 6 elemen ditolak (state
`error`) — arm tidak bergerak sebagian saja untuk payload yang salah bentuk.

**Publish** `<base_topic>/status` (default: `arm/status`), dibaca oleh
`App\Console\Commands\MqttListen::handleStatus()`:

```jsonc
{
  "state": "idle" | "running" | "error",
  "detail": "Menjalankan command baru",
  "last_command": { /* payload command terakhir, diteruskan apa adanya */ },
  "telemetry": {
    "positions_deg": [10, 20, 30, 40, 50, 60], // posisi axis saat ini
    "rssi": -55
  }
}
```

Dipublish saat state berubah (idle→running→idle/error) dan sebagai heartbeat
tiap `STATUS_HEARTBEAT_MS` (default 2 detik) supaya dashboard tahu ESP32
masih online.

## Perilaku gerakan

- Kelima stepper bergerak **bersamaan** (non-blocking, `AccelStepper::run()`
  dipanggil tiap axis di setiap iterasi `loop()`), bukan satu-satu berurutan
  — sesuai gerakan arm 6-axis yang wajar.
  Servo gripper langsung `write()` ke sudut target (gerakan servo instan,
  tidak melalui profil accel/decel seperti stepper).
- Driver stepper (pin ENABLE, active-LOW) di-energize hanya saat ada command
  berjalan, lalu di-de-energize begitu semua axis berhenti — mengurangi
  panas/arus diam saat arm idle. Kalau arm Anda perlu menahan posisi (holding
  torque) terus-menerus, ubah `setAllEnabled(false)` di akhir command supaya
  tidak dipanggil.

## Yang belum ada / batasan sandbox ini

- Belum ada homing/limit switch — firmware asumsi posisi start = 0 step di
  semua axis saat boot. Tambahkan endstop + `AccelStepper::runToNewPosition`
  homing routine kalau arm fisik Anda butuh referensi absolut.
- Belum di-compile/flash ke hardware nyata — lakukan `pio run` dulu untuk cek
  error compile, lalu uji gerakan fisik axis demi axis dengan voltase driver
  rendah/motor lepas dari mekanik sebelum dipasang penuh.
