# Uji integrasi sebelum hardware

Target tahap pertama: web/mobile -> Laravel -> broker MQTT -> simulator ->
Laravel (`mqtt:listen`) -> web/mobile. Simulator meniru satu pose, bukan
siklus pick-and-place atau sensor keberhasilan penempatan.

## Menjalankan simulator (PowerShell)

Jalankan dari root repository:

```powershell
python -m venv .venv-sim
.\.venv-sim\Scripts\python.exe -m pip install -r tools/requirements.txt
.\.venv-sim\Scripts\python.exe tools/arm_simulator.py --host 127.0.0.1
```

Broker harus sudah berjalan. Ganti `127.0.0.1` dengan IP broker jika berada
di komputer lain. Kredensial opsional dibaca dari environment `MQTT_USERNAME`
dan `MQTT_PASSWORD`; jangan masukkan password ke Git. Gunakan `--tls` dan port
broker yang sesuai bila broker membutuhkan TLS.

Default topic simulator adalah `arm-sim/command` dan `arm-sim/status`.
Simulator tidak menerbitkan pesan ke topic command.

Pada instance backend untuk pengujian, isi `.env`:

```dotenv
MQTT_HOST=127.0.0.1
MQTT_PORT=1883
MQTT_BASE_TOPIC=arm-sim
```

Sesuaikan host/port dan kredensial dengan broker yang sama. Di folder backend:

```powershell
php artisan config:clear
php artisan mqtt:listen
```

Restart listener setelah mengubah konfigurasi. Jalankan juga backend HTTP dan
database. Pastikan kategori yang dipilih memiliki TargetZonePreset. Pada mobile,
`EXPO_PUBLIC_API_URL` harus menuju URL backend yang bisa diakses HP, berakhiran
`/api`. Jika memakai MQTT WebSocket mobile, samakan prefix topic melalui
konfigurasi mobile; port WebSocket mengikuti konfigurasi broker, bukan port TCP.

Jika backend pengujian harus memakai `arm`, jalankan simulator dengan
`--base-topic arm` hanya ketika ESP32 sudah terputus dari broker. Dua penerima
akan menerima command yang sama dan dua pengirim status akan saling menimpa
status singleton backend.

## Kontrak yang kompatibel sekarang

Topic command, JSON, tidak retained:

```json
{
  "category": "Food & Beverage",
  "zone": "food-beverage",
  "joint_angles": [10, 20, 30, 40, 50, 60],
  "issued_at": "2026-09-22T12:00:00+07:00",
  "command_id": "demo-001"
}
```

Urutan sudut: base, shoulder, elbow, wrist roll, wrist pitch, gripper; satuan
derajat. Angka contoh hanya untuk simulator, bukan preset robot terkalibrasi.
`command_id` adalah tambahan opsional: backend saat ini dapat meneruskannya
melalui `context.command_id` pada POST `/api/arm/command`.

Simulator menerima 6 angka finite; gripper 0..180. Batas mekanis lima joint
belum diketahui sehingga tidak diasumsikan. Backend tetap mengirim perintah
legacy tanpa command_id; simulator memakai hash seluruh payload untuk dedup.
Cache dedup terbatas pada 256 perintah diterima selama satu proses, tidak
bertahan setelah restart. Payload sama persis dianggap duplikat; gunakan ID
baru untuk pekerjaan baru.

Status tetap memakai `state`, `detail`, `last_command`, dan `telemetry` seperti
firmware/backend sekarang. `state` adalah `running`, `idle`, atau `error`.
`telemetry.simulated=true` dan `telemetry.event` memberi konteks tambahan:
`ready`, `accepted`, `completed`, `failed`, `busy`, `duplicate`, `rejected`,
atau `offline`. Field tambahan ini belum tentu ditampilkan UI saat ini.

Perintah valid: running -> idle setelah 3 detik. Posisi target ditampilkan
hanya setelah simulasi selesai. Heartbeat setiap 2 detik. Perintah saat sibuk
ditolak, tidak diantrikan. Penolakan tidak mengganti last_command atau status
pekerjaan aktif. Last Will memakai error/offline; kehilangan koneksi baru
terlihat setelah broker mendeteksinya. Backend/UI masih perlu menilai umur
`reported_at` untuk status stale.

## Skenario penerimaan

1. Kirim command dari web atau mobile. Log simulator menunjukkan accepted,
   lalu completed; `last_command` di GET `/api/arm` sesuai perintah.
2. Selama running, kirim command kedua. Simulator melaporkan busy dan tetap
   menyelesaikan command pertama. Respons HTTP sukses hanya berarti publish.
3. Kirim ulang command_id yang sama. Tidak ada eksekusi kedua.
4. Jalankan simulator dengan `--fail`. Command berakhir error tanpa mengubah
   posisi menjadi target. Pastikan UI menunjukkan kegagalan.
5. Hentikan simulator dengan Ctrl+C. Periksa status offline/error. Restart
   simulator dan pastikan heartbeat kembali. Uji broker restart juga;
   listener Laravel mungkin perlu dihidupkan ulang.
6. Catat hasil pada tabel di bawah. Jangan mencatat pengujian simulasi sebagai
   keberhasilan gerakan fisik.

| Uji | Hasil yang diharapkan | Hasil aktual |
| --- | --- | --- |
| Web -> simulator | accepted -> completed | Belum diuji lewat broker |
| HP -> simulator | accepted -> completed | Belum diuji lewat broker |
| Busy | Command aktif tidak diganti | Belum diuji lewat broker |
| Duplikat | Tidak dieksekusi ulang | Belum diuji lewat broker |
| Gagal | error, posisi tidak berubah | Belum diuji lewat broker |
| Putus/reconnect | Status terputus lalu pulih | Belum diuji lewat broker |

Tes logika tanpa instalasi dependency atau broker:

```powershell
python -m unittest discover -s tools -p "test_*.py" -v
```

## Tahap hardware berikutnya

- Sepakati posisi awal, arah dan batas tiap joint dengan tim robot.
- Kalibrasikan urutan mendekat, menjepit, mengangkat, memindahkan, melepas,
  kembali; kontrak satu pose belum menjalankan urutan itu.
- Tambahkan validasi, penanganan busy dan dedup ke firmware sebelum menganggap
  perilakunya sama dengan simulator. Firmware belum diubah pada tahap ini.
- Buktikan frame ICAM nyata sampai ke layanan ML dan tampilkan mode sumber.
- Cegah inferensi dari simulator memicu hardware; kunci satu pekerjaan per
  objek, lalu lanjutkan setelah pekerjaan selesai dan objek berikutnya siap.
- Jalankan pengujian integrasi fisik dan rekam hasilnya terpisah.

Implementasi client simulator mengikuti API Paho MQTT Python 2:
https://eclipse.dev/paho/files/paho.mqtt.python/html/client.html
