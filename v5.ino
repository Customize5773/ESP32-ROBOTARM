import socket
import json
import time
import threading


ESP_IP = "192.168.67.174"   # GANTI dengan IP ESP32 yang tampil di Serial Monitor
ESP_PORT = 5000
STREAM_HZ = 10           # frame per detik, meniru kamera sungguhan

# Urutan penyajian objek default: GREEN -> RED -> YELLOW.
# Flag one-hot: GREEN=(G=1,R=0,Y=0), RED=(G=0,R=1,Y=0), YELLOW=(G=0,R=0,Y=1)
DEFAULT_ORDER = ["GREEN", "RED", "YELLOW"]

COLOR_TO_FLAGS = {
    "GREEN": (1, 0, 0),
    "G": (1, 0, 0),
    "RED": (0, 1, 0),
    "R": (0, 1, 0),
    "YELLOW": (0, 0, 1),
    "Y": (0, 0, 1),
}


def color_to_flags(color):
    """Ubah nama warna -> (G, R, Y) one-hot."""
    key = str(color).strip().upper()
    if key not in COLOR_TO_FLAGS:
        raise ValueError(f"Warna '{color}' tidak dikenal. Pakai GREEN/G, RED/R, YELLOW/Y.")
    return COLOR_TO_FLAGS[key]


def flags_to_color(g, r, y_flag):
    """Kebalikan color_to_flags: (G,R,Y) -> nama warna."""
    if (g, r, y_flag) == (1, 0, 0):
        return "GREEN"
    if (g, r, y_flag) == (0, 1, 0):
        return "RED"
    if (g, r, y_flag) == (0, 0, 1):
        return "YELLOW"
    raise ValueError(f"Flag G={g},R={r},Y={y_flag} bukan one-hot valid.")


def validate_flags(g, r, y_flag):
    """Pastikan tepat satu flag = 1 (one-hot)."""
    for v in (g, r, y_flag):
        if v not in (0, 1):
            raise ValueError(f"Flag G,R,Y harus 0/1, dapat G={g},R={r},Y={y_flag}")
    if (g + r + y_flag) != 1:
        raise ValueError(f"Flag harus one-hot (tepat satu =1), dapat G={g},R={r},Y={y_flag}")


def make_payload(x_mm, y_mm, color):
    """Bentuk dict JSON untuk satu frame 'ada objek'."""
    g, r, y_flag = color_to_flags(color)
    validate_flags(g, r, y_flag)
    return {
        "x": round(float(x_mm), 2),
        "y": round(float(y_mm), 2),
        "G": int(g),
        "R": int(r),
        "Y": int(y_flag),
    }


class ICAMXYClient:

    def __init__(self, host=ESP_IP, port=ESP_PORT):
        self.host = host
        self.port = port
        self.sock = None

    def connect(self):

        while True:

            try:
                print(f"[TCP] Connecting to {self.host}:{self.port} ...")

                self.sock = socket.socket(
                    socket.AF_INET,
                    socket.SOCK_STREAM
                )

                self.sock.settimeout(5)
                self.sock.connect((self.host, self.port))

                print("[TCP] Connected.")
                return

            except Exception as e:

                print("[TCP] Connection failed:", e)

                try:
                    if self.sock:
                        self.sock.close()
                except:
                    pass

                self.sock = None

                print("[TCP] Retry in 2 seconds...")
                time.sleep(2)

    # ------------------------------------------------------------------
    # Kirim 1 frame (dengan print lengkap). Hanya untuk tes koneksi:
    # 1 frame TIDAK cukup memicu robot (ESP32 butuh > 6 frame stabil).
    # ------------------------------------------------------------------
    def send_xy(self, x_mm, y_mm, g=0, r=0, y_flag=0, color=None):
        if color is not None:
            g, r, y_flag = color_to_flags(color)
        validate_flags(g, r, y_flag)

        payload = {
            "x": round(float(x_mm), 2),
            "y": round(float(y_mm), 2),
            "G": int(g),
            "R": int(r),
            "Y": int(y_flag)
        }

        message = json.dumps(payload, separators=(",", ":")) + "\n"

        try:
            self.sock.sendall(message.encode("utf-8"))

            print()
            print("[ICAM -> ESP32]")
            print(f"X = {x_mm:.2f} mm")
            print(f"Y = {y_mm:.2f} mm")
            print(f"G = {g}  R = {r}  Y = {y_flag}  ({flags_to_color(g, r, y_flag)})")
            print("JSON:", message.strip())

            response = self.sock.recv(128)

            if response:
                response_text = response.decode("utf-8").strip()
                print("[ESP32 -> ICAM]", response_text)
                return response_text == "OK"

            return False

        except Exception as e:
            print("[TCP] Send error:", e)
            try:
                self.sock.close()
            except:
                pass
            self.sock = None
            print("[TCP] Reconnecting...")
            self.connect()
            return False

    # ------------------------------------------------------------------
    # Kirim 1 frame tanpa banyak print (dipakai oleh streaming)
    # ------------------------------------------------------------------
    def _send_raw(self, payload):
        message = json.dumps(payload, separators=(",", ":")) + "\n"
        try:
            self.sock.sendall(message.encode("utf-8"))
            return self.sock.recv(128).decode("utf-8").strip()
        except Exception as e:
            print("[TCP] Send error:", e)
            try:
                self.sock.close()
            except:
                pass
            self.sock = None
            self.connect()
            return None

    # ------------------------------------------------------------------
    # Meniru kamera: kirim titik yang SAMA berulang-ulang
    # ------------------------------------------------------------------
    def stream_xy(self, x_mm, y_mm, color, seconds=2.0, hz=STREAM_HZ):
        payload = make_payload(x_mm, y_mm, color)
        n = max(1, int(seconds * hz))
        print(f"[STREAM] {color} ({x_mm:.1f}, {y_mm:.1f}) x{n} frame @ {hz} Hz")
        ok = 0
        for _ in range(n):
            if self._send_raw(payload) == "OK":
                ok += 1
            time.sleep(1.0 / hz)
        print(f"[STREAM] ACK OK: {ok}/{n}")
        return ok == n

    # ------------------------------------------------------------------
    # Meniru kamera: tidak ada objek di meja
    # ------------------------------------------------------------------
    def stream_none(self, seconds=2.0, hz=STREAM_HZ):
        n = max(1, int(seconds * hz))
        print(f"[STREAM] Meja kosong x{n} frame @ {hz} Hz")
        for _ in range(n):
            self._send_raw({"found": False})
            time.sleep(1.0 / hz)

    # ------------------------------------------------------------------
    # Emulator scene: objek disajikan satu per satu, terus di-stream.
    # ENTER = "robot sudah mengambil objek ini" -> objek berikutnya.
    # Setelah semua habis, otomatis kirim {"found": false}.
    # ------------------------------------------------------------------
    def run_scene(self, items, hz=STREAM_HZ):
        state = {"i": 0}
        stop = threading.Event()

        def worker():
            while not stop.is_set():
                i = state["i"]
                if i < len(items):
                    x, y, color = items[i]
                    payload = make_payload(x, y, color)
                else:
                    payload = {"found": False}
                self._send_raw(payload)
                time.sleep(1.0 / hz)

        t = threading.Thread(target=worker, daemon=True)
        t.start()

        print("\n[SCENE] Streaming dimulai. Ketik START di Serial Monitor ESP32.")
        for i, (x, y, color) in enumerate(items):
            input(f"[SCENE] Objek {i + 1}/{len(items)} {color} ({x}, {y}) sedang di-stream.\n"
                  f"        Tekan ENTER setelah robot MULAI mengangkatnya... ")
            state["i"] = i + 1

        input("[SCENE] Meja kosong. Tekan ENTER untuk berhenti streaming... ")
        stop.set()
        t.join()
        print("[SCENE] Selesai.")

    def close(self):

        try:
            if self.sock:
                self.sock.close()
        except:
            pass

        self.sock = None


def input_float(text):

    while True:

        try:
            return float(input(text))

        except ValueError:
            print("Masukkan angka.")


def input_color(text="Warna (G=GREEN/R=RED/Y=YELLOW) : "):

    while True:

        c = input(text).strip().upper()

        if c in COLOR_TO_FLAGS:
            if c in ("G", "GREEN"):
                return "GREEN"
            if c in ("R", "RED"):
                return "RED"
            return "YELLOW"

        print("Pilih G / R / Y (atau GREEN/RED/YELLOW).")


if __name__ == "__main__":

    icam = ICAMXYClient()
    icam.connect()

    while True:

        print()
        print("==============================")
        print(" ICAM-300 -> ESP32 XY+G,R,Y")
        print("==============================")
        print("1 = Kirim 1 frame (tes koneksi saja)")
        print("2 = Stream 1 titik (ambil 1 objek)")
        print("3 = Scene G,R,Y (emulator kamera, 3 objek)")
        print("4 = Stream 'meja kosong'")
        print("q = Quit")

        choice = input("> ").strip().lower()

        if choice == "1":

            x_mm = input_float("X (mm) : ")
            y_mm = input_float("Y (mm) : ")
            color = input_color()
            icam.send_xy(x_mm, y_mm, color=color)

        elif choice == "2":

            print("\nKetik START di Serial Monitor ESP32 DULU, lalu isi data ini.")
            x_mm = input_float("X (mm) : ")
            y_mm = input_float("Y (mm) : ")
            color = input_color()
            icam.stream_xy(x_mm, y_mm, color, seconds=3.0)

        elif choice == "3":

            print(f"\nScene urutan {DEFAULT_ORDER}. Masukkan 3 koordinat:")
            items = []
            for color in DEFAULT_ORDER:
                print(f"\n-- Objek {color} : G,R,Y = {color_to_flags(color)} --")
                x_mm = input_float(f"X {color} (mm) : ")
                y_mm = input_float(f"Y {color} (mm) : ")
                items.append((x_mm, y_mm, color))

            icam.run_scene(items)

        elif choice == "4":

            icam.stream_none(seconds=3.0)

        elif choice == "q":

            icam.close()
            print("\nConnection closed.")
            break

        else:

            print("Pilihan tidak dikenal.")