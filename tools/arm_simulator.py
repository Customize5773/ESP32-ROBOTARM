"""MQTT pose simulator; never publishes commands or accesses motor hardware."""
import argparse
import hashlib
import json
import math
import os
import queue
import time
from collections import OrderedDict


class ArmSimulator:
    def __init__(self, duration=3.0, fail=False):
        self.duration = duration
        self.fail = fail
        self.state = "idle"
        self.detail = "SIMULATOR siap"
        self.last_command = None
        self.positions = [0, 0, 0, 0, 0, 90]
        self.deadline = None
        self.seen = OrderedDict()
        self.event = "ready"

    def snapshot(self):
        return {"state": self.state, "detail": self.detail,
                "last_command": self.last_command,
                "telemetry": {"simulated": True, "positions_deg": list(self.positions),
                              "event": self.event}}

    def receive(self, payload, now, retained=False):
        try:
            if retained:
                raise ValueError("retained command ditolak")
            if len(payload) > 8192:
                raise ValueError("payload melebihi 8192 byte")
            data = json.loads(payload)
            if not isinstance(data, dict):
                raise ValueError("payload harus object JSON")
            angles = data.get("joint_angles")
            if not isinstance(angles, list) or len(angles) != 6:
                raise ValueError("joint_angles harus 6 angka")
            if any(type(x) not in (float, int) or not math.isfinite(x) for x in angles):
                raise ValueError("sudut harus angka finite, bukan string/bool")
            if not 0 <= angles[5] <= 180:
                raise ValueError("gripper harus 0..180 derajat")
            command_id = data.get("command_id")
            if command_id is not None and (not isinstance(command_id, str) or not command_id.strip()):
                raise ValueError("command_id harus string nonkosong")
            canonical = json.dumps(data, sort_keys=True, allow_nan=False)
            key = command_id or hashlib.sha256(canonical.encode()).hexdigest()
        except (ValueError, TypeError, UnicodeError, OverflowError) as exc:
            return self.reject(str(exc))
        if key in self.seen:
            return self.reject("command duplikat", "duplicate")
        if self.deadline is not None:
            return self.reject("robot sibuk", "busy")
        self.seen[key] = True
        if len(self.seen) > 256:
            self.seen.popitem(last=False)
        self.last_command = data
        self.state = "running"
        self.detail = "SIMULATOR menerima pose; menunggu durasi simulasi"
        self.event = "accepted"
        self.deadline = now + self.duration
        return self.snapshot()

    def reject(self, reason, event="rejected"):
        # Rejection must not replace the active job or its eventual completion.
        result = self.snapshot()
        result["detail"] = "SIMULATOR: " + reason
        result["telemetry"]["event"] = event
        return result

    def tick(self, now):
        if self.deadline is None or now < self.deadline:
            return None
        self.deadline = None
        self.state = "error" if self.fail else "idle"
        self.event = "failed" if self.fail else "completed"
        self.detail = "SIMULATOR gagal (uji error)" if self.fail else "SIMULATOR pose selesai"
        if not self.fail:
            self.positions = list(self.last_command["joint_angles"])
        return self.snapshot()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.getenv("MQTT_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MQTT_PORT", "1883")))
    parser.add_argument("--base-topic", default="arm-sim")
    parser.add_argument("--duration", type=float, default=3)
    parser.add_argument("--fail", action="store_true", help="Setiap pose berakhir error")
    parser.add_argument("--tls", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("duration harus positif dan finite")
    base = args.base_topic.strip("/")
    if not base or any(c in base for c in ("+", "#", "\x00")):
        parser.error("base-topic tidak boleh kosong atau mengandung wildcard")
    try:
        import paho.mqtt.client as mqtt
    except ImportError:
        parser.exit(1, "Install dependency: python -m pip install -r tools/requirements.txt\n")
    simulator = ArmSimulator(args.duration, args.fail)
    incoming = queue.Queue(maxsize=100)
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                         client_id=f"sortvision-simulator-{os.getpid()}")
    username = os.getenv("MQTT_USERNAME")
    if username:
        client.username_pw_set(username, os.getenv("MQTT_PASSWORD"))
    if args.tls:
        client.tls_set()
    client.reconnect_delay_set(min_delay=1, max_delay=10)

    def publish(status):
        payload = json.dumps(status, allow_nan=False)
        result = client.publish(base + "/status", payload, qos=0, retain=False)
        print(payload, flush=True)
        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            print("Status belum terkirim: broker tidak terhubung", flush=True)

    def on_connect(connection, userdata, flags, reason_code, properties):
        if reason_code.is_failure:
            print(f"Broker menolak koneksi: {reason_code}", flush=True)
            return
        connection.subscribe(base + "/command", qos=1)
        print(f"Terhubung; subscribe {base}/command", flush=True)

    def on_message(connection, userdata, message):
        try:
            incoming.put_nowait((message.payload, message.retain))
        except queue.Full:
            print("Antrean input penuh; pesan dibuang", flush=True)

    client.on_connect = on_connect
    client.on_message = on_message
    offline = simulator.snapshot()
    offline.update(state="error", detail="SIMULATOR terputus")
    offline["telemetry"]["event"] = "offline"
    client.will_set(base + "/status", json.dumps(offline), retain=False)
    print(f"SIMULATOR ONLY | {args.host}:{args.port} | topic {base}", flush=True)
    client.connect_async(args.host, args.port, keepalive=15)
    client.loop_start()
    heartbeat = 0
    try:
        while True:
            now = time.monotonic()
            completed = simulator.tick(now)
            if completed:
                publish(completed)
            try:
                payload, retained = incoming.get(timeout=0.05)
                publish(simulator.receive(payload, time.monotonic(), retained))
            except queue.Empty:
                pass
            if client.is_connected() and now >= heartbeat:
                publish(simulator.snapshot())
                heartbeat = now + 2
    except KeyboardInterrupt:
        if client.is_connected():
            client.publish(base + "/status", json.dumps(offline)).wait_for_publish(timeout=2)
    finally:
        client.disconnect()
        client.loop_stop()


if __name__ == "__main__":
    main()
