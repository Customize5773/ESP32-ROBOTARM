import json
import unittest
from arm_simulator import ArmSimulator


def command(identifier="one", angles=None):
    return json.dumps({"command_id": identifier, "joint_angles": angles or [1, 2, 3, 4, 5, 90]})


class SimulatorTests(unittest.TestCase):
    def test_completion_preserves_command_and_positions(self):
        sim = ArmSimulator()
        self.assertEqual(sim.receive(command(), 0)["state"], "running")
        self.assertIsNone(sim.tick(2))
        result = sim.tick(3)
        self.assertEqual(result["state"], "idle")
        self.assertEqual(result["last_command"]["command_id"], "one")
        self.assertEqual(result["telemetry"]["positions_deg"], [1, 2, 3, 4, 5, 90])

    def test_busy_and_duplicate_do_not_replace_active_command(self):
        sim = ArmSimulator()
        sim.receive(command(), 0)
        self.assertEqual(sim.receive(command("two"), 1)["telemetry"]["event"], "busy")
        self.assertEqual(sim.receive(command(), 1)["telemetry"]["event"], "duplicate")
        self.assertEqual(sim.tick(3)["last_command"]["command_id"], "one")
        self.assertEqual(sim.receive(command(), 4)["telemetry"]["event"], "duplicate")
        self.assertEqual(sim.receive(command("two"), 4)["state"], "running")

    def test_malformed_and_retained_commands_never_execute(self):
        sim = ArmSimulator()
        for payload in [b"\xff", "[]", "{", command(angles=[1]*5),
                        command(angles=[True, 2, 3, 4, 5, 90]),
                        command(angles=[float("nan"), 2, 3, 4, 5, 90]),
                        command(angles=[1, 2, 3, 4, 5, 181])]:
            with self.subTest(payload=payload):
                self.assertEqual(sim.receive(payload, 0)["telemetry"]["event"], "rejected")
                self.assertIsNone(sim.deadline)
        self.assertEqual(sim.receive(command(), 0, True)["telemetry"]["event"], "rejected")

    def test_failure_does_not_claim_target_reached(self):
        sim = ArmSimulator(fail=True)
        sim.receive(command(), 0)
        result = sim.tick(3)
        self.assertEqual(result["state"], "error")
        self.assertEqual(result["telemetry"]["positions_deg"], [0, 0, 0, 0, 0, 90])

    def test_legacy_payload_deduplicates_without_command_id(self):
        sim = ArmSimulator()
        payload = json.dumps({"joint_angles": [0, 0, 0, 0, 0, 90], "issued_at": "demo"})
        sim.receive(payload, 0)
        sim.tick(3)
        self.assertEqual(sim.receive(payload, 4)["telemetry"]["event"], "duplicate")


if __name__ == "__main__":
    unittest.main()
