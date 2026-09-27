"""Evidence gates for the isolated unopened-port QEMU proof."""

import unittest

from examples.realworld_boltons26.probe_unopened_action_port_snapshot import (
    _chardev_disconnected, _qtree_port, validate_evidence,
)


PORT = {"port_number": 1, "guest": "off", "host": "off",
        "throttle": "off", "name": "fpb.control", "chardev": "fpbctl"}


class FakeMonitor:
    def __init__(self, *, host="off", chardev_disconnected=True):
        self.host = host
        self.chardev_disconnected = chardev_disconnected

    def _hmp(self, command):
        if command == "info qtree":
            return ("dev: virtserialport, id \"\"\n"
                    "  chardev = \"fpbctl\"\n"
                    "  nr = 1 (0x1)\n"
                    "  name = \"fpb.control\"\n"
                    f"  port 1, guest off, host {self.host}, throttle off\n")
        if command == "info chardev":
            prefix = "disconnected:" if self.chardev_disconnected else ""
            return f"fpbctl: filename={prefix}unix:/redacted/action.sock,server=on\n"
        raise AssertionError(command)


class UnopenedActionPortProbeTests(unittest.TestCase):
    def test_exact_qemu_device_and_host_backend_are_separate_observations(self):
        monitor = FakeMonitor(host="off")
        self.assertEqual(_qtree_port(monitor), PORT)
        self.assertTrue(_chardev_disconnected(monitor))
        monitor.host = "on"
        monitor.chardev_disconnected = False
        self.assertEqual(_qtree_port(monitor)["host"], "on")
        self.assertFalse(_chardev_disconnected(monitor))

    def test_positive_claim_rejects_partial_post_close_state(self):
        report = {
            "status": "supported_only_when_unopened",
            "production_guard_closed_before_after": True,
            "guest_agent_running": False,
            "guest_port_fd_holders_before": 0,
            "guest_port_fd_holders_after": 0,
            "guest_agent_processes_before": 0,
            "guest_agent_processes_after": 0,
            "ram_restored": True,
            "workspace_restored": True,
            "host_listener_inode_preserved": True,
            "host_chardev_disconnected_before_after": True,
            "post_load_rpc": {"rpc_client_failed_closed": True,
                              "no_initial_bytes_before_rpc": True,
                              "host_disconnected_after_failure": True,
                              "saved_session_epoch_restored": False},
            "device_state_before": dict(PORT),
            "device_state_after": dict(PORT),
        }
        self.assertTrue(validate_evidence(report))
        report["post_load_rpc"]["host_disconnected_after_failure"] = False
        with self.assertRaisesRegex(ValueError, "evidence_incomplete"):
            validate_evidence(report)
        report["post_load_rpc"]["host_disconnected_after_failure"] = True
        report["device_state_after"]["host"] = "on"
        with self.assertRaisesRegex(RuntimeError, "device_state_changed"):
            validate_evidence(report)


if __name__ == "__main__":
    unittest.main()
