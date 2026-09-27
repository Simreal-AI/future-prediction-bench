"""Commit-boundary and fail-closed checks for the bounded Crab probe.

These are host unit tests for error paths, not Linux kernel or QEMU
performance evidence. The real guest calibration and restore checks remain
in examples/official_crab/check_microvm.py.
"""
import base64
import importlib.util
import io
import json
from pathlib import Path
import socket
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from examples.official_crab import check_microvm as host


def load_guest_helpers():
    """Load helper functions without forking or requiring host /proc."""
    monitor = ModuleType("fpb_crab_process_monitor")
    monitor.PAGE_SIZE = 4096
    monitor.PAGEMAP_ENTRY_SIZE = 8
    path = Path(__file__).resolve().parents[1] / "examples/official_crab/guest_probe.py"
    spec = importlib.util.spec_from_file_location("fpb_crab_guest_unit_test", path)
    module = importlib.util.module_from_spec(spec)
    original_path = list(sys.path)
    try:
        with patch.dict(sys.modules, {"fpb_crab_process_monitor": monitor}):
            spec.loader.exec_module(module)
    finally:
        sys.path[:] = original_path
    return module


guest = load_guest_helpers()


class GuestInspectionBoundaryTests(unittest.TestCase):
    def probe(self):
        probe = guest.Probe.__new__(guest.Probe)
        probe.pid, probe.identity = 7, 42
        probe.soft_dirty_supported = True
        probe.baseline_file = {"sha256": "baseline"}
        probe._identity = Mock(return_value=42)
        probe.file_digest = Mock(return_value=dict(probe.baseline_file))
        return probe

    def inspect(self, probe, *, pagemap=b"\x00" * 8, dirty=(), ranges=((4096, 8192),)):
        monitor = SimpleNamespace(PAGE_SIZE=4096, PAGEMAP_ENTRY_SIZE=8,
            parse_writable_ranges=Mock(return_value=list(ranges)),
            dirty_pids=Mock(return_value=set(dirty)))
        with patch.object(guest, "monitor", monitor), patch.object(guest, "open",
                create=True, return_value=io.BytesIO(b"\x00" * 8 + pagemap)):
            result = probe.inspect()
        return result, monitor

    def assert_unknown_requires_both_components(self, result):
        self.assertFalse(result["known"])
        self.assertTrue(result["process_changed"])
        self.assertTrue(result["filesystem_changed"])

    def test_failed_kernel_calibration_never_authorizes_a_clean_skip(self):
        probe = self.probe()
        probe.soft_dirty_supported = False
        result, monitor = self.inspect(probe)
        self.assert_unknown_requires_both_components(result)
        monitor.dirty_pids.assert_not_called()
        probe._identity.assert_not_called()

    def test_replaced_worker_identity_is_unknown(self):
        probe = self.probe()
        probe._identity.return_value = 43
        result, monitor = self.inspect(probe)
        self.assert_unknown_requires_both_components(result)
        monitor.parse_writable_ranges.assert_not_called()

    def test_incomplete_pagemap_is_not_treated_as_clean(self):
        result, monitor = self.inspect(self.probe(), pagemap=b"\x00" * 7)
        self.assert_unknown_requires_both_components(result)
        monitor.dirty_pids.assert_not_called()

    def test_missing_writable_mapping_is_unknown(self):
        result, monitor = self.inspect(self.probe(), ranges=())
        self.assert_unknown_requires_both_components(result)
        monitor.dirty_pids.assert_not_called()

    def test_inspection_permission_failure_is_unknown(self):
        probe = self.probe()
        monitor = SimpleNamespace(PAGE_SIZE=4096, PAGEMAP_ENTRY_SIZE=8,
            parse_writable_ranges=Mock(return_value=[(4096, 8192)]))
        with patch.object(guest, "monitor", monitor), patch.object(guest, "open",
                create=True, side_effect=PermissionError("denied")):
            result = probe.inspect()
        self.assert_unknown_requires_both_components(result)

    def test_clean_and_dirty_states_remain_distinct(self):
        probe = self.probe()
        clean, _ = self.inspect(probe)
        self.assertTrue(clean["known"])
        self.assertFalse(clean["process_changed"])
        self.assertFalse(clean["filesystem_changed"])
        ram_dirty, _ = self.inspect(probe, dirty={probe.pid})
        self.assertTrue(ram_dirty["process_changed"])
        self.assertFalse(ram_dirty["filesystem_changed"])
        probe.file_digest.return_value = {"sha256": "changed"}
        disk_dirty, _ = self.inspect(probe)
        self.assertFalse(disk_dirty["process_changed"])
        self.assertTrue(disk_dirty["filesystem_changed"])

    def test_packed_step_captures_inspection_before_resetting_baseline(self):
        probe = self.probe()
        operations = []
        real_call = probe.call
        def record_call(request):
            operations.append(request["op"])
            return {"value": 123} if request["op"] == "state" else {}
        probe.call = record_call
        probe.inspect = Mock(side_effect=lambda: operations.append("inspect") or {"known": True})
        result = real_call({"op": "step", "action": {"op": "transient"}, "inspect": True})
        self.assertEqual(operations, ["transient", "inspect", "state", "baseline"])
        self.assertTrue(result["baseline_prepared"])

    def test_packed_action_failure_does_not_reset_baseline(self):
        probe = self.probe()
        real_call = probe.call
        probe.call = Mock(side_effect=RuntimeError("action failed"))
        probe.inspect = Mock()
        with self.assertRaisesRegex(RuntimeError, "action failed"):
            real_call({"op": "step", "action": {"op": "memory", "value": 456}, "inspect": True})
        probe.call.assert_called_once_with({"op": "memory", "value": 456})
        probe.inspect.assert_not_called()

    def test_packed_step_rejects_non_boolean_inspection_flag(self):
        for invalid in (None, 0, 1, "true"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "inspect_flag"):
                self.probe().call({"op": "step", "action": {"op": "state"}, "inspect": invalid})


class HostCommitBoundaryTests(unittest.TestCase):
    def test_failed_snapshot_does_not_continue_after_speculative_reset(self):
        vm = SimpleNamespace(load_snapshot=Mock(), save_snapshot=Mock(side_effect=RuntimeError("save failed")))
        prepared = {"state": {"value": 123, "file": "seed"}, "baseline_prepared": True,
                    "inspection": None}
        with patch.object(host, "call", return_value=prepared) as call:
            with self.assertRaisesRegex(RuntimeError, "save failed"):
                host.trial(vm, condition="every_turn", repetition=0, policy=None,
                           snapshot_type=None, packed=True)
        call.assert_called_once()
        vm.save_snapshot.assert_called_once_with("e0t0")
        vm.load_snapshot.assert_called_once_with("crabstart", resume=True)

    def test_unknown_inspection_cannot_continue_after_policy_skip(self):
        vm = SimpleNamespace(load_snapshot=Mock(), save_snapshot=Mock())
        prepared = {"state": {"value": 123, "file": "seed"}, "baseline_prepared": True,
                    "inspection": {"known": False, "process_changed": True, "filesystem_changed": True}}
        policy = SimpleNamespace(evaluate=Mock(return_value=SimpleNamespace(should_checkpoint=False)))
        with patch.object(host, "call", return_value=prepared) as call:
            with self.assertRaisesRegex(RuntimeError, "known_clean_skip"):
                host.trial(vm, condition="selective", repetition=0, policy=policy,
                           snapshot_type=lambda **fields: fields, packed=True)
        call.assert_called_once()
        vm.save_snapshot.assert_not_called()

    def test_clean_claim_with_changed_component_cannot_authorize_skip(self):
        for field in ("process_changed", "filesystem_changed"):
            inspection = {"known": True, "process_changed": False, "filesystem_changed": False}
            inspection[field] = True
            prepared = {"state": {}, "inspection": inspection, "baseline_prepared": True}
            vm = SimpleNamespace(load_snapshot=Mock(), save_snapshot=Mock())
            policy = SimpleNamespace(evaluate=Mock(return_value=SimpleNamespace(should_checkpoint=False)))
            with self.subTest(field=field), patch.object(host, "call", return_value=prepared):
                with self.assertRaisesRegex(RuntimeError, "known_clean_skip"):
                    host.trial(vm, condition="selective", repetition=0, policy=policy,
                               snapshot_type=lambda **fields: fields, packed=True)
            vm.save_snapshot.assert_not_called()

    def test_error_response_aborts_the_host_call(self):
        error = base64.b64encode(json.dumps({"error": "worker vanished"}).encode()).decode()
        with patch.object(host, "_required", return_value="FPB_CRAB_RESULT:" + error):
            with self.assertRaisesRegex(RuntimeError, "worker vanished"):
                host.call(None, {"op": "state"})

    def test_unknown_inspection_response_is_preserved_for_conservative_policy(self):
        value = {"known": False, "error": "PermissionError", "process_changed": True, "filesystem_changed": True}
        encoded = base64.b64encode(json.dumps(value).encode()).decode()
        with patch.object(host, "_required", return_value="FPB_CRAB_RESULT:" + encoded):
            self.assertEqual(host.call(None, {"op": "inspect"}), value)


class GuestFramingTests(unittest.TestCase):
    def test_real_socket_pair_receives_split_frame(self):
        writer, reader = socket.socketpair()
        with writer, reader:
            writer.sendall(b'{"op":')
            writer.sendall(b'"state"}\n')
            self.assertEqual(guest.receive(reader), {"op": "state"})

    def test_closed_or_oversized_frames_are_rejected(self):
        for data in (b"", b"x" * (guest.LIMIT + 1)):
            writer, reader = socket.socketpair()
            with self.subTest(size=len(data)), writer, reader:
                if data:
                    writer.sendall(data)
                writer.shutdown(socket.SHUT_WR)
                with self.assertRaisesRegex(ValueError, "invalid_frame"):
                    guest.receive(reader)


if __name__ == "__main__":
    unittest.main()
