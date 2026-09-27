"""Fail-closed QEMU-side barrier for retired virtio action ports."""

import hashlib
import socket
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from future_prediction_bench.microvm_runtime import MicroVMRuntime, MicroVMRuntimeError


def _qtree(*, guest="off", host="off", port=1):
    return ("bus: virtio-serial-bus.0\n"
            "  dev: virtserialport, id \"\"\n"
            "    chardev = \"fpbctl\"\n"
            f"    nr = {port} (0x{port:x})\n"
            "    name = \"fpb.control\"\n"
            f"    port {port}, guest {guest}, host {host}, throttle off\n")


class ActionPortSnapshotBarrierTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.root = root
        self.kernel, self.initramfs, self.disk = (
            root / "vmlinuz", root / "initramfs", root / "guest.qcow2")
        for path in (self.kernel, self.initramfs, self.disk):
            path.write_bytes(b"fixture")

    def runtime(self):
        return MicroVMRuntime(
            self.kernel, self.initramfs, self.disk,
            kernel_sha256=hashlib.sha256(b"fixture").hexdigest(),
            initramfs_sha256=hashlib.sha256(b"fixture").hexdigest(),
            enable_action_port=True)

    def retired_runtime(self):
        runtime = self.runtime()
        runtime._socket_dir = self.root
        runtime._action_port_retired = True
        runtime._action_port_listener_unlinked = True
        runtime._action_port_disconnect_attested = True
        return runtime

    @staticmethod
    def monitor(*, initial="running", reconnect_after_save=False,
                existing_tag=False, fail_delete=False,
                fail_save_after_create=False):
        state = {"status": initial, "saved": existing_tag}
        commands = []

        def hmp(command, *, timeout=None):
            commands.append(command)
            if command == "info qtree":
                return _qtree(guest="on" if reconnect_after_save and state["saved"] else "off")
            if command == "info chardev":
                return "fpbctl: filename=disconnected:unix:/tmp/action.sock,server=on\n"
            if command == "info status":
                return "VM status: " + state["status"] + "\n"
            if command == "stop":
                state["status"] = "paused"
            elif command == "cont":
                state["status"] = "running"
            elif command == "savevm submitted":
                state["saved"] = True
                if fail_save_after_create:
                    raise MicroVMRuntimeError("injected_save_response_failure")
            elif command == "delvm submitted":
                if fail_delete:
                    raise MicroVMRuntimeError("injected_delete_failure")
                state["saved"] = False
            elif command == "info snapshots":
                return "1 submitted saved\n" if state["saved"] else "No snapshots\n"
            else:
                raise AssertionError(command)
            return ""
        return hmp, state, commands

    def test_retirement_closes_host_fd_but_rejects_stale_qemu_host_on(self):
        runtime = self.runtime()
        runtime._process = SimpleNamespace(poll=lambda: None)
        client_socket, peer = socket.socketpair()
        self.addCleanup(peer.close)
        runtime._action_port = client_socket
        runtime._socket_dir = self.root
        client = SimpleNamespace(socket=client_socket, failed=False, stopped=True)
        commands = []

        def hmp(command, *, timeout=None):
            commands.append(command)
            if command == "info qtree":
                return _qtree(host="on")
            if command == "info chardev":
                return "fpbctl: filename=unix:/tmp/action.sock,server=on\n"
            return ""

        runtime._hmp = hmp
        with self.assertRaisesRegex(MicroVMRuntimeError, "vm_action_port_qemu_not_disconnected"):
            runtime.retire_action_port(client)
        self.assertEqual(commands, ["info qtree", "info chardev"])
        self.assertEqual(client_socket.fileno(), -1)
        self.assertFalse((self.root / "action.sock").exists())
        self.assertTrue(runtime._action_port_retired)
        self.assertFalse(runtime.get_state()["checkpoint_operations_supported"])
        with self.assertRaisesRegex(MicroVMRuntimeError, "vm_action_port_snapshot_not_supported"):
            runtime.save_snapshot("submitted")
        self.assertEqual(runtime.metrics["snapshot_saves"], 0)

    def test_snapshot_gate_requires_exact_qemu_device_and_chardev_state(self):
        for guest, host, port, chardev in (
                ("on", "off", 1, True), ("off", "on", 1, True),
                ("off", "off", 2, True), ("off", "off", 1, False)):
            with self.subTest(guest=guest, host=host, port=port, chardev=chardev):
                runtime = self.runtime()
                runtime._action_port_retired = True
                runtime._action_port_disconnect_attested = True
                runtime._action_port_listener_unlinked = True
                runtime._socket_dir = self.root
                commands = []

                def hmp(command, *, timeout=None):
                    commands.append(command)
                    if command == "info qtree":
                        return _qtree(guest=guest, host=host, port=port)
                    if command == "info chardev":
                        prefix = "disconnected:" if chardev else ""
                        return f"fpbctl: filename={prefix}unix:/tmp/action.sock,server=on\n"
                    raise AssertionError("savevm must not be reached")

                runtime._hmp = hmp
                with self.assertRaisesRegex(MicroVMRuntimeError, "vm_action_port_qemu_(not_disconnected|state_invalid)"):
                    runtime.save_snapshot("submitted")
                self.assertNotIn("savevm submitted", commands)
                self.assertFalse(runtime._action_port_disconnect_attested)

    def test_load_checks_qemu_state_before_and_after_restore(self):
        runtime = self.runtime()
        runtime._action_port_retired = True
        runtime._action_port_disconnect_attested = True
        runtime._action_port_listener_unlinked = True
        runtime._socket_dir = self.root
        runtime._snapshots.add("submitted")
        commands = []

        def hmp(command, *, timeout=None):
            commands.append(command)
            if command == "info qtree":
                return _qtree(host="off" if commands.count("info qtree") == 1 else "on")
            if command == "info chardev":
                return "fpbctl: filename=disconnected:unix:/tmp/action.sock,server=on\n"
            if command == "loadvm submitted":
                return ""
            raise AssertionError(command)

        runtime._hmp = hmp
        with self.assertRaisesRegex(MicroVMRuntimeError, "vm_action_port_qemu_not_disconnected"):
            runtime.load_snapshot("submitted")
        self.assertEqual(commands, ["info qtree", "info chardev", "loadvm submitted",
                                "info qtree", "info chardev"])
        self.assertEqual(runtime.metrics["snapshot_loads"], 0)
        self.assertFalse(runtime._action_port_disconnect_attested)
        self.assertIsNone(runtime._socket_dir)

    def test_preload_reconnect_revokes_attestation_without_loading(self):
        runtime = self.retired_runtime()
        commands = []

        def hmp(command, *, timeout=None):
            commands.append(command)
            if command == "info qtree":
                return _qtree(host="on")
            if command == "info chardev":
                return "fpbctl: filename=unix:/tmp/action.sock,server=on\n"
            raise AssertionError("loadvm must not be reached")

        runtime._hmp = hmp
        with self.assertRaisesRegex(MicroVMRuntimeError, "vm_action_port_qemu_not_disconnected"):
            runtime.load_snapshot("submitted")
        self.assertEqual(commands, ["info qtree", "info chardev"])
        self.assertFalse(runtime._action_port_disconnect_attested)

    def test_action_port_save_pauses_and_preserves_initial_run_state(self):
        for initial in ("running", "paused"):
            with self.subTest(initial=initial):
                runtime = self.retired_runtime()
                runtime._hmp, state, commands = self.monitor(initial=initial)
                snapshot = runtime.save_snapshot("submitted")
                self.assertEqual(snapshot["tag"], "submitted")
                self.assertEqual(state["status"], initial)
                self.assertEqual(commands.count("savevm submitted"), 1)
                self.assertEqual("stop" in commands, initial == "running")
                self.assertEqual("cont" in commands, initial == "running")
                self.assertEqual(runtime.metrics["snapshot_saves"], 1)
                self.assertIn("submitted", runtime._snapshots)

    def test_post_save_reconnect_deletes_tag_and_disables_snapshots(self):
        runtime = self.retired_runtime()
        runtime._process = SimpleNamespace(poll=lambda: None)
        runtime._hmp, state, commands = self.monitor(reconnect_after_save=True)
        with self.assertRaisesRegex(MicroVMRuntimeError, "vm_action_port_qemu_not_disconnected"):
            runtime.save_snapshot("submitted")
        self.assertIn("delvm submitted", commands)
        self.assertFalse(state["saved"])
        self.assertEqual(state["status"], "running")
        self.assertFalse(runtime._action_port_disconnect_attested)
        self.assertEqual(runtime.metrics["snapshot_saves"], 0)
        self.assertNotIn("submitted", runtime._snapshots)
        with self.assertRaisesRegex(MicroVMRuntimeError, "snapshot_not_supported"):
            runtime.load_snapshot("submitted")

    def test_failed_delete_closes_and_never_publishes_uncertain_tag(self):
        runtime = self.retired_runtime()
        runtime._hmp, state, commands = self.monitor(
            reconnect_after_save=True, fail_delete=True)
        with patch.object(runtime, "close") as close:
            with self.assertRaisesRegex(MicroVMRuntimeError, "cleanup_unverified"):
                runtime.save_snapshot("submitted")
            close.assert_called_once()
        self.assertIn("delvm submitted", commands)
        self.assertTrue(state["saved"])
        self.assertFalse(runtime._action_port_disconnect_attested)
        self.assertNotIn("submitted", runtime._snapshots)

    def test_save_response_failure_still_removes_created_tag(self):
        runtime = self.retired_runtime()
        runtime._process = SimpleNamespace(poll=lambda: None)
        runtime._hmp, state, commands = self.monitor(fail_save_after_create=True)
        with self.assertRaisesRegex(MicroVMRuntimeError, "injected_save_response_failure"):
            runtime.save_snapshot("submitted")
        self.assertIn("delvm submitted", commands)
        self.assertFalse(state["saved"])
        self.assertEqual(state["status"], "running")
        self.assertFalse(runtime._action_port_disconnect_attested)

    def test_existing_tag_is_not_overwritten_or_deleted(self):
        runtime = self.retired_runtime()
        runtime._process = SimpleNamespace(poll=lambda: None)
        runtime._hmp, state, commands = self.monitor(existing_tag=True)
        with self.assertRaisesRegex(MicroVMRuntimeError, "tag_already_exists"):
            runtime.save_snapshot("submitted")
        self.assertNotIn("savevm submitted", commands)
        self.assertNotIn("delvm submitted", commands)
        self.assertTrue(state["saved"])
        self.assertEqual(state["status"], "running")

    def test_listener_unlink_failure_blocks_retirement_attestation(self):
        runtime = self.runtime()
        runtime._process = SimpleNamespace(poll=lambda: None)
        runtime._socket_dir = self.root
        host, peer = socket.socketpair()
        self.addCleanup(peer.close)
        runtime._action_port = host
        client = SimpleNamespace(socket=host, failed=False, stopped=True)
        with patch.object(Path, "lstat", return_value=SimpleNamespace(st_mode=stat.S_IFSOCK)), \
             patch.object(Path, "unlink", side_effect=PermissionError("injected")):
            with self.assertRaisesRegex(MicroVMRuntimeError, "listener_unlink_failed"):
                runtime.begin_action_port_retirement(client)
        self.assertFalse(runtime._action_port_listener_unlinked)
        self.assertFalse(runtime._action_port_disconnect_attested)
        with self.assertRaisesRegex(MicroVMRuntimeError, "retirement_invalid"):
            runtime.attest_action_port_retirement()

    def test_reappeared_listener_blocks_attest_save_and_load(self):
        runtime = self.retired_runtime()
        listener = self.root / "action.sock"
        listener.write_bytes(b"replacement")
        runtime._action_port_disconnect_attested = False
        with self.assertRaisesRegex(MicroVMRuntimeError, "listener_reappeared"):
            runtime.attest_action_port_retirement()
        runtime._action_port_disconnect_attested = True
        with self.assertRaisesRegex(MicroVMRuntimeError, "listener_reappeared"):
            runtime.save_snapshot("submitted")
        self.assertFalse(runtime._action_port_disconnect_attested)
        runtime._action_port_disconnect_attested = True
        with self.assertRaisesRegex(MicroVMRuntimeError, "listener_reappeared"):
            runtime.load_snapshot("submitted")
        self.assertFalse(runtime._action_port_disconnect_attested)


if __name__ == "__main__":
    unittest.main()
