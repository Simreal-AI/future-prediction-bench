"""Offline framing and read-only guest agent tests using socket pairs."""

import hashlib
import json
import socket
import struct
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from future_prediction_bench import guest_action_rpc
from future_prediction_bench.microvm_runtime import MicroVMRuntime, MicroVMRuntimeError
from future_prediction_bench.virtio_action import (MAX_RESPONSE, VirtioActionClient,
                                                  VirtioActionError,
                                                  VirtioSerialReadFallback)


class FakeRuntime:
    def __init__(self, sock):
        self.sock = sock

    def action_port_socket(self):
        return self.sock


class VirtioActionTests(unittest.TestCase):
    def setUp(self):
        self.host, self.guest = socket.socketpair()
        self.addCleanup(self.host.close)
        self.addCleanup(self.guest.close)

    def _serve(self, workspace):
        errors = []
        def target():
            try:
                with patch.object(guest_action_rpc, "WORKSPACE", str(workspace)):
                    guest_action_rpc.serve(self.guest.fileno())
            except Exception as exc:
                errors.append(exc)
        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        return thread, errors

    def test_ping_read_truncation_missing_and_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "small.txt").write_text("hello\n", encoding="utf-8")
            (root / "large.txt").write_text("A" * 20000, encoding="ascii")
            thread, errors = self._serve(root)
            client = VirtioActionClient(FakeRuntime(self.host), timeout=1)
            self.assertTrue(client.ping())
            small = client.read_file("small.txt")
            self.assertEqual(small, {"path": "small.txt", "text": "hello\n",
                                     "sha256": hashlib.sha256(b"hello\n").hexdigest(),
                                     "truncated": False})
            large = client.read_file("large.txt")
            self.assertEqual(large["text"], "A" * 16000)
            self.assertTrue(large["truncated"])
            with self.assertRaisesRegex(ValueError, "not found"):
                client.read_file("missing.txt")
            with self.assertRaises(ValueError):
                client.read_file("../escape")
            self.assertTrue(client.stop())
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(client.sequence, 5)

    def test_guest_rejects_workspace_symlink_without_reading_target(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside_dir:
            root = Path(directory)
            outside = Path(outside_dir) / "secret.txt"
            outside.write_text("secret", encoding="utf-8")
            (root / "link.txt").symlink_to(outside)
            thread, errors = self._serve(root)
            client = VirtioActionClient(FakeRuntime(self.host), timeout=1)
            with self.assertRaisesRegex(VirtioActionError, "read_failed"):
                client.read_file("link.txt")
            client.stop()
            thread.join(timeout=2)
            self.assertEqual(errors, [])

    def test_nonregular_matches_missing_and_oversize_requests_serial_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "folder").mkdir()
            with (root / "huge.bin").open("wb") as stream:
                stream.truncate(50_000_001)
            thread, errors = self._serve(root)
            client = VirtioActionClient(FakeRuntime(self.host), timeout=1)
            with self.assertRaisesRegex(ValueError, "not found"):
                client.read_file("folder")
            with self.assertRaisesRegex(VirtioSerialReadFallback, "serial_fallback"):
                client.read_file("huge.bin")
            self.assertFalse(client.failed)
            self.assertTrue(client.stop())
            thread.join(timeout=2)
            self.assertEqual(errors, [])

    def test_host_rejects_sequence_mismatch_and_closes_session(self):
        def wrong_reply():
            size = struct.unpack(">I", self.guest.recv(4))[0]
            request = json.loads(self.guest.recv(size))
            reply = {"v": 1, "session": request["session"], "seq": request["seq"] + 1,
                     "op": "PING", "ok": True, "data": {"pong": True}, "error": None}
            data = json.dumps(reply).encode()
            self.guest.sendall(struct.pack(">I", len(data)) + data)
        thread = threading.Thread(target=wrong_reply, daemon=True)
        thread.start()
        client = VirtioActionClient(FakeRuntime(self.host), timeout=1)
        with self.assertRaisesRegex(VirtioActionError, "response_mismatch"):
            client.ping()
        self.assertTrue(client.failed)
        thread.join(timeout=1)

    def test_host_rejects_oversize_frame_and_timeout(self):
        def too_large():
            size = struct.unpack(">I", self.guest.recv(4))[0]
            self.guest.recv(size)
            self.guest.sendall(struct.pack(">I", MAX_RESPONSE + 1))
        thread = threading.Thread(target=too_large, daemon=True)
        thread.start()
        client = VirtioActionClient(FakeRuntime(self.host), timeout=1)
        with self.assertRaisesRegex(VirtioActionError, "response_length_invalid"):
            client.ping()
        thread.join(timeout=1)

        second_host, second_guest = socket.socketpair()
        self.addCleanup(second_host.close)
        self.addCleanup(second_guest.close)
        timeout_client = VirtioActionClient(FakeRuntime(second_host), timeout=0.02)
        with self.assertRaisesRegex(VirtioActionError, "timed_out"):
            timeout_client.ping()
        self.assertTrue(timeout_client.failed)

    def test_host_closes_session_on_invalid_read_result(self):
        def invalid_reply():
            size = struct.unpack(">I", self.guest.recv(4))[0]
            request = json.loads(self.guest.recv(size))
            reply = {"v": 1, "session": request["session"], "seq": request["seq"],
                     "op": "READ_FILE", "ok": True,
                     "data": {"path": "sample.txt", "size": 1,
                              "sha256": hashlib.sha256(b"x").hexdigest(),
                              "head_b64": "not-base64!"}, "error": None}
            data = json.dumps(reply).encode()
            self.guest.sendall(struct.pack(">I", len(data)) + data)
        thread = threading.Thread(target=invalid_reply, daemon=True)
        thread.start()
        client = VirtioActionClient(FakeRuntime(self.host), timeout=1)
        with self.assertRaisesRegex(VirtioActionError, "read_result_invalid"):
            client.read_file("sample.txt")
        self.assertTrue(client.failed)
        thread.join(timeout=1)

    def test_second_host_caller_fails_while_request_in_flight(self):
        received = threading.Event()
        release = threading.Event()
        errors = []

        def delayed_reply():
            try:
                size = struct.unpack(">I", self.guest.recv(4))[0]
                request = json.loads(self.guest.recv(size))
                received.set()
                if not release.wait(2):
                    raise AssertionError("test reply was not released")
                reply = {"v": 1, "session": request["session"], "seq": request["seq"],
                         "op": "PING", "ok": True, "data": {"pong": True},
                         "error": None}
                data = json.dumps(reply).encode()
                self.guest.sendall(struct.pack(">I", len(data)) + data)
            except Exception as exc:
                errors.append(exc)

        server = threading.Thread(target=delayed_reply, daemon=True)
        server.start()
        client = VirtioActionClient(FakeRuntime(self.host), timeout=2)
        first_results = []
        first = threading.Thread(target=lambda: first_results.append(client.ping()), daemon=True)
        first.start()
        self.assertTrue(received.wait(1))
        with self.assertRaisesRegex(VirtioActionError, "concurrent_request_unsupported"):
            client.ping()
        release.set()
        first.join(timeout=2)
        server.join(timeout=2)
        self.assertEqual(errors, [])
        self.assertEqual(first_results, [True])
        self.assertEqual(client.sequence, 1)

    def test_opt_in_port_explicitly_disallows_unverified_vm_snapshots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = MicroVMRuntime(root / "kernel", root / "initramfs", root / "disk",
                                     kernel_sha256="0" * 64,
                                     initramfs_sha256="1" * 64,
                                     enable_action_port=True)
            with self.assertRaisesRegex(MicroVMRuntimeError, "action_port_snapshot_not_supported"):
                runtime.save_snapshot("warm")
            with self.assertRaisesRegex(MicroVMRuntimeError, "action_port_snapshot_not_supported"):
                runtime.load_snapshot("warm")
            with self.assertRaisesRegex(MicroVMRuntimeError, "action_port_fork_not_supported"):
                runtime.fork_snapshot("warm", [root / "child.qcow2"])
            opt_in_state = runtime.get_state()
            self.assertIsNone(opt_in_state["checkpoint_kind"])
            self.assertIs(opt_in_state["checkpoint_operations_supported"], False)
            self.assertEqual(opt_in_state["action_transport"], "virtio_serial_readonly_v1")

            default_runtime = MicroVMRuntime(root / "kernel", root / "initramfs", root / "disk",
                                             kernel_sha256="0" * 64,
                                             initramfs_sha256="1" * 64)
            default_state = default_runtime.get_state()
            self.assertEqual(default_state["checkpoint_kind"], "full_vm_state_qcow2_v1")
            self.assertNotIn("checkpoint_operations_supported", default_state)
            self.assertNotIn("action_transport", default_state)

    def test_retired_port_is_one_way_and_snapshot_needs_qemu_disconnect(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = MicroVMRuntime(root / "kernel", root / "initramfs", root / "disk",
                                     kernel_sha256="0" * 64,
                                     initramfs_sha256="1" * 64,
                                     enable_action_port=True)
            host, guest = socket.socketpair()
            self.addCleanup(guest.close)
            runtime._action_port = host
            runtime._socket_dir = root

            class StoppedClient:
                socket = host
                stopped = True
                failed = False

            with patch.object(runtime, "_check_running"):
                monitor_state = {"status": "running", "saved": False}
                def monitor(command, **_):
                    if command == "info qtree":
                        return ('dev: virtserialport, id ""\n'
                                '  chardev = "fpbctl"\n'
                                '  nr = 1 (0x1)\n'
                                '  name = "fpb.control"\n'
                                '  port 1, guest off, host off, throttle off\n')
                    if command == "info chardev":
                        return "fpbctl: filename=disconnected:unix:/tmp/action.sock,server=on\n"
                    if command == "info status":
                        return "VM status: " + monitor_state["status"] + "\n"
                    if command == "stop":
                        monitor_state["status"] = "paused"
                    elif command == "cont":
                        monitor_state["status"] = "running"
                    elif command == "savevm submitted":
                        monitor_state["saved"] = True
                    elif command == "info snapshots":
                        return "1 submitted saved\n" if monitor_state["saved"] else "No snapshots\n"
                    return ""

                with patch.object(runtime, "_hmp", side_effect=monitor):
                    runtime.retire_action_port(StoppedClient())
                    self.assertEqual(host.fileno(), -1)
                    with self.assertRaisesRegex(MicroVMRuntimeError, "not_enabled"):
                        runtime.action_port_socket()
                    with self.assertRaisesRegex(MicroVMRuntimeError, "retirement_invalid"):
                        runtime.retire_action_port(StoppedClient())
                    runtime.save_snapshot("submitted")
                    runtime.load_snapshot("submitted")
                    with self.assertRaisesRegex(MicroVMRuntimeError, "fork_not_supported"):
                        runtime.fork_snapshot("submitted", [root / "child.qcow2"])
                    state = runtime.get_state()
                    self.assertEqual(state["checkpoint_kind"], "full_vm_state_qcow2_v1")
                    self.assertIs(state["checkpoint_operations_supported"], True)
                    self.assertEqual(state["action_transport"],
                                     "virtio_serial_readonly_retired_v1")
                    self.assertEqual(state["metrics"]["snapshot_saves"], 1)
                    self.assertEqual(state["metrics"]["snapshot_loads"], 1)


if __name__ == "__main__":
    unittest.main()
