"""RealWorldEnv contracts for the VM adapter, using a deterministic fake guest."""

import base64
import ast
import hashlib
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import zlib
from unittest import mock
from datetime import datetime, timedelta, timezone
from pathlib import Path

from future_prediction_bench.microvm_coding import (
    MicroVMCodingAdapter, replace_text_helper_binding)
from future_prediction_bench import guest_action_rpc
from future_prediction_bench.replace_text import apply as apply_replace_text
from future_prediction_bench.realworld import RealWorldEnv, validate_task
from future_prediction_bench.virtio_action import VirtioSerialReadFallback


class FakeRuntime:
    kernel_sha256 = "a" * 64
    initramfs_sha256 = "b" * 64
    readonly_disk_paths = ()

    def __init__(self, disk_path, *, probe=None):
        self.disk_path = disk_path
        self.probe = probe or "/dev/vda:68737173:0000\n/dev/vdb:00000000:53ef\n"
        self.commands = []
        self.files = {"/mnt/root/workspace/bug.py": b"x = 4\n"}
        self.hardlinks = {}
        self.saved_files = None
        self.restores = 0
        self.closed = False

    def start(self):
        self.commands.append("VM_START")

    def wait_for_serial(self, marker, *, timeout):
        self.commands.append("WAIT_READY")

    def run_shell(self, command, *, timeout):
        self.commands.append(command)
        if command.startswith("for d in /dev/vd?"):
            return {"return_code": 0, "stdout": self.probe}
        if "find /mnt/root/workspace -type l -print" in command:
            return {"return_code": 0, "stdout": ""}
        if "find /mnt/root/workspace -type f -print0 | base64" in command:
            data = b"\x00".join(name.encode() for name in self.files) + b"\x00"
            return {"return_code": 0, "stdout": base64.b64encode(data).decode() + "\n"}
        if "sha256sum /mnt/root/workspace/bug.py" in command:
            data = self.files["/mnt/root/workspace/bug.py"]
            digest = hashlib.sha256(data).hexdigest()
            text = base64.b64encode(data[:16000]).decode()
            return {"return_code": 0, "stdout":
                    f"{digest}  /mnt/root/workspace/bug.py\n{len(data)}\n{text}\n"}
        parts = shlex.split(command)
        if len(parts) >= 2 and parts[-2].startswith("exec(__import__('zlib').decompress"):
            action = json.loads(base64.b64decode(parts[-1]))
            with tempfile.TemporaryDirectory() as root:
                for path, data in self.files.items():
                    target = Path(root) / path.removeprefix("/mnt/root/workspace/")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
                for alias, source in self.hardlinks.items():
                    target = Path(root) / alias
                    target.unlink(missing_ok=True)
                    os.link(Path(root) / source, target)
                try:
                    result = apply_replace_text(root, action)
                except ValueError:
                    result = {"status": "error", "reason": "adapter_error",
                              "error_type": "ValueError"}
                except OSError:
                    return {"return_code": 1, "stdout": ""}
                if "sha256" in result:
                    path = "/mnt/root/workspace/" + action["path"]
                    self.files[path] = (Path(root) / action["path"]).read_bytes()
            encoded = base64.b64encode(json.dumps(result).encode()).decode()
            return {"return_code": 0, "stdout": "FPB_REPLACE_RESULT=" + encoded + "\n"}
        if "chroot /mnt/root /usr/local/bin/python3.12" in command:
            wrapper_expression = shlex.split(command)[-1]
            call = ast.parse(wrapper_expression).body[0].value
            wrapper = ast.literal_eval(call.args[0])
            encoded = re.search(r"code=base64.b64decode\('([^']+)'\)", wrapper).group(1)
            code = base64.b64decode(encoded).decode()
            candidate_output = {"print(2+3)": b"5\n", "print('visible')": b"visible\n",
                                "import sys; sys.stdout.buffer.write(bytes([255]))": b"\xff"}.get(code)
            payload = {"return_code": 0 if candidate_output is not None else 1,
                       "stdout_b64": base64.b64encode(
                           candidate_output if candidate_output is not None
                           else b"unexpected Python case\n").decode(),
                       "truncated": False}
            return {"return_code": 0, "stdout": "FPB_CASE_RESULT=" +
                    base64.b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode() + "\n"}
        return {"return_code": 0, "stdout": ""}

    def save_snapshot(self, tag):
        self.saved_files = dict(self.files)
        return {"kind": "full_vm_state_qcow2_v1", "tag": tag}

    def load_snapshot(self, tag):
        self.files = dict(self.saved_files)
        self.restores += 1

    def get_state(self):
        return {"running": not self.closed, "checkpoint_kind": "full_vm_state_qcow2_v1"}

    def close(self):
        self.closed = True


class VirtioFakeRuntime(FakeRuntime):
    enable_action_port = True

    def __init__(self, disk_path, workspace):
        super().__init__(disk_path)
        self.workspace = workspace
        self.host_socket, self.guest_socket = socket.socketpair()
        self.action_retired = False
        self.agent_thread = None
        self.agent_errors = []
        self.preinstalled_mismatch = False
        self.qemu_host_on_after_close = False
        self.reopen_fails = False
        self.guest_reopened = False
        self.disconnect_attested = False

    def start(self):
        super().start()

        def serve():
            try:
                with mock.patch.object(guest_action_rpc, "WORKSPACE", str(self.workspace)):
                    guest_action_rpc.serve(self.guest_socket.fileno())
            except Exception as exc:
                self.agent_errors.append(exc)

        self.agent_thread = threading.Thread(target=serve, daemon=True)
        self.agent_thread.start()

    def action_port_socket(self):
        if self.action_retired:
            raise RuntimeError("port retired")
        return self.host_socket

    def begin_action_port_retirement(self, client):
        if (self.action_retired or client.socket is not self.host_socket
                or not client.stopped or client.failed):
            raise RuntimeError("invalid retirement")
        self.host_socket.close()
        self.action_retired = True

    def action_port_qemu_state(self):
        return ({"guest": "off", "host": "on", "chardev_disconnected": False}
                if self.qemu_host_on_after_close else
                {"guest": "off", "host": "off", "chardev_disconnected": True})

    def attest_action_port_retirement(self):
        if not self.action_retired:
            raise RuntimeError("retirement not begun")
        if self.qemu_host_on_after_close:
            raise RuntimeError("QEMU still connected")
        self.disconnect_attested = True

    def run_shell(self, command, *, timeout):
        if "-I -S -B -c" in command and "b64decode" in command:
            encoded = re.search(r"b64decode\('([^']+)'\)",
                                shlex.split(command)[-1])
            if encoded is not None:
                source = base64.b64decode(encoded.group(1)).decode("utf-8")
                if "FPB_REOPEN_OK" in source:
                    self.commands.append(command)
                    if self.reopen_fails:
                        return {"return_code": 1, "stdout": ""}
                    self.guest_reopened = True
                    self.qemu_host_on_after_close = False
                    return {"return_code": 0, "stdout": "FPB_REOPEN_OK\n"}
                marker = ("FPB_PROCESS_SET" if "FPB_PROCESS_SET" in source
                          else "FPB_PORT_HOLDERS" if "FPB_PORT_HOLDERS" in source
                          else None)
                if marker is not None:
                    self.commands.append(command)
                    payload = ([[1, 1, "/usr/bin/busybox"],
                                [2, 2, "/usr/bin/busybox"]]
                               if marker == "FPB_PROCESS_SET" else [])
                    return {"return_code": 0, "stdout": marker + "=" +
                            base64.b64encode(json.dumps(payload).encode()).decode() + "\n"}
        if command == "sha256sum /mnt/root/fpb_guest_action_rpc.py":
            self.commands.append(command)
            return {"return_code": 0, "stdout": (
                ("0" * 64 if self.preinstalled_mismatch else
                 hashlib.sha256(Path(guest_action_rpc.__file__).read_bytes()).hexdigest())
                + "  /mnt/root/fpb_guest_action_rpc.py\n")}
        if "FPB_AGENT_PID=" in command:
            self.commands.append(command)
            return {"return_code": 0, "stdout": "FPB_AGENT_PID=12345\n"}
        if command.startswith("if kill -0 12345"):
            self.commands.append(command)
            return {"return_code": 0, "stdout": (
                "EXITED" if self.agent_thread and not self.agent_thread.is_alive()
                else "ALIVE")}
        return super().run_shell(command, timeout=timeout)

    def save_snapshot(self, tag):
        if not self.action_retired:
            raise RuntimeError("live action port cannot be snapshotted")
        return super().save_snapshot(tag)

    def load_snapshot(self, tag):
        if not self.action_retired:
            raise RuntimeError("live action port cannot be restored")
        return super().load_snapshot(tag)

    def close(self):
        self.host_socket.close()
        self.guest_socket.close()
        if self.agent_thread:
            self.agent_thread.join(timeout=1)
        super().close()


def task():
    now = datetime.now(timezone.utc)
    return {"schema_version": "realworld-0.1", "task_id": "vm-fixture",
            "event_id": "vm-fixture", "cluster_id": "vm-fixture", "split": "train",
            "prompt": "Repair a file.", "issued_at": (now - timedelta(minutes=1)).isoformat(),
            "action_deadline": (now + timedelta(minutes=10)).isoformat(),
            "outcome_not_before": (now - timedelta(minutes=1)).isoformat(),
            "verify_after": (now - timedelta(minutes=1)).isoformat(),
            "tool_manifest": [{"name": name, "description": name} for name in
                              ("list_files", "read_file", "write_file", "run_visible_checks", "submit")],
            "reward_contract": {"id": "host_cases", "description": "Private cases",
                                "min_reward": 0, "max_reward": 1},
            "budgets": {"max_actions": 8, "max_wall_seconds": 300},
            "is_fixture": True}


class MicroVMCodingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.disk = root / "guest.qcow2"
        self.disk.write_bytes(b"fake qcow2")
        self.verifier = root / "verifier"
        self.verifier.mkdir()
        (self.verifier / "verify.json").write_text(json.dumps({
            "kind": "command_cases_v1", "cases": [
                {"argv": ["python3", "-B", "-c", "print(2+3)"],
                 "expected_stdout": "5\n", "expected_returncode": 0},
                {"argv": ["python3", "-B", "-c", "print(2+3)"],
                 "expected_stdout": "5\n", "expected_returncode": 0},
            ]}), encoding="utf-8")

    def adapter(self, *, probe=None):
        runtime = FakeRuntime(self.disk, probe=probe)
        adapter = MicroVMCodingAdapter(runtime, verifier_dir=self.verifier,
                                        visible_check=["python3", "-B", "-c", "print('visible')"])
        return adapter, runtime

    def test_x86_backend_rejected_before_verifier_access_or_vm_start(self):
        for backend in ("x86_64_tcg", "x86_64_kvm"):
            runtime = FakeRuntime(self.disk)
            runtime.backend = backend
            with self.subTest(backend=backend), self.assertRaisesRegex(ValueError, "aarch64_hvf backend"):
                MicroVMCodingAdapter(runtime, verifier_dir=self.disk.parent / "missing-verifier",
                    visible_check=["python3", "-B", "-c", "print('visible')"])
            self.assertEqual(runtime.commands, [])

    def test_backend_change_cannot_be_frozen_under_the_hvf_artifact_kind(self):
        adapter, runtime = self.adapter()
        runtime.backend = "x86_64_tcg"
        with self.assertRaisesRegex(ValueError, "aarch64_hvf backend"):
            adapter.artifact_binding()
        with self.assertRaisesRegex(ValueError, "aarch64_hvf backend"):
            adapter.reset(task(), now=datetime.now(timezone.utc))
        self.assertNotIn("VM_START", runtime.commands)

    def test_reset_freezes_complete_validated_task_identity(self):
        original = task()
        stamped = validate_task(original)
        adapter, runtime = self.adapter()
        try:
            adapter.reset(stamped, now=datetime.now(timezone.utc))
            self.assertEqual(adapter._task_sha256, stamped["task_sha256"])
            self.assertIn("VM_START", runtime.commands)
        finally:
            adapter.close()
        changed = dict(stamped, prompt="Changed prompt with stale digest")
        rejected, unused_runtime = self.adapter()
        with self.assertRaisesRegex(ValueError, "digest fields differ"):
            rejected.reset(changed, now=datetime.now(timezone.utc))
        self.assertNotIn("VM_START", unused_runtime.commands)

    def virtio_adapter(self, *, preinstalled=False):
        workspace = self.disk.parent / "workspace"
        workspace.mkdir(exist_ok=True)
        (workspace / "bug.py").write_bytes(b"x = 4\n")
        runtime = VirtioFakeRuntime(self.disk, workspace)
        adapter = MicroVMCodingAdapter(
            runtime, verifier_dir=self.verifier,
            visible_check=["python3", "-B", "-c", "print('visible')"],
            read_transport="virtio_serial_readonly_v1",
            preinstalled_read_agent=preinstalled)
        return adapter, runtime

    def test_preinstalled_agent_skips_upload_and_fail_closes_on_digest_mismatch(self):
        adapter, runtime = self.virtio_adapter(preinstalled=True)
        env = RealWorldEnv(task(), adapter)
        try:
            env.reset("preinstalled")
            self.assertFalse(any(command == ": > /mnt/root/fpb_guest_action_rpc.py"
                                 or "base64 -d >> /mnt/root/fpb_guest_action_rpc.py" in command
                                 for command in runtime.commands))
            self.assertTrue(any(command == "sha256sum /mnt/root/fpb_guest_action_rpc.py"
                                for command in runtime.commands))
            self.assertEqual(env.step({"action": "read_file", "path": "bug.py"})[
                "observation"]["text"], "x = 4\n")
            env.step({"action": "submit"})
            self.assertEqual(env.verify()["reward"], 1.0)
            self.assertEqual(adapter.metrics["virtio_agent_upload_seconds"], 0.0)
            self.assertGreater(adapter.metrics["virtio_agent_prepare_seconds"], 0.0)
        finally:
            adapter.close()

        mismatch_adapter, mismatch_runtime = self.virtio_adapter(preinstalled=True)
        mismatch_runtime.preinstalled_mismatch = True
        mismatch_env = RealWorldEnv(task(), mismatch_adapter)
        with self.assertRaisesRegex(ValueError, "Adapter reset failed"):
            mismatch_env.reset("mismatch")
        self.assertTrue(mismatch_runtime.closed)
        self.assertIsNone(mismatch_runtime.saved_files)

    def test_opt_in_virtio_read_preserves_graded_episode_and_retires_before_snapshot(self):
        serial_adapter, serial_runtime = self.adapter()
        virtio_adapter, virtio_runtime = self.virtio_adapter()
        serial_env = RealWorldEnv(task(), serial_adapter)
        virtio_env = RealWorldEnv(task(), virtio_adapter)
        try:
            serial_env.reset("serial")
            virtio_env.reset("virtio")
            serial_read = serial_env.step({"action": "read_file", "path": "bug.py"})
            virtio_read = virtio_env.step({"action": "read_file", "path": "bug.py"})
            self.assertEqual(serial_read["observation"], virtio_read["observation"])
            self.assertEqual(virtio_adapter.metrics["virtio_reads"], 1)
            self.assertFalse(virtio_runtime.action_retired)
            self.assertEqual(serial_env.step({"action": "submit"})["observation"],
                             virtio_env.step({"action": "submit"})["observation"])
            self.assertTrue(virtio_runtime.action_retired)
            self.assertTrue(virtio_adapter._virtio_client.stopped)
            with self.assertRaisesRegex(Exception, "session_unavailable"):
                virtio_adapter._virtio_client.read_file("bug.py")
            self.assertEqual(serial_env.verify()["reward"], virtio_env.verify()["reward"])
            self.assertEqual(serial_adapter.metrics["full_vm_restores"],
                             virtio_adapter.metrics["full_vm_restores"])
            self.assertEqual(virtio_runtime.agent_errors, [])
        finally:
            serial_adapter.close()
            virtio_adapter.close()

    def test_stale_qemu_host_bit_requires_zero_byte_guest_reopen_and_recheck(self):
        adapter, runtime = self.virtio_adapter(preinstalled=True)
        runtime.qemu_host_on_after_close = True
        env = RealWorldEnv(task(), adapter)
        try:
            env.reset("stale-host-bit")
            env.step({"action": "read_file", "path": "bug.py"})
            result = env.step({"action": "submit"})
            self.assertEqual(result["observation"]["status"], "submitted")
            self.assertTrue(runtime.action_retired)
            self.assertTrue(runtime.disconnect_attested)
            self.assertFalse(runtime.qemu_host_on_after_close)
            self.assertIsNotNone(runtime.saved_files)
            self.assertTrue(runtime.guest_reopened)
            self.assertGreater(adapter.metrics["virtio_agent_disconnect_probe_seconds"], 0)
            self.assertEqual(env.verify()["reward"], 1.0)
        finally:
            adapter.close()

    def test_failed_guest_reopen_leaves_snapshot_and_reward_unavailable(self):
        adapter, runtime = self.virtio_adapter(preinstalled=True)
        runtime.qemu_host_on_after_close = True
        runtime.reopen_fails = True
        env = RealWorldEnv(task(), adapter)
        try:
            env.reset("reopen-fails")
            env.step({"action": "read_file", "path": "bug.py"})
            result = env.step({"action": "submit"})
            self.assertEqual(result["info"]["status"], "interrupted")
            self.assertIsNone(result["reward"])
            self.assertTrue(runtime.action_retired)
            self.assertFalse(runtime.disconnect_attested)
            self.assertFalse(runtime.guest_reopened)
            self.assertIsNone(runtime.saved_files)
        finally:
            adapter.close()

    def test_opt_in_virtio_transport_failure_interrupts_without_reward_or_snapshot(self):
        adapter, runtime = self.virtio_adapter()
        env = RealWorldEnv(task(), adapter)
        try:
            env.reset("revision-1")
            adapter._virtio_client.failed = True
            result = env.step({"action": "read_file", "path": "bug.py"})
            self.assertEqual(result["info"]["status"], "interrupted")
            self.assertIsNone(result["reward"])
            self.assertIsNone(runtime.saved_files)
            self.assertFalse(runtime.action_retired)
        finally:
            adapter.close()

    def test_opt_in_virtio_missing_file_matches_serial_policy_error(self):
        serial_adapter, serial_runtime = self.adapter()
        virtio_adapter, _ = self.virtio_adapter()
        original = serial_runtime.run_shell

        def serial_missing(command, *, timeout):
            if "missing.py" in command:
                serial_runtime.commands.append(command)
                return {"return_code": 0, "stdout": "MISSING\n"}
            return original(command, timeout=timeout)

        serial_runtime.run_shell = serial_missing
        serial_env = RealWorldEnv(task(), serial_adapter)
        virtio_env = RealWorldEnv(task(), virtio_adapter)
        try:
            serial_env.reset("serial")
            virtio_env.reset("virtio")
            first = serial_env.step({"action": "read_file", "path": "missing.py"})
            second = virtio_env.step({"action": "read_file", "path": "missing.py"})
            self.assertEqual(first["observation"], second["observation"])
            self.assertEqual(first["observation"], {
                "status": "error", "reason": "adapter_error", "error_type": "ValueError"})
            self.assertEqual(first["info"]["status"], "active")
            self.assertEqual(second["info"]["status"], "active")
        finally:
            serial_adapter.close()
            virtio_adapter.close()

    def test_opt_in_stuck_agent_blocks_submit_and_reward(self):
        adapter, runtime = self.virtio_adapter()
        original = runtime.run_shell

        def stuck(command, *, timeout):
            if command.startswith("if kill -0 12345"):
                runtime.commands.append(command)
                return {"return_code": 0, "stdout": "ALIVE"}
            return original(command, timeout=timeout)

        runtime.run_shell = stuck
        env = RealWorldEnv(task(), adapter)
        try:
            env.reset("revision-1")
            adapter.command_timeout = 0.001  # Keep the injected timeout fast.
            result = env.step({"action": "submit"})
            self.assertEqual(result["info"]["status"], "interrupted")
            self.assertIsNone(result["reward"])
            self.assertFalse(runtime.action_retired)
            self.assertIsNone(runtime.saved_files)
        finally:
            adapter.close()

    def test_opt_in_large_regular_file_falls_back_to_serial_observation(self):
        adapter, runtime = self.virtio_adapter()
        env = RealWorldEnv(task(), adapter)
        try:
            env.reset("revision-1")
            with mock.patch.object(adapter._virtio_client, "read_file",
                                   side_effect=VirtioSerialReadFallback("large")):
                result = env.step({"action": "read_file", "path": "bug.py"})
            self.assertEqual(result["observation"]["text"], "x = 4\n")
            self.assertEqual(adapter.metrics["virtio_read_fallbacks"], 1)
            self.assertEqual(adapter.metrics["virtio_reads"], 0)
            self.assertEqual(adapter.metrics["file_reads"], 1)
            self.assertTrue(any("sha256sum /mnt/root/workspace/bug.py" in command
                                for command in runtime.commands))
        finally:
            adapter.close()

    def test_opt_in_extra_process_or_open_port_fd_blocks_snapshot(self):
        for probe in ("process", "fd"):
            with self.subTest(probe=probe):
                adapter, runtime = self.virtio_adapter()
                env = RealWorldEnv(task(), adapter)
                try:
                    env.reset("revision-1")
                    if probe == "process":
                        adapter._scan_guest_processes = lambda: {**adapter._virtio_process_baseline,
                                                                  (99, 1): "/usr/local/bin/python3.12"}
                    else:
                        adapter._scan_guest_action_port_holders = lambda: [[99, 3]]
                    result = env.step({"action": "submit"})
                    self.assertEqual(result["info"]["status"], "interrupted")
                    self.assertIsNone(result["reward"])
                    self.assertFalse(runtime.action_retired)
                    self.assertIsNone(runtime.saved_files)
                finally:
                    adapter.close()

    def test_opt_in_requires_matching_runtime_and_rejects_branch_checkpoint(self):
        serial_runtime = FakeRuntime(self.disk)
        with self.assertRaisesRegex(ValueError, "must match"):
            MicroVMCodingAdapter(
                serial_runtime, verifier_dir=self.verifier,
                visible_check=["python3", "-B", "-c", "print('visible')"],
                read_transport="virtio_serial_readonly_v1")
        adapter, _ = self.virtio_adapter()
        env = RealWorldEnv(task(), adapter)
        try:
            env.reset("revision-1")
            with self.assertRaisesRegex(ValueError, "does not support branch"):
                adapter.create_branch_checkpoint()
        finally:
            adapter.close()

    def test_magic_detection_private_cases_and_full_vm_restore(self):
        adapter, runtime = self.adapter()
        env = RealWorldEnv(task(), adapter)
        try:
            opening = env.reset("revision-1")
            self.assertEqual(opening["observation"]["runtime_kind"], "qemu_hvf_full_vm_qcow2_v1")
            self.assertEqual(adapter.block_devices, {"squashfs": "/dev/vda", "ext4": "/dev/vdb"})
            self.assertTrue(any("mknod -m 666 /mnt/root/dev/null" in command
                                for command in runtime.commands))
            self.assertEqual(env.step({"action": "list_files"})["observation"]["files"], ["bug.py"])
            self.assertEqual(env.step({"action": "read_file", "path": "bug.py"})["observation"]["text"], "x = 4\n")
            self.assertTrue(env.step({"action": "run_visible_checks"})["observation"]["passed"])
            submitted = env.step({"action": "submit"})
            self.assertIsNone(submitted["reward"])
            self.assertNotIn("print(2+3)", json.dumps(opening) + json.dumps(submitted))
            # Simulate a hidden case changing the VM disk between cases; each
            # case must restore the same submitted memory and disk state.
            runtime.files["/mnt/root/workspace/bug.py"] = b"corrupt\n"
            self.assertEqual(env.verify()["reward"], 1.0)
            self.assertEqual(runtime.restores, 2)
            self.assertEqual(runtime.files["/mnt/root/workspace/bug.py"], b"x = 4\n")
            self.assertEqual(env.export_trajectory()["reward"], 1.0)
        finally:
            adapter.close()

    def test_ambiguous_disk_magic_aborts_before_policy_actions(self):
        adapter, runtime = self.adapter(probe="/dev/vda:68737173:53ef\n")
        env = RealWorldEnv(task(), adapter)
        with self.assertRaises(ValueError):
            env.reset("revision-1")
        self.assertEqual(env.get_state()["status"], "setup_error")
        self.assertTrue(runtime.closed)

    def test_candidate_identity_is_bound_and_used_for_visible_and_hidden_python(self):
        adapter, runtime = self.adapter()
        self.assertEqual(adapter.artifact_binding()["candidate_python_identity"],
                         "guest_uid_gid_65534_v2")
        env = RealWorldEnv(task(), adapter)
        try:
            env.reset("revision-1")
            self.assertTrue(env.step({"action": "run_visible_checks"})["observation"]["passed"])
            env.step({"action": "submit"})
            self.assertEqual(env.verify()["reward"], 1.0)
            wrappers = []
            for command in runtime.commands:
                if "chroot /mnt/root /usr/local/bin/python3.12 -I -c" not in command:
                    continue
                expression = shlex.split(command)[-1]
                call = ast.parse(expression).body[0].value
                wrappers.append(ast.literal_eval(call.args[0]))
            self.assertEqual(len(wrappers), 3)
            for wrapper in wrappers:
                self.assertIn("user=65534,group=65534,extra_groups=[]", wrapper)
                self.assertIn("start_new_session=True", wrapper)
        finally:
            adapter.close()

    def test_hidden_expected_output_stays_on_host_and_controls_reward(self):
        specification = json.loads((self.verifier / "verify.json").read_text())
        specification["cases"][0]["expected_stdout"] = "6\n"
        (self.verifier / "verify.json").write_text(json.dumps(specification), encoding="utf-8")
        adapter, runtime = self.adapter()
        env = RealWorldEnv(task(), adapter)
        try:
            env.reset("revision-1")
            denied = env.step({"action": "shell", "command": "cat /verifier/verify.json"})
            self.assertEqual(denied["observation"]["reason"], "tool_not_available")
            with self.assertRaises(ValueError):
                adapter.step({"action": "read_file", "path": "../verify.json"}, now=datetime.now(timezone.utc))
            env.step({"action": "submit"})
            self.assertEqual(env.verify()["reward"], 0.0)
            self.assertNotIn("6\\n", json.dumps(env.export_trajectory()))
            self.assertNotIn("6\\n", "".join(runtime.commands))
        finally:
            adapter.close()

    def test_hidden_cases_compare_exact_bytes_not_lossy_decoded_text(self):
        specification = {"kind": "command_cases_v1", "cases": [{
            "argv": ["python3", "-B", "-c",
                     "import sys; sys.stdout.buffer.write(bytes([255]))"],
            "expected_stdout": "\ufffd", "expected_returncode": 0}]}
        (self.verifier / "verify.json").write_text(json.dumps(specification), encoding="utf-8")
        adapter, _ = self.adapter()
        env = RealWorldEnv(task(), adapter)
        try:
            env.reset("revision-1")
            env.step({"action": "submit"})
            self.assertEqual(env.verify()["reward"], 0.0)
        finally:
            adapter.close()

    def test_case_stdout_eof_waits_for_exit_or_scores_overall_timeout(self):
        adapter, _ = self.adapter()
        adapter.command_timeout = 3.0  # One-second inner case deadline.

        def run_case(code):
            command = adapter._python_command(["python3", "-B", "-c", code])
            expression = shlex.split(command)[-1]
            wrapper = ast.literal_eval(ast.parse(expression).body[0].value.args[0])
            self.assertIn("'/usr/local/bin/python3.12'", wrapper)
            self.assertIn("cwd='/workspace'", wrapper)
            wrapper = wrapper.replace("'/usr/local/bin/python3.12'", repr(sys.executable), 1)
            wrapper = wrapper.replace("cwd='/workspace'", f"cwd={self.temp.name!r}", 1)
            started = time.monotonic()
            result = subprocess.run([sys.executable, "-c", wrapper],
                                    capture_output=True, text=True, timeout=4)
            elapsed = time.monotonic() - started
            self.assertEqual(result.returncode, 0, result.stderr)
            match = re.fullmatch(r"FPB_CASE_RESULT=([A-Za-z0-9+/=]+)\n?", result.stdout)
            self.assertIsNotNone(match, result.stdout)
            payload = json.loads(base64.b64decode(match.group(1), validate=True))
            return payload, elapsed

        redirect_stdout = ("import os,time; "
                           "fd=os.open(os.devnull,os.O_WRONLY); "
                           "os.dup2(fd,1); os.close(fd); ")
        try:
            finished, _ = run_case(redirect_stdout + "time.sleep(0.05)")
            self.assertEqual(finished, {"return_code": 0, "stdout_b64": "",
                                        "truncated": False})
            timed_out, elapsed = run_case(redirect_stdout + "time.sleep(5)")
            self.assertEqual(timed_out, {"return_code": 124, "stdout_b64": "",
                                         "truncated": False})
            self.assertLess(elapsed, 3.0)
        finally:
            adapter.close()

    def test_large_utf8_source_upload_reassembles_with_bounded_serial_calls(self):
        adapter, runtime = self.adapter()
        env = RealWorldEnv(task(), adapter)
        content = "界" * 15000
        raw = content.encode("utf-8")
        try:
            env.reset("revision-1")
            result = env.step({"action": "write_file", "path": "bug.py",
                               "content": content})
            self.assertEqual(result["observation"]["sha256"],
                             hashlib.sha256(raw).hexdigest())
            chunks = [command for command in runtime.commands
                      if "base64 -d >> /mnt/root/workspace/bug.py.fpb-write-temp"
                      in command]
            self.assertEqual(len(chunks), 17)
            rebuilt = b"".join(base64.b64decode(
                re.search(r"printf '%s' '([^']+)'", command).group(1))
                for command in chunks)
            self.assertEqual(rebuilt, raw)
            self.assertTrue(all(len(command.encode("utf-8")) <= 3800
                                for command in chunks))
        finally:
            adapter.close()

    def test_replace_text_one_bounded_guest_command_and_conflicts(self):
        adapter, runtime = self.adapter()
        manifest = task()
        manifest["tool_manifest"].append({"name": "replace_text", "description": "Exact edit."})
        manifest["metadata"] = {"replace_text_helper_binding": replace_text_helper_binding()}
        env = RealWorldEnv(manifest, adapter)
        original = runtime.files["/mnt/root/workspace/bug.py"]
        action = {"action": "replace_text", "path": "bug.py",
                  "expected_file_sha256": hashlib.sha256(original).hexdigest(),
                  "old_text": "x = 4", "new_text": "x = 5\n# $(touch hacked) ℝ"}
        try:
            env.reset("revision-1")
            result = env.step(action)["observation"]
            changed = runtime.files["/mnt/root/workspace/bug.py"]
            self.assertEqual(result, {"path": "bug.py",
                                      "sha256": hashlib.sha256(changed).hexdigest()})
            self.assertEqual(changed, b"x = 5\n# $(touch hacked) \xe2\x84\x9d\n")
            commands = [command for command in runtime.commands
                        if len(shlex.split(command)) >= 2 and
                        shlex.split(command)[-2].startswith("exec(__import__('zlib').decompress")]
            self.assertEqual(len(commands), 1)
            self.assertLessEqual(len(commands[0].encode("utf-8")), 3800)
            self.assertNotIn("touch hacked", commands[0])
            stale = env.step(action)["observation"]
            self.assertEqual(stale["status"], "conflict")
            self.assertEqual(stale["reason"], "sha256_mismatch")
            self.assertEqual(runtime.files["/mnt/root/workspace/bug.py"], changed)
            self.assertEqual(adapter.metrics["file_writes"], 1)
            missing = dict(action, expected_file_sha256=hashlib.sha256(changed).hexdigest(),
                           old_text="not present")
            self.assertEqual(env.step(missing)["observation"]["reason"], "old_text_not_unique")
            self.assertEqual(runtime.files["/mnt/root/workspace/bug.py"], changed)
        finally:
            adapter.close()

    def test_replace_text_transport_handles_adversarial_path_and_rejects_oversize(self):
        adapter, runtime = self.adapter()
        path = "nested/quote'\nseparator\u2028.py"
        original = b"anchor\n"
        runtime.files["/mnt/root/workspace/" + path] = original
        manifest = task()
        manifest["tool_manifest"].append({"name": "replace_text", "description": "Exact edit."})
        manifest["metadata"] = {"replace_text_helper_binding": replace_text_helper_binding()}
        env = RealWorldEnv(manifest, adapter)
        action = {"action": "replace_text", "path": path,
                  "expected_file_sha256": hashlib.sha256(original).hexdigest(),
                  "old_text": "anchor", "new_text": "replacement"}
        try:
            env.reset("revision-1")
            result = env.step(action)["observation"]
            self.assertEqual(result["sha256"], hashlib.sha256(b"replacement\n").hexdigest())
            command = next(item for item in runtime.commands
                           if len(shlex.split(item)) >= 2 and
                           shlex.split(item)[-2].startswith("exec(__import__('zlib').decompress"))
            self.assertNotIn("separator", command)
            self.assertNotIn("\u2028", command)
            self.assertLessEqual(len(command.encode("utf-8")), 3800)
            before = dict(runtime.files)
            invalid = dict(action, path="../verifier/verify.json")
            self.assertEqual(env.step(invalid)["observation"]["status"], "error")
            self.assertEqual(runtime.files, before)
            oversized = dict(action, old_text="x" * 1025)
            self.assertEqual(env.step(oversized)["observation"]["status"], "error")
            self.assertEqual(runtime.files, before)
        finally:
            adapter.close()

    def test_compressed_guest_program_is_valid_python(self):
        from future_prediction_bench.microvm_coding import _REPLACE_GUEST_PROGRAM
        source = zlib.decompress(base64.b64decode(_REPLACE_GUEST_PROGRAM))
        compile(source, "<trusted-guest-replace>", "exec")
        namespace = {"__name__": "trusted_guest_test"}
        exec(source, namespace)
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / "x.txt"
            target.write_bytes(b"old\n")
            action = {"action": "replace_text", "path": "x.txt",
                      "expected_file_sha256": hashlib.sha256(b"old\n").hexdigest(),
                      "old_text": "old", "new_text": "new"}
            self.assertEqual(namespace["apply"](root, action)["sha256"],
                             hashlib.sha256(b"new\n").hexdigest())
            self.assertEqual(target.read_bytes(), b"new\n")

    def test_committed_guest_program_matches_host_shared_ast(self):
        from future_prediction_bench import replace_text

        host = ast.parse(Path(replace_text.__file__).read_text(encoding="utf-8"))
        host.body = [
            node for node in host.body
            if not (isinstance(node, ast.FunctionDef)
                    and node.name in {"guest_program", "helper_binding"})
        ]
        for node in ast.walk(host):
            if (isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef))
                    and node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body.pop(0)
        guest_bytes = zlib.decompress(base64.b64decode(
            replace_text.guest_program(), validate=True))
        guest = ast.parse(guest_bytes)
        self.assertEqual(ast.dump(host, include_attributes=False),
                         ast.dump(guest, include_attributes=False))
        self.assertEqual(hashlib.sha256(guest_bytes).hexdigest(),
                         replace_text.helper_binding()["guest_program_sha256"])
        self.assertEqual(hashlib.sha256(Path(replace_text.__file__).read_bytes()).hexdigest(),
                         replace_text.helper_binding()["source_sha256"])

    def test_replace_text_requires_exact_guest_helper_binding_before_vm_start(self):
        adapter, runtime = self.adapter()
        manifest = task()
        manifest["tool_manifest"].append({"name": "replace_text", "description": "Exact edit."})
        manifest["metadata"] = {"replace_text_helper_binding": {
            **replace_text_helper_binding(), "guest_program_sha256": "0" * 64}}
        env = RealWorldEnv(manifest, adapter)
        with self.assertRaises(ValueError):
            env.reset("revision-1")
        self.assertEqual(runtime.commands, [])

    def test_replace_text_action_bound_and_hardlink_error_match_docker(self):
        adapter, runtime = self.adapter()
        manifest = task()
        manifest["tool_manifest"].append({"name": "replace_text", "description": "Exact edit."})
        manifest["metadata"] = {"replace_text_helper_binding": replace_text_helper_binding()}
        env = RealWorldEnv(manifest, adapter)
        original = runtime.files["/mnt/root/workspace/bug.py"]
        action = {"action": "replace_text", "path": "bug.py",
                  "expected_file_sha256": hashlib.sha256(original).hexdigest(),
                  "old_text": "x" * 400, "new_text": ""}
        try:
            env.reset("revision-1")
            self.assertEqual(env.step(action)["observation"]["reason"],
                             "old_text_not_unique")
            before_commands = len(runtime.commands)
            for invalid in (dict(action, old_text="x" * 512),
                            dict(action, old_text="x" * 800),
                            dict(action, old_text="x" * 1024),
                            dict(action, path="界" * 200)):
                self.assertEqual(env.step(invalid)["observation"]["status"], "error")
            self.assertEqual(len(runtime.commands), before_commands)
            self.assertEqual(runtime.files["/mnt/root/workspace/bug.py"], original)
            runtime.files["/mnt/root/workspace/alias.py"] = original
            runtime.hardlinks["alias.py"] = "bug.py"
            hardlink = dict(action, old_text="x = 4")
            self.assertEqual(env.step(hardlink)["observation"],
                             {"status": "error", "reason": "adapter_error",
                              "error_type": "ValueError"})
            self.assertEqual(runtime.files["/mnt/root/workspace/bug.py"], original)
            self.assertEqual(adapter.metrics["file_writes"], 0)
        finally:
            adapter.close()

    def test_replace_text_guest_io_failure_interrupts_without_reward(self):
        adapter, runtime = self.adapter()
        manifest = task()
        manifest["tool_manifest"].append({"name": "replace_text", "description": "Exact edit."})
        manifest["metadata"] = {"replace_text_helper_binding": replace_text_helper_binding()}
        env = RealWorldEnv(manifest, adapter)
        original = runtime.files["/mnt/root/workspace/bug.py"]
        action = {"action": "replace_text", "path": "bug.py",
                  "expected_file_sha256": hashlib.sha256(original).hexdigest(),
                  "old_text": "x = 4", "new_text": "x = 5"}
        try:
            env.reset("revision-1")
            with mock.patch(__name__ + ".apply_replace_text",
                            side_effect=OSError("guest I/O failure")):
                result = env.step(action)
            self.assertEqual(result["info"]["status"], "interrupted")
            self.assertIsNone(result["reward"])
            self.assertEqual(runtime.files["/mnt/root/workspace/bug.py"], original)
        finally:
            adapter.close()

    def test_hidden_case_transport_and_timeout_contract_fail_before_vm_start(self):
        specification = json.loads((self.verifier / "verify.json").read_text())
        for code, expected_rc in (("x" * 1900, 0), ("print(1)", 124)):
            with self.subTest(code_length=len(code), expected_returncode=expected_rc):
                case = dict(specification["cases"][0])
                case["argv"] = ["python3", "-B", "-c", code]
                case["expected_returncode"] = expected_rc
                (self.verifier / "verify.json").write_text(json.dumps({
                    "kind": "command_cases_v1", "cases": [case]}))
                runtime = FakeRuntime(self.disk)
                with self.assertRaises(ValueError):
                    MicroVMCodingAdapter(
                        runtime, verifier_dir=self.verifier,
                        visible_check=["python3", "-B", "-c", "print('visible')"])
                self.assertEqual(runtime.commands, [])

    def test_verifier_replaced_after_constructor_is_rechecked_before_binding(self):
        adapter, runtime = self.adapter()
        try:
            specification = json.loads((self.verifier / "verify.json").read_text())
            specification["cases"][0]["argv"][3] = "x" * 1900
            (self.verifier / "verify.json").write_text(json.dumps(specification))
            with self.assertRaisesRegex(ValueError, "bounded guest command length"):
                adapter.artifact_binding()
            self.assertEqual(runtime.commands, [])
        finally:
            adapter.close()

    def test_long_quoted_and_newline_paths_use_bounded_encoded_shell_transport(self):
        adapter, runtime = self.adapter()
        env = RealWorldEnv(task(), adapter)
        original_run = runtime.run_shell
        read_bytes = b"path-safe\n"
        digest = hashlib.sha256(read_bytes).hexdigest()

        def run_shell(command, *, timeout):
            if 'sha256sum < "$fpb_target"' in command:
                runtime.commands.append(command)
                return {"return_code": 0, "stdout": (
                    f"{digest}  -\n{len(read_bytes)}\n"
                    + base64.b64encode(read_bytes).decode() + "\n")}
            return original_run(command, timeout=timeout)

        runtime.run_shell = run_shell
        try:
            env.reset("revision-1")
            paths = ["/".join(["'" * 127] * 4), "a\nb", "a\u2028b"]
            for path in paths:
                with self.subTest(path_kind="separator" if not path.isprintable()
                                  else "quoted"):
                    before = len(runtime.commands)
                    content = "path-safe\n"
                    env.step({"action": "write_file", "path": path,
                              "content": content})
                    self.assertEqual(env.status, "active")
                    observation = env.step({"action": "read_file", "path": path})
                    self.assertEqual(observation["observation"]["sha256"], digest)
                    commands = runtime.commands[before:]
                    encoded_path = base64.b64encode(
                        (adapter.workspace_root + "/" + path).encode()).decode()
                    self.assertTrue(any(encoded_path in command for command in commands))
                    self.assertTrue(all(len(command.encode("utf-8")) <= 3800
                                        and "\n" not in command and "\r" not in command
                                        for command in commands))
                    chunks = [re.findall(r"printf '%s' '([^']+)' \| base64 -d >>",
                                         command)
                              for command in commands]
                    rebuilt = b"".join(base64.b64decode(matches[-1])
                                       for matches in chunks if matches)
                    self.assertEqual(rebuilt, content.encode())
        finally:
            adapter.close()


if __name__ == "__main__":
    unittest.main()
