"""QEMU/HVF transport contracts exercised with a fake VM socket process."""

import hashlib
import io
import os
import re
import socket
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from future_prediction_bench.microvm_runtime import (
    MicroVMRuntime, MicroVMRuntimeError, _clone_or_copy_qcow2,
    _file_identity, _quarantine_marker, _sha256_file,
)


class FakeQEMUProcess:
    def __init__(self, args, *, silent_serial=False, **kwargs):
        self.args = args
        self.silent_serial = silent_serial
        self.stderr = io.BytesIO()
        self.returncode = None
        self.snapshots = set()
        self.paused = "-S" in args
        self.host_sockets = []
        self.connections = []
        self.threads = []
        for target in (self._monitor, self._serial):
            host, guest = socket.socketpair()
            self.host_sockets.append(host)
            self.connections.append(guest)
            thread = threading.Thread(target=target, args=(guest,), daemon=True)
            thread.start()
            self.threads.append(thread)

    def _monitor(self, conn):
        try:
            conn.sendall(b"QEMU monitor\r\n(qemu) ")
            buffer = b""
            while self.returncode is None:
                data = conn.recv(4096)
                if not data:
                    break
                buffer += data
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    command = line.decode("ascii").strip()
                    if command.startswith("savevm "):
                        self.snapshots.add(command.split(" ", 1)[1])
                        response = b""
                    elif command.startswith("loadvm "):
                        response = b"" if command.split(" ", 1)[1] in self.snapshots else b"Error: missing\r\n"
                    elif command.startswith("delvm "):
                        tag = command.split(" ", 1)[1]
                        if tag in self.snapshots:
                            self.snapshots.remove(tag)
                            response = b""
                        else:
                            response = b"Error: missing\r\n"
                    elif command == "info snapshots":
                        response = b"ID TAG VM SIZE\r\n" + b"".join(
                            f"1 {tag} 83.4 MiB\r\n".encode() for tag in sorted(self.snapshots))
                    elif command == "info status":
                        response = f"VM status: {'paused' if self.paused else 'running'}\r\n".encode()
                    elif command == "stop":
                        self.paused = True
                        response = b""
                    elif command == "cont":
                        self.paused = False
                        response = b""
                    else:
                        response = b""
                    conn.sendall(response + b"(qemu) ")
        except OSError:
            pass

    def _serial(self, conn):
        try:
            conn.sendall(b"Launching initramfs emergency recovery shell\r\n")
            buffer = b""
            while self.returncode is None:
                data = conn.recv(4096)
                if not data:
                    break
                buffer += data
                if b"'END__'" not in buffer or self.silent_serial:
                    continue
                match = re.search(rb"__FPB_[0-9a-f]{32}_", buffer)
                if match:
                    prefix = match.group(0)
                    # Echo the full input first: it must not contain a joined
                    # BEGIN/END marker and must not fool the host parser.
                    conn.sendall(buffer + b"\r\n" + prefix + b"BEGIN__\r\n"
                                 + b"guest-ok\r\n\r\n" + prefix + b"END__:0\r\n")
                buffer = b""
        except OSError:
            pass

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = 0
        for sock in self.connections + self.host_sockets:
            try:
                sock.close()
            except OSError:
                pass

    def kill(self):
        self.terminate()

    def wait(self, timeout=None):
        return self.returncode


class MicroVMRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.kernel = root / "vmlinuz"
        self.initramfs = root / "initramfs"
        self.disk = root / "guest.qcow2"
        self.modloop = root / "modloop-virt"
        self.kernel.write_bytes(b"a pinned kernel")
        self.initramfs.write_bytes(b"a pinned initramfs")
        self.disk.write_bytes(b"fake qcow2")
        self.modloop.write_bytes(b"read-only guest modules")

    def runtime(self, *, popen_factory=None, kernel_sha256=None, readonly_disk_paths=()):
        return MicroVMRuntime(
            self.kernel, self.initramfs, self.disk,
            kernel_sha256=kernel_sha256 or hashlib.sha256(self.kernel.read_bytes()).hexdigest(),
            initramfs_sha256=hashlib.sha256(self.initramfs.read_bytes()).hexdigest(),
            readonly_disk_paths=readonly_disk_paths,
            popen_factory=popen_factory)

    def _started_fake_runtime(self):
        created = []
        def factory(args, **kwargs):
            process = FakeQEMUProcess(args, **kwargs)
            created.append(process)
            return process
        runtime = self.runtime(popen_factory=factory)
        runtime._check_artifacts = lambda: None
        runtime._connect = lambda path, deadline: created[0].host_sockets.pop(0)
        runtime.start()
        return runtime, created[0]

    def test_save_rejects_existing_qemu_tag_without_overwriting_it(self):
        runtime, process = self._started_fake_runtime()
        try:
            process.snapshots.add("inherited")
            with self.assertRaisesRegex(MicroVMRuntimeError, "tag_already_exists"):
                runtime.save_snapshot("inherited")
            self.assertEqual(process.snapshots, {"inherited"})
            self.assertEqual(runtime.metrics["snapshot_saves"], 0)
            runtime.save_snapshot("warm")
            with self.assertRaisesRegex(MicroVMRuntimeError, "tag_already_exists"):
                runtime.save_snapshot("warm")
            self.assertEqual(process.snapshots, {"inherited", "warm"})
            self.assertEqual(runtime.metrics["snapshot_saves"], 1)
        finally:
            runtime.close()

    def test_failed_save_response_deletes_uncommitted_tag_and_closes_runtime(self):
        runtime, process = self._started_fake_runtime()
        actual_hmp = runtime._hmp
        def lost_response(command, *, timeout=None):
            result = actual_hmp(command, timeout=timeout)
            if command == "savevm warm":
                raise MicroVMRuntimeError("injected_save_reply_loss")
            return result
        try:
            runtime._hmp = lost_response
            with self.assertRaisesRegex(MicroVMRuntimeError, "injected_save_reply_loss"):
                runtime.save_snapshot("warm")
            self.assertNotIn("warm", process.snapshots)
            self.assertFalse(runtime.get_state()["running"])
            self.assertEqual(runtime.metrics["snapshot_saves"], 0)
            self.assertFalse(_quarantine_marker(self.disk).exists())
        finally:
            runtime.close()

    def test_failed_save_cleanup_closes_runtime_when_tag_cannot_be_deleted(self):
        runtime, process = self._started_fake_runtime()
        actual_hmp = runtime._hmp
        def uncertain_save(command, *, timeout=None):
            if command == "delvm warm":
                raise MicroVMRuntimeError("injected_delete_failure")
            result = actual_hmp(command, timeout=timeout)
            if command == "savevm warm":
                raise MicroVMRuntimeError("injected_save_reply_loss")
            return result
        runtime._hmp = uncertain_save
        with self.assertRaisesRegex(MicroVMRuntimeError, "vm_snapshot_cleanup_unverified"):
            runtime.save_snapshot("warm")
        self.assertEqual(runtime.metrics["snapshot_saves"], 0)
        self.assertEqual(process.poll(), 0)
        self.assertFalse(runtime.get_state()["running"])
        self.assertTrue(_quarantine_marker(self.disk).is_file())
        with self.assertRaisesRegex(MicroVMRuntimeError, "qcow2_disk_quarantined"):
            self.runtime().start()
        with self.assertRaisesRegex(MicroVMRuntimeError, "qcow2_disk_quarantined"):
            _clone_or_copy_qcow2(self.disk, self.disk.with_name("child.qcow2"))
        self.assertFalse(self.disk.with_name("child.qcow2").exists())

    def test_pinned_kernel_is_checked_before_vm_launch(self):
        runtime = self.runtime(kernel_sha256="0" * 64)
        with self.assertRaisesRegex(MicroVMRuntimeError, "kernel_digest_mismatch"):
            runtime.start()
        self.assertIsNone(runtime._process)

    def test_real_vm_protocol_shape_and_full_state_snapshot_lifecycle(self):
        created = []
        def factory(args, **kwargs):
            process = FakeQEMUProcess(args, **kwargs)
            created.append(process)
            return process
        runtime = self.runtime(popen_factory=factory, readonly_disk_paths=(self.modloop,))
        def fake_artifacts():  # Fake writable disk is not qcow2.
            runtime._readonly_disk_sha256s = (_sha256_file(self.modloop),)
            runtime._readonly_disk_identities = (_file_identity(self.modloop),)
        runtime._check_artifacts = fake_artifacts
        runtime._connect = lambda path, deadline: created[0].host_sockets.pop(0)
        try:
            runtime.start()
            args = created[0].args
            self.assertEqual(args[args.index("-nic") + 1], "none")
            self.assertIn("virt,accel=hvf", args)
            self.assertFalse(any("-virtfs" in arg or "-fsdev" in arg or "-netdev" in arg
                                 for arg in args))
            self.assertTrue(any(f"file={self.modloop.resolve()},format=raw,readonly=on" in arg
                                for arg in args))
            self.assertLess(len(args[args.index("-monitor") + 1].split(",", 1)[0]), 100)
            runtime.wait_for_serial("Launching initramfs emergency recovery shell", timeout=1)
            result = runtime.run_shell("printf 'guest-ok\\n'", timeout=1)
            self.assertEqual(result, {"stdout": "guest-ok\n", "return_code": 0})
            with self.assertRaisesRegex(ValueError, "canonical TTY bound"):
                runtime.run_shell("x" * 3900, timeout=1)
            self.assertTrue(runtime.get_state()["running"])
            snapshot = runtime.save_snapshot("warm")
            self.assertEqual(snapshot["kind"], "full_vm_state_qcow2_v1")
            runtime.load_snapshot("warm")
            self.assertEqual(runtime.get_state()["metrics"]["snapshot_loads"], 1)
        finally:
            runtime.close()
        self.assertEqual(created[0].poll(), 0)
        self.assertFalse(runtime._socket_dir)

    def test_guest_timeout_stops_vm_and_cleans_control_sockets(self):
        created = []
        def factory(args, **kwargs):
            process = FakeQEMUProcess(args, silent_serial=True, **kwargs)
            created.append(process)
            return process
        runtime = self.runtime(popen_factory=factory)
        runtime._check_artifacts = lambda: None
        runtime._connect = lambda path, deadline: created[0].host_sockets.pop(0)
        runtime.start()
        runtime.wait_for_serial("Launching initramfs emergency recovery shell", timeout=1)
        with self.assertRaisesRegex(MicroVMRuntimeError, "serial_timed_out"):
            runtime.run_shell("true", timeout=0.1)
        self.assertFalse(runtime.get_state()["running"])
        self.assertIsNone(runtime._socket_dir)

    def test_readonly_guest_disk_changes_invalidate_restore(self):
        runtime = self.runtime(readonly_disk_paths=(self.modloop,))
        runtime._readonly_disk_sha256s = (hashlib.sha256(self.modloop.read_bytes()).hexdigest(),)
        self.modloop.write_bytes(b"changed guest modules")
        with self.assertRaisesRegex(MicroVMRuntimeError, "readonly_disk_changed"):
            runtime.load_snapshot("warm")

    def test_local_snapshot_fast_restore_skips_redundant_index_and_digest(self):
        created = []
        def factory(args, **kwargs):
            process = FakeQEMUProcess(args, **kwargs)
            created.append(process)
            return process
        runtime = self.runtime(popen_factory=factory, readonly_disk_paths=(self.modloop,))
        def pinned_artifacts():
            runtime._readonly_disk_sha256s = (_sha256_file(self.modloop),)
            runtime._readonly_disk_identities = (_file_identity(self.modloop),)
        runtime._check_artifacts = pinned_artifacts
        runtime._connect = lambda path, deadline: created[0].host_sockets.pop(0)
        try:
            runtime.start()
            runtime.save_snapshot("warm")
            commands = []
            actual_hmp = runtime._hmp
            def recorded_hmp(command, *, timeout=None):
                commands.append(command)
                return actual_hmp(command, timeout=timeout)
            runtime._hmp = recorded_hmp
            with patch("future_prediction_bench.microvm_runtime._sha256_file",
                       side_effect=AssertionError("unexpected digest")):
                runtime.load_snapshot("warm")
            self.assertEqual(commands, ["loadvm warm"])
            commands.clear()
            runtime.load_snapshot("warm", full_validation=True)
            self.assertEqual(commands, ["info snapshots", "loadvm warm"])
            self.assertEqual(runtime.metrics["snapshot_loads"], 2)
            commands.clear()
            with self.assertRaisesRegex(MicroVMRuntimeError, "qemu_snapshot_not_found"):
                runtime.load_snapshot("unknown")
            self.assertEqual(commands, ["info snapshots"])
            # A stale local tag must still be rejected by authoritative loadvm.
            created[0].snapshots.clear()
            commands.clear()
            with self.assertRaisesRegex(MicroVMRuntimeError, "qemu_monitor_command_failed"):
                runtime.load_snapshot("warm")
            self.assertEqual(commands, ["loadvm warm"])
            self.assertEqual(runtime.metrics["snapshot_loads"], 2)
        finally:
            runtime.close()

    def test_same_size_readonly_mutation_with_restored_mtime_still_invalidates(self):
        runtime = self.runtime(readonly_disk_paths=(self.modloop,))
        runtime._readonly_disk_sha256s = (_sha256_file(self.modloop),)
        runtime._readonly_disk_identities = (_file_identity(self.modloop),)
        old = self.modloop.stat()
        self.modloop.write_bytes(b"X" * len(b"read-only guest modules"))
        with self.assertRaisesRegex(MicroVMRuntimeError, "readonly_disk_changed"):
            runtime.load_snapshot("warm")
        # Restoring mtime cannot undo the kernel-maintained ctime/identity.
        os.utime(self.modloop, ns=(old.st_atime_ns, old.st_mtime_ns))
        with self.assertRaisesRegex(MicroVMRuntimeError, "readonly_disk_changed"):
            runtime.load_snapshot("warm")

    def test_readonly_path_replacement_invalidates_cached_identity(self):
        runtime = self.runtime(readonly_disk_paths=(self.modloop,))
        runtime._readonly_disk_sha256s = (_sha256_file(self.modloop),)
        runtime._readonly_disk_identities = (_file_identity(self.modloop),)
        replacement = self.modloop.with_name("replacement.raw")
        replacement.write_bytes(self.modloop.read_bytes())
        os.replace(replacement, self.modloop)
        with self.assertRaisesRegex(MicroVMRuntimeError, "readonly_disk_changed"):
            runtime.load_snapshot("warm")

    def _fork_fixture(self, *, fail_second_child=False):
        created = []
        def factory(args, **kwargs):
            if fail_second_child and len(created) == 2:
                raise OSError("child launch failed")
            process = FakeQEMUProcess(args, **kwargs)
            if "-S" in args:
                process.snapshots.add("warm")  # copied qcow2 internal snapshot
            created.append(process)
            return process
        runtime = self.runtime(popen_factory=factory, readonly_disk_paths=(self.modloop,))
        def fake_artifacts(instance):
            instance._readonly_disk_sha256s = tuple(
                _sha256_file(path) for path in instance.readonly_disk_paths)
        def fake_connect(instance, path, deadline):
            return instance._process.host_sockets.pop(0)
        return runtime, created, fake_artifacts, fake_connect

    def test_full_vm_fork_has_independent_qcow2_disks_and_running_children(self):
        runtime, created, fake_artifacts, fake_connect = self._fork_fixture()
        targets = [self.disk.parent / f"child-{index}.qcow2" for index in range(2)]
        with patch.object(MicroVMRuntime, "_check_artifacts", fake_artifacts), \
             patch.object(MicroVMRuntime, "_connect", fake_connect):
            runtime.start()
            runtime.save_snapshot("warm")
            children = runtime.fork_snapshot("warm", targets)
            try:
                self.assertEqual(len(children), 2)
                self.assertFalse(created[0].paused)
                self.assertTrue(all("-S" in process.args and not process.paused
                                    for process in created[1:]))
                self.assertEqual([path.read_bytes() for path in targets], [b"fake qcow2"] * 2)
                targets[0].write_bytes(b"changed child A")
                self.assertEqual(targets[1].read_bytes(), b"fake qcow2")
                self.assertEqual(self.disk.read_bytes(), b"fake qcow2")
                self.assertEqual(runtime.metrics["forks_spawned"], 2)
                self.assertGreater(runtime.metrics["fork_child_start_restore_seconds"], 0)
                self.assertEqual(runtime.metrics["fork_reflink_disks"] +
                                 runtime.metrics["fork_copied_disks"], 2)
            finally:
                for child in children:
                    child.close()
                runtime.close()

    def test_parallel_fork_overlaps_child_start_and_restore_in_input_order(self):
        runtime, created, fake_artifacts, fake_connect = self._fork_fixture()
        targets = [self.disk.parent / f"parallel-{index}.qcow2" for index in range(2)]
        start_barrier = threading.Barrier(2)
        load_barrier = threading.Barrier(2)
        parent_paused_during_restore = []
        original_start = MicroVMRuntime.start
        original_load = MicroVMRuntime.load_snapshot

        def parallel_start(child, *, paused=False):
            self.assertTrue(all(path.exists() for path in targets))
            self.assertTrue(all(path.read_bytes() == b"fake qcow2" for path in targets))
            self.assertTrue(created[0].paused)
            start_barrier.wait(timeout=5)
            return original_start(child, paused=paused)

        def parallel_load(child, tag="warm", *, resume=False):
            load_barrier.wait(timeout=5)
            parent_paused_during_restore.append(created[0].paused)
            return original_load(child, tag, resume=resume)

        with patch.object(MicroVMRuntime, "_check_artifacts", fake_artifacts), \
             patch.object(MicroVMRuntime, "_connect", fake_connect):
            runtime.start()
            runtime.save_snapshot("warm")
            with patch.object(MicroVMRuntime, "start", parallel_start), \
                 patch.object(MicroVMRuntime, "load_snapshot", parallel_load):
                children = runtime.fork_snapshot("warm", targets, parallel_children=True)
            try:
                self.assertEqual([child.disk_path for child in children],
                                 [path.resolve() for path in targets])
                self.assertEqual(parent_paused_during_restore, [True, True])
                self.assertFalse(created[0].paused)
                self.assertTrue(all(not process.paused for process in created[1:]))
                self.assertEqual(runtime.metrics["forks_spawned"], 2)
                self.assertGreater(runtime.metrics["fork_child_start_restore_seconds"], 0)
            finally:
                for child in children:
                    child.close()
                runtime.close()

    def test_parallel_fork_supports_four_child_vm_starts(self):
        runtime, created, fake_artifacts, fake_connect = self._fork_fixture()
        targets = [self.disk.parent / f"parallel-four-{index}.qcow2" for index in range(4)]
        start_barrier = threading.Barrier(4)
        original_start = MicroVMRuntime.start

        def four_way_start(child, *, paused=False):
            start_barrier.wait(timeout=5)
            return original_start(child, paused=paused)

        with patch.object(MicroVMRuntime, "_check_artifacts", fake_artifacts), \
             patch.object(MicroVMRuntime, "_connect", fake_connect):
            runtime.start()
            runtime.save_snapshot("warm")
            with patch.object(MicroVMRuntime, "start", four_way_start):
                children = runtime.fork_snapshot("warm", targets, parallel_children=True)
            try:
                self.assertEqual([child.disk_path for child in children],
                                 [path.resolve() for path in targets])
                self.assertEqual(len(created), 5)
                self.assertFalse(created[0].paused)
                self.assertEqual(runtime.metrics["forks_spawned"], 4)
            finally:
                for child in children:
                    child.close()
                runtime.close()

    def test_parallel_fork_restore_failure_closes_all_children_and_cleans_disks(self):
        runtime, created, fake_artifacts, fake_connect = self._fork_fixture()
        targets = [self.disk.parent / f"parallel-fail-{index}.qcow2" for index in range(2)]
        preserved = self.disk.parent / "preserved.qcow2"
        preserved.write_bytes(b"leave this file alone")
        load_barrier = threading.Barrier(2)
        original_load = MicroVMRuntime.load_snapshot

        def one_restore_fails(child, tag="warm", *, resume=False):
            load_barrier.wait(timeout=5)
            if child.disk_path == targets[1].resolve():
                raise MicroVMRuntimeError("injected_load_failure")
            return original_load(child, tag, resume=resume)

        with patch.object(MicroVMRuntime, "_check_artifacts", fake_artifacts), \
             patch.object(MicroVMRuntime, "_connect", fake_connect):
            runtime.start()
            runtime.save_snapshot("warm")
            try:
                with patch.object(MicroVMRuntime, "load_snapshot", one_restore_fails):
                    with self.assertRaisesRegex(MicroVMRuntimeError, "injected_load_failure"):
                        runtime.fork_snapshot("warm", targets, parallel_children=True)
                self.assertEqual(len(created), 3)
                self.assertFalse(created[0].paused)
                self.assertTrue(all(process.poll() == 0 for process in created[1:]))
                self.assertFalse(any(path.exists() for path in targets))
                self.assertEqual(preserved.read_bytes(), b"leave this file alone")
                self.assertEqual(runtime.metrics["forks_spawned"], 0)
            finally:
                runtime.close()

    def test_parallel_fork_rejects_more_than_four_children_before_parent_pause(self):
        runtime, created, fake_artifacts, fake_connect = self._fork_fixture()
        targets = [self.disk.parent / f"parallel-limit-{index}.qcow2" for index in range(5)]
        with patch.object(MicroVMRuntime, "_check_artifacts", fake_artifacts), \
             patch.object(MicroVMRuntime, "_connect", fake_connect):
            runtime.start()
            runtime.save_snapshot("warm")
            try:
                with self.assertRaisesRegex(ValueError, "at most four"):
                    runtime.fork_snapshot("warm", targets, parallel_children=True)
                self.assertFalse(created[0].paused)
                self.assertFalse(any(path.exists() for path in targets))
            finally:
                runtime.close()

    def test_fork_rejects_changed_pinned_artifact_and_preserves_parent(self):
        runtime, created, fake_artifacts, fake_connect = self._fork_fixture()
        target = self.disk.parent / "child.qcow2"
        with patch.object(MicroVMRuntime, "_check_artifacts", fake_artifacts), \
             patch.object(MicroVMRuntime, "_connect", fake_connect):
            runtime.start()
            runtime.save_snapshot("warm")
            self.kernel.write_bytes(b"mutated kernel")
            try:
                with self.assertRaisesRegex(MicroVMRuntimeError, "fork_kernel_digest_mismatch"):
                    runtime.fork_snapshot("warm", [target])
                self.assertFalse(created[0].paused)
                self.assertFalse(target.exists())
            finally:
                runtime.close()

    def test_fork_child_boot_failure_closes_siblings_and_removes_only_new_disks(self):
        runtime, created, fake_artifacts, fake_connect = self._fork_fixture(fail_second_child=True)
        targets = [self.disk.parent / f"child-{index}.qcow2" for index in range(2)]
        preserved = self.disk.parent / "preserved.qcow2"
        preserved.write_bytes(b"do not touch")
        with patch.object(MicroVMRuntime, "_check_artifacts", fake_artifacts), \
             patch.object(MicroVMRuntime, "_connect", fake_connect):
            runtime.start()
            runtime.save_snapshot("warm")
            try:
                with self.assertRaisesRegex(MicroVMRuntimeError, "vm_start_failed"):
                    runtime.fork_snapshot("warm", targets)
                self.assertFalse(created[0].paused)
                self.assertEqual(created[1].poll(), 0)
                self.assertFalse(any(path.exists() for path in targets))
                self.assertEqual(preserved.read_bytes(), b"do not touch")
            finally:
                runtime.close()

    def test_fork_requires_saved_tag_without_creating_child_disk(self):
        runtime, created, fake_artifacts, fake_connect = self._fork_fixture()
        target = self.disk.parent / "child.qcow2"
        with patch.object(MicroVMRuntime, "_check_artifacts", fake_artifacts), \
             patch.object(MicroVMRuntime, "_connect", fake_connect):
            runtime.start()
            try:
                with self.assertRaisesRegex(MicroVMRuntimeError, "qemu_snapshot_not_found"):
                    runtime.fork_snapshot("missing", [target])
                self.assertFalse(created[0].paused)
                self.assertFalse(target.exists())
            finally:
                runtime.close()

    def test_fork_rejects_broken_target_and_parent_symlinks_before_resolution(self):
        runtime, created, fake_artifacts, fake_connect = self._fork_fixture()
        broken = self.disk.parent / "broken.qcow2"
        missing = self.disk.parent / "missing.qcow2"
        broken.symlink_to(missing)
        alias = self.disk.parent / "alias"
        alias.symlink_to(self.disk.parent, target_is_directory=True)
        with patch.object(MicroVMRuntime, "_check_artifacts", fake_artifacts), \
             patch.object(MicroVMRuntime, "_connect", fake_connect):
            runtime.start()
            runtime.save_snapshot("warm")
            try:
                with self.assertRaisesRegex(ValueError, "symlinks"):
                    runtime.fork_snapshot("warm", [broken])
                with self.assertRaisesRegex(ValueError, "symlinks"):
                    runtime.fork_snapshot("warm", [alias / "child.qcow2"])
                self.assertTrue(broken.is_symlink())
                self.assertFalse(missing.exists())
                self.assertFalse((self.disk.parent / "child.qcow2").exists())
                self.assertFalse(created[0].paused)
            finally:
                runtime.close()


if __name__ == "__main__":
    unittest.main()
