"""The opt-in template fan-out retains the all-children verification barrier."""

import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import future_prediction_bench.microvm_template as template_module
from future_prediction_bench.microvm_runtime import MicroVMRuntime, MicroVMRuntimeError, _sha256_file
from future_prediction_bench.microvm_template import MicroVMTemplate
from test_microvm_runtime import FakeQEMUProcess


class ParallelTemplateProvisionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.kernel = self.root / "kernel"
        self.initramfs = self.root / "initramfs"
        self.modloop = self.root / "modloop"
        self.parent_disk = self.root / "parent.qcow2"
        for path, payload in ((self.kernel, b"kernel"),
                              (self.initramfs, b"initramfs"),
                              (self.modloop, b"modloop"),
                              (self.parent_disk, b"qcow2 before export")):
            path.write_bytes(payload)
        self.processes = []

    def _factory(self, args, **kwargs):
        process = FakeQEMUProcess(args, **kwargs)
        if "-S" in args:
            process.snapshots.add("warm")
        self.processes.append(process)
        return process

    @staticmethod
    def _fake_artifacts(runtime):
        runtime._readonly_disk_sha256s = tuple(
            _sha256_file(path) for path in runtime.readonly_disk_paths)

    @staticmethod
    def _fake_connect(runtime, path, deadline):
        return runtime._process.host_sockets.pop(0)

    def _template(self):
        parent = MicroVMRuntime(
            self.kernel, self.initramfs, self.parent_disk,
            kernel_sha256=_sha256_file(self.kernel),
            initramfs_sha256=_sha256_file(self.initramfs),
            readonly_disk_paths=(self.modloop,), popen_factory=self._factory)
        parent.start()
        try:
            return MicroVMTemplate.export(parent, self.root / "warm.qcow2")
        finally:
            parent.close()

    def test_four_clones_execute_concurrently_and_all_hash_before_boot(self):
        with patch.object(MicroVMRuntime, "_check_artifacts", self._fake_artifacts), \
             patch.object(MicroVMRuntime, "_connect", self._fake_connect):
            template = self._template()
            paths = [self.root / f"child-{index}.qcow2" for index in range(4)]
            resolved_paths = {path.resolve() for path in paths}
            original_clone = template_module._clone_or_copy_qcow2
            original_hash = template_module._sha256_file
            rendezvous = threading.Barrier(4, timeout=3)
            hashed = set()
            hashed_lock = threading.Lock()

            def overlapping_clone(source, target):
                mode = original_clone(source, target)
                rendezvous.wait()
                return mode

            def record_hash(path):
                value = original_hash(path)
                if path in resolved_paths:
                    with hashed_lock:
                        hashed.add(path)
                return value

            def check_start(child, *, paused=False):
                self.assertEqual(hashed, resolved_paths)
                return original_start(child, paused=paused)

            original_start = MicroVMRuntime.start
            with patch.object(template_module, "_clone_or_copy_qcow2", overlapping_clone), \
                 patch.object(template_module, "_sha256_file", record_hash), \
                 patch.object(MicroVMRuntime, "start", check_start):
                children = template.spawn(paths, max_workers=4,
                                          popen_factory=self._factory,
                                          parallel_clone_verification=True)
            try:
                self.assertEqual(len(children), 4)
                self.assertEqual([path.read_bytes() for path in paths],
                                 [b"qcow2 before export"] * 4)
                paths[0].write_bytes(b"only child zero changed")
                self.assertEqual(paths[1].read_bytes(), b"qcow2 before export")
                self.assertEqual(template.disk_path.read_bytes(), b"qcow2 before export")
            finally:
                for child in children:
                    child.close()

    def test_one_bad_parallel_clone_blocks_all_child_boots_and_cleans_batch(self):
        with patch.object(MicroVMRuntime, "_check_artifacts", self._fake_artifacts), \
             patch.object(MicroVMRuntime, "_connect", self._fake_connect):
            template = self._template()
            paths = [self.root / f"bad-child-{index}.qcow2" for index in range(3)]
            original_clone = template_module._clone_or_copy_qcow2
            rendezvous = threading.Barrier(3, timeout=3)
            calls = 0
            call_lock = threading.Lock()

            def one_bad_clone(source, target):
                nonlocal calls
                with call_lock:
                    ordinal = calls
                    calls += 1
                mode = original_clone(source, target)
                rendezvous.wait()
                if ordinal == 1:
                    target.chmod(0o600)
                    target.write_bytes(b"corrupted child")
                return mode

            with patch.object(template_module, "_clone_or_copy_qcow2", one_bad_clone):
                with self.assertRaisesRegex(MicroVMRuntimeError,
                                            "template_child_clone_digest_mismatch"):
                    template.spawn(paths, max_workers=3, popen_factory=self._factory,
                                   parallel_clone_verification=True)
            self.assertFalse(any(path.exists() for path in paths))
            self.assertEqual(len(self.processes), 1)  # Export parent only.
            self.assertEqual(template.disk_path.read_bytes(), b"qcow2 before export")

    def test_clone_exception_drains_siblings_before_cleanup(self):
        with patch.object(MicroVMRuntime, "_check_artifacts", self._fake_artifacts), \
             patch.object(MicroVMRuntime, "_connect", self._fake_connect):
            template = self._template()
            paths = [self.root / f"exception-child-{index}.qcow2"
                     for index in range(3)]
            original_clone = template_module._clone_or_copy_qcow2
            rendezvous = threading.Barrier(3, timeout=3)
            drained = set()
            lock = threading.Lock()
            calls = 0

            def one_clone_raises(source, target):
                nonlocal calls
                with lock:
                    ordinal = calls
                    calls += 1
                if ordinal == 0:
                    target.write_bytes(b"partial private clone")
                else:
                    original_clone(source, target)
                rendezvous.wait()
                with lock:
                    drained.add(target)
                if ordinal == 0:
                    raise MicroVMRuntimeError("injected_clone_failure")
                return "clonefile"

            with patch.object(template_module, "_clone_or_copy_qcow2", one_clone_raises):
                with self.assertRaisesRegex(MicroVMRuntimeError,
                                            "injected_clone_failure"):
                    template.spawn(paths, max_workers=3, popen_factory=self._factory,
                                   parallel_clone_verification=True)
            self.assertEqual(len(drained), 3)
            self.assertFalse(any(path.exists() for path in paths))
            self.assertFalse(list(self.root.glob(".fpb-template-child-*")))
            self.assertEqual(len(self.processes), 1)

    def test_raced_external_target_survives_failed_no_replace_publish(self):
        with patch.object(MicroVMRuntime, "_check_artifacts", self._fake_artifacts), \
             patch.object(MicroVMRuntime, "_connect", self._fake_connect):
            template = self._template()
            target = self.root / "raced-external.qcow2"
            original_clone = template_module._clone_or_copy_qcow2

            def external_writer_after_clone(source, private_target):
                mode = original_clone(source, private_target)
                target.write_bytes(b"unrelated external file")
                return mode

            with patch.object(template_module, "_clone_or_copy_qcow2",
                              external_writer_after_clone):
                with self.assertRaises(FileExistsError):
                    template.spawn([target], popen_factory=self._factory,
                                   parallel_clone_verification=True)
            self.assertEqual(target.read_bytes(), b"unrelated external file")
            self.assertFalse(list(self.root.glob(".fpb-template-child-*")))
            self.assertEqual(len(self.processes), 1)

    def test_parallel_option_requires_a_boolean(self):
        with patch.object(MicroVMRuntime, "_check_artifacts", self._fake_artifacts), \
             patch.object(MicroVMRuntime, "_connect", self._fake_connect):
            template = self._template()
            target = self.root / "child.qcow2"
            with self.assertRaisesRegex(ValueError, "parallel_clone_verification"):
                template.spawn([target], parallel_clone_verification=1)
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
