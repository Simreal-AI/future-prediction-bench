"""Host-only reusable QEMU template contracts with a fake monitor transport."""

import copy
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import future_prediction_bench.microvm_template as template_module
from examples.realworld_boltons26.benchmark_template_hash_ab import (
    _legacy_spawn, _same_semantics,
)
from future_prediction_bench.microvm_runtime import MicroVMRuntime, MicroVMRuntimeError, _sha256_file
from future_prediction_bench.microvm_template import MicroVMTemplate
from test_microvm_runtime import FakeQEMUProcess


class MicroVMTemplateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.kernel = self.root / "kernel"
        self.initramfs = self.root / "initramfs"
        self.modloop = self.root / "modloop"
        self.parent_disk = self.root / "parent.qcow2"
        for path, payload in ((self.kernel, b"kernel"),
                              (self.initramfs, b"initramfs"),
                              (self.modloop, b"modloop"),
                              (self.parent_disk, b"qcow2 before export")):
            path.write_bytes(payload)
        self.created = []

    def _runtime(self):
        def factory(args, **kwargs):
            process = FakeQEMUProcess(args, **kwargs)
            if "-S" in args:
                process.snapshots.add("warm")
            self.created.append(process)
            return process
        self.factory = factory
        return MicroVMRuntime(
            self.kernel, self.initramfs, self.parent_disk,
            kernel_sha256=_sha256_file(self.kernel),
            initramfs_sha256=_sha256_file(self.initramfs),
            readonly_disk_paths=(self.modloop,), popen_factory=factory)

    @staticmethod
    def _fake_artifacts(instance):
        instance._readonly_disk_sha256s = tuple(
            _sha256_file(path) for path in instance.readonly_disk_paths)

    @staticmethod
    def _fake_connect(instance, path, deadline):
        return instance._process.host_sockets.pop(0)

    def test_template_survives_parent_exit_and_spawns_independent_writable_children(self):
        parent = self._runtime()
        with patch.object(MicroVMRuntime, "_check_artifacts", self._fake_artifacts), \
             patch.object(MicroVMRuntime, "_connect", self._fake_connect):
            parent.start()
            template = MicroVMTemplate.export(parent, self.root / "warm.qcow2", tag="warm")
            self.assertFalse(self.created[0].paused)
            self.assertGreater(template.export_seconds, 0)
            self.assertIn(template.export_clone_mode, {"clonefile", "copy"})
            self.assertEqual(template.disk_path.read_bytes(), b"qcow2 before export")
            self.assertFalse(os.stat(template.disk_path).st_mode & 0o222)
            parent.close()
            self.parent_disk.write_bytes(b"parent changed after export")
            reopened = MicroVMTemplate.open(template.manifest_path,
                                            expected_template_id=template.template_id)
            targets = [self.root / f"child-{index}.qcow2" for index in range(2)]
            children = reopened.spawn(targets, max_workers=2, popen_factory=self.factory)
            try:
                self.assertEqual(len(children), 2)
                self.assertEqual([path.read_bytes() for path in targets],
                                 [b"qcow2 before export"] * 2)
                self.assertTrue(all(os.stat(path).st_mode & 0o200 for path in targets))
                self.assertTrue(all(not process.paused for process in self.created[1:]))
                targets[0].write_bytes(b"child A changed")
                self.assertEqual(targets[1].read_bytes(), b"qcow2 before export")
                self.assertEqual(template.disk_path.read_bytes(), b"qcow2 before export")
                self.assertEqual(self.parent_disk.read_bytes(), b"parent changed after export")
            finally:
                for child in children:
                    child.close()

    def test_failed_export_cleans_template_and_resumes_parent(self):
        parent = self._runtime()
        target = self.root / "bad.qcow2"
        with patch.object(MicroVMRuntime, "_check_artifacts", self._fake_artifacts), \
             patch.object(MicroVMRuntime, "_connect", self._fake_connect):
            parent.start()
            try:
                with patch("future_prediction_bench.microvm_template._clone_or_copy_qcow2",
                           side_effect=MicroVMRuntimeError("injected_clone_failure")):
                    with self.assertRaisesRegex(MicroVMRuntimeError, "injected_clone_failure"):
                        MicroVMTemplate.export(parent, target)
                self.assertFalse(self.created[0].paused)
                self.assertFalse(target.exists())
                self.assertFalse(Path(str(target) + ".json").exists())
            finally:
                parent.close()

    def test_retired_action_port_parent_rejected_before_template_side_effects(self):
        parent = MicroVMRuntime(
            self.kernel, self.initramfs, self.parent_disk,
            kernel_sha256=_sha256_file(self.kernel),
            initramfs_sha256=_sha256_file(self.initramfs),
            enable_action_port=True)
        parent._action_port_retired = True
        parent._action_port_disconnect_attested = True
        target = self.root / "forbidden.qcow2"
        with patch.object(parent, "_check_running", side_effect=AssertionError("side effect")):
            with self.assertRaisesRegex(MicroVMRuntimeError, "template_action_port_not_supported"):
                MicroVMTemplate.export(parent, target)
        self.assertFalse(target.exists())
        self.assertFalse(Path(str(target) + ".json").exists())

    def test_tamper_and_unsafe_targets_are_rejected_before_spawning(self):
        parent = self._runtime()
        with patch.object(MicroVMRuntime, "_check_artifacts", self._fake_artifacts), \
             patch.object(MicroVMRuntime, "_connect", self._fake_connect):
            parent.start()
            template = MicroVMTemplate.export(parent, self.root / "warm.qcow2")
            parent.close()
            with self.assertRaisesRegex(MicroVMRuntimeError, "template_id_mismatch"):
                MicroVMTemplate.open(template.manifest_path,
                                     expected_template_id="0" * 64)
            alias = self.root / "alias.qcow2"
            alias.symlink_to(self.root / "missing.qcow2")
            with self.assertRaisesRegex(ValueError, "symlink"):
                template.spawn([alias], popen_factory=self.factory)
            with self.assertRaisesRegex(ValueError, "overlaps"):
                template.spawn([template.disk_path], popen_factory=self.factory)
            with self.assertRaisesRegex(ValueError, "unique"):
                template.spawn([self.root / "same.qcow2"] * 2,
                               popen_factory=self.factory)
            template.disk_path.chmod(0o600)
            template.disk_path.write_bytes(b"tampered")
            tampered_child = self.root / "new.qcow2"
            with self.assertRaisesRegex(MicroVMRuntimeError,
                                        "template_child_clone_digest_mismatch"):
                template.spawn([tampered_child], popen_factory=self.factory)
            self.assertFalse(tampered_child.exists())
            self.assertEqual(len(self.created), 1)

    def test_failed_restore_closes_children_and_removes_only_its_disks(self):
        parent = self._runtime()
        with patch.object(MicroVMRuntime, "_check_artifacts", self._fake_artifacts), \
             patch.object(MicroVMRuntime, "_connect", self._fake_connect):
            parent.start()
            template = MicroVMTemplate.export(parent, self.root / "warm.qcow2")
            parent.close()
            targets = [self.root / "child-0.qcow2", self.root / "child-1.qcow2"]
            original_load = MicroVMRuntime.load_snapshot

            def fail_second(child, tag="warm", *, resume=False):
                if child.disk_path == targets[1].resolve():
                    raise MicroVMRuntimeError("injected_restore_failure")
                return original_load(child, tag, resume=resume)

            with patch.object(MicroVMRuntime, "load_snapshot", fail_second):
                with self.assertRaisesRegex(MicroVMRuntimeError,
                                            "injected_restore_failure"):
                    template.spawn(targets, max_workers=2, popen_factory=self.factory)
            self.assertFalse(any(path.exists() for path in targets))
            self.assertEqual(template.disk_path.read_bytes(), b"qcow2 before export")
            self.assertTrue(all(process.poll() == 0 for process in self.created[1:]))

    def test_spawn_hashes_every_independent_child_without_rehashing_source(self):
        parent = self._runtime()
        with patch.object(MicroVMRuntime, "_check_artifacts", self._fake_artifacts), \
             patch.object(MicroVMRuntime, "_connect", self._fake_connect):
            parent.start()
            template = MicroVMTemplate.export(parent, self.root / "warm.qcow2")
            parent.close()
            targets = [self.root / "child-0.qcow2", self.root / "child-1.qcow2"]
            hashed = []
            original_hash = template_module._sha256_file

            def record_hash(path):
                hashed.append(Path(path))
                return original_hash(path)

            with patch.object(template_module, "_sha256_file", record_hash):
                children = template.spawn(targets, max_workers=2,
                                          popen_factory=self.factory)
            try:
                self.assertEqual(hashed.count(template.disk_path), 0)
                self.assertEqual([hashed.count(path.resolve()) for path in targets],
                                 [1, 1])
            finally:
                for child in children:
                    child.close()

    def test_public_ab_legacy_arm_replays_three_source_hashes(self):
        parent = self._runtime()
        with patch.object(MicroVMRuntime, "_check_artifacts", self._fake_artifacts), \
             patch.object(MicroVMRuntime, "_connect", self._fake_connect):
            parent.start()
            template = MicroVMTemplate.export(parent, self.root / "warm.qcow2")
            parent.close()
            target = self.root / "child.qcow2"
            hashed = []
            original_hash = template_module._sha256_file

            def record_hash(path):
                hashed.append(Path(path))
                return original_hash(path)

            with patch.object(template_module, "_sha256_file", record_hash):
                children = _legacy_spawn(template, [target],
                                         popen_factory=self.factory)
            try:
                self.assertEqual(hashed.count(template.disk_path), 3)
                self.assertEqual(hashed.count(target.resolve()), 1)
            finally:
                for child in children:
                    child.close()

    def test_public_ab_comparator_requires_exact_actions_and_private_cases(self):
        row = {
            "method": "replace_text", "branch": "repair", "condition": "prepared",
            "task_sha256": "a" * 64,
            "opening_observation": {"status": "ready"},
            "action_observation_sha256s": ["b" * 64],
            "case_results": [{"passed": True, "stdout_sha256": "c" * 64}],
            "reward": 1.0, "passed_cases": 1,
            "final_source_sha256": "d" * 64,
            "adapter_metrics": {"full_vm_restores": 1, "stateless_batches": 1},
        }
        left = {"runs": [dict(copy.deepcopy(row), method=method, branch=branch,
                              condition=condition)
                         for method in ("full_write", "replace_text")
                         for branch in ("repair", "baseline")
                         for condition in ("cold", "prepared")]}
        right = copy.deepcopy(left)
        _same_semantics(left, right)
        right["runs"][0]["case_results"][0]["stdout_sha256"] = "e" * 64
        with self.assertRaisesRegex(RuntimeError, "case_results"):
            _same_semantics(left, right)
        right = copy.deepcopy(left)
        right["runs"][0]["action_observation_sha256s"] = ["f" * 64]
        with self.assertRaisesRegex(RuntimeError, "action_observation_sha256s"):
            _same_semantics(left, right)

    def test_source_change_during_clone_rejects_child_digest(self):
        parent = self._runtime()
        with patch.object(MicroVMRuntime, "_check_artifacts", self._fake_artifacts), \
             patch.object(MicroVMRuntime, "_connect", self._fake_connect):
            parent.start()
            template = MicroVMTemplate.export(parent, self.root / "warm.qcow2")
            parent.close()
            target = self.root / "child.qcow2"
            original_clone = template_module._clone_or_copy_qcow2

            def change_source_then_clone(source, destination):
                source.chmod(0o600)
                source.write_bytes(b"changed after initial template check")
                return original_clone(source, destination)

            with patch.object(template_module, "_clone_or_copy_qcow2",
                              change_source_then_clone):
                with self.assertRaisesRegex(MicroVMRuntimeError,
                                            "template_child_clone_digest_mismatch"):
                    template.spawn([target], popen_factory=self.factory)
            self.assertFalse(target.exists())
            self.assertEqual(len(self.created), 1)
            with self.assertRaisesRegex(MicroVMRuntimeError,
                                        "template_child_clone_digest_mismatch"):
                template.spawn([target], popen_factory=self.factory)

    def test_source_change_after_clone_preserves_child_and_blocks_next_spawn(self):
        parent = self._runtime()
        with patch.object(MicroVMRuntime, "_check_artifacts", self._fake_artifacts), \
             patch.object(MicroVMRuntime, "_connect", self._fake_connect):
            parent.start()
            template = MicroVMTemplate.export(parent, self.root / "warm.qcow2")
            parent.close()
            target = self.root / "child.qcow2"
            original_clone = template_module._clone_or_copy_qcow2

            def clone_then_change_source(source, destination):
                mode = original_clone(source, destination)
                source.chmod(0o600)
                source.write_bytes(b"changed after independent clone")
                return mode

            with patch.object(template_module, "_clone_or_copy_qcow2",
                              clone_then_change_source):
                children = template.spawn([target], popen_factory=self.factory)
            try:
                self.assertEqual(target.read_bytes(), b"qcow2 before export")
                with self.assertRaisesRegex(MicroVMRuntimeError,
                                            "template_child_clone_digest_mismatch"):
                    template.spawn([self.root / "next-child.qcow2"],
                                   popen_factory=self.factory)
            finally:
                for child in children:
                    child.close()

    def test_corrupt_child_clone_is_rejected_before_any_vm_boot(self):
        parent = self._runtime()
        with patch.object(MicroVMRuntime, "_check_artifacts", self._fake_artifacts), \
             patch.object(MicroVMRuntime, "_connect", self._fake_connect):
            parent.start()
            template = MicroVMTemplate.export(parent, self.root / "warm.qcow2")
            parent.close()
            target = self.root / "child.qcow2"
            original_clone = template_module._clone_or_copy_qcow2

            def clone_then_corrupt_child(source, destination):
                mode = original_clone(source, destination)
                destination.chmod(0o600)
                destination.write_bytes(b"corrupt child")
                return mode

            with patch.object(template_module, "_clone_or_copy_qcow2",
                              clone_then_corrupt_child):
                with self.assertRaisesRegex(MicroVMRuntimeError,
                                            "template_child_clone_digest_mismatch"):
                    template.spawn([target], popen_factory=self.factory)
            self.assertFalse(target.exists())
            self.assertEqual(len(self.created), 1)

    def test_source_mutation_between_two_clones_blocks_both_before_boot(self):
        parent = self._runtime()
        with patch.object(MicroVMRuntime, "_check_artifacts", self._fake_artifacts), \
             patch.object(MicroVMRuntime, "_connect", self._fake_connect):
            parent.start()
            template = MicroVMTemplate.export(parent, self.root / "warm.qcow2")
            parent.close()
            targets = [self.root / "child-0.qcow2", self.root / "child-1.qcow2"]
            original_clone = template_module._clone_or_copy_qcow2
            clone_count = 0

            def mutate_after_first_clone(source, destination):
                nonlocal clone_count
                mode = original_clone(source, destination)
                clone_count += 1
                if clone_count == 1:
                    source.chmod(0o600)
                    source.write_bytes(b"changed between child clones")
                return mode

            with patch.object(template_module, "_clone_or_copy_qcow2",
                              mutate_after_first_clone):
                with self.assertRaisesRegex(MicroVMRuntimeError,
                                            "template_child_clone_digest_mismatch"):
                    template.spawn(targets, max_workers=2,
                                   popen_factory=self.factory)
            self.assertEqual(clone_count, 2)
            self.assertFalse(any(path.exists() for path in targets))
            self.assertEqual(len(self.created), 1)


if __name__ == "__main__":
    unittest.main()
