"""Host-coordinator restart faults for a committed full-VM terminal result."""

from __future__ import annotations

import copy
import hashlib
import json
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from future_prediction_bench.semantic_vm_recovery import (
    CodingVMRecoveryJournal, RecoveryError,
)
from future_prediction_bench.realworld import validate_task


class _Process:
    pid = 99_999_991

    def poll(self):
        return None


class _VM:
    def __init__(self, disk, saved=None, *, fresh=False):
        self.disk_path = disk
        self.kernel_path = disk.parent / "kernel"
        self.initramfs_path = disk.parent / "initramfs"
        self.readonly_disk_paths = ()
        self.kernel_sha256 = "a" * 64
        self.initramfs_sha256 = "b" * 64
        self.memory_mib = 128
        self.vcpus = 1
        self.kernel_append = "console=ttyAMA0"
        self.enable_action_port = False
        self.qemu_binary = "qemu-system-aarch64"
        self.qemu_img_binary = "qemu-img"
        self.command_timeout = 30.0
        self._readonly_disk_sha256s = ()
        self._process = None if fresh else _Process()
        self.state = {"source": "original", "case_marker": None}
        self.saved = {} if saved is None else saved
        self.starts = 0
        self.loads = 0

    def save_snapshot(self, tag):
        self.saved[tag] = copy.deepcopy(self.state)
        return {"kind": "full_vm_state_qcow2_v1", "tag": tag,
                "disk_path": str(self.disk_path),
                "kernel_sha256": self.kernel_sha256,
                "initramfs_sha256": self.initramfs_sha256,
                "readonly_disk_sha256s": []}

    def start(self, *, paused=False):
        if not paused:
            raise ValueError("restart must boot paused")
        self.starts += 1
        self._process = _Process()

    def load_snapshot(self, tag, *, resume=False, full_validation=False):
        self.loads += 1
        self.state = copy.deepcopy(self.saved[tag])

    def close(self):
        self._process = None


class _Adapter:
    def __init__(self, runtime, binding, verifier_dir, *, fresh=False):
        self.runtime = runtime
        self.verifier_dir = verifier_dir
        self.visible_check = ("python3", "-B", "-c", "import target")
        self.workspace_root = "/mnt/root/workspace"
        self.read_transport = "serial_shell"
        self.preinstalled_read_agent = False
        self.command_timeout = 30.0
        self.stateless_verifier = None
        self._binding = copy.deepcopy(binding)
        self.started = not fresh
        self.submitted = False
        self.expected_binding = None if fresh else copy.deepcopy(binding)
        self.snapshot = None
        self.verified = None
        self.submit_calls = 0
        self.verify_calls = 0

    def artifact_binding(self):
        return copy.deepcopy(self._binding)

    def step(self, action, *, now):
        if action["action"] == "submit":
            self.submit_calls += 1
            self.snapshot = self.runtime.save_snapshot("submitted")
            self.submitted = True
            return {"observation": {"status": "submitted",
                                    "snapshot_kind": "full_vm_state_qcow2_v1"},
                    "terminated": True}
        raise ValueError("unexpected policy action")

    def verify(self, *, now):
        self.verify_calls += 1
        self.runtime.load_snapshot("submitted")
        if self.runtime.state["case_marker"] is not None:
            raise AssertionError("interrupted hidden case leaked into retry")
        self.verified = {"status": "resolved", "reward": 1.0,
                         "evidence": {"source": self.runtime.state["source"]}}
        return copy.deepcopy(self.verified)


def _task(binding):
    return {"schema_version": "realworld-0.1", "task_id": "host-restart-fixture",
            "event_id": "host-restart", "cluster_id": "host-restart",
            "split": "dev", "prompt": "Repair and submit.",
            "issued_at": "2026-01-01T00:00:00+00:00",
            "action_deadline": "2027-01-01T00:00:00+00:00",
            "outcome_not_before": "2026-01-01T00:00:00+00:00",
            "verify_after": "2026-01-01T00:00:00+00:00",
            "tool_manifest": [{"name": "submit", "description": "Submit repair"}],
            "reward_contract": {"id": "fixed", "description": "Private tests",
                                "min_reward": 0, "max_reward": 1},
            "budgets": {"max_actions": 10, "max_wall_seconds": 600},
            "is_fixture": True, "metadata": {"artifact_binding": copy.deepcopy(binding)}}


class HostRestartTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.disk = self.root / "vm.qcow2"
        self.disk.write_bytes(b"fake qcow2; changed only through fake snapshots")
        self.verifier_dir = self.root / "private-verifier"
        self.verifier_dir.mkdir()
        self.binding = {"runtime_kind": "qemu_hvf_full_vm_qcow2_v1",
                        "disk_seed_sha256": "c" * 64,
                        "readonly_disk_sha256s": [],
                        "verifier_sha256": "d" * 64}
        self.task = _task(self.binding)
        self.vm = _VM(self.disk)
        self.adapter = _Adapter(self.vm, self.binding, self.verifier_dir)
        self.adapter._task_sha256 = validate_task(self.task)["task_sha256"]
        self.journal = CodingVMRecoveryJournal(
            self.adapter, self.root / "journal", inspector=lambda: {
                "processes": [], "tree": "unchanged"})
        self.journal.begin()
        self.journal.enable_host_restart(self.task)

    def test_enable_rejects_different_prompt_with_identical_artifact_binding(self):
        # Reproduce the first opt-in call after reset, not a later re-arm.
        self.journal._host_restart_task = None
        changed = copy.deepcopy(self.task)
        changed["prompt"] = "A different policy task with the same VM artifacts."
        with self.assertRaisesRegex(RecoveryError, "host_restart_task_differs_from_reset"):
            self.journal.enable_host_restart(changed)
        self.assertFalse(self.journal.host_restart_path.exists())
        self.assertIsNone(self.journal._host_restart_task)

    def _fresh(self, *, binding=None, disk=None):
        vm = _VM(disk or self.disk, self.vm.saved, fresh=True)
        adapter = _Adapter(vm, binding or self.binding, self.verifier_dir, fresh=True)
        journal = CodingVMRecoveryJournal(adapter, self.journal.directory,
                                          inspector=lambda: {"processes": [],
                                                             "tree": "unchanged"})
        return vm, adapter, journal

    def _submit_without_result(self):
        def crash_in_case(*, now):
            self.vm.state["case_marker"] = "partial-hidden-case"
            raise KeyboardInterrupt("host coordinator killed")

        self.adapter.verify = crash_in_case
        with self.assertRaises(KeyboardInterrupt):
            self.journal.submit_and_verify(now=None)
        self.assertTrue(self.journal.terminal_submit_path.is_file())
        self.assertFalse(self.journal.terminal_result_path.exists())
        self.assertEqual(self.adapter.submit_calls, 1)

    def test_fresh_process_rebuilds_adapter_and_retries_hidden_case_once(self):
        self._submit_without_result()
        vm, adapter, journal = self._fresh()
        with mock.patch.object(journal, "_old_host_vm_stopped", return_value=True):
            result = journal.resume_verification_after_host_restart(self.task, now=None)
        self.assertEqual(result["grading"]["reward"], 1.0)
        self.assertEqual(result["grading"]["evidence"]["source"], "original")
        self.assertEqual(vm.starts, 1)
        self.assertEqual(adapter.verify_calls, 1)
        self.assertEqual(adapter.submit_calls, 0)
        self.assertEqual(self.adapter.submit_calls, 1)
        self.assertTrue(journal.terminal_result_path.is_file())
        self.assertEqual(journal.submit_and_verify(now=None), result)

    def test_live_old_qemu_and_changed_disk_fail_before_start(self):
        self._submit_without_result()
        vm, _, journal = self._fresh()
        with mock.patch.object(journal, "_old_host_vm_stopped", return_value=False):
            with self.assertRaisesRegex(RecoveryError, "old_host_qemu_still_running"):
                journal.resume_verification_after_host_restart(self.task, now=None)
        self.assertEqual(vm.starts, 0)
        other_disk = self.root / "replaced.qcow2"
        other_disk.write_bytes(b"new inode")
        self.disk.unlink()
        other_disk.rename(self.disk)
        vm, _, journal = self._fresh()
        with self.assertRaisesRegex(RecoveryError, "host_restart_frozen_identity_mismatch"):
            journal.resume_verification_after_host_restart(self.task, now=None)
        self.assertEqual(vm.starts, 0)

    def test_changed_task_binding_and_missing_marker_fail_closed(self):
        self._submit_without_result()
        vm, _, journal = self._fresh()
        changed = copy.deepcopy(self.task)
        changed["prompt"] = "Another task."
        with self.assertRaisesRegex(RecoveryError, "host_restart_frozen_identity_mismatch"):
            journal.resume_verification_after_host_restart(changed, now=None)
        self.assertEqual(vm.starts, 0)
        self.journal.terminal_submit_path.unlink()
        vm, _, journal = self._fresh()
        with mock.patch.object(journal, "_old_host_vm_stopped", return_value=True):
            with self.assertRaisesRegex(RecoveryError, "terminal_submit_record_corrupt"):
                journal.resume_verification_after_host_restart(self.task, now=None)
        self.assertEqual(vm.starts, 0)

    def test_durable_result_reloads_without_vm_boot(self):
        original = self.journal.submit_and_verify(now=None)
        vm, adapter, journal = self._fresh()
        with mock.patch.object(journal, "_old_host_vm_stopped", return_value=True):
            reloaded = journal.resume_verification_after_host_restart(
                validate_task(self.task), now=None)
        self.assertEqual(reloaded, original)
        self.assertEqual(vm.starts, 0)
        self.assertEqual(adapter.submit_calls, 0)
        self.assertEqual(adapter.verify_calls, 0)

    def test_changed_digest_stamped_task_fails_before_boot(self):
        self._submit_without_result()
        stamped = validate_task(self.task)
        stamped["prompt"] = "Tampered after validation"
        vm, _, journal = self._fresh()
        with mock.patch.object(journal, "_old_host_vm_stopped", return_value=True):
            with self.assertRaisesRegex(RecoveryError, "host_restart_task_or_assets_invalid"):
                journal.resume_verification_after_host_restart(stamped, now=None)
        self.assertEqual(vm.starts, 0)

    def test_corrupt_private_result_or_context_never_releases_reward(self):
        self.journal.submit_and_verify(now=None)
        result_path = self.journal.terminal_result_path
        result = json.loads(result_path.read_text())
        result["grading"]["reward"] = 0.0
        result_path.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
        vm, _, journal = self._fresh()
        with mock.patch.object(journal, "_old_host_vm_stopped", return_value=True):
            with self.assertRaisesRegex(RecoveryError, "terminal_result_record_corrupt"):
                journal.resume_verification_after_host_restart(self.task, now=None)
        self.assertEqual(vm.starts, 0)
        self.journal.host_restart_path.write_text("{}\n")
        vm, _, journal = self._fresh()
        with self.assertRaisesRegex(RecoveryError, "host_restart_context_record_corrupt"):
            journal.resume_verification_after_host_restart(self.task, now=None)
        self.assertEqual(vm.starts, 0)

    def test_context_records_current_qemu_after_prior_vm_replacement(self):
        # enable_host_restart arms the episode; publication occurs just before
        # submit so an earlier VM-process recovery cannot leave a stale PID.
        self.vm._process.pid = 99_999_992
        self.journal.submit_and_verify(now=None)
        context = json.loads(self.journal.host_restart_path.read_text())
        self.assertEqual(context["old_qemu_pid"], 99_999_992)
        self.assertEqual(stat.S_IMODE(self.journal.host_restart_path.stat().st_mode), 0o600)

    def test_context_binds_latest_committed_action_manifest(self):
        self.journal.apply_opaque(
            lambda: self.vm.state.update(source="repaired") or {"done": True})
        terminal = self.journal.submit_and_verify(now=None)
        context = json.loads(self.journal.host_restart_path.read_text())
        self.assertEqual(context["journal_sha256"],
                         hashlib.sha256(self.journal.path.read_bytes()).hexdigest())
        self.assertEqual(terminal["grading"]["evidence"]["source"], "repaired")

    def test_missing_opt_in_context_fails_before_boot(self):
        self.journal._host_restart_task = None
        self.journal.submit_and_verify(now=None)
        vm, _, journal = self._fresh()
        with self.assertRaisesRegex(RecoveryError, "host_restart_context_record_corrupt"):
            journal.resume_verification_after_host_restart(self.task, now=None)
        self.assertEqual(vm.starts, 0)

    def test_changed_private_verifier_binding_fails_before_boot(self):
        self._submit_without_result()
        vm, adapter, journal = self._fresh()
        adapter._binding["verifier_sha256"] = "e" * 64
        with mock.patch.object(journal, "_old_host_vm_stopped", return_value=True):
            with self.assertRaisesRegex(RecoveryError, "host_restart_frozen_identity_mismatch"):
                journal.resume_verification_after_host_restart(self.task, now=None)
        self.assertEqual(vm.starts, 0)

    def test_changed_timeout_contract_fails_before_boot(self):
        self._submit_without_result()
        for field in ("runtime", "adapter"):
            vm, adapter, journal = self._fresh()
            target = vm if field == "runtime" else adapter
            target.command_timeout = 5.0
            with mock.patch.object(journal, "_old_host_vm_stopped", return_value=True):
                with self.assertRaisesRegex(RecoveryError, "host_restart_frozen_identity_mismatch"):
                    journal.resume_verification_after_host_restart(self.task, now=None)
            self.assertEqual(vm.starts, 0)

    def test_replacement_start_failure_requires_new_runtime_for_retry(self):
        self._submit_without_result()
        vm, _, journal = self._fresh()
        with mock.patch.object(journal, "_old_host_vm_stopped", return_value=True):
            with mock.patch.object(vm, "start", side_effect=RuntimeError("boot failed")):
                with self.assertRaisesRegex(RuntimeError, "boot failed"):
                    journal.resume_verification_after_host_restart(self.task, now=None)
        self.assertFalse(journal.terminal_result_path.exists())
        vm2, adapter2, journal2 = self._fresh()
        with mock.patch.object(journal2, "_old_host_vm_stopped", return_value=True):
            result = journal2.resume_verification_after_host_restart(self.task, now=None)
        self.assertEqual(result["grading"]["reward"], 1.0)
        self.assertEqual(vm2.starts, 1)
        self.assertEqual(adapter2.submit_calls, 0)

    def test_attempt_publication_failure_cannot_be_reopened_as_submitted(self):
        original = self.journal._publish_terminal_record

        def fail_attempt(path, value, label):
            if path == self.journal.terminal_attempt_path:
                raise OSError("attempt commit lost")
            return original(path, value, label)

        self.journal._publish_terminal_record = fail_attempt
        with self.assertRaisesRegex(OSError, "attempt commit lost"):
            self.journal.submit_and_verify(now=None)
        self.assertTrue(self.journal.host_restart_path.is_file())
        self.assertFalse(self.journal.terminal_attempt_path.exists())
        self.assertFalse(self.journal.terminal_submit_path.exists())
        vm, _, journal = self._fresh()
        with mock.patch.object(journal, "_old_host_vm_stopped", return_value=True):
            with self.assertRaisesRegex(RecoveryError, "terminal_attempt_record_corrupt"):
                journal.resume_verification_after_host_restart(self.task, now=None)
        self.assertEqual(vm.starts, 0)

    def test_changed_recovery_code_identity_fails_before_boot(self):
        self._submit_without_result()
        vm, _, journal = self._fresh()
        with mock.patch.object(journal, "_old_host_vm_stopped", return_value=True):
            with mock.patch.object(journal, "_source_binding", return_value={"forged": True}):
                with self.assertRaisesRegex(RecoveryError, "host_restart_frozen_identity_mismatch"):
                    journal.resume_verification_after_host_restart(self.task, now=None)
        self.assertEqual(vm.starts, 0)


if __name__ == "__main__":
    unittest.main()
