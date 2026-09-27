"""Fault-injection and classification checks for the full-VM recovery journal."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from future_prediction_bench.semantic_vm_recovery import (
    CodingVMRecoveryJournal, InjectedAfterSave, RecoveryError,
)


class FakeVM:
    def __init__(self, disk, saved=None):
        self.disk_path = disk
        self.state = {"files": {"source.py": "old"}, "processes": [(1, 1, "init")],
                      "ram": "baseline"}
        self.saved = {} if saved is None else saved
        self.saves = 0
        self.kernel_sha256 = "a" * 64
        self.initramfs_sha256 = "b" * 64
        self.readonly_disk_paths = ()
        self._readonly_disk_sha256s = ()
        self.memory_mib = 128
        self.vcpus = 1
        self._process = FakeProcess()

    def start(self, *, paused=False):
        if not paused:
            raise ValueError("Expected paused recovery boot")
        self._process = FakeProcess()

    def close(self):
        self._process = None

    def save_snapshot(self, tag):
        self.saved[tag] = copy.deepcopy(self.state)
        self.saves += 1
        return {"kind": "full_vm_state_qcow2_v1", "tag": tag,
                "disk_path": str(self.disk_path),
                "kernel_sha256": self.kernel_sha256,
                "initramfs_sha256": self.initramfs_sha256,
                "readonly_disk_sha256s": list(self._readonly_disk_sha256s)}

    def load_snapshot(self, tag, *, resume=False):
        self.state = copy.deepcopy(self.saved[tag])


class FakeProcess:
    def __init__(self):
        self.crashed = False

    def poll(self):
        return 1 if self.crashed else None


class FakeAdapter:
    def __init__(self, runtime):
        self.runtime = runtime
        self.started = True
        self.submitted = False
        self.verifications = 0
        self.submit_calls = 0
        self.snapshot = None
        self.verified = None
        self.expected_binding = {"runtime_kind": "qemu_hvf_full_vm_qcow2_v1",
                                 "readonly_disk_sha256s": []}
        self.read_side_effect = None

    def step(self, action, *, now):
        kind = action["action"]
        if kind == "list_files":
            result = {"observation": {"files": sorted(self.runtime.state["files"]),
                                      "truncated": False}, "terminated": False}
        elif kind == "read_file":
            content = self.runtime.state["files"][action["path"]]
            result = {"observation": {"path": action["path"], "text": content,
                                      "sha256": hashlib.sha256(content.encode()).hexdigest(),
                                      "truncated": False}, "terminated": False}
            if self.read_side_effect == "file":
                self.runtime.state["files"]["hidden"] = "mutation"
            elif self.read_side_effect == "process":
                self.runtime.state["processes"].append((2, 2, "worker"))
            elif self.read_side_effect == "ram":
                self.runtime.state["ram"] = "changed"
        elif kind == "write_file":
            self.runtime.state["files"][action["path"]] = action["content"]
            result = {"observation": {"path": action["path"],
                                      "sha256": hashlib.sha256(action["content"].encode()).hexdigest()},
                      "terminated": False}
        elif kind == "run_visible_checks":
            self.runtime.state["ram"] = "opaque-change"
            result = {"observation": {"passed": True}, "terminated": False}
        elif kind == "submit":
            self.submit_calls += 1
            self.snapshot = self.runtime.save_snapshot("submitted")
            self.submitted = True
            result = {"observation": {"status": "submitted",
                                      "snapshot_kind": self.snapshot["kind"]},
                      "terminated": True}
        else:
            raise ValueError(kind)
        return result

    def verify(self, *, now):
        if self.verified is not None:
            return copy.deepcopy(self.verified)
        self.verifications += 1
        self.verified = {"status": "resolved", "reward": 1.0}
        return copy.deepcopy(self.verified)


def inspect(adapter):
    state = adapter.runtime.state
    return {"processes": [list(item) for item in state["processes"]],
            "tree": hashlib.sha256(json.dumps(state["files"], sort_keys=True).encode()).hexdigest()}


class JournalTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        disk = Path(self.directory.name) / "vm.qcow2"
        disk.write_bytes(b"fake qcow2 for fsync")
        self.vm = FakeVM(disk)
        self.adapter = FakeAdapter(self.vm)
        self.journal = CodingVMRecoveryJournal(
            self.adapter, Path(self.directory.name) / "journal", inspector=lambda: inspect(self.adapter))
        self.journal.begin()

    def _read(self):
        return {"action": "read_file", "path": "source.py"}

    def test_read_only_skips_and_replays_same_observation(self):
        for _ in range(3):
            result = self.journal.apply(self._read(), now=None)
            self.assertFalse(result["checkpointed"])
        self.assertEqual(self.vm.saves, 1)
        self.vm.state["files"]["uncommitted"] = "discard me"
        replay = self.journal.recover(now=None)
        self.assertEqual(replay["replayed_turns"], 3)
        self.assertNotIn("uncommitted", self.vm.state["files"])
        self.assertEqual(self.journal.metrics["replayed_turns"], 3)

    def test_mutated_caller_action_does_not_change_replay_record(self):
        caller_action = self._read()
        original_step = self.adapter.step

        def mutate_caller_after_step(action, *, now):
            result = original_step(action, now=now)
            caller_action["path"] = "changed-after-execution.py"
            return result

        self.adapter.step = mutate_caller_after_step
        result = self.journal.apply(caller_action, now=None)
        self.assertFalse(result["checkpointed"])
        self.assertEqual(caller_action["path"], "changed-after-execution.py")
        self.assertEqual(self.journal.manifest["replay"][0]["action"], self._read())
        self.journal.recover(now=None)
        self.assertFalse(self.journal.needs_recovery)

    def test_file_mutation_during_nominal_read_forces_full_snapshot(self):
        self.adapter.read_side_effect = "file"
        result = self.journal.apply(self._read(), now=None)
        self.assertTrue(result["checkpointed"])
        self.assertEqual(self.vm.saves, 2)
        self.vm.state["files"].clear()
        self.journal.recover(now=None)
        self.assertEqual(self.vm.state["files"]["hidden"], "mutation")

    def test_live_process_during_nominal_read_forces_full_snapshot(self):
        self.adapter.read_side_effect = "process"
        result = self.journal.apply(self._read(), now=None)
        self.assertTrue(result["checkpointed"])
        self.vm.state["processes"].clear()
        self.journal.recover(now=None)
        self.assertIn((2, 2, "worker"), self.vm.state["processes"])

    def test_opaque_ram_and_process_state_restores_and_taints(self):
        self.journal.apply_opaque(lambda: self.vm.state.update({
            "ram": "updated", "processes": [(1, 1, "init"), (4, 4, "server")]}))
        self.assertEqual(self.vm.saves, 2)
        self.vm.state["ram"] = "lost"
        self.vm.state["processes"].clear()
        self.journal.recover(now=None)
        self.assertEqual(self.vm.state["ram"], "updated")
        self.assertIn((4, 4, "server"), self.vm.state["processes"])
        # The process could mutate memory without changing PID.  Once an
        # opaque tool has run, later reads may no longer skip snapshots.
        self.adapter.read_side_effect = None
        self.assertTrue(self.journal.apply(self._read(), now=None)["checkpointed"])

    def test_actual_vm_process_restart_path_rebinds_snapshot(self):
        self.journal.apply_opaque(lambda: self.vm.state.update({
            "ram": "saved", "processes": [(1, 1, "init"), (4, 4, "server")]}))
        self.vm.state["ram"] = "uncommitted"
        with self.assertRaises(RecoveryError):
            self.journal.recover_after_vm_crash(FakeVM(self.vm.disk_path, self.vm.saved),
                                                now=None)
        self.vm._process.crashed = True
        replacement = FakeVM(self.vm.disk_path, self.vm.saved)
        self.journal.recover_after_vm_crash(replacement, now=None)
        self.assertIs(self.adapter.runtime, replacement)
        self.assertEqual(replacement.state["ram"], "saved")
        self.assertIn((4, 4, "server"), replacement.state["processes"])

    def test_vm_restart_rejects_different_disk(self):
        self.vm._process.crashed = True
        different = Path(self.directory.name) / "other.qcow2"
        different.write_bytes(b"different")
        with self.assertRaises(RecoveryError):
            self.journal.recover_after_vm_crash(FakeVM(different, self.vm.saved), now=None)
        self.assertIs(self.adapter.runtime, self.vm)

    def test_recovery_journal_rejects_non_arm_backend_at_construction(self):
        self.vm.backend = "x86_64_tcg"
        unused = Path(self.directory.name) / "non-arm-journal"
        with self.assertRaisesRegex(RecoveryError, "requires_aarch64_hvf_backend"):
            CodingVMRecoveryJournal(self.adapter, unused)
        self.assertFalse(unused.exists())

    def test_vm_restart_rejects_changed_backend_before_closing_old_runtime(self):
        self.vm._process.crashed = True
        old_process = self.vm._process
        replacement = FakeVM(self.vm.disk_path, self.vm.saved)
        replacement.backend = "x86_64_tcg"
        replacement.start = mock.Mock(side_effect=AssertionError("must not start"))
        with self.assertRaisesRegex(RecoveryError, "replacement_vm_differs"):
            self.journal.recover_after_vm_crash(replacement, now=None)
        replacement.start.assert_not_called()
        self.assertIs(self.vm._process, old_process)
        self.assertIs(self.adapter.runtime, self.vm)

    def test_vm_restart_rejects_changed_readonly_disk_binding(self):
        self.vm._process.crashed = True
        replacement = FakeVM(self.vm.disk_path, self.vm.saved)
        replacement._readonly_disk_sha256s = ("0" * 64,)
        with self.assertRaisesRegex(RecoveryError, "replacement_readonly_disk_mismatch"):
            self.journal.recover_after_vm_crash(replacement, now=None)
        self.assertTrue(self.journal.needs_recovery)
        self.assertIs(self.adapter.runtime, self.vm)
        self.assertEqual(self.adapter.verifications, 0)

    def test_vm_restart_can_retry_after_transient_start_failure(self):
        self.vm._process.crashed = True
        failed = FakeVM(self.vm.disk_path, self.vm.saved)

        def fail_start(*, paused=False):
            raise RuntimeError("transient QEMU boot failure")

        failed.start = fail_start
        with self.assertRaisesRegex(RuntimeError, "transient QEMU boot failure"):
            self.journal.recover_after_vm_crash(failed, now=None)
        self.assertTrue(self.journal.needs_recovery)
        with self.assertRaisesRegex(RecoveryError, "recover_before_next_turn"):
            self.journal.submit_and_verify(now=None)
        replacement = FakeVM(self.vm.disk_path, self.vm.saved)
        self.journal.recover_after_vm_crash(replacement, now=None)
        self.assertIs(self.adapter.runtime, replacement)
        self.assertFalse(self.journal.needs_recovery)

    def test_vm_restart_can_retry_after_transient_load_failure(self):
        self.vm._process.crashed = True
        failed = FakeVM(self.vm.disk_path, self.vm.saved)

        def fail_load(tag, *, resume=False):
            raise RuntimeError("transient QEMU load failure")

        failed.load_snapshot = fail_load
        with self.assertRaisesRegex(RecoveryError, "vm_recovery_failed"):
            self.journal.recover_after_vm_crash(failed, now=None)
        self.assertTrue(self.journal.needs_recovery)
        replacement = FakeVM(self.vm.disk_path, self.vm.saved)
        self.journal.recover_after_vm_crash(replacement, now=None)
        self.assertIs(self.adapter.runtime, replacement)
        self.assertFalse(self.journal.needs_recovery)

    def test_failed_partial_action_blocks_until_rollback(self):
        def partial_write_then_fail(action, *, now):
            self.vm.state["files"]["partial"] = "uncommitted"
            raise RuntimeError("guest command failed")

        self.adapter.step = partial_write_then_fail
        with self.assertRaises(RuntimeError):
            self.journal.apply({"action": "write_file", "path": "source.py",
                                "content": "updated"}, now=None)
        self.assertTrue(self.journal.needs_recovery)
        with self.assertRaises(RecoveryError):
            self.journal.apply(self._read(), now=None)
        self.adapter.step = FakeAdapter.step.__get__(self.adapter, FakeAdapter)
        self.journal.recover(now=None)
        self.assertNotIn("partial", self.vm.state["files"])

    def test_keyboard_interrupt_after_partial_write_cannot_release_reward(self):
        def interrupted_step(action, *, now):
            self.vm.state["files"]["partial"] = "uncommitted"
            raise KeyboardInterrupt()

        self.adapter.step = interrupted_step
        with self.assertRaises(KeyboardInterrupt):
            self.journal.apply({"action": "write_file", "path": "source.py",
                                "content": "updated"}, now=None)
        self.assertTrue(self.journal.needs_recovery)
        with self.assertRaisesRegex(RecoveryError, "recover_before_next_turn"):
            self.journal.submit_and_verify(now=None)
        self.assertEqual(self.adapter.verifications, 0)

    def test_keyboard_interrupt_after_snapshot_cannot_release_reward(self):
        committed = self.journal.path.read_bytes()
        original_save = self.vm.save_snapshot

        def interrupted_save(tag):
            original_save(tag)
            raise KeyboardInterrupt()

        self.vm.save_snapshot = interrupted_save
        with self.assertRaises(KeyboardInterrupt):
            self.journal.apply({"action": "write_file", "path": "source.py",
                                "content": "uncommitted"}, now=None)
        self.assertTrue(self.journal.needs_recovery)
        self.assertEqual(self.journal.path.read_bytes(), committed)
        with self.assertRaisesRegex(RecoveryError, "recover_before_next_turn"):
            self.journal.submit_and_verify(now=None)
        self.assertEqual(self.adapter.verifications, 0)

    def test_pre_inspector_interrupt_keeps_reward_gate_closed(self):
        def interrupted_inspector():
            self.vm.state["files"]["inspector_partial"] = "uncommitted"
            raise KeyboardInterrupt()

        self.journal.inspector = interrupted_inspector
        with self.assertRaises(KeyboardInterrupt):
            self.journal.apply(self._read(), now=None)
        self.assertTrue(self.journal.needs_recovery)
        with self.assertRaisesRegex(RecoveryError, "recover_before_next_turn"):
            self.journal.submit_and_verify(now=None)
        self.assertEqual(self.adapter.verifications, 0)

    def test_post_action_commit_interrupt_keeps_reward_gate_closed(self):
        def interrupted_commit(*args, **kwargs):
            raise KeyboardInterrupt()

        self.journal._commit = interrupted_commit
        with self.assertRaises(KeyboardInterrupt):
            self.journal.apply({"action": "write_file", "path": "source.py",
                                "content": "uncommitted"}, now=None)
        self.assertEqual(self.vm.state["files"]["source.py"], "uncommitted")
        self.assertTrue(self.journal.needs_recovery)
        with self.assertRaisesRegex(RecoveryError, "recover_before_next_turn"):
            self.journal.submit_and_verify(now=None)
        self.assertEqual(self.adapter.verifications, 0)

    def test_malformed_submission_blocks_grading_until_recovery(self):
        self.adapter.step = lambda action, *, now: {"observation": None}
        with self.assertRaisesRegex(RecoveryError, "vm_submission_failed"):
            self.journal.submit_and_verify(now=None)
        self.assertTrue(self.journal.needs_recovery)
        self.assertEqual(self.adapter.verifications, 0)

    def test_verifier_failure_closes_reward_gate(self):
        def failed_verify(*, now):
            raise KeyboardInterrupt()

        self.adapter.verify = failed_verify
        with self.assertRaises(KeyboardInterrupt):
            self.journal.submit_and_verify(now=None)
        self.assertTrue(self.journal.needs_recovery)

    def test_interrupted_hidden_case_retries_from_submitted_full_vm(self):
        seen = []
        attempts = 0

        def interrupted_then_complete(*, now):
            nonlocal attempts
            attempts += 1
            for case in range(2):
                self.vm.load_snapshot("submitted", resume=True)
                seen.append((attempts, case,
                             self.vm.state["files"].get("case_marker")))
                self.vm.state["files"]["case_marker"] = str(case)
                if attempts == 1:
                    raise KeyboardInterrupt()
            self.adapter.verified = {"status": "resolved", "reward": 1.0}
            return copy.deepcopy(self.adapter.verified)

        self.adapter.verify = interrupted_then_complete
        with self.assertRaises(KeyboardInterrupt):
            self.journal.submit_and_verify(now=None)
        self.assertTrue(self.journal.terminal_submit_path.is_file())
        self.assertFalse(self.journal.terminal_result_path.exists())
        self.assertTrue(self.journal.needs_recovery)
        with self.assertRaisesRegex(RecoveryError, "terminal_episode_already_submitted"):
            self.journal.apply(self._read(), now=None)
        with self.assertRaisesRegex(RecoveryError, "old_vm_has_not_crashed"):
            self.journal.resume_verification_after_vm_crash(
                FakeVM(self.vm.disk_path, self.vm.saved), now=None)
        terminal = self.journal.resume_verification(now=None)
        self.assertEqual(terminal["grading"]["reward"], 1.0)
        self.assertEqual(seen, [(1, 0, None), (2, 0, None), (2, 1, None)])
        self.assertEqual(self.adapter.submit_calls, 1)
        self.assertEqual(attempts, 2)
        self.assertTrue(self.journal.terminal_result_path.is_file())

    def test_crashed_vm_during_hidden_case_restarts_from_committed_submission(self):
        seen = []
        attempts = 0

        def crash_once_then_grade(*, now):
            nonlocal attempts
            attempts += 1
            runtime = self.adapter.runtime
            runtime.load_snapshot("submitted", resume=True)
            seen.append((attempts, runtime.state["files"].get("case_marker")))
            runtime.state["files"]["case_marker"] = "case-side-effect"
            if attempts == 1:
                runtime._process.crashed = True
                raise KeyboardInterrupt()
            self.adapter.verifications += 1
            self.adapter.verified = {"status": "resolved", "reward": 1.0}
            return copy.deepcopy(self.adapter.verified)

        self.adapter.verify = crash_once_then_grade
        with self.assertRaises(KeyboardInterrupt):
            self.journal.submit_and_verify(now=None)
        self.assertTrue(self.journal.terminal_submit_path.is_file())
        self.assertFalse(self.journal.terminal_result_path.exists())
        self.assertEqual(self.adapter.submit_calls, 1)
        replacement = FakeVM(self.vm.disk_path, self.vm.saved)
        result = self.journal.resume_verification_after_vm_crash(replacement, now=None)
        self.assertEqual(result["grading"]["reward"], 1.0)
        self.assertEqual(seen, [(1, None), (2, None)])
        self.assertIs(self.adapter.runtime, replacement)
        self.assertEqual(self.adapter.submit_calls, 1)
        self.assertEqual(self.adapter.verifications, 1)
        self.assertTrue(self.journal.terminal_result_path.is_file())
        self.assertEqual(self.journal.submit_and_verify(now=None), result)

    def test_terminal_vm_restart_load_failure_is_retryable_without_reward(self):
        def crash_before_grading(*, now):
            # A real serial transport may close QEMU while surfacing its
            # failure, leaving no process handle for the recovery caller.
            self.adapter.runtime.close()
            raise KeyboardInterrupt()

        self.adapter.verify = crash_before_grading
        with self.assertRaises(KeyboardInterrupt):
            self.journal.submit_and_verify(now=None)
        failed = FakeVM(self.vm.disk_path, self.vm.saved)

        def interrupted_load(tag, *, resume=False):
            raise RuntimeError("injected terminal snapshot load failure")

        failed.load_snapshot = interrupted_load
        with self.assertRaisesRegex(RuntimeError, "terminal snapshot load failure"):
            self.journal.resume_verification_after_vm_crash(failed, now=None)
        self.assertTrue(self.journal.needs_recovery)
        self.assertFalse(self.journal.terminal_result_path.exists())
        self.assertEqual(self.adapter.submit_calls, 1)
        with self.assertRaisesRegex(RecoveryError, "terminal_episode_already_submitted"):
            self.journal.apply(self._read(), now=None)

        self.adapter.verify = FakeAdapter.verify.__get__(self.adapter, FakeAdapter)
        replacement = FakeVM(self.vm.disk_path, self.vm.saved)
        result = self.journal.resume_verification_after_vm_crash(replacement, now=None)
        self.assertEqual(result["grading"]["reward"], 1.0)
        self.assertEqual(self.adapter.submit_calls, 1)
        self.assertEqual(self.adapter.verifications, 1)

    def test_terminal_vm_restart_rejects_changed_disk_before_boot(self):
        def crash_before_grading(*, now):
            self.adapter.runtime._process.crashed = True
            raise KeyboardInterrupt()

        self.adapter.verify = crash_before_grading
        with self.assertRaises(KeyboardInterrupt):
            self.journal.submit_and_verify(now=None)
        other_disk = Path(self.directory.name) / "other-terminal.qcow2"
        other_disk.write_bytes(b"different")
        wrong = FakeVM(other_disk, self.vm.saved)
        with self.assertRaisesRegex(RecoveryError, "replacement_vm_differs"):
            self.journal.resume_verification_after_vm_crash(wrong, now=None)
        self.vm.kernel_append = "console=ttyAMA0"
        wrong_boot = FakeVM(self.vm.disk_path, self.vm.saved)
        wrong_boot.kernel_append = "console=ttyAMA0 altered=1"
        with self.assertRaisesRegex(RecoveryError, "replacement_vm_differs"):
            self.journal.resume_verification_after_vm_crash(wrong_boot, now=None)
        self.assertIs(self.adapter.runtime, self.vm)
        self.assertFalse(self.journal.terminal_result_path.exists())
        self.assertEqual(self.adapter.submit_calls, 1)

    def test_terminal_vm_restart_rejects_x86_backend_before_boot(self):
        def crash_before_grading(*, now):
            self.vm._process.crashed = True
            raise KeyboardInterrupt()
        self.adapter.verify = crash_before_grading
        with self.assertRaises(KeyboardInterrupt):
            self.journal.submit_and_verify(now=None)
        replacement = FakeVM(self.vm.disk_path, self.vm.saved)
        replacement.backend = "x86_64_kvm"
        replacement.start = mock.Mock(side_effect=AssertionError("must not start"))
        old_process = self.vm._process
        with self.assertRaisesRegex(RecoveryError, "replacement_vm_differs"):
            self.journal.resume_verification_after_vm_crash(replacement, now=None)
        replacement.start.assert_not_called()
        self.assertIs(self.vm._process, old_process)
        self.assertIs(self.adapter.runtime, self.vm)
        self.assertFalse(self.journal.terminal_result_path.exists())

    def test_terminal_vm_restart_requires_durable_submit_marker(self):
        publish = self.journal._publish_terminal_record

        def fail_before_marker(path, value, label):
            if path == self.journal.terminal_submit_path:
                raise OSError("injected terminal marker failure")
            return publish(path, value, label)

        self.journal._publish_terminal_record = fail_before_marker
        with self.assertRaisesRegex(OSError, "terminal marker failure"):
            self.journal.submit_and_verify(now=None)
        self.vm._process.crashed = True
        replacement = FakeVM(self.vm.disk_path, self.vm.saved)
        with self.assertRaisesRegex(RecoveryError, "terminal_submit_record_corrupt"):
            self.journal.resume_verification_after_vm_crash(replacement, now=None)
        self.assertIs(self.adapter.runtime, self.vm)
        self.assertEqual(self.adapter.submit_calls, 1)
        self.assertEqual(self.adapter.verifications, 0)
        self.assertFalse(self.journal.terminal_result_path.exists())

    def test_terminal_result_is_durable_and_duplicate_calls_are_idempotent(self):
        first = self.journal.submit_and_verify(now=None)
        submit_bytes = self.journal.terminal_submit_path.read_bytes()
        result_bytes = self.journal.terminal_result_path.read_bytes()
        second = self.journal.submit_and_verify(now=None)
        self.assertEqual(first, second)
        self.assertEqual(self.journal.terminal_submit_path.read_bytes(), submit_bytes)
        self.assertEqual(self.journal.terminal_result_path.read_bytes(), result_bytes)
        self.assertEqual(self.adapter.submit_calls, 1)
        self.assertEqual(self.adapter.verifications, 1)
        self.assertRegex(first["terminal_id"], r"^[0-9a-f]{64}$")
        # A new journal object can use the same live adapter and private
        # durable records; no QEMU/adapter reconstruction is implied.
        reopened = CodingVMRecoveryJournal(
            self.adapter, self.journal.directory, inspector=lambda: inspect(self.adapter))
        self.assertEqual(reopened.resume_verification(now=None), first)
        self.assertEqual(self.adapter.verifications, 1)

    def test_submit_before_marker_failure_retries_without_second_submit(self):
        publish = self.journal._publish_terminal_record
        failures = 0

        def fail_first_marker(path, value, label):
            nonlocal failures
            if path == self.journal.terminal_submit_path and failures == 0:
                failures += 1
                raise OSError("injected marker publication failure")
            return publish(path, value, label)

        self.journal._publish_terminal_record = fail_first_marker
        with self.assertRaisesRegex(OSError, "marker publication failure"):
            self.journal.submit_and_verify(now=None)
        self.assertTrue(self.adapter.submitted)
        self.assertEqual(self.adapter.submit_calls, 1)
        self.assertEqual(self.adapter.verifications, 0)
        self.assertFalse(self.journal.terminal_submit_path.exists())
        with self.assertRaisesRegex(RecoveryError, "terminal_episode_already_submitted"):
            self.journal.apply(self._read(), now=None)
        self.journal._publish_terminal_record = publish
        resumed = self.journal.resume_verification(now=None)
        self.assertEqual(resumed["grading"]["reward"], 1.0)
        self.assertEqual(self.adapter.submit_calls, 1)
        self.assertEqual(self.adapter.verifications, 1)

    def test_pending_grade_can_retry_without_reward_or_second_submit(self):
        verify = self.adapter.verify
        calls = 0

        def pending_once(*, now):
            nonlocal calls
            calls += 1
            if calls == 1:
                return {"status": "pending", "reason": "transient_verifier_error"}
            return verify(now=now)

        self.adapter.verify = pending_once
        pending = self.journal.submit_and_verify(now=None)
        self.assertEqual(pending["grading"]["status"], "pending")
        self.assertNotIn("reward", pending["grading"])
        self.assertFalse(self.journal.terminal_result_path.exists())
        resolved = self.journal.resume_verification(now=None)
        self.assertEqual(resolved["grading"]["reward"], 1.0)
        self.assertEqual(resolved["terminal_id"], pending["terminal_id"])
        self.assertEqual(self.adapter.submit_calls, 1)

    def test_result_publication_failure_does_not_return_reward(self):
        publish = self.journal._publish_terminal_record
        failures = 0

        def fail_first_result(path, value, label):
            nonlocal failures
            if path == self.journal.terminal_result_path and failures == 0:
                failures += 1
                raise OSError("injected result publication failure")
            return publish(path, value, label)

        self.journal._publish_terminal_record = fail_first_result
        with self.assertRaisesRegex(OSError, "result publication failure"):
            self.journal.submit_and_verify(now=None)
        self.assertTrue(self.journal.terminal_submit_path.is_file())
        self.assertFalse(self.journal.terminal_result_path.exists())
        self.assertEqual(self.adapter.submit_calls, 1)
        self.assertEqual(self.adapter.verifications, 1)
        self.journal._publish_terminal_record = publish
        resolved = self.journal.resume_verification(now=None)
        self.assertEqual(resolved["grading"]["reward"], 1.0)
        self.assertTrue(self.journal.terminal_result_path.is_file())
        self.assertEqual(self.adapter.submit_calls, 1)
        self.assertEqual(self.adapter.verifications, 1)

    def _assert_existing_terminal_record_resync(self, initial_directory_sync,
                                                retry_directory_sync,
                                                expected_verifications):
        original_fsync = os.fsync
        count = 0
        failing_directory_sync = initial_directory_sync

        def fail_on_directory_sync(fd):
            nonlocal count
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                count += 1
                if count == failing_directory_sync:
                    raise OSError("injected directory fsync failure")
            return original_fsync(fd)

        with mock.patch("future_prediction_bench.semantic_vm_recovery.os.fsync",
                        side_effect=fail_on_directory_sync):
            with self.assertRaisesRegex(OSError, "directory fsync failure"):
                self.journal.submit_and_verify(now=None)
        self.assertTrue(self.journal.terminal_submit_path.exists())
        self.assertEqual(self.adapter.verifications, expected_verifications)
        count = 0
        failing_directory_sync = retry_directory_sync
        with mock.patch("future_prediction_bench.semantic_vm_recovery.os.fsync",
                        side_effect=fail_on_directory_sync):
            with self.assertRaisesRegex(OSError, "directory fsync failure"):
                self.journal.resume_verification(now=None)
        self.assertEqual(self.adapter.verifications, expected_verifications)
        result = self.journal.resume_verification(now=None)
        self.assertEqual(result["grading"]["reward"], 1.0)
        self.assertEqual(self.adapter.submit_calls, 1)
        self.assertEqual(self.adapter.verifications, 1)

    def test_existing_terminal_marker_is_resynced_before_verification(self):
        self._assert_existing_terminal_record_resync(3, 2, 0)

    def test_existing_terminal_result_is_resynced_before_reward(self):
        self._assert_existing_terminal_record_resync(4, 3, 1)

    def test_unknown_submit_outcome_blocks_recovery_and_second_submit(self):
        def interrupted_submit(action, *, now):
            self.adapter.submit_calls += 1
            self.vm.save_snapshot("submitted")
            raise KeyboardInterrupt()

        self.adapter.step = interrupted_submit
        with self.assertRaises(KeyboardInterrupt):
            self.journal.submit_and_verify(now=None)
        self.assertTrue(self.journal.terminal_attempt_path.is_file())
        self.assertFalse(self.journal.terminal_submit_path.exists())
        self.assertEqual(self.adapter.verifications, 0)
        with self.assertRaisesRegex(RecoveryError, "terminal_submit_attempt_uncertain"):
            self.journal.recover(now=None)
        with self.assertRaisesRegex(RecoveryError, "terminal_submit_attempt_uncertain"):
            self.journal.submit_and_verify(now=None)
        with self.assertRaisesRegex(RecoveryError, "terminal_episode_already_submitted"):
            self.journal.apply(self._read(), now=None)
        reopened = CodingVMRecoveryJournal(
            self.adapter, self.journal.directory, inspector=lambda: inspect(self.adapter))
        with self.assertRaisesRegex(RecoveryError, "terminal_submit_attempt_uncertain"):
            reopened.recover(now=None)
        self.assertEqual(self.adapter.submit_calls, 1)

    def test_terminal_record_corruption_fails_closed(self):
        self.journal.submit_and_verify(now=None)
        attempt_bytes = self.journal.terminal_attempt_path.read_bytes()
        marker_bytes = self.journal.terminal_submit_path.read_bytes()
        verify_count = self.adapter.verifications
        cached_grade = self.adapter.verified
        self.adapter.verified = None
        with self.assertRaisesRegex(RecoveryError, "terminal_result_record_corrupt"):
            self.journal.resume_verification(now=None)
        self.adapter.verified = cached_grade
        self.journal.terminal_attempt_path.write_text('{"schema":"forged"}\n')
        with self.assertRaisesRegex(RecoveryError, "terminal_attempt_record_corrupt"):
            self.journal.resume_verification(now=None)
        self.journal.terminal_attempt_path.write_bytes(attempt_bytes)
        self.journal.terminal_submit_path.write_text('{"schema":"forged"}\n')
        with self.assertRaisesRegex(RecoveryError, "terminal_submit_record_corrupt"):
            self.journal.resume_verification(now=None)
        self.assertEqual(self.adapter.verifications, verify_count)
        self.journal.terminal_submit_path.write_bytes(marker_bytes)
        self.journal.terminal_result_path.write_text('{"schema":"forged"}\n')
        with self.assertRaisesRegex(RecoveryError, "terminal_result_record_corrupt"):
            self.journal.submit_and_verify(now=None)
        self.assertEqual(self.adapter.submit_calls, 1)
        self.assertEqual(self.adapter.verifications, verify_count)

    def test_failure_after_artifact_before_manifest_uses_old_snapshot(self):
        self.journal.apply(self._read(), now=None)
        committed = self.journal.path.read_bytes()
        with self.assertRaises(InjectedAfterSave):
            self.journal.apply({"action": "write_file", "path": "source.py",
                                "content": "uncommitted"}, now=None,
                               inject_after_save=True)
        with self.assertRaises(RecoveryError):
            self.journal.submit_and_verify(now=None)
        self.assertEqual(self.adapter.verifications, 0)
        self.assertEqual(self.journal.path.read_bytes(), committed)
        self.assertEqual(self.vm.state["files"]["source.py"], "uncommitted")
        self.assertEqual(self.vm.saves, 2)  # The second tag is an orphan.
        reopened = CodingVMRecoveryJournal(
            self.adapter, self.journal.directory, inspector=lambda: inspect(self.adapter))
        result = reopened.recover(now=None)
        self.assertEqual(result, {"turn": 1, "checkpoint_turn": 0, "replayed_turns": 1})
        self.assertEqual(self.vm.state["files"]["source.py"], "old")
        retried = reopened.apply({"action": "write_file", "path": "source.py",
                                 "content": "committed"}, now=None)
        self.assertTrue(retried["checkpointed"])
        self.vm.state["files"]["source.py"] = "wrong"
        reopened.recover(now=None)
        self.assertEqual(self.vm.state["files"]["source.py"], "committed")

    def test_open_cannot_release_uncommitted_guest_state_for_reward(self):
        self.journal.apply(self._read(), now=None)
        with self.assertRaises(InjectedAfterSave):
            self.journal.apply({"action": "write_file", "path": "source.py",
                                "content": "uncommitted"}, now=None,
                               inject_after_save=True)
        self.assertEqual(self.vm.state["files"]["source.py"], "uncommitted")
        self.journal.open()
        self.assertTrue(self.journal.needs_recovery)
        with self.assertRaisesRegex(RecoveryError, "recover_before_next_turn"):
            self.journal.submit_and_verify(now=None)
        reopened = CodingVMRecoveryJournal(
            self.adapter, self.journal.directory, inspector=lambda: inspect(self.adapter))
        reopened.open()
        with self.assertRaisesRegex(RecoveryError, "recover_before_next_turn"):
            reopened.submit_and_verify(now=None)
        self.assertEqual(self.adapter.verifications, 0)
        reopened.recover(now=None)
        self.assertEqual(self.vm.state["files"]["source.py"], "old")
        self.assertEqual(reopened.submit_and_verify(now=None)["grading"]["reward"], 1.0)
        self.assertEqual(self.adapter.verifications, 1)

    def test_replay_divergence_is_infrastructure_error(self):
        self.journal.apply(self._read(), now=None)
        self.adapter.step = lambda *args, **kwargs: {"observation": "different"}
        with self.assertRaises(RecoveryError):
            self.journal.recover(now=None)
        self.assertTrue(self.journal.needs_recovery)

    def test_replayed_read_with_hidden_mutation_keeps_reward_gate_closed(self):
        original = self.journal.apply(self._read(), now=None)
        self.assertFalse(original["checkpointed"])
        self.adapter.read_side_effect = "file"
        with self.assertRaisesRegex(RecoveryError, "vm_recovery_failed"):
            self.journal.recover(now=None)
        self.assertTrue(self.journal.needs_recovery)
        self.assertEqual(self.adapter.verifications, 0)
        with self.assertRaisesRegex(RecoveryError, "recover_before_next_turn"):
            self.journal.submit_and_verify(now=None)
        self.adapter.read_side_effect = None
        self.journal.recover(now=None)
        self.assertNotIn("hidden", self.vm.state["files"])

    def test_malformed_manifest_and_symlink_fail_closed(self):
        self.journal.path.write_text('{"turn":999}', encoding="utf-8")
        with self.assertRaises(RecoveryError):
            self.journal.open()
        self.journal.path.write_text('{"turn":1,"turn":2}', encoding="utf-8")
        with self.assertRaises(RecoveryError):
            self.journal.open()
        self.journal.path.unlink()
        self.journal.path.symlink_to(self.vm.disk_path)
        with self.assertRaises(RecoveryError):
            self.journal.open()


if __name__ == "__main__":
    unittest.main()
