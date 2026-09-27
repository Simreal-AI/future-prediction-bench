"""The async release gate must survive real thread interleavings and faults."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import future_prediction_bench.async_vm_checkpoint as checkpoint_module
from future_prediction_bench.async_vm_checkpoint import (
    AsyncVMCheckpointCoordinator, PolicyChoice,
)
from future_prediction_bench.semantic_vm_recovery import RecoveryError


class FakeVM:
    def __init__(self, disk):
        self.disk_path = disk
        self.state = {"text": "before", "ram": "first"}
        self.saved = {}
        self.saves = 0
        self.kernel_sha256 = "a" * 64
        self.initramfs_sha256 = "b" * 64
        self.entered = threading.Event()
        self.release = threading.Event()
        self.fail_save = False
        self.hold_save = False

    def save_snapshot(self, tag):
        if self.saves:
            self.entered.set()
            if self.hold_save and not self.release.wait(timeout=2):
                raise RuntimeError("test worker timed out")
            if self.fail_save:
                raise RuntimeError("injected snapshot failure")
        self.saved[tag] = copy.deepcopy(self.state)
        self.saves += 1
        return {"kind": "full_vm_state_qcow2_v1", "tag": tag,
                "disk_path": str(self.disk_path),
                "kernel_sha256": self.kernel_sha256,
                "initramfs_sha256": self.initramfs_sha256,
                "readonly_disk_sha256s": []}

    def load_snapshot(self, tag, *, resume=False):
        self.state = copy.deepcopy(self.saved[tag])


class FakeAdapter:
    def __init__(self, runtime):
        self.runtime = runtime
        self.started = True
        self.submitted = False
        self.expected_binding = {"kind": "fake-full-vm",
                                 "readonly_disk_sha256s": []}
        self.verifications = 0
        self.snapshot = None
        self.verified = None

    def step(self, action, *, now):
        if action["action"] == "write_file":
            self.runtime.state["text"] = action["content"]
            observation = {"sha256": hashlib.sha256(action["content"].encode()).hexdigest()}
        elif action["action"] == "run_visible_checks":
            observation = {"passed": self.runtime.state["text"] == "repaired"}
        elif action["action"] == "submit":
            self.snapshot = self.runtime.save_snapshot("submitted")
            self.submitted = True
            observation = {"status": "submitted",
                           "snapshot_kind": self.snapshot["kind"]}
        else:
            raise ValueError(action)
        return {"observation": observation,
                "terminated": action["action"] == "submit"}

    def verify(self, *, now):
        if self.verified is not None:
            return copy.deepcopy(self.verified)
        self.verifications += 1
        self.verified = {"status": "resolved", "reward": 1.0}
        return copy.deepcopy(self.verified)


def _next(next_action):
    def choose(prompt):
        assert json.loads(prompt.observation_json)
        return PolicyChoice(prompt.observation_sha256, next_action)
    return choose


class AsyncCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        disk = Path(self.temp.name) / "vm.qcow2"
        disk.write_bytes(b"fsync-able fake disk")
        self.vm = FakeVM(disk)
        self.adapter = FakeAdapter(self.vm)
        self.coordinator = AsyncVMCheckpointCoordinator(
            self.adapter, Path(self.temp.name) / "journal", mode="overlap",
            inspector=lambda: {"processes": [], "tree": "fake"})
        self.addCleanup(self.coordinator.close)
        self.coordinator.begin()
        self.action = {"action": "write_file", "path": "source.py",
                       "content": "repaired"}

    def test_policy_runs_while_real_checkpoint_worker_waits_and_gate_blocks(self):
        self.vm.hold_save = True
        seen = []

        def policy(prompt):
            self.assertTrue(self.vm.entered.wait(timeout=1))
            seen.append(json.loads(prompt.observation_json))
            self.assertEqual(self.coordinator.journal.manifest["turn"], 0)
            with self.assertRaisesRegex(RecoveryError, "checkpoint_boundary_in_flight"):
                self.coordinator.submit_and_verify(now=None)
            with self.assertRaisesRegex(RecoveryError, "checkpoint_boundary_in_flight"):
                self.coordinator.run_turn(self.action, _next({"action": "submit"}), now=None)
            self.vm.release.set()
            return PolicyChoice(prompt.observation_sha256,
                                {"action": "run_visible_checks"})

        outcome = self.coordinator.run_turn(self.action, policy, now=None)
        self.assertEqual(len(seen), 1)
        self.assertGreater(outcome["measurements"]["checkpoint_policy_overlap_seconds"], 0)
        self.assertEqual(self.coordinator.journal.manifest["turn"], 1)
        self.assertTrue((self.coordinator.journal.directory / "policy-turn-000001.json").is_file())
        with self.assertRaisesRegex(RecoveryError, "submit_not_committed_policy_choice"):
            self.coordinator.submit_and_verify(now=None)
        self.coordinator.run_turn({"action": "run_visible_checks"},
                                  _next({"action": "submit"}), now=None)
        self.assertEqual(self.coordinator.submit_and_verify(now=None)["grading"]["reward"], 1)
        self.assertEqual(self.adapter.verifications, 1)

    def test_save_failure_halts_and_reward_is_never_released(self):
        self.vm.fail_save = True
        with self.assertRaisesRegex(RuntimeError, "injected snapshot failure"):
            self.coordinator.run_turn(self.action, _next({"action": "submit"}), now=None)
        self.assertEqual(self.coordinator.journal.manifest["turn"], 0)
        self.assertFalse((self.coordinator.journal.directory / "policy-turn-000001.json").exists())
        with self.assertRaises(RecoveryError):
            self.coordinator.submit_and_verify(now=None)
        self.assertEqual(self.adapter.verifications, 0)

    def test_interrupt_after_action_before_worker_submission_halts(self):
        def interrupted_prompt(*args, **kwargs):
            raise KeyboardInterrupt()

        with patch.object(checkpoint_module, "PolicyPrompt", interrupted_prompt):
            with self.assertRaises(KeyboardInterrupt):
                self.coordinator.run_turn(
                    self.action, _next({"action": "submit"}), now=None)
        self.assertEqual(self.vm.state["text"], "repaired")
        self.assertEqual(self.coordinator.journal.manifest["turn"], 0)
        self.assertTrue(self.coordinator.journal.needs_recovery)
        self.assertTrue(self.coordinator._halted)
        with self.assertRaises(RecoveryError):
            self.coordinator.submit_and_verify(now=None)
        self.assertEqual(self.adapter.verifications, 0)

    def test_after_save_before_manifest_fault_is_not_policy_released(self):
        manifest_before = self.coordinator.journal.path.read_bytes()
        with self.assertRaisesRegex(RecoveryError, "injected_after_snapshot_before_manifest"):
            self.coordinator.run_turn(
                self.action, _next({"action": "submit"}), now=None,
                inject_after_save=True)
        self.assertEqual(self.coordinator.journal.path.read_bytes(), manifest_before)
        self.assertEqual(self.vm.saves, 2)  # One uncommitted orphan tag.
        self.assertFalse((self.coordinator.journal.directory / "policy-turn-000001.json").exists())
        with self.assertRaises(RecoveryError):
            self.coordinator.submit_and_verify(now=None)
        self.assertEqual(self.adapter.verifications, 0)

    def test_policy_error_awaits_worker_then_halts(self):
        self.vm.hold_save = True
        worker_finished = threading.Event()
        original_save = self.vm.save_snapshot

        def observed_save(tag):
            try:
                return original_save(tag)
            finally:
                worker_finished.set()

        self.vm.save_snapshot = observed_save

        def bad_policy(prompt):
            self.assertTrue(self.vm.entered.wait(timeout=1))
            threading.Timer(0.03, self.vm.release.set).start()
            raise RuntimeError("policy unavailable")

        with self.assertRaisesRegex(RuntimeError, "policy unavailable"):
            self.coordinator.run_turn(self.action, bad_policy, now=None)
        self.assertTrue(worker_finished.is_set())
        self.assertEqual(self.coordinator.journal.manifest["turn"], 1)
        with self.assertRaises(RecoveryError):
            self.coordinator.submit_and_verify(now=None)
        self.assertEqual(self.adapter.verifications, 0)

    def test_caller_mutation_cannot_change_checkpoint_action_record(self):
        self.vm.hold_save = True
        action = copy.deepcopy(self.action)

        def mutate_while_saving(prompt):
            self.assertTrue(self.vm.entered.wait(timeout=1))
            action["action"] = "run_visible_checks"
            action["content"] = "not the executed edit"
            self.vm.release.set()
            return PolicyChoice(prompt.observation_sha256, {"action": "submit"})

        result = self.coordinator.run_turn(action, mutate_while_saving, now=None)
        self.assertEqual(result["measurements"]["action"], "write_file")
        self.assertFalse(self.coordinator.journal.manifest["opaque_taint"])
        self.assertEqual(self.vm.state["text"], "repaired")
        self.assertEqual(self.coordinator.submit_and_verify(now=None)["grading"]["reward"], 1)

    def test_observation_mismatch_and_boundary_tamper_are_rejected(self):
        def wrong(prompt):
            return PolicyChoice("0" * 64, {"action": "submit"})
        with self.assertRaisesRegex(RecoveryError, "policy_choice_not_bound"):
            self.coordinator.run_turn(self.action, wrong, now=None)
        self.assertEqual(self.adapter.verifications, 0)
        self.assertFalse((self.coordinator.journal.directory / "policy-turn-000001.json").exists())

    def test_next_action_and_committed_boundary_are_checked(self):
        self.coordinator.run_turn(self.action,
                                  _next({"action": "run_visible_checks"}), now=None)
        with self.assertRaisesRegex(RecoveryError, "action_differs"):
            self.coordinator.run_turn(self.action, _next({"action": "submit"}), now=None)
        boundary = self.coordinator.journal.directory / "policy-turn-000001.json"
        content = json.loads(boundary.read_text())
        content["observation_sha256"] = "0" * 64
        boundary.write_text(json.dumps(content))
        with self.assertRaisesRegex(RecoveryError, "policy_boundary_missing_or_changed"):
            self.coordinator.run_turn({"action": "run_visible_checks"},
                                      _next({"action": "submit"}), now=None)
        self.assertEqual(self.coordinator.journal.manifest["turn"], 1)

    def test_serial_control_runs_policy_only_after_checkpoint(self):
        self.coordinator.close()
        self.coordinator = AsyncVMCheckpointCoordinator(
            self.adapter, Path(self.temp.name) / "serial-journal", mode="serial",
            inspector=lambda: {"processes": [], "tree": "fake"})
        self.coordinator.begin()

        def policy(prompt):
            self.assertTrue(self.vm.entered.is_set())
            self.assertEqual(self.coordinator.journal.manifest["turn"], 1)
            return PolicyChoice(prompt.observation_sha256, {"action": "submit"})

        result = self.coordinator.run_turn(self.action, policy, now=None)
        self.assertEqual(result["measurements"]["checkpoint_policy_overlap_seconds"], 0)
        self.assertEqual(self.coordinator.submit_and_verify(now=None)["grading"]["reward"], 1)


if __name__ == "__main__":
    unittest.main()
