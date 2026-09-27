"""RealWorldEnv integration gates for the opt-in namespaced verifier path."""

import hashlib
import base64
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from future_prediction_bench.microvm_coding import MicroVMCodingAdapter
from future_prediction_bench.microvm_runtime import MicroVMRuntimeError
from future_prediction_bench.realworld import RealWorldEnv
from future_prediction_bench.stateless_verifier import GUEST_PROGRAM
from future_prediction_bench.stateless_verifier import BATCH_OVERFLOW_MARKER
from test_microvm_coding import FakeRuntime, task
from test_stateless_verifier import FakeGuest


class IntegratedRuntime(FakeRuntime):
    def __init__(self, disk_path):
        super().__init__(disk_path)
        self.batch_data = b""
        self.outputs = {"print('private')": b"private\n",
                        "print('second')": b"second\n"}
        self.fail_batch = False
        self.force_batch_overflow = False
        self.background_writer = False
        self.spawn_on_visible = False
        self.fork_with_extra_process = False
        self.forked_child = None
        self.inject_saved_background_writer = False
        self.missing_baseline_process = False
        self.changed_baseline_executable = False
        self.inject_saved_missing_baseline = False
        self.inject_saved_changed_executable = False
        self.fail_restores = 0

    def run_shell(self, command, *, timeout):
        if "chroot /mnt/root /usr/local/bin/python3.12 -I -S -B -c " in command:
            self.commands.append(command)
            processes = [[1, 0, "/usr/bin/busybox"], [2, 20, "/usr/bin/busybox"]]
            if self.missing_baseline_process:
                processes.pop()
            if self.changed_baseline_executable:
                processes[0][2] = "/usr/local/bin/python3.12"
            if self.background_writer:
                processes.append([900, 9000, "/usr/local/bin/python3.12"])
            encoded = base64.b64encode(json.dumps(processes).encode()).decode()
            return {"return_code": 0, "stdout": "FPB_PROCESS_SET=" + encoded}
        if self.spawn_on_visible and "chroot /mnt/root /usr/local/bin/python3.12 -I -c" in command:
            self.background_writer = True
        stateless = ("fpb_guest_stateless_verifier.py" in command
                     or "fpb_stateless_cases.json" in command
                     or command == "modprobe overlay"
                     or "/mnt/root/tmp" in command)
        if stateless:
            if self.fail_batch and "--batch" in command:
                self.commands.append(command)
                return {"return_code": 1, "stdout": "infrastructure failure"}
            if self.force_batch_overflow and "--batch" in command:
                self.commands.append(command)
                return {"return_code": 0, "stdout": BATCH_OVERFLOW_MARKER}
            return FakeGuest.run_shell(self, command, timeout=timeout)
        return super().run_shell(command, timeout=timeout)

    def save_snapshot(self, tag):
        self.commands.append("SNAPSHOT:" + tag)
        self.saved_background_writer = (self.background_writer
                                        or self.inject_saved_background_writer)
        self.saved_missing_baseline_process = (self.missing_baseline_process
                                               or self.inject_saved_missing_baseline)
        self.saved_changed_baseline_executable = (self.changed_baseline_executable
                                                  or self.inject_saved_changed_executable)
        return super().save_snapshot(tag)

    def load_snapshot(self, tag):
        if self.fail_restores:
            self.fail_restores -= 1
            self.commands.append("LOAD_FAILED:" + tag)
            raise MicroVMRuntimeError("synthetic_restore_failure")
        self.commands.append("LOAD:" + tag)
        self.background_writer = self.saved_background_writer
        self.missing_baseline_process = self.saved_missing_baseline_process
        self.changed_baseline_executable = self.saved_changed_baseline_executable
        return super().load_snapshot(tag)

    def fork_snapshot(self, tag, child_disk_paths):
        if not tag.startswith("br") or len(child_disk_paths) != 1:
            raise ValueError("Expected one issued branch snapshot")
        child_path = Path(child_disk_paths[0])
        child_path.write_bytes(self.disk_path.read_bytes())
        child = IntegratedRuntime(child_path)
        child.files = dict(self.saved_files)
        child.background_writer = (self.saved_background_writer
                                   or self.fork_with_extra_process)
        child.commands.append("RESTORE_BRANCH:" + tag)
        self.forked_child = child
        return [child]


class MicroVMStatelessIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.verifier_dir = self.root / "verifier"
        self.verifier_dir.mkdir()
        self.disk = self.root / "disk.qcow2"
        self.disk.write_bytes(b"fake qcow2")
        self.task = task()
        (self.root / "task.json").write_text(json.dumps(self.task), encoding="utf-8")
        self.spec = {"kind": "command_cases_v1", "cases": [
            {"argv": ["python3", "-B", "-c", "print('private')"],
             "expected_stdout": "private\n", "expected_returncode": 0},
            {"argv": ["python3", "-B", "-c", "print('second')"],
             "expected_stdout": "second\n", "expected_returncode": 0}]}
        self.verifier_file = self.verifier_dir / "verify.json"
        self.verifier_file.write_text(json.dumps(self.spec), encoding="utf-8")
        self.contract = self.root / "stateless_contract.json"
        self.contract.write_text(json.dumps({
            "kind": "stateless_python_cases_overlay_v1",
            "task_id": self.task["task_id"],
            "verifier_sha256": hashlib.sha256(self.verifier_file.read_bytes()).hexdigest(),
            "requires_live_background_process_state": False,
            "requires_shared_case_filesystem_state": False,
            "requires_quiescent_submitted_state": True,
            "allow_unprivileged_case_execution": True}), encoding="utf-8")

    def _environment(self):
        runtime = IntegratedRuntime(self.disk)
        adapter = MicroVMCodingAdapter(
            runtime, verifier_dir=self.verifier_dir,
            visible_check=["python3", "-B", "-c", "print('visible')"],
            stateless_verifier_contract=self.contract,
            stateless_task_path=self.root / "task.json")
        bound = dict(self.task)
        bound["metadata"] = {"artifact_binding": adapter.artifact_binding()}
        env = RealWorldEnv(bound, adapter)
        return env, adapter, runtime

    def test_submit_prepares_private_case_codes_before_snapshot_then_grades_one_batch(self):
        env, adapter, runtime = self._environment()
        try:
            binding = adapter.artifact_binding()
            self.assertEqual(binding["stateless_contract_sha256"],
                             hashlib.sha256(self.contract.read_bytes()).hexdigest())
            self.assertEqual(binding["stateless_helper_sha256"],
                             hashlib.sha256(GUEST_PROGRAM.read_bytes()).hexdigest())
            env.reset("policy")
            self.assertFalse(any("fpb_stateless_cases.json" in command
                                 for command in runtime.commands))
            env.step({"action": "submit"})
            commands = runtime.commands
            self.assertLess(next(i for i, command in enumerate(commands)
                                 if "sha256sum /mnt/root/fpb_stateless_cases.json" in command),
                            commands.index("SNAPSHOT:submitted"))
            result = env.verify()
            self.assertEqual((result["status"], result["reward"]), ("graded", 1.0))
            self.assertEqual(runtime.restores, 1)
            self.assertEqual(adapter.metrics["hidden_cases"], 2)
            self.assertEqual(adapter.metrics["stateless_batches"], 1)
            self.assertEqual(adapter.metrics["stateless_batch_fallbacks"], 0)
            self.assertGreaterEqual(adapter.metrics["vm_submit_snapshot_seconds"], 0)
            self.assertGreaterEqual(adapter.metrics["stateless_verify_restore_seconds"], 0)
            self.assertGreaterEqual(adapter.metrics["stateless_verify_precheck_seconds"], 0)
            self.assertGreaterEqual(adapter.metrics["stateless_verify_batch_seconds"], 0)
            self.assertEqual(sum("--batch" in command for command in commands), 1)
            self.assertNotIn("expected_stdout", "".join(commands))
            self.assertEqual(env.get_state()["adapter_state"]["verifier_mode"],
                             "stateless_namespaced_batch_v1")
            self.assertEqual(result["evidence"]["kind"],
                             "host_checked_guest_stateless_namespaced_cases_v1")
            self.assertFalse(result["evidence"]["batch_fallback_used"])
        finally:
            adapter.close()

    def test_snapshot_contaminated_during_save_stays_pending_before_hidden_cases(self):
        env, adapter, runtime = self._environment()
        try:
            env.reset("policy")
            runtime.inject_saved_background_writer = True
            submitted = env.step({"action": "submit"})
            self.assertEqual(submitted["observation"]["status"], "submitted")
            self.assertEqual(runtime.restores, 0)
            for _ in range(2):
                result = env.verify()
                self.assertEqual((result["status"], result["reward"]),
                                 ("pending", None))
                self.assertEqual(adapter.metrics["hidden_cases"], 0)
                self.assertEqual(adapter.metrics["stateless_batches"], 0)
            self.assertFalse(any("--batch" in command for command in runtime.commands))
        finally:
            adapter.close()

    def test_saved_snapshot_missing_baseline_process_stays_pending(self):
        env, adapter, runtime = self._environment()
        try:
            env.reset("policy")
            runtime.inject_saved_missing_baseline = True
            submitted = env.step({"action": "submit"})
            self.assertEqual(submitted["observation"]["status"], "submitted")
            result = env.verify()
            self.assertEqual((result["status"], result["reward"]), ("pending", None))
            self.assertEqual(adapter.metrics["hidden_cases"], 0)
            self.assertEqual(adapter.metrics["stateless_batches"], 0)
            self.assertFalse(any("--batch" in command for command in runtime.commands))
        finally:
            adapter.close()

    def test_saved_snapshot_changed_baseline_executable_stays_pending(self):
        env, adapter, runtime = self._environment()
        try:
            env.reset("policy")
            runtime.inject_saved_changed_executable = True
            submitted = env.step({"action": "submit"})
            self.assertEqual(submitted["observation"]["status"], "submitted")
            result = env.verify()
            self.assertEqual((result["status"], result["reward"]), ("pending", None))
            self.assertEqual(adapter.metrics["hidden_cases"], 0)
            self.assertEqual(adapter.metrics["stateless_batches"], 0)
            self.assertFalse(any("--batch" in command for command in runtime.commands))
        finally:
            adapter.close()

    def test_expected_only_value_never_reaches_guest_transport(self):
        sentinel = "EXPECTED_ONLY_SECRET_SENTINEL_8f2c\n"
        self.spec["cases"][0]["expected_stdout"] = sentinel
        self.verifier_file.write_text(json.dumps(self.spec), encoding="utf-8")
        contract = json.loads(self.contract.read_text(encoding="utf-8"))
        contract["verifier_sha256"] = hashlib.sha256(self.verifier_file.read_bytes()).hexdigest()
        self.contract.write_text(json.dumps(contract), encoding="utf-8")
        env, adapter, runtime = self._environment()
        try:
            env.reset("policy")
            env.step({"action": "submit"})
            result = env.verify()
            self.assertEqual((result["status"], result["reward"]), ("graded", 0.0))
            guest_commands = "".join(runtime.commands)
            self.assertNotIn(sentinel, guest_commands)
            self.assertNotIn("EXPECTED_ONLY_SECRET_SENTINEL_8f2c", guest_commands)
            self.assertNotIn(sentinel.encode(), runtime.batch_data)
            self.assertNotIn("EXPECTED_ONLY_SECRET_SENTINEL_8f2c".encode(), runtime.batch_data)
            self.assertNotIn(b"expected_stdout", runtime.batch_data)
        finally:
            adapter.close()

    def test_transient_post_submit_process_is_erased_by_mandatory_restore(self):
        env, adapter, runtime = self._environment()
        try:
            env.reset("policy")
            env.step({"action": "submit"})
            runtime.background_writer = True
            result = env.verify()
            self.assertEqual((result["status"], result["reward"]), ("graded", 1.0))
            self.assertEqual(runtime.restores, 1)
            self.assertEqual(adapter.metrics["hidden_cases"], 2)
            self.assertFalse(runtime.background_writer)
        finally:
            adapter.close()

    def test_restore_failure_retries_without_scoring_hidden_cases(self):
        env, adapter, runtime = self._environment()
        try:
            env.reset("policy")
            env.step({"action": "submit"})
            runtime.fail_restores = 1
            first = env.verify()
            self.assertEqual((first["status"], first["reward"]), ("pending", None))
            self.assertEqual(adapter.metrics["hidden_cases"], 0)
            self.assertEqual(adapter.metrics["stateless_batches"], 0)
            self.assertFalse(any("--batch" in command for command in runtime.commands))
            second = env.verify()
            self.assertEqual((second["status"], second["reward"]), ("graded", 1.0))
            self.assertEqual(runtime.restores, 1)
            self.assertEqual(adapter.metrics["hidden_cases"], 2)
            self.assertEqual(sum("--batch" in command for command in runtime.commands), 1)
        finally:
            adapter.close()

    def test_delayed_verify_after_defers_restore_and_hidden_batch(self):
        current = datetime.now(timezone.utc)
        due = current + timedelta(minutes=5)
        self.task["verify_after"] = due.isoformat()
        (self.root / "task.json").write_text(json.dumps(self.task), encoding="utf-8")
        env, adapter, runtime = self._environment()
        try:
            env.clock = lambda: current
            env.reset("policy")
            env.step({"action": "submit"})
            before = list(runtime.commands)
            pending = env.verify()
            self.assertEqual((pending["status"], pending["reason"], pending["reward"]),
                             ("pending", "not_due", None))
            self.assertEqual(runtime.commands, before)
            self.assertEqual(runtime.restores, 0)
            self.assertEqual(adapter.metrics["hidden_cases"], 0)
            env.clock = lambda: due + timedelta(seconds=1)
            graded = env.verify()
            self.assertEqual((graded["status"], graded["reward"]), ("graded", 1.0))
            self.assertEqual(runtime.restores, 1)
            self.assertEqual(adapter.metrics["hidden_cases"], 2)
        finally:
            adapter.close()

    def test_post_batch_process_blocks_reward_until_clean_retry(self):
        env, adapter, runtime = self._environment()
        try:
            env.reset("policy")
            env.step({"action": "submit"})
            original_grade = adapter.stateless_verifier.grade_batch

            def grade_with_writer():
                result = original_grade()
                runtime.background_writer = True
                return result

            adapter.stateless_verifier.grade_batch = grade_with_writer
            first = env.verify()
            self.assertEqual((first["status"], first["reward"]), ("pending", None))
            self.assertEqual(adapter.metrics["hidden_cases"], 0)
            self.assertEqual(adapter.metrics["stateless_batches"], 0)
            adapter.stateless_verifier.grade_batch = original_grade
            second = env.verify()
            self.assertEqual((second["status"], second["reward"]), ("graded", 1.0))
            self.assertEqual(runtime.restores, 2)
            self.assertEqual(adapter.metrics["hidden_cases"], 2)
        finally:
            adapter.close()

    def test_batch_infrastructure_failure_keeps_reward_pending(self):
        env, adapter, runtime = self._environment()
        try:
            env.reset("policy")
            env.step({"action": "submit"})
            runtime.fail_batch = True
            result = env.verify()
            self.assertEqual(result["status"], "pending")
            self.assertIsNone(result["reward"])
            self.assertEqual(env.get_state()["status"], "pending")
            self.assertEqual(adapter.metrics["hidden_cases"], 0)
        finally:
            adapter.close()

    def test_batch_overflow_falls_back_to_exact_scored_cases(self):
        env, adapter, runtime = self._environment()
        try:
            env.reset("policy")
            env.step({"action": "submit"})
            runtime.force_batch_overflow = True
            runtime.outputs["print('private')"] = b"x" * 8000
            result = env.verify()
            self.assertEqual((result["status"], result["reward"]), ("graded", 0.0))
            self.assertTrue(result["evidence"]["batch_fallback_used"])
            self.assertEqual(adapter.metrics["hidden_cases"], 2)
            self.assertEqual(adapter.metrics["stateless_batch_fallbacks"], 1)
            self.assertEqual(sum("/fpb_guest_stateless_verifier.py '" in command
                                 for command in runtime.commands), 2)
        finally:
            adapter.close()

    def test_modified_contract_before_submit_interrupts_instead_of_grading(self):
        env, adapter, runtime = self._environment()
        try:
            env.reset("policy")
            self.contract.write_text(self.contract.read_text() + " ", encoding="utf-8")
            outcome = env.step({"action": "submit"})
            self.assertEqual(outcome["info"]["status"], "interrupted")
            self.assertIsNone(outcome["reward"])
            self.assertFalse(any(command == "SNAPSHOT:submitted"
                                 for command in runtime.commands))
        finally:
            adapter.close()

    def test_visible_check_background_writer_fails_closed_before_submit(self):
        env, adapter, runtime = self._environment()
        try:
            env.reset("policy")
            runtime.spawn_on_visible = True
            self.assertTrue(env.step({"action": "run_visible_checks"})["observation"]["passed"])
            self.assertTrue(any("user=65534,group=65534" in command
                                for command in runtime.commands))
            outcome = env.step({"action": "submit"})
            self.assertEqual(outcome["info"]["status"], "interrupted")
            self.assertIsNone(outcome["reward"])
            self.assertFalse(any(command == "SNAPSHOT:submitted"
                                 for command in runtime.commands))
        finally:
            adapter.close()

    def test_malformed_contract_rejected_before_guest_start(self):
        contract = json.loads(self.contract.read_text(encoding="utf-8"))
        contract["requires_quiescent_submitted_state"] = False
        self.contract.write_text(json.dumps(contract), encoding="utf-8")
        with self.assertRaises(ValueError):
            self._environment()

    def test_source_task_with_existing_frozen_binding_matches_episode(self):
        source = dict(self.task)
        source["metadata"] = {"artifact_binding": {"prior_registration": True}}
        (self.root / "task.json").write_text(json.dumps(source), encoding="utf-8")
        env, adapter, _ = self._environment()
        try:
            self.assertEqual(env.reset("policy")["task"]["task_id"], self.task["task_id"])
        finally:
            adapter.close()

    def test_unexpected_baseline_daemon_fails_before_policy_actions(self):
        env, adapter, runtime = self._environment()
        runtime.background_writer = True
        try:
            with self.assertRaisesRegex(ValueError, "Adapter reset failed") as captured:
                env.reset("policy")
            self.assertEqual(str(captured.exception.__cause__), "vm_episode_setup_failed")
            self.assertEqual(env.status, "setup_error")
            self.assertFalse(adapter.started)
        finally:
            adapter.close()

    def test_stateless_branch_restores_clean_baseline_and_grades(self):
        env, adapter, runtime = self._environment()
        branch_adapter = None
        try:
            env.reset("policy")
            checkpoint = env.create_branch_checkpoint()
            branch_adapter = adapter.branch_adapter(self.root / "branch.qcow2")
            branch = env.fork_from_checkpoint(checkpoint, branch_adapter,
                                              branch_id="clean-sibling")
            self.assertEqual(branch_adapter._baseline_processes,
                             adapter._baseline_processes)
            branch.step({"action": "submit"})
            result = branch.verify()
            self.assertEqual((result["status"], result["reward"]),
                             ("graded", 1.0))
            self.assertEqual(branch_adapter.metrics["hidden_cases"], 2)
            self.assertEqual(branch_adapter.metrics["stateless_batches"], 1)
            self.assertFalse(runtime.closed)
        finally:
            if branch_adapter is not None:
                branch_adapter.close()
            adapter.close()

    def test_stateless_branch_rejects_live_background_process_at_checkpoint(self):
        env, adapter, runtime = self._environment()
        try:
            env.reset("policy")
            runtime.background_writer = True
            with self.assertRaisesRegex(Exception, "stateless_branch_not_quiescent"):
                env.create_branch_checkpoint()
            self.assertFalse(any(command.startswith("SNAPSHOT:br")
                                 for command in runtime.commands))
        finally:
            adapter.close()

    def test_stateless_branch_rejects_unexpected_process_after_restore(self):
        env, adapter, runtime = self._environment()
        branch_adapter = None
        try:
            env.reset("policy")
            checkpoint = env.create_branch_checkpoint()
            runtime.fork_with_extra_process = True
            branch_adapter = adapter.branch_adapter(self.root / "dirty-branch.qcow2")
            with self.assertRaisesRegex(ValueError, "Branch adapter reset failed"):
                env.fork_from_checkpoint(checkpoint, branch_adapter,
                                         branch_id="dirty-sibling")
            self.assertFalse(branch_adapter.started)
            self.assertTrue(runtime.forked_child.closed)
            self.assertEqual(branch_adapter.metrics["hidden_cases"], 0)
        finally:
            if branch_adapter is not None:
                branch_adapter.close()
            adapter.close()


if __name__ == "__main__":
    unittest.main()
