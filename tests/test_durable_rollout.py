"""Lease fencing and submitted-state binding without a Docker daemon."""

from __future__ import annotations

import copy
import json
import shutil
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from future_prediction_bench import durable_rollout
from future_prediction_bench.coding_env import DockerCodingAdapter, _workspace_digest
from future_prediction_bench.durable_rollout import DurableDockerRolloutQueue, LeaseLost
from future_prediction_bench.realworld import RealWorldEnv, RealWorldTaskRegistry


class _LocalSubmittedAdapter(DockerCodingAdapter):
    """Exercise the real host submission boundary; replace only Docker I/O."""

    IMAGE_ID = "sha256:" + "a" * 64

    def artifact_binding(self):
        return {"seed_workspace_sha256": _workspace_digest(self.seed_dir),
                "verifier_sha256": _workspace_digest(self.verifier_dir),
                "image_sha256": self.IMAGE_ID,
                "visible_check_sha256": "b" * 64}

    def reset(self, task, *, now):
        self.expected_binding = self.artifact_binding()
        self.image_id = self.IMAGE_ID
        self.task_sha256 = task["task_sha256"]
        self.instance_root = self.output_root / "local-instance"
        self.instance_root.mkdir(parents=True)
        self.workspace = self.instance_root / "workspace"
        shutil.copytree(self.seed_dir, self.workspace)
        (self.instance_root / "snapshots").mkdir()
        return {"workspace_sha256": _workspace_digest(self.workspace)}

    def step(self, action, *, now):
        if action != {"action": "submit"}:
            raise ValueError("This test only submits")
        digest = _workspace_digest(self.workspace)
        self.submitted_workspace = self.instance_root / "snapshots" / digest
        shutil.copytree(self.workspace, self.submitted_workspace)
        self.submitted = True
        return {"observation": {"status": "submitted", "workspace_sha256": digest},
                "terminated": True}

    def verify(self, *, now):
        raise AssertionError("The unit test must not invoke Docker")


class _LocalResolvedVerifier(_LocalSubmittedAdapter):
    def verify(self, *, now):
        result = {"status": "resolved", "reward": 1.0,
                  "available_at": datetime.now(timezone.utc).isoformat(),
                  "evidence": {"workspace_sha256": self.submitted_workspace.name,
                               "verifier_sha256": self.expected_binding["verifier_sha256"],
                               "image_sha256": self.image_id,
                               "case_results": [{"passed": True}]}}
        self.verified = result
        return result


class _LocalPendingVerifier(_LocalSubmittedAdapter):
    calls = 0

    def verify(self, *, now):
        type(self).calls += 1
        return {"status": "pending", "reason": "awaiting_external_outcome"}


class _LocalVoidVerifier(_LocalSubmittedAdapter):
    def verify(self, *, now):
        return {"status": "void", "reason": "source_retracted",
                "evidence": {"source_status": "retracted"}}


def _task():
    now = datetime.now(timezone.utc)
    return {"schema_version": "realworld-0.1", "task_id": "durable-docker-test",
            "event_id": "durable-docker-test", "cluster_id": "durable-docker-test",
            "split": "train", "prompt": "Submit a frozen file.",
            "issued_at": (now - timedelta(minutes=1)).isoformat(),
            "action_deadline": (now + timedelta(minutes=10)).isoformat(),
            "outcome_not_before": (now - timedelta(minutes=1)).isoformat(),
            "verify_after": (now - timedelta(minutes=1)).isoformat(),
            "tool_manifest": [{"name": "submit", "description": "Freeze workspace"}],
            "reward_contract": {"id": "exact", "description": "Host check", "min_reward": 0,
                                "max_reward": 1},
            "budgets": {"max_actions": 1, "max_wall_seconds": 600,
                        "max_verifications": 1},
            "is_fixture": True}


class DurableRolloutTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        seed, verifier = self.root / "seed", self.root / "verifier"
        seed.mkdir()
        verifier.mkdir()
        (seed / "sample.py").write_text("answer = 42\n", encoding="utf-8")
        (verifier / "verify.json").write_text(json.dumps({"kind": "command_cases_v1",
                                                       "cases": []}), encoding="utf-8")
        self.adapter = _LocalSubmittedAdapter(
            seed_dir=seed, verifier_dir=verifier, image="test-image",
            output_root=self.root / "episodes" / "first",
            visible_check=("python3", "-B", "-c", "pass"))
        task = _task()
        task["metadata"] = {"artifact_binding": self.adapter.artifact_binding()}
        self.env = RealWorldEnv(task, self.adapter)
        self.env.reset("fixed-policy")
        self.env.step({"action": "submit"})
        self.queue = DurableDockerRolloutQueue(self.root / "queue" / "work.sqlite")

    def _enqueue(self, *, require_queue_revision=False):
        return self.queue.enqueue_submitted("one", self.env,
                                            policy_revision="weights-sha256:abc",
                                            require_queue_revision=require_queue_revision)

    def test_crash_reclaim_fences_old_worker_and_publishes_one_reward(self):
        receipt = self._enqueue()
        self.assertEqual(receipt["snapshot_sha256"], self.adapter.submitted_workspace.name)
        self.assertEqual(receipt, self._enqueue())
        first = self.queue.claim("old-worker", lease_seconds=0.2)
        self.assertIsNone(self.queue.claim("other", lease_seconds=0.2))
        time.sleep(0.24)
        second = DurableDockerRolloutQueue(self.queue.path).claim("replacement", lease_seconds=2)
        self.assertEqual(second.attempt, 2)
        result = {"reward": 1.0, "status": "graded"}
        with self.assertRaises(LeaseLost):
            self.queue._finish(first, result=result)
        self.queue._finish(second, result=result)
        self.assertIsNone(self.queue.claim("third"))
        row = self.queue.get("one")
        self.assertEqual((row["state"], row["reward"], row["attempt"]), ("graded", 1.0, 2))
        with self.assertRaises(LeaseLost):
            self.queue._finish(second, result=result)

    def test_pending_has_no_reward_and_can_retry(self):
        self._enqueue()
        first = self.queue.claim("worker", lease_seconds=1)
        self.queue._finish(first, pending_reason="verifier_unavailable", retry_seconds=0)
        self.assertIsNone(self.queue.get("one")["reward"])
        self.assertEqual(self.queue.get("one")["last_pending_reason"], "verifier_unavailable")
        second = self.queue.claim("replacement", lease_seconds=1)
        self.assertEqual(second.attempt, 2)

    def test_changed_submission_and_candidate_visible_queue_are_rejected(self):
        self._enqueue()
        with self.assertRaisesRegex(ValueError, "different submission"):
            self.queue.enqueue_submitted("one", self.env, policy_revision="other")
        with self.assertRaisesRegex(ValueError, "already assigned"):
            self.queue.enqueue_submitted("different-job", self.env,
                                         policy_revision="weights-sha256:abc")
        candidate_queue = DurableDockerRolloutQueue(
            self.adapter.submitted_workspace / "work.sqlite")
        with self.assertRaisesRegex(ValueError, "disjoint"):
            candidate_queue.enqueue_submitted("bad", self.env, policy_revision="revision")
        writable_queue = DurableDockerRolloutQueue(self.adapter.workspace / "queue.sqlite")
        with self.assertRaisesRegex(ValueError, "disjoint"):
            writable_queue.enqueue_submitted("also-bad", self.env, policy_revision="revision")

    def test_nonfixture_registry_cannot_be_candidate_writable(self):
        adapter = _LocalSubmittedAdapter(
            seed_dir=self.root / "seed", verifier_dir=self.root / "verifier",
            image="test-image", output_root=self.root / "episodes" / "nonfixture",
            visible_check=("python3", "-B", "-c", "pass"))
        task = _task()
        task["is_fixture"] = False
        task["metadata"] = {"artifact_binding": adapter.artifact_binding()}
        env = RealWorldEnv(task, adapter)
        env.reset("fixed-policy")
        env.step({"action": "submit"})
        registry_path = adapter.workspace / "registry" / "tasks.sqlite"
        registry = RealWorldTaskRegistry(registry_path)
        try:
            registry.register(task)
        finally:
            registry.close()
        with self.assertRaisesRegex(ValueError, "disjoint"):
            self.queue.enqueue_submitted(
                "nonfixture", env, policy_revision="weights-sha256:abc",
                registry_path=registry_path)

    def test_mutated_snapshot_remains_ungraded(self):
        self._enqueue()
        (self.adapter.submitted_workspace / "sample.py").write_text("answer = 0\n", encoding="utf-8")
        outcome = self.queue.process_one("worker", revision_provider=lambda: "weights-sha256:abc",
                                         lease_seconds=1, retry_seconds=0)
        self.assertEqual(outcome["status"], "pending")
        self.assertIn("submitted_snapshot_changed", outcome["reason"])
        self.assertIsNone(self.queue.get("one")["reward"])

    def test_revision_mismatch_does_not_run_verifier_or_publish_reward(self):
        self._enqueue()
        outcome = self.queue.process_one("stale-worker", revision_provider=lambda: "new-revision",
                                         lease_seconds=1, retry_seconds=0)
        self.assertEqual(outcome["status"], "pending")
        self.assertIn("policy_revision_changed", outcome["reason"])
        self.assertIsNone(self.queue.get("one")["reward"])

    def test_bounded_backlog_rejects_second_submission(self):
        queue = DurableDockerRolloutQueue(self.root / "bounded" / "work.sqlite",
                                          max_outstanding=1)
        queue.enqueue_submitted("first", self.env, policy_revision="revision")
        second_adapter = _LocalSubmittedAdapter(
            seed_dir=self.root / "seed", verifier_dir=self.root / "verifier",
            image="test-image", output_root=self.root / "episodes" / "second",
            visible_check=("python3", "-B", "-c", "pass"))
        task = _task()
        task["metadata"] = {"artifact_binding": second_adapter.artifact_binding()}
        second_env = RealWorldEnv(task, second_adapter)
        second_env.reset("fixed-policy")
        second_env.step({"action": "submit"})
        with self.assertRaisesRegex(RuntimeError, "backlog is full"):
            queue.enqueue_submitted("second", second_env, policy_revision="revision")

    def test_two_process_compatible_connections_claim_once(self):
        self._enqueue()
        with ThreadPoolExecutor(max_workers=2) as pool:
            claims = list(pool.map(lambda number: DurableDockerRolloutQueue(
                self.queue.path).claim(f"worker-{number}", lease_seconds=2), range(2)))
        self.assertEqual(sum(lease is not None for lease in claims), 1)
        self.assertEqual(self.queue.get("one")["attempt"], 1)

    def test_verifier_code_change_and_mid_verify_revision_change_gate_reward(self):
        self._enqueue()
        with mock.patch("future_prediction_bench.durable_rollout._code_binding",
                        return_value={"changed": "source"}):
            outcome = self.queue.process_one("worker", revision_provider=lambda: "weights-sha256:abc",
                                             lease_seconds=1, retry_seconds=0)
        self.assertEqual(outcome["status"], "pending")
        self.assertIn("trusted_verifier_code_changed", outcome["reason"])
        self.assertIsNone(self.queue.get("one")["reward"])
        revisions = iter(("weights-sha256:abc", "new-revision"))
        with mock.patch.object(self.queue, "_verify", return_value={"status": "graded", "reward": 1}):
            outcome = self.queue.process_one("worker", revision_provider=lambda: next(revisions),
                                             lease_seconds=1, retry_seconds=0)
        self.assertEqual(outcome["status"], "pending")
        self.assertIn("policy_revision_changed_before_reward_commit", outcome["reason"])
        self.assertIsNone(self.queue.get("one")["reward"])

    def test_reopened_env_uses_canonical_verify_and_exports_text_audit(self):
        self._enqueue()
        code = durable_rollout._code_binding()
        with mock.patch.object(durable_rollout, "DockerCodingAdapter", _LocalResolvedVerifier), \
             mock.patch.object(durable_rollout, "_code_binding", return_value=code):
            outcome = self.queue.process_one("worker", revision_provider=lambda: "weights-sha256:abc",
                                             lease_seconds=1)
        row = self.queue.get("one")
        self.assertEqual((outcome["status"], row["state"], row["reward"]),
                         ("graded", "graded", 1.0))
        self.assertEqual(row["result"]["final_state"]["metrics"]["verifications_used"], 1)
        self.assertEqual(row["result"]["final_state"]["events"][-1]["kind"], "verification")
        self.assertEqual(row["result"]["text_trajectory"]["reward"], 1.0)
        self.assertFalse(row["result"]["text_trajectory"]["trainer_ready"])

    def test_pending_metering_persists_and_budget_exhausts_without_second_case_run(self):
        self._enqueue()
        code = durable_rollout._code_binding()
        _LocalPendingVerifier.calls = 0
        with mock.patch.object(durable_rollout, "DockerCodingAdapter", _LocalPendingVerifier), \
             mock.patch.object(durable_rollout, "_code_binding", return_value=code):
            first = self.queue.process_one("worker", revision_provider=lambda: "weights-sha256:abc",
                                           lease_seconds=1, retry_seconds=0)
            self.assertEqual(first["status"], "pending")
            second = self.queue.process_one("worker", revision_provider=lambda: "weights-sha256:abc",
                                            lease_seconds=1, retry_seconds=0)
        self.assertEqual(second["status"], "exhausted")
        self.assertIn("verification_attempt_budget_exhausted", second["reason"])
        self.assertEqual(_LocalPendingVerifier.calls, 1)
        row = self.queue.get("one")
        self.assertEqual((row["state"], row["reward"], row["attempt"]),
                         ("exhausted", None, 2))
        self.assertIsNone(self.queue.claim("third"))

    def test_void_is_terminal_ungraded_and_never_retried(self):
        self._enqueue()
        code = durable_rollout._code_binding()
        with mock.patch.object(durable_rollout, "DockerCodingAdapter", _LocalVoidVerifier), \
             mock.patch.object(durable_rollout, "_code_binding", return_value=code):
            outcome = self.queue.process_one(
                "worker", revision_provider=lambda: "weights-sha256:abc",
                lease_seconds=1)
        row = self.queue.get("one")
        self.assertEqual((outcome["status"], row["state"], row["reward"]),
                         ("void", "void", None))
        self.assertEqual(row["result"]["reason"], "source_retracted")
        self.assertEqual(row["result"]["final_state"]["status"], "void")
        self.assertFalse(row["result"]["trainer_ready"])
        self.assertIsNone(self.queue.claim("another-worker"))

    def test_atomic_queue_revision_fence_commits_matching_generation(self):
        first = self.queue.set_current_revision("fixed-policy", "weights-sha256:abc")
        same = self.queue.set_current_revision("fixed-policy", "weights-sha256:abc")
        self.assertEqual(first, same)
        self.assertEqual(first["generation"], 1)
        receipt = self._enqueue(require_queue_revision=True)
        self.assertEqual(receipt["queue_revision_generation"], 1)
        with mock.patch.object(self.queue, "_verify", return_value={
                "status": "graded", "reward": 1.0}):
            outcome = self.queue.process_one(
                "worker", revision_provider=lambda: "weights-sha256:abc",
                atomic_revision_fence=True, lease_seconds=1)
        row = self.queue.get("one")
        self.assertEqual((outcome["status"], row["state"], row["reward"]),
                         ("graded", "graded", 1.0))
        self.assertEqual(row["result"]["queue_revision_fence"], first)
        self.assertTrue(row["queue_revision_matches_job"])
        self.assertTrue(row["queue_revision_fence_current"])
        self.queue.set_current_revision("fixed-policy", "weights-sha256:new")
        later = self.queue.get("one")
        self.assertEqual((later["state"], later["reward"]), ("graded", 1.0))
        self.assertFalse(later["queue_revision_matches_job"])
        self.assertFalse(later["queue_revision_fence_current"])
        self.assertEqual(later["result"]["queue_revision_fence"]["generation"], 1)

    def test_atomic_revision_flip_before_commit_fences_reward(self):
        self.queue.set_current_revision("fixed-policy", "weights-sha256:abc")
        self._enqueue(require_queue_revision=True)
        with mock.patch.object(self.queue, "_verify", return_value={
                "status": "graded", "reward": 1.0}):
            outcome = self.queue.process_one(
                "worker", revision_provider=lambda: "weights-sha256:abc",
                after_verify=lambda _lease, _result: self.queue.set_current_revision(
                    "fixed-policy", "weights-sha256:new"),
                atomic_revision_fence=True, lease_seconds=1)
        row = self.queue.get("one")
        self.assertEqual(outcome["status"], "stale")
        self.assertIsNone(outcome["reward"])
        self.assertEqual((row["state"], row["reward"], row["result"]),
                         ("stale", None, None))
        self.assertEqual(row["last_pending_reason"], "queue_revision_mismatch_at_commit")
        self.assertEqual(row["queue_revision_generation"], 2)
        self.assertFalse(row["queue_revision_matches_job"])
        self.assertIsNone(self.queue.claim("another-worker"))

    def test_strict_submission_cannot_be_processed_without_worker_fence_flag(self):
        self.queue.set_current_revision("fixed-policy", "weights-sha256:abc")
        self._enqueue(require_queue_revision=True)
        self.queue.set_current_revision("fixed-policy", "weights-sha256:new")
        with mock.patch.object(self.queue, "_verify") as verifier:
            # The provider is stale and the worker omits the optional flag.
            outcome = self.queue.process_one(
                "worker", revision_provider=lambda: "weights-sha256:abc",
                lease_seconds=1)
        verifier.assert_not_called()
        self.assertEqual((outcome["status"], self.queue.get("one")["state"]),
                         ("stale", "stale"))
        self.assertIsNone(self.queue.get("one")["reward"])

    def test_strict_submission_auto_fences_aba_even_without_worker_flag(self):
        self.queue.set_current_revision("fixed-policy", "weights-sha256:abc")
        self._enqueue(require_queue_revision=True)

        def flip_twice(_lease, _result):
            self.queue.set_current_revision("fixed-policy", "weights-sha256:new")
            self.queue.set_current_revision("fixed-policy", "weights-sha256:abc")

        with mock.patch.object(self.queue, "_verify", return_value={
                "status": "graded", "reward": 1.0}):
            outcome = self.queue.process_one(
                "worker", revision_provider=lambda: "weights-sha256:abc",
                after_verify=flip_twice, lease_seconds=1)
        row = self.queue.get("one")
        self.assertEqual((outcome["status"], row["state"], row["reward"]),
                         ("stale", "stale", None))
        self.assertFalse(row["queue_revision_fence_current"])

    def test_atomic_fence_missing_revision_row_fails_closed(self):
        self._enqueue()
        with mock.patch.object(self.queue, "_verify", return_value={
                "status": "graded", "reward": 1.0}) as verifier:
            outcome = self.queue.process_one(
                "worker", revision_provider=lambda: "weights-sha256:abc",
                atomic_revision_fence=True, lease_seconds=1)
        self.assertEqual(outcome["status"], "stale")
        verifier.assert_not_called()
        self.assertIsNone(self.queue.get("one")["reward"])

    def test_atomic_fence_blocks_aba_revision_cycle(self):
        self.queue.set_current_revision("fixed-policy", "weights-sha256:abc")
        receipt = self._enqueue(require_queue_revision=True)
        self.assertEqual(receipt["queue_revision_generation"], 1)

        def flip_twice(_lease, _result):
            self.queue.set_current_revision("fixed-policy", "weights-sha256:new")
            self.queue.set_current_revision("fixed-policy", "weights-sha256:abc")

        with mock.patch.object(self.queue, "_verify", return_value={
                "status": "graded", "reward": 1.0}):
            outcome = self.queue.process_one(
                "worker", revision_provider=lambda: "weights-sha256:abc",
                after_verify=flip_twice, atomic_revision_fence=True,
                lease_seconds=1)
        row = self.queue.get("one")
        self.assertEqual((outcome["status"], row["state"], row["reward"]),
                         ("stale", "stale", None))
        self.assertTrue(row["queue_revision_matches_job"])
        self.assertEqual(row["queue_revision_generation"], 3)
        self.assertFalse(row["queue_revision_generation_matches_job"])
        self.assertFalse(row["queue_revision_fence_current"])
        self.assertEqual(self._enqueue(require_queue_revision=True)["queue_revision_generation"], 1)

    def test_atomic_generation_flip_before_claim_skips_verifier(self):
        self.queue.set_current_revision("fixed-policy", "weights-sha256:abc")
        self._enqueue(require_queue_revision=True)
        self.queue.set_current_revision("fixed-policy", "weights-sha256:new")
        with mock.patch.object(self.queue, "_verify") as verifier:
            outcome = self.queue.process_one(
                "worker", revision_provider=lambda: "weights-sha256:abc",
                atomic_revision_fence=True, lease_seconds=1)
        verifier.assert_not_called()
        self.assertEqual(outcome["status"], "stale")
        self.assertIsNone(self.queue.get("one")["reward"])

    def test_strict_enqueue_requires_matching_queue_revision(self):
        with self.assertRaisesRegex(ValueError, "Queue current revision"):
            self._enqueue(require_queue_revision=True)
        self.queue.set_current_revision("fixed-policy", "weights-sha256:new")
        with self.assertRaisesRegex(ValueError, "Queue current revision"):
            self._enqueue(require_queue_revision=True)
        self.assertIsNone(self.queue.get("one"))


if __name__ == "__main__":
    unittest.main()
