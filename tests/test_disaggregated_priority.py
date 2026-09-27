"""Trusted verifier priority must reorder only queued work, never policy inputs."""

from __future__ import annotations

import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from examples.realworld_boltons26.benchmark_verifier_priority import benchmark
from future_prediction_bench.disaggregated_rollout import run_disaggregated_coding_batch


def _task(index):
    now = datetime.now(timezone.utc)
    return {
        "schema_version": "realworld-0.1", "task_id": f"priority-{index}",
        "event_id": f"priority-{index}", "cluster_id": f"priority-{index}",
        "split": "train", "prompt": "Repair the supplied module.",
        "issued_at": (now - timedelta(minutes=1)).isoformat(),
        "action_deadline": (now + timedelta(minutes=5)).isoformat(),
        "outcome_not_before": (now - timedelta(minutes=1)).isoformat(),
        "verify_after": (now - timedelta(minutes=1)).isoformat(),
        "tool_manifest": [{"name": "submit", "description": "Submit the workspace."}],
        "reward_contract": {"id": "hidden-tests", "description": "All tests pass.",
                            "min_reward": 0, "max_reward": 1},
        "budgets": {"max_actions": 2, "max_wall_seconds": 30},
        "is_fixture": True,
    }


def _job(index, priority=None):
    job = {"task": _task(index), "seed_dir": "/unused-seed",
           "verifier_dir": "/unused-verifier", "image": "test-image",
           "actions": [{"action": "submit"}], "policy_id": f"policy-{index}"}
    if priority is not None:
        job["verifier_priority"] = priority
    return job


class _Trace:
    def __init__(self):
        self.lock = threading.Lock()
        self.first_verify_started = threading.Event()
        self.all_submitted = threading.Event()
        self.submitted = 0
        self.verify_order = []
        self.adapter_kwargs = []
        self.task_keys = []
        self.action_keys = []


class _Adapter:
    def __init__(self, trace, **kwargs):
        self.trace = trace
        self.task_id = None
        self.image_id = "sha256:" + "0" * 64
        with trace.lock:
            trace.adapter_kwargs.append(set(kwargs))

    def artifact_binding(self):
        return {"seed_workspace_sha256": "1" * 64,
                "verifier_sha256": "2" * 64,
                "image_sha256": self.image_id,
                "visible_check_sha256": "3" * 64}

    def reset(self, task, *, now):
        self.task_id = task["task_id"]
        with self.trace.lock:
            self.trace.task_keys.append(set(task))
        if self.task_id == "priority-1":
            if not self.trace.first_verify_started.wait(timeout=3):
                raise TimeoutError("First verifier did not start")
        return {"status": "ready"}

    def step(self, action, *, now):
        with self.trace.lock:
            self.trace.action_keys.append(set(action))
            self.trace.submitted += 1
            if self.trace.submitted == 4:
                self.trace.all_submitted.set()
        return {"observation": {"status": "submitted"}, "terminated": True}

    def verify(self, *, now):
        with self.trace.lock:
            self.trace.verify_order.append(self.task_id)
        if self.task_id == "priority-0":
            self.trace.first_verify_started.set()
            if not self.trace.all_submitted.wait(timeout=3):
                raise TimeoutError("Actor did not hand off all episodes")
        return {"status": "resolved", "reward": 1.0,
                "evidence": {"kind": "test-verifier"},
                "available_at": datetime.now(timezone.utc).isoformat()}

    def get_state(self):
        return {"metrics": {"verify_seconds": 0.0}}

    def close(self):
        pass


class VerifierPriorityTests(unittest.TestCase):
    def test_controlled_benchmark_uses_equal_workers_and_exact_rewards(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = benchmark(output=Path(tmp) / "benchmark", repetitions=1,
                               slow_seconds=0.03, fast_seconds=0.001)
        self.assertEqual(result["worker_budget"]["actor_workers"], 1)
        self.assertEqual(result["worker_budget"]["verification_workers"], 1)
        self.assertFalse(result["measures_rl_training"])
        self.assertEqual([run["verification_order"] for run in result["runs"]],
                         [[0, 1, 2, 3, 4], [0, 2, 3, 4, 1]])
        self.assertTrue(all(run["rewards"] == [1.0, 0.0, 1.0, 0.0, 1.0]
                            for run in result["runs"]))

    def test_priority_reorders_queued_work_without_policy_injection(self):
        trace = _Trace()
        jobs = [_job(0), _job(1, 8), _job(2, 0), _job(3, 1)]
        with tempfile.TemporaryDirectory() as tmp:
            report = run_disaggregated_coding_batch(
                jobs=jobs, output_root=Path(tmp) / "priority",
                actor_workers=1, verification_workers=1,
                actor_queue_capacity=4, verifier_queue_capacity=4,
                adapter_factory=lambda **kwargs: _Adapter(trace, **kwargs))
        self.assertEqual(trace.verify_order,
                         ["priority-0", "priority-2", "priority-3", "priority-1"])
        self.assertEqual(report["graded_count"], 4)
        self.assertEqual(report["error_count"], 0)
        self.assertEqual([item["reward"] for item in report["jobs"]], [1.0] * 4)
        self.assertEqual([item["verifier_priority"] for item in report["jobs"]], [5, 8, 0, 1])
        self.assertTrue(all("verifier_priority" not in kwargs for kwargs in trace.adapter_kwargs))
        self.assertTrue(all("verifier_priority" not in keys for keys in trace.task_keys))
        self.assertTrue(all("verifier_priority" not in keys for keys in trace.action_keys))
        self.assertTrue(all(item["timings"]["completion_offset_seconds"] is not None
                            for item in report["jobs"]))

    def test_equal_priorities_preserve_handoff_order(self):
        trace = _Trace()
        with tempfile.TemporaryDirectory() as tmp:
            report = run_disaggregated_coding_batch(
                jobs=[_job(index) for index in range(4)],
                output_root=Path(tmp) / "fifo", actor_workers=1,
                verification_workers=1, actor_queue_capacity=4,
                verifier_queue_capacity=4,
                adapter_factory=lambda **kwargs: _Adapter(trace, **kwargs))
        self.assertEqual(trace.verify_order, [f"priority-{index}" for index in range(4)])
        self.assertEqual(report["error_count"], 0)

    def test_rejects_invalid_priorities_as_job_errors(self):
        for invalid in (True, -1, 10, "0", 1.5):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as tmp:
                job = _job(0)
                job["verifier_priority"] = invalid
                report = run_disaggregated_coding_batch(
                    jobs=[job], output_root=Path(tmp) / "bad",
                    adapter_factory=lambda **kwargs: _Adapter(_Trace(), **kwargs))
                self.assertEqual(report["error_count"], 1)
                self.assertIn("verifier_priority", report["jobs"][0]["error"])


if __name__ == "__main__":
    unittest.main()
