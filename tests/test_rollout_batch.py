"""Behavioral checks for the bounded real-world coding episode queue."""

from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from future_prediction_bench.rollout_batch import run_coding_batch


def _task(index, *, fixture=True):
    now = datetime.now(timezone.utc)
    return {
        "schema_version": "realworld-0.1", "task_id": f"batch-task-{index}",
        "event_id": f"batch-event-{index}", "cluster_id": f"batch-cluster-{index}",
        "split": "train", "prompt": "Repair the supplied module.",
        "issued_at": (now - timedelta(minutes=1)).isoformat(),
        "action_deadline": (now + timedelta(minutes=5)).isoformat(),
        "outcome_not_before": (now - timedelta(minutes=1)).isoformat(),
        "verify_after": (now - timedelta(minutes=1)).isoformat(),
        "tool_manifest": [{"name": "submit", "description": "Submit the workspace."}],
        "reward_contract": {"id": "hidden-tests", "description": "All tests pass.",
                            "min_reward": 0, "max_reward": 1},
        "budgets": {"max_actions": 2, "max_wall_seconds": 30},
        "is_fixture": fixture,
    }


def _job(index, *, fixture=True):
    return {"task": _task(index, fixture=fixture), "seed_dir": "/seed",
            "verifier_dir": "/verifier", "image": "local-image",
            "actions": [{"action": "submit"}], "policy_id": f"policy-{index}"}


class CodingBatchTests(unittest.TestCase):
    def test_bounded_concurrency_order_and_stage_timing(self):
        active = 0
        highest = 0
        completion_order = []
        lock = threading.Lock()
        first_pair = threading.Barrier(2, timeout=3)

        def runner(**kwargs):
            nonlocal active, highest
            index = int(kwargs["task"]["task_id"].rsplit("-", 1)[1])
            with lock:
                active += 1
                highest = max(highest, active)
            try:
                if index < 2:
                    first_pair.wait()
                time.sleep(0.03 if index == 0 else 0.005)
                output = Path(kwargs["output"])
                output.mkdir(parents=True)
                (output / "report.json").write_text(json.dumps({
                    "environment_metrics": {"reset_seconds": 0.002,
                                            "action_seconds": 0.003,
                                            "verifier_seconds": 0.004},
                    "adapter_metrics": {"docker_start_seconds": 0.001,
                                        "checkpoint_seconds": 0.0005,
                                        "verify_seconds": 0.004},
                }), encoding="utf-8")
                with lock:
                    completion_order.append(index)
                return {"status": "graded", "reward": 1.0, "output": str(output)}
            finally:
                with lock:
                    active -= 1

        with tempfile.TemporaryDirectory() as tmp:
            report = run_coding_batch(jobs=[_job(index) for index in range(4)],
                                      output_root=Path(tmp) / "batch",
                                      max_workers=2, runner=runner)
            persisted = json.loads((Path(tmp) / "batch" / "batch_report.json").read_text())
        self.assertEqual(highest, 2)
        self.assertNotEqual(completion_order, [0, 1, 2, 3])
        self.assertEqual([item["index"] for item in report["jobs"]], [0, 1, 2, 3])
        self.assertEqual(report, persisted)
        self.assertEqual(report["completed_count"], 4)
        self.assertEqual(report["error_count"], 0)
        self.assertFalse(report["phase_separation"])
        self.assertEqual(report["jobs"][0]["timings"]["episode_stage_seconds"]["environment_verify_seconds"], 0.004)
        self.assertTrue(all(item["timings"]["total_seconds"] >= item["timings"]["episode_seconds"] for item in report["jobs"]))
        self.assertTrue(all(item["task_sha256"] for item in report["jobs"]))

    def test_failure_isolation_and_no_invented_stage_timing(self):
        called = []

        def runner(**kwargs):
            task_id = kwargs["task"]["task_id"]
            called.append(task_id)
            if task_id.endswith("-1"):
                raise RuntimeError("verifier unavailable")
            return {"status": "pending", "reward": None}

        with tempfile.TemporaryDirectory() as tmp:
            report = run_coding_batch(jobs=[_job(index) for index in range(3)],
                                      output_root=Path(tmp) / "batch",
                                      max_workers=1, runner=runner)
        self.assertEqual(called, ["batch-task-0", "batch-task-1", "batch-task-2"])
        self.assertEqual(report["completed_count"], 2)
        self.assertEqual(report["error_count"], 1)
        self.assertEqual(report["jobs"][1]["error_type"], "RuntimeError")
        self.assertEqual(report["jobs"][1]["error"], "verifier unavailable")
        self.assertIsNone(report["jobs"][0]["timings"]["episode_stage_seconds"])
        self.assertEqual(report["jobs"][2]["episode_result"]["status"], "pending")

    def test_requires_policy_id_and_registry_for_non_fixture(self):
        called = []

        def runner(**kwargs):
            called.append(kwargs)
            return {"status": "graded"}

        missing_policy = _job(0)
        del missing_policy["policy_id"]
        with tempfile.TemporaryDirectory() as tmp:
            report = run_coding_batch(jobs=[missing_policy, _job(1, fixture=False), _job(2)],
                                      output_root=Path(tmp) / "batch",
                                      max_workers=1, runner=runner)
        self.assertEqual(report["error_count"], 2)
        self.assertEqual([item["task_id"] for item in report["jobs"]], [None, "batch-task-1", "batch-task-2"])
        self.assertEqual(len(called), 1)
        self.assertEqual(called[0]["policy_id"], "policy-2")
        self.assertEqual(report["jobs"][0]["error_type"], "ValueError")
        self.assertIn("registry", report["jobs"][1]["error"])

    def test_batch_level_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "batch"
            output.mkdir()
            (output / "previous.txt").write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "new or empty"):
                run_coding_batch(jobs=[_job(0)], output_root=output)
            with self.assertRaisesRegex(ValueError, "max_workers"):
                run_coding_batch(jobs=[_job(0)], output_root=Path(tmp) / "new", max_workers=True)
            with self.assertRaisesRegex(ValueError, "nonempty sequence"):
                run_coding_batch(jobs=[], output_root=Path(tmp) / "new")


if __name__ == "__main__":
    unittest.main()
