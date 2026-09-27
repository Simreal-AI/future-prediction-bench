"""Pipeline behavior with the real environment core and a deterministic adapter."""

from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from future_prediction_bench.disaggregated_rollout import run_disaggregated_coding_batch


def _task(index):
    now = datetime.now(timezone.utc)
    return {
        "schema_version": "realworld-0.1", "task_id": f"pipeline-{index}",
        "event_id": f"pipeline-{index}", "cluster_id": f"pipeline-{index}",
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


def _job(index):
    return {"task": _task(index), "seed_dir": "/unused-seed",
            "verifier_dir": "/unused-verifier", "image": "test-image",
            "actions": [{"action": "submit"}], "policy_id": f"policy-{index}"}


class _Recorder:
    def __init__(self):
        self.verify_started = threading.Event()
        self.verify_finished = threading.Event()
        self.actor_overlapped_verifier = False
        self.lock = threading.Lock()
        self.closed = []
        self.started = []
        self.block_first_verification = False
        self.release_first_verification = threading.Event()


class _Adapter:
    def __init__(self, recorder, **kwargs):
        self.recorder = recorder
        self.task_id = None
        self.image_id = "sha256:" + "0" * 64
        self.submitted = False

    def artifact_binding(self):
        return {"seed_workspace_sha256": "1" * 64,
                "verifier_sha256": "2" * 64,
                "image_sha256": self.image_id,
                "visible_check_sha256": "3" * 64}

    def reset(self, task, *, now):
        self.task_id = task["task_id"]
        with self.recorder.lock:
            self.recorder.started.append(self.task_id)
        if self.task_id == "pipeline-1":
            # With one actor worker, actor 1 can start while actor 0 verifies
            # only if ownership really moved into a separate verifier pool.
            started = self.recorder.verify_started.wait(timeout=2)
            self.recorder.actor_overlapped_verifier = started and not self.recorder.verify_finished.is_set()
        elif self.task_id != "pipeline-0" and self.recorder.block_first_verification:
            self.recorder.verify_started.wait(timeout=2)
        return {"status": "ready"}

    def step(self, action, *, now):
        self.submitted = True
        return {"observation": {"status": "submitted"}, "terminated": True}

    def verify(self, *, now):
        if self.task_id == "pipeline-0":
            self.recorder.verify_started.set()
            if self.recorder.block_first_verification:
                self.recorder.release_first_verification.wait(timeout=3)
            else:
                time.sleep(0.1)
            self.recorder.verify_finished.set()
        else:
            time.sleep(0.01)
        return {"status": "resolved", "reward": 1.0,
                "evidence": {"kind": "test-verifier"},
                "available_at": datetime.now(timezone.utc).isoformat()}

    def get_state(self):
        return {"submitted": self.submitted, "metrics": {"verify_seconds": 0.0}}

    def close(self):
        with self.recorder.lock:
            self.recorder.closed.append(self.task_id)


class DisaggregatedRolloutTests(unittest.TestCase):
    def test_actor_continues_while_prior_episode_verifies(self):
        recorder = _Recorder()
        with tempfile.TemporaryDirectory() as tmp:
            report = run_disaggregated_coding_batch(
                jobs=[_job(0), _job(1), _job(2)], output_root=Path(tmp) / "pipeline",
                actor_workers=1, verification_workers=1,
                actor_queue_capacity=1, verifier_queue_capacity=1,
                adapter_factory=lambda **kwargs: _Adapter(recorder, **kwargs))
            persisted = json.loads((Path(tmp) / "pipeline" / "batch_report.json").read_text())
            episode_reports = [json.loads((Path(item["output"]) / "report.json").read_text())
                               for item in report["jobs"]]
            trajectories = [(Path(item["output"]) / "training_trajectory.json").is_file()
                            for item in report["jobs"]]
        self.assertTrue(recorder.actor_overlapped_verifier)
        self.assertEqual(report, persisted)
        self.assertEqual(report["architecture"], "same_host_actor_verifier_pipeline")
        self.assertEqual(report["graded_count"], 3)
        self.assertEqual(report["error_count"], 0)
        self.assertEqual([item["index"] for item in report["jobs"]], [0, 1, 2])
        self.assertEqual([item["reward"] for item in report["jobs"]], [1.0] * 3)
        self.assertEqual([item["status"] for item in episode_reports], ["graded"] * 3)
        self.assertTrue(all(trajectories))
        self.assertEqual(set(recorder.closed), {"pipeline-0", "pipeline-1", "pipeline-2"})
        self.assertEqual(report["queue_metrics"]["actor_handoffs"], 3)
        self.assertLessEqual(report["queue_metrics"]["verifier_queue_high_water"], 1)
        self.assertTrue(all(item["timings"]["verification_seconds"] is not None
                            for item in report["jobs"]))

    def test_bad_job_isolated_and_no_private_reward_before_submit(self):
        recorder = _Recorder()
        bad = _job(1)
        bad["policy_id"] = ""
        unsubmitted = _job(2)
        unsubmitted["actions"] = []
        with tempfile.TemporaryDirectory() as tmp:
            report = run_disaggregated_coding_batch(
                jobs=[_job(0), bad, unsubmitted], output_root=Path(tmp) / "pipeline",
                actor_workers=2, verification_workers=1,
                adapter_factory=lambda **kwargs: _Adapter(recorder, **kwargs))
        self.assertEqual(report["error_count"], 1)
        self.assertEqual(report["jobs"][1]["execution_status"], "error")
        self.assertEqual(report["jobs"][2]["episode_status"], "active")
        self.assertIsNone(report["jobs"][2]["reward"])
        self.assertIsNone(report["jobs"][2]["timings"]["verification_seconds"])
        self.assertEqual(report["queue_metrics"]["actor_handoffs"], 1)

    def test_full_verifier_queue_applies_backpressure(self):
        recorder = _Recorder()
        recorder.block_first_verification = True
        result = {}
        with tempfile.TemporaryDirectory() as tmp:
            def run():
                try:
                    result["report"] = run_disaggregated_coding_batch(
                        jobs=[_job(index) for index in range(4)],
                        output_root=Path(tmp) / "pipeline",
                        actor_workers=2, verification_workers=1,
                        actor_queue_capacity=2, verifier_queue_capacity=1,
                        adapter_factory=lambda **kwargs: _Adapter(recorder, **kwargs))
                except Exception as exc:
                    result["error"] = exc

            thread = threading.Thread(target=run, daemon=True)
            thread.start()
            try:
                self.assertTrue(recorder.verify_started.wait(timeout=3))
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    with recorder.lock:
                        started_count = len(recorder.started)
                    if started_count >= 3:
                        break
                    time.sleep(0.005)
                self.assertGreaterEqual(started_count, 3)
                time.sleep(0.05)
                self.assertTrue(thread.is_alive())
            finally:
                recorder.release_first_verification.set()
                thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
            self.assertNotIn("error", result)
            report = result["report"]
        self.assertEqual(report["graded_count"], 4)
        self.assertEqual(report["error_count"], 0)
        self.assertLessEqual(report["queue_metrics"]["verifier_queue_high_water"], 1)
        self.assertGreater(report["queue_metrics"]["actor_handoff_backpressure_seconds"], 0.01)

    def test_worker_and_output_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pipeline"
            path.mkdir()
            (path / "existing").write_text("x", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "new or empty"):
                run_disaggregated_coding_batch(jobs=[_job(0)], output_root=path)
            with self.assertRaisesRegex(ValueError, "actor_workers"):
                run_disaggregated_coding_batch(jobs=[_job(0)], output_root=Path(tmp) / "new",
                                                actor_workers=True)
            with self.assertRaisesRegex(ValueError, "nonempty sequence"):
                run_disaggregated_coding_batch(jobs=[], output_root=Path(tmp) / "new")


if __name__ == "__main__":
    unittest.main()
