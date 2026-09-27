"""Controlled straggler benchmark for the same-host actor/verifier scheduler.

This synthetic *environment* fixture isolates verifier scheduling. Five
distinct task IDs receive the same fixed service times under FIFO and trusted
priority. The two conditions use the same one actor and one verifier worker.
It measures queue latency, not model inference, optimizer updates, or RL
training throughput. Repetitions of these five fixtures are correlated.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from future_prediction_bench.disaggregated_rollout import run_disaggregated_coding_batch


def _task(index):
    now = datetime.now(timezone.utc)
    return {
        "schema_version": "realworld-0.1", "task_id": f"scheduler-synthetic-{index}",
        "event_id": f"scheduler-synthetic-{index}",
        "cluster_id": f"scheduler-synthetic-{index}",
        "split": "train", "prompt": "Submit the synthetic coding fixture.",
        "issued_at": (now - timedelta(minutes=1)).isoformat(),
        "action_deadline": (now + timedelta(minutes=5)).isoformat(),
        "outcome_not_before": (now - timedelta(minutes=1)).isoformat(),
        "verify_after": (now - timedelta(minutes=1)).isoformat(),
        "tool_manifest": [{"name": "submit", "description": "Submit the workspace."}],
        "reward_contract": {"id": "synthetic-hidden-check",
                            "description": "Fixed controlled outcome.",
                            "min_reward": 0, "max_reward": 1},
        "budgets": {"max_actions": 2, "max_wall_seconds": 30},
        "is_fixture": True,
    }


class _Trace:
    def __init__(self, episode_count):
        self.lock = threading.Lock()
        self.first_started = threading.Event()
        self.all_submitted = threading.Event()
        self.episode_count = episode_count
        self.submitted = 0
        self.verify_order = []


class _Adapter:
    def __init__(self, trace, *, slow_seconds, fast_seconds, **kwargs):
        self.trace = trace
        self.slow_seconds = slow_seconds
        self.fast_seconds = fast_seconds
        self.index = None
        self.image_id = "sha256:" + "0" * 64

    def artifact_binding(self):
        return {"seed_workspace_sha256": "1" * 64,
                "verifier_sha256": "2" * 64,
                "image_sha256": self.image_id,
                "visible_check_sha256": "3" * 64}

    def reset(self, task, *, now):
        self.index = int(task["task_id"].rsplit("-", 1)[1])
        if self.index == 1 and not self.trace.first_started.wait(timeout=3):
            raise TimeoutError("First verifier did not start")
        return {"status": "ready"}

    def step(self, action, *, now):
        with self.trace.lock:
            self.trace.submitted += 1
            if self.trace.submitted == self.trace.episode_count:
                self.trace.all_submitted.set()
        return {"observation": {"status": "submitted"}, "terminated": True}

    def verify(self, *, now):
        with self.trace.lock:
            self.trace.verify_order.append(self.index)
        if self.index == 0:
            self.trace.first_started.set()
            if not self.trace.all_submitted.wait(timeout=3):
                raise TimeoutError("Actor did not hand off all episodes")
        time.sleep(self.slow_seconds if self.index in (0, 1) else self.fast_seconds)
        return {"status": "resolved", "reward": float(self.index % 2 == 0),
                "evidence": {"kind": "synthetic-controlled-verifier"},
                "available_at": datetime.now(timezone.utc).isoformat()}

    def get_state(self):
        return {"metrics": {"verify_seconds": 0.0}}

    def close(self):
        pass


def benchmark(*, output, repetitions=3, slow_seconds=0.30, fast_seconds=0.01):
    if type(repetitions) is not int or not 1 <= repetitions <= 20:
        raise ValueError("repetitions must be an integer in [1, 20]")
    for name, value in (("slow_seconds", slow_seconds), ("fast_seconds", fast_seconds)):
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not 0 < value <= 10:
            raise ValueError(f"{name} must be in (0, 10]")
    if slow_seconds <= fast_seconds:
        raise ValueError("slow_seconds must exceed fast_seconds")
    output = Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    jobs = [{"task": _task(index), "seed_dir": "/synthetic-seed",
             "verifier_dir": "/synthetic-verifier", "image": "synthetic-image",
             "actions": [{"action": "submit"}], "policy_id": "synthetic-fixed-policy"}
            for index in range(5)]
    expected_rewards = [1.0, 0.0, 1.0, 0.0, 1.0]
    runs = []
    for repetition in range(repetitions):
        order = ("fifo", "priority") if repetition % 2 == 0 else ("priority", "fifo")
        for condition in order:
            condition_jobs = copy.deepcopy(jobs)
            if condition == "priority":
                condition_jobs[1]["verifier_priority"] = 9
                for index in (2, 3, 4):
                    condition_jobs[index]["verifier_priority"] = 0
            trace = _Trace(len(jobs))
            report = run_disaggregated_coding_batch(
                jobs=condition_jobs,
                output_root=output / f"rep-{repetition}-{condition}",
                actor_workers=1, verification_workers=1,
                actor_queue_capacity=5, verifier_queue_capacity=5,
                adapter_factory=lambda **kwargs: _Adapter(
                    trace, slow_seconds=slow_seconds,
                    fast_seconds=fast_seconds, **kwargs))
            rewards = [item["reward"] for item in report["jobs"]]
            expected_order = [0, 1, 2, 3, 4] if condition == "fifo" else [0, 2, 3, 4, 1]
            if report["error_count"] or rewards != expected_rewards or trace.verify_order != expected_order:
                raise RuntimeError(f"Unexpected {condition} result: errors={report['error_count']}, "
                                   f"rewards={rewards}, order={trace.verify_order}")
            times = {str(item["index"]): item["timings"] for item in report["jobs"]}
            fast_completion = [times[str(index)]["completion_offset_seconds"] for index in (2, 3, 4)]
            runs.append({"repetition": repetition, "condition": condition,
                         "verification_order": trace.verify_order,
                         "rewards": rewards, "graded_count": report["graded_count"],
                         "wall_seconds": report["wall_seconds"],
                         "median_fast_completion_seconds": statistics.median(fast_completion),
                         "per_job_stage_seconds": times})
            print(f"{condition} rep={repetition} wall={report['wall_seconds']:.3f}s "
                  f"fast-median={statistics.median(fast_completion):.3f}s", flush=True)
    medians = {condition: {
        "median_fast_completion_seconds": statistics.median(
            run["median_fast_completion_seconds"] for run in runs if run["condition"] == condition),
        "median_batch_wall_seconds": statistics.median(
            run["wall_seconds"] for run in runs if run["condition"] == condition)}
        for condition in ("fifo", "priority")}
    result = {"scope": "synthetic same-host verifier scheduling; five distinct fixed fixtures",
              "measures_model_inference": False, "measures_rl_training": False,
              "repetitions_are_independent_outcomes": False,
              "worker_budget": {"actor_workers": 1, "verification_workers": 1,
                                "case_workers_per_episode": 1},
              "episode_count": 5, "repetitions": repetitions,
              "service_seconds": {"slow": slow_seconds, "fast": fast_seconds},
              "expected_rewards": expected_rewards, "runs": runs,
              "medians": medians,
              "fast_job_completion_latency_ratio": (
                  medians["fifo"]["median_fast_completion_seconds"] /
                  medians["priority"]["median_fast_completion_seconds"]),
              "batch_throughput_ratio": (
                  medians["fifo"]["median_batch_wall_seconds"] /
                  medians["priority"]["median_batch_wall_seconds"])}
    (output / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--slow-seconds", type=float, default=0.30)
    parser.add_argument("--fast-seconds", type=float, default=0.01)
    arguments = parser.parse_args()
    print(json.dumps(benchmark(output=arguments.output, repetitions=arguments.repetitions,
                               slow_seconds=arguments.slow_seconds,
                               fast_seconds=arguments.fast_seconds)["medians"], indent=2))
