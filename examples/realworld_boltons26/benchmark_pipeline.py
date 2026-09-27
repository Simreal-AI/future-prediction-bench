"""Compare serial episodes with a bounded actor/verifier pipeline on Boltons.

This is a fixed, public repair fixture with scripted actions. It measures
graded environment episodes, not model inference or RL weight updates.
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from future_prediction_bench.disaggregated_rollout import run_disaggregated_coding_batch
from future_prediction_bench.realworld_demo import load_coding_actions
from future_prediction_bench.rollout_batch import run_coding_batch


def benchmark(*, task_dir, image, output, repetitions=2, episode_count=3):
    if type(repetitions) is not int or not 1 <= repetitions <= 10:
        raise ValueError("repetitions must be an integer in [1, 10]")
    if type(episode_count) is not int or not 2 <= episode_count <= 8:
        raise ValueError("episode_count must be an integer in [2, 8]")
    task_dir, output = Path(task_dir).resolve(), Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output directory must be new or empty")
    task = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    deadline = datetime.fromisoformat(task["action_deadline"].replace("Z", "+00:00"))
    if deadline.astimezone(timezone.utc) <= datetime.now(timezone.utc):
        raise ValueError("Task action deadline has passed; generate a fresh pinned task")
    baseline = load_coding_actions(task_dir / "actions.baseline.jsonl")
    solution = load_coding_actions(task_dir / "actions.solution.jsonl")
    visible = ("python3", "-B", "-c", "import boltons.strutils")
    jobs = [{"task": task, "seed_dir": task_dir / "seed",
             "verifier_dir": task_dir / "verifier", "image": image,
             "actions": solution if index % 2 == 0 else baseline,
             "policy_id": "scripted-public-boltons-fixture",
             "visible_check": visible, "verifier_workers": 1}
            for index in range(episode_count)]
    expected = [1.0 if index % 2 == 0 else 0.0 for index in range(episode_count)]
    output.mkdir(parents=True, exist_ok=True)
    runs = []
    for repetition in range(repetitions):
        order = ("serial", "pipeline") if repetition % 2 == 0 else ("pipeline", "serial")
        for condition in order:
            destination = output / f"rep-{repetition}-{condition}"
            if condition == "serial":
                report = run_coding_batch(jobs=jobs, output_root=destination, max_workers=1)
                rewards = [item["episode_result"]["reward"]
                           if item["episode_result"] else None for item in report["jobs"]]
            else:
                report = run_disaggregated_coding_batch(
                    jobs=jobs, output_root=destination, actor_workers=1,
                    verification_workers=1, actor_queue_capacity=1,
                    verifier_queue_capacity=1)
                rewards = [item["reward"] for item in report["jobs"]]
            if report["error_count"] or rewards != expected:
                raise RuntimeError(f"Unexpected {condition} results: errors={report['error_count']}, rewards={rewards}")
            runs.append({"repetition": repetition, "condition": condition,
                         "wall_seconds": report["wall_seconds"],
                         "rewards": rewards, "output": str(destination)})
            print(f"{condition} rep={repetition} wall={report['wall_seconds']:.3f}s", flush=True)
    medians = {condition: statistics.median(item["wall_seconds"] for item in runs
                                             if item["condition"] == condition)
               for condition in ("serial", "pipeline")}
    result = {"scope": "same-host public Boltons 26.0.0 solved fixture",
              "measures_model_training": False,
              "task_dir": str(task_dir), "episode_count": episode_count,
              "repetitions": repetitions, "case_workers_per_episode": 1,
              "actor_workers": 1, "verification_workers": 1,
              "expected_rewards": expected, "runs": runs,
              "median_wall_seconds": medians,
              "median_graded_episode_throughput_ratio": medians["serial"] / medians["pipeline"]}
    (output / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--episode-count", type=int, default=3)
    arguments = parser.parse_args()
    print(json.dumps(benchmark(task_dir=arguments.task_dir, image=arguments.image,
                               output=arguments.output, repetitions=arguments.repetitions,
                               episode_count=arguments.episode_count)["median_wall_seconds"], indent=2))
