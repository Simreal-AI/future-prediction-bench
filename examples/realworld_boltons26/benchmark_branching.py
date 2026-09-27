"""Measure cold replay against shared-prefix filesystem branches, offline.

This is an environment-throughput integration benchmark on one public solved
fixture. It does not measure model inference, optimizer throughput, or RL
sample efficiency. The generated task must still be within its action window.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from future_prediction_bench.branch_rollouts import run_coding_branches
from future_prediction_bench.realworld_demo import load_coding_actions, run_coding_task
from future_prediction_bench.rollout_batch import run_coding_batch


def benchmark(*, task_dir, image, output, repetitions=2, branch_count=4):
    if not 1 <= repetitions <= 20 or not 2 <= branch_count <= 8:
        raise ValueError("repetitions must be 1..20 and branch_count 2..8")
    task_dir, output = Path(task_dir).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    task = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    solution = load_coding_actions(task_dir / "actions.solution.jsonl")
    prefix = [solution[0], {"action": "run_visible_checks"}]
    suffixes = [solution[1:] if index % 2 == 0 else [{"action": "submit"}]
                for index in range(branch_count)]
    expected = [1.0 if index % 2 == 0 else 0.0 for index in range(branch_count)]
    visible = ("python3", "-B", "-c", "import boltons.strutils")
    runs = []
    # Alternate order to reduce a warm-cache or host-load ordering artifact.
    for repetition in range(repetitions):
        conditions = ["cold_serial", "shared_serial", "cold_parallel", "shared_parallel"]
        if repetition % 2:
            conditions.reverse()
        for condition in conditions:
            location = output / f"rep-{repetition}-{condition}"
            started = time.monotonic()
            if condition == "cold_serial":
                rewards = []
                for index, suffix in enumerate(suffixes):
                    episode = run_coding_task(
                        task=task, seed_dir=task_dir / "seed", verifier_dir=task_dir / "verifier",
                        image=image, actions=[*prefix, *suffix],
                        output=location / f"episode-{index}", visible_check=visible,
                        policy_id="scripted-branch-benchmark", verifier_workers=1)
                    rewards.append(episode["reward"])
            elif condition == "cold_parallel":
                report = run_coding_batch(
                    jobs=[{"task": task, "seed_dir": task_dir / "seed",
                           "verifier_dir": task_dir / "verifier", "image": image,
                           "actions": [*prefix, *suffix],
                           "policy_id": "scripted-branch-benchmark",
                           "visible_check": visible, "verifier_workers": 1}
                          for suffix in suffixes],
                    output_root=location, max_workers=min(4, branch_count))
                rewards = [item["episode_result"]["reward"]
                           if item["episode_result"] is not None else None
                           for item in report["jobs"]]
            else:
                report = run_coding_branches(
                    task=task, seed_dir=task_dir / "seed", verifier_dir=task_dir / "verifier",
                    image=image, prefix_actions=prefix,
                    branches=[{"branch_id": f"suffix-{index}", "actions": suffix}
                              for index, suffix in enumerate(suffixes)],
                    output_root=location, policy_id="scripted-branch-benchmark",
                    visible_check=visible, verifier_workers=1,
                    max_workers=1 if condition == "shared_serial" else min(4, branch_count))
                rewards = [item["reward"] for item in report["results"]]
            elapsed = time.monotonic() - started
            if rewards != expected:
                raise RuntimeError(f"Unexpected rewards for {condition}: {rewards}")
            runs.append({"repetition": repetition, "condition": condition,
                         "wall_seconds": elapsed, "rewards": rewards,
                         "output": str(location)})
            print(f"{condition} rep={repetition} wall={elapsed:.3f}s", flush=True)
    medians = {condition: statistics.median(item["wall_seconds"] for item in runs
                                             if item["condition"] == condition)
               for condition in ("cold_serial", "shared_serial", "cold_parallel", "shared_parallel")}
    result = {"scope": "one public solved Boltons repository fixture on this host",
              "branch_count": branch_count, "repetitions": repetitions,
              "prefix_actions": [item["action"] for item in prefix],
              "verifier_workers_per_episode": 1,
              "shared_prefix_process_state": False,
              "measures_rl_training": False,
              "runs": runs, "median_wall_seconds": medians}
    (output / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--branch-count", type=int, default=4)
    arguments = parser.parse_args()
    print(json.dumps(benchmark(task_dir=arguments.task_dir, image=arguments.image,
                               output=arguments.output, repetitions=arguments.repetitions,
                               branch_count=arguments.branch_count)["median_wall_seconds"], indent=2))
