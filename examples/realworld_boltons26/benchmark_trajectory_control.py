"""Real Docker A/B for bounded, multi-turn actor/environment/reward scheduling.

Both arms use the same RealWorldEnv, pinned Boltons source, local Docker image,
scripted action sequences, 14-case host verifier, and one worker per stage.
Only in-flight admission changes. A trusted verifier delay is injected into
the first episode of each batch at 0, 2, and 5 seconds. This measures valid
graded environment episodes, not model inference or RL training.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from future_prediction_bench.realworld_demo import load_coding_actions
from future_prediction_bench.trajectory_control import (
    TrajectoryControlPlane, TrajectoryJob, coding_episode_factory,
)


def benchmark(*, task_dir, image, output, repetitions=2, episode_count=4,
              reward_workers=1, pipeline_max_in_flight=None):
    if type(repetitions) is not int or not 1 <= repetitions <= 10:
        raise ValueError("repetitions must be an integer in [1, 10]")
    if type(episode_count) is not int or not 2 <= episode_count <= 16:
        raise ValueError("episode_count must be an integer in [2, 16]")
    if type(reward_workers) is not int or not 1 <= reward_workers <= 4:
        raise ValueError("reward_workers must be an integer in [1, 4]")
    if pipeline_max_in_flight is None:
        pipeline_max_in_flight = episode_count
    if type(pipeline_max_in_flight) is not int or not 2 <= pipeline_max_in_flight <= episode_count:
        raise ValueError("pipeline_max_in_flight must be in [2, episode_count]")
    task_dir, output = Path(task_dir).resolve(), Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output directory must be new or empty")
    task = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    deadline = datetime.fromisoformat(task["action_deadline"].replace("Z", "+00:00"))
    if deadline.astimezone(timezone.utc) <= datetime.now(timezone.utc):
        raise ValueError("Task action deadline has passed; generate a fresh pinned task")
    baseline = load_coding_actions(task_dir / "actions.baseline.jsonl")
    solution = load_coding_actions(task_dir / "actions.solution.jsonl")
    expected = [1.0 if index % 2 == 0 else 0.0 for index in range(episode_count)]
    output.mkdir(parents=True, exist_ok=True)
    runs = []
    pairs = []
    visible = ("python3", "-B", "-c", "import boltons.strutils")
    for delay in (0, 2, 5):
        for repetition in range(repetitions):
            by_condition = {}
            order = ("serial", "pipeline") if repetition % 2 == 0 else ("pipeline", "serial")
            for condition in order:
                run_dir = output / f"delay-{delay}-rep-{repetition}-{condition}"
                factory = coding_episode_factory(
                    task=task, seed_dir=task_dir / "seed",
                    verifier_dir=task_dir / "verifier", image=image,
                    output_root=run_dir, visible_check=visible,
                    verifier_workers=1)
                jobs = []
                for index in range(episode_count):
                    actions = solution if index % 2 == 0 else baseline

                    def action_fn(observation, history, revision, *, selected=actions):
                        turn = len(history) - 1
                        return selected[turn] if turn < len(selected) else None

                    jobs.append(TrajectoryJob(
                        job_id=f"episode-{index}",
                        policy_id="scripted-public-boltons-fixture",
                        open_episode=lambda selected=index: factory(selected),
                        generate_action=action_fn))
                manager = TrajectoryControlPlane(
                    jobs, revision_provider=lambda: "scripted-revision-v1",
                    actor_workers=1, environment_workers=1, reward_workers=reward_workers,
                    actor_queue_capacity=1, environment_queue_capacity=1,
                    reward_queue_capacity=1,
                    max_in_flight=1 if condition == "serial" else pipeline_max_in_flight,
                    before_verify=(lambda job_id, seconds=delay: time.sleep(seconds)
                                   if job_id == "episode-0" else None))
                load_before = os.getloadavg()
                try:
                    report = manager.run()
                finally:
                    manager.close()
                load_after = os.getloadavg()
                rewards = [item["reward"] for item in report["episodes"]]
                statuses = [item["status"] for item in report["episodes"]]
                if (rewards != expected or statuses != ["graded"] * episode_count
                        or report["infrastructure_error_count"]
                        or report["pending_count"]):
                    raise RuntimeError(f"Unexpected {condition} result at delay {delay}: "
                                       f"{statuses}, {rewards}")
                item = {"delay_seconds": delay, "repetition": repetition,
                        "condition": condition, "wall_seconds": report["wall_seconds"],
                        "host_loadavg_before": load_before,
                        "host_loadavg_after": load_after,
                        "valid_graded_episodes_per_hour": report["valid_graded_episodes_per_hour"],
                        "queue_wait_p95_seconds": report["queue_wait_p95_seconds"],
                        "queue_high_water": report["queue_high_water"],
                        "statuses": statuses, "rewards": rewards,
                        "evidence_sha256": [episode["verification_evidence_sha256"]
                                            for episode in report["episodes"]],
                        "per_episode_completion_seconds": [episode["completion_offset_seconds"]
                                                           for episode in report["episodes"]]}
                (run_dir / "trajectory_report.json").parent.mkdir(parents=True, exist_ok=True)
                (run_dir / "trajectory_report.json").write_text(
                    json.dumps(report, indent=2) + "\n", encoding="utf-8")
                runs.append(item)
                by_condition[condition] = item
                print(f"delay={delay}s rep={repetition} {condition} "
                      f"wall={item['wall_seconds']:.3f}s", flush=True)
            if (by_condition["serial"]["evidence_sha256"]
                    != by_condition["pipeline"]["evidence_sha256"]):
                raise RuntimeError("Trusted verifier evidence differs between A/B arms")
            pairs.append({"delay_seconds": delay, "repetition": repetition,
                          "serial_wall_seconds": by_condition["serial"]["wall_seconds"],
                          "pipeline_wall_seconds": by_condition["pipeline"]["wall_seconds"],
                          "evidence_equal": True, "reward_equal": True})
    summaries = {}
    for delay in (0, 2, 5):
        arms = {condition: [run for run in runs if run["delay_seconds"] == delay
                            and run["condition"] == condition]
                for condition in ("serial", "pipeline")}
        medians = {condition: statistics.median(item["wall_seconds"] for item in arm)
                   for condition, arm in arms.items()}
        summaries[str(delay)] = {
            "median_wall_seconds": medians,
            "median_valid_graded_episodes_per_hour": {
                condition: episode_count * 3600 / seconds for condition, seconds in medians.items()},
            "throughput_ratio_pipeline_over_serial": medians["serial"] / medians["pipeline"],
            "max_queue_wait_p95_seconds": {
                condition: {stage: max(item["queue_wait_p95_seconds"][stage] for item in arm)
                            for stage in ("actor", "environment", "reward")}
                for condition, arm in arms.items()}}
    result = {"schema_version": "rollart-control-ab-0.1",
              "scope": "same-host pinned public Boltons 26.0.0 solved fixture",
              "measures_model_inference": False, "measures_rl_training": False,
              "task_id": task["task_id"], "task_dir": str(task_dir),
              "image": image, "host": platform.platform(),
              "episode_count": episode_count, "repetitions": repetitions,
              "worker_counts": {"actor": 1, "environment": 1, "reward": reward_workers},
              "queue_capacities": {"actor": 1, "environment": 1, "reward": 1},
              "serial_max_in_flight": 1,
              "pipeline_max_in_flight": pipeline_max_in_flight,
              "hidden_cases_per_episode": 14,
              "expected_rewards": expected,
              "runs": runs, "pairs": pairs, "summary_by_delay_seconds": summaries}
    (output / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--episode-count", type=int, default=4)
    parser.add_argument("--reward-workers", type=int, default=1)
    parser.add_argument("--pipeline-max-in-flight", type=int)
    arguments = parser.parse_args()
    summary = benchmark(task_dir=arguments.task_dir, image=arguments.image,
                        output=arguments.output, repetitions=arguments.repetitions,
                        episode_count=arguments.episode_count,
                        reward_workers=arguments.reward_workers,
                        pipeline_max_in_flight=arguments.pipeline_max_in_flight)
    print(json.dumps(summary["summary_by_delay_seconds"], indent=2))
