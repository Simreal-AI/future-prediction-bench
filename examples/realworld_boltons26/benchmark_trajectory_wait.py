"""Compare reward-worker waiting with bounded scheduler-side readiness timers.

The same real Docker coding episodes, verifier, actions, stage workers, and
in-flight bound run in both arms. Only placement of a task-authored short
verification-readiness delay differs. This is an environment throughput test,
not policy inference or model training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _identity(episode):
    return {"status": episode["status"], "reward": episode["reward"],
            "action_sha256": [action["action_sha256"] for action in episode["actions"]],
            "policy_revisions": [action["policy_revision"] for action in episode["actions"]],
            "freshness_gate": episode["freshness_gate"],
            "verification_status": episode["verification_status"],
            "verification_evidence_sha256": episode["verification_evidence_sha256"]}


def _stage_diagnostics(report):
    phases = [phase for episode in report["episodes"] for phase in episode["phase_timestamps"]]
    first = report["episodes"][0]["phase_timestamps"]
    reward = next(phase for phase in first if phase["stage"] == "reward")
    submit = [phase for phase in first if phase["stage"] == "environment"][-1]
    return {"work_seconds_by_stage": {
                stage: sum(phase["work_seconds"] for phase in phases if phase["stage"] == stage)
                for stage in ("actor", "environment", "reward")},
            "queue_wait_p95_seconds_by_stage": report["queue_wait_p95_seconds"],
            "first_episode_submit_to_reward_start_seconds": (
                reward["started_offset_seconds"] - submit["finished_offset_seconds"]),
            "first_episode_reward_work_seconds": reward["work_seconds"],
            "first_episode_reward_queue_seconds": reward["queue_seconds"]}


def benchmark(*, task_dir, image, output, public_output=None, repetitions=3,
              episode_count=4, delays=(0, 2, 5)):
    if type(repetitions) is not int or not 1 <= repetitions <= 10:
        raise ValueError("repetitions must be an integer in [1, 10]")
    if type(episode_count) is not int or not 2 <= episode_count <= 16:
        raise ValueError("episode_count must be an integer in [2, 16]")
    if (not delays or any(isinstance(delay, bool) or not isinstance(delay, (int, float))
                          or not math.isfinite(delay) or not 0 <= delay <= 30
                          for delay in delays)):
        raise ValueError("delays must be finite seconds in [0, 30]")
    task_dir, output = Path(task_dir).resolve(), Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output directory must be new or empty")
    task = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    deadline = datetime.fromisoformat(task["action_deadline"].replace("Z", "+00:00"))
    if deadline.astimezone(timezone.utc) <= datetime.now(timezone.utc):
        raise ValueError("Task action deadline has passed; generate a fresh pinned task")
    solution = load_coding_actions(task_dir / "actions.solution.jsonl")
    baseline = load_coding_actions(task_dir / "actions.baseline.jsonl")
    expected = [float(index % 2 == 0) for index in range(episode_count)]
    output.mkdir(parents=True)
    runs, pairs = [], []
    visible = ("python3", "-B", "-c", "import boltons.strutils")
    for delay in delays:
        for repetition in range(repetitions):
            by_arm = {}
            order = ("worker", "scheduler") if repetition % 2 == 0 else ("scheduler", "worker")
            for placement in order:
                run_dir = output / f"delay-{delay:g}-rep-{repetition}-{placement}"
                batch_start = time.monotonic()
                factory = coding_episode_factory(
                    task=task, seed_dir=task_dir / "seed",
                    verifier_dir=task_dir / "verifier", image=image,
                    output_root=run_dir, visible_check=visible,
                    verifier_workers=1)
                jobs = []
                for index in range(episode_count):
                    actions = solution if index % 2 == 0 else baseline

                    def generate(observation, history, revision, *, selected=actions):
                        turn = len(history) - 1
                        return selected[turn] if turn < len(selected) else None

                    jobs.append(TrajectoryJob(
                        job_id=f"episode-{index}",
                        policy_id="scripted-public-boltons-fixture",
                        open_episode=lambda selected=index: factory(selected),
                        generate_action=generate))
                manager = TrajectoryControlPlane(
                    jobs, revision_provider=lambda: "scripted-revision-v1",
                    actor_workers=1, environment_workers=1, reward_workers=1,
                    actor_queue_capacity=1, environment_queue_capacity=1,
                    reward_queue_capacity=1, max_in_flight=episode_count,
                    verification_delay_seconds=(
                        lambda job_id, seconds=delay: seconds if job_id == "episode-0" else 0),
                    verification_wait_placement=placement)
                load_before = os.getloadavg()
                try:
                    report = manager.run()
                finally:
                    manager.close()
                full_wall = time.monotonic() - batch_start
                load_after = os.getloadavg()
                rewards = [episode["reward"] for episode in report["episodes"]]
                statuses = [episode["status"] for episode in report["episodes"]]
                if (rewards != expected or statuses != ["graded"] * episode_count
                        or report["infrastructure_error_count"] or report["pending_count"]
                        or report["fresh_graded_count"] != episode_count):
                    raise RuntimeError(f"Unexpected {placement} grading at delay={delay}: "
                                       f"{statuses}, {rewards}")
                item = {"delay_seconds": delay, "repetition": repetition,
                        "placement": placement,
                        "full_batch_wall_seconds": full_wall,
                        "episode_run_wall_seconds": report["wall_seconds"],
                        "valid_graded_episodes_per_hour": episode_count * 3600 / full_wall,
                        "queue_wait_p95_seconds": report["queue_wait_p95_seconds"],
                        "host_loadavg_before": load_before,
                        "host_loadavg_after": load_after,
                        "verification_wait_seconds": [episode["verification_wait_seconds"]
                                                      for episode in report["episodes"]],
                        "stage_diagnostics": _stage_diagnostics(report),
                        "identities": [_identity(episode) for episode in report["episodes"]]}
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "trajectory_report.json").write_text(
                    json.dumps(report, indent=2) + "\n", encoding="utf-8")
                runs.append(item)
                by_arm[placement] = item
                print(f"delay={delay:g}s rep={repetition} {placement} "
                      f"full_wall={full_wall:.3f}s", flush=True)
            if by_arm["worker"]["identities"] != by_arm["scheduler"]["identities"]:
                raise RuntimeError("Action, revision, reward, or verifier evidence differs between arms")
            pairs.append({"delay_seconds": delay, "repetition": repetition,
                          "worker_full_batch_seconds": by_arm["worker"]["full_batch_wall_seconds"],
                          "scheduler_full_batch_seconds": by_arm["scheduler"]["full_batch_wall_seconds"],
                          "exact_action_revision_reward_evidence_parity": True})
    summaries = {}
    for delay in delays:
        matched = [pair for pair in pairs if pair["delay_seconds"] == delay]
        worker = statistics.median(pair["worker_full_batch_seconds"] for pair in matched)
        scheduler = statistics.median(pair["scheduler_full_batch_seconds"] for pair in matched)
        summaries[str(delay)] = {"worker_median_full_batch_seconds": worker,
                                 "scheduler_median_full_batch_seconds": scheduler,
                                 "median_throughput_ratio": worker / scheduler,
                                 "paired_deltas_seconds": [
                                     pair["worker_full_batch_seconds"]
                                     - pair["scheduler_full_batch_seconds"] for pair in matched]}
    controller = Path(__file__).resolve().parents[2] / "future_prediction_bench" / "trajectory_control.py"
    raw = {"schema_version": "rollart-readiness-ab-0.1",
           "task_dir": str(task_dir), "output": str(output),
           "image": image, "host": platform.platform(),
           "task_id": task["task_id"], "source_sdist_sha256": task["metadata"]["source_sdist_sha256"],
           "controller_sha256": _sha256(controller),
           "episode_count": episode_count, "repetitions": repetitions,
           "delays_seconds": list(delays),
           "fixed_resources": {"actor_workers": 1, "environment_workers": 1,
                               "reward_workers": 1, "verifier_case_workers_per_episode": 1,
                               "max_in_flight": episode_count,
                               "queue_capacities": {"actor": 1, "environment": 1, "reward": 1}},
           "measures_model_inference": False, "measures_rl_training": False,
           "runs": runs, "pairs": pairs, "summary_by_delay_seconds": summaries}
    (output / "benchmark.json").write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    if public_output is not None:
        public = {key: value for key, value in raw.items()
                  if key not in {"task_dir", "output", "runs"}}
        public["run_diagnostics"] = [
            {"delay_seconds": run["delay_seconds"], "repetition": run["repetition"],
             "placement": run["placement"],
             "stage_diagnostics": run["stage_diagnostics"]} for run in runs]
        public["measurement_scope"] = (
            "Full real Docker graded batches including factory setup, all coding actions, "
            "14-case host verification per episode, and cleanup. The 2/5-second "
            "first-episode readiness waits are injected; they are not measured "
            "model or verifier latency. This does not measure RL training.")
        public["limitations"] = [
            "One public solved repository fixture and fixed scripted actions; repetitions share host and Docker cache.",
            "Readiness delays are artificial and only test control-plane scheduling under delayed feedback.",
            "The held lease remains in the in-flight bound; a full bound can still prevent admission of new episodes.",
            "No token-level behavior likelihoods, optimizer step, GPU, or model-quality measurement."]
        path = Path(public_output).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(public, indent=2) + "\n", encoding="utf-8")
    return raw


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--public-output")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--episode-count", type=int, default=4)
    parser.add_argument("--delays", type=float, nargs="+", default=[0, 2, 5])
    args = parser.parse_args()
    result = benchmark(task_dir=args.task_dir, image=args.image,
                       output=args.output, public_output=args.public_output,
                       repetitions=args.repetitions,
                       episode_count=args.episode_count, delays=tuple(args.delays))
    print(json.dumps(result["summary_by_delay_seconds"], indent=2))
