"""Real Docker stable-revision A/B for the opt-in trajectory dispatch fence.

Each arm runs the pinned public solved Boltons v2 repair and untouched
baseline through RealWorldEnv and the same 14 host-side cases. This tests
full graded batch overhead and exact action/reward/evidence parity, not model
inference, training, or a synthetic verifier wait.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from future_prediction_bench.coding_env import DockerCodingAdapter, _workspace_digest
from future_prediction_bench.realworld import validate_task
from future_prediction_bench.trajectory_control import (
    TrajectoryControlPlane, TrajectoryJob, coding_episode_factory,
)


VISIBLE = ("python3", "-B", "-c", "import boltons.strutils")
TASK_ID = "boltons-26-singularize-ss-v2"
SOURCE_SHA256 = "5566d6cfd5a1e873d8e8476496287a9f92979964611ad9a9cecb6b0ef29b1edd"
SEED_SHA256 = "4463d7c6acfced27ada226f4d26229fc3d78c8eec8fcde7734b8c73fd66a31b6"
POLICY_REVISION = "pinned-scripted-revision-v1"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _refreshed_task(source):
    task = copy.deepcopy(source)
    now = datetime.now(timezone.utc)
    task.update({"issued_at": (now - timedelta(minutes=1)).isoformat(),
                 "action_deadline": (now + timedelta(minutes=30)).isoformat(),
                 "outcome_not_before": (now - timedelta(minutes=1)).isoformat(),
                 "verify_after": (now - timedelta(minutes=1)).isoformat()})
    return task


def _inputs(task_dir, image, output):
    source = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    if (source.get("task_id") != TASK_ID or source.get("is_fixture") is not True
            or source.get("metadata", {}).get("source_sdist_sha256") != SOURCE_SHA256
            or _workspace_digest(task_dir / "seed") != SEED_SHA256):
        raise ValueError("Expected the pinned public Boltons v2 Docker fixture")
    verifier = json.loads((task_dir / "verifier" / "verify.json").read_text(
        encoding="utf-8"))
    if verifier.get("kind") != "command_cases_v1" or len(verifier.get("cases", [])) != 14:
        raise ValueError("Expected the pinned 14-case verifier")
    def actions(name):
        return [json.loads(line) for line in (task_dir / name).read_text(
            encoding="utf-8").splitlines() if line.strip()]
    baseline = actions("actions.baseline.jsonl")
    repair = actions("actions.solution.replace_text.jsonl")
    if (baseline != [{"action": "submit"}]
            or [item.get("action") for item in repair]
               != ["read_file", "replace_text", "run_visible_checks", "submit"]):
        raise ValueError("Pinned baseline or repair script differs")
    task = _refreshed_task(source)
    adapter = DockerCodingAdapter(
        seed_dir=task_dir / "seed", verifier_dir=task_dir / "verifier",
        image=image, output_root=output / "binding-probe",
        visible_check=VISIBLE, verifier_workers=1)
    try:
        binding = adapter.artifact_binding()
    finally:
        adapter.close()
    task.setdefault("metadata", {})["artifact_binding"] = binding
    frozen = validate_task(task)
    return task, frozen, binding, {"repair": repair, "baseline": baseline}


def _schedule(pairs):
    if type(pairs) is not int or not 2 <= pairs <= 5:
        raise ValueError("pairs must be in [2, 5]")
    for pair in range(1, pairs + 1):
        for arm in (("report_only", "fence") if pair % 2 else ("fence", "report_only")):
            yield pair, arm


def _batch(pair, arm, task, actions, task_dir, image, output):
    started = time.monotonic()
    run_dir = output / f"pair-{pair}-{arm}"
    factory = coding_episode_factory(
        task=task, seed_dir=task_dir / "seed",
        verifier_dir=task_dir / "verifier", image=image,
        output_root=run_dir, visible_check=VISIBLE, verifier_workers=1)
    jobs = []
    for branch in ("repair", "baseline"):
        selected = actions[branch]

        def generate(observation, history, revision, *, sequence=selected):
            turn = len(history) - 1
            return copy.deepcopy(sequence[turn]) if turn < len(sequence) else None

        jobs.append(TrajectoryJob(
            job_id=branch, policy_id="scripted-public-boltons-v2",
            open_episode=lambda name=branch: factory(0 if name == "repair" else 1),
            generate_action=generate))
    manager = TrajectoryControlPlane(
        jobs, revision_provider=lambda: POLICY_REVISION,
        stale_revision_policy=arm,
        actor_workers=1, environment_workers=1, reward_workers=1,
        actor_queue_capacity=1, environment_queue_capacity=1,
        reward_queue_capacity=1, max_in_flight=2)
    try:
        report = manager.run()
        exported = manager.text_trajectories()
    finally:
        manager.close()
    wall = time.monotonic() - started
    episodes = report["episodes"]
    if ([item["job_id"] for item in episodes] != ["repair", "baseline"]
            or [item["status"] for item in episodes] != ["graded", "graded"]
            or [item["reward"] for item in episodes] != [1.0, 0.0]
            or any(item["freshness_gate"] != "current_revision"
                   or item["stale_action_rejections"] != 0
                   or item["verification_evidence_sha256"] is None
                   for item in episodes)
            or set(exported) != {"repair", "baseline"}):
        raise RuntimeError("Docker stable-revision batch failed contract")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "trajectory_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    phases = [phase for episode in episodes for phase in episode["phase_timestamps"]]
    return {"pair": pair, "arm": arm, "wall_seconds": wall,
            "control_run_seconds": report["wall_seconds"],
            "setup_and_cleanup_seconds": wall - report["wall_seconds"],
            "graded_episodes_per_hour": 2 * 3600 / wall,
            "queue_wait_p95_seconds": report["queue_wait_p95_seconds"],
            "queue_wait_median_seconds": report["queue_wait_median_seconds"],
            "stage_work_sum_seconds": {stage: sum(item["work_seconds"] for item in phases
                                             if item["stage"] == stage)
                                       for stage in ("actor", "environment", "reward")},
            "status": [item["status"] for item in episodes],
            "reward": [item["reward"] for item in episodes],
            "evidence_sha256": [item["verification_evidence_sha256"]
                                for item in episodes],
            "action_sha256s": [[action["action_sha256"] for action in item["actions"]]
                               for item in episodes],
            "policy_revisions": [[action["policy_revision"] for action in item["actions"]]
                                 for item in episodes],
            "freshness": [item["freshness_gate"] for item in episodes],
            "stale_action_rejections": [item["stale_action_rejections"] for item in episodes]}


def _assert_parity(rows, pairs):
    if [(row["pair"], row["arm"]) for row in rows] != list(_schedule(pairs)):
        raise RuntimeError("incomplete_ab_schedule")
    first = rows[0]
    for row in rows:
        for key in ("status", "reward", "evidence_sha256", "action_sha256s",
                    "policy_revisions", "freshness", "stale_action_rejections"):
            if row[key] != first[key]:
                raise RuntimeError("stable_revision_parity_failed: " + key)


def benchmark(*, task_dir, image, output, pairs=3):
    schedule = list(_schedule(pairs))
    task_dir, output = Path(task_dir).resolve(), Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output must be new or empty")
    if output.is_relative_to(task_dir) or task_dir.is_relative_to(output):
        raise ValueError("Output must be separate from fixture")
    output.mkdir(parents=True, exist_ok=True)
    task, frozen, binding, actions = _inputs(task_dir, image, output)
    rows = []
    for pair, arm in schedule:
        row = _batch(pair, arm, task, actions, task_dir, image, output)
        rows.append(row)
        print(f"pair {pair} {arm}: {row['wall_seconds']:.3f}s / 2 graded", flush=True)
    _assert_parity(rows, pairs)
    medians = {arm: statistics.median(row["wall_seconds"] for row in rows
                                    if row["arm"] == arm)
               for arm in ("report_only", "fence")}
    report = {"kind": "real_docker_trajectory_revision_fence_ab_v1",
              "scope": "pinned public solved Boltons v2; scripted actions, no model or optimizer",
              "task_id": task["task_id"], "task_sha256": frozen["task_sha256"],
              "artifact_binding": binding,
              "source_sha256": {"trajectory_control.py": _sha(Path(
                  __file__).parents[2] / "future_prediction_bench" / "trajectory_control.py"),
                                "coding_env.py": _sha(Path(
                                    __file__).parents[2] / "future_prediction_bench" / "coding_env.py"),
                                "realworld.py": _sha(Path(
                                    __file__).parents[2] / "future_prediction_bench" / "realworld.py"),
                                "benchmark_revision_fence.py": _sha(__file__)},
              "fixture_sha256": {"task.json": _sha(task_dir / "task.json"),
                                  "verify.json": _sha(task_dir / "verifier" / "verify.json"),
                                  "actions.baseline.jsonl": _sha(task_dir / "actions.baseline.jsonl"),
                                  "actions.solution.replace_text.jsonl": _sha(
                                      task_dir / "actions.solution.replace_text.jsonl")},
              "pairs": pairs, "batch_size": 2, "graded_episodes": len(rows) * 2,
              "schedule": [list(item) for item in schedule], "runs": rows,
              "median_full_graded_batch_seconds": medians,
              "fence_over_report_only_ratio": medians["fence"] / medians["report_only"],
              "parity": {"statuses_rewards_evidence_actions_revisions_freshness_equal": True,
                         "all_batches_graded": True,
                         "stale_rejections_on_stable_revision": 0}}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n",
                                         encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=3)
    arguments = parser.parse_args()
    report = benchmark(task_dir=arguments.task_dir, image=arguments.image,
                       output=arguments.output, pairs=arguments.pairs)
    print(json.dumps(report["median_full_graded_batch_seconds"], indent=2))


if __name__ == "__main__":
    main()
