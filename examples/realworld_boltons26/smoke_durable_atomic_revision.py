"""One real-Docker, separate-process smoke of the opt-in SQLite revision fence.

This uses the pinned public Boltons v2 repair and its 14 host-side cases. It
tests reward publication at a stable queue-local revision, not a speedup,
model inference, or an optimizer. Race/ABA cases are covered by focused tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import platform
import time
from pathlib import Path

from future_prediction_bench import coding_env, durable_rollout, realworld
from future_prediction_bench.coding_env import DockerCodingAdapter, _workspace_digest
from future_prediction_bench.durable_rollout import DurableDockerRolloutQueue
from future_prediction_bench.realworld import RealWorldEnv
from .benchmark_revision_fence import VISIBLE, _inputs


POLICY = "scripted-public-boltons-v2"
REVISION = "pinned-scripted-weight-revision-v1"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n",
                          encoding="utf-8")


def _worker(queue_path, outcome_path):
    queue = DurableDockerRolloutQueue(queue_path)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        result = queue.process_one(
            "atomic-fence-smoke-worker", revision_provider=lambda: REVISION,
            lease_seconds=10, atomic_revision_fence=True)
        if result is not None:
            _write(outcome_path, result)
            return
        time.sleep(0.05)
    raise RuntimeError("Submitted job was not claimable")


def smoke(*, task_dir, image, output):
    task_dir, output = Path(task_dir).resolve(), Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output must be new or empty")
    if output.is_relative_to(task_dir) or task_dir.is_relative_to(output):
        raise ValueError("Output must be separate from fixture")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    task, frozen, binding, actions = _inputs(task_dir, image, output)
    queue_path = output / "queue" / "work.sqlite"
    queue = DurableDockerRolloutQueue(queue_path)
    revision = queue.set_current_revision(POLICY, REVISION)

    actor_started = time.monotonic()
    adapter = DockerCodingAdapter(
        seed_dir=task_dir / "seed", verifier_dir=task_dir / "verifier",
        image=image, output_root=output / "actor" / "instances",
        visible_check=VISIBLE, verifier_workers=1)
    try:
        env = RealWorldEnv(task, adapter)
        env.reset(POLICY)
        for action in actions["repair"]:
            env.step(action)
        if env.status != "pending":
            raise RuntimeError("Scripted actor did not submit")
        receipt = queue.enqueue_submitted(
            "repair", env, policy_revision=REVISION,
            require_queue_revision=True)
        actor_state = env.get_state()
        if receipt["queue_revision_generation"] != revision["generation"]:
            raise RuntimeError("Enqueue revision generation differs")
        submitted_sha = _workspace_digest(adapter.submitted_workspace)
    finally:
        adapter.close()
    actor_seconds = time.monotonic() - actor_started

    worker_started = time.monotonic()
    outcome_path = output / "worker_outcome.json"
    worker = mp.get_context("spawn").Process(
        target=_worker, args=(str(queue_path), str(outcome_path)))
    worker.start()
    worker.join(timeout=180)
    if worker.is_alive():
        worker.terminate()
        worker.join(timeout=10)
        raise RuntimeError("Verifier worker timed out")
    if worker.exitcode != 0 or not outcome_path.is_file():
        raise RuntimeError(f"Verifier worker failed with exit code {worker.exitcode}")
    worker_seconds = time.monotonic() - worker_started
    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    row = queue.get("repair")
    if row is None:
        raise RuntimeError("Durable queue lost the submitted job")
    result = row["result"]
    evidence = result["evidence"] if result else None
    cases = evidence["case_results"] if evidence else None
    if (outcome["status"] != "graded" or outcome["reward"] != 1.0
            or outcome["attempt"] != 1 or row["state"] != "graded"
            or row["attempt"] != 1 or row["reward"] != 1.0
            or result["queue_revision_fence"] != revision
            or row["queue_revision_fence_current"] is not True
            or row["queue_revision_generation_bound"] != revision["generation"]
            or result["snapshot_sha256"] != submitted_sha
            or evidence["workspace_sha256"] != submitted_sha
            or evidence["verifier_sha256"] != binding["verifier_sha256"]
            or evidence["image_sha256"] != binding["image_sha256"]
            or not isinstance(cases, list) or len(cases) != 14
            or not all(case["passed"] for case in cases)
            or result["final_state"]["status"] != "graded"
            or result["final_state"]["metrics"]["verifications_used"] != 1
            or actor_state["metrics"]["actions_used"] != 4
            or result["trainer_ready"] is not False):
        raise RuntimeError("Real-Docker atomic revision fence smoke failed")

    report = {
        "kind": "real_docker_durable_atomic_revision_smoke_v1",
        "scope": "One pinned public solved Boltons v2 episode; scripted actions; stable revision",
        "claims_excluded": ["throughput improvement", "model training", "remote weight atomicity",
                            "microVM isolation", "revision-flip race from this smoke"],
        "host": {"system": platform.system(), "machine": platform.machine()},
        "source_sha256": {"durable_rollout.py": _sha(durable_rollout.__file__),
                          "coding_env.py": _sha(coding_env.__file__),
                          "realworld.py": _sha(realworld.__file__),
                          "smoke_durable_atomic_revision.py": _sha(__file__)},
        "fixture_sha256": {
            "task.json": _sha(task_dir / "task.json"),
            "verify.json": _sha(task_dir / "verifier" / "verify.json"),
            "actions.solution.replace_text.jsonl": _sha(
                task_dir / "actions.solution.replace_text.jsonl")},
        "task_id": frozen["task_id"], "task_sha256": frozen["task_sha256"],
        "artifact_binding": binding,
        "queue_revision_fence": revision,
        "snapshot_sha256": submitted_sha,
        "submission_sha256": receipt["submission_sha256"],
        "verifier_separate_process": True,
        "case_count": len(cases), "passed_cases": sum(case["passed"] for case in cases),
        "case_results": cases,
        "status": row["state"], "reward": row["reward"], "attempt": row["attempt"],
        "queue_revision_fence_current_at_read": row["queue_revision_fence_current"],
        "trainer_ready": result["trainer_ready"],
        "wall_seconds": time.monotonic() - started,
        "actor_and_enqueue_seconds": actor_seconds,
        "worker_process_and_verify_seconds": worker_seconds,
    }
    _write(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", required=True, type=Path)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = smoke(task_dir=args.task_dir, image=args.image, output=args.output)
    print(f"graded {report['case_count']} Docker cases; reward {report['reward']}; "
          f"wall {report['wall_seconds']:.3f}s")


if __name__ == "__main__":
    main()
