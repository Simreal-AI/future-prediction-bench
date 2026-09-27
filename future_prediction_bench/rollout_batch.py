"""Bounded collection of independent, real-world coding episodes.

Each job runs a complete episode through ``run_coding_task``. The current
episode API verifies synchronously, so this is a bounded episode queue, not a
disaggregated actor/verifier/trainer runtime. In particular, it does not
generate model actions, compute log probabilities, or update weights.
"""

from __future__ import annotations

import copy
import json
import math
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

from .realworld import validate_task
from .realworld_demo import run_coding_task
from .realworld_demo import load_coding_actions


_REQUIRED_JOB_FIELDS = {"task", "seed_dir", "verifier_dir", "image", "actions", "policy_id"}
_OPTIONAL_JOB_FIELDS = {"visible_check", "verifier_workers"}


def _iso_now():
    return datetime.now(timezone.utc).isoformat()


def _stage_metric(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0 else None


def _episode_timing(path):
    """Read trusted host timings; never infer missing phases from wall time."""
    report_file = path / "report.json"
    if not report_file.is_file():
        return None
    report = json.loads(report_file.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("Episode report must be an object")
    environment = report.get("environment_metrics", {})
    adapter = report.get("adapter_metrics", {})
    if not isinstance(environment, dict) or not isinstance(adapter, dict):
        raise ValueError("Episode timing metrics are malformed")
    return {
        "environment_reset_seconds": _stage_metric(environment.get("reset_seconds")),
        "environment_action_seconds": _stage_metric(environment.get("action_seconds")),
        "environment_verify_seconds": _stage_metric(environment.get("verifier_seconds")),
        "docker_start_seconds": _stage_metric(adapter.get("docker_start_seconds")),
        "checkpoint_seconds": _stage_metric(adapter.get("checkpoint_seconds")),
        "adapter_verify_seconds": _stage_metric(adapter.get("verify_seconds")),
    }


def _prepare_job(job):
    if not isinstance(job, dict) or not _REQUIRED_JOB_FIELDS <= job.keys() or job.keys() - _REQUIRED_JOB_FIELDS - _OPTIONAL_JOB_FIELDS:
        raise ValueError("Invalid batch job fields")
    if not isinstance(job["policy_id"], str) or not job["policy_id"].strip():
        raise ValueError("Each job requires an explicit nonblank policy_id")
    frozen = validate_task(job["task"])
    if not isinstance(job["actions"], (list, tuple)) or any(not isinstance(action, dict) for action in job["actions"]):
        raise ValueError("actions must be a sequence of objects")
    return frozen, copy.deepcopy(job)


def load_coding_batch_jobs(path):
    """Load an operator-authored JSON job list, resolving paths from its file."""
    manifest = Path(path).resolve()
    jobs = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("Batch manifest must be a nonempty JSON array")
    loaded = []
    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError("Each batch job must be an object")
        item = dict(job)
        for field in ("seed_dir", "verifier_dir"):
            if isinstance(item.get(field), str):
                item[field] = str((manifest.parent / item[field]).resolve())
        if isinstance(item.get("task"), str):
            item["task"] = json.loads((manifest.parent / item["task"]).read_text(encoding="utf-8"))
        if isinstance(item.get("actions"), str):
            item["actions"] = load_coding_actions(manifest.parent / item["actions"])
        loaded.append(item)
    return loaded


def run_coding_batch(*, jobs, output_root, max_workers=4, registry_path=None,
                     runner=run_coding_task):
    """Run independent episodes with an upper bound on concurrent workers.

    ``jobs`` is a sequence of mappings with task, seed_dir, verifier_dir,
    image, actions, and policy_id. visible_check and verifier_workers are
    optional per-job settings. The same persistent registry is passed to all
    episodes. Each result occupies ``episodes/000000``, ``episodes/000001``,
    etc. Results remain in input order even when completion order differs.

    A job error is recorded and does not cancel other jobs. An invalid output
    root or worker count is a batch-level error. The caller owns the policy and
    controls how actions are generated; this function does not train a model.
    """
    if type(max_workers) is not int or not 1 <= max_workers <= 64:
        raise ValueError("max_workers must be an integer in [1, 64]")
    if not callable(runner):
        raise ValueError("runner must be callable")
    if isinstance(jobs, (str, bytes, dict)):
        raise ValueError("jobs must be a nonempty sequence")
    try:
        planned_jobs = list(jobs)
    except TypeError as exc:
        raise ValueError("jobs must be a nonempty sequence") from exc
    if not planned_jobs:
        raise ValueError("jobs must be a nonempty sequence")
    root = Path(output_root).resolve()
    if root.exists() and not root.is_dir():
        raise ValueError("Batch output path must be a directory")
    if root.exists() and any(root.iterdir()):
        raise ValueError("Batch output directory must be new or empty")
    root.mkdir(parents=True, exist_ok=True)
    (root / "episodes").mkdir()
    batch_started = time.monotonic()
    created_at = _iso_now()
    results = [None] * len(planned_jobs)

    def execute(index, job, enqueued_at):
        episode_root = root / "episodes" / f"{index:06d}"
        started = time.monotonic()
        queue_seconds = max(0.0, started - enqueued_at)
        policy_id = job.get("policy_id") if isinstance(job, dict) else None
        item = {"index": index, "task_id": None, "task_sha256": None,
                "policy_id": policy_id if isinstance(policy_id, str) else None,
                "output": str(episode_root), "execution_status": "error",
                "episode_result": None, "error_type": None, "error": None,
                "timings": {"queue_seconds": queue_seconds, "episode_seconds": None,
                            "total_seconds": None, "episode_stage_seconds": None}}
        try:
            frozen, detached = _prepare_job(job)
            item["task_id"] = frozen["task_id"]
            item["task_sha256"] = frozen["task_sha256"]
            if not frozen["is_fixture"] and registry_path is None:
                raise ValueError("A persistent task registry is required for non-fixture tasks")
            kwargs = {key: detached[key] for key in _REQUIRED_JOB_FIELDS}
            for key in _OPTIONAL_JOB_FIELDS & detached.keys():
                kwargs[key] = detached[key]
            kwargs["output"] = episode_root
            kwargs["registry_path"] = registry_path
            result = runner(**kwargs)
            if not isinstance(result, dict):
                raise TypeError("Episode runner must return a result object")
            json.dumps(result, allow_nan=False)
            item["episode_result"] = copy.deepcopy(result)
            item["execution_status"] = "completed"
            try:
                item["timings"]["episode_stage_seconds"] = _episode_timing(episode_root)
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                item["timing_detail_error"] = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            item["execution_status"] = "error"
            item["episode_result"] = None
            item["error_type"] = type(exc).__name__
            item["error"] = str(exc)
        finally:
            ended = time.monotonic()
            item["timings"]["episode_seconds"] = max(0.0, ended - started)
            item["timings"]["total_seconds"] = max(0.0, ended - enqueued_at)
        return item

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="coding-episode") as pool:
        futures = {}
        next_index = 0
        in_flight_limit = min(len(planned_jobs), 2 * max_workers)
        while next_index < len(planned_jobs) or futures:
            while next_index < len(planned_jobs) and len(futures) < in_flight_limit:
                enqueued_at = time.monotonic()
                future = pool.submit(execute, next_index, planned_jobs[next_index], enqueued_at)
                futures[future] = next_index
                next_index += 1
            completed, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in completed:
                index = futures.pop(future)
                results[index] = future.result()

    report = {"schema_version": "realworld-batch-0.1",
              "queue_kind": "bounded_complete_episodes",
              "phase_separation": False,
              "created_at": created_at, "completed_at": _iso_now(),
              "max_workers": max_workers, "job_count": len(results),
              "max_in_flight": in_flight_limit,
              "registry_path": str(Path(registry_path).resolve()) if registry_path is not None else None,
              "completed_count": sum(item["execution_status"] == "completed" for item in results),
              "error_count": sum(item["execution_status"] == "error" for item in results),
              "wall_seconds": max(0.0, time.monotonic() - batch_started),
              "jobs": results}
    encoded = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    temporary = root / "batch_report.json.tmp"
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(root / "batch_report.json")
    return report
