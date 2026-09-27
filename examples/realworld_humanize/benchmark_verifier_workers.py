"""Three-pair Humanize Docker A/B: one versus four verifier workers.

Each pair runs both the untouched baseline and scripted repair. The order
alternates by pair to reduce simple time-order bias. All 12 episodes use the
same pinned source, task, actions, verifier, and resolved local Docker image.
Raw episode reports stay in the requested run directory; the summary contains
only hashes, aggregate case counts, and timings. No model or training runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

from future_prediction_bench.coding_env import _workspace_digest
from future_prediction_bench.realworld_demo import load_coding_actions, run_coding_task

from .make_task import (ARCHIVE_SHA256, SOURCE_COMMIT, SOURCE_FILE, VISIBLE_CHECK,
                        make_task)


TASK_ID = "humanize-4150-naturalsize-rounding-v1"
VERIFIER_SPEC_SHA256 = "4e36079f09ed50c6b9d1c23aed746e9d6ffca22cf62878411717b8329dcb7305"
PAIRS = 3
EXPECTED_PASS = {"baseline": 5, "solution": 14}
EXPECTED_REWARD = {"baseline": 0.0, "solution": 1.0}
EXPECTED_ACTIONS = {"baseline": 1, "solution": 4}
EXPECTED_CHECKPOINTS = {"baseline": 1, "solution": 2}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _schedule():
    schedule = []
    for pair in range(PAIRS):
        modes = ("baseline", "solution") if pair % 2 == 0 else ("solution", "baseline")
        workers = (1, 4) if pair % 2 == 0 else (4, 1)
        for mode in modes:
            for worker in workers:
                schedule.append((pair, mode, worker))
    return schedule


def _preflight(task_dir):
    task_dir = Path(task_dir)
    task = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    metadata = task.get("metadata", {})
    archive = task_dir / "humanize-4.15.0.tar.gz"
    if (task.get("task_id") != TASK_ID
            or metadata.get("source_commit") != SOURCE_COMMIT
            or metadata.get("source_sdist_sha256") != ARCHIVE_SHA256
            or archive.is_symlink() or _sha(archive.read_bytes()) != ARCHIVE_SHA256
            or metadata.get("source_workspace_sha256") != _workspace_digest(task_dir / "seed")
            or not (task_dir / "seed" / SOURCE_FILE).is_file()):
        raise ValueError("pinned_humanize_source_mismatch")
    verifier_path = task_dir / "verifier/verify.json"
    if _sha(verifier_path.read_bytes()) != VERIFIER_SPEC_SHA256:
        raise ValueError("pinned_humanize_verifier_changed")
    verifier = json.loads(verifier_path.read_text(encoding="utf-8"))
    if (verifier.get("kind") != "command_cases_v1"
            or not isinstance(verifier.get("cases"), list)
            or len(verifier["cases"]) != 14):
        raise ValueError("humanize_verifier_not_14_cases")
    deadline = datetime.fromisoformat(task["action_deadline"].replace("Z", "+00:00"))
    if deadline.astimezone(timezone.utc) <= datetime.now(timezone.utc) + timedelta(minutes=5):
        raise ValueError("task_deadline_too_close_for_12_episodes")
    actions = {}
    action_sha = {}
    for mode in ("baseline", "solution"):
        path = task_dir / f"actions.{mode}.jsonl"
        actions[mode] = load_coding_actions(path)
        action_sha[mode] = _sha(path.read_bytes())
    if ([item.get("action") for item in actions["baseline"]] != ["submit"]
            or [item.get("action") for item in actions["solution"]]
               != ["read_file", "write_file", "run_visible_checks", "submit"]):
        raise ValueError("humanize_action_script_changed")
    return task, actions, action_sha


def _episode(raw, *, mode, workers, source_sha, action_sha):
    evidence = raw["verification"]["evidence"]
    cases = evidence["case_results"]
    metrics = raw["adapter_metrics"]
    observation = raw["opening"]["observation"]
    transitions = raw["transitions"]
    visible = {"opening": observation,
               "actions": [item["observation"] for item in transitions]}
    complete = raw["elapsed_seconds"]
    verify = metrics["verify_seconds"]
    if (raw["task_id"] != TASK_ID or raw["status"] != "graded"
            or raw["reward"] != EXPECTED_REWARD[mode]
            or len(cases) != 14
            or sum(item["passed"] for item in cases) != EXPECTED_PASS[mode]
            or raw["environment_metrics"]["actions_used"] != EXPECTED_ACTIONS[mode]
            or metrics["checkpoints_created"] != EXPECTED_CHECKPOINTS[mode]
            or metrics["verifier_workers"] != workers or metrics["verifier_cases"] != 14
            or len(transitions) != EXPECTED_ACTIONS[mode]
            or evidence["workspace_sha256"] != transitions[-1]["observation"]["workspace_sha256"]
            or observation["workspace_sha256"] != source_sha
            or evidence["image_sha256"] != raw["image_sha256"]
            or any(not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
                   for value in (complete, verify))
            or verify > complete):
        raise RuntimeError("humanize_worker_episode_contract_failed")
    return {
        "mode": mode, "workers": workers,
        "reward": raw["reward"], "passed_cases": EXPECTED_PASS[mode],
        "case_count": len(cases),
        "case_vector_sha256": _sha(_canonical(cases).encode("utf-8")),
        "visible_observations_sha256": _sha(_canonical(visible).encode("utf-8")),
        "action_file_sha256": action_sha[mode],
        "source_workspace_sha256": source_sha,
        "submitted_workspace_sha256": evidence["workspace_sha256"],
        "verifier_sha256": evidence["verifier_sha256"],
        "image_sha256": raw["image_sha256"],
        "task_sha256": raw["opening"]["task"]["task_sha256"],
        "complete_episode_seconds": complete,
        "verifier_stage_seconds": verify,
    }


def _summary(values):
    return {"n": len(values), "median": statistics.median(values),
            "min": min(values), "max": max(values)}


def _finalize(records, *, source_sha, action_sha):
    if len(records) != PAIRS * 2 * 2:
        raise RuntimeError("humanize_worker_run_incomplete")
    by_mode = {mode: [item for item in records if item["mode"] == mode]
               for mode in ("baseline", "solution")}
    for mode, items in by_mode.items():
        if sorted((item["pair"], item["workers"]) for item in items) != [
                (pair, worker) for pair in range(PAIRS) for worker in (1, 4)]:
            raise RuntimeError("humanize_worker_schedule_incomplete")
        for key in ("reward", "passed_cases", "case_count", "case_vector_sha256",
                    "visible_observations_sha256", "action_file_sha256",
                    "source_workspace_sha256", "submitted_workspace_sha256",
                    "verifier_sha256", "image_sha256", "task_sha256"):
            if len({item[key] for item in items}) != 1:
                raise RuntimeError("humanize_worker_parity_failed: " + key)
    if (len({item["image_sha256"] for item in records}) != 1
            or len({item["source_workspace_sha256"] for item in records}) != 1
            or len({item["verifier_sha256"] for item in records}) != 1
            or len({item["task_sha256"] for item in records}) != 1
            or by_mode["baseline"][0]["submitted_workspace_sha256"] ==
               by_mode["solution"][0]["submitted_workspace_sha256"]):
        raise RuntimeError("humanize_worker_cross_mode_binding_failed")
    summaries = {}
    for mode, items in by_mode.items():
        summaries[mode] = {}
        for worker in (1, 4):
            arm = [item for item in items if item["workers"] == worker]
            summaries[mode][str(worker)] = {
                "complete_episode_seconds": _summary(
                    [item["complete_episode_seconds"] for item in arm]),
                "verifier_stage_seconds": _summary(
                    [item["verifier_stage_seconds"] for item in arm]),
            }
        summaries[mode]["paired_differences_seconds_1_minus_4"] = [
            {"pair": pair,
             "complete_episode": next(item["complete_episode_seconds"] for item in items
                                      if item["pair"] == pair and item["workers"] == 1)
             - next(item["complete_episode_seconds"] for item in items
                    if item["pair"] == pair and item["workers"] == 4),
             "verifier_stage": next(item["verifier_stage_seconds"] for item in items
                                    if item["pair"] == pair and item["workers"] == 1)
             - next(item["verifier_stage_seconds"] for item in items
                    if item["pair"] == pair and item["workers"] == 4)}
            for pair in range(PAIRS)]
    return {
        "schema_version": "humanize-docker-verifier-workers-ab-0.1",
        "classification": "public_solved_repository_integration_fixture",
        "status": "passed", "task_id": TASK_ID,
        "source_commit": SOURCE_COMMIT,
        "source_sdist_sha256": ARCHIVE_SHA256,
        "source_workspace_sha256": source_sha,
        "task_sha256": records[0]["task_sha256"],
        "verifier_sha256": records[0]["verifier_sha256"],
        "image_sha256": records[0]["image_sha256"],
        "action_file_sha256": action_sha,
        "pairs_per_mode": PAIRS, "total_episodes": len(records),
        "order": [{"pair": pair, "mode": mode, "workers": worker}
                  for pair, mode, worker in _schedule()],
        "episodes": records, "summary": summaries,
        "model_inference_measured": False, "training_measured": False,
        "limits": [
            "Public solved Humanize fixture and scripted actions, not held-out repair ability.",
            "One local Docker image and host; no policy inference or optimizer time.",
            "Three pairs per mode provide a small controlled timing sample, not a broad throughput estimate.",
            "Four workers run verifier containers concurrently and may increase host CPU and memory use.",
        ],
    }


def _write_public(path, report):
    encoded = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if any(marker in encoded for marker in (
            "/Users/", "/private/tmp/", "expected_stdout", "stdout_sha256",
            "case_results", "episode_id", "observations\": [")):
        raise RuntimeError("humanize_public_report_contains_private_detail")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8")


def benchmark(*, image, output, task_dir=None, sdist=None, public_output=None,
              runner=run_coding_task):
    if (task_dir is None) == (sdist is None):
        raise ValueError("provide_exactly_one_of_task_dir_or_sdist")
    output = Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output_must_be_new_or_empty")
    if sdist is not None:
        output.mkdir(parents=True)
        task_dir = output / "fixture"
        make_task(task_dir, sdist=sdist)
    task_dir = Path(task_dir).resolve()
    task, actions, action_sha = _preflight(task_dir)
    source_sha = task["metadata"]["source_workspace_sha256"]
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for pair, mode, workers in _schedule():
        episode_dir = output / "episodes" / f"pair{pair}-{mode}-w{workers}"
        runner(task={k: v for k, v in task.items() if k != "task_sha256"},
               seed_dir=task_dir / "seed", verifier_dir=task_dir / "verifier",
               image=image, actions=actions[mode], output=episode_dir,
               visible_check=VISIBLE_CHECK,
               policy_id=f"scripted-humanize-{mode}-w{workers}",
               verifier_workers=workers)
        raw = json.loads((episode_dir / "report.json").read_text(encoding="utf-8"))
        record = _episode(raw, mode=mode, workers=workers,
                          source_sha=source_sha, action_sha=action_sha)
        record["pair"] = pair
        records.append(record)
    report = _finalize(records, source_sha=source_sha,
                       action_sha=action_sha)
    _write_public(output / "summary.json", report)
    if public_output is not None:
        _write_public(public_output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--task-dir")
    group.add_argument("--sdist")
    parser.add_argument("--public-output")
    args = parser.parse_args(argv)
    result = benchmark(image=args.image, output=args.output, task_dir=args.task_dir,
                       sdist=args.sdist, public_output=args.public_output)
    print(json.dumps({"status": result["status"], "episodes": result["total_episodes"]}))


if __name__ == "__main__":
    main()
