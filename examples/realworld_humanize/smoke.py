"""Run baseline and scripted repair through real Docker RealWorldEnv episodes.

This is a correctness smoke on one already-public solved repository defect.
It does not evaluate a learned model, held-out task success, or throughput.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from future_prediction_bench.coding_env import _workspace_digest
from future_prediction_bench.realworld_demo import load_coding_actions, run_coding_task

from .make_task import ARCHIVE_SHA256, SOURCE_COMMIT, SOURCE_FILE, UPSTREAM_REPAIR, VISIBLE_CHECK


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def smoke(*, task_dir, image, output, public_output=None):
    task_dir, output = Path(task_dir).resolve(), Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output directory must be new or empty")
    task = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    deadline = datetime.fromisoformat(task["action_deadline"].replace("Z", "+00:00"))
    if deadline.astimezone(timezone.utc) <= datetime.now(timezone.utc):
        raise ValueError("Task action deadline passed; build a fresh pinned task")
    if (task["metadata"].get("source_commit") != SOURCE_COMMIT
            or task["metadata"].get("source_sdist_sha256") != ARCHIVE_SHA256
            or _sha((task_dir / "humanize-4.15.0.tar.gz").read_bytes()) != ARCHIVE_SHA256
            or task["metadata"].get("source_workspace_sha256")
               != _workspace_digest(task_dir / "seed")):
        raise ValueError("Task source does not match the pinned Humanize archive")
    results = {}
    for arm in ("baseline", "solution"):
        actions_path = task_dir / f"actions.{arm}.jsonl"
        run_coding_task(
            task=task, seed_dir=task_dir / "seed", verifier_dir=task_dir / "verifier",
            image=image, actions=load_coding_actions(actions_path),
            output=output / arm, visible_check=VISIBLE_CHECK,
            policy_id=f"scripted-public-humanize-{arm}", verifier_workers=1)
        raw = json.loads((output / arm / "report.json").read_text(encoding="utf-8"))
        evidence = raw["verification"]["evidence"]
        cases = evidence["case_results"]
        results[arm] = {
            "status": raw["status"], "reward": raw["reward"],
            "case_count": len(cases),
            "case_pass_count": sum(case["passed"] for case in cases),
            "visible_actions_used": raw["environment_metrics"]["actions_used"],
            "checkpoint_count": raw["adapter_metrics"]["checkpoints_created"],
            "submitted_workspace_sha256": evidence["workspace_sha256"],
            "evidence_sha256": _sha(json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()),
            "action_file_sha256": _sha(actions_path.read_bytes()),
            "image_sha256": raw["image_sha256"],
        }
    if (results["baseline"]["status"] != "graded" or results["baseline"]["reward"] != 0.0
            or results["solution"]["status"] != "graded" or results["solution"]["reward"] != 1.0
            or results["baseline"]["case_count"] != 14
            or results["solution"]["case_count"] != 14
            or results["baseline"]["case_pass_count"] != 5
            or results["solution"]["case_pass_count"] != 14
            or results["baseline"]["image_sha256"] != results["solution"]["image_sha256"]
            or results["baseline"]["submitted_workspace_sha256"]
               == results["solution"]["submitted_workspace_sha256"]):
        raise RuntimeError("Humanize baseline/repair grading contract did not hold")
    report = {
        "schema_version": "realworld-humanize-docker-smoke-0.1",
        "classification": "public_solved_repository_integration_fixture",
        "task_id": task["task_id"],
        "repository": "https://github.com/python-humanize/humanize",
        "source_tag": "4.15.0", "source_commit": SOURCE_COMMIT,
        "source_sdist_sha256": ARCHIVE_SHA256,
        "source_workspace_sha256": task["metadata"]["source_workspace_sha256"],
        "source_file": SOURCE_FILE,
        "upstream_repair": UPSTREAM_REPAIR,
        "host_verifier_cases": 14,
        "model_inference_measured": False,
        "training_measured": False,
        "throughput_claim": False,
        "arms": results,
        "limits": [
            "One public solved Humanize defect with scripted actions, not held-out repair ability.",
            "Verifier files were kept on the host during execution; their construction is visible in the public fixture builder.",
            "Only one baseline and one repair Docker episode; timings are not a throughput estimate.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "smoke.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if public_output is not None:
        path = Path(public_output).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--public-output")
    args = parser.parse_args()
    report = smoke(task_dir=args.task_dir, image=args.image,
                   output=args.output, public_output=args.public_output)
    print(json.dumps({"task_id": report["task_id"],
                      "baseline": report["arms"]["baseline"]["case_pass_count"],
                      "solution": report["arms"]["solution"]["case_pass_count"]}, indent=2))
