"""Compare complete pinned RealWorldEnv episodes with optional stateless grading.

Each paired order alternates. Both modes replay the same public Boltons repair,
visible import check, and 14 host-private hidden cases in separate QEMU VMs.
The reported times exclude model inference and optimizer updates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

from future_prediction_bench.coding_env import _workspace_digest
from future_prediction_bench.http import strict_json_loads
from future_prediction_bench.microvm_demo import run_microvm_task
from future_prediction_bench.realworld_demo import load_coding_actions
from future_prediction_bench.stateless_verifier import GUEST_PROGRAM


def benchmark(task_dir, assets_dir, contract_path, output, *, pairs=3):
    if type(pairs) is not int or not 1 <= pairs <= 5:
        raise ValueError("pairs must be 1..5")
    task_dir, assets_dir = Path(task_dir).resolve(), Path(assets_dir).resolve()
    contract_path, output = Path(contract_path).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be new or empty")
    task_path = task_dir / "task.json"
    task = strict_json_loads(task_path.read_text(encoding="utf-8"))
    manifest = strict_json_loads((assets_dir / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != "boltons-microvm-assets-v2"
            or manifest.get("task_id") != task.get("task_id")
            or manifest.get("seed_workspace_sha256") != _workspace_digest(task_dir / "seed")
            or manifest.get("rootfs_qcow2_sha256") != hashlib.sha256(
                (assets_dir / "rootfs.qcow2").read_bytes()).hexdigest()):
        raise ValueError("Prepared image differs from pinned Boltons fixture")
    actions = load_coding_actions(task_dir / "actions.solution.jsonl")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for pair in range(pairs):
        order = ("default", "stateless") if pair % 2 == 0 else ("stateless", "default")
        for mode in order:
            episode_output = output / f"pair-{pair + 1}-{mode}"
            started = time.monotonic()
            result = run_microvm_task(
                task=task, verifier_dir=task_dir / "verifier", assets_dir=assets_dir,
                actions=actions, output=episode_output, stateless_task_path=(task_path if mode == "stateless" else None),
                stateless_verifier_contract=(contract_path if mode == "stateless" else None),
                visible_check=["python3", "-B", "-c", "import boltons.strutils"],
                policy_id="pinned-scripted-benchmark")
            wall = time.monotonic() - started
            report = strict_json_loads((episode_output / "report.json").read_text(encoding="utf-8"))
            evidence = report.get("verification", {}).get("evidence", {})
            cases = evidence.get("case_results", [])
            if (result["status"] != "graded" or result["reward"] != 1.0
                    or len(cases) != 14 or any(item.get("passed") is not True for item in cases)):
                raise RuntimeError("Pinned repaired episode failed parity check")
            metrics = report["environment_metrics"]
            records.append({"pair": pair + 1, "mode": mode,
                            "reward": result["reward"], "passed_cases": len(cases),
                            "setup_seconds": metrics["reset_seconds"],
                            "action_seconds": metrics["action_seconds"],
                            "verification_seconds": metrics["verifier_seconds"],
                            "episode_seconds": report["elapsed_seconds"],
                            "outer_wall_seconds": wall,
                            "full_vm_restores": report["adapter_metrics"]["full_vm_restores"],
                            "stateless_batches": report["adapter_metrics"]["stateless_batches"],
                            "verifier_evidence_kind": evidence.get("kind")})
    summaries = {}
    for mode in ("default", "stateless"):
        subset = [item for item in records if item["mode"] == mode]
        summaries[mode] = {name + "_median": statistics.median(item[name] for item in subset)
                           for name in ("setup_seconds", "action_seconds", "verification_seconds",
                                        "episode_seconds", "outer_wall_seconds")}
    ratio = (summaries["default"]["episode_seconds_median"]
             / summaries["stateless"]["episode_seconds_median"])
    report = {"kind": "pinned_microvm_realworld_stateless_episode_comparison_v1",
              "pairs": pairs, "records": records, "medians": summaries,
              "graded_episode_speed_ratio": ratio,
              "asset_binding": {
                  "manifest_schema_version": manifest["schema_version"],
                  "rootfs_seed_sha256": manifest["rootfs_qcow2_sha256"],
                  "source_sdist_sha256": manifest["source_sdist_sha256"],
                  "seed_workspace_sha256": manifest["seed_workspace_sha256"],
                  "task_id": task["task_id"],
                  "task_source_sha256": hashlib.sha256(task_path.read_bytes()).hexdigest(),
                  "verifier_sha256": hashlib.sha256(
                      (task_dir / "verifier" / "verify.json").read_bytes()).hexdigest(),
                  "contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
                  "guest_helper_sha256": hashlib.sha256(GUEST_PROGRAM.read_bytes()).hexdigest()},
              "limits": ["pinned public scripted repair, not model-generated actions",
                         "default restores a full VM per hidden case",
                         "stateless uses one batch of per-case mount/PID namespaces after one VM restore",
                         "stateless also restores once at submit to validate the frozen process census",
                         "one Apple Silicon host; no model inference or optimizer training"]}
    (output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pairs", type=int, default=3)
    args = parser.parse_args()
    result = benchmark(args.task_dir, args.assets_dir, args.contract, args.output,
                       pairs=args.pairs)
    print(json.dumps({"medians": result["medians"],
                      "graded_episode_speed_ratio": result["graded_episode_speed_ratio"]},
                     indent=2))
