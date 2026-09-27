"""Check non-root candidate execution with real QEMU/HVF full-VM grading.

The third arm appends a PID-1 kill attempt to the public Boltons module. It
must receive a resolved zero reward across all host-private cases rather than
turning a policy-caused guest failure into a pending infrastructure result.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

from future_prediction_bench.coding_env import _workspace_digest
from future_prediction_bench.microvm_demo import run_microvm_task
from future_prediction_bench.microvm_coding import MicroVMCodingAdapter
from future_prediction_bench.realworld_demo import load_coding_actions

from .check_microvm_branch_env import _fixture_task


_SEED_SOURCE_SHA256 = "f7f4873406d3913372c9d2b1296cc5e3efb87e88808457e93fc212a5df2de18e"
_ATTACK_SUFFIX = "\nimport os, signal\nos.kill(1, signal.SIGKILL)\n"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def probe(task_dir: Path, assets_dir: Path, output: Path, *, public_output=None):
    task_dir, assets_dir, output = (Path(p).resolve()
                                    for p in (task_dir, assets_dir, output))
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output directory must be new or empty")
    source_path = task_dir / "seed" / "boltons" / "strutils.py"
    source_bytes = source_path.read_bytes()
    if _sha(source_bytes) != _SEED_SOURCE_SHA256:
        raise ValueError("Pinned Boltons seed source changed")
    manifest = json.loads((assets_dir / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != "boltons-microvm-assets-v2"
            or manifest.get("seed_workspace_sha256")
               != _workspace_digest(task_dir / "seed")):
        raise ValueError("Pinned Boltons asset/seed binding changed")
    source = source_bytes.decode("utf-8")
    attack = source + _ATTACK_SUFFIX
    if len(attack.encode("utf-8")) > 65536:
        raise ValueError("Adversarial source exceeds the bounded write action")
    task = _fixture_task(json.loads((task_dir / "task.json").read_text(encoding="utf-8")))
    arms = {
        "baseline": load_coding_actions(task_dir / "actions.baseline.jsonl"),
        "repair": load_coding_actions(task_dir / "actions.solution.replace_text.jsonl"),
        "kill_pid1": [{"action": "write_file", "path": "boltons/strutils.py",
                       "content": attack}, {"action": "submit"}],
    }
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    bound_task_sha = None
    for arm, actions in arms.items():
        episode = output / arm
        run_microvm_task(task=task, verifier_dir=task_dir / "verifier",
                         assets_dir=assets_dir, actions=actions, output=episode)
        report_bytes = (episode / "report.json").read_bytes()
        report = json.loads(report_bytes)
        cases = report["verification"]["evidence"]["case_results"]
        if (report["status"] != "graded" or len(cases) != 14
                or report["adapter_metrics"]["full_vm_restores"] != 14):
            raise RuntimeError(f"{arm} did not complete all 14 full-VM cases")
        task_sha = report["opening"]["task"]["task_sha256"]
        if bound_task_sha is None:
            bound_task_sha = task_sha
        elif task_sha != bound_task_sha:
            raise RuntimeError("Execution arms had different frozen task bindings")
        results[arm] = {
            "status": report["status"], "reward": report["reward"],
            "passed_cases": sum(case["passed"] for case in cases),
            "case_count": len(cases),
            "return_codes": sorted({case["return_code"] for case in cases}),
            "full_vm_restores": report["adapter_metrics"]["full_vm_restores"],
            "raw_report_sha256": _sha(report_bytes),
        }
    if (results["baseline"]["reward"] != 0.0
            or results["baseline"]["passed_cases"] != 7
            or results["repair"]["reward"] != 1.0
            or results["repair"]["passed_cases"] != 14
            or results["kill_pid1"]["reward"] != 0.0
            or results["kill_pid1"]["passed_cases"] != 0
            or results["kill_pid1"]["return_codes"] != [1]):
        raise RuntimeError("Non-root candidate reward contract changed")
    report = {
        "schema_version": "microvm-guest-identity-probe-v1",
        "scope": "three offline QEMU/HVF full-VM graded episodes on one solved public task",
        "task_id": task["task_id"], "bound_task_sha256": bound_task_sha,
        "candidate_python_identity": "guest_uid_gid_65534_v2",
        "seed_source_sha256": _SEED_SOURCE_SHA256,
        "asset_binding": {
            "manifest_schema_version": manifest["schema_version"],
            "rootfs_seed_sha256": manifest["rootfs_qcow2_sha256"],
            "source_sdist_sha256": manifest["source_sdist_sha256"],
            "seed_workspace_sha256": manifest["seed_workspace_sha256"],
            "kernel_sha256": manifest["alpine_sha256"]["vmlinuz-virt"],
            "initramfs_sha256": manifest["alpine_sha256"]["initramfs-virt"],
            "readonly_disk_sha256": manifest["modloop_disk_sha256"],
        },
        "attack_source_sha256": _sha(attack.encode("utf-8")),
        "attack_kind": "module_import_attempts_to_kill_guest_pid1",
        "adapter_source_sha256": _sha(Path(inspect.getsourcefile(MicroVMCodingAdapter)).read_bytes()),
        "arms": results,
        "limits": [
            "One public solved Boltons task and one PID-1 kill attempt; not a general containment proof.",
            "No model inference, optimizer update, or throughput comparison was measured.",
            "Historical reports that used guest-root Python have a different execution binding.",
        ],
    }
    (output / "identity_probe.json").write_text(json.dumps(report, indent=2) + "\n",
                                                  encoding="utf-8")
    if public_output is not None:
        public_output = Path(public_output).resolve()
        public_output.parent.mkdir(parents=True, exist_ok=True)
        public_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", required=True, type=Path)
    parser.add_argument("--assets-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--public-output", type=Path)
    args = parser.parse_args()
    result = probe(args.task_dir, args.assets_dir, args.output,
                   public_output=args.public_output)
    print(json.dumps({"candidate_python_identity": result["candidate_python_identity"],
                      "rewards": {arm: entry["reward"] for arm, entry in result["arms"].items()}},
                     indent=2))
