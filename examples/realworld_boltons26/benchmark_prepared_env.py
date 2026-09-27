"""Compare complete cold and prepared-template RealWorldEnv episodes.

The same pinned task, scripted actions, and host-private 14-case verifier run
in both conditions. Time includes child disk provisioning, VM start/restore,
RealWorldEnv.reset, actions, and verification. It excludes policy inference
and gradient updates. The one-time template preparation is reported separately
and included in the amortized comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
import time
from pathlib import Path

from future_prediction_bench.microvm_coding import MicroVMCodingAdapter
from future_prediction_bench.microvm_runtime import _clone_or_copy_qcow2, _sha256_file
from future_prediction_bench.prepared_microvm import PreparedMicroVMTemplate
from future_prediction_bench.realworld import RealWorldEnv, validate_task

from .check_microvm_branch_env import _fixture_task
from .microvm_benchmark import _runtime
from .microvm_template_benchmark import _fixture


VISIBLE = ("python3", "-B", "-c", "import boltons.strutils")


def _episode(task, adapter, actions, expected_reward, *, condition, branch):
    env = RealWorldEnv(task, adapter)
    opening = env.reset("scripted-prepared-template-comparison")
    if (opening["status"] != "active"
            or opening["observation"]["runtime_kind"] != "qemu_hvf_full_vm_qcow2_v1"):
        raise RuntimeError("Episode did not open in the expected VM runtime")
    action_observation_sha256s = []
    for action in actions:
        result = env.step(action)
        if result["observation"].get("status") in {"error", "missed", "interrupted"}:
            raise RuntimeError("Scripted policy action failed")
        action_observation_sha256s.append(hashlib.sha256(json.dumps(
            result["observation"], sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest())
    if env.status != "pending":
        raise RuntimeError("Scripted policy did not submit")
    graded = env.verify()
    if graded["status"] != "graded" or graded["reward"] != expected_reward:
        raise RuntimeError("Cold/prepared episode reward differs")
    cases = graded["evidence"].get("case_results", [])
    if len(cases) != 14:
        raise RuntimeError("The pinned hidden-case count differs")
    state = env.get_state()
    return {
        "condition": condition, "branch": branch,
        "reward": graded["reward"], "task_sha256": opening["task"]["task_sha256"],
        "opening_observation": opening["observation"],
        "action_observation_sha256s": action_observation_sha256s,
        "evidence_kind": graded["evidence"]["kind"],
        "case_results": cases,
        "passed_cases": sum(case["passed"] for case in cases),
        "environment_metrics": state["metrics"],
        "adapter_metrics": state["adapter_state"]["metrics"],
        "vm_metrics": state["adapter_state"]["runtime"]["metrics"],
    }


def benchmark(task_dir, assets_dir, output_dir, *, repetitions=2,
              stateless_contract=None, preinstall_stateless_helper=False):
    if type(repetitions) is not int or not 1 <= repetitions <= 5:
        raise ValueError("repetitions must be in [1, 5]")
    if preinstall_stateless_helper and stateless_contract is None:
        raise ValueError("Helper preinstallation requires a stateless contract")
    task_dir, assets, output = (Path(item).resolve() for item in
                                (task_dir, assets_dir, output_dir))
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output must be new or empty")
    if (output.is_relative_to(task_dir) or task_dir.is_relative_to(output)
            or output.is_relative_to(assets) or assets.is_relative_to(output)):
        raise ValueError("Output, task, and immutable assets must be disjoint")
    pristine, cases, verifier_sha, manifest = _fixture(task_dir, assets)
    task = _fixture_task(json.loads((task_dir / "task.json").read_text(encoding="utf-8")))
    if task["task_id"] != manifest["task_id"] or len(cases) != 14:
        raise ValueError("Pinned fixture differs")
    actions = [json.loads(line) for line in
               (task_dir / "actions.solution.jsonl").read_text(encoding="utf-8").splitlines()
               if line.strip()]
    if ([action.get("action") for action in actions]
            != ["read_file", "write_file", "run_visible_checks", "submit"]):
        raise ValueError("Expected the pinned four-action scripted repair")
    output.mkdir(parents=True, exist_ok=True)
    verifier_dir = task_dir / "verifier"
    stateless_task_path = None
    if stateless_contract is not None:
        # The contract binds the exact operator-authored task.json next to its
        # verifier. The local fixture deadline is refreshed before binding;
        # neither file is ever mounted into a guest.
        stateless_contract = Path(stateless_contract).resolve()
        local_source = output / "stateless-task-source"
        (local_source / "verifier").mkdir(parents=True)
        stateless_task_path = local_source / "task.json"
        stateless_task_path.write_text(json.dumps(task, indent=2) + "\n", encoding="utf-8")
        shutil.copy2(verifier_dir / "verify.json", local_source / "verifier" / "verify.json")
        verifier_dir = local_source / "verifier"
    adapter_kwargs = {
        "verifier_dir": verifier_dir, "visible_check": VISIBLE,
        "stateless_verifier_contract": stateless_contract,
        "stateless_task_path": stateless_task_path,
    }
    parent_disk = output / "template-parent.qcow2"
    _clone_or_copy_qcow2(pristine, parent_disk)
    parent = MicroVMCodingAdapter(_runtime(assets, parent_disk), **adapter_kwargs)
    task.setdefault("metadata", {})["artifact_binding"] = parent.artifact_binding()
    frozen = validate_task(task)
    prepared_started = time.monotonic()
    try:
        prepared = PreparedMicroVMTemplate.prepare(
            parent, task, seed_dir=task_dir / "seed",
            assets_manifest=assets / "manifest.json",
            template_disk_path=output / "clean-template.qcow2",
            preinstall_stateless_helper=preinstall_stateless_helper)
    except BaseException:
        parent.close()
        parent_disk.unlink(missing_ok=True)
        if stateless_task_path is not None:
            shutil.rmtree(stateless_task_path.parent, ignore_errors=True)
        raise
    prepare_seconds = time.monotonic() - prepared_started
    prepared = PreparedMicroVMTemplate.open(
        prepared.manifest_path, expected_prepared_id=prepared.prepared_id)
    template_disk = Path(str(prepared.manifest_path).removesuffix(".prepared.json"))
    template_sha = _sha256_file(template_disk)
    runs = []
    try:
        for repetition in range(repetitions):
            # Alternate both condition and branch order to expose host-cache
            # and gradual thermal/load effects in the raw paired data.
            conditions = ("cold", "prepared") if repetition % 2 == 0 else (
                "prepared", "cold")
            branches = ("repair", "baseline") if repetition % 2 == 0 else (
                "baseline", "repair")
            pair = {}
            for branch in branches:
                for condition in conditions:
                    disk = output / f"{condition}-r{repetition}-{branch}.qcow2"
                    adapter = None
                    started = time.monotonic()
                    try:
                        if condition == "cold":
                            _clone_or_copy_qcow2(pristine, disk)
                            adapter = MicroVMCodingAdapter(
                                _runtime(assets, disk), **adapter_kwargs)
                        else:
                            adapter = prepared.spawn_adapters(task, [disk], max_workers=1)[0]
                        selected = actions if branch == "repair" else [actions[0], actions[-1]]
                        row = _episode(task, adapter, selected,
                                       1.0 if branch == "repair" else 0.0,
                                       condition=condition, branch=branch)
                        # Mutate only this child's RAM/tmpfs and ext4 after
                        # grading. The next child must start from a clean
                        # template, independent of this finished episode.
                        check = adapter.runtime.run_shell(
                            "test ! -e /tmp/fpb-prepared-prior-child && "
                            "test ! -e /mnt/root/.fpb-prepared-prior-child")
                        if check["return_code"]:
                            raise RuntimeError("A previous child leaked RAM or disk state")
                        written = adapter.runtime.run_shell(
                            "printf child > /tmp/fpb-prepared-prior-child && "
                            "printf child > /mnt/root/.fpb-prepared-prior-child")
                        if written["return_code"]:
                            raise RuntimeError("Child-isolation marker write failed")
                        row["wall_seconds"] = time.monotonic() - started
                        pair[(branch, condition)] = row
                        runs.append(row)
                        print(f"{condition} {branch} r{repetition}: "
                              f"{row['wall_seconds']:.3f}s reward={row['reward']}", flush=True)
                    finally:
                        if adapter is not None:
                            adapter.close()
                        disk.unlink(missing_ok=True)
                cold, warm = pair[(branch, "cold")], pair[(branch, "prepared")]
                if (cold["task_sha256"] != warm["task_sha256"]
                        or cold["task_sha256"] != frozen["task_sha256"]
                        or cold["opening_observation"] != warm["opening_observation"]
                        or cold["action_observation_sha256s"]
                           != warm["action_observation_sha256s"]
                        or cold["case_results"] != warm["case_results"]
                        or cold["evidence_kind"] != warm["evidence_kind"]):
                    raise RuntimeError("Prepared episode differs from cold RealWorldEnv semantics")
            if (_sha256_file(template_disk) != template_sha
                    or _sha256_file(task_dir / "verifier" / "verify.json") != verifier_sha
                    or _sha256_file(pristine) != manifest["rootfs_qcow2_sha256"]):
                raise RuntimeError("Immutable template, verifier, or seed changed")
    finally:
        parent_disk.unlink(missing_ok=True)
        template_disk.unlink(missing_ok=True)
        prepared.manifest_path.with_name(template_disk.name + ".json").unlink(missing_ok=True)
        prepared.manifest_path.unlink(missing_ok=True)
        if stateless_task_path is not None:
            shutil.rmtree(stateless_task_path.parent)
    cold = [row for row in runs if row["condition"] == "cold"]
    warm = [row for row in runs if row["condition"] == "prepared"]
    cold_total = sum(row["wall_seconds"] for row in cold)
    warm_total = sum(row["wall_seconds"] for row in warm)
    report = {
        "kind": "prepared_realworld_full_vm_episodes_v1",
        "scope": "pinned public Boltons 26.0.0 scripted RealWorldEnv fixture",
        "uses_realworld_env": True, "measures_model_training": False,
        "verifier_mode": ("stateless_namespaced_batch_v1" if stateless_contract
                          else "full_vm_per_case_v1"),
        "preinstalled_stateless_helper": preinstall_stateless_helper,
        "task_sha256": frozen["task_sha256"],
        "prepared_id": prepared.prepared_id,
        "host_private_verifier_sha256": verifier_sha,
        "template_disk_sha256": template_sha,
        "template_prepare_seconds": prepare_seconds,
        "repetitions": repetitions,
        "episode_count_per_condition": len(cold),
        "condition_order": "alternated by repetition within branch",
        "runs": runs,
        "median_cold_episode_seconds": statistics.median(r["wall_seconds"] for r in cold),
        "median_prepared_episode_seconds": statistics.median(r["wall_seconds"] for r in warm),
        "steady_state_time_ratio": (statistics.median(r["wall_seconds"] for r in cold)
                                    / statistics.median(r["wall_seconds"] for r in warm)),
        "cold_total_seconds": cold_total,
        "prepared_total_including_setup_seconds": prepare_seconds + warm_total,
        "amortized_time_ratio_including_setup": cold_total / (prepare_seconds + warm_total),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                             encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--stateless-contract", help="Exact task-specific opt-in contract")
    parser.add_argument("--preinstall-stateless-helper", action="store_true",
                        help="Seal the trusted generic stateless helper in the clean template")
    args = parser.parse_args()
    report = benchmark(args.task_dir, args.assets_dir, args.output,
                       repetitions=args.repetitions,
                       stateless_contract=args.stateless_contract,
                       preinstall_stateless_helper=args.preinstall_stateless_helper)
    print(json.dumps({key: report[key] for key in (
        "median_cold_episode_seconds", "median_prepared_episode_seconds",
        "steady_state_time_ratio", "template_prepare_seconds",
        "amortized_time_ratio_including_setup")}, indent=2))
