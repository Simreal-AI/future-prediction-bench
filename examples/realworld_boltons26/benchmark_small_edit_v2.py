"""Pair cold/prepared Boltons v2 episodes with full-write/replace-text repair.

Every method uses the same task, pinned seed, 14 host-only cases, VM assets,
visible check, and final source bytes. Timings include child provisioning,
RealWorldEnv reset/actions/verification and final-source attestation. They do
not include policy inference, optimization, or the one-time asset build.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import statistics
import time
from pathlib import Path

from future_prediction_bench.microvm_coding import MicroVMCodingAdapter
from future_prediction_bench.microvm_runtime import _clone_or_copy_qcow2, _sha256_file
from future_prediction_bench.prepared_microvm import PreparedMicroVMTemplate
from future_prediction_bench.realworld import RealWorldEnv, validate_task

from .benchmark_prepared_env import VISIBLE
from .check_microvm_branch_env import _fixture_task
from .make_task_v2 import NEW_TEXT, OLD_TEXT, PATH, TASK_ID
from .microvm_benchmark import _runtime
from .microvm_template_benchmark import _fixture


METHOD_FILES = {
    "full_write": "actions.solution.full_write.jsonl",
    "replace_text": "actions.solution.replace_text.jsonl",
}


def _episode_v2(task, adapter, actions, expected_reward, *, condition, branch):
    env = RealWorldEnv(task, adapter)
    opening = env.reset("scripted-small-edit-v2-comparison")
    if (opening["status"] != "active"
            or opening["observation"].get("runtime_kind")
               != "qemu_hvf_full_vm_qcow2_v1"):
        raise RuntimeError("Episode did not open in the expected VM runtime")
    action_observation_sha256s = []
    action_wall_seconds = []
    for action in actions:
        action_started = time.monotonic()
        result = env.step(action)
        elapsed = time.monotonic() - action_started
        if result["observation"].get("status") in {"error", "missed", "interrupted", "conflict"}:
            raise RuntimeError(f"Scripted {action['action']} action failed: "
                               f"{result['observation']!r}")
        action_wall_seconds.append({"action": action["action"], "wall_seconds": elapsed})
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
    edits = [item["wall_seconds"] for item in action_wall_seconds
             if item["action"] in {"write_file", "replace_text"}]
    return {
        "condition": condition, "branch": branch,
        "reward": graded["reward"], "task_sha256": opening["task"]["task_sha256"],
        "opening_observation": opening["observation"],
        "action_observation_sha256s": action_observation_sha256s,
        "action_wall_seconds": action_wall_seconds,
        "edit_action_seconds": edits[0] if edits else None,
        "evidence_kind": graded["evidence"]["kind"],
        "case_results": cases,
        "passed_cases": sum(case["passed"] for case in cases),
        "environment_metrics": state["metrics"],
        "adapter_metrics": state["adapter_state"]["metrics"],
        "vm_metrics": state["adapter_state"]["runtime"]["metrics"],
    }


def _actions(task_dir, original):
    scripted = {}
    fixed = original.replace(OLD_TEXT, NEW_TEXT)
    if original.count(OLD_TEXT) != 1:
        raise ValueError("Pinned old-text anchor is absent or duplicated")
    old_sha = hashlib.sha256(original.encode("utf-8")).hexdigest()
    fixed_sha = hashlib.sha256(fixed.encode("utf-8")).hexdigest()
    for method, name in METHOD_FILES.items():
        actions = [json.loads(line) for line in
                   (task_dir / name).read_text(encoding="utf-8").splitlines()
                   if line.strip()]
        kind = "write_file" if method == "full_write" else "replace_text"
        if ([item.get("action") for item in actions]
                != ["read_file", kind, "run_visible_checks", "submit"]
                or actions[0].get("path") != PATH
                or actions[1].get("path") != PATH):
            raise ValueError("V2 action sequence differs from the pinned four steps")
        edit = actions[1]
        if method == "full_write":
            if edit != {"action": "write_file", "path": PATH, "content": fixed}:
                raise ValueError("Full-write action differs from pinned repair")
        elif edit != {"action": "replace_text", "path": PATH,
                      "expected_file_sha256": old_sha,
                      "old_text": OLD_TEXT, "new_text": NEW_TEXT}:
            raise ValueError("Replace-text action differs from pinned repair")
        scripted[method] = actions
    return scripted, old_sha, fixed_sha


def _guest_source_sha(adapter):
    result = adapter.runtime.run_shell("sha256sum /mnt/root/workspace/boltons/strutils.py")
    match = re.fullmatch(r"([0-9a-f]{64})\s+\S+\n?", result["stdout"])
    if result["return_code"] or match is None:
        raise RuntimeError("Guest final source digest is unavailable")
    return match.group(1)


def _paired_semantics(rows, task_sha):
    """Require equal semantics within a method and equal solved state across methods."""
    for branch in ("repair", "baseline"):
        for method in METHOD_FILES:
            cold = rows[(method, branch, "cold")]
            warm = rows[(method, branch, "prepared")]
            for key in ("task_sha256", "opening_observation",
                        "action_observation_sha256s", "evidence_kind",
                        "case_results", "reward", "final_source_sha256"):
                if cold[key] != warm[key]:
                    raise RuntimeError(f"Cold/prepared {method} {branch} {key} differs")
            if cold["task_sha256"] != task_sha:
                raise RuntimeError("Episode task differs from one frozen v2 task")
        for condition in ("cold", "prepared"):
            full = rows[("full_write", branch, condition)]
            edit = rows[("replace_text", branch, condition)]
            for key in ("task_sha256", "opening_observation", "evidence_kind",
                        "case_results", "reward", "final_source_sha256"):
                if full[key] != edit[key]:
                    raise RuntimeError(f"Full-write/replace-text {branch} {key} differs")
            if branch == "repair":
                for index in (0, 2, 3):  # Read, visible check, submit are identical actions.
                    if (full["action_observation_sha256s"][index]
                            != edit["action_observation_sha256s"][index]):
                        raise RuntimeError("Common policy action observation differs across edits")
            if (branch == "baseline" and full["action_observation_sha256s"]
                    != edit["action_observation_sha256s"]):
                raise RuntimeError("Identical baseline policy has different observations")


def benchmark(task_dir, assets_dir, output_dir, *, repetitions=2,
              stateless_contract=None, preinstall_stateless_helper=False):
    if type(repetitions) is not int or not 1 <= repetitions <= 5:
        raise ValueError("repetitions must be in [1, 5]")
    if preinstall_stateless_helper and stateless_contract is None:
        raise ValueError("Helper preinstallation requires a stateless contract")
    task_dir, assets, output = (Path(item).resolve() for item in
                                (task_dir, assets_dir, output_dir))
    if (output.exists() and (not output.is_dir() or any(output.iterdir()))) or any(
            output.is_relative_to(root) or root.is_relative_to(output)
            for root in (task_dir, assets)):
        raise ValueError("Output must be new, empty, and disjoint")
    pristine, cases, verifier_sha, manifest = _fixture(task_dir, assets)
    task = _fixture_task(json.loads((task_dir / "task.json").read_text(encoding="utf-8")))
    if (task["task_id"] != TASK_ID or task["task_id"] != manifest["task_id"]
            or len(cases) != 14):
        raise ValueError("Expected exact v2 task and 14-case asset binding")
    source = (task_dir / "seed" / PATH).read_text(encoding="utf-8")
    methods, original_sha, repaired_sha = _actions(task_dir, source)
    output.mkdir(parents=True, exist_ok=True)
    verifier_dir = task_dir / "verifier"
    stateless_task_path = None
    if stateless_contract is not None:
        contract = Path(stateless_contract).resolve()
        source_root = output / "stateless-task-source"
        (source_root / "verifier").mkdir(parents=True)
        stateless_task_path = source_root / "task.json"
        stateless_task_path.write_text(json.dumps(task, indent=2) + "\n", encoding="utf-8")
        shutil.copy2(verifier_dir / "verify.json", source_root / "verifier" / "verify.json")
        verifier_dir = source_root / "verifier"
    else:
        contract = None
    adapter_kwargs = {
        "verifier_dir": verifier_dir, "visible_check": VISIBLE,
        "stateless_verifier_contract": contract,
        "stateless_task_path": stateless_task_path,
    }
    template_disk = output / "clean-template.qcow2"
    parent_disk = output / "template-parent.qcow2"
    setup_started = time.monotonic()
    parent = parent_runtime = None
    try:
        _clone_or_copy_qcow2(pristine, parent_disk)
        parent_runtime = _runtime(assets, parent_disk)
        parent = MicroVMCodingAdapter(parent_runtime, **adapter_kwargs)
        task.setdefault("metadata", {})["artifact_binding"] = parent.artifact_binding()
        frozen = validate_task(task)
        prepared = PreparedMicroVMTemplate.prepare(
            parent, task, seed_dir=task_dir / "seed",
            assets_manifest=assets / "manifest.json",
            template_disk_path=template_disk,
            preinstall_stateless_helper=preinstall_stateless_helper)
        prepared = PreparedMicroVMTemplate.open(
            prepared.manifest_path, expected_prepared_id=prepared.prepared_id)
        template_sha = _sha256_file(template_disk)
        template_prepare_seconds = time.monotonic() - setup_started
    except BaseException:
        if parent is not None:
            parent.close()
        elif parent_runtime is not None:
            parent_runtime.close()
        parent_disk.unlink(missing_ok=True)
        template_disk.unlink(missing_ok=True)
        Path(str(template_disk) + ".json").unlink(missing_ok=True)
        Path(str(template_disk) + ".prepared.json").unlink(missing_ok=True)
        if stateless_task_path is not None:
            shutil.rmtree(stateless_task_path.parent, ignore_errors=True)
        raise
    runs = []
    try:
        for repetition in range(repetitions):
            pair = {}
            method_order = tuple(METHOD_FILES) if repetition % 2 == 0 else tuple(
                reversed(tuple(METHOD_FILES)))
            branch_order = ("repair", "baseline") if repetition % 2 == 0 else (
                "baseline", "repair")
            for method_index, method in enumerate(method_order):
                for branch_index, branch in enumerate(branch_order):
                    condition_order = (("cold", "prepared")
                                       if (repetition + method_index + branch_index) % 2 == 0
                                       else ("prepared", "cold"))
                    for condition in condition_order:
                        disk = output / f"{condition}-r{repetition}-{method}-{branch}.qcow2"
                        adapter = None
                        started = time.monotonic()
                        try:
                            if condition == "cold":
                                _clone_or_copy_qcow2(pristine, disk)
                                adapter = MicroVMCodingAdapter(
                                    _runtime(assets, disk), **adapter_kwargs)
                            else:
                                adapter = prepared.spawn_adapters(task, [disk], max_workers=1)[0]
                            actions = (methods[method] if branch == "repair"
                                       else [methods[method][0], methods[method][-1]])
                            row = _episode_v2(task, adapter, actions,
                                           1.0 if branch == "repair" else 0.0,
                                           condition=condition, branch=branch)
                            row["method"] = method
                            row["repetition"] = repetition
                            row["final_source_sha256"] = _guest_source_sha(adapter)
                            expected_sha = repaired_sha if branch == "repair" else original_sha
                            if row["final_source_sha256"] != expected_sha:
                                raise RuntimeError("Episode final source differs from expected bytes")
                            # Check each child independently after grading.
                            check = adapter.runtime.run_shell(
                                "test ! -e /tmp/fpb-v2-prior-child && "
                                "test ! -e /mnt/root/.fpb-v2-prior-child")
                            if check["return_code"]:
                                raise RuntimeError("Prior child RAM or ext4 state leaked")
                            mark = adapter.runtime.run_shell(
                                "printf child > /tmp/fpb-v2-prior-child && "
                                "printf child > /mnt/root/.fpb-v2-prior-child")
                            if mark["return_code"]:
                                raise RuntimeError("Child-isolation marker failed")
                            final_state = adapter.get_state()
                            row["adapter_metrics"] = final_state["metrics"]
                            row["vm_metrics"] = final_state["runtime"]["metrics"]
                            row["wall_seconds"] = time.monotonic() - started
                            pair[(method, branch, condition)] = row
                            runs.append(row)
                            print(f"{condition} {method} {branch} r{repetition}: "
                                  f"{row['wall_seconds']:.3f}s reward={row['reward']}",
                                  flush=True)
                        finally:
                            if adapter is not None:
                                adapter.close()
                            disk.unlink(missing_ok=True)
            _paired_semantics(pair, frozen["task_sha256"])
            if (_sha256_file(template_disk) != template_sha
                    or _sha256_file(task_dir / "verifier" / "verify.json") != verifier_sha
                    or _sha256_file(pristine) != manifest["rootfs_qcow2_sha256"]):
                raise RuntimeError("Pinned template, verifier, or pristine disk changed")
    finally:
        parent_disk.unlink(missing_ok=True)
        template_disk.unlink(missing_ok=True)
        prepared.manifest_path.with_name(template_disk.name + ".json").unlink(missing_ok=True)
        prepared.manifest_path.unlink(missing_ok=True)
        if stateless_task_path is not None:
            shutil.rmtree(stateless_task_path.parent, ignore_errors=True)
    by_method = {}
    for method in METHOD_FILES:
        cold = [r for r in runs if r["method"] == method and r["condition"] == "cold"]
        warm = [r for r in runs if r["method"] == method and r["condition"] == "prepared"]
        cold_sum = sum(r["wall_seconds"] for r in cold)
        warm_sum = sum(r["wall_seconds"] for r in warm)
        median_cold = statistics.median(r["wall_seconds"] for r in cold)
        median_warm = statistics.median(r["wall_seconds"] for r in warm)
        by_method[method] = {
            "episode_count_per_condition": len(cold),
            "median_cold_episode_seconds": median_cold,
            "median_prepared_episode_seconds": median_warm,
            "cold_total_seconds": cold_sum,
            "prepared_total_seconds": warm_sum,
            "steady_state_time_ratio": median_cold / median_warm,
            "throughput_ratio_from_total_time": cold_sum / warm_sum,
            "amortized_ratio_if_run_alone": cold_sum / (template_prepare_seconds + warm_sum),
            "repair_only_median_cold_seconds": statistics.median(
                r["wall_seconds"] for r in cold if r["branch"] == "repair"),
            "repair_only_median_prepared_seconds": statistics.median(
                r["wall_seconds"] for r in warm if r["branch"] == "repair"),
            "repair_edit_action_median_cold_seconds": statistics.median(
                r["edit_action_seconds"] for r in cold if r["branch"] == "repair"),
            "repair_edit_action_median_prepared_seconds": statistics.median(
                r["edit_action_seconds"] for r in warm if r["branch"] == "repair"),
        }
    cold_total = sum(v["cold_total_seconds"] for v in by_method.values())
    warm_total = sum(v["prepared_total_seconds"] for v in by_method.values())
    report = {
        "kind": "boltons_v2_full_write_replace_text_prepared_pairs_v1",
        "scope": "public solved Boltons 26.0.0 fixture; full QEMU/HVF RealWorldEnv episodes",
        "uses_realworld_env": True,
        "measures_model_training": False,
        "environment_metrics_scope": "through RealWorldEnv grading; adapter and VM metrics include final attestation",
        "task_id": TASK_ID,
        "task_sha256": frozen["task_sha256"],
        "original_source_sha256": original_sha,
        "repaired_source_sha256": repaired_sha,
        "prepared_id": prepared.prepared_id,
        "template_disk_sha256": template_sha,
        "host_private_verifier_sha256": verifier_sha,
        "host_private_case_count": len(cases),
        "verifier_mode": ("stateless_namespaced_batch_v1" if contract
                          else "full_vm_per_case_v1"),
        "preinstalled_stateless_helper": preinstall_stateless_helper,
        "repetitions": repetitions,
        "order": "method, branch, and condition order alternated in paired blocks",
        "template_prepare_seconds": template_prepare_seconds,
        "methods": by_method,
        "combined_cold_total_seconds": cold_total,
        "combined_prepared_total_including_one_setup_seconds": template_prepare_seconds + warm_total,
        "combined_amortized_time_ratio": cold_total / (template_prepare_seconds + warm_total),
        "runs": runs,
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
    parser.add_argument("--stateless-contract")
    parser.add_argument("--preinstall-stateless-helper", action="store_true")
    args = parser.parse_args()
    result = benchmark(args.task_dir, args.assets_dir, args.output,
                       repetitions=args.repetitions,
                       stateless_contract=args.stateless_contract,
                       preinstall_stateless_helper=args.preinstall_stateless_helper)
    print(json.dumps({key: result[key] for key in (
        "task_id", "template_prepare_seconds", "methods",
        "combined_amortized_time_ratio")}, indent=2))
