"""Alternating complete RealWorldEnv episodes on pinned Humanize 4.15.0.

The cooperative guest and prepared full-VM control execute the same scripted
read/replace/submit actions and the same 14 host-private cases. The guest
receives case programs only. Setup is charged separately; each row covers
one graded episode including child creation, actions, verification and cleanup.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import statistics
import time
from pathlib import Path

from examples.realworld_boltons26.check_microvm_branch_env import _fixture_task
from examples.realworld_boltons26.microvm_benchmark import _boot, _runtime
from examples.realworld_humanize.make_task import VISIBLE_CHECK as VISIBLE
from future_prediction_bench.microvm_coding import MicroVMCodingAdapter
from future_prediction_bench.microvm_runtime import _clone_or_copy_qcow2
from future_prediction_bench.prepared_microvm import PreparedMicroVMTemplate
from future_prediction_bench.realworld import RealWorldEnv, validate_task

from .host_runtime import (HumanizePrivateCases, connect_booted_vm,
                           make_env, verify_assets)
from examples.cooperative_realworld.benchmark import (
    _episode, _public_report as _base_public_report, _stats,
)
from examples.resident_guest_candidate.smoke_qemu import _outer_workspace_sha


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _canon(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _source_binding(task, assets):
    package = Path(__file__).resolve().parents[1]
    core = package.parent / "future_prediction_bench"
    named_files = {
        "benchmark_sha256": Path(__file__),
        "task_builder_sha256": package / "cooperative_realworld_humanize/make_task_v2.py",
        "asset_builder_sha256": package / "cooperative_realworld_humanize/prepare_microvm.py",
        "cooperative_host_sha256": package / "cooperative_realworld/host_runtime.py",
        "cooperative_guest_sha256": package / "cooperative_realworld/guest_service.py",
        "cooperative_case_runner_sha256": package / "cooperative_realworld/guest_case_runner.py",
        "prepared_template_sha256": core / "prepared_microvm.py",
        "microvm_runtime_sha256": core / "microvm_runtime.py",
        "microvm_template_sha256": core / "microvm_template.py",
        "microvm_coding_sha256": core / "microvm_coding.py",
        "stateless_guest_sha256": core / "guest_stateless_verifier.py",
        "task_sha256": task / "task.json",
        "verifier_sha256": task / "verifier/verify.json",
        "cooperative_contract_sha256": package / "cooperative_realworld_humanize/cooperative_contract.json",
        "resident_contract_sha256": package / "cooperative_realworld_humanize/resident_contract.json",
        "stateless_contract_sha256": package / "cooperative_realworld_humanize/stateless_contract.json",
        "asset_manifest_sha256": assets / "manifest.json",
    }
    return {name: _sha(path.read_bytes()) for name, path in named_files.items()}


def _public_report(report):
    public = _base_public_report(report)
    public["source_binding"] = report["source_binding"]
    public["common_contract"] = report["common_contract"]
    public["image_build_timing_scope"] = (
        "The offline pinned image build is not included in setup or episode timing")
    return public


def _actions(task, private):
    rows = [json.loads(line) for line in
            (task / "actions.solution.replace_text.jsonl").read_text(
                encoding="utf-8").splitlines() if line.strip()]
    if ([row.get("action") for row in rows]
            != ["read_file", "replace_text", "run_visible_checks", "submit"]
            or rows[0].get("path") != private.source_path
            or rows[1].get("path") != private.source_path
            or rows[1].get("expected_file_sha256") != private.seed_file_sha256):
        raise ValueError("pinned_action_script_changed")
    old = rows[1]["old_text"].encode("utf-8")
    new = rows[1]["new_text"].encode("utf-8")
    source = private.seed_file_path.read_bytes()
    if source.count(old) != 1 or len(old) > 256 or len(new) > 256:
        raise ValueError("pinned_repair_anchor_changed")
    repair = [rows[0], rows[1], rows[-1]]
    baseline = [rows[0], rows[-1]]
    return {"repair": repair, "baseline": baseline}, _sha(source.replace(old, new, 1))


def _control_setup(task, assets, output, source_task):
    source_task = copy.deepcopy(source_task)
    validate_task(source_task)
    private_source = output / "control-source"
    (private_source / "verifier").mkdir(parents=True)
    (private_source / "task.json").write_text(
        json.dumps(source_task, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(task / "verifier/verify.json", private_source / "verifier/verify.json")
    adapter_kwargs = {
        "verifier_dir": private_source / "verifier", "visible_check": VISIBLE,
        "stateless_verifier_contract": (Path(__file__).resolve().parents[1] /
                                        "cooperative_realworld_humanize/stateless_contract.json"),
        "stateless_task_path": private_source / "task.json",
    }
    parent_disk = output / "control-parent.qcow2"
    template_disk = output / "control-template.qcow2"
    started = time.monotonic()
    _clone_or_copy_qcow2(assets / "rootfs.qcow2", parent_disk)
    parent = MicroVMCodingAdapter(_runtime(assets, parent_disk), **adapter_kwargs)
    try:
        source_task.setdefault("metadata", {})["artifact_binding"] = parent.artifact_binding()
        prepared = PreparedMicroVMTemplate.prepare(
            parent, source_task, seed_dir=task / "seed",
            assets_manifest=assets / "manifest.json",
            template_disk_path=template_disk,
            preinstall_stateless_helper=True)
    finally:
        parent.close()
    prepared = PreparedMicroVMTemplate.open(
        prepared.manifest_path, expected_prepared_id=prepared.prepared_id)
    return source_task, prepared, time.monotonic() - started


def _equivalence(first, second, *, mode, repaired_sha, original_sha):
    coop, full = (first, second) if first["condition"] == "cooperative" else (second, first)
    if coop["mode"] != mode or full["mode"] != mode:
        raise RuntimeError("paired_mode_mismatch")
    if coop["task_sha256"] == full["task_sha256"]:
        raise RuntimeError("distinct_runtime_contracts_share_task_hash")
    if coop["reward"] != full["reward"] or coop["case_results"] != full["case_results"]:
        raise RuntimeError("host_private_case_result_or_reward_mismatch")
    if coop["passed_cases"] != (14 if mode == "repair" else 5):
        raise RuntimeError("known_fixture_result_changed")
    for opening in (coop["opening_observation"], full["opening_observation"]):
        if (opening["task_id"] != "humanize-4150-naturalsize-rounding-v2"
                or opening["workspace_root"] != "/workspace"
                or opening["visible_check"] != list(VISIBLE)):
            raise RuntimeError("opening_shared_contract_mismatch")
    if set(coop["opening_observation"]["tools"]) != {
            "read_file", "replace_text", "submit"}:
        raise RuntimeError("cooperative_tool_scope_changed")
    if not set(coop["opening_observation"]["tools"]).issubset(
            full["opening_observation"]["tools"]):
        raise RuntimeError("control_missing_common_tools")
    if len(coop["action_observations"]) != len(full["action_observations"]):
        raise RuntimeError("action_count_mismatch")
    for index, (left, right) in enumerate(zip(coop["action_observations"],
                                            full["action_observations"], strict=True)):
        if index == len(coop["action_observations"]) - 1:
            if (left.get("status") != "submitted"
                    or right.get("status") != "submitted"
                    or left.get("snapshot_kind") !=
                       "quiescent_process_fork_frozen_overlay_v1"
                    or right.get("snapshot_kind") != "full_vm_state_qcow2_v1"):
                raise RuntimeError("submit_status_or_declared_snapshot_kind_mismatch")
        elif left != right:
            raise RuntimeError("policy_action_observation_mismatch_at_" + str(index))
    expected_sha = repaired_sha if mode == "repair" else original_sha
    if coop["case_results"] and (coop["action_observations"][0]["sha256"] != original_sha
                                 or full["action_observations"][0]["sha256"] != original_sha):
        raise RuntimeError("read_did_not_observe_pristine_source")
    if mode == "repair" and (coop["action_observations"][1]["sha256"] != expected_sha
                             or full["action_observations"][1]["sha256"] != expected_sha):
        raise RuntimeError("repair_digest_mismatch")
    return {"equal_read_and_edit_observations": True,
            "equal_per_case_results": True, "equal_reward": True,
            "opening_declared_runtime_differs": True,
            "submit_declared_snapshot_kind_differs": True,
            "task_tool_manifests_differ_by_opt_in_scope": True}


def benchmark(task_dir, assets_dir, output_dir, *, repetitions=2):
    task, assets, output = map(lambda path: Path(path).resolve(),
                               (task_dir, assets_dir, output_dir))
    if type(repetitions) is not int or not 1 <= repetitions <= 5:
        raise ValueError("repetitions_must_be_1_to_5")
    if output.exists() and any(output.iterdir()):
        raise ValueError("output_must_be_new_or_empty")
    if any(output.is_relative_to(root) or root.is_relative_to(output)
           for root in (task, assets)):
        raise ValueError("output_must_be_disjoint_from_inputs")
    shared_control_task = _fixture_task(
        json.loads((task / "task.json").read_text(encoding="utf-8")))
    time_window = {key: shared_control_task[key] for key in
                   ("issued_at", "action_deadline", "outcome_not_before", "verify_after")}
    private = HumanizePrivateCases(task / "task.json", task / "verifier/verify.json",
                                   assets / "manifest.json", task_window=time_window)
    verify_assets(private, assets)
    if (shared_control_task["prompt"] != private.source_task["prompt"]
            or shared_control_task["reward_contract"] !=
               private.source_task["reward_contract"]):
        raise RuntimeError("humanize_logical_task_contract_differs")
    actions, repaired_sha = _actions(task, private)
    original_sha = private.seed_file_sha256
    output.mkdir(parents=True)
    cooperative_disk = output / "cooperative.qcow2"
    cooperative_runtime = None
    report = {"kind": "cooperative_humanize_prepared_full_vm_ab_v1",
              "status": "incomplete", "fixture": "public_solved_humanize_4150_v2",
              "realworld_env_used": True, "model_or_optimizer_used": False,
              "repetitions": repetitions, "pairs": [],
              "source_binding": _source_binding(task, assets),
              "common_contract": {
                  "task_id": private.task_id,
                  "source_path": private.source_path,
                  "source_seed_sha256": private.seed_file_sha256,
                  "verifier_sha256": private.file_digests["verifier"],
                  "case_count": len(private.cases),
                  "reward_contract_sha256": _sha(_canon(
                      private.source_task["reward_contract"]).encode("utf-8")),
                  "prompt_sha256": _sha(private.source_task["prompt"].encode("utf-8")),
                  "common_policy_actions": ["read_file", "replace_text", "submit"],
              }}
    try:
        coop_setup_started = time.monotonic()
        _clone_or_copy_qcow2(assets / "rootfs.qcow2", cooperative_disk)
        cooperative_runtime = _runtime(assets, cooperative_disk)
        _boot(cooperative_runtime)
        outer_sha_before = _outer_workspace_sha(cooperative_runtime)
        client = connect_booted_vm(cooperative_runtime, private)
        report["cooperative_one_time_setup_seconds"] = time.monotonic() - coop_setup_started
        report["cooperative_checkpoint_ns"] = client.checkpoint["checkpoint_ns"]
        report["cooperative_service_sha256"] = private.guest_service_sha256
        report["cooperative_case_runner_sha256"] = private.guest_case_runner_sha256
        report["resident_wire_and_verifier_helper_sha256"] = client.guest_helper_sha256
        control_task, template, control_setup = _control_setup(
            task, assets, output, shared_control_task)
        report["prepared_full_vm_one_time_setup_seconds"] = control_setup
        report["outer_workspace_sha256"] = outer_sha_before
        report["task_time_window_identical"] = all(
            private.experimental_task()[key] == control_task[key]
            for key in time_window)
        if not report["task_time_window_identical"]:
            raise RuntimeError("task_time_window_differs_between_arms")
        seen_nonces = set()
        for repetition in range(repetitions):
            for mode in (("repair", "baseline") if repetition % 2 == 0
                         else ("baseline", "repair")):
                order = (("cooperative", "prepared_full_vm")
                         if (repetition + (mode == "baseline")) % 2 == 0
                         else ("prepared_full_vm", "cooperative"))
                pair = {"repetition": repetition, "mode": mode,
                        "condition_order": list(order), "episodes": []}
                report["pairs"].append(pair)
                for condition in order:
                    if condition == "cooperative":
                        if _outer_workspace_sha(cooperative_runtime) != outer_sha_before:
                            raise RuntimeError("outer_workspace_changed_before_episode")
                        env = make_env(client, private, mode=mode)
                        row = _episode(env, actions[mode],
                                       expected_reward=1.0 if mode == "repair" else 0.0,
                                       condition=condition, mode=mode)
                        nonce = env.adapter.branch_process_nonce
                        if nonce in seen_nonces:
                            raise RuntimeError("cooperative_process_nonce_reused")
                        seen_nonces.add(nonce)
                        if client.state != "idle" or _outer_workspace_sha(
                                cooperative_runtime) != outer_sha_before:
                            raise RuntimeError("cooperative_cleanup_or_outer_workspace_changed")
                        row["guest_reset_ns"] = env.adapter.metrics["guest_reset_ns"]
                        row["guest_cleanup_ns"] = env.adapter.metrics["guest_cleanup_ns"]
                        row["source_after_submit_sha256"] = env.adapter.submitted_source_sha
                    else:
                        disk = output / f"full-r{repetition}-{mode}.qcow2"
                        provision_started = time.monotonic()
                        adapter = template.spawn_adapters(control_task, [disk],
                                                          max_workers=1)[0]
                        provision_seconds = time.monotonic() - provision_started
                        control_row = None
                        try:
                            env = RealWorldEnv(control_task, adapter)
                            row = _episode(env, actions[mode],
                                           expected_reward=1.0 if mode == "repair" else 0.0,
                                           condition=condition, mode=mode)
                            control_row = row
                            row["wall_seconds"] += provision_seconds
                            row["child_provision_seconds"] = provision_seconds
                            attest_started = time.monotonic()
                            source_result = adapter.runtime.run_shell(
                                "sha256sum /mnt/root/workspace/" + private.source_path,
                                timeout=10)
                            match = re.fullmatch(
                                r"([0-9a-f]{64})\s+\S+\n?",
                                source_result.get("stdout", ""))
                            expected_source_sha = (
                                repaired_sha if mode == "repair" else original_sha)
                            if (source_result.get("return_code") != 0
                                    or match is None
                                    or match.group(1) != expected_source_sha):
                                raise RuntimeError("prepared_full_vm_final_source_mismatch")
                            row["source_after_submit_sha256"] = match.group(1)
                            row["source_attestation_seconds"] = (
                                time.monotonic() - attest_started)
                        finally:
                            teardown_started = time.monotonic()
                            adapter.close()
                            disk.unlink(missing_ok=True)
                            if control_row is not None:
                                control_row["child_teardown_seconds"] = (
                                    time.monotonic() - teardown_started)
                                control_row["wall_seconds"] += control_row[
                                    "child_teardown_seconds"]
                    pair["episodes"].append(row)
                pair["parity"] = _equivalence(
                    *pair["episodes"], mode=mode,
                    repaired_sha=repaired_sha, original_sha=original_sha)
        cooperative = [row["wall_seconds"] for pair in report["pairs"]
                       for row in pair["episodes"] if row["condition"] == "cooperative"]
        full = [row["wall_seconds"] for pair in report["pairs"]
                for row in pair["episodes"] if row["condition"] == "prepared_full_vm"]
        report["cooperative_graded_episode_stats"] = _stats(cooperative)
        report["prepared_full_vm_graded_episode_stats"] = _stats(full)
        report["steady_state_median_ratio_full_over_cooperative"] = round(
            statistics.median(full) / statistics.median(cooperative), 4)
        report["amortized_total_seconds"] = {
            "cooperative": round(sum(cooperative) + report["cooperative_one_time_setup_seconds"], 6),
            "prepared_full_vm": round(sum(full) + report["prepared_full_vm_one_time_setup_seconds"], 6)}
        report["status"] = "passed"
    finally:
        if cooperative_runtime is not None:
            cooperative_runtime.close()
        cooperative_disk.unlink(missing_ok=True)
        encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if any(marker in encoded for marker in ("/Users/", "/private/tmp/",
                                              "expected_stdout", "stdout_b64")):
            raise RuntimeError("report_contains_private_detail")
        (output / "report.json").write_text(encoded, encoding="utf-8")
        if report["status"] == "passed":
            public = json.dumps(_public_report(report), indent=2, sort_keys=True,
                                allow_nan=False) + "\n"
            if any(marker in public for marker in (
                    "/Users/", "/private/tmp/", "expected_stdout", "stdout_b64",
                    "stdout_sha256", "case_results", "action_observations",
                    "whole_result_parity_sha256")):
                raise RuntimeError("public_report_contains_private_detail")
            (output / "public-report.json").write_text(public, encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repetitions", type=int, default=2)
    args = parser.parse_args(argv)
    result = benchmark(args.task_dir, args.assets_dir, args.output,
                       repetitions=args.repetitions)
    print(json.dumps({key: result[key] for key in (
        "status", "cooperative_one_time_setup_seconds",
        "prepared_full_vm_one_time_setup_seconds",
        "cooperative_graded_episode_stats",
        "prepared_full_vm_graded_episode_stats",
        "steady_state_median_ratio_full_over_cooperative")}, indent=2))


if __name__ == "__main__":
    main()
