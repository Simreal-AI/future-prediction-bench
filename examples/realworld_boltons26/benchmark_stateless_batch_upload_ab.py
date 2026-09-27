"""Same-contract A/B of serial versus one-command stateless case-code upload.

All arms use one prepared Boltons v2 template, the same task SHA, the same
preinstalled guest helper, and the same 14 host-private cases. The optimized
arm changes only the host's *trusted* upload of case source at submit. Its
single shell line has a strict 3,800-byte bound and falls back to the
unchanged implementation if too long. Expected outputs never enter the VM.
This is a scripted environment timing, not model inference or RL training.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import re
import shutil
import statistics
import time
from pathlib import Path

from future_prediction_bench import replace_text
from future_prediction_bench.microvm_coding import (
    MicroVMCodingAdapter, replace_text_helper_binding,
)
from future_prediction_bench.microvm_runtime import _clone_or_copy_qcow2, _sha256_file
from future_prediction_bench.prepared_microvm import PreparedMicroVMTemplate
from future_prediction_bench.realworld import validate_task
from future_prediction_bench.stateless_verifier import (
    BATCH_TARGET, _batch_payload, validate_python_argv,
    validate_stateless_contract,
)

from .benchmark_prepared_env import VISIBLE
from .benchmark_small_edit_v2 import _actions, _episode_v2, _guest_source_sha
from .check_microvm_branch_env import _fixture_task
from .make_task_v2 import PATH, TASK_ID
from .microvm_benchmark import _runtime
from .microvm_template_benchmark import _fixture


PAIRS = 3
MAX_SHELL_BYTES = 3800
CURRENT_HELPER_BINDING = {
    "source_sha256": "681c6a05f2eb00a5b6b415fc295fdf3a35dc85b7e5e539cd1a968ca39e2ec1b7",
    "guest_program_sha256": "db5d96a53be748bf375005489e0888bb90a7eb9a53b56d87545d023c9b06b30e",
}


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _one_call_upload(verifier, codes, original):
    """Use one bounded guest command, retaining the original method as fallback."""
    verifier._check_host_contract()
    if not verifier.installed:
        raise RuntimeError("Install the trusted guest helper before grading")
    expected_codes = [validate_python_argv(case["argv"]) for case in verifier.cases]
    if list(codes) != expected_codes:
        raise ValueError("case_source_differs_from_frozen_verifier")
    data = _batch_payload(codes)
    encoded = base64.b64encode(data).decode("ascii")
    command = ("printf '%s' '" + encoded + "' | base64 -d > " + BATCH_TARGET
               + " && chmod 600 " + BATCH_TARGET + " && sha256sum " + BATCH_TARGET)
    if len(command.encode("ascii")) > MAX_SHELL_BYTES:
        return original(codes)
    output = verifier._required(command)
    match = re.fullmatch(r"([0-9a-f]{64})\s+\S+\n?", output)
    digest = hashlib.sha256(data).hexdigest()
    if match is None or match.group(1) != digest:
        raise RuntimeError("Uploaded guest case code digest differs")
    verifier.batch_code_sha256 = digest
    return digest


def _enable_one_call(adapter):
    verifier = adapter.stateless_verifier
    if verifier is None or verifier.batch_code_sha256 is not None:
        raise ValueError("fresh_stateless_verifier_required")
    original = verifier.prepare_batch_codes
    verifier.prepare_batch_codes = lambda codes: _one_call_upload(verifier, codes, original)


def _schedule():
    schedule = []
    for pair in range(PAIRS):
        branches = ("repair", "baseline") if pair % 2 == 0 else ("baseline", "repair")
        arms = ("serial", "one_call") if pair % 2 == 0 else ("one_call", "serial")
        for branch in branches:
            for arm in arms:
                schedule.append((pair, branch, arm))
    return schedule


def _assert_parity(rows, frozen_sha, original_sha, repaired_sha):
    if len(rows) != 4 * PAIRS:
        raise RuntimeError("ab_episode_count_incomplete")
    for branch in ("repair", "baseline"):
        subset = [row for row in rows if row["branch"] == branch]
        if sorted((row["pair"], row["arm"]) for row in subset) != [
                (pair, arm) for pair in range(PAIRS)
                for arm in ("one_call", "serial")]:
            raise RuntimeError("ab_schedule_incomplete")
        expected_reward = 1.0 if branch == "repair" else 0.0
        expected_passes = 14 if branch == "repair" else 7
        expected_source = repaired_sha if branch == "repair" else original_sha
        for row in subset:
            if (row["reward"] != expected_reward
                    or row["passed_cases"] != expected_passes
                    or len(row["case_results"]) != 14
                    or row["task_sha256"] != frozen_sha
                    or row["final_source_sha256"] != expected_source
                    or row["adapter_metrics"]["full_vm_restores"] != 1
                    or row["adapter_metrics"]["stateless_batches"] != 1
                    or row["adapter_metrics"]["stateless_batch_fallbacks"] != 0
                    or sum(case["passed"] for case in row["case_results"])
                       != row["passed_cases"]
                    or row["evidence_kind"] != "host_checked_guest_stateless_namespaced_cases_v1"):
                raise RuntimeError("same_contract_episode_grade_or_boundary_failed")
        for key in ("task_sha256", "opening_observation",
                    "action_observation_sha256s", "evidence_kind",
                    "case_results", "reward", "final_source_sha256"):
            if any(row[key] != subset[0][key] for row in subset[1:]):
                raise RuntimeError("same_contract_parity_failed: " + key)


def _stats(values):
    return {"n": len(values), "median": statistics.median(values),
            "min": min(values), "max": max(values)}


def _asset_binding(manifest):
    """Publish only path-free digests from the verified v2 asset manifest."""
    if (manifest.get("schema_version") != "boltons-microvm-assets-v2"
            or manifest.get("task_id") != TASK_ID
            or not isinstance(manifest.get("alpine_sha256"), dict)):
        raise ValueError("pinned_v2_asset_manifest_required")
    digests = {
        "rootfs_seed_sha256": manifest.get("rootfs_qcow2_sha256"),
        "source_sdist_sha256": manifest.get("source_sdist_sha256"),
        "seed_workspace_sha256": manifest.get("seed_workspace_sha256"),
        "kernel_sha256": manifest["alpine_sha256"].get("vmlinuz-virt"),
        "initramfs_sha256": manifest["alpine_sha256"].get("initramfs-virt"),
        "readonly_modloop_sha256": manifest.get("modloop_disk_sha256"),
    }
    if any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
           for value in digests.values()):
        raise ValueError("pinned_v2_asset_digest_missing_or_malformed")
    return {"manifest_schema_version": manifest["schema_version"],
            "task_id": manifest["task_id"], **digests}


def _report(rows, *, setup_seconds, frozen_sha, prepared_id, template_sha,
            verifier_sha, contract_sha, original_sha, repaired_sha,
            original_task_sha, original_task_file_sha, original_helper_sha,
            assets_manifest):
    _assert_parity(rows, frozen_sha, original_sha, repaired_sha)
    public_rows = []
    for row in rows:
        public_rows.append({
            "pair": row["pair"], "branch": row["branch"], "arm": row["arm"],
            "reward": row["reward"], "passed_cases": row["passed_cases"],
            "task_sha256": row["task_sha256"],
            "case_vector_sha256": hashlib.sha256(_canonical(row["case_results"]).encode()).hexdigest(),
            "opening_observation_sha256": hashlib.sha256(
                _canonical(row["opening_observation"]).encode()).hexdigest(),
            "action_observation_sha256s": row["action_observation_sha256s"],
            "final_source_sha256": row["final_source_sha256"],
            "episode_wall_seconds": row["wall_seconds"],
            "environment_verify_seconds": row["environment_metrics"]["verifier_seconds"],
            "batch_upload_seconds": row["adapter_metrics"]["stateless_submit_batch_upload_seconds"],
            "submit_snapshot_seconds": row["adapter_metrics"]["vm_submit_snapshot_seconds"],
            "verify_batch_seconds": row["adapter_metrics"]["stateless_verify_batch_seconds"],
            "verify_restore_seconds": row["adapter_metrics"]["stateless_verify_restore_seconds"],
            "guest_tool_seconds": row["adapter_metrics"]["guest_tool_seconds"],
        })
    if any(not math.isfinite(item[key]) or item[key] < 0 for item in public_rows
           for key in ("episode_wall_seconds", "environment_verify_seconds",
                       "batch_upload_seconds", "submit_snapshot_seconds",
                       "verify_batch_seconds", "verify_restore_seconds")):
        raise RuntimeError("nonfinite_or_negative_timing")
    summary = {}
    for branch in ("repair", "baseline"):
        summary[branch] = {}
        subset = [item for item in public_rows if item["branch"] == branch]
        for arm in ("serial", "one_call"):
            arm_rows = [item for item in subset if item["arm"] == arm]
            summary[branch][arm] = {
                key: _stats([item[key] for item in arm_rows]) for key in (
                    "episode_wall_seconds", "batch_upload_seconds", "submit_snapshot_seconds",
                    "verify_batch_seconds", "verify_restore_seconds")}
        summary[branch]["paired_serial_minus_one_call_seconds"] = [
            {"pair": pair, "episode_wall": next(item["episode_wall_seconds"] for item in subset
                                                if item["pair"] == pair and item["arm"] == "serial")
             - next(item["episode_wall_seconds"] for item in subset
                    if item["pair"] == pair and item["arm"] == "one_call"),
             "batch_upload": next(item["batch_upload_seconds"] for item in subset
                                  if item["pair"] == pair and item["arm"] == "serial")
             - next(item["batch_upload_seconds"] for item in subset
                    if item["pair"] == pair and item["arm"] == "one_call")}
            for pair in range(PAIRS)]
    serial_total = sum(item["episode_wall_seconds"] for item in public_rows
                       if item["arm"] == "serial")
    one_call_total = sum(item["episode_wall_seconds"] for item in public_rows
                         if item["arm"] == "one_call")
    return {
        "kind": "boltons_v2_prepared_stateless_batch_upload_ab_v1",
        "status": "passed", "task_id": TASK_ID, "task_sha256": frozen_sha,
        "asset_binding": _asset_binding(assets_manifest),
        "original_refreshed_task_sha256": original_task_sha,
        "original_task_file_sha256": original_task_file_sha,
        "original_replace_text_helper_sha256": original_helper_sha,
        "effective_replace_text_helper_sha256": CURRENT_HELPER_BINDING["source_sha256"],
        "same_contract_and_prepared_template": True,
        "candidate_python_identity": "guest_uid_gid_65534_v2",
        "host_private_case_count": 14,
        "host_private_verifier_sha256": verifier_sha,
        "stateless_contract_sha256": contract_sha,
        "prepared_id": prepared_id, "template_disk_sha256": template_sha,
        "original_source_sha256": original_sha,
        "repaired_source_sha256": repaired_sha,
        "template_setup_seconds": setup_seconds,
        "pairs_per_branch": PAIRS, "episode_count": len(public_rows),
        "order": [{"pair": pair, "branch": branch, "arm": arm}
                  for pair, branch, arm in _schedule()],
        "rows": public_rows, "summary": summary,
        "serial_total_seconds": serial_total,
        "one_call_total_seconds": one_call_total,
        "serial_total_including_setup_seconds": setup_seconds + serial_total,
        "one_call_total_including_setup_seconds": setup_seconds + one_call_total,
        "steady_total_time_ratio": serial_total / one_call_total,
        "setup_amortized_ratio_at_six_episodes": (
            setup_seconds + serial_total) / (setup_seconds + one_call_total),
        "model_inference_measured": False, "rl_training_measured": False,
        "limits": [
            "Public solved Boltons v2 scripted fixture, not held-out agent repair.",
            "One Apple Silicon host and three pairs per branch; timing gains may be smaller than noise.",
            "The same 14 cases run serially inside one guest batch in both arms.",
            "One full-VM save at submit and one restore at verify remain required.",
            "The one-time clean template setup is excluded from steady episode timing and included separately.",
        ],
    }


def benchmark(task_dir, assets_dir, contract_path, output_dir):
    task_dir, assets, contract, output = (Path(path).resolve() for path in
                                          (task_dir, assets_dir, contract_path, output_dir))
    if (output.exists() and (not output.is_dir() or any(output.iterdir()))) or any(
            output.is_relative_to(root) or root.is_relative_to(output)
            for root in (task_dir, assets, contract.parent)):
        raise ValueError("output_must_be_new_empty_and_disjoint")
    pristine, cases, verifier_sha, manifest = _fixture(task_dir, assets)
    task_file = task_dir / "task.json"
    task = _fixture_task(json.loads(task_file.read_text(encoding="utf-8")))
    if (task["task_id"] != TASK_ID or task["task_id"] != manifest["task_id"]
            or len(cases) != 14):
        raise ValueError("pinned_v2_fixture_required")
    original_task_sha = validate_task(task)["task_sha256"]
    original_helper_sha = task["metadata"]["replace_text_helper_binding"]["source_sha256"]
    if (replace_text_helper_binding() != CURRENT_HELPER_BINDING
            or _sha256_file(Path(replace_text.__file__))
               != CURRENT_HELPER_BINDING["source_sha256"]):
        raise RuntimeError("current_replace_text_helper_differs_from_pinned_source")
    # The published task predates the current verified replace-text helper.
    # Rebind one in-memory task, identically for both arms; leave fixture files intact.
    task["metadata"]["replace_text_helper_binding"] = dict(CURRENT_HELPER_BINDING)
    methods, original_sha, repaired_sha = _actions(
        task_dir, (task_dir / "seed" / PATH).read_text(encoding="utf-8"))
    validate_stateless_contract(task_dir, contract)
    output.mkdir(parents=True, exist_ok=True)
    source_root = output / "stateless-task-source"
    (source_root / "verifier").mkdir(parents=True)
    stateless_task_path = source_root / "task.json"
    stateless_task_path.write_text(json.dumps(task, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(task_dir / "verifier/verify.json", source_root / "verifier/verify.json")
    adapter_kwargs = {
        "verifier_dir": source_root / "verifier", "visible_check": VISIBLE,
        "stateless_verifier_contract": contract,
        "stateless_task_path": stateless_task_path,
    }
    parent_disk = output / "template-parent.qcow2"
    template_disk = output / "clean-template.qcow2"
    setup_started = time.monotonic()
    prepared = None
    parent = parent_runtime = None
    try:
        _clone_or_copy_qcow2(pristine, parent_disk)
        parent_runtime = _runtime(assets, parent_disk)
        parent = MicroVMCodingAdapter(parent_runtime, **adapter_kwargs)
        binding = parent.artifact_binding()
        if binding.get("candidate_python_identity") != "guest_uid_gid_65534_v2":
            raise RuntimeError("candidate_user_identity_changed")
        task.setdefault("metadata", {})["artifact_binding"] = binding
        frozen = validate_task(task)
        prepared = PreparedMicroVMTemplate.prepare(
            parent, task, seed_dir=task_dir / "seed",
            assets_manifest=assets / "manifest.json",
            template_disk_path=template_disk, preinstall_stateless_helper=True)
        prepared = PreparedMicroVMTemplate.open(
            prepared.manifest_path, expected_prepared_id=prepared.prepared_id)
        setup_seconds = time.monotonic() - setup_started
        template_sha = _sha256_file(template_disk)
        rows = []
        for pair, branch, arm in _schedule():
            disk = output / f"{arm}-p{pair}-{branch}.qcow2"
            adapter = None
            row = None
            started = time.monotonic()
            try:
                adapter = prepared.spawn_adapters(task, [disk], max_workers=1)[0]
                if arm == "one_call":
                    _enable_one_call(adapter)
                actions = (methods["replace_text"] if branch == "repair"
                           else [methods["replace_text"][0], methods["replace_text"][-1]])
                row = _episode_v2(task, adapter, actions,
                                  1.0 if branch == "repair" else 0.0,
                                  condition="prepared", branch=branch)
                row["final_source_sha256"] = _guest_source_sha(adapter)
                row["pair"] = pair
                row["arm"] = arm
            finally:
                if adapter is not None:
                    adapter.close()
                disk.unlink(missing_ok=True)
            row["wall_seconds"] = time.monotonic() - started
            rows.append(row)
            print(f"pair={pair} branch={branch} arm={arm} "
                  f"wall={row['wall_seconds']:.3f}s", flush=True)
        if (_sha256_file(template_disk) != template_sha
                or _sha256_file(pristine) != manifest["rootfs_qcow2_sha256"]
                or _sha256_file(task_dir / "verifier/verify.json") != verifier_sha):
            raise RuntimeError("pinned_template_or_fixture_changed")
        report = _report(rows, setup_seconds=setup_seconds,
                         frozen_sha=frozen["task_sha256"], prepared_id=prepared.prepared_id,
                         template_sha=template_sha, verifier_sha=verifier_sha,
                         contract_sha=_sha256_file(contract), original_sha=original_sha,
                         repaired_sha=repaired_sha,
                         original_task_sha=original_task_sha,
                         original_task_file_sha=_sha256_file(task_file),
                         original_helper_sha=original_helper_sha,
                         assets_manifest=manifest)
        encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if any(marker in encoded for marker in ("/Users/", "/private/tmp/",
                                                 "expected_stdout", "case_results")):
            raise RuntimeError("public_report_contains_private_detail")
        (output / "report.json").write_text(encoded, encoding="utf-8")
        return report
    finally:
        if parent is not None:
            parent.close()
        elif parent_runtime is not None:
            parent_runtime.close()
        parent_disk.unlink(missing_ok=True)
        template_disk.unlink(missing_ok=True)
        Path(str(template_disk) + ".json").unlink(missing_ok=True)
        Path(str(template_disk) + ".prepared.json").unlink(missing_ok=True)
        shutil.rmtree(source_root, ignore_errors=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = benchmark(args.task_dir, args.assets_dir, args.contract, args.output)
    print(json.dumps({"status": result["status"],
                      "steady_total_time_ratio": result["steady_total_time_ratio"],
                      "setup_amortized_ratio_at_six_episodes": result[
                          "setup_amortized_ratio_at_six_episodes"]}))


if __name__ == "__main__":
    main()
