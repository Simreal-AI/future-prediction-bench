"""Matched complete-episode A/B for stateless helper-only VM preinstallation.

Both conditions use independent, task-bound prepared full-VM templates and the
same explicit host-authored stateless contract. One installs the trusted
generic helper at submit; the other seals that helper, but no hidden case
code, before export. No model inference or optimizer update is timed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import time
from pathlib import Path

from future_prediction_bench.microvm_coding import MicroVMCodingAdapter
from future_prediction_bench.microvm_runtime import _clone_or_copy_qcow2, _sha256_file
from future_prediction_bench.prepared_microvm import PreparedMicroVMTemplate
from future_prediction_bench.realworld import validate_task

from .benchmark_prepared_env import VISIBLE, _episode
from .check_microvm_branch_env import _fixture_task
from .microvm_benchmark import _runtime
from .microvm_template_benchmark import _fixture


def benchmark(task_dir, assets_dir, contract_path, output_dir, *, repetitions=3):
    if type(repetitions) is not int or not 1 <= repetitions <= 5:
        raise ValueError("repetitions must be in [1, 5]")
    task_dir, assets, contract, output = (Path(item).resolve() for item in
                                          (task_dir, assets_dir, contract_path,
                                           output_dir))
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output must be new or empty")
    if (output.is_relative_to(task_dir) or task_dir.is_relative_to(output)
            or output.is_relative_to(assets) or assets.is_relative_to(output)):
        raise ValueError("Output, task, and assets must be disjoint")
    pristine, cases, verifier_sha, manifest = _fixture(task_dir, assets)
    if len(cases) != 14:
        raise ValueError("Expected pinned 14-case verifier")
    task = _fixture_task(json.loads((task_dir / "task.json").read_text(encoding="utf-8")))
    actions = [json.loads(line) for line in
               (task_dir / "actions.solution.jsonl").read_text(encoding="utf-8").splitlines()
               if line.strip()]
    if [row.get("action") for row in actions] != [
            "read_file", "write_file", "run_visible_checks", "submit"]:
        raise ValueError("Expected pinned repair actions")
    output.mkdir(parents=True, exist_ok=True)
    source = output / "stateless-task-source"
    (source / "verifier").mkdir(parents=True)
    (source / "task.json").write_text(json.dumps(task, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(task_dir / "verifier" / "verify.json", source / "verifier" / "verify.json")
    adapter_kwargs = {"verifier_dir": source / "verifier", "visible_check": VISIBLE,
                      "stateless_verifier_contract": contract,
                      "stateless_task_path": source / "task.json"}
    templates = {}
    template_hashes = {}
    setup_seconds = {}
    parent_disks = []
    try:
        for condition, preinstall in (("uploaded_helper", False),
                                      ("preinstalled_helper", True)):
            parent_disk = output / f"{condition}-parent.qcow2"
            _clone_or_copy_qcow2(pristine, parent_disk)
            parent_disks.append(parent_disk)
            adapter = MicroVMCodingAdapter(_runtime(assets, parent_disk), **adapter_kwargs)
            if "artifact_binding" not in task.get("metadata", {}):
                task.setdefault("metadata", {})["artifact_binding"] = adapter.artifact_binding()
            elif adapter.artifact_binding() != task["metadata"]["artifact_binding"]:
                raise RuntimeError("The two parent templates have different task artifacts")
            start = time.monotonic()
            templates[condition] = PreparedMicroVMTemplate.prepare(
                adapter, task, seed_dir=task_dir / "seed",
                assets_manifest=assets / "manifest.json",
                template_disk_path=output / f"{condition}-template.qcow2",
                preinstall_stateless_helper=preinstall)
            setup_seconds[condition] = time.monotonic() - start
            templates[condition] = PreparedMicroVMTemplate.open(
                templates[condition].manifest_path,
                expected_prepared_id=templates[condition].prepared_id)
            template_hashes[condition] = _sha256_file(
                output / f"{condition}-template.qcow2")
        frozen = validate_task(task)
        runs = []
        for repetition in range(repetitions):
            conditions = ("uploaded_helper", "preinstalled_helper") if repetition % 2 == 0 \
                else ("preinstalled_helper", "uploaded_helper")
            branches = ("repair", "baseline") if repetition % 2 == 0 \
                else ("baseline", "repair")
            for branch in branches:
                pair = {}
                for condition in conditions:
                    disk = output / f"{condition}-r{repetition}-{branch}.qcow2"
                    adapter = None
                    start = time.monotonic()
                    try:
                        adapter = templates[condition].spawn_adapters(task, [disk],
                                                                       max_workers=1)[0]
                        row = _episode(task, adapter,
                                       actions if branch == "repair" else [actions[0], actions[-1]],
                                       1.0 if branch == "repair" else 0.0,
                                       condition=condition, branch=branch)
                        probe = adapter.runtime.run_shell(
                            "test ! -e /tmp/fpb-helper-prior-child && "
                            "test ! -e /mnt/root/.fpb-helper-prior-child")
                        if probe["return_code"]:
                            raise RuntimeError("Previous helper child state leaked")
                        marker = adapter.runtime.run_shell(
                            "printf prior > /tmp/fpb-helper-prior-child && "
                            "printf prior > /mnt/root/.fpb-helper-prior-child")
                        if marker["return_code"]:
                            raise RuntimeError("Helper child marker write failed")
                        row["wall_seconds"] = time.monotonic() - start
                        runs.append(row)
                        pair[condition] = row
                        print(f"{condition} {branch} r{repetition}: "
                              f"{row['wall_seconds']:.3f}s reward={row['reward']}", flush=True)
                    finally:
                        if adapter is not None:
                            adapter.close()
                        disk.unlink(missing_ok=True)
                uploaded, preinstalled = pair["uploaded_helper"], pair["preinstalled_helper"]
                if (uploaded["task_sha256"] != preinstalled["task_sha256"]
                        or uploaded["task_sha256"] != frozen["task_sha256"]
                        or uploaded["opening_observation"] != preinstalled["opening_observation"]
                        or uploaded["action_observation_sha256s"]
                           != preinstalled["action_observation_sha256s"]
                        or uploaded["case_results"] != preinstalled["case_results"]
                        or uploaded["evidence_kind"] != preinstalled["evidence_kind"]):
                    raise RuntimeError("Helper-only preinstall changed episode semantics")
            if (_sha256_file(task_dir / "verifier" / "verify.json") != verifier_sha
                    or _sha256_file(pristine) != manifest["rootfs_qcow2_sha256"]
                    or any(_sha256_file(output / f"{key}-template.qcow2") != digest
                           for key, digest in template_hashes.items())):
                raise RuntimeError("A pinned helper benchmark artifact changed")
    finally:
        for parent in parent_disks:
            parent.unlink(missing_ok=True)
        for condition, template in templates.items():
            (output / f"{condition}-template.qcow2").unlink(missing_ok=True)
            template.manifest_path.unlink(missing_ok=True)
            (output / f"{condition}-template.qcow2.json").unlink(missing_ok=True)
        shutil.rmtree(source, ignore_errors=True)
    uploaded = [row for row in runs if row["condition"] == "uploaded_helper"]
    preinstalled = [row for row in runs if row["condition"] == "preinstalled_helper"]
    uploaded_total = sum(row["wall_seconds"] for row in uploaded)
    preinstalled_total = sum(row["wall_seconds"] for row in preinstalled)
    report = {
        "kind": "prepared_stateless_helper_only_ab_v1",
        "task_sha256": frozen["task_sha256"],
        "host_private_verifier_sha256": verifier_sha,
        "helper_is_generic_only": True,
        "hidden_case_code_present_at_template_export": False,
        "expected_outputs_in_guest": False,
        "uses_realworld_env": True, "measures_model_training": False,
        "repetitions": repetitions, "episodes_per_condition": len(uploaded),
        "condition_order": "alternated by repetition within branch",
        "template_prepare_seconds": setup_seconds,
        "runs": runs,
        "median_uploaded_helper_episode_seconds": statistics.median(
            row["wall_seconds"] for row in uploaded),
        "median_preinstalled_helper_episode_seconds": statistics.median(
            row["wall_seconds"] for row in preinstalled),
        "steady_state_time_ratio": statistics.median(
            row["wall_seconds"] for row in uploaded) / statistics.median(
                row["wall_seconds"] for row in preinstalled),
        "uploaded_total_including_setup_seconds": (uploaded_total
                                                   + setup_seconds["uploaded_helper"]),
        "preinstalled_total_including_setup_seconds": (preinstalled_total
                                                       + setup_seconds["preinstalled_helper"]),
        "amortized_time_ratio_including_setup": (
            (uploaded_total + setup_seconds["uploaded_helper"])
            / (preinstalled_total + setup_seconds["preinstalled_helper"])),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                             encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--stateless-contract", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    report = benchmark(args.task_dir, args.assets_dir, args.stateless_contract,
                       args.output, repetitions=args.repetitions)
    print(json.dumps({key: report[key] for key in (
        "median_uploaded_helper_episode_seconds",
        "median_preinstalled_helper_episode_seconds", "steady_state_time_ratio",
        "amortized_time_ratio_including_setup")}, indent=2))
