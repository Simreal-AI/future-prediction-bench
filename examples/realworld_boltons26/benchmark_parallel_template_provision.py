"""A/B serial versus parallel template clone+SHA provisioning on real QEMU.

Both arms use the same sealed template, child count, parallel VM startup,
parallel grading, 14 host-private cases, and per-child full SHA-256 checks.
Only the preboot provisioning schedule changes. This is a solved-fixture
sandbox experiment, not policy inference or end-to-end RL training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from future_prediction_bench.microvm_runtime import _clone_or_copy_qcow2
from future_prediction_bench.microvm_template import MicroVMTemplate
from future_prediction_bench import microvm_template as template_module

from .microvm_benchmark import _boot, _grade, _patch, _required, _runtime, _sha
from .microvm_fork_benchmark import _absent, _value
from .microvm_template_benchmark import _fixture


def _host_ram_bytes():
    if platform.system() != "Darwin":
        return None
    try:
        result = subprocess.run(["sysctl", "-n", "hw.memsize"],
                                capture_output=True, text=True, check=False, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None
    value = result.stdout.strip()
    return int(value) if result.returncode == 0 and value.isdecimal() else None


def _source_sha(vm):
    output = _required(vm, "sha256sum /mnt/root/workspace/boltons/strutils.py")
    match = re.fullmatch(r"([0-9a-f]{64})\s+\S+\n?", output)
    if match is None:
        raise RuntimeError("Guest source digest framing failed")
    return match.group(1)


def _isolation(children, label):
    for index, vm in enumerate(children):
        _required(vm, f"printf '{label}-{index}' > /tmp/fpb-parallel-{index}")
        _required(vm, f"printf '{label}-{index}' > /mnt/root/workspace/.fpb-parallel-{index}")
    for index, vm in enumerate(children):
        for other in range(len(children)):
            if index == other:
                continue
            if not _absent(vm, f"/tmp/fpb-parallel-{other}"):
                raise RuntimeError("Sibling VM shared RAM")
            if not _absent(vm, f"/mnt/root/workspace/.fpb-parallel-{other}"):
                raise RuntimeError("Sibling VM shared ext4 disk")
    return True


def _graded_branch(vm, cases, index):
    expected_reward = 1.0 if index % 2 == 0 else 0.0
    if expected_reward:
        _patch(vm)
    value = _value(vm)
    if value != ("glass" if expected_reward else "glas"):
        raise RuntimeError("Guest code state differs from the requested branch")
    source_sha = _source_sha(vm)
    vm.save_snapshot("submitted")
    grade = _grade(vm, cases, "submitted")
    if grade["reward"] != expected_reward or grade["case_count"] != len(cases):
        raise RuntimeError("Pinned hidden-case grade differs from branch target")
    return {"index": index, "value": value, "final_source_sha256": source_sha,
            "reward": grade["reward"], "passed_cases": grade["passed_cases"],
            "case_results": grade["case_results"],
            "vm_metrics": vm.get_state()["metrics"]}


def _comparable(rows):
    return [{key: branch[key] for key in
             ("index", "value", "final_source_sha256", "reward", "passed_cases",
              "case_results")}
            for branch in rows]


def benchmark(task_dir, assets_dir, output, *, pairs=2, branches=4,
              grading_workers=4):
    if type(pairs) is not int or not 1 <= pairs <= 5:
        raise ValueError("pairs must be in [1, 5]")
    if type(branches) is not int or not 2 <= branches <= 4:
        raise ValueError("branches must be in [2, 4]")
    if type(grading_workers) is not int or not 1 <= grading_workers <= branches:
        raise ValueError("grading_workers must be in [1, branches]")
    task_dir, assets, output = (Path(path).resolve() for path in
                                (task_dir, assets_dir, output))
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output must be new or empty")
    if (output.is_relative_to(task_dir) or output.is_relative_to(assets)
            or task_dir.is_relative_to(output) or assets.is_relative_to(output)):
        raise ValueError("Output and frozen inputs must be disjoint")
    base, cases, verifier_sha, asset_manifest = _fixture(task_dir, assets)
    task = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    parent_disk = output / "parent.qcow2"
    template_disk = output / "template.qcow2"
    template = None
    rows = []
    load_before = list(os.getloadavg())
    template_start = time.monotonic()
    try:
        _clone_or_copy_qcow2(base, parent_disk)
        with _runtime(assets, parent_disk) as parent:
            _boot(parent)
            _required(parent, "printf before-export > /tmp/fpb-parallel-parent")
            template = MicroVMTemplate.export(parent, template_disk, tag="warm")
            _required(parent, "printf after-export > /tmp/fpb-parallel-later")
        template_setup_seconds = time.monotonic() - template_start
        template_sha = _sha(template.disk_path)
        for pair in range(pairs):
            order = (("serial", "parallel") if pair % 2 == 0
                     else ("parallel", "serial"))
            paired = {}
            for condition in order:
                paths = [output / f"{condition}-p{pair}-b{index}.qcow2"
                         for index in range(branches)]
                children = []
                path_indexes = {path.resolve(): index for index, path in enumerate(paths)}
                phases = []
                phase_lock = threading.Lock()
                original_clone = template_module._clone_or_copy_qcow2
                original_hash = template_module._sha256_file

                def timed_clone(source, target):
                    phase_started = time.monotonic()
                    try:
                        return original_clone(source, target)
                    finally:
                        with phase_lock:
                            phases.append(("clone", None,
                                           phase_started, time.monotonic()))

                def timed_hash(path):
                    phase_started = time.monotonic()
                    try:
                        return original_hash(path)
                    finally:
                        if path in path_indexes:
                            with phase_lock:
                                phases.append(("hash", path_indexes[path],
                                               phase_started, time.monotonic()))

                started = time.monotonic()
                try:
                    with patch.object(template_module, "_clone_or_copy_qcow2", timed_clone), \
                         patch.object(template_module, "_sha256_file", timed_hash):
                        children = template.spawn(
                            paths, max_workers=branches,
                            parallel_clone_verification=condition == "parallel")
                    setup_seconds = time.monotonic() - started
                    clones = [phase for phase in phases if phase[0] == "clone"]
                    hashes = [phase for phase in phases if phase[0] == "hash"]
                    if (len(clones) != branches or len(hashes) != branches
                            or {phase[1] for phase in hashes} != set(range(branches))):
                        raise RuntimeError("Every child must be cloned and SHA-checked once")
                    provisioning_window = (max(phase[3] for phase in hashes)
                                           - min(phase[2] for phase in clones))
                    for child in children:
                        if _required(child, "cat /tmp/fpb-parallel-parent") != "before-export":
                            raise RuntimeError("Child did not restore sealed parent RAM")
                        if not _absent(child, "/tmp/fpb-parallel-later"):
                            raise RuntimeError("Child inherited post-export RAM")
                    isolation_start = time.monotonic()
                    _isolation(children, f"{condition}-p{pair}")
                    isolation_seconds = time.monotonic() - isolation_start
                    grade_start = time.monotonic()
                    with ThreadPoolExecutor(max_workers=grading_workers) as pool:
                        grades = list(pool.map(
                            lambda indexed: _graded_branch(indexed[1], cases, indexed[0]),
                            enumerate(children)))
                    grade_seconds = time.monotonic() - grade_start
                    if _sha(template.disk_path) != template_sha:
                        raise RuntimeError("Sealed template changed")
                    if _sha(task_dir / "verifier" / "verify.json") != verifier_sha:
                        raise RuntimeError("Host-private verifier changed")
                finally:
                    for child in children:
                        child.close()
                wall_seconds = time.monotonic() - started
                for path in paths:
                    path.unlink()
                row = {"pair": pair, "condition": condition,
                       "wall_seconds": wall_seconds, "setup_seconds": setup_seconds,
                       "clone_seconds_sum": sum(phase[3] - phase[2] for phase in clones),
                       "child_sha_seconds_sum": sum(phase[3] - phase[2] for phase in hashes),
                       "clone_and_child_sha_makespan_seconds": provisioning_window,
                       "isolation_seconds": isolation_seconds,
                       "grading_seconds": grade_seconds,
                       "all_ram_and_ext4_isolated": True, "branches": grades,
                       "host_load_average_after": list(os.getloadavg())}
                rows.append(row)
                paired[condition] = row
                print(f"pair={pair} {condition} setup={setup_seconds:.3f}s "
                      f"graded_total={wall_seconds:.3f}s", flush=True)
            if _comparable(paired["serial"]["branches"]) != _comparable(
                    paired["parallel"]["branches"]):
                raise RuntimeError("A/B source, cases, or reward diverged")
        grouped = {condition: [row for row in rows if row["condition"] == condition]
                   for condition in ("serial", "parallel")}
        setup_median = {key: statistics.median(row["setup_seconds"] for row in group)
                        for key, group in grouped.items()}
        total_median = {key: statistics.median(row["wall_seconds"] for row in group)
                        for key, group in grouped.items()}
        report = {
            "schema": "fpb-qemu-template-parallel-provision-ab-v1",
            "scope": "Solved pinned Boltons fixture, four independent QEMU/HVF branch VMs",
            "task_id": task["task_id"],
            "task_source_sha256": _sha(task_dir / "task.json"),
            "asset_manifest_sha256": _sha(assets / "manifest.json"),
            "base_disk_sha256": asset_manifest["rootfs_qcow2_sha256"],
            "template_sha256": template_sha,
            "template_disk_size_bytes": template.disk_path.stat().st_size,
            "verifier_sha256": verifier_sha,
            "template_module_sha256": _sha(Path(template_module.__file__)),
            "benchmark_source_sha256": _sha(Path(__file__)),
            "host_architecture": platform.machine(),
            "host_logical_cpu_count": os.cpu_count(),
            "host_ram_bytes": _host_ram_bytes(),
            "vm_memory_mib_each": 512,
            "nominal_aggregate_vm_ram_mib": branches * 512,
            "host_load_average_before": load_before,
            "pairs": pairs, "branches_per_arm": branches,
            "grading_workers": grading_workers,
            "hidden_cases_per_branch": len(cases),
            "template_setup_seconds": template_setup_seconds,
            "condition_order": "AB/BA alternating",
            "serial_and_parallel_verify_every_child_sha256_before_any_vm_boot": True,
            "all_source_case_reward_parity_passed": True,
            "measures_policy_inference_or_optimizer": False,
            "median_setup_seconds": setup_median,
            "median_graded_batch_wall_seconds": total_median,
            "setup_speed_ratio_serial_over_parallel": setup_median["serial"] / setup_median["parallel"],
            "complete_batch_speed_ratio_serial_over_parallel": total_median["serial"] / total_median["parallel"],
            "rows": rows,
        }
        (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                                 encoding="utf-8")
        return report
    finally:
        parent_disk.unlink(missing_ok=True)
        if template is not None:
            template.disk_path.unlink(missing_ok=True)
            template.manifest_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--branches", type=int, default=4)
    parser.add_argument("--grading-workers", type=int, default=4)
    args = parser.parse_args()
    result = benchmark(args.task_dir, args.assets_dir, args.output,
                       pairs=args.pairs, branches=args.branches,
                       grading_workers=args.grading_workers)
    print(json.dumps({key: result[key] for key in
                      ("median_setup_seconds", "median_graded_batch_wall_seconds",
                       "setup_speed_ratio_serial_over_parallel",
                       "complete_batch_speed_ratio_serial_over_parallel")}, indent=2))


if __name__ == "__main__":
    main()
