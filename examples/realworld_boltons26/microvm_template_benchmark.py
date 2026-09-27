"""Fair cold-parallel versus persistent warm-template VM branch experiment.

The parent is closed before any template child starts. Both conditions use
independent writable child qcow2 files, concurrent VM startup, the same two
grading workers, and the same 14 host-private Boltons cases. Timings cover
scripted environment work, not policy inference or optimizer updates.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from future_prediction_bench.coding_env import _workspace_digest
from future_prediction_bench.microvm_runtime import _clone_or_copy_qcow2
from future_prediction_bench.microvm_template import MicroVMTemplate

from .microvm_benchmark import _boot, _required, _runtime, _sha
from .microvm_fork_benchmark import _absent, _grade_branch, _value


def _fixture(task_dir: Path, assets: Path):
    manifest = json.loads((assets / "manifest.json").read_text(encoding="utf-8"))
    task = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != "boltons-microvm-assets-v2"
            or manifest.get("task_id") != task.get("task_id")
            or manifest.get("source_sdist_sha256")
               != task.get("metadata", {}).get("source_sdist_sha256")
            or manifest.get("seed_workspace_sha256")
               != _workspace_digest(task_dir / "seed")):
        raise ValueError("Assets differ from the pinned Boltons task")
    base = assets / "rootfs.qcow2"
    if _sha(base) != manifest.get("rootfs_qcow2_sha256"):
        raise ValueError("Pristine microVM disk changed")
    verifier = task_dir / "verifier" / "verify.json"
    if verifier.is_symlink():
        raise ValueError("Host-private verifier cannot be a symlink")
    specification = json.loads(verifier.read_text(encoding="utf-8"))
    if (specification.get("kind") != "command_cases_v1"
            or not isinstance(specification.get("cases"), list)
            or len(specification["cases"]) != 14):
        raise ValueError("Expected the pinned 14-case Boltons verifier")
    return base, specification["cases"], _sha(verifier), manifest


def _isolation(children, *, condition, repetition):
    ram_paths = [f"/tmp/fpb-template-{repetition}-{index}" for index in range(len(children))]
    disk_paths = [f"/mnt/root/workspace/.fpb-template-{repetition}-{index}"
                  for index in range(len(children))]
    for index, vm in enumerate(children):
        _required(vm, f"printf '{condition}-{index}' > {ram_paths[index]}")
        _required(vm, f"printf '{condition}-{index}' > {disk_paths[index]}")
    for index, vm in enumerate(children):
        for other in range(len(children)):
            if other == index:
                continue
            if not _absent(vm, ram_paths[other]) or not _absent(vm, disk_paths[other]):
                raise RuntimeError("Template siblings shared RAM or writable disk state")
    return {"independent_ram_markers": len(children),
            "independent_disk_markers": len(children)}


def benchmark(task_dir, assets_dir, output, *, repetitions=2, branch_count=2,
              grading_workers=2, keep_disks=False):
    if type(repetitions) is not int or not 1 <= repetitions <= 5:
        raise ValueError("repetitions must be in [1, 5]")
    if type(branch_count) is not int or not 2 <= branch_count <= 4:
        raise ValueError("branch_count must be in [2, 4]")
    if type(grading_workers) is not int or not 1 <= grading_workers <= branch_count:
        raise ValueError("grading_workers must be in [1, branch_count]")
    task_dir, assets, output = map(lambda item: Path(item).resolve(),
                                   (task_dir, assets_dir, output))
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output must be new or empty")
    if output.is_relative_to(assets) or assets.is_relative_to(output):
        raise ValueError("Output and immutable assets must be disjoint")
    verifier_dir = task_dir / "verifier"
    if output.is_relative_to(verifier_dir) or verifier_dir.is_relative_to(output):
        raise ValueError("Output and host-private verifier must be disjoint")
    base, cases, verifier_sha256, manifest = _fixture(task_dir, assets)
    output.mkdir(parents=True, exist_ok=True)

    template_setup_started = time.monotonic()
    parent_disk = output / "warm-parent.qcow2"
    parent_disk_clone_mode = _clone_or_copy_qcow2(base, parent_disk)
    with _runtime(assets, parent_disk) as parent:
        _boot(parent)
        _required(parent, "printf 'before-export' > /tmp/fpb-template-parent-ram")
        export_started = time.monotonic()
        template = MicroVMTemplate.export(parent, output / "warm-template.qcow2", tag="warm")
        template_export_seconds = time.monotonic() - export_started
        _required(parent, "printf 'after-export' > /tmp/fpb-parent-later-ram")
        _required(parent, "printf 'after-export' > /mnt/root/workspace/.fpb-parent-later-disk")
        if _value(parent) != "glas":
            raise RuntimeError("Parent changed pinned Boltons source unexpectedly")
    template_setup_seconds = time.monotonic() - template_setup_started
    template = MicroVMTemplate.open(template.manifest_path,
                                    expected_template_id=template.template_id)
    template_disk_sha256 = _sha(template.disk_path)
    runs = []
    for repetition in range(repetitions):
        conditions = ("cold_parallel", "template_spawn") if repetition % 2 == 0 else (
            "template_spawn", "cold_parallel")
        for condition in conditions:
            disk_paths = [output / f"{condition}-r{repetition}-b{index}.qcow2"
                          for index in range(branch_count)]
            children = []
            started = time.monotonic()
            try:
                if condition == "cold_parallel":
                    clone_modes = [_clone_or_copy_qcow2(base, path) for path in disk_paths]
                    children = [_runtime(assets, path) for path in disk_paths]
                    with ThreadPoolExecutor(max_workers=branch_count) as pool:
                        list(pool.map(_boot, children))
                    clone_mode_counts = {"clonefile": clone_modes.count("clonefile"),
                                         "copy": clone_modes.count("copy")}
                else:
                    children = template.spawn(disk_paths, max_workers=branch_count)
                    clone_mode_counts = None  # The template API owns disk provisioning.
                    for child in children:
                        if _required(child, "cat /tmp/fpb-template-parent-ram") != "before-export":
                            raise RuntimeError("Child did not restore template RAM state")
                        if not _absent(child, "/tmp/fpb-parent-later-ram") \
                                or not _absent(child, "/mnt/root/workspace/.fpb-parent-later-disk"):
                            raise RuntimeError("Child inherited post-export parent state")
                setup_seconds = time.monotonic() - started
                isolation = _isolation(children, condition=condition,
                                       repetition=repetition)
                grading_started = time.monotonic()
                with ThreadPoolExecutor(max_workers=grading_workers) as pool:
                    grades = list(pool.map(
                        lambda pair: _grade_branch(pair[1], cases, pair[0]),
                        enumerate(children)))
                grading_seconds = time.monotonic() - grading_started
                expected = [1.0 if index % 2 == 0 else 0.0
                            for index in range(branch_count)]
                if [item["reward"] for item in grades] != expected:
                    raise RuntimeError("Unexpected pinned Boltons branch rewards")
                if _sha(template.disk_path) != template_disk_sha256:
                    raise RuntimeError("Reusable template changed during grading")
                if _sha(verifier_dir / "verify.json") != verifier_sha256:
                    raise RuntimeError("Host-private verifier changed during grading")
            finally:
                for child in children:
                    child.close()
            elapsed = time.monotonic() - started
            details = [{"name": path.name, "bytes": path.stat().st_size,
                        "sha256": _sha(path)} for path in disk_paths]
            if not keep_disks:
                for path in disk_paths:
                    path.unlink()
            runs.append({"condition": condition, "repetition": repetition,
                         "wall_seconds": elapsed, "setup_seconds": setup_seconds,
                         "grading_seconds": grading_seconds,
                         "disk_clone_mode_counts": clone_mode_counts,
                         "rewards": [item["reward"] for item in grades],
                         "isolation": isolation, "branches": grades,
                         "child_disks": details})
            print(f"{condition} rep={repetition} wall={elapsed:.3f}s", flush=True)
    if not keep_disks:
        parent_disk.unlink()
        template.disk_path.unlink()
        template.manifest_path.unlink()
    cold = [row for row in runs if row["condition"] == "cold_parallel"]
    warm = [row for row in runs if row["condition"] == "template_spawn"]
    cold_total = sum(row["wall_seconds"] for row in cold)
    warm_total = template_setup_seconds + sum(row["wall_seconds"] for row in warm)
    median_cold = statistics.median(row["wall_seconds"] for row in cold)
    median_warm = statistics.median(row["wall_seconds"] for row in warm)
    graded_count = repetitions * branch_count
    report = {
        "scope": "pinned public Boltons 26.0.0 repair fixture on one QEMU/HVF host",
        "runtime": "qemu_hvf_full_vm_template_v1",
        "asset_schema_version": manifest["schema_version"],
        "host_private_case_count": len(cases),
        "host_private_verifier_sha256": verifier_sha256,
        "measures_model_training": False, "uses_realworld_env": False,
        "policy_action_window_checked": False,
        "repetitions": repetitions, "branch_count": branch_count,
        "grading_workers": grading_workers,
        "condition_order": "alternated within pairs",
        "cold_setup": "independent disks and concurrent boots",
        "template_setup_seconds": template_setup_seconds,
        "template_export_seconds": template_export_seconds,
        "parent_disk_clone_mode": parent_disk_clone_mode,
        "template_disk_sha256": template_disk_sha256,
        "parent_closed_before_spawning": True,
        "expected_rewards": [1.0 if index % 2 == 0 else 0.0
                             for index in range(branch_count)],
        "runs": runs,
        "median_cold_seconds": median_cold,
        "median_template_seconds": median_warm,
        "steady_state_throughput_ratio": median_cold / median_warm,
        "cold_total_seconds": cold_total,
        "template_total_including_setup_seconds": warm_total,
        "amortized_throughput_ratio_including_setup": cold_total / warm_total,
        "graded_branches_per_second_cold": graded_count / cold_total,
        "graded_branches_per_second_template_amortized": graded_count / warm_total,
    }
    (output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n",
                                             encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--branch-count", type=int, default=2)
    parser.add_argument("--grading-workers", type=int, default=2)
    parser.add_argument("--keep-disks", action="store_true")
    args = parser.parse_args()
    result = benchmark(args.task_dir, args.assets_dir, args.output,
                       repetitions=args.repetitions, branch_count=args.branch_count,
                       grading_workers=args.grading_workers,
                       keep_disks=args.keep_disks)
    print(json.dumps({key: result[key] for key in (
        "median_cold_seconds", "median_template_seconds",
        "steady_state_throughput_ratio", "template_setup_seconds",
        "amortized_throughput_ratio_including_setup")}, indent=2))
