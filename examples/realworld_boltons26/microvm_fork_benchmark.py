"""Compare independent cold VMs with siblings forked from one warm VM snapshot.

This operates directly on pinned QEMU/HVF assets and host-private Boltons
cases. It is an infrastructure experiment on a publicly solved fixture, not
an RL policy episode or a model-training throughput measurement.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from future_prediction_bench.coding_env import _workspace_digest
from future_prediction_bench.microvm_runtime import MicroVMRuntime, _clone_or_copy_qcow2

from .microvm_benchmark import _boot, _grade, _patch, _python, _required, _runtime, _sha


def _absent(vm, path):
    return vm.run_shell(f"test ! -e {path}")["return_code"] == 0


def _value(vm):
    result = _python(vm, "from boltons.strutils import singularize; print('FPB_VALUE='+singularize('glass'))")
    if result["return_code"]:
        raise RuntimeError("Pinned Boltons import failed in guest")
    if "FPB_VALUE=glass" in result["stdout"]:
        return "glass"
    if "FPB_VALUE=glas" in result["stdout"]:
        return "glas"
    raise RuntimeError("Unexpected pinned Boltons result")


def _verify_isolation(parent, children, *, condition, repetition):
    """Check per-VM RAM (/tmp) and ext4 disk markers while siblings are live."""
    ram_paths = [f"/tmp/fpb-fork-{repetition}-{index}" for index in range(len(children))]
    disk_paths = [f"/mnt/root/workspace/.fpb-fork-{repetition}-{index}" for index in range(len(children))]
    for index, vm in enumerate(children):
        _required(vm, f"printf '{condition}-{index}' > {ram_paths[index]}")
        _required(vm, f"printf '{condition}-{index}' > {disk_paths[index]}")
    for index, vm in enumerate(children):
        for other in range(len(children)):
            if other == index:
                continue
            if not _absent(vm, ram_paths[other]) or not _absent(vm, disk_paths[other]):
                raise RuntimeError("Sibling VM shared RAM or writable disk state")
    for ram_path, disk_path in zip(ram_paths, disk_paths):
        if not _absent(parent, ram_path) or not _absent(parent, disk_path):
            raise RuntimeError("Parent VM was changed by a sibling")
    return {"independent_ram_markers": len(children),
            "independent_disk_markers": len(children),
            "parent_unchanged": True}


def _grade_branch(vm, cases, index):
    expected = 1.0 if index % 2 == 0 else 0.0
    if expected:
        _patch(vm)
    observed = _value(vm)
    if observed != ("glass" if expected else "glas"):
        raise RuntimeError("Sibling code state differs from its requested suffix")
    vm.save_snapshot("submitted")
    grade = _grade(vm, cases, "submitted")
    if grade["reward"] != expected or grade["case_count"] != len(cases):
        raise RuntimeError("Host-private verifier disagrees with scripted suffix")
    return {"index": index, "expected_reward": expected, "reward": grade["reward"],
            "passed_cases": grade["passed_cases"], "case_count": grade["case_count"],
            "value": observed, "vm_metrics": vm.get_state()["metrics"]}


def benchmark(task_dir, assets_dir, output, *, repetitions=2, branch_count=2,
              grading_workers=2, keep_disks=False):
    """Run alternating paired conditions with one persistent warm parent VM.

    Cold and forked conditions both keep all child VMs live during isolation
    checks and use the same bounded grading worker count. Parent setup is
    measured once and included separately in the amortized comparison.
    """
    if type(repetitions) is not int or not 1 <= repetitions <= 5:
        raise ValueError("repetitions must be an integer in [1, 5]")
    if type(branch_count) is not int or not 2 <= branch_count <= 4:
        raise ValueError("branch_count must be an integer in [2, 4]")
    if type(grading_workers) is not int or not 1 <= grading_workers <= branch_count:
        raise ValueError("grading_workers must be in [1, branch_count]")
    if type(keep_disks) is not bool:
        raise ValueError("keep_disks must be boolean")
    task_dir, assets, output = map(lambda path: Path(path).resolve(),
                                   (task_dir, assets_dir, output))
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output must be new or empty")
    if output.is_relative_to(assets) or assets.is_relative_to(output):
        raise ValueError("Output and immutable assets must be disjoint")
    if output.is_relative_to(task_dir / "verifier") or (task_dir / "verifier").is_relative_to(output):
        raise ValueError("Output and host-private verifier must be disjoint")
    manifest = json.loads((assets / "manifest.json").read_text(encoding="utf-8"))
    task = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != "boltons-microvm-assets-v2"
            or manifest.get("task_id") != task.get("task_id")
            or manifest.get("source_sdist_sha256") != task.get("metadata", {}).get("source_sdist_sha256")
            or manifest.get("seed_workspace_sha256") != _workspace_digest(task_dir / "seed")):
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
    cases = specification["cases"]
    verifier_sha256 = _sha(verifier)
    if not callable(getattr(MicroVMRuntime, "fork_snapshot", None)):
        raise RuntimeError("This MicroVMRuntime does not implement fork_snapshot")

    output.mkdir(parents=True, exist_ok=True)
    parent_setup_started = time.monotonic()
    parent_disk = output / "warm-parent.qcow2"
    parent_disk_clone_mode = _clone_or_copy_qcow2(base, parent_disk)
    runs = []
    with _runtime(assets, parent_disk) as parent:
        _boot(parent)
        _required(parent, "printf 'parent-warm-state' > /tmp/fpb-parent-ram")
        parent.save_snapshot("warm")
        parent_setup_seconds = time.monotonic() - parent_setup_started
        if _value(parent) != "glas":
            raise RuntimeError("Warm parent is not the pristine Boltons state")

        for repetition in range(repetitions):
            order = ("cold_parallel", "forked_siblings") if repetition % 2 == 0 else (
                "forked_siblings", "cold_parallel")
            for condition in order:
                disk_paths = [output / f"{condition}-r{repetition}-b{index}.qcow2"
                              for index in range(branch_count)]
                children = []
                started = time.monotonic()
                try:
                    if condition == "cold_parallel":
                        clone_modes = [_clone_or_copy_qcow2(base, disk_path)
                                       for disk_path in disk_paths]
                        clone_mode_counts = {"clonefile": clone_modes.count("clonefile"),
                                             "copy": clone_modes.count("copy")}
                        children = [_runtime(assets, disk_path) for disk_path in disk_paths]
                        # Cold independent VMs may be provisioned in parallel.
                        # A serial cold boot would bias the comparison in favor
                        # of forked siblings.
                        with ThreadPoolExecutor(max_workers=branch_count) as pool:
                            list(pool.map(_boot, children))
                    else:
                        before_metrics = parent.get_state()["metrics"]
                        children = parent.fork_snapshot("warm", disk_paths)
                        after_metrics = parent.get_state()["metrics"]
                        clone_mode_counts = {
                            "clonefile": after_metrics["fork_reflink_disks"] - before_metrics["fork_reflink_disks"],
                            "copy": after_metrics["fork_copied_disks"] - before_metrics["fork_copied_disks"]}
                        if len(children) != branch_count:
                            raise RuntimeError("MicroVM fork returned the wrong sibling count")
                        if any(_required(child, "cat /tmp/fpb-parent-ram") != "parent-warm-state"
                               for child in children):
                            raise RuntimeError("Forked child did not inherit parent RAM state")
                    setup_seconds = time.monotonic() - started
                    if len({str(child.disk_path) for child in children}) != branch_count:
                        raise RuntimeError("Sibling VMs did not receive unique writable disk paths")
                    isolation = _verify_isolation(parent, children, condition=condition,
                                                  repetition=repetition)
                    grading_started = time.monotonic()
                    with ThreadPoolExecutor(max_workers=grading_workers) as pool:
                        grades = list(pool.map(lambda pair: _grade_branch(pair[1], cases, pair[0]),
                                               enumerate(children)))
                    grading_seconds = time.monotonic() - grading_started
                    if _value(parent) != "glas" or _required(parent, "cat /tmp/fpb-parent-ram") != "parent-warm-state":
                        raise RuntimeError("Sibling operation changed the warm parent")
                    expected = [1.0 if index % 2 == 0 else 0.0
                                for index in range(branch_count)]
                    if [item["reward"] for item in grades] != expected:
                        raise RuntimeError("Unexpected branch rewards")
                    if _sha(verifier) != verifier_sha256:
                        raise RuntimeError("Host-private verifier changed during grading")
                finally:
                    for child in children:
                        child.close()
                elapsed = time.monotonic() - started
                disk_details = [{"path": str(path), "bytes": path.stat().st_size,
                                 "sha256": _sha(path)} for path in disk_paths]
                if not keep_disks:
                    for path in disk_paths:
                        path.unlink()
                runs.append({"condition": condition, "repetition": repetition,
                             "wall_seconds": elapsed, "setup_seconds": setup_seconds,
                             "grading_seconds": grading_seconds,
                             "disk_clone_mode_counts": clone_mode_counts,
                             "rewards": [item["reward"] for item in grades],
                             "branches": grades, "isolation": isolation,
                             "child_disks": disk_details})
                print(f"{condition} rep={repetition} wall={elapsed:.3f}s", flush=True)
    if not keep_disks:
        parent_disk.unlink()
    cold = [item for item in runs if item["condition"] == "cold_parallel"]
    forked = [item for item in runs if item["condition"] == "forked_siblings"]
    cold_total = sum(item["wall_seconds"] for item in cold)
    fork_total = parent_setup_seconds + sum(item["wall_seconds"] for item in forked)
    median_cold = statistics.median(item["wall_seconds"] for item in cold)
    median_forked = statistics.median(item["wall_seconds"] for item in forked)
    graded_count = repetitions * branch_count
    result = {"scope": "pinned public Boltons 26.0.0 repair fixture on one QEMU/HVF host",
              "runtime": "qemu_hvf_full_vm_qcow2_v1",
              "asset_schema_version": manifest["schema_version"],
              "host_private_case_count": len(cases),
              "host_private_verifier_sha256": verifier_sha256,
              "measures_model_training": False,
              "uses_realworld_env": False,
              "policy_action_window_checked": False,
              "branch_count": branch_count, "grading_workers": grading_workers,
              "cold_setup": "all disks copied, then parallel VM boots",
              "repetitions": repetitions, "condition_order": "alternated within pairs",
              "parent_setup_seconds": parent_setup_seconds,
              "parent_disk_clone_mode": parent_disk_clone_mode,
              "expected_rewards": [1.0 if index % 2 == 0 else 0.0
                                   for index in range(branch_count)],
              "all_child_disks_independent": True,
              "all_child_ram_independent": True,
              "runs": runs, "median_cold_seconds": median_cold,
              "median_forked_seconds": median_forked,
              "steady_state_throughput_ratio": median_cold / median_forked,
              "cold_total_seconds": cold_total,
              "fork_total_including_parent_setup_seconds": fork_total,
              "amortized_throughput_ratio_including_parent_setup": cold_total / fork_total,
              "graded_branches_per_second_cold": graded_count / cold_total,
              "graded_branches_per_second_forked_amortized": graded_count / fork_total}
    (output / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


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
        "median_cold_seconds", "median_forked_seconds", "steady_state_throughput_ratio",
        "parent_setup_seconds", "amortized_throughput_ratio_including_parent_setup")}, indent=2))
