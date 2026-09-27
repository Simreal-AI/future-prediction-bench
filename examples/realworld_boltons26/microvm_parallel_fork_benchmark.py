"""Compare serial and parallel restoration of independent warm QEMU/HVF VMs.

This is a pinned, scripted Boltons infrastructure experiment. It measures
child startup, complete fork setup, and host-private graded branch batches;
it does not measure policy inference or an RL optimizer step.
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

from .microvm_benchmark import _boot, _required, _runtime, _sha
from .microvm_fork_benchmark import _grade_branch, _value, _verify_isolation


def _validated_inputs(task_dir, assets_dir, output):
    task_dir, assets, output = map(lambda path: Path(path).resolve(),
                                   (task_dir, assets_dir, output))
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output must be new or empty")
    if output.is_relative_to(assets) or assets.is_relative_to(output):
        raise ValueError("Output and immutable assets must be disjoint")
    verifier_dir = task_dir / "verifier"
    if output.is_relative_to(verifier_dir) or verifier_dir.is_relative_to(output):
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
    verifier = verifier_dir / "verify.json"
    if verifier.is_symlink():
        raise ValueError("Host-private verifier cannot be a symlink")
    specification = json.loads(verifier.read_text(encoding="utf-8"))
    if (specification.get("kind") != "command_cases_v1"
            or not isinstance(specification.get("cases"), list)
            or len(specification["cases"]) != 14):
        raise ValueError("Expected the pinned 14-case Boltons verifier")
    return task_dir, assets, output, manifest, base, verifier, specification["cases"]


def benchmark(task_dir, assets_dir, output, *, repetitions=3, branch_count=2,
              grading_workers=2, keep_disks=False):
    """Run alternating serial/parallel warm-fork pairs on one live parent VM."""
    if type(repetitions) is not int or not 1 <= repetitions <= 5:
        raise ValueError("repetitions must be an integer in [1, 5]")
    if type(branch_count) is not int or not 2 <= branch_count <= 4:
        raise ValueError("branch_count must be an integer in [2, 4]")
    if type(grading_workers) is not int or not 1 <= grading_workers <= branch_count:
        raise ValueError("grading_workers must be in [1, branch_count]")
    if type(keep_disks) is not bool:
        raise ValueError("keep_disks must be boolean")
    (task_dir, assets, output, manifest, base, verifier, cases) = _validated_inputs(
        task_dir, assets_dir, output)
    if not callable(getattr(MicroVMRuntime, "fork_snapshot", None)):
        raise RuntimeError("This MicroVMRuntime does not implement fork_snapshot")
    verifier_sha256 = _sha(verifier)
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
            order = ("fork_serial", "fork_parallel") if repetition % 2 == 0 else (
                "fork_parallel", "fork_serial")
            for condition in order:
                parallel = condition == "fork_parallel"
                disk_paths = [output / f"{condition}-r{repetition}-b{index}.qcow2"
                              for index in range(branch_count)]
                children = []
                started = time.monotonic()
                try:
                    before = parent.get_state()["metrics"]
                    fork_started = time.monotonic()
                    children = parent.fork_snapshot("warm", disk_paths,
                                                    parallel_children=parallel)
                    fork_call_seconds = time.monotonic() - fork_started
                    after = parent.get_state()["metrics"]
                    if len(children) != branch_count:
                        raise RuntimeError("MicroVM fork returned the wrong sibling count")
                    if any(_required(child, "cat /tmp/fpb-parent-ram") != "parent-warm-state"
                           for child in children):
                        raise RuntimeError("Forked child did not inherit parent RAM state")
                    setup_seconds = time.monotonic() - started
                    if len({str(child.disk_path) for child in children}) != branch_count:
                        raise RuntimeError("Sibling VMs did not receive unique writable disk paths")
                    clone_mode_counts = {
                        "clonefile": after["fork_reflink_disks"] - before["fork_reflink_disks"],
                        "copy": after["fork_copied_disks"] - before["fork_copied_disks"]}
                    child_start_restore_seconds = (
                        after["fork_child_start_restore_seconds"]
                        - before["fork_child_start_restore_seconds"])
                    disk_clone_seconds = (after["fork_disk_clone_seconds"]
                                          - before["fork_disk_clone_seconds"])
                    isolation_started = time.monotonic()
                    isolation = _verify_isolation(parent, children, condition=condition,
                                                  repetition=repetition)
                    isolation_seconds = time.monotonic() - isolation_started
                    grading_started = time.monotonic()
                    with ThreadPoolExecutor(max_workers=grading_workers) as pool:
                        grades = list(pool.map(lambda pair: _grade_branch(pair[1], cases, pair[0]),
                                               enumerate(children)))
                    grading_seconds = time.monotonic() - grading_started
                    if (_value(parent) != "glas"
                            or _required(parent, "cat /tmp/fpb-parent-ram") != "parent-warm-state"):
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
                wall_seconds = time.monotonic() - started
                disk_details = [{"name": path.name, "bytes": path.stat().st_size,
                                 "sha256": _sha(path)} for path in disk_paths]
                if not keep_disks:
                    for path in disk_paths:
                        path.unlink()
                runs.append({"condition": condition, "repetition": repetition,
                             "wall_seconds": wall_seconds, "setup_seconds": setup_seconds,
                             "fork_call_seconds": fork_call_seconds,
                             "child_start_restore_seconds": child_start_restore_seconds,
                             "disk_clone_seconds": disk_clone_seconds,
                             "isolation_seconds": isolation_seconds,
                             "grading_seconds": grading_seconds,
                             "disk_clone_mode_counts": clone_mode_counts,
                             "rewards": [item["reward"] for item in grades],
                             "branches": grades, "isolation": isolation,
                             "child_disks": disk_details})
                print(f"{condition} rep={repetition} setup={setup_seconds:.3f}s "
                      f"graded_wall={wall_seconds:.3f}s", flush=True)
    if not keep_disks:
        parent_disk.unlink()

    serial = [item for item in runs if item["condition"] == "fork_serial"]
    parallel = [item for item in runs if item["condition"] == "fork_parallel"]
    median = lambda samples, field: statistics.median(item[field] for item in samples)
    serial_total = sum(item["wall_seconds"] for item in serial)
    parallel_total = sum(item["wall_seconds"] for item in parallel)
    result = {
        "scope": "pinned public Boltons 26.0.0 repair fixture on one QEMU/HVF host",
        "runtime": "qemu_hvf_full_vm_qcow2_v1",
        "asset_schema_version": manifest["schema_version"],
        "host_private_case_count": len(cases),
        "host_private_verifier_sha256": verifier_sha256,
        "measures_model_training": False,
        "uses_realworld_env": False,
        "policy_action_window_checked": False,
        "branch_count": branch_count, "grading_workers": grading_workers,
        "repetitions": repetitions, "condition_order": "alternated within pairs",
        "parent_setup_seconds": parent_setup_seconds,
        "parent_disk_clone_mode": parent_disk_clone_mode,
        "expected_rewards": [1.0 if index % 2 == 0 else 0.0
                             for index in range(branch_count)],
        "all_child_disks_independent": True,
        "all_child_ram_independent": True,
        "runs": runs,
        "median_serial_child_start_restore_seconds": median(serial, "child_start_restore_seconds"),
        "median_parallel_child_start_restore_seconds": median(parallel, "child_start_restore_seconds"),
        "child_start_restore_ratio": (median(serial, "child_start_restore_seconds")
                                      / median(parallel, "child_start_restore_seconds")),
        "median_serial_setup_seconds": median(serial, "setup_seconds"),
        "median_parallel_setup_seconds": median(parallel, "setup_seconds"),
        "setup_ratio": median(serial, "setup_seconds") / median(parallel, "setup_seconds"),
        "median_serial_batch_seconds": median(serial, "wall_seconds"),
        "median_parallel_batch_seconds": median(parallel, "wall_seconds"),
        "graded_batch_throughput_ratio": (median(serial, "wall_seconds")
                                          / median(parallel, "wall_seconds")),
        "serial_total_seconds": serial_total,
        "parallel_total_seconds": parallel_total,
        "serial_total_including_parent_setup_seconds": parent_setup_seconds + serial_total,
        "parallel_total_including_parent_setup_seconds": parent_setup_seconds + parallel_total,
        "amortized_graded_batch_throughput_ratio": (
            (parent_setup_seconds + serial_total) / (parent_setup_seconds + parallel_total)),
    }
    (output / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n",
                                            encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--branch-count", type=int, default=2)
    parser.add_argument("--grading-workers", type=int, default=2)
    parser.add_argument("--keep-disks", action="store_true")
    args = parser.parse_args()
    result = benchmark(args.task_dir, args.assets_dir, args.output,
                       repetitions=args.repetitions, branch_count=args.branch_count,
                       grading_workers=args.grading_workers, keep_disks=args.keep_disks)
    print(json.dumps({key: result[key] for key in (
        "median_serial_child_start_restore_seconds",
        "median_parallel_child_start_restore_seconds", "child_start_restore_ratio",
        "median_serial_setup_seconds", "median_parallel_setup_seconds", "setup_ratio",
        "median_serial_batch_seconds", "median_parallel_batch_seconds",
        "graded_batch_throughput_ratio",
        "amortized_graded_batch_throughput_ratio")}, indent=2))
