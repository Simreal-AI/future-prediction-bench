"""Measure QEMU/HVF full-VM savevm and loadvm primitives on pinned assets.

No policy, verifier, or optimizer runs here. The guest is a real ARM64 Linux
VM, and each load must restore both an initramfs RAM marker and an ext4 disk
marker. This is not a DeltaBox implementation or a training-speed result.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from pathlib import Path

from future_prediction_bench.coding_env import _workspace_digest
from future_prediction_bench.microvm_runtime import (
    MicroVMRuntime, _clone_or_copy_qcow2, _sha256_file,
)

from .microvm_benchmark import _boot, _required


_RAM = "/tmp/fpb-primitive-ram"
_DISK = "/mnt/root/workspace/.fpb-primitive-disk"


def _stats_ms(seconds):
    values = sorted(float(value) * 1000 for value in seconds)
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Expected nonnegative finite timing samples")
    return {"count": len(values), "p50_ms": statistics.median(values),
            "p95_ms": values[math.ceil(0.95 * len(values)) - 1],
            "min_ms": values[0], "max_ms": values[-1]}


def _snapshot_vm_size(listing, tag):
    """Return QEMU's own VM_SIZE text and approximate bytes, or raw evidence."""
    for line in listing.splitlines():
        columns = line.split()
        if tag not in columns:
            continue
        index = columns.index(tag)
        if index + 1 >= len(columns):
            continue
        number = columns[index + 1]
        unit = columns[index + 2] if index + 2 < len(columns) else ""
        size = f"{number} {unit}".strip()
        match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([KMGT]?i?B|[KMGT]?B)", size,
                             flags=re.IGNORECASE)
        if match is None:
            match = re.fullmatch(r"(\d+(?:\.\d+)?)([KMGT]?i?B|[KMGT]?B)", number,
                                 flags=re.IGNORECASE)
        if match is None:
            return {"raw_line": line.strip(), "vm_size": size, "vm_size_bytes": None}
        units = {"B": 1, "KB": 1000, "MB": 1000**2, "GB": 1000**3,
                 "TB": 1000**4, "KIB": 1024, "MIB": 1024**2,
                 "GIB": 1024**3, "TIB": 1024**4}
        multiplier = units.get(match.group(2).upper())
        return {"raw_line": line.strip(), "vm_size": match.group(0),
                "vm_size_bytes": int(float(match.group(1)) * multiplier) if multiplier else None}
    raise RuntimeError("QEMU did not list the saved snapshot tag")


def _read_state(vm):
    return _required(vm, f"cat {_RAM}") == "baseline" and _required(vm, f"cat {_DISK}") == "baseline"


def _change_state(vm, index):
    _required(vm, f"printf 'changed-{index}' > {_RAM}; printf 'changed-{index}' > {_DISK}")


def _runtime(assets, manifest, disk, memory_mib):
    return MicroVMRuntime(
        assets / "vmlinuz-virt", assets / "initramfs-virt", disk,
        kernel_sha256=manifest["alpine_sha256"]["vmlinuz-virt"],
        initramfs_sha256=manifest["alpine_sha256"]["initramfs-virt"],
        readonly_disk_paths=(assets / "modloop-virt-padded.raw",),
        memory_mib=memory_mib, command_timeout=30)


def _one_memory(assets, manifest, base, output, memory_mib, *, measured_loads,
                warmup_loads, save_count, monitor_count, keep_disks):
    disk = output / f"memory-{memory_mib}.qcow2"
    clone_mode = _clone_or_copy_qcow2(base, disk)
    event_log = []
    report = {"memory_mib": memory_mib, "status": "error", "disk_clone_mode": clone_mode}
    try:
        with _runtime(assets, manifest, disk, memory_mib) as vm:
            boot_started = time.monotonic()
            _boot(vm)
            boot_seconds = time.monotonic() - boot_started
            original_hmp = vm._hmp

            def timed_hmp(command, *, timeout=None):
                started = time.monotonic()
                try:
                    return original_hmp(command, timeout=timeout)
                finally:
                    event_log.append({"command": command,
                                      "seconds": max(0.0, time.monotonic() - started)})

            vm._hmp = timed_hmp
            for _ in range(3):
                vm._hmp("info status")
            _required(vm, f"printf 'baseline' > {_RAM}; printf 'baseline' > {_DISK}")
            save_samples = []
            public_saves = []
            save_monitor_commands = []
            for index in range(save_count):
                tag = "baseline" if index == 0 else f"baseline{index}"
                before = len(event_log)
                save_started = time.monotonic()
                vm.save_snapshot(tag)
                public_saves.append(time.monotonic() - save_started)
                commands = [event["command"] for event in event_log[before:]]
                if commands != ["info snapshots", f"savevm {tag}", "info snapshots"]:
                    raise RuntimeError("Unexpected current-source save monitor sequence")
                save_monitor_commands.append(commands)
                primitive = [event["seconds"] for event in event_log[before:]
                             if event["command"] == f"savevm {tag}"]
                if len(primitive) != 1:
                    raise RuntimeError("Could not isolate one savevm monitor command")
                save_samples.append(primitive[0])
            snapshot_size = _snapshot_vm_size(vm._hmp("info snapshots"), "baseline")
            if not _read_state(vm):
                raise RuntimeError("Baseline guest state changed before the load loop")

            def load_once(index):
                _change_state(vm, index)
                before = len(event_log)
                started = time.monotonic()
                vm.load_snapshot("baseline")
                public_seconds = time.monotonic() - started
                primitive = [event["seconds"] for event in event_log[before:]
                             if event["command"] == "loadvm baseline"]
                if len(primitive) != 1 or not _read_state(vm):
                    raise RuntimeError("Full-VM load did not restore RAM and disk state")
                return primitive[0], public_seconds

            for index in range(warmup_loads):
                load_once(f"warmup-{index}")
            loads = []
            public_loads = []
            monitor = []
            for index in range(measured_loads):
                before = len(event_log)
                vm._hmp("info status")
                baseline = [event["seconds"] for event in event_log[before:]
                            if event["command"] == "info status"]
                if len(baseline) != 1:
                    raise RuntimeError("Could not isolate monitor roundtrip")
                if index < monitor_count:
                    monitor.append(baseline[0])
                primitive, public_seconds = load_once(index)
                loads.append(primitive)
                public_loads.append(public_seconds)
            report.update({"status": "measured", "boot_seconds": boot_seconds,
                           "warmup_loads": warmup_loads,
                           "correct_state_resets": warmup_loads + measured_loads,
                           "snapshot_vm_size": snapshot_size,
                           "savevm_hmp_seconds": save_samples,
                           "savevm_hmp_stats": _stats_ms(save_samples),
                           "public_save_snapshot_seconds": public_saves,
                           "public_save_snapshot_stats": _stats_ms(public_saves),
                           "save_monitor_commands": save_monitor_commands,
                           "loadvm_hmp_seconds": loads,
                           "loadvm_hmp_stats": _stats_ms(loads),
                           "public_load_snapshot_seconds": public_loads,
                           "public_load_snapshot_stats": _stats_ms(public_loads),
                           "monitor_roundtrip_seconds": monitor,
                           "monitor_roundtrip_stats": _stats_ms(monitor),
                           "vm_metrics": vm.get_state()["metrics"]})
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
    finally:
        if not keep_disks and disk.exists():
            disk.unlink()
    return report


def benchmark(task_dir, assets_dir, output, *, memory_mib=(128, 256, 512),
              measured_loads=20, warmup_loads=3, save_count=2,
              monitor_count=20, keep_disks=False):
    if (not isinstance(memory_mib, (list, tuple)) or not memory_mib
            or len(memory_mib) > 4 or len(set(memory_mib)) != len(memory_mib)
            or any(type(value) is not int or not 128 <= value <= 8192 for value in memory_mib)):
        raise ValueError("memory_mib must be 1..4 distinct integer sizes in [128, 8192]")
    for name, value, minimum, maximum in (("measured_loads", measured_loads, 20, 200),
                                          ("warmup_loads", warmup_loads, 1, 10),
                                          ("save_count", save_count, 1, 3),
                                          ("monitor_count", monitor_count, 20, 200)):
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    if type(keep_disks) is not bool:
        raise ValueError("keep_disks must be boolean")
    if monitor_count > measured_loads:
        raise ValueError("monitor_count cannot exceed measured_loads")
    task_dir, assets, output = map(lambda path: Path(path).resolve(),
                                   (task_dir, assets_dir, output))
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output must be new or empty")
    if output.is_relative_to(assets) or assets.is_relative_to(output):
        raise ValueError("Output and pinned VM assets must be disjoint")
    if output.is_relative_to(task_dir) or task_dir.is_relative_to(output):
        raise ValueError("Output and pinned task inputs must be disjoint")
    manifest = json.loads((assets / "manifest.json").read_text(encoding="utf-8"))
    task = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != "boltons-microvm-assets-v2"
            or manifest.get("task_id") != task.get("task_id")
            or manifest.get("source_sdist_sha256") != task.get("metadata", {}).get("source_sdist_sha256")
            or manifest.get("seed_workspace_sha256") != _workspace_digest(task_dir / "seed")):
        raise ValueError("Expected a pinned Boltons asset manifest v2")
    base = assets / "rootfs.qcow2"
    modloop = assets / "modloop-virt-padded.raw"
    if (_sha256_file(base) != manifest.get("rootfs_qcow2_sha256")
            or _sha256_file(modloop) != manifest.get("modloop_disk_sha256")):
        raise ValueError("Pinned VM disks changed")
    output.mkdir(parents=True, exist_ok=True)
    source_paths = {
        "runtime": Path(__file__).parents[2] / "future_prediction_bench" / "microvm_runtime.py",
        "benchmark": Path(__file__),
    }
    source_sha256 = {name: _sha256_file(path) for name, path in source_paths.items()}
    asset_manifest_sha256 = _sha256_file(assets / "manifest.json")
    results = []
    for size in memory_mib:
        result = _one_memory(assets, manifest, base, output, size,
                             measured_loads=measured_loads,
                             warmup_loads=warmup_loads,
                             save_count=save_count,
                             monitor_count=monitor_count,
                             keep_disks=keep_disks)
        results.append(result)
        if source_sha256 != {name: _sha256_file(path) for name, path in source_paths.items()}:
            raise RuntimeError("Benchmark source changed during QEMU run")
        if _sha256_file(assets / "manifest.json") != asset_manifest_sha256:
            raise RuntimeError("Pinned asset manifest changed during QEMU run")
        print(f"memory={size} MiB status={result['status']}", flush=True)
        (output / "benchmark.json").write_text(json.dumps({
            "scope": "same-host QEMU/HVF full-VM snapshot primitives on pinned Boltons assets",
            "method": "timed current-source save/load wrappers and separate HMP command/monitor roundtrips",
            "source_sha256": source_sha256,
            "asset_binding": {
                "manifest_schema_version": manifest["schema_version"],
                "manifest_sha256": asset_manifest_sha256,
                "rootfs_seed_sha256": manifest["rootfs_qcow2_sha256"],
                "source_sdist_sha256": manifest["source_sdist_sha256"],
                "seed_workspace_sha256": manifest["seed_workspace_sha256"],
            },
            "asset_manifest_sha256": asset_manifest_sha256,
            "source_sdist_sha256": manifest["source_sdist_sha256"],
            "seed_workspace_sha256": manifest["seed_workspace_sha256"],
            "not_training_or_delta_box": True,
            "snapshot_save_count_per_memory": save_count,
            "measured_loads_per_memory": measured_loads,
            "warmup_loads_per_memory": warmup_loads,
            "monitor_roundtrips_per_memory": monitor_count,
            "p95_method": "nearest rank",
            "results": results}, indent=2) + "\n", encoding="utf-8")
    if not any(item["status"] == "measured" for item in results):
        raise RuntimeError("No requested guest memory size completed the primitive benchmark")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--memory-mib", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--measured-loads", type=int, default=20)
    parser.add_argument("--warmup-loads", type=int, default=3)
    parser.add_argument("--save-count", type=int, default=2)
    parser.add_argument("--monitor-count", type=int, default=20)
    parser.add_argument("--keep-disks", action="store_true")
    args = parser.parse_args()
    results = benchmark(args.task_dir, args.assets_dir, args.output,
                        memory_mib=args.memory_mib,
                        measured_loads=args.measured_loads,
                        warmup_loads=args.warmup_loads,
                        save_count=args.save_count,
                        monitor_count=args.monitor_count,
                        keep_disks=args.keep_disks)
    print(json.dumps({item["memory_mib"]: item["loadvm_hmp_stats"]
                      for item in results if item["status"] == "measured"}, indent=2))
