"""Matched strict-vs-cached QEMU full-VM restore A/B on pinned Boltons assets.

This tests a *transport* optimization, not DeltaBox's DeltaFS/DeltaCR. Every
restore must reconstruct an ext4 file, a tmpfs file, and a live shell process's
in-memory variable. Timings exclude mutation and validation commands but
include all work in MicroVMRuntime.load_snapshot.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import platform
import statistics
import subprocess
import time
from pathlib import Path

from examples.realworld_boltons26.benchmark_vm_primitives import _runtime
from examples.realworld_boltons26.microvm_benchmark import _boot, _required
from future_prediction_bench.coding_env import _workspace_digest
from future_prediction_bench.microvm_runtime import _clone_or_copy_qcow2, _sha256_file


RAM = "/tmp/fpb-fast-restore-ram"
DISK = "/mnt/root/workspace/.fpb-fast-restore-disk"
OBS = "/tmp/fpb-fast-restore-observed"
WORKER = "/tmp/fpb-fast-restore-worker.sh"
WORKER_SOURCE = '''#!/bin/sh
value=baseline
trap 'value=changed' USR2
trap 'printf "%s" "$value" > /tmp/fpb-fast-restore-observed' USR1
while :; do sleep 0.05; done
'''


def _stats(samples):
    ordered = sorted(samples)
    if not ordered or any(not math.isfinite(value) or value < 0 for value in ordered):
        raise ValueError("invalid samples")
    return {"count": len(ordered), "p50_ms": statistics.median(ordered) * 1000,
            "p95_ms": ordered[math.ceil(.95 * len(ordered)) - 1] * 1000,
            "min_ms": ordered[0] * 1000, "max_ms": ordered[-1] * 1000}


def _check_assets(task_dir, assets_dir):
    task = json.loads((task_dir / "task.json").read_text())
    manifest_bytes = (assets_dir / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if (manifest.get("schema_version") != "boltons-microvm-assets-v2"
            or manifest.get("task_id") != task.get("task_id")
            or manifest.get("source_sdist_sha256") !=
            task.get("metadata", {}).get("source_sdist_sha256")
            or manifest.get("seed_workspace_sha256") !=
            _workspace_digest(task_dir / "seed")):
        raise ValueError("pinned Boltons task/asset binding mismatch")
    for name, digest in (("rootfs.qcow2", manifest["rootfs_qcow2_sha256"]),
                         ("modloop-virt-padded.raw", manifest["modloop_disk_sha256"]),
                         ("vmlinuz-virt", manifest["alpine_sha256"]["vmlinuz-virt"]),
                         ("initramfs-virt", manifest["alpine_sha256"]["initramfs-virt"])):
        if _sha256_file(assets_dir / name) != digest:
            raise ValueError("pinned VM asset changed: " + name)
    return task, manifest, hashlib.sha256(manifest_bytes).hexdigest()


def _observe_worker(vm, pid, expected):
    _required(vm, f"kill -USR1 {pid}")
    output = None
    for _ in range(10):
        output = _required(vm, f"sleep 0.05; cat {OBS}")
        if output == expected:
            return
    raise RuntimeError(f"live process variable: expected {expected}, got {output!r}")


def _run(task_dir, assets_dir, output_dir, pairs):
    task_dir, assets_dir, output_dir = (Path(path).resolve() for path in
                                      (task_dir, assets_dir, output_dir))
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError("output directory must be new or empty")
    if output_dir.is_relative_to(task_dir) or output_dir.is_relative_to(assets_dir):
        raise ValueError("output cannot contain task or asset inputs")
    task, manifest, manifest_sha = _check_assets(task_dir, assets_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    disk = output_dir / "clone.qcow2"
    clone_start = time.monotonic()
    clone_mode = _clone_or_copy_qcow2(assets_dir / "rootfs.qcow2", disk)
    clone_seconds = time.monotonic() - clone_start
    report = {"schema_version": "fpb-qemu-cached-restore-ab-v1",
              "scope": "QEMU/HVF full-VM restore transport A/B, not DeltaBox",
              "task_id": task["task_id"], "asset_manifest_sha256": manifest_sha,
              "asset_schema": manifest["schema_version"],
              "host_architecture": platform.machine(), "memory_mib": 128,
              "pairs": pairs, "clone_mode": clone_mode,
              "clone_seconds": clone_seconds, "samples": [],
              "cold_boot_seconds": None,
              "excludes": "task policy, model, verifier, mutation and post-restore validation"}
    try:
        with _runtime(assets_dir, manifest, disk, 128) as vm:
            started = time.monotonic()
            _boot(vm)
            report["cold_boot_seconds"] = time.monotonic() - started
            encoded = base64.b64encode(WORKER_SOURCE.encode()).decode()
            _required(vm, f"printf '%s' '{encoded}' | base64 -d > {WORKER}; chmod 700 {WORKER}")
            pid_text = _required(vm, f"sh {WORKER} >/dev/null 2>&1 & echo $!")
            if not pid_text.strip().isdigit():
                raise RuntimeError("worker pid missing")
            pid = int(pid_text.strip())
            _observe_worker(vm, pid, "baseline")
            _required(vm, f"printf baseline > {RAM}; printf baseline > {DISK}")
            vm.save_snapshot("baseline")
            original_hmp = vm._hmp
            hmp_events = []
            def timed_hmp(command, *, timeout=None):
                start = time.monotonic()
                try:
                    return original_hmp(command, timeout=timeout)
                finally:
                    hmp_events.append({"command": command,
                                       "seconds": time.monotonic() - start})
            vm._hmp = timed_hmp
            for index in range(pairs + 2):
                # Two warm-up pairs, then AB/BA to balance cache/order drift.
                order = ("strict", "cached") if index % 2 == 0 else ("cached", "strict")
                for mode in order:
                    _required(vm, f"kill -USR2 {pid}; sleep 0.10; printf changed > {RAM}; "
                                  f"printf changed > {DISK}; sync")
                    _observe_worker(vm, pid, "changed")
                    before = len(hmp_events)
                    started = time.monotonic()
                    vm.load_snapshot("baseline", full_validation=(mode == "strict"))
                    wall = time.monotonic() - started
                    events = hmp_events[before:]
                    loads = [item["seconds"] for item in events
                             if item["command"] == "loadvm baseline"]
                    if len(loads) != 1:
                        raise RuntimeError("missing one actual QEMU loadvm")
                    commands = [item["command"] for item in events]
                    expected_commands = (["info snapshots", "loadvm baseline"]
                                         if mode == "strict" else ["loadvm baseline"])
                    if commands != expected_commands:
                        raise RuntimeError("unexpected monitor protocol: " + repr(commands))
                    if _required(vm, f"cat {RAM}; cat {DISK}") != "baselinebaseline":
                        raise RuntimeError("RAM or ext4 was not restored")
                    _observe_worker(vm, pid, "baseline")
                    if index >= 2:
                        report["samples"].append({"pair": index - 2, "mode": mode,
                                                  "wall_seconds": wall,
                                                  "hmp_loadvm_seconds": loads[0],
                                                  "ram_disk_process_restored": True})
        grouped = {mode: [item["wall_seconds"] for item in report["samples"]
                         if item["mode"] == mode] for mode in ("strict", "cached")}
        report["wall_stats"] = {mode: _stats(grouped[mode]) for mode in grouped}
        report["hmp_stats"] = {mode: _stats([item["hmp_loadvm_seconds"]
                                           for item in report["samples"]
                                           if item["mode"] == mode])
                               for mode in grouped}
        report["p50_speedup"] = (report["wall_stats"]["strict"]["p50_ms"] /
                                 report["wall_stats"]["cached"]["p50_ms"])
        report["all_state_checks_passed"] = len(report["samples"]) == 2 * pairs
        report["status"] = "measured"
    except Exception as exc:
        report["status"] = "error"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        raise
    finally:
        (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
        disk.unlink(missing_ok=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pairs", type=int, default=20)
    args = parser.parse_args()
    if not 5 <= args.pairs <= 100:
        parser.error("--pairs must be between 5 and 100")
    result = _run(args.task_dir, args.assets_dir, args.output_dir, args.pairs)
    print(json.dumps({key: result[key] for key in
                      ("status", "wall_stats", "hmp_stats", "p50_speedup")},
                     indent=2))


if __name__ == "__main__":
    main()
