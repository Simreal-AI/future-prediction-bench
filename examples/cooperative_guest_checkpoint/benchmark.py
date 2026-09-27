"""Measure a cooperative coupled checkpoint in the pinned ARM64 QEMU guest.

The guest receives only verifier program strings. Expected outputs and all
reward calculations stay on the trusted host. This is a reproducible public
research example, not a general runtime or adversarial policy sandbox.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import shutil
import statistics
import time
from pathlib import Path

from examples.realworld_boltons26.microvm_benchmark import _boot, _runtime
from future_prediction_bench.coding_env import _workspace_digest


HERE = Path(__file__).resolve().parent
GUEST_PROGRAM = HERE / "guest_coupled.py"
PINNED_TASK = "boltons-26-singularize-ss-v2"
PINNED_SOURCE_SHA = "f7f4873406d3913372c9d2b1296cc5e3efb87e88808457e93fc212a5df2de18e"
PINNED_VERIFIER_SHA = "2fbdc59f999b489b13052102d9b9b31b0e136ec9271242870084fb3d63b5eb58"


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_fixture(task, assets):
    task_json = json.loads((task / "task.json").read_text(encoding="utf-8"))
    manifest = json.loads((assets / "manifest.json").read_text(encoding="utf-8"))
    verifier_path = task / "verifier" / "verify.json"
    verifier = json.loads(verifier_path.read_text(encoding="utf-8"))
    cases = verifier.get("cases")
    if (task_json.get("task_id") != PINNED_TASK
            or manifest.get("schema_version") != "boltons-microvm-assets-v2"
            or manifest.get("task_id") != PINNED_TASK
            or manifest.get("seed_workspace_sha256") != _workspace_digest(task / "seed")
            or _sha(assets / "rootfs.qcow2") != manifest.get("rootfs_qcow2_sha256")
            or _sha(task / "seed/boltons/strutils.py") != PINNED_SOURCE_SHA
            or _sha(verifier_path) != PINNED_VERIFIER_SHA
            or verifier.get("kind") != "command_cases_v1"
            or not isinstance(cases, list) or len(cases) != 14):
        raise ValueError("pinned task, verifier, or VM assets changed")
    codes = []
    for case in cases:
        argv = case.get("argv")
        if (not isinstance(argv, list) or len(argv) != 4
                or argv[:3] != ["python3", "-B", "-c"]
                or type(argv[3]) is not str or not 0 < len(argv[3]) <= 2048
                or type(case.get("expected_returncode")) is not int
                or type(case.get("expected_stdout")) is not str):
            raise ValueError("pinned case contract changed")
        codes.append(argv[3])
    return manifest, cases, codes


def _required(vm, command, *, timeout=30):
    result = vm.run_shell(command, timeout=timeout)
    if result["return_code"] != 0:
        raise RuntimeError("guest command failed: " + result["stdout"][-500:])
    return result["stdout"]


def _upload(vm, payload, target):
    if not 0 < len(payload) <= 100_000:
        raise ValueError("guest payload too large")
    _required(vm, ": > " + target)
    for offset in range(0, len(payload), 1500):
        chunk = base64.b64encode(payload[offset:offset + 1500]).decode("ascii")
        _required(vm, "printf '%s' '" + chunk + "' | base64 -d >> " + target)
    if _required(vm, "sha256sum " + target).split()[0] != hashlib.sha256(payload).hexdigest():
        raise RuntimeError("guest payload digest mismatch")
    return hashlib.sha256(payload).hexdigest()


def _stats(values):
    if not values:
        raise ValueError("missing samples")
    ordered = sorted(values)
    return {"n": len(values), "p50_ms": round(statistics.median(ordered) * 1000, 6),
            "p95_ms": round(ordered[(95 * len(ordered) + 99) // 100 - 1] * 1000, 6),
            "min_ms": round(ordered[0] * 1000, 6),
            "max_ms": round(ordered[-1] * 1000, 6)}


def _decode_guest(stdout, cycles, restores):
    matches = re.findall(r"FPB_COUPLED_RESULT:([A-Za-z0-9+/=]+)", stdout)
    if len(matches) != 1:
        raise RuntimeError("missing or repeated guest report frame")
    payload = json.loads(base64.b64decode(matches[0], validate=True))
    if (payload.get("kind") != "cooperative_coupled_fork_overlay_checkpoint_v1"
            or payload.get("cycles") != cycles
            or payload.get("restores_per_cycle") != restores
            or payload.get("seed_source_sha256") != PINNED_SOURCE_SHA
            or len(payload.get("checkpoint_ns", [])) != cycles
            or len(payload.get("restore_ns", [])) != cycles * restores
            or len(payload.get("no_case_branch_cycle_ns", [])) != cycles * restores - 2
            or len(payload.get("cycles_data", [])) != cycles
            or payload.get("outer_ext4_source_unchanged") is not True
            or payload.get("outer_ext4_tree_unchanged") is not True
            or not re.fullmatch(r"[0-9a-f]{64}",
                                payload.get("outer_ext4_tree_sha256", ""))):
        raise RuntimeError("guest checkpoint report contract changed")
    for cycle_index, row in enumerate(payload["cycles_data"]):
        if (row.get("parent_turn_after_checkpoint") != 999
                or row.get("template_turn_after_restores") != 17
                or row.get("frozen_prefix_unchanged") is not True
                or row.get("frozen_source_unchanged") is not True
                or not re.fullmatch(r"mnt:\[[0-9]+\]",
                                    row.get("template_mount_namespace", ""))
                or len(row.get("branch_cycle_ns", [])) != restores
                or len(row.get("branches", [])) != restores):
            raise RuntimeError("checkpoint process or filesystem parity failed")
        for index, branch in enumerate(row["branches"]):
            mode = "repair" if index % 2 == 0 else "baseline"
            if (branch.get("mode") != mode
                    or branch.get("inherited_turn") != 17
                    or branch.get("local_turn") != (111 if mode == "repair" else 222)
                    or branch.get("prefix_seen") is not True
                    or branch.get("sibling_markers_absent") is not True
                    or not re.fullmatch(r"mnt:\[[0-9]+\]",
                                        branch.get("mount_namespace", ""))
                    or branch["mount_namespace"] == row["template_mount_namespace"]
                    or (cycle_index != 0 or index >= 2) and "case_results" in branch):
                raise RuntimeError("sibling branch isolation failed")
    return payload


def _grade_and_redact(guest, cases):
    results = []
    for index in (0, 1):
        branch = guest["cycles_data"][0]["branches"][index]
        actual = branch.pop("case_results")
        if not isinstance(actual, list) or len(actual) != len(cases):
            raise RuntimeError("guest did not execute all verifier cases")
        passed = 0
        for result, case in zip(actual, cases):
            value = base64.b64decode(result["stdout_b64"], validate=True)
            if (result["return_code"] == case["expected_returncode"]
                    and value == case["expected_stdout"].encode("utf-8")):
                passed += 1
        results.append({"mode": branch["mode"], "case_count": len(cases),
                        "passed_cases": passed,
                        "reward": 1.0 if passed == len(cases) else 0.0})
    if results != [{"mode": "repair", "case_count": 14,
                    "passed_cases": 14, "reward": 1.0},
                   {"mode": "baseline", "case_count": 14,
                    "passed_cases": 7, "reward": 0.0}]:
        raise RuntimeError("pinned repaired/baseline reward parity failed")
    return results


def benchmark(task_dir, assets_dir, output, *, cycles=10, restores=10,
              vm_loads=30):
    task, assets, output = map(lambda item: Path(item).resolve(),
                               (task_dir, assets_dir, output))
    if not 2 <= cycles <= 30 or not 2 <= restores <= 100 or not 5 <= vm_loads <= 100:
        raise ValueError("cycles=2..30, restores=2..100, vm_loads=5..100")
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be new or empty")
    manifest, cases, codes = _check_fixture(task, assets)
    output.mkdir(parents=True, exist_ok=True)
    disk = output / "child.qcow2"
    shutil.copy2(assets / "rootfs.qcow2", disk)
    started = time.monotonic()
    with _runtime(assets, disk) as vm:
        _boot(vm)
        boot_seconds = time.monotonic() - started
        _required(vm, "modprobe overlay")
        helper_sha = _upload(vm, GUEST_PROGRAM.read_bytes(),
                             "/mnt/root/fpb_coupled_guest.py")
        case_payload = json.dumps(codes, separators=(",", ":")).encode("utf-8")
        case_sha = _upload(vm, case_payload, "/mnt/root/fpb_coupled_cases.json")
        _required(vm, "printf 'checkpointed' > /tmp/fpb-coupled-ram-marker")
        _required(vm, "printf 'checkpointed' > /mnt/root/fpb-coupled-disk-marker")
        vm.save_snapshot("coupled_fullvm_baseline")
        call_started = time.monotonic()
        stdout = _required(
            vm,
            "LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/aarch64-linux-gnu "
            "chroot /mnt/root /usr/local/bin/python3.12 -I -B "
            f"/fpb_coupled_guest.py {cycles} {restores}", timeout=300)
        guest_call_seconds = time.monotonic() - call_started
        guest = _decode_guest(stdout, cycles, restores)
        grade = _grade_and_redact(guest, cases)
        _required(vm, "printf 'diverged' > /tmp/fpb-coupled-ram-marker")
        _required(vm, "printf 'diverged' > /mnt/root/fpb-coupled-disk-marker")
        load_seconds = []
        for _ in range(vm_loads):
            begin = time.monotonic()
            vm.load_snapshot("coupled_fullvm_baseline")
            load_seconds.append(time.monotonic() - begin)
            if (_required(vm, "cat /tmp/fpb-coupled-ram-marker") != "checkpointed"
                    or _required(vm, "cat /mnt/root/fpb-coupled-disk-marker") != "checkpointed"):
                raise RuntimeError("full VM RAM or ext4 restore parity failed")
            _required(vm, "printf 'diverged' > /tmp/fpb-coupled-ram-marker")
            _required(vm, "printf 'diverged' > /mnt/root/fpb-coupled-disk-marker")
        vm_metrics = vm.get_state()["metrics"]
    report = {"kind": "cooperative_coupled_checkpoint_qemu_benchmark_v1",
              "scope": "one pinned Boltons v2 ARM64 QEMU/HVF task, one booted VM",
              "guest": guest, "host_private_grading": grade,
              "host": {"boot_seconds": round(boot_seconds, 6),
                       "guest_call_seconds": round(guest_call_seconds, 6),
                       "total_seconds": round(time.monotonic() - started, 6),
                       "full_vm_restore_seconds": load_seconds,
                       "full_vm_restore_stats": _stats(load_seconds),
                       "full_vm_ram_ext4_restore_checks": vm_loads,
                       "vm_metrics": vm_metrics},
              "assets": {"rootfs_seed_sha256": manifest["rootfs_qcow2_sha256"],
                         "kernel_sha256": manifest["alpine_sha256"]["vmlinuz-virt"],
                         "initramfs_sha256": manifest["alpine_sha256"]["initramfs-virt"],
                         "verifier_sha256": PINNED_VERIFIER_SHA,
                         "guest_program_sha256": helper_sha,
                         "case_program_payload_sha256": case_sha},
              "limits": ["cooperative one-process template only; no arbitrary writable FDs or background processes",
                         "guest branches share one VM and Linux kernel; untrusted-root security is not provided",
                         "guest checkpoint includes remount-readonly plus process fork, but no CRIU artifact or durable crash recovery",
                         "guest restore includes child fork and nested overlay mount; full VM loadvm restores CPU/RAM/device/ext4 and is a different contract",
                         "guest-clock restore-to-ready timings exclude branch actions, verifier execution, cleanup, VM boot, serial transfer, and model/trainer work; no-case branch-cycle timings include cleanup"]}
    serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if any(marker in serialized for marker in ("/Users/", "/private/tmp/", "expected_stdout", "stdout_b64")):
        raise RuntimeError("report contains path or private verifier output")
    (output / "report.json").write_text(serialized, encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--restores", type=int, default=10)
    parser.add_argument("--vm-loads", type=int, default=30)
    args = parser.parse_args()
    report = benchmark(args.task_dir, args.assets_dir, args.output,
                       cycles=args.cycles, restores=args.restores,
                       vm_loads=args.vm_loads)
    print(json.dumps({"checkpoint": report["guest"]["checkpoint_stats"],
                      "restore": report["guest"]["restore_stats"],
                      "no_case_branch_cycle": report["guest"]["no_case_branch_cycle_stats"],
                      "full_vm_restore": report["host"]["full_vm_restore_stats"],
                      "grade": report["host_private_grading"]}, indent=2))


if __name__ == "__main__":
    main()
