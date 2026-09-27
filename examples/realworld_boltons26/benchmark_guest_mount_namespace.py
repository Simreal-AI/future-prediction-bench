"""Measure private mount-namespace plus tmpfs/overlay branches inside one VM.

The trusted host uploads a fixed Python program to the pinned ARM64 guest.
Guest-clock operation timings exclude QEMU boot, transport, model inference,
and optimizer updates. This is a trusted process experiment, not an agent
policy or a secure untrusted code sandbox.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import shutil
import time
from pathlib import Path

from future_prediction_bench.guest_mount_namespace import __file__ as GUEST_SCRIPT
from future_prediction_bench.http import strict_json_loads

from .benchmark_guest_cow import _required
from .microvm_benchmark import _boot, _runtime, _sha


def _upload(vm):
    source = Path(GUEST_SCRIPT).read_bytes()
    if len(source) > 100_000:
        raise ValueError("Guest namespace program exceeds upload bound")
    target = "/mnt/root/fpb_guest_mount_namespace.py"
    _required(vm, f": > {target}")
    for offset in range(0, len(source), 1500):
        encoded = base64.b64encode(source[offset:offset + 1500]).decode("ascii")
        _required(vm, f"printf '%s' '{encoded}' | base64 -d >> {target}")
    expected = hashlib.sha256(source).hexdigest()
    actual = _required(vm, f"sha256sum {target}").split()[0]
    if actual != expected:
        raise RuntimeError("Uploaded namespace program hash differs")
    return expected


def _parse_report(stdout: str, repetitions: int):
    matches = re.findall(r"FPB_MOUNT_NS_RESULT:([A-Za-z0-9+/=]+)", stdout)
    if len(matches) != 1:
        raise RuntimeError("Guest namespace report framing failed")
    try:
        decoded = base64.b64decode(matches[0], validate=True)
        report = strict_json_loads(decoded.decode("utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise RuntimeError("Guest namespace report is invalid") from exc
    if (not isinstance(report, dict)
            or report.get("kind") != "guest_linux_mount_namespace_overlayfs_microbenchmark_v1"
            or report.get("repetitions") != repetitions
            or not isinstance(report.get("correctness"), dict)
            or not isinstance(report.get("timings"), dict)):
        raise RuntimeError("Guest namespace report contract differs")
    check = report["correctness"]
    if (check.get("fix_branch", {}).get("glass") != "glass"
            or check.get("baseline_branch", {}).get("glass") != "glas"
            or check.get("distinct_child_mount_namespaces") is not True
            or check.get("parent_view_empty") is not True
            or check.get("shared_lower_sha256_unchanged") is not True
            or check.get("parent_heap_unchanged") is not True):
        raise RuntimeError("Guest namespace isolation or repair checks failed")
    names = [check.get("parent_namespace"),
             check["fix_branch"].get("mount_namespace"),
             check["baseline_branch"].get("mount_namespace")]
    devices = [check.get("parent_upper_device"),
               check["fix_branch"].get("upper_device"),
               check["baseline_branch"].get("upper_device")]
    if (any(not isinstance(name, str) or not re.fullmatch(r"mnt:\[\d+\]", name)
            for name in names) or len(set(names)) != 3
            or any(type(device) is not int for device in devices)
            or len(set(devices)) != 3):
        raise RuntimeError("Guest namespace identifiers or devices are not distinct")
    for name in ("unshare_ns", "private_propagation_ns", "tmpfs_mount_ns",
                 "overlay_mount_ns", "cleanup_mounts_ns", "fork_to_ready_ns",
                 "release_to_reap_ns", "complete_cycle_ns"):
        timing = report["timings"].get(name)
        if (not isinstance(timing, dict) or timing.get("count") != repetitions
                or type(timing.get("median_ms")) not in (int, float)
                or type(timing.get("p95_ms")) not in (int, float)
                or not 0 <= timing["median_ms"] <= timing["p95_ms"] <= 300_000):
            raise RuntimeError("Guest namespace timing report is invalid")
    return report


def benchmark(assets_dir, output, *, repetitions=100):
    assets, output = Path(assets_dir).resolve(), Path(output).resolve()
    if type(repetitions) is not int or not 10 <= repetitions <= 500:
        raise ValueError("repetitions must be 10..500")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be new or empty")
    manifest = strict_json_loads((assets / "manifest.json").read_text(encoding="utf-8"))
    pristine = assets / "rootfs.qcow2"
    if _sha(pristine) != manifest["rootfs_qcow2_sha256"]:
        raise ValueError("Pinned pristine VM disk changed")
    output.mkdir(parents=True, exist_ok=True)
    disk = output / "guest-mount-namespace.qcow2"
    shutil.copy2(pristine, disk)
    started = time.monotonic()
    with _runtime(assets, disk) as vm:
        _boot(vm)
        boot_seconds = time.monotonic() - started
        _required(vm, "modprobe overlay")
        _required(vm, "mkdir -p /mnt/root/proc && mount -t proc proc /mnt/root/proc")
        program_sha = _upload(vm)
        guest_started = time.monotonic()
        stdout = _required(
            vm,
            "LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/aarch64-linux-gnu "
            "PYTHONPATH=/workspace chroot /mnt/root "
            f"/usr/local/bin/python3.12 -B /fpb_guest_mount_namespace.py {repetitions}",
            timeout=300)
        call_seconds = time.monotonic() - guest_started
        guest = _parse_report(stdout, repetitions)
        vm_metrics = vm.get_state()["metrics"]
    report = {"kind": "host_observed_guest_mount_namespace_benchmark_v1",
              "guest": guest,
              "host": {"boot_seconds": round(boot_seconds, 6),
                       "guest_call_seconds": round(call_seconds, 6),
                       "total_seconds": round(time.monotonic() - started, 6),
                       "vm_metrics": vm_metrics},
              "assets": {"kernel_sha256": manifest["alpine_sha256"]["vmlinuz-virt"],
                         "initramfs_sha256": manifest["alpine_sha256"]["initramfs-virt"],
                         "rootfs_seed_sha256": manifest["rootfs_qcow2_sha256"],
                         "task_id": manifest["task_id"],
                         "seed_workspace_sha256": manifest["seed_workspace_sha256"],
                         "source_sdist_sha256": manifest["source_sdist_sha256"],
                         "guest_program_sha256": program_sha},
              "limits": ["trusted sibling processes share one guest Linux kernel",
                         "root authority in the guest can inspect or affect siblings",
                         "not an adversarial policy sandbox or full VM checkpoint",
                         "no model inference or optimizer training measured"]}
    (output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n",
                                             encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repetitions", type=int, default=100)
    args = parser.parse_args()
    result = benchmark(args.assets_dir, args.output, repetitions=args.repetitions)
    print(json.dumps({"host": result["host"], "guest_timings": result["guest"]["timings"]},
                     indent=2))
