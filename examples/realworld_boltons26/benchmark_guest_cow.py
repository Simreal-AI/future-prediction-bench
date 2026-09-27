"""Measure fork + overlayfs branch primitives inside an already-running VM.

The host uploads a trusted self-contained Python experiment to the pinned
Boltons ARM64 guest, then captures its guest-clock timings. No policy shell or
hidden verifier is involved. Reported primitive milliseconds exclude VM boot,
script upload, model inference, and optimizer updates; host overhead is saved
separately in the JSON report.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import shutil
import time
from pathlib import Path

from future_prediction_bench.guest_cow_branch import __file__ as GUEST_SCRIPT
from future_prediction_bench.http import strict_json_loads

from .microvm_benchmark import _boot, _runtime, _sha


def _required(vm, command: str, *, timeout=30):
    result = vm.run_shell(command, timeout=timeout)
    if result["return_code"] != 0:
        raise RuntimeError("Guest command failed: " + result["stdout"][-2000:])
    return result["stdout"]


def _upload_guest_program(vm):
    source = Path(GUEST_SCRIPT).read_bytes()
    if len(source) > 100_000:
        raise ValueError("Guest branch program exceeds upload bound")
    target = "/mnt/root/fpb_guest_cow_branch.py"
    _required(vm, f": > {target}")
    for offset in range(0, len(source), 1500):
        encoded = base64.b64encode(source[offset:offset + 1500]).decode("ascii")
        _required(vm, f"printf '%s' '{encoded}' | base64 -d >> {target}")
    actual = _required(vm, f"sha256sum {target}").split()[0]
    import hashlib
    expected = hashlib.sha256(source).hexdigest()
    if actual != expected:
        raise RuntimeError("Guest branch program digest differs from trusted host source")
    return expected


def _parse_guest_report(stdout: str, repetitions: int):
    matches = re.findall(r"FPB_COW_RESULT:([A-Za-z0-9+/=]+)", stdout)
    if len(matches) != 1:
        raise RuntimeError("Guest COW report framing failed")
    try:
        decoded = base64.b64decode(matches[0], validate=True)
        report = strict_json_loads(decoded.decode("utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise RuntimeError("Guest COW report is invalid") from exc
    if (not isinstance(report, dict)
            or report.get("kind") != "guest_linux_fork_overlayfs_microbenchmark_v1"
            or report.get("repetitions") != repetitions
            or not isinstance(report.get("timings"), dict)
            or not isinstance(report.get("correctness"), dict)):
        raise RuntimeError("Guest COW report contract differs")
    correctness = report["correctness"]
    if (correctness.get("parent_glass") != "glas"
            or correctness.get("fix_branch", {}).get("glass") != "glass"
            or correctness.get("baseline_branch", {}).get("glass") != "glas"
            or correctness.get("parent_counter") != 7
            or correctness.get("shared_lower_sha256_unchanged") is not True
            or correctness.get("isolated_overlay_writes") is not True):
        raise RuntimeError("Guest branch isolation or repair correctness failed")
    for key in ("overlay_create", "fork_ready", "process_reap",
                "overlay_rollback", "rollback_total", "combined_branch_cycle"):
        item = report["timings"].get(key)
        if (not isinstance(item, dict) or item.get("count") != repetitions
                or type(item.get("median_ms")) not in (int, float)
                or not 0 <= item["median_ms"] <= 300_000):
            raise RuntimeError("Guest COW timing report is invalid")
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
    disk = output / "guest-cow.qcow2"
    shutil.copy2(pristine, disk)
    started = time.monotonic()
    with _runtime(assets, disk) as vm:
        _boot(vm)
        boot_seconds = time.monotonic() - started
        _required(vm, "modprobe overlay")
        uploaded_sha = _upload_guest_program(vm)
        guest_started = time.monotonic()
        stdout = _required(
            vm,
            "LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/aarch64-linux-gnu "
            "PYTHONPATH=/workspace chroot /mnt/root "
            f"/usr/local/bin/python3.12 -B /fpb_guest_cow_branch.py {repetitions}",
            timeout=300)
        guest_call_seconds = time.monotonic() - guest_started
        guest = _parse_guest_report(stdout, repetitions)
        vm_metrics = vm.get_state()["metrics"]
    report = {"kind": "host_observed_guest_cow_benchmark_v1",
              "guest": guest,
              "host": {"boot_seconds": round(boot_seconds, 6),
                       "guest_call_seconds": round(guest_call_seconds, 6),
                       "total_seconds": round(time.monotonic() - started, 6),
                       "vm_metrics": vm_metrics},
              "assets": {"kernel_sha256": manifest["alpine_sha256"]["vmlinuz-virt"],
                         "initramfs_sha256": manifest["alpine_sha256"]["initramfs-virt"],
                         "rootfs_seed_sha256": manifest["rootfs_qcow2_sha256"],
                         "guest_program_sha256": uploaded_sha},
              "limits": ["trusted guest-local branches share one Linux kernel",
                         "not an adversarial policy sandbox",
                         "not a full VM checkpoint or restore",
                         "no model inference or optimizer training measured"]}
    (output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
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
