"""Paired read/PING transport timing in one already-running QEMU/HVF VM.

This opt-in experiment does not route policy writes, submit, snapshots, or
hidden grading over the virtio port. It measures only a trusted read-only RPC
and keeps the existing serial shell path as the default implementation.
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

from future_prediction_bench.guest_action_rpc import __file__ as GUEST_AGENT
from future_prediction_bench.microvm_runtime import MicroVMRuntime
from future_prediction_bench.virtio_action import VirtioActionClient

from .microvm_benchmark import _boot, _sha


def _required(vm, command, *, timeout=30):
    result = vm.run_shell(command, timeout=timeout)
    if result["return_code"] != 0:
        raise RuntimeError("Guest setup or read command failed: " + result["stdout"][-1000:])
    return result["stdout"]


def _upload_agent(vm):
    source = Path(GUEST_AGENT).read_bytes()
    if len(source) > 100_000:
        raise ValueError("Guest agent exceeds upload bound")
    target = "/mnt/root/fpb_guest_action_rpc.py"
    _required(vm, f": > {target}")
    for offset in range(0, len(source), 1500):
        encoded = base64.b64encode(source[offset:offset + 1500]).decode("ascii")
        _required(vm, f"printf '%s' '{encoded}' | base64 -d >> {target}")
    expected = hashlib.sha256(source).hexdigest()
    actual = _required(vm, f"sha256sum {target}").split()[0]
    if actual != expected:
        raise RuntimeError("Guest agent differs from trusted host source")
    return expected


def _start_agent(vm):
    _required(vm, "test -c /dev/virtio-ports/fpb.control")
    _required(vm, "mkdir -p /mnt/root/dev && { test -c /mnt/root/dev/null || "
             "mknod -m 666 /mnt/root/dev/null c 1 3; }")
    _required(vm, "touch /mnt/root/dev/fpb.control && "
             "mount --bind /dev/virtio-ports/fpb.control /mnt/root/dev/fpb.control")
    output = _required(
        vm, "LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/aarch64-linux-gnu "
        "chroot /mnt/root /usr/local/bin/python3.12 -I -B "
        "/fpb_guest_action_rpc.py >/dev/null 2>&1 & "
        "printf 'FPB_AGENT_PID=%s\\n' \"$!\"")
    match = re.fullmatch(r"FPB_AGENT_PID=([0-9]+)\n?", output)
    if match is None:
        raise RuntimeError("Guest agent PID framing failed")
    return int(match.group(1))


def _shell_read(vm, relative):
    if relative != "boltons/strutils.py":
        raise ValueError("Pinned read benchmark path required")
    guest = "/mnt/root/workspace/boltons/strutils.py"
    result = _required(vm, f"sha256sum {guest}; wc -c < {guest}; "
                       f"head -c 16000 {guest} | base64")
    lines = result.splitlines()
    if len(lines) < 2:
        raise RuntimeError("Serial read result invalid")
    match = re.fullmatch(r"([0-9a-f]{64})\s+.+", lines[0])
    if match is None or not lines[1].strip().isdigit():
        raise RuntimeError("Serial read result invalid")
    size = int(lines[1].strip())
    head = base64.b64decode("".join(lines[2:]), validate=True)
    if len(head) != min(size, 16000):
        raise RuntimeError("Serial read result invalid")
    return {"path": relative, "text": head.decode("utf-8", "replace"),
            "sha256": match.group(1), "truncated": size > 16000}


def _summary(samples):
    ordered = sorted(samples)
    return {"count": len(ordered), "median_ms": round(statistics.median(ordered) * 1000, 6),
            "p95_ms": round(ordered[(95 * len(ordered) + 99) // 100 - 1] * 1000, 6),
            "min_ms": round(ordered[0] * 1000, 6),
            "max_ms": round(ordered[-1] * 1000, 6)}


def benchmark(assets_dir, output, *, repetitions=100, warmup=10):
    assets, output = Path(assets_dir).resolve(), Path(output).resolve()
    if type(repetitions) is not int or not 10 <= repetitions <= 500:
        raise ValueError("repetitions must be 10..500")
    if type(warmup) is not int or not 0 <= warmup <= 100:
        raise ValueError("warmup must be 0..100")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be new or empty")
    manifest = json.loads((assets / "manifest.json").read_text(encoding="utf-8"))
    pristine = assets / "rootfs.qcow2"
    if _sha(pristine) != manifest["rootfs_qcow2_sha256"]:
        raise ValueError("Pinned pristine VM disk changed")
    output.mkdir(parents=True, exist_ok=True)
    disk = output / "transport.qcow2"
    shutil.copy2(pristine, disk)
    vm = MicroVMRuntime(
        assets / "vmlinuz-virt", assets / "initramfs-virt", disk,
        kernel_sha256=manifest["alpine_sha256"]["vmlinuz-virt"],
        initramfs_sha256=manifest["alpine_sha256"]["initramfs-virt"],
        readonly_disk_paths=(assets / "modloop-virt-padded.raw",),
        enable_action_port=True, command_timeout=30)
    started = time.monotonic()
    agent_pid = None
    with vm:
        _boot(vm)
        boot_seconds = time.monotonic() - started
        agent_sha = _upload_agent(vm)
        agent_pid = _start_agent(vm)
        client = VirtioActionClient(vm, timeout=10)
        if not client.ping():
            raise RuntimeError("Guest action port did not answer PING")
        relative = "boltons/strutils.py"
        baseline_read = client.read_file(relative)
        if baseline_read != _shell_read(vm, relative):
            raise RuntimeError("Virtio and serial file observations differ")
        source_sha256 = baseline_read["sha256"]
        ping_shell, ping_virtio, read_shell, read_virtio = [], [], [], []
        pairs_checked = 0
        for index in range(warmup + repetitions):
            order = ("shell", "virtio") if index % 2 == 0 else ("virtio", "shell")
            for condition in order:
                begun = time.monotonic()
                if condition == "shell":
                    _required(vm, "true")
                    elapsed = time.monotonic() - begun
                    if index >= warmup:
                        ping_shell.append(elapsed)
                else:
                    client.ping()
                    elapsed = time.monotonic() - begun
                    if index >= warmup:
                        ping_virtio.append(elapsed)
            pair_values = {}
            for condition in reversed(order):
                begun = time.monotonic()
                if condition == "shell":
                    value = _shell_read(vm, relative)
                    elapsed = time.monotonic() - begun
                    if index >= warmup:
                        read_shell.append(elapsed)
                else:
                    value = client.read_file(relative)
                    elapsed = time.monotonic() - begun
                    if index >= warmup:
                        read_virtio.append(elapsed)
                if value["sha256"] != source_sha256:
                    raise RuntimeError("Pinned source digest changed")
                pair_values[condition] = value
            if pair_values["shell"] != pair_values["virtio"]:
                raise RuntimeError("Virtio and serial read observations differ within pair")
            pairs_checked += 1
        if client.read_file(relative) != _shell_read(vm, relative):
            raise RuntimeError("Virtio and serial reads diverged after timing")
        client.stop()
        _required(vm, f"wait {agent_pid}")
        _required(vm, "umount /mnt/root/dev/fpb.control")
        vm_metrics = vm.get_state()["metrics"]
        rpc_metrics = client.metrics
    report = {"kind": "virtio_action_read_transport_benchmark_v1",
              "scope": "same booted ARM64 QEMU/HVF VM, trusted read-only agent",
              "repetitions": repetitions, "warmup": warmup,
              "pair_order": "alternated; read phase reverses PING order",
              "latency": {"shell_true": _summary(ping_shell), "virtio_ping": _summary(ping_virtio),
                          "shell_read": _summary(read_shell), "virtio_read": _summary(read_virtio)},
              "parity": {"read_observation_equal": True,
                         "complete_observation_pairs_checked": pairs_checked,
                         "measured_pairs_checked": repetitions,
                         "source_sha256": source_sha256},
              "setup": {"boot_seconds": round(boot_seconds, 6),
                        "total_seconds": round(time.monotonic() - started, 6),
                        "vm_metrics": vm_metrics, "rpc_metrics": rpc_metrics},
              "assets": {"kernel_sha256": manifest["alpine_sha256"]["vmlinuz-virt"],
                         "initramfs_sha256": manifest["alpine_sha256"]["initramfs-virt"],
                         "rootfs_seed_sha256": manifest["rootfs_qcow2_sha256"],
                         "agent_sha256": agent_sha},
              "limitations": ["read-only prototype; no write, submit, snapshot, or reward RPC",
                              "shell read timing omits the separate adapter symlink scan",
                              "no policy inference, optimizer update, or whole-episode timing"]}
    (output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    report = benchmark(args.assets_dir, args.output,
                       repetitions=args.repetitions, warmup=args.warmup)
    print(json.dumps({"latency": report["latency"], "setup": {
        "boot_seconds": report["setup"]["boot_seconds"],
        "total_seconds": report["setup"]["total_seconds"]}}, indent=2))
