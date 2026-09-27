"""Isolated graded proof: can a trusted guest reopen clear QEMU's stale host bit?

The current production QEMU barrier stays active. This proof varies the
trusted guest hold interval, records QEMU state, and intentionally interrupts
submit before any savevm or reward. No live service or policy action uses the
temporary FD.
"""

from __future__ import annotations

import argparse
import base64
import json
import shlex
import time
from pathlib import Path

from future_prediction_bench.microvm_coding import MicroVMCodingAdapter
from future_prediction_bench.microvm_runtime import (
    MicroVMRuntime, MicroVMRuntimeError, _clone_or_copy_qcow2,
)
from future_prediction_bench.realworld import RealWorldEnv, validate_task

from .benchmark_semantic_recovery import _assets
from .benchmark_virtio_graded import _fixture_task
from .probe_unopened_action_port_snapshot import _chardev_disconnected, _qtree_port


def _required(vm, command):
    result = vm.run_shell(command, timeout=5)
    if result["return_code"]:
        raise RuntimeError("trusted_guest_reopen_command_failed")
    return result["stdout"]


def _trusted_guest_reopen(vm, *, hold_seconds=0.2):
    if not isinstance(hold_seconds, (int, float)) or not 0 <= hold_seconds <= 0.2:
        raise ValueError("Guest reopen hold must be between 0 and 200 ms")
    target = "/mnt/root/dev/fpb.reopen-probe"
    source = (
        "import os,time\n"
        "fd=os.open('/dev/fpb.reopen-probe',"
        "os.O_RDWR|os.O_NONBLOCK|os.O_CLOEXEC|os.O_NOFOLLOW)\n"
        "try:\n"
        " print('FPB_REOPEN_OK')\n"
        f" time.sleep({hold_seconds!r})\n"
        "finally: os.close(fd)\n")
    expression = "exec(__import__('base64').b64decode('" + base64.b64encode(
        source.encode()).decode("ascii") + "'))"
    command = ("LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/aarch64-linux-gnu "
               "chroot /mnt/root /usr/local/bin/python3.12 -I -S -B -c "
               + shlex.quote(expression))
    _required(vm, f"test ! -e {target} && touch {target} && "
                  f"mount --bind /dev/virtio-ports/fpb.control {target}")
    try:
        output = _required(vm, command)
        if output.strip() != "FPB_REOPEN_OK":
            raise RuntimeError("trusted_guest_reopen_output_invalid")
    finally:
        _required(vm, f"umount {target} && rm {target}")
    return {"guest_opened_nonblocking": True, "guest_wrote_bytes": False,
            "guest_read_bytes": False, "guest_fd_closed": True,
            "temporary_bind_removed": True}


class GuestReopenRuntime(MicroVMRuntime):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reopen = None
        self.submitted_snapshot_would_be_allowed = False

    def save_snapshot(self, tag="warm"):
        if tag == "submitted":
            self.submitted_snapshot_would_be_allowed = (
                self._action_port_disconnect_attested
                and self.action_port_qemu_state() == {
                    "guest": "off", "host": "off", "chardev_disconnected": True})
            raise MicroVMRuntimeError("proof_intentionally_blocks_submitted_snapshot")
        return super().save_snapshot(tag)


class GuestReopenAdapter(MicroVMCodingAdapter):
    HOLD_SECONDS = 0.2

    def _trusted_guest_reopen_for_qemu_disconnect(self):
        runtime = self.runtime
        before = {"device_state": _qtree_port(runtime),
                  "chardev_disconnected": _chardev_disconnected(runtime)}
        started = time.monotonic()
        try:
            guest = _trusted_guest_reopen(runtime, hold_seconds=self.HOLD_SECONDS)
            after = {"device_state": _qtree_port(runtime),
                     "chardev_disconnected": _chardev_disconnected(runtime)}
            runtime.reopen = {"guest_probe": guest,
                              "seconds": time.monotonic() - started,
                              "before": before, "after": after,
                              "qemu_disconnected_after_reopen":
                                  after["device_state"]["guest"] == "off"
                                  and after["device_state"]["host"] == "off"
                                  and after["chardev_disconnected"]}
        except Exception as probe_exc:
            runtime.reopen = {"before": before,
                              "probe_failure_type": type(probe_exc).__name__,
                              "probe_failure_reason": str(probe_exc)[:120],
                              "qemu_disconnected_after_reopen": False}
            raise
        finally:
            self.metrics["virtio_agent_disconnect_probe_seconds"] += (
                time.monotonic() - started)


def probe(task_dir, assets_dir, output_dir, *, hold_seconds=0.2):
    task_dir, assets, output = (Path(value).resolve() for value in
                                (task_dir, assets_dir, output_dir))
    if ((output.exists() and (not output.is_dir() or any(output.iterdir())))
            or output.is_relative_to(task_dir) or output.is_relative_to(assets)
            or task_dir.is_relative_to(output) or assets.is_relative_to(output)):
        raise ValueError("Output must be new, empty, and disjoint")
    task_source, manifest, actions = _assets(task_dir, assets)
    if (manifest.get("image_variant") != "preinstalled_virtio_read_agent_v1"
            or manifest.get("preinstalled_guest_action_agent_path")
               != "/fpb_guest_action_rpc.py"):
        raise ValueError("Expected pinned preinstalled virtio agent image")
    output.mkdir(parents=True, exist_ok=True)
    disk = output / "guest-reopen.qcow2"
    clone_mode = _clone_or_copy_qcow2(assets / "rootfs.qcow2", disk)
    runtime = GuestReopenRuntime(
        assets / "vmlinuz-virt", assets / "initramfs-virt", disk,
        kernel_sha256=manifest["alpine_sha256"]["vmlinuz-virt"],
        initramfs_sha256=manifest["alpine_sha256"]["initramfs-virt"],
        readonly_disk_paths=(assets / "modloop-virt-padded.raw",),
        memory_mib=128, command_timeout=30, enable_action_port=True)
    adapter = GuestReopenAdapter(
        runtime, verifier_dir=task_dir / "verifier",
        visible_check=("python3", "-B", "-c", "import boltons.strutils"),
        read_transport="virtio_serial_readonly_v1", preinstalled_read_agent=True)
    adapter.HOLD_SECONDS = hold_seconds
    task = _fixture_task(task_source)
    task.setdefault("metadata", {})["artifact_binding"] = adapter.artifact_binding()
    validate_task(task)
    env = RealWorldEnv(task, adapter)
    report = {"kind": "graded_guest_reopen_qemu_disconnect_probe_v1",
              "task_id": manifest["task_id"],
              "rootfs_seed_sha256": manifest["rootfs_qcow2_sha256"],
              "clone_mode": clone_mode, "hold_seconds": hold_seconds,
              "status": "incomplete"}
    try:
        env.reset("guest-reopen-probe")
        statuses = []
        for action in actions:
            transition = env.step(action)
            statuses.append({"action": action["action"],
                             "status": transition["info"]["status"]})
            if transition["info"]["status"] == "interrupted":
                break
        report["action_statuses"] = statuses
        report["reopen"] = runtime.reopen
        report["snapshot_saves"] = runtime.metrics["snapshot_saves"]
        report["submitted_snapshot_would_be_allowed"] = runtime.submitted_snapshot_would_be_allowed
        report["checkpoint_operations_supported"] = runtime.get_state()[
            "checkpoint_operations_supported"]
        report["guest_process_baseline_restored"] = (
            adapter._scan_guest_processes() == adapter._virtio_process_baseline)
        report["guest_action_port_fd_holders"] = len(adapter._scan_guest_action_port_holders())
        report["agent_stop_acknowledged"] = bool(
            adapter._virtio_client and adapter._virtio_client.stopped)
        report["old_host_client_fd_closed"] = bool(
            adapter._virtio_client and adapter._virtio_client.socket.fileno() == -1)
        if runtime.reopen is None:
            report["status"] = "reopen_not_attempted"
        elif runtime.reopen["qemu_disconnected_after_reopen"]:
            report["status"] = "guest_reopen_cleared_qemu_host_state_proof_only"
        else:
            report["status"] = "guest_reopen_did_not_clear_qemu_host_state"
        if (report["snapshot_saves"] != 0
                or not report["guest_process_baseline_restored"]
                or report["guest_action_port_fd_holders"] != 0):
            raise RuntimeError("proof_isolation_or_quiescence_failed")
        if (report["status"] == "guest_reopen_cleared_qemu_host_state_proof_only"
                and (not report["checkpoint_operations_supported"]
                     or not report["submitted_snapshot_would_be_allowed"])):
            raise RuntimeError("proof_qemu_attestation_missing")
    except Exception as exc:
        report["status"] = "failed"
        report["failure_type"] = type(exc).__name__
        report["failure_reason"] = str(exc)[:120]
        raise
    finally:
        adapter.close()
        disk.unlink(missing_ok=True)
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n",
                                            encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hold-ms", type=float, default=200.0)
    args = parser.parse_args()
    result = probe(args.task_dir, args.assets_dir, args.output,
                   hold_seconds=args.hold_ms / 1000.0)
    print(json.dumps({key: result[key] for key in
                      ("status", "reopen", "snapshot_saves", "checkpoint_operations_supported")},
                     indent=2))


if __name__ == "__main__":
    main()
