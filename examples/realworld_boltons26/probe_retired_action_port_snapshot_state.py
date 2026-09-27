"""Inspect QEMU's action-port state after the real graded STOP/retire sequence.

This proof-only runtime intercepts the submitted savevm boundary. It never
allows a snapshot while QEMU still reports a connected action backend.
Production runtime and adapter behavior are not modified by this module.
"""

from __future__ import annotations

import argparse
import json
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


class RetirementBoundaryRuntime(MicroVMRuntime):
    """Gate this single experiment on QEMU state, not the Python FD alone."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.retired_at = None
        self.boundary = None

    def retire_action_port(self, client):
        try:
            super().retire_action_port(client)
        except MicroVMRuntimeError as exc:
            self.retired_at = time.monotonic()
            if str(exc) == "vm_action_port_qemu_not_disconnected":
                first = {"seconds_after_socket_close": 0.0,
                         "device_state": _qtree_port(self),
                         "chardev_disconnected": _chardev_disconnected(self)}
                time.sleep(2.0)
                second = {"seconds_after_socket_close": time.monotonic() - self.retired_at,
                          "device_state": _qtree_port(self),
                          "chardev_disconnected": _chardev_disconnected(self)}
                self.boundary = {"samples": [first, second],
                                 "python_host_socket_closed": self._action_port is None,
                                 "runtime_transport_retired": self._action_port_retired,
                                 "qemu_disconnect_attested": self._action_port_disconnect_attested,
                                 "barrier_error": str(exc),
                                 "qemu_disconnected_before_snapshot": False}
            raise
        self.retired_at = time.monotonic()

    def save_snapshot(self, tag="warm"):
        if tag == "submitted" and self.retired_at is not None:
            first = {"seconds_after_socket_close": time.monotonic() - self.retired_at,
                     "device_state": _qtree_port(self),
                     "chardev_disconnected": _chardev_disconnected(self)}
            samples = [first]
            if first["device_state"]["host"] != "off" or not first["chardev_disconnected"]:
                time.sleep(2.0)
                samples.append({"seconds_after_socket_close": time.monotonic() - self.retired_at,
                                "device_state": _qtree_port(self),
                                "chardev_disconnected": _chardev_disconnected(self)})
            self.boundary = {"samples": samples,
                             "python_host_socket_closed": self._action_port is None,
                             "runtime_transport_retired": self._action_port_retired,
                             "qemu_disconnected_before_snapshot":
                                 (samples[-1]["device_state"]["host"] == "off"
                                  and samples[-1]["chardev_disconnected"])}
            if not self.boundary["qemu_disconnected_before_snapshot"]:
                raise MicroVMRuntimeError("proof_qemu_host_port_still_connected")
        return super().save_snapshot(tag)


def probe(task_dir, assets_dir, output_dir):
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
    disk = output / "retired-port.qcow2"
    clone_mode = _clone_or_copy_qcow2(assets / "rootfs.qcow2", disk)
    runtime = RetirementBoundaryRuntime(
        assets / "vmlinuz-virt", assets / "initramfs-virt", disk,
        kernel_sha256=manifest["alpine_sha256"]["vmlinuz-virt"],
        initramfs_sha256=manifest["alpine_sha256"]["initramfs-virt"],
        readonly_disk_paths=(assets / "modloop-virt-padded.raw",),
        memory_mib=128, command_timeout=30, enable_action_port=True)
    adapter = MicroVMCodingAdapter(
        runtime, verifier_dir=task_dir / "verifier",
        visible_check=("python3", "-B", "-c", "import boltons.strutils"),
        read_transport="virtio_serial_readonly_v1", preinstalled_read_agent=True)
    task = _fixture_task(task_source)
    task.setdefault("metadata", {})["artifact_binding"] = adapter.artifact_binding()
    validate_task(task)
    env = RealWorldEnv(task, adapter)
    report = {"kind": "graded_retired_action_port_qemu_state_probe_v1",
              "task_id": manifest["task_id"],
              "rootfs_seed_sha256": manifest["rootfs_qcow2_sha256"],
              "clone_mode": clone_mode,
              "production_runtime_modified": False,
              "status": "incomplete"}
    try:
        env.reset("retirement-boundary-probe")
        results = []
        for action in actions:
            transition = env.step(action)
            results.append({"action": action["action"],
                            "status": transition["info"]["status"]})
            if transition["info"]["status"] == "interrupted":
                break
        report["action_statuses"] = results
        report["boundary"] = runtime.boundary
        report["agent_stop_acknowledged"] = bool(
            adapter._virtio_client and adapter._virtio_client.stopped)
        report["host_client_fd_closed"] = bool(
            adapter._virtio_client and adapter._virtio_client.socket.fileno() == -1)
        report["guest_port_unmounted"] = (
            runtime.run_shell("test ! -e /mnt/root/dev/fpb.control", timeout=30)["return_code"] == 0)
        report["guest_process_baseline_restored"] = (
            adapter._scan_guest_processes() == adapter._virtio_process_baseline)
        report["guest_action_port_fd_holders"] = len(adapter._scan_guest_action_port_holders())
        report["snapshot_saves"] = runtime.metrics["snapshot_saves"]
        report["checkpoint_operations_supported"] = runtime.get_state()[
            "checkpoint_operations_supported"]
        if runtime.boundary is None:
            report["status"] = "retirement_sequence_incomplete"
        elif runtime.boundary.get("barrier_error") == "vm_action_port_qemu_not_disconnected":
            report["status"] = "retirement_barrier_blocked_qemu_host_still_connected"
        elif runtime.boundary["qemu_disconnected_before_snapshot"]:
            report["status"] = "qemu_disconnected_before_snapshot"
        else:
            report["status"] = "snapshot_blocked_qemu_host_still_connected"
        if not all((report["agent_stop_acknowledged"], report["host_client_fd_closed"],
                    report["guest_port_unmounted"], report["guest_process_baseline_restored"],
                    report["guest_action_port_fd_holders"] == 0)):
            raise RuntimeError("graded_retirement_preconditions_not_proven")
    except Exception as exc:
        report["status"] = "failed"
        report["failure_type"] = type(exc).__name__
        report["failure_reason"] = str(exc)[:200]
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
    args = parser.parse_args()
    report = probe(args.task_dir, args.assets_dir, args.output)
    print(json.dumps({key: report[key] for key in
                      ("status", "action_statuses", "boundary",
                       "agent_stop_acknowledged", "host_client_fd_closed")}, indent=2))


if __name__ == "__main__":
    main()
