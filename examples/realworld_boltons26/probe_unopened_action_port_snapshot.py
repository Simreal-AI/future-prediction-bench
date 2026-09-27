"""Experimental QEMU savevm/loadvm probe with an unopened virtio port.

This deliberately bypasses MicroVMRuntime's production snapshot guard only
through direct HMP calls in this stand-alone proof. The guest never opens
fpb.control, no RPC agent runs, and the host does not connect action.sock
until *after* loadvm for a negative no-agent RPC check. It does not test or
authorize checkpointing a connected or live guest action service.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import select
import shlex
import socket
import stat
import time
from pathlib import Path

from future_prediction_bench.microvm_runtime import (
    MicroVMRuntime, MicroVMRuntimeError, _clone_or_copy_qcow2, _sha256_file,
)
from future_prediction_bench.virtio_action import VirtioActionClient, VirtioActionError

from .microvm_benchmark import _boot


TAG = "unopened_probe"
RAM_MARKER = "/tmp/fpb-unopened-port-ram"
DISK_MARKER = "/mnt/root/workspace/.fpb-unopened-port-disk"


class DeferredActionPortRuntime(MicroVMRuntime):
    """Proof-only runtime: leave QEMU's action listener unconnected."""

    def _connect(self, path, deadline):
        if Path(path).name == "action.sock":
            return None
        return super()._connect(path, deadline)


def _required(vm, command):
    result = vm.run_shell(command, timeout=30)
    if result["return_code"]:
        raise RuntimeError("trusted_guest_probe_failed")
    return result["stdout"]


def _qtree_port(vm):
    lines = vm._hmp("info qtree").splitlines()
    blocks = []
    for index, line in enumerate(lines):
        if "dev: virtserialport" in line:
            block = "\n".join(lines[index:index + 6])
            if 'chardev = "fpbctl"' in block and 'name = "fpb.control"' in block:
                blocks.append(block)
    if len(blocks) != 1:
        raise RuntimeError("virtserialport_identity_ambiguous")
    match = re.search(r"port ([0-9]+), guest (on|off), host (on|off), "
                      r"throttle (on|off)", blocks[0])
    if match is None:
        raise RuntimeError("virtserialport_state_unavailable")
    return {"port_number": int(match.group(1)), "guest": match.group(2),
            "host": match.group(3), "throttle": match.group(4),
            "name": "fpb.control", "chardev": "fpbctl"}


def _chardev_disconnected(vm):
    lines = vm._hmp("info chardev").splitlines()
    matches = [line for line in lines if line.startswith("fpbctl: filename=")]
    if len(matches) != 1:
        raise RuntimeError("action_chardev_identity_ambiguous")
    return matches[0].startswith("fpbctl: filename=disconnected:unix:")


def _guest_port_census(vm):
    source = (
        "import base64,json,os\n"
        "holders=[]\n"
        "agents=[]\n"
        "for name in os.listdir('/proc'):\n"
        " if not name.isdecimal(): continue\n"
        " try: cmdline=open('/proc/'+name+'/cmdline','rb').read()\n"
        " except OSError: cmdline=b''\n"
        " if b'guest_action_rpc.py' in cmdline: agents.append(int(name))\n"
        " try: fds=os.listdir('/proc/'+name+'/fd')\n"
        " except OSError: continue\n"
        " for fd in fds:\n"
        "  try: target=os.readlink('/proc/'+name+'/fd/'+fd)\n"
        "  except OSError: continue\n"
        "  if 'fpb.control' in target: holders.append([int(name),int(fd)])\n"
        "print('FPB_CENSUS='+base64.b64encode(json.dumps({'holders':sorted(holders),'agents':sorted(agents)}).encode()).decode())")
    encoded = base64.b64encode(source.encode()).decode()
    command = ("LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/aarch64-linux-gnu "
               "chroot /mnt/root /usr/local/bin/python3.12 -I -S -B -c "
               + shlex.quote(f"exec(__import__('base64').b64decode('{encoded}'))"))
    _required(vm, "mkdir -p /mnt/root/proc && mount -t proc proc /mnt/root/proc")
    try:
        output = _required(vm, command)
    finally:
        _required(vm, "umount /mnt/root/proc")
    match = re.fullmatch(r"FPB_CENSUS=([A-Za-z0-9+/=]+)\n?", output)
    if match is None:
        raise RuntimeError("action_fd_census_invalid")
    value = json.loads(base64.b64decode(match.group(1), validate=True))
    if (not isinstance(value, dict) or set(value) != {"holders", "agents"}
            or not isinstance(value["holders"], list) or len(value["holders"]) > 1024
            or not isinstance(value["agents"], list) or len(value["agents"]) > 1024
            or any(type(pid) is not int or pid < 0 for pid in value["agents"])
            or any(not isinstance(item, list) or len(item) != 2
                   or any(type(number) is not int or number < 0 for number in item)
                   for item in value["holders"])):
        raise RuntimeError("action_fd_census_invalid")
    return value


def _guard_still_closed(vm):
    for operation in (lambda: vm.save_snapshot("guard"),
                      lambda: vm.load_snapshot("guard")):
        try:
            operation()
        except MicroVMRuntimeError as exc:
            if str(exc) != "vm_action_port_snapshot_not_supported":
                raise
        else:
            raise RuntimeError("production_snapshot_guard_opened")
    return True


def _assert_unopened(state):
    if (state != {"port_number": 1, "guest": "off", "host": "off",
                  "throttle": "off", "name": "fpb.control", "chardev": "fpbctl"}):
        raise RuntimeError("unopened_port_device_state_changed")


def _after_load_no_agent_rpc(vm, action_path):
    """Fresh host connection cannot silently resume a nonexistent session."""
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        connection.connect(str(action_path))
        deadline = time.monotonic() + 2.0
        while True:
            connected = _qtree_port(vm)
            if connected["host"] == "on":
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("new_host_connection_not_visible_to_qemu")
            time.sleep(0.02)
        if connected["guest"] != "off":
            raise RuntimeError("guest_port_opened_without_agent")
        readable, _, _ = select.select([connection], [], [], 0.1)
        if readable:
            initial = connection.recv(1, socket.MSG_PEEK)
            if initial:
                raise RuntimeError("unexpected_initial_action_port_bytes")
            raise RuntimeError("fresh_action_port_socket_closed_before_rpc")

        class FreshSocketProvider:
            def action_port_socket(self):
                return connection

        client = VirtioActionClient(FreshSocketProvider(), timeout=0.2)
        initial_sequence = client.sequence
        fresh_session = bool(re.fullmatch(r"[0-9a-f]{32}", client.session))
        try:
            client.ping()
        except VirtioActionError as exc:
            failed_reason = str(exc)
        else:
            raise RuntimeError("unopened_guest_responded_to_rpc")
        if not client.failed or not fresh_session or initial_sequence != 0:
            raise RuntimeError("fresh_rpc_session_did_not_fail_closed")
        closed_at = time.monotonic()
        post_failure_trace = [{"after_client_close_seconds": 0.0,
                               "device_state": _qtree_port(vm),
                               "chardev_disconnected": _chardev_disconnected(vm)}]
        deadline = time.monotonic() + 2.0
        while True:
            disconnected = _qtree_port(vm)
            if disconnected["host"] == "off":
                break
            if time.monotonic() >= deadline:
                break
            time.sleep(0.02)
        chardev_disconnected = _chardev_disconnected(vm)
        post_failure_trace.append({"after_client_close_seconds": time.monotonic() - closed_at,
                                   "device_state": disconnected,
                                   "chardev_disconnected": chardev_disconnected})
        return {"fresh_host_connection_observed": True,
                "guest_stayed_unopened": True,
                "fresh_session_nonce_generated": True,
                "no_initial_bytes_before_rpc": True,
                "initial_sequence_zero": True,
                "ping_failed_reason": failed_reason,
                "rpc_client_failed_closed": True,
                "host_disconnected_after_failure": disconnected["host"] == "off",
                "post_failure_device_state": disconnected,
                "chardev_disconnected_after_failure": chardev_disconnected,
                "post_failure_trace": post_failure_trace,
                "saved_session_epoch_present_at_snapshot": False,
                "saved_session_epoch_restored": False}
    finally:
        connection.close()


def validate_evidence(report):
    """Reject a positive claim unless every independent gate is proven."""
    if (report.get("status") != "supported_only_when_unopened"
            or not report.get("production_guard_closed_before_after")
            or report.get("guest_agent_running") is not False
            or report.get("guest_port_fd_holders_before") != 0
            or report.get("guest_port_fd_holders_after") != 0
            or report.get("guest_agent_processes_before") != 0
            or report.get("guest_agent_processes_after") != 0
            or report.get("ram_restored") is not True
            or report.get("workspace_restored") is not True
            or report.get("host_listener_inode_preserved") is not True
            or report.get("host_chardev_disconnected_before_after") is not True
            or report.get("post_load_rpc", {}).get("rpc_client_failed_closed") is not True
            or report.get("post_load_rpc", {}).get("no_initial_bytes_before_rpc") is not True
            or report.get("post_load_rpc", {}).get("host_disconnected_after_failure") is not True
            or report.get("post_load_rpc", {}).get("saved_session_epoch_restored") is not False):
        raise ValueError("unopened_port_snapshot_evidence_incomplete")
    _assert_unopened(report["device_state_before"])
    _assert_unopened(report["device_state_after"])
    return True


def probe(assets_dir, output_dir):
    assets, output = Path(assets_dir).resolve(), Path(output_dir).resolve()
    if ((output.exists() and (not output.is_dir() or any(output.iterdir())))
            or output.is_relative_to(assets) or assets.is_relative_to(output)):
        raise ValueError("Output must be new, empty, and disjoint from assets")
    manifest_path = assets / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("Pinned asset manifest cannot be a symlink")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != "boltons-microvm-assets-v2"
            or _sha256_file(assets / "rootfs.qcow2")
               != manifest.get("rootfs_qcow2_sha256")
            or _sha256_file(assets / "modloop-virt-padded.raw")
               != manifest.get("modloop_disk_sha256")):
        raise ValueError("Pinned QEMU assets changed")
    output.mkdir(parents=True, exist_ok=True)
    disk = output / "unopened-port.qcow2"
    clone_mode = _clone_or_copy_qcow2(assets / "rootfs.qcow2", disk)
    vm = DeferredActionPortRuntime(
        assets / "vmlinuz-virt", assets / "initramfs-virt", disk,
        kernel_sha256=manifest["alpine_sha256"]["vmlinuz-virt"],
        initramfs_sha256=manifest["alpine_sha256"]["initramfs-virt"],
        readonly_disk_paths=(assets / "modloop-virt-padded.raw",),
        enable_action_port=True, command_timeout=30)
    report = {"kind": "qemu_unopened_virtio_port_snapshot_probe_v1",
              "status": "incomplete", "task_id": manifest["task_id"],
              "rootfs_seed_sha256": manifest["rootfs_qcow2_sha256"],
              "guest_agent_running": False,
              "clone_mode": clone_mode,
              "production_runtime_modified": False,
              "experimental_bypass": "direct_HMP_savevm_loadvm_only"}
    try:
        with vm:
            _boot(vm)
            action_path = vm._socket_dir / "action.sock"
            listener_before = action_path.stat()
            if (not stat.S_ISSOCK(listener_before.st_mode)
                    or vm._action_port is not None):
                raise RuntimeError("action_listener_not_deferred")
            _guard_still_closed(vm)
            state_before = _qtree_port(vm)
            _assert_unopened(state_before)
            if not _chardev_disconnected(vm):
                raise RuntimeError("action_chardev_connected_before_save")
            census_before = _guest_port_census(vm)
            if census_before["holders"] or census_before["agents"]:
                raise RuntimeError("guest_action_port_open_before_save")
            node_before = _required(vm, "stat -Lc '%F:%t:%T' /dev/virtio-ports/fpb.control")
            if not node_before.startswith("character special file:"):
                raise RuntimeError("guest_action_port_node_invalid")
            _required(vm, f"printf frozen > {RAM_MARKER} && printf frozen > {DISK_MARKER}")
            if _required(vm, f"cat {RAM_MARKER} {DISK_MARKER}") != "frozenfrozen":
                raise RuntimeError("snapshot_markers_not_set")
            save_started = time.monotonic()
            vm._hmp("savevm " + TAG, timeout=30)
            save_seconds = time.monotonic() - save_started
            listing = vm._hmp("info snapshots")
            if not re.search(rf"(?m)^\s*\S+\s+{TAG}(?:\s|$)", listing):
                raise RuntimeError("qemu_snapshot_tag_not_published")
            _required(vm, f"printf mutated > {RAM_MARKER} && printf mutated > {DISK_MARKER}")
            if _required(vm, f"cat {RAM_MARKER} {DISK_MARKER}") != "mutatedmutated":
                raise RuntimeError("post_save_mutation_not_observed")
            load_started = time.monotonic()
            vm._hmp("loadvm " + TAG, timeout=30)
            load_seconds = time.monotonic() - load_started
            vm._serial_buffer = b""  # The proof owns HMP loadvm, not the runtime API.
            status = vm._hmp("info status")
            if "VM status: paused" in status:
                vm._hmp("cont")
            elif "VM status: running" not in status:
                raise RuntimeError("post_load_vm_status_ambiguous")
            state_after = _qtree_port(vm)
            _assert_unopened(state_after)
            if not _chardev_disconnected(vm):
                raise RuntimeError("action_chardev_connected_after_load")
            listener_after = action_path.stat()
            if (not stat.S_ISSOCK(listener_after.st_mode)
                    or listener_after.st_ino != listener_before.st_ino
                    or vm._action_port is not None):
                raise RuntimeError("host_action_listener_changed_after_load")
            node_after = _required(vm, "stat -Lc '%F:%t:%T' /dev/virtio-ports/fpb.control")
            ram_restored = _required(vm, f"cat {RAM_MARKER}") == "frozen"
            workspace_restored = _required(vm, f"cat {DISK_MARKER}") == "frozen"
            if node_after != node_before or not ram_restored or not workspace_restored:
                raise RuntimeError("vm_ram_disk_or_device_not_restored")
            census_after = _guest_port_census(vm)
            if census_after["holders"] or census_after["agents"]:
                raise RuntimeError("guest_action_port_open_after_load")
            _guard_still_closed(vm)
            post_load_rpc = _after_load_no_agent_rpc(vm, action_path)
            report.update({
                "status": ("supported_only_when_unopened" if post_load_rpc["host_disconnected_after_failure"]
                           else "partial_save_load_host_cleanup_failed"),
                "device_state_before": state_before,
                "device_state_after": state_after,
                "guest_port_node_type_major_minor_preserved": True,
                "guest_port_fd_holders_before": 0,
                "guest_port_fd_holders_after": 0,
                "guest_agent_processes_before": 0,
                "guest_agent_processes_after": 0,
                "host_listener_inode_preserved": True,
                "host_chardev_disconnected_before_after": True,
                "ram_restored": ram_restored,
                "workspace_restored": workspace_restored,
                "production_guard_closed_before_after": True,
                "hmp_savevm_seconds": save_seconds,
                "hmp_loadvm_seconds": load_seconds,
                "post_load_rpc": post_load_rpc,
                "scope_limit": "no guest open, no host action connection at snapshot, no live RPC session or epoch restoration claim",
            })
            if report["status"] == "supported_only_when_unopened":
                validate_evidence(report)
    except Exception as exc:
        report["status"] = "failed"
        report["failure_type"] = type(exc).__name__
        report["failure_reason"] = str(exc)[:200]
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n",
                                            encoding="utf-8")
        raise
    finally:
        disk.unlink(missing_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n",
                                        encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = probe(args.assets_dir, args.output)
    print(json.dumps({"status": result["status"],
                      "hmp_savevm_seconds": result["hmp_savevm_seconds"],
                      "hmp_loadvm_seconds": result["hmp_loadvm_seconds"],
                      "post_load_rpc": result["post_load_rpc"]}, indent=2))


if __name__ == "__main__":
    main()
