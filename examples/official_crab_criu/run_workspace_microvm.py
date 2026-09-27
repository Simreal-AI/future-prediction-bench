"""Run actual upstream Crab RAM + ZFS recovery in one disposable x86 TCG VM.

Use ``python -m examples.official_crab_criu.run_workspace_microvm``. No network
device, host directory share, host kernel change, or host pool is provided.
Failed disks and per-attempt raw CRIU logs are retained as evidence.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import stat
import time

from examples.official_crab_criu.check_chain import (
    CRAB_COMMIT, CRAB_PYTHON_SHA256, INTEGRATIONS_PYTHON_SHA256)
from examples.official_crab_criu.workspace_driver import GUEST_SOURCES, MODES, sha
from future_prediction_bench import microvm_runtime
from future_prediction_bench.microvm_runtime import MicroVMRuntime, _clone_or_copy_qcow2

RELEASE = "6.18.53-0-virt"
KERNEL_SHA256 = "3b6e001d41938fdf4fab7d87826fc40dfab733973a5ef96a14c892dbdec87502"
CONFIG_SHA256 = "453545679d2852e55c8a219b43b1386d46991f5c267454045604d91a35f80d89"
SNAPSHOT_PHASES = ("before_live_damage", "after_live_damage",
                   "after_negative_process_restore_and_delete")
FD_KEYS = ("fd", "target", "offset", "inode", "device", "size", "namespace_pid")
GUEST_SOURCE_ROOT = "/tmp/frozen-workspace-source"
GUEST_PROBE_ROOT = "/tmp/owned-workspace-probe"


def regular(path):
    path = Path(path)
    if (any(p.is_symlink() for p in (path, *path.parents)) or
            not stat.S_ISREG(path.stat().st_mode)):
        raise ValueError("regular_nonsymlink_input_required:" + str(path))
    return path


def validate_assets(assets):
    """Fail before disk creation unless the reviewed kernel cohort is pinned."""
    assets = Path(assets).absolute()
    manifest_path = regular(assets / "manifest.json")
    manifest = json.loads(manifest_path.read_bytes())
    if (not isinstance(manifest, dict) or
            manifest.get("schema_version") != "official-crab-zfs-guest-assets-v1" or
            manifest.get("architecture") != "linux/amd64" or
            manifest.get("kernel_release") != RELEASE):
        raise ValueError("reviewed_ZFS_guest_assets_required")
    if (manifest.get("kernel_sha256") != KERNEL_SHA256 or
            manifest.get("kernel_config_sha256") != CONFIG_SHA256):
        raise ValueError("reviewed_kernel_and_config_pins_required")
    initrd = manifest.get("initramfs", {})
    if (not isinstance(initrd, dict) or
            initrd.get("exact_full_matching_kernel_and_zfs_modules_included") is not True or
            initrd.get("nonmodule_bytes_and_modes_preserved") is not True):
        raise ValueError("full_matching_initramfs_required")
    if type(manifest.get("rootfs_bytes")) is not int or manifest["rootfs_bytes"] != 3 * 1024 ** 3:
        raise ValueError("reviewed_3GiB_owned_root_required")
    crab = manifest.get("base_guest_crab", {})
    if (not isinstance(crab, dict) or crab.get("commit") != CRAB_COMMIT or
            crab.get("python_package_sha256") != CRAB_PYTHON_SHA256 or
            crab.get("integrations_python_sha256") != INTEGRATIONS_PYTHON_SHA256 or
            crab.get("modified_upstream_files") != []):
        raise ValueError("original_guest_Crab_pins_required")
    pins = {"manifest.json": sha(manifest_path), "vmlinuz-virt": KERNEL_SHA256,
        "config-" + RELEASE: CONFIG_SHA256, "initramfs-virt": initrd.get("sha256"),
        "rootfs.qcow2": manifest.get("rootfs_qcow2_sha256")}
    for name, expected in pins.items():
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("complete_artifact_SHA256_required:" + name)
        if sha(regular(assets / name)) != expected:
            raise ValueError("guest_artifact_pin_mismatch:" + name)
    if (type(manifest.get("rootfs_qcow2_bytes")) is not int or
            manifest["rootfs_qcow2_bytes"] != (assets / "rootfs.qcow2").stat().st_size):
        raise ValueError("guest_disk_byte_length_mismatch")
    return assets, manifest, pins


def validate_output(output, assets):
    output = Path(output).absolute()
    if (output.exists() or output.is_symlink() or any(p.is_symlink() for p in output.parents) or
            any(c in str(output) for c in ",\r\n\x00")):
        raise ValueError("new_disjoint_nonsymlink_output_required")
    output = output.resolve()
    source = Path(__file__).resolve().parent
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in (assets.resolve(), source)):
        raise ValueError("new_disjoint_nonsymlink_output_required")
    return output


def recovery_failures(guest, *, mode, memory_mib, source_pins):
    """Derive host acceptance from actual evidence, never just a passed flag."""
    failures = []

    def need(value, name):
        if value is not True:
            failures.append(name)

    try:
        need(isinstance(guest, dict) and guest.get("schema_version") ==
             "official-crab-workspace-guest-driver-v1", "guest_schema")
        need(guest.get("passed") is True, "guest_passed")
        need(guest.get("pool_created") is True and guest.get("pool_destroyed") is True, "pool_lifecycle")
        need(guest.get("memory_mib") == memory_mib and type(guest.get("memory_mib")) is int and
             guest.get("filesystem_recovery_mode") == mode, "guest_scope")
        expected = {name: source_pins[name] for name in GUEST_SOURCES}
        need(guest.get("source_sha256") == expected == guest.get("source_sha256_after"), "guest_source_pins")
        need(guest.get("probe_execution", {}).get("returncode") == 0, "probe_execution")
        probe = guest["probe_result"]
        need(probe.get("schema_version") == "crab-zfs-owned-workspace-recovery-v1", "probe_schema")
        need(probe.get("passed") is True and probe.get("positive_composite_recovery_passed") is True, "positive_passed")
        need(type(probe.get("memory_mib")) is int and probe["memory_mib"] == memory_mib and
             probe.get("filesystem_recovery_mode") == mode, "probe_scope")
        need(probe.get("probe_source_sha256") == source_pins["workspace_probe.py"], "actual_probe_source")
        need(probe.get("worker_binary_sha256") == guest.get("actual_compiled_worker_sha256") and
             re.fullmatch(r"[0-9a-f]{64}", probe.get("worker_binary_sha256", "")) is not None, "actual_worker_binary")
        need(probe.get("crab_commit") == CRAB_COMMIT and
             probe.get("original_crab_python_sha256") == CRAB_PYTHON_SHA256 and
             probe.get("original_integrations_python_sha256") == INTEGRATIONS_PYTHON_SHA256, "original_Crab_source")
        for flag in ("upstream_runtime_replaced_or_stubbed", "original_upstream_source_patched",
                     "network_lock_bypass", "device_inode_fd_contract_relaxed",
                     "global_drop_caches_executed", "corrective_file_rewrites_executed"):
            need(probe.get(flag) is False, "forbidden_change:" + flag)
        for flag in ("capture_quiescence_verified", "filesystem_checkpoint_executed",
                     "private_ram_restored_exactly", "entire_owned_workspace_restored_exactly",
                     "held_fd_object_and_offset_restored_exactly", "identity_file_restored_exactly",
                     "continued_write_exactly_once_at_saved_offset", "post_restore_progress_verified"):
            need(probe.get(flag) is True, flag)
        for key, phases in (("checkpoint_result", ("process_checkpoint", "filesystem_checkpoint")),
                            ("restore_result", ("filesystem_restore", "process_restore"))):
            result = probe[key]
            operations = result["operations"]
            need(result.get("status") == "succeeded" and result.get("failure_code") == "none" and
                 len(operations) == 2 and all(row.get("executed") is True for row in operations) and
                 tuple(row["metadata"]["phase"] for row in operations) == phases, key)
            for row in operations:
                phase = row["metadata"]["phase"]
                command = row["command"]
                verb = {"process_checkpoint": "checkpoint", "process_restore": "restore",
                        "filesystem_checkpoint": "snapshot", "filesystem_restore": "rollback"}[phase]
                need(isinstance(command, list) and all(isinstance(s, str) for s in command) and
                     command[0] == ("runc" if phase.startswith("process") else "zfs") and
                     verb in command, "actual_command:" + phase)
        ram = probe["saved_private_ram_sha256"]
        need(isinstance(ram, str) and re.fullmatch(r"[0-9a-f]{64}", ram) is not None and
             ram == probe["restored_private_ram_sha256"] != probe["damaged_private_ram_sha256"], "whole_RAM_hashes")
        saved = probe["saved_workspace"]
        need(bool(saved["entries"]) and saved == probe["restored_workspace"] and
             saved != probe["damaged_workspace"], "whole_workspace_comparison")
        workspace_digest = hashlib.sha256(json.dumps(saved["entries"], sort_keys=True,
            separators=(",", ":")).encode()).hexdigest()
        need(saved.get("sha256") == workspace_digest, "workspace_evidence_digest")
        ledger = probe["saved_ledger"]
        raw = bytes.fromhex(ledger["hex"])
        expected_ledger = bytes((13 * i + 7) % 251 for i in range(512))
        need(raw == expected_ledger and ledger["bytes"] == 512 and
             hashlib.sha256(raw).hexdigest() == ledger["sha256"], "complete_saved_ledger")
        identity = probe["identity"]
        need(identity == probe["restored_identity_file"] and identity["bytes"] == memory_mib * 1024 ** 2 and
             identity["namespace_pid"] == 1, "identity_comparison")
        fd, restored = probe["saved_fd"], probe["restored_fd"]
        need(all(fd[k] == restored[k] for k in FD_KEYS) and fd["offset"] == 64 and
             fd["size"] == 512 and fd["device"] == identity["file_device"] and
             fd["inode"] == identity["file_inode"], "strict_FD_object_offset")
        continued = probe["continued_fd"]
        need(continued["offset"] == 72 and all(continued[k] == fd[k] for k in
             ("fd", "target", "inode", "device", "size", "namespace_pid")) and
             probe["continued_counters"] == [42, 12] and
             probe["continued_file_bytes_sha256"] == hashlib.sha256(raw[:64] + b"PH000042" + raw[72:]).hexdigest(),
             "continued_write_bytes_FD_counters")
        negative = probe["negative_control"]
        need(negative.get("expected_failure_witness_verified") is True and negative.get("ram_recovered") is True and
             negative.get("workspace_recovered") is False and
             negative["restored_private_ram_sha256"] == ram and
             negative["workspace_after_process_only"] != saved and
             negative["operation"].get("executed") is True, "process_only_negative_witness")
        for name, attempt in (("negative-process-only", negative["restore_attempt_evidence"]),
                              ("positive-composite", probe["positive_restore_attempt_evidence"])):
            log = attempt["restore_log"]
            need(log.get("available") is True and log.get("stable_during_read") is True and
                 type(log.get("bytes")) is int and log["bytes"] > 0 and
                 log.get("sha256") == log.get("retained_sha256") and
                 log.get("bytes") == log.get("retained_bytes") and
                 re.fullmatch(r"[0-9a-f]{64}", log.get("sha256", "")) is not None,
                 "per_attempt_raw_restore_log:" + name)
        snapshots = probe["snapshot_view"]
        observations = snapshots["observations"]
        need(set(observations) == set(SNAPSHOT_PHASES) and
             snapshots.get("saved_snapshot_bytes_verified_correct") is True and
             snapshots.get("clone_destroyed") is True and
             snapshots.get("destroyed_before_positive_restore") is True, "all_three_snapshot_observations")
        for phase in SNAPSHOT_PHASES:
            row = observations[phase]
            need(row.get("whole_workspace_matches_saved") is True and row.get("ledger_bytes_match_saved") is True and
                 row["workspace"] == saved and row["ledger"]["hex"] == ledger["hex"] and
                 row["ledger"]["sha256"] == ledger["sha256"], "immutable_snapshot:" + phase)
        observer = probe["filesystem_restore_observer"]
        need(observer.get("original_step_success") is True and observer["workspace"] == saved and
             observer["ledger"]["hex"] == ledger["hex"], "before_process_filesystem_recovery")
        absence = probe["runtime_deleted_before_positive_restore"]
        need(absence.get("no_owned_live_process_verified") is True and
             absence.get("container_absent_from_actual_runc_list") is True and
             absence.get("container_absence_error_confirmed") is True and
             absence.get("process_scan_errors") == [] and
             absence.get("live_processes_matching_owned_worker_inode") == [] and
             len(absence["known_pids"]) >= 2 and
             all(row.get("known_owned_process_not_live_verified") is True for row in absence["known_pids"]),
             "process_absence_before_filesystem_lifecycle")
        lifecycle = probe["filesystem_lifecycle"]
        need(lifecycle.get("mode") == mode and lifecycle.get("original_rollback_adapter_call_unchanged") is True,
             "original_rollback_lifecycle")
        if mode == "unmount-rollback-mount":
            need(lifecycle.get("actual_unmount_succeeded") is True and lifecycle.get("actual_mount_succeeded") is True and
                 lifecycle.get("mounted_property_after_unmount") == "no" and
                 lifecycle.get("mounted_property_after_mount") == "yes", "genuine_unmount_mount")
    except (KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
        failures.append("incomplete_or_malformed_recovery_evidence:" + str(exc))
    return failures


def required(vm, report, command, *, timeout=30):
    value = vm.run_shell(command, timeout=timeout)
    report.setdefault("setup_commands", []).append({"command": command, **value})
    if value["return_code"] != 0:
        raise RuntimeError("actual_guest_command_failed:" + json.dumps(report["setup_commands"][-1]))
    return value["stdout"]


def boot_guest(vm, report):
    vm.start()
    report["boot_console"] = vm.wait_for_serial(
        "Launching initramfs emergency recovery shell", timeout=120).decode("utf-8", "replace")
    if required(vm, report, "uname -r").strip() != RELEASE:
        raise RuntimeError("actual_reviewed_kernel_release_required")
    required(vm, report, "/usr/bin/kmod --version")
    required(vm, report, "sha256sum /usr/bin/kmod /lib/modules/" + RELEASE + "/modules.dep")
    required(vm, report, "ln -s /usr/bin/kmod /tmp/depmod")
    required(vm, report, "/tmp/depmod -a " + RELEASE, timeout=90)
    required(vm, report, "sha256sum /lib/modules/" + RELEASE + "/modules.dep /lib/modules/" + RELEASE + "/modules.dep.bin")
    for module in ("virtio_pci", "virtio_blk", "ext4"):
        required(vm, report, "modprobe " + module)
    required(vm, report, "ls -l /dev/vda")
    required(vm, report, "mkdir -p /mnt/root")
    required(vm, report, "mount -t ext4 /dev/vda /mnt/root")
    for command in (
        "mkdir -p /mnt/root/proc /mnt/root/sys /mnt/root/dev /mnt/root/run /mnt/root/tmp /sys/fs/cgroup /dev/pts",
        "mountpoint -q /sys/fs/cgroup || mount -t cgroup2 none /sys/fs/cgroup",
        "mountpoint -q /dev/pts || mount -t devpts devpts /dev/pts",
        "mount --bind /proc /mnt/root/proc", "mount --bind /sys /mnt/root/sys",
        "mount --bind /sys/fs/cgroup /mnt/root/sys/fs/cgroup", "mount --bind /dev /mnt/root/dev",
        "mount --bind /dev/pts /mnt/root/dev/pts", "mount -t tmpfs tmpfs /mnt/root/run",
        # File-vdev paths must be visible in the initial and moved-root namespaces.
        "mount --bind /mnt/root/tmp /tmp", "ip link set lo up"):
        required(vm, report, command)
    for module in ("unix_diag", "inet_diag", "tcp_diag", "udp_diag", "netlink_diag",
                   "af_packet_diag", "tun", "nf_tables", "x_tables", "nft_compat", "xt_mark"):
        required(vm, report, "modprobe " + module)
    required(vm, report, "modprobe zfs zfs_arc_max=134217728 zfs_prefetch_disable=1", timeout=90)
    required(vm, report, "ls -l /dev/zfs")
    for name in ("zfs", "spl"):
        if required(vm, report, "cat /sys/module/" + name + "/version").strip() != "2.4.4-1":
            raise RuntimeError("actual_reviewed_ZFS_module_version_required")


def transfer(vm, report, local, guest_path):
    encoded = base64.b64encode(local.read_bytes()).decode("ascii")
    for index in range(0, len(encoded), 2000):
        required(vm, report, "printf %s " + shlex.quote(encoded[index:index + 2000]) +
                 (">" if index == 0 else ">>") + shlex.quote(guest_path + ".b64"))
    required(vm, report, "base64 -d " + shlex.quote(guest_path + ".b64") + " >" + shlex.quote(guest_path))
    observed = required(vm, report, "sha256sum " + shlex.quote(guest_path)).split()
    if not observed or observed[0] != sha(local):
        raise RuntimeError("transferred_source_pin_mismatch:" + local.name)


def retrieve(vm, report, guest_path, local, *, limit=16 * 1024 ** 2):
    """Retrieve exact bytes in bounded serial frames, including full raw logs."""
    size = int(required(vm, report, "wc -c <" + shlex.quote(guest_path)).strip())
    if not 0 <= size <= limit:
        raise ValueError("bounded_guest_evidence_size_required")
    digest = required(vm, report, "sha256sum " + shlex.quote(guest_path)).split()[0]
    payload = bytearray()
    for offset in range(0, size, 32768):
        encoded = required(vm, report, "dd if=" + shlex.quote(guest_path) +
            " bs=32768 skip=" + str(offset // 32768) + " count=1 2>/dev/null | base64")
        payload.extend(base64.b64decode("".join(encoded.split()), validate=True))
    if len(payload) != size or hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError("retrieved_raw_evidence_pin_mismatch")
    local.write_bytes(payload)
    return {"guest_path": guest_path, "output_file": local.name, "bytes": size, "sha256": digest}


def check(assets, output, *, mode, memory_mib=8):
    if mode not in MODES or type(memory_mib) is not int or memory_mib not in (8, 64):
        raise ValueError("reviewed_mode_and_memory_scope_required")
    assets, manifest, asset_pins = validate_assets(assets)
    output = validate_output(output, assets)
    source = Path(__file__).absolute().parent
    paths = {name: regular(source / name) for name in (*GUEST_SOURCES, Path(__file__).name)}
    paths["microvm_runtime.py"] = regular(Path(microvm_runtime.__file__).absolute())
    pins = {name: sha(path) for name, path in paths.items()}
    output.mkdir(parents=True)
    frozen = output / "source"
    frozen.mkdir()
    for name, path in paths.items():
        (frozen / name).write_bytes(path.read_bytes())
        if sha(frozen / name) != pins[name]:
            raise ValueError("source_changed_before_freeze:" + name)
    (frozen / "manifest.json").write_text(json.dumps({
        "schema_version": "official-crab-workspace-frozen-source-v1", "sha256": pins}, indent=2) + "\n")
    frozen_manifest_sha = sha(frozen / "manifest.json")
    report = {"schema_version": "official-crab-workspace-microvm-driver-v1", "passed": False,
        "backend": "x86_64_tcg", "guest_memory_mib": 1024, "memory_mib": memory_mib,
        "filesystem_recovery_mode": mode, "source_sha256": pins,
        "asset_sha256": asset_pins, "asset_manifest_sha256": asset_pins["manifest.json"],
        "frozen_manifest_sha256": frozen_manifest_sha,
        "guest_network": False, "host_directory_shared_with_guest": False,
        "model_rollout_or_optimizer_executed": False, "graded_repository_episode": False,
        "disk_retained": True, "setup_commands": [], "retained_guest_evidence": []}
    disk = output / "disposable.qcow2"
    started = time.perf_counter()
    vm = None
    try:
        _clone_or_copy_qcow2(assets / "rootfs.qcow2", disk)
        vm = MicroVMRuntime(assets / "vmlinuz-virt", assets / "initramfs-virt", disk,
            kernel_sha256=manifest["kernel_sha256"], initramfs_sha256=manifest["initramfs"]["sha256"],
            backend="x86_64_tcg", memory_mib=1024, command_timeout=90)
        with vm:
            boot_guest(vm, report)
            required(vm, report, "mkdir -p /mnt/root" + GUEST_SOURCE_ROOT)
            for name in (*GUEST_SOURCES, "manifest.json"):
                transfer(vm, report, frozen / name, "/mnt/root" + GUEST_SOURCE_ROOT + "/" + name)
            program = ("/bin/busybox --install -s; exec env PATH=/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin "
                "LD_LIBRARY_PATH=/usr/local/lib:/usr/lib:/lib /usr/local/bin/python3.12 -B " +
                GUEST_SOURCE_ROOT + "/workspace_driver.py --source " + GUEST_SOURCE_ROOT +
                " --mode " + mode + " --memory-mib " + str(memory_mib))
            namespace = ("mount --make-rprivate /; cd /mnt/root; mount --move . /; "
                         "exec /bin/busybox chroot . /bin/sh -ec " + shlex.quote(program))
            command = ("unshare -m /bin/sh -ec " + shlex.quote(namespace) +
                " >/mnt/root/tmp/workspace-result.json 2>/mnt/root/tmp/workspace-stderr.log")
            report["guest_execution"] = {"command": command, **vm.run_shell(command, timeout=300)}
            for name in ("workspace-result.json", "workspace-stderr.log"):
                report["retained_guest_evidence"].append(retrieve(vm, report,
                    "/mnt/root/tmp/" + name, output / ("guest-result.json" if name.endswith("result.json") else "guest-stderr.log")))
            guest = json.loads((output / "guest-result.json").read_bytes())
            report["guest_result"] = guest
            report["recovery_acceptance_failures"] = recovery_failures(guest,
                mode=mode, memory_mib=memory_mib, source_pins=pins)
            if not (guest.get("frozen_manifest_sha256") == frozen_manifest_sha ==
                    guest.get("frozen_manifest_sha256_after")):
                report["recovery_acceptance_failures"].append("guest_frozen_manifest_guard")
            # Preserve each available attempt record/log before deleting a successful disk.
            for attempt in ("negative-process-only", "positive-composite"):
                for name in ("attempt.json", "restore.log"):
                    guest_path = "/mnt/root" + GUEST_PROBE_ROOT + "/restore-attempts/" + attempt + "/" + name
                    present = vm.run_shell("test -f " + shlex.quote(guest_path))
                    report.setdefault("guest_evidence_availability", []).append({"guest_path": guest_path, **present})
                    if present["return_code"] == 0:
                        report["retained_guest_evidence"].append(retrieve(vm, report, guest_path, output / (attempt + "-" + name)))
                    else:
                        report["recovery_acceptance_failures"].append("missing_raw_attempt_evidence:" + attempt + "/" + name)
            for attempt, log in (
                ("negative-process-only", guest.get("probe_result", {}).get("negative_control", {}).get("restore_attempt_evidence", {}).get("restore_log", {})),
                ("positive-composite", guest.get("probe_result", {}).get("positive_restore_attempt_evidence", {}).get("restore_log", {}))):
                local = output / (attempt + "-restore.log")
                if not local.is_file() or sha(local) != log.get("retained_sha256"):
                    report["recovery_acceptance_failures"].append("actual_retained_log_pin_mismatch:" + attempt)
            report["guest_sync"] = vm.run_shell("sync", timeout=90)
            report["kernel_tail"] = vm.run_shell("dmesg | tail -c 16000")
            report["passed"] = (report["guest_execution"]["return_code"] == 0 and
                report["guest_sync"]["return_code"] == 0 and not report["recovery_acceptance_failures"])
    except Exception as exc:
        report["error"] = str(exc)
        report["passed"] = False
    finally:
        if vm is not None:
            vm.close()
        report["driver_wall_seconds"] = time.perf_counter() - started
        for label, paths_to_check, expected in (
            ("source", paths, pins), ("frozen_source", {name: frozen / name for name in pins}, pins),
            ("asset", {name: assets / name for name in asset_pins}, asset_pins)):
            try:
                after = {name: sha(regular(path)) for name, path in paths_to_check.items()}
                report[label + "_sha256_after"] = after
                if after != expected:
                    raise ValueError(label + "_changed_during_probe")
            except Exception as exc:
                report[label + "_guard_error"] = str(exc)
                report["passed"] = False
        try:
            report["frozen_manifest_sha256_after"] = sha(regular(frozen / "manifest.json"))
            if report["frozen_manifest_sha256_after"] != frozen_manifest_sha:
                raise ValueError("frozen_manifest_changed")
        except Exception as exc:
            report["passed"] = False
            report["frozen_manifest_guard_error"] = str(exc)
        if report["passed"]:
            try:
                # VM has exited; flush the owned host file before removing it.
                descriptor = os.open(disk, os.O_RDWR)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                disk.unlink()
                report["disk_retained"] = False
            except Exception as exc:
                report["disk_cleanup_error"] = str(exc)
                report["passed"] = False
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--memory-mib", type=int, choices=(8, 64), default=8)
    args = parser.parse_args()
    result = check(args.assets, args.output, mode=args.mode, memory_mib=args.memory_mib)
    print(json.dumps({"passed": result["passed"], "error": result.get("error"),
        "recovery_acceptance_failures": result.get("recovery_acceptance_failures"),
        "output": str(Path(args.output).absolute())}, indent=2))
    raise SystemExit(0 if result["passed"] else 2)
