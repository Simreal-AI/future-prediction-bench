"""Real-QEMU probe for non-action-port snapshot tag and failure boundaries.

The injected failures happen only after real QEMU HMP operations have run.
This checks the production runtime against a real qcow2 snapshot table; it
does not simulate a storage-controller or host-power failure.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

from future_prediction_bench.microvm_runtime import (
    MicroVMRuntimeError, _clone_or_copy_qcow2, _quarantine_marker, _sha256_file,
)

from .microvm_benchmark import _boot, _required, _runtime


_RAM = "/tmp/fpb-transaction-ram"
_DISK = "/mnt/root/workspace/.fpb-transaction-disk"


def _state(vm):
    return _required(vm, f"cat {_RAM}") == "baseline" and _required(vm, f"cat {_DISK}") == "baseline"


def _listed(tag, listing):
    return re.search(rf"(?m)^\s*\S+\s+{re.escape(tag)}(?:\s|$)", listing) is not None


def _offline_tags(disk):
    completed = subprocess.run(
        ["qemu-img", "snapshot", "-l", str(disk)], check=True,
        capture_output=True, text=True, timeout=30)
    return completed.stdout


def _require_assets(assets):
    if assets.is_symlink() or not assets.is_dir():
        raise ValueError("Pinned asset directory is missing or symlinked")
    manifest_path = assets / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "boltons-microvm-assets-v2":
        raise ValueError("Expected pinned Boltons v2 assets")
    for path, expected in (
        (assets / "vmlinuz-virt", manifest["alpine_sha256"]["vmlinuz-virt"]),
        (assets / "initramfs-virt", manifest["alpine_sha256"]["initramfs-virt"]),
        (assets / "modloop-virt-padded.raw", manifest["modloop_disk_sha256"]),
        (assets / "rootfs.qcow2", manifest["rootfs_qcow2_sha256"]),
    ):
        if path.is_symlink() or not path.is_file() or _sha256_file(path) != expected:
            raise ValueError("Pinned VM asset digest changed")
    return manifest_path, manifest


def run(assets_dir, output_dir):
    if Path(assets_dir).is_symlink():
        raise ValueError("Pinned asset directory cannot be a symlink")
    assets = Path(assets_dir).resolve()
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path, manifest = _require_assets(assets)
    manifest_digest = _sha256_file(manifest_path)
    source_paths = {
        "runtime": Path(__file__).parents[2] / "future_prediction_bench" / "microvm_runtime.py",
        "probe": Path(__file__),
        "asset_and_boot_helper": Path(__file__).with_name("microvm_benchmark.py"),
    }
    source_binding = {name: _sha256_file(path) for name, path in source_paths.items()}
    disk = output / "transaction.qcow2"
    fail_disk = output / "cleanup-fail.qcow2"
    _clone_or_copy_qcow2(assets / "rootfs.qcow2", disk)
    results = {}
    try:
        with _runtime(assets, disk) as vm:
            _boot(vm)
            _required(vm, f"printf baseline > {_RAM}; printf baseline > {_DISK}")
            vm.save_snapshot("committed")
            original_tag_listing = vm._hmp("info snapshots")
            try:
                vm.save_snapshot("committed")
            except MicroVMRuntimeError as exc:
                if str(exc) != "vm_snapshot_tag_already_exists":
                    raise
            else:
                raise RuntimeError("Duplicate snapshot tag was not rejected")
            if vm._hmp("info snapshots") != original_tag_listing:
                raise RuntimeError("Duplicate save changed QEMU snapshot table")
            _required(vm, f"printf changed > {_RAM}; printf changed > {_DISK}")
            vm.load_snapshot("committed")
            if not _state(vm):
                raise RuntimeError("Committed RAM/ext4 state did not restore")
            results["existing_tag_rejected_without_replacement"] = True
            results["committed_ram_and_ext4_restored"] = True

            actual_hmp = vm._hmp
            def lost_save_reply(command, *, timeout=None):
                response = actual_hmp(command, timeout=timeout)
                if command == "savevm orphan":
                    raise MicroVMRuntimeError("injected_reply_loss_after_real_savevm")
                return response
            vm._hmp = lost_save_reply
            try:
                vm.save_snapshot("orphan")
            except MicroVMRuntimeError as exc:
                if str(exc) != "injected_reply_loss_after_real_savevm":
                    raise
            else:
                raise RuntimeError("Injected lost save response was not rejected")
            if vm.get_state()["running"]:
                raise RuntimeError("Failed save did not close VM")
            results["lost_ack_closed_runtime"] = True

        listing = _offline_tags(disk)
        if not _listed("committed", listing) or _listed("orphan", listing):
            raise RuntimeError("Offline qcow2 tag table failed cleanup check")
        results["qcow2_committed_tag_preserved"] = True
        results["qcow2_uncommitted_tag_deleted"] = True

        with _runtime(assets, disk) as replacement:
            replacement.start(paused=True)
            replacement.load_snapshot("committed", resume=True)
            if not _state(replacement):
                raise RuntimeError("Replacement VM could not restore committed state")
        results["replacement_vm_restored_committed_ram_and_ext4"] = True

        _clone_or_copy_qcow2(assets / "rootfs.qcow2", fail_disk)
        with _runtime(assets, fail_disk) as unclean:
            _boot(unclean)
            actual_hmp = unclean._hmp
            def failed_delete(command, *, timeout=None):
                if command == "delvm orphan":
                    raise MicroVMRuntimeError("injected_delete_failure")
                response = actual_hmp(command, timeout=timeout)
                if command == "savevm orphan":
                    raise MicroVMRuntimeError("injected_reply_loss_after_real_savevm")
                return response
            unclean._hmp = failed_delete
            try:
                unclean.save_snapshot("orphan")
            except MicroVMRuntimeError as exc:
                if str(exc) != "vm_snapshot_cleanup_unverified":
                    raise
            else:
                raise RuntimeError("Unverified tag deletion was accepted")
            if unclean.get_state()["running"]:
                raise RuntimeError("Unverified tag deletion left VM running")
        if not _listed("orphan", _offline_tags(fail_disk)):
            raise RuntimeError("Fault probe did not leave expected uncommitted tag")
        if not _quarantine_marker(fail_disk).is_file():
            raise RuntimeError("Uncertain disk was not persistently marked")
        try:
            _runtime(assets, fail_disk).start()
        except MicroVMRuntimeError as exc:
            if str(exc) != "qcow2_disk_quarantined":
                raise
        else:
            raise RuntimeError("Uncertain disk reopened in replacement runtime")
        try:
            _clone_or_copy_qcow2(fail_disk, output / "forbidden-clone.qcow2")
        except MicroVMRuntimeError as exc:
            if str(exc) != "qcow2_disk_quarantined":
                raise
        else:
            raise RuntimeError("Uncertain disk cloned into a new runtime disk")
        results["failed_delete_quarantined_runtime_and_disk"] = True

        if source_binding != {name: _sha256_file(path) for name, path in source_paths.items()}:
            raise RuntimeError("Probe source changed during QEMU run")
        if (_require_assets(assets)[1] != manifest
                or _sha256_file(manifest_path) != manifest_digest):
            raise RuntimeError("Pinned assets changed during QEMU run")

        report = {
            "kind": "real_qemu_snapshot_transaction_probe_v1",
            "status": "passed",
            "scope": "One pinned Boltons v2 VM; real HMP operations with host-injected lost acknowledgement after savevm completed",
            "source_sha256": source_binding,
            "source_and_assets_stable_through_run": True,
            "asset_binding": {
                "manifest_schema_version": manifest["schema_version"],
                "manifest_sha256": manifest_digest,
                "source_sdist_sha256": manifest["source_sdist_sha256"],
                "seed_workspace_sha256": manifest["seed_workspace_sha256"],
                "rootfs_qcow2_sha256": manifest["rootfs_qcow2_sha256"],
                "kernel_sha256": manifest["alpine_sha256"]["vmlinuz-virt"],
                "initramfs_sha256": manifest["alpine_sha256"]["initramfs-virt"],
                "modloop_disk_sha256": manifest["modloop_disk_sha256"],
            },
            "checks": results,
            "model_inference_or_training": False,
            "host_power_or_storage_failure_tested": False,
        }
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report
    finally:
        disk.unlink(missing_ok=True)
        fail_disk.unlink(missing_ok=True)
        _quarantine_marker(disk).unlink(missing_ok=True)
        _quarantine_marker(fail_disk).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    started = time.monotonic()
    report = run(args.assets_dir, args.output)
    print(json.dumps({"status": report["status"],
                      "checks": report["checks"],
                      "elapsed_seconds": time.monotonic() - started}, indent=2))


if __name__ == "__main__":
    main()
