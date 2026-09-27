"""Host fail-closed guards; these tests do not execute QEMU or claim speed."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from examples.official_crab_criu import run_workspace_microvm as host
from examples.official_crab_criu import workspace_driver as driver


def digest(data):
    return hashlib.sha256(data).hexdigest()


def valid_evidence(mode="unmount-rollback-mount", memory_mib=8):
    """Synthetic evidence only for verifying acceptance/error-path logic."""
    pins = {name: "e" * 64 for name in driver.GUEST_SOURCES}
    ledger = bytes((13 * i + 7) % 251 for i in range(512))
    entries = [{"path": "ledger.bin", "kind": "file", "mode": 0o600,
                "bytes": 512, "sha256": digest(ledger)},
               {"path": "saved.txt", "kind": "file", "mode": 0o600,
                "bytes": 18, "sha256": digest(b"saved-at-phase-41\n")}]
    workspace = {"entries": entries, "sha256": digest(json.dumps(entries,
        sort_keys=True, separators=(",", ":")).encode())}
    saved_ledger = {"hex": ledger.hex(), "sha256": digest(ledger), "bytes": 512}
    identity = {"address": 4096, "bytes": memory_mib * 1024 ** 2, "page_size": 4096,
        "held_fd": 3, "file_inode": 10, "file_device": 27, "namespace_pid": 1,
        "initial_file_offset": 64}
    fd = {"fd": 3, "target": "/workspace/ledger.bin", "offset": 64,
          "inode": 10, "device": 27, "size": 512, "namespace_pid": 1}
    damaged = {"entries": [{"path": "damaged"}]}

    def operation(phase):
        verb = {"process_checkpoint": "checkpoint", "process_restore": "restore",
                "filesystem_checkpoint": "snapshot", "filesystem_restore": "rollback"}[phase]
        return {"executed": True, "metadata": {"phase": phase},
                "command": ["runc" if phase.startswith("process") else "zfs", verb]}

    def attempt():
        return {"restore_log": {"available": True, "stable_during_read": True,
            "bytes": 20, "retained_bytes": 20, "sha256": "c" * 64,
            "retained_sha256": "c" * 64}}

    probe = {"schema_version": "crab-zfs-owned-workspace-recovery-v1", "passed": True,
        "positive_composite_recovery_passed": True, "memory_mib": memory_mib,
        "filesystem_recovery_mode": mode, "probe_source_sha256": pins["workspace_probe.py"],
        "worker_binary_sha256": "f" * 64, "crab_commit": host.CRAB_COMMIT,
        "original_crab_python_sha256": host.CRAB_PYTHON_SHA256,
        "original_integrations_python_sha256": host.INTEGRATIONS_PYTHON_SHA256,
        "checkpoint_result": {"status": "succeeded", "failure_code": "none", "operations":
            [operation("process_checkpoint"), operation("filesystem_checkpoint")]},
        "restore_result": {"status": "succeeded", "failure_code": "none", "operations":
            [operation("filesystem_restore"), operation("process_restore")]},
        "saved_private_ram_sha256": "a" * 64, "restored_private_ram_sha256": "a" * 64,
        "damaged_private_ram_sha256": "b" * 64,
        "saved_workspace": workspace, "restored_workspace": copy.deepcopy(workspace),
        "damaged_workspace": damaged, "saved_ledger": saved_ledger,
        "identity": identity, "restored_identity_file": dict(identity),
        "saved_fd": fd, "restored_fd": dict(fd), "continued_fd": {**fd, "offset": 72},
        "continued_counters": [42, 12],
        "continued_file_bytes_sha256": digest(ledger[:64] + b"PH000042" + ledger[72:]),
        "negative_control": {"expected_failure_witness_verified": True, "ram_recovered": True,
            "workspace_recovered": False, "restored_private_ram_sha256": "a" * 64,
            "workspace_after_process_only": damaged, "operation": operation("process_restore"),
            "restore_attempt_evidence": attempt()},
        "positive_restore_attempt_evidence": attempt(),
        "snapshot_view": {"saved_snapshot_bytes_verified_correct": True,
            "clone_destroyed": True, "destroyed_before_positive_restore": True,
            "observations": {phase: {"whole_workspace_matches_saved": True,
                "ledger_bytes_match_saved": True, "workspace": copy.deepcopy(workspace),
                "ledger": dict(saved_ledger)} for phase in host.SNAPSHOT_PHASES}},
        "filesystem_restore_observer": {"original_step_success": True,
            "workspace": copy.deepcopy(workspace), "ledger": dict(saved_ledger)},
        "runtime_deleted_before_positive_restore": {"no_owned_live_process_verified": True,
            "container_absent_from_actual_runc_list": True, "container_absence_error_confirmed": True,
            "process_scan_errors": [], "live_processes_matching_owned_worker_inode": [],
            "known_pids": [{"pid": 1, "known_owned_process_not_live_verified": True},
                           {"pid": 2, "known_owned_process_not_live_verified": True}]},
        "filesystem_lifecycle": {"mode": mode, "original_rollback_adapter_call_unchanged": True,
            "actual_unmount_succeeded": True, "actual_mount_succeeded": True,
            "mounted_property_after_unmount": "no", "mounted_property_after_mount": "yes"}}
    for flag in ("upstream_runtime_replaced_or_stubbed", "original_upstream_source_patched",
                 "network_lock_bypass", "device_inode_fd_contract_relaxed",
                 "global_drop_caches_executed", "corrective_file_rewrites_executed"):
        probe[flag] = False
    for flag in ("capture_quiescence_verified", "filesystem_checkpoint_executed",
                 "private_ram_restored_exactly", "entire_owned_workspace_restored_exactly",
                 "held_fd_object_and_offset_restored_exactly", "identity_file_restored_exactly",
                 "continued_write_exactly_once_at_saved_offset", "post_restore_progress_verified"):
        probe[flag] = True
    guest = {"schema_version": "official-crab-workspace-guest-driver-v1", "passed": True,
        "pool_created": True, "pool_destroyed": True, "memory_mib": memory_mib,
        "filesystem_recovery_mode": mode, "source_sha256": pins, "source_sha256_after": dict(pins),
        "probe_execution": {"returncode": 0}, "actual_compiled_worker_sha256": "f" * 64,
        "probe_result": probe}
    return guest, pins


class RecoveryAcceptanceTests(unittest.TestCase):
    def failures(self, guest, pins, mode="unmount-rollback-mount", memory_mib=8):
        return host.recovery_failures(guest, mode=mode, memory_mib=memory_mib, source_pins=pins)

    def test_complete_fixed_scope_evidence_accepts_both_modes_and_sizes(self):
        for mode in driver.MODES:
            for size in (8, 64):
                with self.subTest(mode=mode, size=size):
                    guest, pins = valid_evidence(mode, size)
                    self.assertEqual(self.failures(guest, pins, mode, size), [])

    def test_true_summary_alone_never_proves_recovery(self):
        _, pins = valid_evidence()
        for value in ({"passed": True}, None, [], {"probe_result": {"passed": True}}):
            with self.subTest(value=value):
                self.assertTrue(self.failures(value, pins))

    def test_missing_snapshot_phases_cannot_use_vacuous_all(self):
        for phases in ({}, {"before_live_damage": {}}):
            guest, pins = valid_evidence()
            guest["probe_result"]["snapshot_view"]["observations"] = phases
            self.assertTrue(self.failures(guest, pins))

    def test_unchanged_summary_flags_cannot_hide_corrupted_actual_evidence(self):
        changes = [
            lambda p: p["restored_fd"].update(device=28),
            lambda p: p["restored_fd"].update(offset=72),
            lambda p: p.update(restored_private_ram_sha256="b" * 64),
            lambda p: p["restored_workspace"]["entries"].pop(),
            lambda p: p["negative_control"].update(workspace_after_process_only=p["saved_workspace"]),
            lambda p: p["snapshot_view"]["observations"]["after_live_damage"]["ledger"].update(hex="00" * 512),
            lambda p: p.update(continued_counters=[42, 11]),
            lambda p: p.update(continued_file_bytes_sha256="d" * 64),
            lambda p: p["restore_result"]["operations"].reverse(),
            lambda p: p["restore_result"]["operations"][0].update(executed=False),
            lambda p: p["positive_restore_attempt_evidence"]["restore_log"].update(available=False),
            lambda p: p["runtime_deleted_before_positive_restore"].update(process_scan_errors=[{"pid": 3}]),
        ]
        for index, change in enumerate(changes):
            with self.subTest(change=index):
                guest, pins = valid_evidence()
                change(guest["probe_result"])
                self.assertTrue(self.failures(guest, pins))

    def test_boolean_coercion_is_rejected_for_witnesses(self):
        for value in (1, "true", None):
            guest, pins = valid_evidence()
            guest["probe_result"]["private_ram_restored_exactly"] = value
            self.assertTrue(self.failures(guest, pins))

    def test_source_pin_mismatch_is_rejected_even_when_all_results_pass(self):
        guest, pins = valid_evidence()
        guest["source_sha256_after"]["preflight.py"] = "d" * 64
        self.assertTrue(self.failures(guest, pins))


class AssetBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.assets = self.root / "assets"
        self.assets.mkdir()
        contents = {"vmlinuz-virt": b"unit-test-kernel", "initramfs-virt": b"unit-test-initrd",
                    "config-" + host.RELEASE: b"unit-test-config", "rootfs.qcow2": b"unit-test-disk"}
        for name, raw in contents.items():
            (self.assets / name).write_bytes(raw)
        self.patches = [patch.object(host, "KERNEL_SHA256", digest(contents["vmlinuz-virt"])),
                        patch.object(host, "CONFIG_SHA256", digest(contents["config-" + host.RELEASE]))]
        for item in self.patches:
            item.start()
        self.manifest = {"schema_version": "official-crab-zfs-guest-assets-v1", "architecture": "linux/amd64",
            "kernel_release": host.RELEASE, "kernel_sha256": host.KERNEL_SHA256,
            "kernel_config_sha256": host.CONFIG_SHA256, "rootfs_bytes": 3 * 1024 ** 3,
            "rootfs_qcow2_bytes": len(contents["rootfs.qcow2"]),
            "rootfs_qcow2_sha256": digest(contents["rootfs.qcow2"]),
            "initramfs": {"sha256": digest(contents["initramfs-virt"]),
                "exact_full_matching_kernel_and_zfs_modules_included": True,
                "nonmodule_bytes_and_modes_preserved": True},
            "base_guest_crab": {"commit": host.CRAB_COMMIT, "python_package_sha256": host.CRAB_PYTHON_SHA256,
                "integrations_python_sha256": host.INTEGRATIONS_PYTHON_SHA256, "modified_upstream_files": []}}
        self.write_manifest()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temporary.cleanup()

    def write_manifest(self):
        (self.assets / "manifest.json").write_text(json.dumps(self.manifest))

    def test_actual_bytes_are_checked_before_any_vm_is_constructed(self):
        host.validate_assets(self.assets)
        for name in ("vmlinuz-virt", "initramfs-virt", "config-" + host.RELEASE, "rootfs.qcow2"):
            path = self.assets / name
            saved = path.read_bytes()
            path.write_bytes(saved + b"tampered")
            with self.subTest(name=name), patch.object(host, "MicroVMRuntime") as vm:
                with self.assertRaisesRegex(ValueError, "pin_mismatch"):
                    host.check(self.assets, self.root / "output", mode="original-rollback")
                vm.assert_not_called()
            path.write_bytes(saved)

    def test_malformed_or_wrong_cohort_manifests_fail_closed(self):
        changes = [{"schema_version": "private-candidate"}, {"architecture": "linux/arm64"},
            {"kernel_release": "6.18.52-0-virt"}, {"kernel_config_sha256": "a" * 64},
            {"rootfs_bytes": True}, {"rootfs_bytes": 256 * 1024 ** 2}, {"base_guest_crab": {}},
            {"initramfs": {}}, {"rootfs_qcow2_bytes": 1}, {"rootfs_qcow2_sha256": None}]
        original = copy.deepcopy(self.manifest)
        for change in changes:
            self.manifest = {**original, **change}
            self.write_manifest()
            with self.subTest(change=change), self.assertRaises(ValueError):
                host.validate_assets(self.assets)

    def test_symlink_inputs_are_rejected(self):
        path = self.assets / "initramfs-virt"
        moved = self.root / "elsewhere"
        path.rename(moved)
        path.symlink_to(moved)
        with self.assertRaisesRegex(ValueError, "nonsymlink"):
            host.validate_assets(self.assets)

    def test_existing_nested_parent_and_symlink_outputs_are_rejected(self):
        existing = self.root / "existing"
        existing.mkdir()
        link = self.root / "linked"
        link.symlink_to(existing, target_is_directory=True)
        for value in (existing, self.assets / "child", self.root, link / "new", self.root / "unsafe,name",
                      self.root / "other/../assets/new"):
            with self.subTest(output=value), self.assertRaisesRegex(ValueError, "output_required"):
                host.validate_output(value, self.assets)

    def test_unreviewed_sizes_and_modes_fail_before_asset_read_or_vm(self):
        with patch.object(host, "validate_assets") as validate, patch.object(host, "MicroVMRuntime") as vm:
            for size in (True, None, "8", 0, 1, 7, 9, 65):
                with self.subTest(size=size), self.assertRaises(ValueError):
                    host.check(self.assets, self.root / "output", mode="original-rollback", memory_mib=size)
            with self.assertRaises(ValueError):
                host.check(self.assets, self.root / "output", mode="force-rollback")
            validate.assert_not_called()
            vm.assert_not_called()

    def test_guest_boot_failure_keeps_disk_and_records_frozen_source(self):
        output = self.root / "failed"
        vm = Mock()
        vm.__enter__ = Mock(return_value=vm)
        vm.__exit__ = Mock(return_value=False)
        with patch.object(host, "_clone_or_copy_qcow2", side_effect=lambda a, b: b.write_bytes(b"owned-disk")), \
                patch.object(host, "MicroVMRuntime", return_value=vm), \
                patch.object(host, "boot_guest", side_effect=RuntimeError("actual_boot_failure")):
            result = host.check(self.assets, output, mode="original-rollback")
        self.assertFalse(result["passed"])
        self.assertTrue((output / "disposable.qcow2").is_file())
        self.assertEqual(result["error"], "actual_boot_failure")
        self.assertEqual(result["source_sha256"], result["source_sha256_after"])
        self.assertEqual(result["source_sha256"], result["frozen_source_sha256_after"])
        self.assertEqual(result["asset_sha256"], result["asset_sha256_after"])
        self.assertTrue((output / "source/workspace_worker.c").is_file())
        vm.close.assert_called_once()


class GuestSafetyTests(unittest.TestCase):
    def test_guest_marker_is_required_before_compilation_or_pool_creation(self):
        with patch.object(driver, "require_guest", side_effect=RuntimeError("guest_required")), \
                patch.object(driver, "subprocess") as subprocess:
            with self.assertRaisesRegex(RuntimeError, "guest_required"):
                driver.run("/tmp/nonexistent", mode="original-rollback", memory_mib=8)
            subprocess.run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
