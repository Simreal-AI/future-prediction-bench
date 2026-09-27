"""Host-only tests for the opt-in clean RealWorldEnv VM template."""

import copy
import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from future_prediction_bench import prepared_microvm as prepared_module
from future_prediction_bench.coding_env import _workspace_digest
from future_prediction_bench.microvm_coding import MicroVMCodingAdapter
from future_prediction_bench.microvm_runtime import MicroVMRuntime, _sha256_file
from future_prediction_bench.prepared_microvm import (
    PreparedMicroVMTemplate, _PreparedCodingAdapter, _clean_task,
)
from future_prediction_bench.realworld import RealWorldEnv, validate_task


BASELINE = {(1, 100): "/usr/bin/busybox", (2, 100): "/usr/bin/busybox"}


class _FakeTemplate:
    template_id = "d" * 64

    def __init__(self, path):
        self.disk_path = path
        self.manifest_path = Path(str(path) + ".json")


class PreparedMicroVMTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.seed = self.root / "seed"
        (self.seed / "boltons").mkdir(parents=True)
        (self.seed / "boltons" / "strutils.py").write_text("def singularize(x): return x[:-1]\n")
        self.verifier = self.root / "verifier"
        self.verifier.mkdir()
        (self.verifier / "verify.json").write_text(json.dumps({
            "kind": "command_cases_v1", "cases": [{"argv": ["python3", "-B", "-c", "print(1)"],
                                                   "expected_stdout": "1\n",
                                                   "expected_returncode": 0}]}))
        self.kernel = self.root / "kernel"
        self.initramfs = self.root / "initramfs"
        self.modloop = self.root / "modloop"
        self.disk = self.root / "pristine.qcow2"
        for path, content in ((self.kernel, b"kernel"), (self.initramfs, b"initramfs"),
                              (self.modloop, b"modloop"), (self.disk, b"qcow2 seed")):
            path.write_bytes(content)
        self.runtime = MicroVMRuntime(
            self.kernel, self.initramfs, self.disk,
            kernel_sha256=_sha256_file(self.kernel),
            initramfs_sha256=_sha256_file(self.initramfs),
            readonly_disk_paths=(self.modloop,))
        self.adapter = MicroVMCodingAdapter(
            self.runtime, verifier_dir=self.verifier,
            visible_check=["python3", "-B", "-c", "import boltons.strutils"])
        now = datetime.now(timezone.utc)
        self.task = {
            "schema_version": "realworld-0.1", "task_id": "boltons-26-singularize-ss-v1",
            "event_id": "boltons-26-singularize-ss", "cluster_id": "boltons-strutils-singularize",
            "split": "train", "prompt": "Repair the pinned Boltons fixture.",
            "issued_at": (now - timedelta(minutes=1)).isoformat(),
            "action_deadline": (now + timedelta(minutes=10)).isoformat(),
            "outcome_not_before": (now - timedelta(minutes=1)).isoformat(),
            "verify_after": (now - timedelta(minutes=1)).isoformat(),
            "tool_manifest": [{"name": name, "description": name} for name in
                              ("list_files", "read_file", "write_file", "run_visible_checks", "submit")],
            "reward_contract": {"id": "host_cases", "description": "Private cases",
                                "min_reward": 0, "max_reward": 1},
            "budgets": {"max_actions": 8, "max_wall_seconds": 300},
            "is_fixture": True,
            "metadata": {"source_sdist_sha256": prepared_module._SOURCE_SDIST_SHA256,
                         "artifact_binding": self.adapter.artifact_binding()},
        }
        # These synthetic tiny files exercise the mechanism without bundling
        # the full pinned Boltons source or private verifier into unit tests.
        for name, value in (
            ("_SEED_TREE_SHA256", _workspace_digest(self.seed)),
            ("_SEED_FILE_SHA256", _sha256_file(self.seed / "boltons" / "strutils.py")),
            ("_VERIFIER_SHA256", _sha256_file(self.verifier / "verify.json")),
            ("_ROOTFS_SHA256", _sha256_file(self.disk)),
            ("_KERNEL_SHA256", _sha256_file(self.kernel)),
            ("_INITRAMFS_SHA256", _sha256_file(self.initramfs)),
            ("_MODLOOP_SHA256", _sha256_file(self.modloop)),
        ):
            setting = patch.object(prepared_module, name, value)
            setting.start()
            self.addCleanup(setting.stop)
        self.assets = self.root / "assets.json"
        self.assets.write_text(json.dumps({
            "schema_version": "boltons-microvm-assets-v2",
            "task_id": self.task["task_id"],
            "source_sdist_sha256": self.task["metadata"]["source_sdist_sha256"],
            "seed_workspace_sha256": _workspace_digest(self.seed),
            "rootfs_qcow2_sha256": _sha256_file(self.disk),
            "modloop_disk_sha256": _sha256_file(self.modloop),
            "alpine_sha256": {"vmlinuz-virt": _sha256_file(self.kernel),
                              "initramfs-virt": _sha256_file(self.initramfs),
                              "modloop-virt": _sha256_file(self.modloop)},
        }))

    def prepare(self):
        def fake_reset(adapter, task, *, now):
            adapter.started = True
            adapter.expected_binding = adapter.artifact_binding()
            adapter.block_devices = {"squashfs": "/dev/vda", "ext4": "/dev/vdb"}

        def fake_export(parent, disk_path, *, tag):
            self.assertEqual(tag, "prepared")
            result = _FakeTemplate(Path(disk_path))
            result.disk_path.write_bytes(b"sealed snapshot")
            result.manifest_path.write_text("{}")
            return result

        with patch.object(MicroVMCodingAdapter, "reset", fake_reset), \
                patch.object(MicroVMCodingAdapter, "_guest_ok", return_value=""), \
                patch("future_prediction_bench.prepared_microvm._guest_source_digest",
                      return_value=_sha256_file(self.seed / "boltons" / "strutils.py")), \
                patch("future_prediction_bench.prepared_microvm._processes", return_value=BASELINE), \
                patch("future_prediction_bench.prepared_microvm.MicroVMTemplate.export",
                      side_effect=fake_export):
            return PreparedMicroVMTemplate.prepare(
                self.adapter, self.task, seed_dir=self.seed,
                assets_manifest=self.assets,
                template_disk_path=self.root / "prepared.qcow2")

    def test_clean_task_and_asset_binding_fail_closed(self):
        self.assertEqual(prepared_module._SOURCE_SDIST_SHA256,
                         "5566d6cfd5a1e873d8e8476496287a9f92979964611ad9a9cecb6b0ef29b1edd")
        bad = copy.deepcopy(self.task)
        bad["metadata"]["source_sdist_sha256"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "pinned public Boltons"):
            _clean_task(bad)
        bad = copy.deepcopy(self.task)
        bad["metadata"]["artifact_binding"]["stateless_verifier_kind"] = "x"
        with self.assertRaisesRegex(ValueError, "stateless binding"):
            _clean_task(bad)
        bad["metadata"]["artifact_binding"].update({
            "stateless_verifier_kind": "stateless_python_cases_overlay_v1",
            "stateless_task_source_sha256": "a" * 64,
            "stateless_contract_sha256": "b" * 64,
            "stateless_helper_sha256": "c" * 64,
        })
        with self.assertRaisesRegex(ValueError, "stateless binding"):
            _clean_task(bad)
        self.assertEqual(_clean_task(bad, allow_stateless=True)[0]["task_id"],
                         self.task["task_id"])
        del bad["metadata"]["artifact_binding"]["stateless_helper_sha256"]
        with self.assertRaisesRegex(ValueError, "stateless binding"):
            _clean_task(bad, allow_stateless=True)
        bad = copy.deepcopy(self.task)
        bad["task_id"] = "different-task"
        with self.assertRaisesRegex(ValueError, "pinned public Boltons"):
            _clean_task(bad)
        self.disk.write_bytes(b"changed seed")
        with self.assertRaisesRegex(ValueError, "frozen task artifacts"):
            self.prepare()

    def test_self_consistent_replacement_rootfs_still_fails_exact_pin(self):
        self.disk.write_bytes(b"different but self-consistent qcow2")
        self.task["metadata"]["artifact_binding"] = self.adapter.artifact_binding()
        manifest = json.loads(self.assets.read_text())
        manifest["rootfs_qcow2_sha256"] = _sha256_file(self.disk)
        self.assets.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "assets differ from the pinned"):
            self.prepare()

    def test_prepared_manifest_rejects_tampering_and_changed_private_verifier(self):
        prepared = self.prepare()
        opened = PreparedMicroVMTemplate.open(
            prepared.manifest_path, expected_prepared_id=prepared.prepared_id)
        self.assertEqual(opened.prepared_id, prepared.prepared_id)
        wrong_task = copy.deepcopy(self.task)
        wrong_task["prompt"] += " changed"
        with self.assertRaisesRegex(ValueError, "requested task"):
            opened.spawn_adapters(wrong_task, [self.root / "child.qcow2"])
        (self.verifier / "verify.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "source or verifier changed"):
            opened.spawn_adapters(self.task, [self.root / "child.qcow2"])
        record = json.loads(prepared.manifest_path.read_text())
        record["seed_file_sha256"] = "0" * 64
        prepared.manifest_path.chmod(0o600)
        prepared.manifest_path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "manifest digest differs"):
            opened.spawn_adapters(self.task, [self.root / "child.qcow2"])

    def test_spawned_adapter_uses_same_realworld_opening_and_rejects_background_state(self):
        prepared = self.prepare()
        child = self.root / "child.qcow2"
        child.write_bytes(b"sealed child disk")
        child_runtime = MicroVMRuntime(
            self.kernel, self.initramfs, child,
            kernel_sha256=_sha256_file(self.kernel),
            initramfs_sha256=_sha256_file(self.initramfs),
            readonly_disk_paths=(self.modloop,))

        class FakeOpened:
            def spawn(self, paths, *, max_workers):
                self.assertion = (paths, max_workers)
                return [child_runtime]

        with patch("future_prediction_bench.prepared_microvm.MicroVMTemplate.open",
                   return_value=FakeOpened()):
            adapters = prepared.spawn_adapters(self.task, [child], max_workers=1)
        self.assertEqual(len(adapters), 1)
        adapter = adapters[0]
        self.assertIsInstance(adapter, _PreparedCodingAdapter)
        env = RealWorldEnv(self.task, adapter)
        with patch.object(_PreparedCodingAdapter, "_guest_ok", return_value=""), \
                patch.object(_PreparedCodingAdapter, "_check_no_symlinks"), \
                patch("future_prediction_bench.prepared_microvm._guest_source_digest",
                      return_value=_sha256_file(self.seed / "boltons" / "strutils.py")), \
                patch("future_prediction_bench.prepared_microvm._processes", return_value=BASELINE):
            opening = env.reset("policy")
        self.assertEqual(opening["observation"]["runtime_kind"],
                         "qemu_hvf_full_vm_qcow2_v1")
        self.assertEqual(opening["task"]["task_sha256"],
                         prepared._verified()["task_sha256"])
        adapter.close()

        second = _PreparedCodingAdapter(child_runtime, record=prepared._verified(),
                                        verifier_dir=self.verifier)
        other = RealWorldEnv(self.task, second)
        with patch.object(_PreparedCodingAdapter, "_guest_ok", return_value=""), \
                patch.object(_PreparedCodingAdapter, "_check_no_symlinks"), \
                patch("future_prediction_bench.prepared_microvm._guest_source_digest",
                      return_value=_sha256_file(self.seed / "boltons" / "strutils.py")), \
                patch("future_prediction_bench.prepared_microvm._processes",
                      return_value={(1, 100): "/usr/bin/busybox", (3, 101): "/bin/python"}):
            with self.assertRaisesRegex(ValueError, "Adapter reset failed"):
                other.reset("policy")
        self.assertEqual(other.status, "setup_error")
        mismatched = _PreparedCodingAdapter(child_runtime, record=prepared._verified(),
                                            verifier_dir=self.verifier)
        changed = copy.deepcopy(self.task)
        changed["prompt"] += " different"
        with patch.object(child_runtime, "close") as close:
            with self.assertRaisesRegex(ValueError, "task differs"):
                mismatched.reset(validate_task(changed), now=None)
            close.assert_called_once()

    def test_adapter_construction_failure_cleans_spawned_disk(self):
        prepared = self.prepare()
        child = self.root / "child-on-failure.qcow2"
        child.write_bytes(b"sealed child disk")
        child_runtime = MicroVMRuntime(
            self.kernel, self.initramfs, child,
            kernel_sha256=_sha256_file(self.kernel),
            initramfs_sha256=_sha256_file(self.initramfs),
            readonly_disk_paths=(self.modloop,))

        class FakeOpened:
            def spawn(self, paths, *, max_workers):
                return [child_runtime]

        with patch("future_prediction_bench.prepared_microvm.MicroVMTemplate.open",
                   return_value=FakeOpened()), \
                patch("future_prediction_bench.prepared_microvm._PreparedCodingAdapter",
                      side_effect=ValueError("constructor failed")):
            with self.assertRaisesRegex(ValueError, "constructor failed"):
                prepared.spawn_adapters(self.task, [child])
        self.assertFalse(child.exists())

    def test_exclusive_sidecar_create_race_never_deletes_another_writer(self):
        target = Path(str((self.root / "prepared.qcow2").resolve()) + ".prepared.json")
        original_open = Path.open

        def raced_open(path, mode="r", *args, **kwargs):
            if Path(path) == target and mode == "x":
                with original_open(target, "w", encoding="utf-8") as stream:
                    stream.write("other writer's file")
                raise FileExistsError(target)
            return original_open(path, mode, *args, **kwargs)

        with patch.object(Path, "open", raced_open):
            with self.assertRaises(FileExistsError):
                self.prepare()
        self.assertEqual(target.read_text(), "other writer's file")
        self.assertFalse(Path(str(target).removesuffix(".prepared.json")).exists())
        self.assertFalse(Path(str(target).removesuffix(".prepared.json") + ".json").exists())

    def test_helper_only_preinstall_seals_no_hidden_case_code(self):
        with self.assertRaisesRegex(ValueError, "requires an explicit stateless"):
            PreparedMicroVMTemplate.prepare(
                self.adapter, self.task, seed_dir=self.seed,
                assets_manifest=self.assets,
                template_disk_path=self.root / "invalid.qcow2",
                preinstall_stateless_helper=True)
        helper_sha = "c" * 64
        binding = copy.deepcopy(self.task["metadata"]["artifact_binding"])
        binding.update({"stateless_verifier_kind": "stateless_python_cases_overlay_v1",
                        "stateless_task_source_sha256": "a" * 64,
                        "stateless_contract_sha256": "b" * 64,
                        "stateless_helper_sha256": helper_sha})
        self.task["metadata"]["artifact_binding"] = binding
        calls = []
        verifier = SimpleNamespace(batch_code_sha256=None)
        verifier.install = lambda: calls.append("install_generic_helper") or helper_sha
        self.adapter.stateless_verifier = verifier
        self.adapter.stateless_contract_path = self.root / "contract.json"
        self.adapter.stateless_contract_path.write_text("{}")
        self.adapter.stateless_task_path = self.root / "task.json"
        self.adapter.stateless_task_path.write_text("{}")
        self.adapter.artifact_binding = lambda: copy.deepcopy(binding)

        def fake_reset(adapter, task, *, now):
            adapter.started = True
            adapter.expected_binding = binding
            adapter.block_devices = {"squashfs": "/dev/vda", "ext4": "/dev/vdb"}

        def fake_export(parent, disk_path, *, tag):
            calls.append("export")
            result = _FakeTemplate(Path(disk_path))
            result.disk_path.write_bytes(b"sealed snapshot")
            result.manifest_path.write_text("{}")
            return result

        def guest_ok(command, *, timeout=None):
            calls.append(command)
            return ""

        with patch.object(MicroVMCodingAdapter, "reset", fake_reset), \
                patch.object(MicroVMCodingAdapter, "_guest_ok", side_effect=guest_ok), \
                patch("future_prediction_bench.prepared_microvm._guest_source_digest",
                      return_value=_sha256_file(self.seed / "boltons" / "strutils.py")), \
                patch("future_prediction_bench.prepared_microvm._guest_file_sha",
                      return_value=helper_sha), \
                patch("future_prediction_bench.prepared_microvm._processes", return_value=BASELINE), \
                patch("future_prediction_bench.prepared_microvm.MicroVMTemplate.export",
                      side_effect=fake_export):
            prepared = PreparedMicroVMTemplate.prepare(
                self.adapter, self.task, seed_dir=self.seed,
                assets_manifest=self.assets,
                template_disk_path=self.root / "helper-only.qcow2",
                preinstall_stateless_helper=True)
        record = prepared._verified()
        self.assertEqual(record["preinstalled_stateless_helper_sha256"], helper_sha)
        self.assertLess(calls.index("install_generic_helper"), calls.index("export"))
        self.assertTrue(any("test ! -e /mnt/root/fpb_stateless_cases.json" in call
                            for call in calls))
        self.assertIsNone(verifier.batch_code_sha256)

    def test_child_rechecks_preinstalled_helper_before_submit(self):
        child = object.__new__(_PreparedCodingAdapter)
        child._prepared_record = {"preinstalled_stateless_helper_sha256": "c" * 64}
        child.stateless_verifier = SimpleNamespace(installed=False, program_sha256=None)
        calls = []
        with patch.object(_PreparedCodingAdapter, "_guest_ok",
                          side_effect=lambda command: calls.append(command) or ""), \
                patch("future_prediction_bench.prepared_microvm._guest_file_sha",
                      return_value="c" * 64):
            child._attest_preinstalled_helper()
        self.assertTrue(child.stateless_verifier.installed)
        self.assertEqual(child.stateless_verifier.program_sha256, "c" * 64)
        self.assertEqual(calls, ["test ! -e /mnt/root/fpb_stateless_cases.json"])
        with patch.object(_PreparedCodingAdapter, "_guest_ok", return_value=""), \
                patch("future_prediction_bench.prepared_microvm._guest_file_sha",
                      return_value="d" * 64):
            with self.assertRaisesRegex(RuntimeError, "helper_changed"):
                child._attest_preinstalled_helper()


if __name__ == "__main__":
    unittest.main()
