"""Static contract and asset-rebinding checks for the separate v2 fixture."""

import copy
import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from examples.realworld_boltons26 import make_task as v1_module
from examples.realworld_boltons26 import rebind_assets_v2 as assets_module
from examples.realworld_boltons26.benchmark_small_edit_v2 import (
    _actions, _paired_semantics,
)
from examples.realworld_boltons26.make_task_v2 import (
    NEW_TEXT, OLD_TEXT, TASK_ID, make_task_v2,
)
from future_prediction_bench import prepared_microvm as pins
from future_prediction_bench.coding_env import _workspace_digest
from future_prediction_bench.microvm_runtime import _sha256_file
from future_prediction_bench.realworld import validate_task


SOURCE = ("def singularize(word):\n"
          "    orig_word = word\n"
          "    if word.endswith('ies'):\n"
          "        singular = word[:-3]\n"
          + OLD_TEXT + "\n")


def _archive_bytes():
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for name, payload in (
            ("LICENSE", b"BSD-3-Clause\n"),
            ("boltons/strutils.py", SOURCE.encode("utf-8")),
        ):
            info = tarfile.TarInfo("boltons-26.0.0/" + name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return output.getvalue()


class BoltonsFixtureV2Tests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        data = _archive_bytes()
        self.archive = self.root / "fake-sdist.tar.gz"
        self.archive.write_bytes(data)
        self.archive_sha = hashlib.sha256(data).hexdigest()
        self.task_dir = self.root / "v2-task"
        with patch.object(v1_module, "ARCHIVE_SHA256", self.archive_sha):
            make_task_v2(self.task_dir, sdist=self.archive)

    def test_both_edits_share_one_task_and_exact_final_bytes(self):
        task = json.loads((self.task_dir / "task.json").read_text(encoding="utf-8"))
        self.assertEqual(task["task_id"], TASK_ID)
        self.assertEqual({tool["name"] for tool in task["tool_manifest"]},
                         pins._TASK_TOOLS[TASK_ID])
        actions, original_sha, repaired_sha = _actions(self.task_dir, SOURCE)
        self.assertEqual(actions["replace_text"][1]["expected_file_sha256"], original_sha)
        self.assertEqual(repaired_sha, hashlib.sha256(
            SOURCE.replace(OLD_TEXT, NEW_TEXT).encode("utf-8")).hexdigest())
        self.assertEqual(actions["full_write"][1]["content"],
                         SOURCE.replace(OLD_TEXT, NEW_TEXT))
        with patch.object(pins, "_SOURCE_SDIST_SHA256", self.archive_sha), \
                patch.object(pins, "_VERIFIER_SHA256", "a" * 64):
            task["metadata"]["artifact_binding"] = {
                "runtime_kind": "qemu_hvf_full_vm_qcow2_v1",
                "verifier_sha256": "a" * 64,
            }
            self.assertEqual(pins._clean_task(task)[0]["task_id"], TASK_ID)
            with patch.object(pins, "replace_text_helper_binding",
                              return_value={"source_sha256": "0" * 64,
                                            "guest_program_sha256": "0" * 64}):
                with self.assertRaisesRegex(ValueError, "text-edit helper differs"):
                    pins._clean_task(task)
            with patch.object(pins, "_sha256_file", return_value="0" * 64):
                with self.assertRaisesRegex(ValueError, "text-edit helper differs"):
                    pins._clean_task(task)
            wrong_helper = copy.deepcopy(task)
            wrong_helper["metadata"]["replace_text_helper_binding"]["source_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "text-edit helper differs"):
                pins._clean_task(wrong_helper)
            bad = copy.deepcopy(task)
            bad["tool_manifest"] = [tool for tool in bad["tool_manifest"]
                                    if tool["name"] != "replace_text"]
            with self.assertRaisesRegex(ValueError, "pinned public Boltons"):
                pins._clean_task(bad)

    def test_v2_asset_rebind_requires_exact_v1_digest_and_task(self):
        fixture = self.task_dir
        contents = {
            "rootfs.qcow2": b"pinned qcow2",
            "vmlinuz-virt": b"pinned kernel",
            "initramfs-virt": b"pinned initramfs",
            "modloop-virt-padded.raw": b"pinned module disk",
        }
        old_assets = self.root / "v1-assets"
        old_assets.mkdir()
        for name, payload in contents.items():
            (old_assets / name).write_bytes(payload)
        hashes = {name: _sha256_file(old_assets / name) for name in contents}
        manifest = {
            "schema_version": "boltons-microvm-assets-v2",
            "task_id": pins._TASK_ID,
            "source_sdist_sha256": self.archive_sha,
            "seed_workspace_sha256": _workspace_digest(fixture / "seed"),
            "rootfs_qcow2_sha256": hashes["rootfs.qcow2"],
            "modloop_disk_sha256": hashes["modloop-virt-padded.raw"],
            "alpine_sha256": {
                "vmlinuz-virt": hashes["vmlinuz-virt"],
                "initramfs-virt": hashes["initramfs-virt"],
                "modloop-virt": hashes["modloop-virt-padded.raw"],
            },
        }
        (old_assets / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        patches = (
            patch.object(pins, "_SOURCE_SDIST_SHA256", self.archive_sha),
            patch.object(pins, "_SEED_TREE_SHA256", _workspace_digest(fixture / "seed")),
            patch.object(pins, "_SEED_FILE_SHA256", _sha256_file(
                fixture / "seed" / "boltons" / "strutils.py")),
            patch.object(pins, "_VERIFIER_SHA256", _sha256_file(
                fixture / "verifier" / "verify.json")),
            patch.object(pins, "_ROOTFS_SHA256", hashes["rootfs.qcow2"]),
            patch.object(pins, "_KERNEL_SHA256", hashes["vmlinuz-virt"]),
            patch.object(pins, "_INITRAMFS_SHA256", hashes["initramfs-virt"]),
            patch.object(pins, "_MODLOOP_SHA256", hashes["modloop-virt-padded.raw"]),
            patch.object(assets_module, "_ASSET_HASHES", hashes),
        )
        for setting in patches:
            setting.start()
            self.addCleanup(setting.stop)
        (self.root / "v2-assets").mkdir()  # An existing empty output is permitted.
        result = assets_module.rebind_assets_v2(
            fixture, old_assets, self.root / "v2-assets")
        self.assertEqual(result["task_id"], TASK_ID)
        rebound = json.loads((self.root / "v2-assets" / "manifest.json").read_text())
        self.assertEqual(rebound["task_id"], TASK_ID)
        self.assertEqual(rebound["rootfs_qcow2_sha256"], manifest["rootfs_qcow2_sha256"])
        (old_assets / "rootfs.qcow2").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "Pinned V1 asset changed"):
            assets_module.rebind_assets_v2(
                fixture, old_assets, self.root / "blocked-assets")

    def test_pair_verifier_detects_semantic_or_observation_difference(self):
        rows = {}
        for branch in ("repair", "baseline"):
            for method in ("full_write", "replace_text"):
                for condition in ("cold", "prepared"):
                    rows[(method, branch, condition)] = {
                        "task_sha256": "a" * 64,
                        "opening_observation": {"tools": ["write_file", "replace_text"]},
                        "action_observation_sha256s": (["read", "submit"]
                                                       if branch == "baseline"
                                                       else ["read", method, "check", "submit"]),
                        "evidence_kind": "command_cases_v1",
                        "case_results": [{"passed": branch == "repair"}] * 14,
                        "reward": 1.0 if branch == "repair" else 0.0,
                        "final_source_sha256": "b" * 64 if branch == "repair" else "c" * 64,
                    }
        _paired_semantics(rows, "a" * 64)
        rows[("replace_text", "repair", "prepared")]["action_observation_sha256s"] = ["different"]
        with self.assertRaisesRegex(RuntimeError, "action_observation_sha256s"):
            _paired_semantics(rows, "a" * 64)
        rows[("replace_text", "repair", "prepared")]["action_observation_sha256s"] = [
            "other-read", "replace_text", "check", "submit"]
        rows[("replace_text", "repair", "cold")]["action_observation_sha256s"] = [
            "other-read", "replace_text", "check", "submit"]
        with self.assertRaisesRegex(RuntimeError, "Common policy action observation"):
            _paired_semantics(rows, "a" * 64)

    def test_prepared_v2_child_enables_bound_replace_tool(self):
        task = json.loads((self.task_dir / "task.json").read_text(encoding="utf-8"))
        task["metadata"]["artifact_binding"] = {"runtime_kind": "qemu_hvf_full_vm_qcow2_v1"}
        frozen = validate_task(task)
        child = object.__new__(pins._PreparedCodingAdapter)
        child.started = False
        child.runtime = SimpleNamespace(close=lambda: None)
        child.workspace_root = child.WORKSPACE_ROOT
        child.visible_check = ("python3", "-B", "-c", "import boltons.strutils")
        child.stateless_verifier = None
        child._prepared_record = {
            "task_sha256": frozen["task_sha256"],
            "artifact_binding": frozen["metadata"]["artifact_binding"],
            "seed_file_sha256": "e" * 64,
            "baseline_processes": [[1, 100, "/usr/bin/busybox"],
                                   [2, 100, "/usr/bin/busybox"]],
            "block_devices": {"ext4": "/dev/vdb"},
            "preinstalled_stateless_helper_sha256": None,
        }
        child.artifact_binding = lambda: frozen["metadata"]["artifact_binding"]
        child._check_stateless_episode_task = lambda _task: None
        child._guest_ok = lambda _command: ""
        child._check_no_symlinks = lambda: None
        with patch.object(pins, "_guest_source_digest", return_value="e" * 64), \
                patch.object(pins, "_processes", return_value={
                    (1, 100): "/usr/bin/busybox", (2, 100): "/usr/bin/busybox"}):
            observation = child.reset(frozen, now=None)
        self.assertTrue(child.replace_text_enabled)
        self.assertIn("replace_text", observation["tools"])


if __name__ == "__main__":
    unittest.main()
