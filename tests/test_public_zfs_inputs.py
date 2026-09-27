"""Reject substituted cohorts and output aliases before offline extraction."""
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from examples.official_crab_criu import fetch_zfs_inputs as inputs
from examples.official_crab_criu.zfs_linux_builder import fixed_original_python_trees


class PublicZFSInputsTests(unittest.TestCase):
    def test_dot_dot_output_alias_cannot_overlap_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            cache = root / "cache"
            cache.mkdir()
            output = cache / "missing" / ".." / "new-output"
            with self.assertRaisesRegex(ValueError, "disjoint_cache"):
                inputs.fetch(output, cache=cache)
            self.assertFalse((cache / "new-output").exists())

    def test_symlink_ancestor_is_rejected_before_resolving(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "real").mkdir()
            (root / "alias").symlink_to(root / "real", target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "nonsymlink"):
                inputs.new_output_path(root / "alias" / "new-output")
            self.assertFalse((root / "real" / "new-output").exists())

    def test_self_certified_package_hash_is_rejected(self):
        plan = inputs.fixed_plan()
        plan["schema_version"] = "official-crab-zfs-inputs-v1"
        plan["packages"][0]["sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "inputs.json").write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, "package_record_mismatch"):
                inputs.validate_inputs(root)

    def test_substituted_package_path_is_not_opened(self):
        plan = inputs.fixed_plan()
        plan["schema_version"] = "official-crab-zfs-inputs-v1"
        plan["packages"][0]["filename"] = "../../unexpected.apk"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "inputs.json").write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, "package_record_mismatch"):
                inputs.validate_inputs(root)

    def test_different_kernel_cohort_is_rejected(self):
        plan = inputs.fixed_plan()
        plan["schema_version"] = "official-crab-zfs-inputs-v1"
        plan["kernel_release"] = "different-cohort"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "inputs.json").write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, "fixed_ZFS_cohort"):
                inputs.validate_inputs(root)

    def test_unchanged_but_unpinned_original_tree_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "crab").mkdir()
            (root / "crab" / "__init__.py").write_text("# substituted source\n")
            with self.assertRaisesRegex(ValueError, "fixed_original_Python_tree"):
                fixed_original_python_trees(root)

    @unittest.skipUnless(os.environ.get("FPB_ZFS_APK_DIR"), "genuine pinned ZFS APK input directory not configured")
    def test_genuine_cohort_and_single_byte_corruption(self):
        root = Path(os.environ["FPB_ZFS_APK_DIR"]).resolve()
        actual = inputs.validate_inputs(root)
        self.assertEqual(actual["kernel_release"], "6.18.53-0-virt")
        record = inputs.fixed_plan()["packages"][0]
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory).resolve() / record["filename"]
            shutil.copyfile(root / record["filename"], changed)
            with changed.open("r+b") as stream:
                first = stream.read(1)
                stream.seek(0)
                stream.write(bytes([first[0] ^ 1]))
            with self.assertRaisesRegex(ValueError, "whole_APK_pin"):
                inputs.apk_identity(changed, record)


if __name__ == "__main__":
    unittest.main()
