"""Fixed GNU tar closure and opt-in exact archive staging checks."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tarfile
import tempfile
import unittest

from examples.official_crab_criu.fetch_inputs import load_inputs
from examples.official_crab_criu.prepare_guest import (
    ACL_LIBS_APK_SHA256, GNU_TAR_APK_SHA256, GNU_TAR_BINARY_SHA256,
    _extract_payload, _require_gnu_tar,
)


class FixedTarClosureTest(unittest.TestCase):
    def test_reviewed_manifest_has_exact_tar_dependency_provider(self):
        _, manifest = load_inputs()
        packages = {row["name"]: row for row in manifest["packages"]}
        self.assertEqual(packages["tar"]["version"], "1.35-r5")
        self.assertEqual(packages["tar"]["sha256"], GNU_TAR_APK_SHA256)
        self.assertEqual(packages["acl-libs"]["version"], "2.3.2-r1")
        self.assertEqual(packages["acl-libs"]["sha256"], ACL_LIBS_APK_SHA256)
        self.assertIn("so:libacl.so.1", packages["tar"]["dependencies"].split())
        self.assertIn("so:libacl.so.1=1.1.2302", packages["acl-libs"]["provides"].split())
        self.assertEqual(packages["acl-libs"]["dependencies"], "so:libc.musl-x86_64.so.1")
        self.assertNotIn("libattr", packages)


class RealPinnedTarStagingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = os.environ.get("FPB_CRIU_APK_DIR")
        if not directory:
            raise unittest.SkipTest("Set FPB_CRIU_APK_DIR for exact GNU tar APK checks")
        cls.source = Path(directory)
        _, manifest = load_inputs()
        cls.inputs = {row["name"]: row for row in manifest["packages"]}

    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.stage = Path(self.scratch.name) / "stage"
        self.stage.mkdir()
        registry = {}
        for name in ("acl-libs", "busybox", "tar"):
            row = self.inputs[name]
            path = self.source / row["filename"]
            self.assertEqual(path.stat().st_size, row["bytes"])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), row["sha256"])
            _extract_payload(path, self.stage, casefold_registry=registry)

    def test_original_gnu_tar_and_busybox_bytes_survive_staging(self):
        provenance = _require_gnu_tar(self.stage)
        self.assertEqual(provenance["path"], "/bin/tar")
        self.assertFalse(provenance["binary_executed_by_builder"])
        self.assertEqual(hashlib.sha256((self.stage / "bin/tar").read_bytes()).hexdigest(),
                         GNU_TAR_BINARY_SHA256)
        with tarfile.open(self.source / self.inputs["busybox"]["filename"],
                          ignore_zeros=True) as archive:
            original = archive.extractfile("bin/busybox").read()
        self.assertEqual((self.stage / "bin/busybox").read_bytes(), original)
        self.assertFalse((self.stage / "bin/tar").is_symlink())

    def test_busybox_tar_alias_is_rejected_without_changing_busybox(self):
        busybox = (self.stage / "bin/busybox").read_bytes()
        command = self.stage / "bin/tar"
        command.unlink()
        command.symlink_to("busybox")
        with self.assertRaisesRegex(ValueError, "original_gnu_tar_regular_binary_required"):
            _require_gnu_tar(self.stage)
        self.assertEqual((self.stage / "bin/busybox").read_bytes(), busybox)

    def test_earlier_path_alias_cannot_shadow_the_gnu_tool(self):
        shadow = self.stage / "usr/bin/tar"
        shadow.parent.mkdir(parents=True, exist_ok=True)
        shadow.symlink_to("../../bin/busybox")
        with self.assertRaisesRegex(ValueError, "gnu_tar_guest_path_shadow_rejected"):
            _require_gnu_tar(self.stage)
        self.assertEqual(hashlib.sha256((self.stage / "bin/tar").read_bytes()).hexdigest(),
                         GNU_TAR_BINARY_SHA256)

    def test_changed_acl_library_cannot_pass_original_dependency_check(self):
        library = self.stage / "usr/lib/libacl.so.1"
        payload = bytearray(library.read_bytes())
        payload[-1] ^= 1
        library.write_bytes(payload)
        with self.assertRaisesRegex(ValueError, "original_gnu_tar_acl_dependency_required"):
            _require_gnu_tar(self.stage)


if __name__ == "__main__":
    unittest.main()
