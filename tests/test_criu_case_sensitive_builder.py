"""Archive guards and opt-in real offline Linux xtables reconstruction."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest

from examples.official_crab_criu.prepare_guest import _extract_payload, _stash_xtables
from examples.official_crab_criu.linux_rootfs_builder import require_case_sensitive


def archive(path, records):
    with tarfile.open(path, "w:gz") as output:
        for name, content in records:
            member = tarfile.TarInfo(name)
            member.size = len(content)
            member.mode = 0o644
            output.addfile(member, io.BytesIO(content))


class ArchiveCaseGuardTest(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.stage = self.root / "stage"
        self.stage.mkdir()

    def test_unhandled_case_pair_fails_before_any_archive_extraction(self):
        path = self.root / "source.tar.gz"
        archive(path, [("usr/lib/Example.so", b"upper"), ("usr/lib/example.so", b"lower")])
        with self.assertRaisesRegex(ValueError, "unhandled_casefold_collision"):
            _extract_payload(path, self.stage)
        self.assertEqual(list(self.stage.iterdir()), [])

    def test_all_managed_case_pair_payloads_stay_off_host_stage(self):
        path = self.root / "source.tar.gz"
        archive(path, [("usr/lib/xtables/libxt_MARK.so", b"upper"),
                       ("usr/lib/xtables/libxt_mark.so", b"lower")])
        self.assertEqual(_extract_payload(path, self.stage, exclude_xtables=True), 0)
        self.assertEqual(list(self.stage.iterdir()), [])

    def test_implicit_parent_case_collision_fails_before_extraction(self):
        path = self.root / "source.tar.gz"
        archive(path, [("usr/Lib/a", b"upper parent"), ("usr/lib/b", b"lower parent")])
        with self.assertRaisesRegex(ValueError, "unhandled_casefold_collision"):
            _extract_payload(path, self.stage)
        self.assertEqual(list(self.stage.iterdir()), [])

    def test_plugin_payload_requires_explicit_managed_archive(self):
        path = self.root / "source.tar.gz"
        archive(path, [("usr/lib/xtables/libxt_MARK.so", b"payload")])
        with self.assertRaisesRegex(ValueError, "unexpected_plugin_payload"):
            _extract_payload(path, self.stage)
        self.assertEqual(list(self.stage.iterdir()), [])

    def test_cross_archive_collision_cannot_overwrite_previous_bytes(self):
        first, second = self.root / "first.tar.gz", self.root / "second.tar.gz"
        archive(first, [("usr/lib/Example.so", b"original")])
        archive(second, [("usr/lib/example.so", b"different")])
        registry = {}
        _extract_payload(first, self.stage, casefold_registry=registry)
        with self.assertRaisesRegex(ValueError, "unhandled_casefold_collision"):
            _extract_payload(second, self.stage, casefold_registry=registry)
        self.assertEqual((self.stage / "usr/lib/Example.so").read_bytes(), b"original")

    def test_case_probe_never_removes_preexisting_named_file(self):
        sentinel = self.stage / ".fpb-case-probe-A"
        sentinel.write_bytes(b"unowned sentinel")
        try:
            require_case_sensitive(self.stage)
        except ValueError as error:
            self.assertIn("case_sensitive", str(error))
        self.assertEqual(sentinel.read_bytes(), b"unowned sentinel")
        self.assertEqual(list(self.stage.iterdir()), [sentinel])


class RealPinnedPluginTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = os.environ.get("FPB_CRIU_APK_DIR")
        if not directory:
            raise unittest.SkipTest("Set FPB_CRIU_APK_DIR for real pinned APK checks")
        cls.source = Path(directory) / "iptables-1.8.13-r0.apk"
        if not cls.source.is_file():
            raise unittest.SkipTest("Pinned iptables APK absent")

    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.objects, self.manifest, self.digest, self.pairs = _stash_xtables(self.source, self.root)
        self.entries = json.loads(self.manifest.read_text())["entries"]

    def test_encoded_manifest_matches_every_original_archive_byte_and_link(self):
        expected = {}
        with tarfile.open(self.source, ignore_zeros=True) as source:
            for member in source.getmembers():
                if member.name.startswith("usr/lib/xtables/") and not member.isdir():
                    if member.isfile():
                        data = source.extractfile(member).read()
                        expected[member.name] = ("file", hashlib.sha256(data).hexdigest(), len(data))
                    else:
                        self.assertTrue(member.issym())
                        expected[member.name] = ("symlink", member.linkname)
        actual = {}
        for entry in self.entries:
            if entry["kind"] == "file":
                payload = (self.objects / entry["object"]).read_bytes()
                self.assertEqual(hashlib.sha256(payload).hexdigest(), entry["sha256"])
                actual[entry["path"]] = ("file", entry["sha256"], entry["bytes"])
            else:
                actual[entry["path"]] = ("symlink", entry["target"])
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), 121)
        self.assertEqual(len(self.pairs), 9)

    def test_real_linux_tmpfs_restores_all_case_distinct_plugin_paths(self):
        image = os.environ.get("FPB_CRIU_BUILD_IMAGE")
        if not image:
            self.skipTest("Set FPB_CRIU_BUILD_IMAGE for real offline Docker assembly")
        helper = Path(__file__).resolve().parents[1] / "examples/official_crab_criu/linux_rootfs_builder.py"
        stage = self.root / "stage"
        stage.mkdir()
        # Public plugin bytes only; permit the isolated utility to write its
        # verification JSON regardless of host/container UID mapping.
        self.root.chmod(0o777)
        result = subprocess.run(["docker", "run", "--rm", "--pull", "never", "--network", "none",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--read-only",
            "--tmpfs", "/linux-stage:rw,nosuid,nodev,size=32m",
            "--mount", f"type=bind,src={stage},dst=/input,readonly",
            "--mount", f"type=bind,src={self.objects},dst=/objects,readonly",
            "--mount", f"type=bind,src={helper},dst=/builder/linux_rootfs_builder.py,readonly",
            "--mount", f"type=bind,src={self.root},dst=/output", "--entrypoint", "python3.12", image,
            "-I", "-B", "/builder/linux_rootfs_builder.py", "--input-stage", "/input",
            "--objects", "/objects", "--plugin-manifest", "/output/xtables-manifest.json",
            "--plugin-manifest-sha256", self.digest, "--linux-stage", "/linux-stage/root",
            "--verification-output", "/output/linux-proof.json"],
            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        proof = json.loads((self.root / "linux-proof.json").read_text())
        self.assertTrue(proof["case_sensitive_probe_passed"])
        self.assertTrue(proof["exact_archive_paths_types_and_bytes"])
        self.assertEqual(proof["plugin_paths_verified"], 121)
        self.assertEqual(proof["regular_files_verified"], 115)
        self.assertEqual(proof["symlinks_verified"], 6)
        self.assertEqual(len(proof["casefold_collision_pairs_preserved"]), 9)
        self.assertFalse(proof["guest_binaries_executed"])
        self.assertFalse(proof["mke2fs_executed"])


if __name__ == "__main__":
    unittest.main()
