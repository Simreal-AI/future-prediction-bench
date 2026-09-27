"""Focused checks for the offline public-materials audit."""

import unittest
from pathlib import Path
from contextlib import contextmanager
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from unittest.mock import patch

from scripts import audit_materials, package_materials


class MaterialsAuditTests(unittest.TestCase):
    @contextmanager
    def clean_checkout(self):
        """Exercise only files shipped in the public source archive."""
        source_root = Path(__file__).resolve().parents[1]
        manifest = json.loads((source_root / "scripts/public_manifest.json").read_text())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve() / "checkout"
            for name in manifest["files"]:
                dest = root / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_root / name, dest)
            with patch.object(package_materials, "ROOT", root), patch.object(audit_materials, "ROOT", root):
                yield root

    def rewrite_bundle(self, bundle, changes):
        with zipfile.ZipFile(bundle) as zipped:
            members = {name: zipped.read(name) for name in zipped.namelist()}
        members.update(changes)
        with zipfile.ZipFile(bundle, "w") as zipped:
            for name, data in members.items():
                package_materials._add_bytes(zipped, data, name)
        bundle.with_name(bundle.name + ".sha256").write_text(
            f"{audit_materials._digest(bundle.read_bytes())}  {bundle.name}\n"
        )

    def test_clean_public_checkout_builds_and_audits_without_briefing(self):
        with self.clean_checkout() as root:
            self.assertFalse((root / "reports").exists())
            result = package_materials.build_materials()
            bundle = Path(result["materials_zip"])
            report = audit_materials.audit(bundle, run_tests=False)
            self.assertEqual(report["findings"], [])
            self.assertEqual(report["status"], "pass")
            self.assertFalse(result["supplemental_briefing"])
            with zipfile.ZipFile(bundle) as zipped:
                self.assertFalse(any("/briefing/" in name for name in zipped.namelist()))
                self.assertNotIn(b"not been published", zipped.read("future-prediction-bench-materials/README.md"))

            # Rebuild with the actual CLI from its own extracted public ZIP.
            with zipfile.ZipFile(result["source_zip"]) as zipped:
                zipped.extractall(root / "extracted")
            extracted_root = root / "extracted/future-prediction-bench"
            process = subprocess.run(
                [sys.executable, "scripts/package_materials.py"], cwd=extracted_root,
                capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            rebuilt = json.loads(process.stdout)
            self.assertEqual(Path(rebuilt["source_zip"]).read_bytes(), Path(result["source_zip"]).read_bytes())
            process = subprocess.run(
                [sys.executable, "scripts/audit_materials.py", rebuilt["materials_zip"], "--skip-tests"],
                cwd=extracted_root, capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(process.returncode, 0, process.stderr + process.stdout)

    def test_local_briefing_is_excluded_unless_explicitly_requested(self):
        with self.clean_checkout() as root:
            briefing = root / "reports" / package_materials.BRIEFING_NAME
            briefing.parent.mkdir()
            briefing.write_text("# 项目成果\n", encoding="utf-8")
            result = package_materials.build_materials()
            self.assertFalse(result["supplemental_briefing"])
            report = audit_materials.audit(Path(result["materials_zip"]), run_tests=False)
            self.assertEqual(report["status"], "pass")
            self.assertFalse(report["supplemental_briefing"])

    def test_explicit_briefing_is_separate_and_checksums_are_audited(self):
        with self.clean_checkout() as root:
            briefing = root / "reports" / package_materials.BRIEFING_NAME
            briefing.parent.mkdir()
            briefing.write_text("# 项目成果\n", encoding="utf-8")
            requested = f"reports/{package_materials.BRIEFING_NAME}"
            result = package_materials.build_materials(requested)
            bundle = Path(result["materials_zip"])
            report = audit_materials.audit(bundle, run_tests=False, briefing_path=requested)
            self.assertEqual(report["status"], "pass")
            self.assertTrue(report["supplemental_briefing"])
            with zipfile.ZipFile(result["source_zip"]) as zipped:
                self.assertFalse(any("project-highlights-zh.md" in name for name in zipped.namelist()))
            self.assertIn("materials: member list differs from public allowlist",
                          audit_materials.audit(bundle, run_tests=False)["findings"])
            member = "future-prediction-bench-materials/briefing/" + package_materials.BRIEFING_NAME
            self.rewrite_bundle(bundle, {member: b"Unexpected replacement\n"})
            report = audit_materials.audit(bundle, run_tests=False, briefing_path=requested)
            self.assertIn("materials: Chinese briefing differs from reviewed file", report["findings"])

    def test_rejects_unsafe_or_unreviewed_briefing_paths(self):
        with self.clean_checkout() as root:
            briefing = root / "reports" / package_materials.BRIEFING_NAME
            briefing.parent.mkdir()
            briefing.write_text("# Summary\n", encoding="utf-8")
            unsafe = ["../reports/project-highlights-zh.md", "reports/../reports/project-highlights-zh.md",
                      "reports/private.md", "reports\\project-highlights-zh.md", root.parent / package_materials.BRIEFING_NAME]
            for path in unsafe:
                with self.subTest(path=str(path)), self.assertRaises(ValueError):
                    package_materials._briefing_file(path)
            self.assertEqual(package_materials._briefing_file(briefing), briefing)
            briefing.unlink()
            other = root / "private.md"
            other.write_text("private\n", encoding="utf-8")
            briefing.symlink_to(other)
            with self.assertRaisesRegex(ValueError, "unsafe"):
                package_materials._briefing_file(f"reports/{package_materials.BRIEFING_NAME}")

    def test_rejects_unexpected_supplemental_archive_data(self):
        with self.clean_checkout():
            result = package_materials.build_materials()
            bundle = Path(result["materials_zip"])
            self.rewrite_bundle(bundle, {"future-prediction-bench-materials/briefing/private.md": b"extra\n"})
            report = audit_materials.audit(bundle, run_tests=False)
            self.assertIn("materials: member list differs from public allowlist", report["findings"])

    def test_explicit_briefing_still_requires_privacy_audit(self):
        with self.clean_checkout() as root:
            briefing = root / "reports" / package_materials.BRIEFING_NAME
            briefing.parent.mkdir()
            briefing.write_bytes(b"Private location: /" + b"Users/Someone/hidden/file\n")
            result = package_materials.build_materials(briefing, output_dir=root / "staged")
            bundle = Path(result["materials_zip"])
            self.assertEqual(bundle.parent, root / "staged")
            report = audit_materials.audit(bundle, run_tests=False, briefing_path=briefing)
            member = "future-prediction-bench-materials/briefing/" + package_materials.BRIEFING_NAME
            self.assertIn(member + ": personal_absolute_path", report["findings"])

    def test_rejects_unsafe_archive_members(self):
        self.assertFalse(audit_materials._safe_name("../private.json"))
        self.assertFalse(audit_materials._safe_name("/absolute.json"))
        self.assertFalse(audit_materials._safe_name("C:/absolute.json"))
        self.assertFalse(audit_materials._safe_name("folder\\private.json"))
        self.assertTrue(audit_materials._safe_name("future-prediction-bench/docs/VALIDATION.md"))

    def test_reports_private_data_without_echoing_value(self):
        errors = []
        name = "future-prediction-bench/docs/example.md"
        data = b"a personal location: /" + b"Users/Someone/hidden/file"
        audit_materials._audit_text(name, data, errors)
        self.assertEqual(errors, [f"{name}: personal_absolute_path"])

    def test_digest_field_must_be_hex_sha256(self):
        errors = []
        audit_materials._audit_digests("report.json", {"asset_binding": {"source_sdist_sha256": "short"}}, errors)
        self.assertEqual(errors, ["report.json: malformed digest at /asset_binding/source_sdist_sha256"])

    def test_markdown_link_cannot_escape_source_root(self):
        errors = []
        audit_materials._audit_links({
            "future-prediction-bench/README.md": b"[bad](../../outside.md)"
        }, errors)
        self.assertEqual(errors, [
            "future-prediction-bench/README.md: relative link escapes source root"
        ])

    def test_package_rejects_new_unreviewed_public_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            scripts = root / "scripts"
            scripts.mkdir()
            manifest = scripts / "public_manifest.json"
            manifest.write_text(json.dumps({
                "schema_version": 1,
                "files": ["scripts/public_manifest.json"],
            }))
            unexpected = root / "future_prediction_bench" / "experimental.py"
            with patch.object(package_materials, "ROOT", root), patch.object(
                package_materials, "_candidate_public_files",
                return_value=[manifest, unexpected],
            ):
                with self.assertRaisesRegex(ValueError, "Unreviewed public candidate"):
                    package_materials._public_files()

    def test_example_json_contract_requires_explicit_review(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            contract = root / "examples" / "new_runtime" / "contract.json"
            contract.parent.mkdir(parents=True)
            contract.write_text('{"kind":"new"}', encoding="utf-8")
            with patch.object(package_materials, "ROOT", root):
                candidates = {
                    path.relative_to(root).as_posix()
                    for path in package_materials._candidate_public_files()
                }
            self.assertIn("examples/new_runtime/contract.json", candidates)

    def test_manifest_cannot_add_ignored_upstream_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            scripts = root / "scripts"
            scripts.mkdir()
            manifest = scripts / "public_manifest.json"
            ignored = root / "runs" / "upstream" / "source.py"
            ignored.parent.mkdir(parents=True)
            ignored.write_text("# copied upstream code\n", encoding="utf-8")
            manifest.write_text(json.dumps({
                "schema_version": 1,
                "files": ["runs/upstream/source.py", "scripts/public_manifest.json"],
            }), encoding="utf-8")
            with patch.object(package_materials, "ROOT", root), patch.object(
                package_materials, "_candidate_public_files", return_value=[manifest]
            ):
                with self.assertRaisesRegex(ValueError, "outside reviewed candidate roots"):
                    package_materials._public_files()

    def test_curated_vm_reports_state_asset_provenance(self):
        report_dir = Path(__file__).resolve().parents[1] / "docs" / "measurements"
        reports = (sorted(report_dir.glob("microvm_*.json"))
                   + sorted(report_dir.glob("guest_*.json"))
                   + sorted(report_dir.glob("stateless_*.json"))
                   + sorted(report_dir.glob("prepared_*.json")))
        self.assertGreaterEqual(len(reports), 9)
        for path in reports:
            with self.subTest(path=path.name):
                errors = []
                audit_materials._audit_text(
                    f"future-prediction-bench/docs/measurements/{path.name}",
                    path.read_bytes(), errors,
                )
                self.assertEqual(errors, [])
                binding = json.loads(path.read_text())["asset_binding"]
                if path.name.startswith("guest_cow_"):
                    self.assertIsNone(binding["source_sdist_sha256"])
                    self.assertIsNone(binding["seed_workspace_sha256"])
                else:
                    self.assertEqual(len(binding["source_sdist_sha256"]), 64)
                    self.assertEqual(len(binding["seed_workspace_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
