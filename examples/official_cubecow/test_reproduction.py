"""Guard tests plus optional real-report verification; no native fixture run.

Set FPB_CUBECOW_RESULT and both FPB_CUBECOW_EXPECTED_* SHA declarations to
exercise the preserved genuine report. Missing evidence is an explicit skip.
"""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location("public_cubecow_" + name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = load("prepare_build")
verify = load("verify_result")


class PublicBuildGuards(unittest.TestCase):
    def test_main_and_lock_are_exact_executed_bytes(self):
        self.assertEqual(build.sha_file(HERE / "src/main.rs"), build.MAIN_SHA)
        self.assertEqual(build.sha_file(HERE / "Cargo.lock"), build.LOCK_SHA)
        pins = json.loads((HERE / "source_pins.json").read_text())
        self.assertEqual(len(pins["upstream_crate_files_sha256"]), 34)
        self.assertEqual(len(build.registry_packages(HERE / "Cargo.lock")), 106)

    def test_canonical_overlap_and_symlinks_reject_before_creating_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source, other = root / "source", root / "other"
            source.mkdir()
            other.mkdir()
            marker = source / "keep"
            marker.write_bytes(b"original")
            descendant = build.canonical(other / ".." / "source" / "new", must_exist=False)
            with self.assertRaisesRegex(ValueError, "overlap"):
                build.require_separate(descendant, (source,))
            with self.assertRaisesRegex(ValueError, "overlap"):
                build.require_separate(root / "fresh", (root / "fresh" / "input",))
            link = root / "link"
            link.symlink_to(source, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                build.canonical(link / "new", must_exist=False)
            self.assertEqual(marker.read_bytes(), b"original")
            self.assertFalse(descendant.exists())

    def test_preexisting_output_and_input_symlink_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            output = root / "existing"
            output.mkdir()
            with self.assertRaisesRegex(ValueError, "fresh"):
                build.require_separate(output, ())
            target = root / "target"
            target.write_text("unchanged")
            (root / "linked-input").symlink_to(target)
            with self.assertRaisesRegex(ValueError, "nonregular"):
                build.inventory(root)
            self.assertEqual(target.read_text(), "unchanged")

    def test_original_compiler_links_are_recorded_without_following(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root / "original-loader-link").symlink_to("/not/a/real/host/file")
            result = build.compiler_inventory(root)
            self.assertEqual(result, {"original-loader-link": {"kind": "symlink", "target": "/not/a/real/host/file"}})

    def test_json_parser_rejects_duplicate_and_nonfinite_values(self):
        for value in ('{"status":"pass","status":"fail"}', '{"cost":NaN}', '{"cost":Infinity}'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                verify.parse_json(value)

    def test_binary_hashes_are_strict_explicit_declarations(self):
        for value in (None, True, "abc", "a" * 63, "A" * 64):
            with self.subTest(value=value), self.assertRaises(ValueError):
                verify.hash_pin(value, "fixture")


class GenuineReportAndNegatives(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = os.environ.get("FPB_CUBECOW_RESULT")
        fixture = os.environ.get("FPB_CUBECOW_EXPECTED_FIXTURE_SHA256")
        utility = os.environ.get("FPB_CUBECOW_EXPECTED_HASH_UTILITY_SHA256")
        if not all((path, fixture, utility)):
            raise unittest.SkipTest("actual result and two explicit operator SHA declarations not supplied")
        cls.path = verify.canonical(path)
        cls.actual = verify.parse_json(cls.path.read_text())
        cls.fixture, cls.utility = fixture, utility
        verify.hash_pin(fixture, "fixture")
        verify.hash_pin(utility, "hash utility")
        cls.oracle = verify.independent_oracle()

    def check(self, actual):
        return verify.verify_result(actual, self.oracle, fixture_sha256=self.fixture, hash_utility_sha256=self.utility)

    def test_actual_report_has_six_processes_and92_complete_witnesses(self):
        result = verify.verify_file(self.path, fixture_sha256=self.fixture, hash_utility_sha256=self.utility, oracle=self.oracle)
        self.assertEqual(result["verified_native_phase_count"], 6)
        self.assertEqual(result["verified_whole_file_witness_count"], 92)
        self.assertEqual(result["verified_full_bytes_compared_total"], 3485466624)
        self.assertEqual(result["raw_runtime_result_sha256"], hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertFalse(result["binary_declarations_are_execution_attestation"])

    def test_changed_binary_and_original_source_pins_are_rejected(self):
        for name in ("fixture_executable_sha256", "sha256sum_executable_sha256", "official_source_commit"):
            data = copy.deepcopy(self.actual)
            data[name] = "0" * len(data[name])
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.check(data)

    def test_phase_failure_missing_phase_and_pid_replay_are_rejected(self):
        for kind in ("missing", "exit", "pid"):
            data = copy.deepcopy(self.actual)
            phases = data["separate_native_process_phases"]
            if kind == "missing":
                phases.pop()
            elif kind == "exit":
                phases[0]["returncode"] = 1
            else:
                phases[1]["spawned_pid"] = phases[0]["spawned_pid"]
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.check(data)

    def test_raw_child_stdout_and_embedded_result_must_agree(self):
        data = copy.deepcopy(self.actual)
        data["separate_native_process_phases"][0]["result"]["pid"] += 1
        with self.assertRaisesRegex(ValueError, "stdout"):
            self.check(data)

    def test_complete_file_hash_length_and_backing_identity_are_required(self):
        for name, value in (("sha256", "0" * 64), ("full_bytes_compared", 4096),
                            ("path", "/tmp/fpb-xfs/populated/../foreign"), ("inode", True)):
            data = copy.deepcopy(self.actual)
            phase = data["separate_native_process_phases"][0]
            phase["result"]["cases"][0]["witnesses"][0][name] = value
            phase["stdout"] = json.dumps(phase["result"])
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.check(data)

    def test_mutation_resize_and_orphan_contracts_are_required(self):
        for index, key, value in ((1, "mutation_regions", {}), (2, "shrink_rejected", False),
                                  (4, "deleted_origin_list_is_empty", False), (5, "deleted_all_branches", False)):
            data = copy.deepcopy(self.actual)
            phase = data["separate_native_process_phases"][index]
            phase["result"]["cases"][0][key] = value
            phase["stdout"] = json.dumps(phase["result"])
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.check(data)


if __name__ == "__main__":
    unittest.main()
