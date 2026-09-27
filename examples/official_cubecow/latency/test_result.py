"""Copied-evidence controls; these tests do not execute XFS or a native VM."""
import copy
import importlib.util
import json
import os
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("cube_latency_verifier", Path(__file__).with_name("verify_result.py"))
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


class ActualReportControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        filename = os.environ.get("FPB_CUBECOW_LATENCY_RESULT")
        fixture = os.environ.get("FPB_CUBECOW_LATENCY_FIXTURE_SHA256")
        utility = os.environ.get("FPB_CUBECOW_LATENCY_HASH_UTILITY_SHA256")
        if not filename or not fixture or not utility:
            raise unittest.SkipTest("Supply an actual raw result and independently recorded executable SHA declarations")
        cls.path = Path(filename)
        cls.original, cls.original_sha = verifier.load(cls.path)
        cls.oracle, _ = verifier.independent_oracle()
        cls.arguments = {
            "root": verifier.canonical_path(os.environ.get("FPB_CUBECOW_LATENCY_ROOT", "/tmp/fpb-xfs/latency")),
            "expected_fixture_sha256": fixture,
            "expected_hash_utility_sha256": utility,
            "expected_executable": os.environ.get("FPB_CUBECOW_LATENCY_EXECUTABLE", "/opt/fpb-cubecow/latency-fixture"),
        }
        verifier.verify_samples(cls.original, cls.oracle, **cls.arguments)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "original_sha"):
            if verifier.sha_file(cls.path) != cls.original_sha:
                raise AssertionError("The preserved actual raw input changed during copied-evidence controls")

    def reject(self, mutation, message):
        altered = copy.deepcopy(self.original)
        mutation(altered)
        with self.assertRaisesRegex(ValueError, message):
            verifier.verify_samples(altered, self.oracle, **self.arguments)

    def test_accept_all_actual_samples(self):
        result = verifier.verify_samples(self.original, self.oracle, **self.arguments)
        self.assertEqual(result["verified_method_trials"], 240)
        self.assertEqual(result["verified_whole_file_witness_count"], 2720)

    def test_missing_sample_rejected(self):
        self.reject(lambda r: r["samples"].pop(), "incomplete or extra")

    def test_extra_sample_rejected(self):
        self.reject(lambda r: r["samples"].append(r["samples"][-1]), "incomplete or extra")

    def test_failed_sample_rejected(self):
        self.reject(lambda r: r["samples"][0].update(status="fail"), "sample sequence, status")

    def test_duplicate_pair_rejected(self):
        self.reject(lambda r: r["samples"][2].update(pair=0), "sample sequence, status")

    def test_unbalanced_execution_order_rejected(self):
        self.reject(lambda r: r["samples"][0].update(order=["full-copy", "original"]), "sample sequence, status")

    def test_warmup_inclusion_rejected(self):
        self.reject(lambda r: r["samples"][0].update(warmup=False), "sample sequence, status")

    def test_boolean_duration_rejected(self):
        self.reject(lambda r: r["samples"][0].update(operation_return_ns=True), "invalid actual operation time")

    def test_return_sample_false_durability_rejected(self):
        self.reject(lambda r: r["samples"][0].update(caller_durable_total_ns=1), "unmeasured persistence")

    def test_durable_elapsed_arithmetic_rejected(self):
        self.reject(lambda r: r["samples"][60].update(caller_durable_extra_ns=1), "durable elapsed arithmetic")

    def test_incomplete_persistence_scope_rejected(self):
        self.reject(lambda r: r["samples"][60]["caller_durable_scope"]["ancestor_directories_synced"].pop(), "persistence scope differs")

    def test_unsynced_new_data_rejected(self):
        self.reject(lambda r: r["samples"][60]["caller_durable_scope"].update(data_files_synced=[]), "persistence scope differs")

    def test_unaccounted_metadata_rejected(self):
        self.reject(lambda r: r["samples"][0].update(generated_layout_has_no_unaccounted_metadata_files=False), "cleanup or layout failed")

    def test_trial_cleanup_failure_rejected(self):
        self.reject(lambda r: r["samples"][0].update(cleanup_completed=False), "cleanup or layout failed")

    def test_final_cleanup_failure_rejected(self):
        self.reject(lambda r: r.update(cleanup_all_prepared_data_completed=False), "final cleanup did not pass")

    def test_wrong_destination_digest_rejected(self):
        self.reject(lambda r: r["samples"][0]["at_return_validation"][0].update(sha256="0" * 64), "whole-file digest differs")

    def test_wrong_source_isolation_digest_rejected(self):
        self.reject(lambda r: r["samples"][0]["destination_write_isolation"][1].update(sha256="0" * 64), "whole-file digest differs")

    def test_inode_alias_rejected(self):
        def mutation(report):
            sample = report["samples"][0]
            target = sample["at_return_validation"][0]["path"]
            inode = sample["before"][0]["inode"]
            for value in sample.values():
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict) and item.get("path") == target:
                            item["inode"] = inode
        self.reject(mutation, "alias inodes")

    def test_escaped_path_rejected(self):
        self.reject(lambda r: r["samples"][0]["at_return_validation"][0].update(path="/tmp/fpb-xfs/latency/../escape"), "noncanonical")

    def test_missing_witness_rejected(self):
        self.reject(lambda r: r["samples"][0]["destination_write_isolation"].pop(), "witness count differs")

    def test_false_resource_gate_rejected(self):
        self.reject(lambda r: r["samples"][0]["filesystem_space_before"].update(used_bytes=512 * verifier.MIB), "invalid filesystem usage")

    def test_namespace_counter_drift_rejected(self):
        self.reject(lambda r: r["samples"][0]["filesystem_space_after_cleanup"].update(snapshot_count=3), "counters did not return")

    def test_invented_statistics_rejected(self):
        self.reject(lambda r: r["groups"][0]["statistics"]["original"]["operation_return"].update(mean_ns=1.0), "distribution differs")

    def test_changed_executable_declaration_rejected(self):
        self.reject(lambda r: r.update(fixture_executable_sha256="0" * 64), "executable/source cohort differs")


class ParserAndStatisticsControls(unittest.TestCase):
    def test_duplicate_json_key_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate JSON"):
            json.loads('{"x":1,"x":2}', object_pairs_hook=verifier.no_duplicates)

    def test_short_sample_p95_is_maximum(self):
        self.assertEqual(verifier.distribution([8, 3, 1, 7, 2, 5, 4, 6])["p95_ns"], 8)


if __name__ == "__main__":
    unittest.main()
