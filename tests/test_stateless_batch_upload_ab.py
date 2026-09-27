"""Offline fault and parity tests for the bounded one-command case upload."""

from __future__ import annotations

import base64
import hashlib
import re
import unittest
from unittest.mock import patch

from examples.realworld_boltons26 import benchmark_stateless_batch_upload_ab as ab
from future_prediction_bench.stateless_verifier import _batch_payload


class FakeVerifier:
    def __init__(self, *, wrong_digest=False):
        self.cases = [{"argv": ["python3", "-B", "-c", "print(1)"]}]
        self.installed = True
        self.batch_code_sha256 = None
        self.calls = []
        self.contract_checks = 0
        self.wrong_digest = wrong_digest

    def _check_host_contract(self):
        self.contract_checks += 1

    def _required(self, command):
        self.calls.append(command)
        match = re.search(r"printf '%s' '([A-Za-z0-9+/=]+)'", command)
        assert match is not None
        payload = base64.b64decode(match.group(1), validate=True)
        digest = hashlib.sha256(payload).hexdigest()
        if self.wrong_digest:
            digest = "0" * 64
        return digest + "  /mnt/root/fpb_stateless_cases.json\n"


def fake_rows():
    rows = []
    for pair, branch, arm in ab._schedule():
        passes = 14 if branch == "repair" else 7
        rows.append({
            "pair": pair, "branch": branch, "arm": arm,
            "reward": 1.0 if branch == "repair" else 0.0,
            "passed_cases": passes,
            "case_results": [{"returncode": 0, "stdout_sha256": f"{i:064x}",
                              "passed": i < passes} for i in range(14)],
            "task_sha256": "a" * 64,
            "final_source_sha256": "c" * 64 if branch == "repair" else "b" * 64,
            "opening_observation": {"source": "b" * 64},
            "action_observation_sha256s": ["1" * 64, "2" * 64] if branch == "repair"
                                          else ["1" * 64],
            "evidence_kind": "host_checked_guest_stateless_namespaced_cases_v1",
            "adapter_metrics": {
                "full_vm_restores": 1, "stateless_batches": 1,
                "stateless_batch_fallbacks": 0,
                "stateless_submit_helper_install_seconds": 0.0,
                "stateless_submit_batch_upload_seconds": 0.025 if arm == "serial" else 0.009,
                "vm_submit_snapshot_seconds": 0.3,
                "stateless_verify_batch_seconds": 0.32,
                "stateless_verify_restore_seconds": 0.13,
                "guest_tool_seconds": 0.4,
            },
            "environment_metrics": {"verifier_seconds": 0.55},
            "wall_seconds": 1.75 if arm == "serial" else 1.73,
        })
    return rows


def fake_asset_manifest():
    return {
        "schema_version": "boltons-microvm-assets-v2",
        "task_id": ab.TASK_ID,
        "rootfs_qcow2_sha256": "4" * 64,
        "source_sdist_sha256": "5" * 64,
        "seed_workspace_sha256": "6" * 64,
        "modloop_disk_sha256": "7" * 64,
        "alpine_sha256": {"vmlinuz-virt": "8" * 64,
                          "initramfs-virt": "9" * 64},
    }


class StatelessBatchUploadABTests(unittest.TestCase):
    def test_one_guest_command_checks_exact_payload_and_digest(self):
        verifier = FakeVerifier()
        fallback = []
        digest = ab._one_call_upload(verifier, ["print(1)"],
                                     lambda codes: fallback.append(codes))
        self.assertEqual(digest, hashlib.sha256(_batch_payload(["print(1)"])).hexdigest())
        self.assertEqual(verifier.batch_code_sha256, digest)
        self.assertEqual(len(verifier.calls), 1)
        self.assertLessEqual(len(verifier.calls[0].encode("ascii")), ab.MAX_SHELL_BYTES)
        self.assertIn("chmod 600", verifier.calls[0])
        self.assertIn("sha256sum", verifier.calls[0])
        self.assertEqual(fallback, [])

    def test_host_contract_wrong_source_and_digest_fail_closed(self):
        verifier = FakeVerifier()
        with self.assertRaisesRegex(ValueError, "case_source_differs"):
            ab._one_call_upload(verifier, ["print(2)"], lambda codes: None)
        self.assertEqual(verifier.calls, [])
        verifier = FakeVerifier(wrong_digest=True)
        with self.assertRaisesRegex(RuntimeError, "digest differs"):
            ab._one_call_upload(verifier, ["print(1)"], lambda codes: None)
        self.assertIsNone(verifier.batch_code_sha256)
        verifier.installed = False
        with self.assertRaisesRegex(RuntimeError, "Install"):
            ab._one_call_upload(verifier, ["print(1)"], lambda codes: None)

    def test_bound_falls_back_to_unchanged_serial_upload(self):
        verifier = FakeVerifier()
        calls = []
        with patch.object(ab, "MAX_SHELL_BYTES", 50):
            result = ab._one_call_upload(verifier, ["print(1)"],
                                         lambda codes: calls.append(codes) or "fallback")
        self.assertEqual(result, "fallback")
        self.assertEqual(calls, [["print(1)"]])
        self.assertEqual(verifier.calls, [])

    def test_three_pairs_and_same_contract_parity(self):
        self.assertEqual(len(ab._schedule()), 12)
        self.assertEqual(ab._schedule()[:4], [
            (0, "repair", "serial"), (0, "repair", "one_call"),
            (0, "baseline", "serial"), (0, "baseline", "one_call")])
        rows = fake_rows()
        report = ab._report(rows, setup_seconds=6.0, frozen_sha="a" * 64,
                            prepared_id="d" * 64, template_sha="e" * 64,
                            verifier_sha="f" * 64, contract_sha="0" * 64,
                            original_sha="b" * 64, repaired_sha="c" * 64,
                            original_task_sha="1" * 64,
                            original_task_file_sha="2" * 64,
                            original_helper_sha="3" * 64,
                            assets_manifest=fake_asset_manifest())
        self.assertEqual(report["episode_count"], 12)
        self.assertEqual(report["setup_amortized_ratio_at_six_episodes"],
                         (6.0 + report["serial_total_seconds"]) /
                         (6.0 + report["one_call_total_seconds"]))
        self.assertEqual(report["serial_total_including_setup_seconds"],
                         6.0 + report["serial_total_seconds"])
        self.assertNotIn("case_results", str(report))
        self.assertEqual(report["original_refreshed_task_sha256"], "1" * 64)
        self.assertEqual(report["effective_replace_text_helper_sha256"],
                         ab.CURRENT_HELPER_BINDING["source_sha256"])
        self.assertEqual(report["asset_binding"]["rootfs_seed_sha256"], "4" * 64)
        rows[1]["case_results"][0]["stdout_sha256"] = "f" * 64
        with self.assertRaisesRegex(RuntimeError, "same_contract_parity_failed"):
            ab._assert_parity(rows, "a" * 64, "b" * 64, "c" * 64)

    def test_asset_binding_requires_complete_valid_v2_provenance(self):
        binding = ab._asset_binding(fake_asset_manifest())
        self.assertEqual(binding["manifest_schema_version"],
                         "boltons-microvm-assets-v2")
        self.assertEqual(binding["kernel_sha256"], "8" * 64)
        malformed = fake_asset_manifest()
        malformed["source_sdist_sha256"] = "short"
        with self.assertRaisesRegex(ValueError, "digest_missing_or_malformed"):
            ab._asset_binding(malformed)
        wrong_task = fake_asset_manifest()
        wrong_task["task_id"] = "other"
        with self.assertRaisesRegex(ValueError, "manifest_required"):
            ab._asset_binding(wrong_task)

    def test_uid_or_batch_boundary_mismatch_fails(self):
        rows = fake_rows()
        rows[0]["adapter_metrics"]["full_vm_restores"] = 0
        with self.assertRaisesRegex(RuntimeError, "boundary_failed"):
            ab._assert_parity(rows, "a" * 64, "b" * 64, "c" * 64)
        rows = fake_rows()
        rows[0]["case_results"][0]["passed"] = False
        with self.assertRaisesRegex(RuntimeError, "boundary_failed"):
            ab._assert_parity(rows, "a" * 64, "b" * 64, "c" * 64)


if __name__ == "__main__":
    unittest.main()
