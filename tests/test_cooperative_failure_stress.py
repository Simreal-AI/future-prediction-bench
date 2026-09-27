"""Offline gates for the real-QEMU cooperative failure stress probe."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from examples.cooperative_realworld.stress_qemu import (
    FAULT_CODES, _census, _fault_branch, _validate_fault_codes,
    public_report, stress,
)


class _Runtime:
    def __init__(self, *, escaped=False, descendant=False, debris=False):
        self.escaped = escaped
        self.descendant = descendant
        self.debris = debris
        self.commands = []

    def run_shell(self, command, *, timeout):
        self.commands.append(command)
        if command.startswith("sha256sum"):
            return {"return_code": 0,
                    "stdout": "a" * 64 + "  /mnt/root/fpb_coop_guest.py\n"}
        if "grep -l" in command:
            return {"return_code": 0,
                    "stdout": "/proc/99/comm\n" if self.descendant else ""}
        if "ls -d" in command:
            return {"return_code": 0,
                    "stdout": "/mnt/root/.fpb-resident-leak\n" if self.debris else ""}
        if command.startswith("test ! -e") and self.escaped:
            return {"return_code": 1, "stdout": ""}
        return {"return_code": 0, "stdout": ""}


class _Client:
    def __init__(self):
        self.state = "idle"
        self.codes = None

    def reset(self, episode_id, mode):
        assert mode == "baseline"
        self.state = "active"
        return {"namespaces": {"process_nonce": "1" * 32}}

    def action(self, name, args):
        assert name == "read_file" and args["path"] == "src/humanize/filesize.py"
        return {"accepted": True, "sha256": "b" * 64}

    def submit(self):
        self.state = "pending"

    def run_case_batch(self, codes, *, expected_branch_sha256):
        self.codes = codes
        assert expected_branch_sha256 == "b" * 64
        expected = [(0, b"spawned\n"), (0, b"denied\n"), (0, b"False\n"),
                    (124, b"")] + [(0, b"ok\n")] * 10
        return [{"return_code": rc, "stdout_bytes": text,
                 "truncated": False, "output_over_batch_cap": False}
                for rc, text in expected]

    def close(self, *, completed):
        assert completed is True
        self.state = "idle"


class CooperativeFailureStressTests(unittest.TestCase):
    def test_fault_codes_fit_one_exact_bounded_fourteen_case_batch(self):
        self.assertEqual(len(FAULT_CODES), 14)
        self.assertLessEqual(_validate_fault_codes(), 2048)
        self.assertIn("os.fork()", FAULT_CODES[0])
        self.assertIn("os.setsid()", FAULT_CODES[0])
        self.assertIn("os.close(1)", FAULT_CODES[3])

    def test_fault_branch_requires_all_results_and_empty_cleanup_census(self):
        private = SimpleNamespace(source_path="src/humanize/filesize.py",
                                  seed_file_sha256="b" * 64,
                                  guest_service_sha256="a" * 64)
        client = _Client()
        result = _fault_branch(client, private, _Runtime(), label="humanize")
        self.assertEqual(len(client.codes), 14)
        self.assertEqual(result["stdout_close_timeout_return_code"], 124)
        self.assertEqual(result["cleanup_after_faults"]["named_candidate_descendants"], 0)
        with self.assertRaisesRegex(RuntimeError, "candidate_descendant_or_branch_pool_survived"):
            _census(_Runtime(descendant=True))
        with self.assertRaisesRegex(RuntimeError, "candidate_descendant_or_branch_pool_survived"):
            _census(_Runtime(debris=True))
        with self.assertRaisesRegex(RuntimeError, "guest_stress_attestation_failed"):
            _fault_branch(_Client(), private, _Runtime(escaped=True), label="humanize")

    def test_public_projection_has_no_candidate_output_or_branch_nonce(self):
        item = {"repository": "humanize", "task_id": "humanize-id",
                "source_sdist_sha256": "a" * 64, "verifier_sha256": "b" * 64,
                "rootfs_qcow2_sha256": "c" * 64,
                "guest_service_sha256": "d" * 64,
                "guest_case_runner_sha256": "e" * 64,
                "outer_workspace_sha256": "f" * 64,
                "fault_branch": {"fault_case_count": 14,
                                 "detached_grandchild_handshake": True,
                                 "symlink_outside_write_denied": True,
                                 "next_case_workspace_copy_clean": True,
                                 "stdout_close_timeout_return_code": 124,
                                 "cleanup_after_faults": {"named_candidate_descendants": 0,
                                                          "root_side_branch_pool_debris": 0},
                                 "case_result_digest": "1" * 64,
                                 "branch_process_nonce": "secret-nonce"},
                "graded_branches": [
                    {"mode": "repair", "reward": 1.0, "passed_cases": 14,
                     "case_result_digest": "2" * 64,
                     "census": {"named_candidate_descendants": 0,
                                "root_side_branch_pool_debris": 0},
                     "secret_case_stdout": "hidden"},
                    {"mode": "baseline", "reward": 0.0, "passed_cases": 5,
                     "case_result_digest": "3" * 64,
                     "census": {"named_candidate_descendants": 0,
                                "root_side_branch_pool_debris": 0}}],
                "unique_branch_nonces": 3}
        boltons = copy.deepcopy(item)
        boltons["repository"] = "boltons"
        boltons["task_id"] = "boltons-id"
        boltons["graded_branches"][1]["passed_cases"] = 7
        report = {"kind": "synthetic", "status": "passed",
                  "branches_per_repository": 2, "fault_case_frame_bytes": 782,
                  "source_binding": {}, "repositories": [item, boltons]}
        public = json.dumps(public_report(report))
        self.assertNotIn("secret-nonce", public)
        self.assertNotIn("hidden", public)
        self.assertNotIn("secret_case_stdout", public)

    def test_output_must_be_new_and_disjoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("bt", "ba", "ht", "ha"):
                (root / name).mkdir()
            (root / "occupied").mkdir()
            (root / "occupied/file").write_text("x")
            inputs = [root / name for name in ("bt", "ba", "ht", "ha")]
            with self.assertRaisesRegex(ValueError, "stress_inputs_or_output_invalid"):
                stress(*inputs, root / "occupied", branches=2)
            with self.assertRaisesRegex(ValueError, "stress_inputs_or_output_invalid"):
                stress(*inputs, root / "new", branches=3)

    def test_published_real_qemu_result_is_sanitized_and_complete(self):
        path = (Path(__file__).resolve().parents[1] / "docs/measurements"
                / "cooperative_failure_stress_2026-09-25.json")
        raw = path.read_text(encoding="utf-8")
        for marker in ("/Users/", "/private/tmp/", "stdout_b64",
                       "expected_stdout", "branch_process_nonce", "graded_branches"):
            self.assertNotIn(marker, raw)
        report = json.loads(raw)
        self.assertEqual((report["status"], report["branches_per_repository"]),
                         ("passed", 20))
        self.assertEqual({row["repository"] for row in report["repositories"]},
                         {"boltons", "humanize"})
        for repo in report["repositories"]:
            self.assertEqual(repo["graded_branch_count"], 20)
            self.assertEqual(repo["unique_branch_nonces"], 21)
            self.assertEqual(repo["fault_branch"]["stdout_close_timeout_return_code"], 124)
            self.assertEqual(repo["fault_branch"]["cleanup_after_faults"], {
                "named_candidate_descendants": 0,
                "root_side_branch_pool_debris": 0})
            self.assertTrue(repo["all_case_results_stable_by_mode"])


if __name__ == "__main__":
    unittest.main()
