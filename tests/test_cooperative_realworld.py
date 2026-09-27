"""Offline gates for the cooperative/full-VM semantic comparison."""

from __future__ import annotations

import copy
import base64
import json
import unittest
from pathlib import Path
import statistics

from examples.cooperative_realworld.benchmark import _equivalence, _public_report
from examples.cooperative_realworld.host_runtime import CooperativeTransport
from wire import frame


def _row(condition, *, mode="repair"):
    repair = mode == "repair"
    cases = [{"return_code": 0, "stdout_sha256": "a" * 64,
              "passed": index < (14 if repair else 7)} for index in range(14)]
    opening = {
        "task_id": "boltons-26-singularize-ss-v2",
        "workspace_root": "/workspace",
        "visible_check": ["python3", "-B", "-c", "import boltons.strutils"],
        "runtime_kind": condition,
        "tools": (["read_file", "replace_text", "submit"]
                  if condition == "cooperative"
                  else ["list_files", "read_file", "write_file", "replace_text",
                        "run_visible_checks", "submit"]),
    }
    actions = [{"path": "boltons/strutils.py", "sha256": "1" * 64,
                "text": "public source preview", "truncated": True}]
    if repair:
        actions.append({"path": "boltons/strutils.py", "sha256": "2" * 64})
    actions.append({"status": "submitted",
                    "snapshot_kind": (
                        "quiescent_process_fork_frozen_overlay_v1"
                        if condition == "cooperative" else "full_vm_state_qcow2_v1")})
    return {"condition": condition, "mode": mode,
            "opening_observation": opening, "action_observations": actions,
            "task_sha256": "c" * 64 if condition == "cooperative" else "f" * 64,
            "case_results": cases, "reward": 1.0 if repair else 0.0,
            "passed_cases": 14 if repair else 7}


class CooperativeRealWorldParityTests(unittest.TestCase):
    def test_repair_and_baseline_common_action_observations_match(self):
        for mode in ("repair", "baseline"):
            with self.subTest(mode=mode):
                result = _equivalence(_row("cooperative", mode=mode),
                                      _row("prepared_full_vm", mode=mode),
                                      mode=mode, repaired_sha="2" * 64,
                                      original_sha="1" * 64)
                self.assertTrue(result["equal_per_case_results"])
                self.assertTrue(result["equal_read_and_edit_observations"])

    def test_different_read_text_is_a_real_parity_failure(self):
        cooperative = _row("cooperative")
        full = _row("prepared_full_vm")
        cooperative["action_observations"][0]["text"] = "shortened preview"
        with self.assertRaisesRegex(RuntimeError, "policy_action_observation_mismatch"):
            _equivalence(cooperative, full, mode="repair",
                         repaired_sha="2" * 64, original_sha="1" * 64)

    def test_hidden_case_difference_cannot_be_hidden_by_equal_reward(self):
        cooperative = _row("cooperative")
        full = _row("prepared_full_vm")
        full["case_results"][5]["stdout_sha256"] = "b" * 64
        with self.assertRaisesRegex(RuntimeError, "host_private_case_result"):
            _equivalence(cooperative, full, mode="repair",
                         repaired_sha="2" * 64, original_sha="1" * 64)

    def test_submission_kind_is_declared_not_normalized_away(self):
        cooperative = _row("cooperative")
        full = _row("prepared_full_vm")
        tampered = copy.deepcopy(cooperative)
        tampered["action_observations"][-1]["snapshot_kind"] = "full_vm_state_qcow2_v1"
        with self.assertRaisesRegex(RuntimeError, "submit_status_or_declared"):
            _equivalence(tampered, full, mode="repair",
                         repaired_sha="2" * 64, original_sha="1" * 64)

    def test_transport_carries_full_sixteen_kib_preview(self):
        answer = {"v": 1, "seq": 0, "ok": True, "error": None,
                  "value": {"text": "line\n" * 3200}}

        class Runtime:
            def run_shell(self, command, *, timeout):
                self.command = command
                return {"return_code": 0, "stdout": "FPB_COOP_V0=" +
                        base64.b64encode(frame(answer, limit=32768)).decode("ascii")}

        runtime = Runtime()
        result = CooperativeTransport(runtime).exchange(
            {"v": 1, "seq": 0, "op": "hello", "args": {}})
        self.assertEqual(result, answer)
        self.assertIn("/fpb_coop_guest.py --call", runtime.command)

    def test_public_report_discards_case_rows_and_policy_text(self):
        cooperative = _row("cooperative")
        full = _row("prepared_full_vm")
        for row in (cooperative, full):
            row.update({"wall_seconds": 1.0, "source_after_submit_sha256": "2" * 64,
                        "guest_reset_ns": 1000, "guest_cleanup_ns": 2000,
                        "child_provision_seconds": 0.1,
                        "child_teardown_seconds": 0.1})
        report = {
            "kind": "synthetic", "status": "passed", "fixture": "public_fixture",
            "repetitions": 1, "cooperative_one_time_setup_seconds": 3.0,
            "prepared_full_vm_one_time_setup_seconds": 4.0,
            "cooperative_checkpoint_ns": 1000,
            "cooperative_service_sha256": "a" * 64,
            "cooperative_case_runner_sha256": "b" * 64,
            "outer_workspace_sha256": "c" * 64,
            "cooperative_graded_episode_stats": {"n": 1},
            "prepared_full_vm_graded_episode_stats": {"n": 1},
            "steady_state_median_ratio_full_over_cooperative": 1.0,
            "amortized_total_seconds": {"cooperative": 4.0,
                                        "prepared_full_vm": 5.0},
            "task_time_window_identical": True,
            "pairs": [{"repetition": 0, "mode": "repair",
                       "condition_order": ["cooperative", "prepared_full_vm"],
                       "parity": {"equal_per_case_results": True},
                       "episodes": [cooperative, full]}],
        }
        public = json.dumps(_public_report(report), sort_keys=True)
        self.assertNotIn("case_results", public)
        self.assertNotIn("stdout_sha256", public)
        self.assertNotIn("public source preview", public)
        self.assertNotIn("whole_result_parity_sha256", public)

    def test_published_fivepair_report_preserves_parity_and_timing_scope(self):
        path = (Path(__file__).resolve().parents[1] / "docs" / "measurements"
                / "cooperative_realworld_ab_fivepair_2026-09-25.json")
        raw = path.read_text(encoding="utf-8")
        for private_field in ("case_results", "stdout_sha256", "expected_stdout",
                              "action_observations", "whole_result_parity_sha256",
                              "/Users/", "/private/tmp/"):
            self.assertNotIn(private_field, raw)
        report = json.loads(raw)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["repetitions"], 5)
        self.assertEqual(len(report["pairs"]), 10)
        self.assertTrue(report["task_time_window_identical"])
        per_arm = {"cooperative": [], "prepared_full_vm": []}
        for pair in report["pairs"]:
            self.assertEqual({row["condition"] for row in pair["episodes"]},
                             set(per_arm))
            self.assertTrue(pair["parity"]["equal_all_hidden_case_outcomes"])
            self.assertTrue(pair["parity"]["equal_read_and_edit_observations"])
            self.assertTrue(pair["parity"]["equal_reward"])
            expected = (14, 1.0) if pair["mode"] == "repair" else (7, 0.0)
            for row in pair["episodes"]:
                self.assertEqual((row["passed_cases"], row["reward"]), expected)
                per_arm[row["condition"]].append(row["wall_seconds"])
        self.assertEqual([len(values) for values in per_arm.values()], [10, 10])
        ratio = (statistics.median(per_arm["prepared_full_vm"])
                 / statistics.median(per_arm["cooperative"]))
        self.assertAlmostEqual(ratio,
                               report["steady_state_median_ratio_full_over_cooperative"],
                               places=4)
        self.assertTrue(report["paired_ratio_stats"]["all_pairs_cooperative_faster"])


if __name__ == "__main__":
    unittest.main()
