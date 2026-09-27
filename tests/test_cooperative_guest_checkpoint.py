"""Offline guards for the public cooperative-checkpoint example."""

from __future__ import annotations

import base64
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
EXAMPLE = PROJECT / "examples/cooperative_guest_checkpoint"
sys.path.insert(0, str(EXAMPLE))
sys.path.insert(0, str(PROJECT))

import guest_coupled
import benchmark as run_qemu


class CoupledOfflineTests(unittest.TestCase):
    def test_pinned_task_and_asset_binding(self):
        task = PROJECT / "runs/boltons-v2-task-pinned2-20260925"
        assets = PROJECT / "runs/microvm-assets-v2task-pinned2-20260925"
        if not task.exists() or not assets.exists():
            self.skipTest("optional generated Boltons task and VM assets are absent")
        manifest, cases, codes = run_qemu._check_fixture(task, assets)
        self.assertEqual(manifest["task_id"], run_qemu.PINNED_TASK)
        self.assertEqual(len(cases), len(codes))
        self.assertEqual(len(codes), 14)

    def test_host_only_grading_removes_candidate_outputs(self):
        cases = [{"expected_returncode": 0, "expected_stdout": str(i)}
                 for i in range(14)]
        clean = [{"return_code": 0,
                  "stdout_b64": base64.b64encode(str(i).encode()).decode()}
                 for i in range(14)]
        baseline = copy.deepcopy(clean)
        for i in range(7, 14):
            baseline[i]["stdout_b64"] = base64.b64encode(b"wrong").decode()
        report = {"cycles_data": [{"branches": [
            {"mode": "repair", "case_results": clean},
            {"mode": "baseline", "case_results": baseline}]}]}
        graded = run_qemu._grade_and_redact(report, cases)
        self.assertEqual([(x["passed_cases"], x["reward"]) for x in graded],
                         [(14, 1.0), (7, 0.0)])
        self.assertNotIn("case_results", report["cycles_data"][0]["branches"][0])
        self.assertNotIn("stdout_b64", json.dumps(report))

    def test_corrupted_restored_process_state_is_rejected(self):
        branches = [{"mode": mode, "inherited_turn": 17,
                     "local_turn": 111 if mode == "repair" else 222,
                     "prefix_seen": True, "sibling_markers_absent": True,
                     "mount_namespace": "mnt:[43210]"}
                    for mode in ("repair", "baseline")]
        report = {"kind": "cooperative_coupled_fork_overlay_checkpoint_v1",
                  "cycles": 2, "restores_per_cycle": 2,
                  "seed_source_sha256": run_qemu.PINNED_SOURCE_SHA,
                  "checkpoint_ns": [1, 2], "restore_ns": [1, 2, 3, 4],
                  "no_case_branch_cycle_ns": [1, 2],
                  "cycles_data": [
                      {"parent_turn_after_checkpoint": 999,
                       "template_turn_after_restores": 17,
                       "frozen_prefix_unchanged": True,
                       "frozen_source_unchanged": True,
                       "template_mount_namespace": "mnt:[12345]",
                       "branch_cycle_ns": [1, 2],
                       "branches": copy.deepcopy(branches)} for _ in range(2)],
                  "outer_ext4_source_unchanged": True}
        report["outer_ext4_tree_unchanged"] = True
        report["outer_ext4_tree_sha256"] = "a" * 64
        report["cycles_data"][1]["branches"][0]["inherited_turn"] = 999
        encoded = base64.b64encode(json.dumps(report).encode()).decode()
        with self.assertRaisesRegex(RuntimeError, "sibling branch isolation"):
            run_qemu._decode_guest("FPB_COUPLED_RESULT:" + encoded, 2, 2)

    def test_timing_quantiles_use_raw_samples(self):
        values = [1_000_000 * value for value in range(1, 21)]
        stats = guest_coupled.ns_stats(values)
        self.assertEqual(stats["n"], 20)
        self.assertEqual(stats["p50_ms"], 10.5)
        self.assertEqual(stats["p95_ms"], 19.0)

    def test_outer_tree_digest_detects_non_source_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "a").mkdir()
            file = root / "a" / "other.py"
            file.write_text("before", encoding="ascii")
            before = guest_coupled._tree_digest(root)
            file.write_text("after", encoding="ascii")
            self.assertNotEqual(guest_coupled._tree_digest(root), before)
            file.unlink()
            file.symlink_to(root / "missing")
            with self.assertRaisesRegex(RuntimeError, "symlink"):
                guest_coupled._tree_digest(root)

    def test_published_report_has_parity_and_no_case_outputs(self):
        path = PROJECT / "docs/measurements/cooperative_guest_checkpoint_v0.9.0.json"
        report_text = path.read_text(encoding="utf-8")
        for private_key in ("stdout_b64", "case_results", "expected_stdout",
                            "/Users/", "/private/tmp/"):
            self.assertNotIn(private_key, report_text)
        report = json.loads(report_text)
        self.assertEqual([row["reward"] for row in report["host_private_grading"]],
                         [1.0, 0.0])
        self.assertEqual(report["guest"]["checkpoint_stats"]["n"], 10)
        self.assertEqual(report["guest"]["restore_stats"]["n"], 100)
        self.assertEqual(report["guest"]["no_case_branch_cycle_stats"]["n"], 98)
        self.assertTrue(report["guest"]["outer_ext4_tree_unchanged"])
        self.assertEqual(report["guest"]["checkpoint_stats"],
                         guest_coupled.ns_stats(report["guest"]["checkpoint_ns"]))
        self.assertEqual(report["guest"]["restore_stats"],
                         guest_coupled.ns_stats(report["guest"]["restore_ns"]))
        self.assertEqual(report["guest"]["no_case_branch_cycle_stats"],
                         guest_coupled.ns_stats(
                             report["guest"]["no_case_branch_cycle_ns"]))
        self.assertEqual(
            [value for cycle, row in enumerate(report["guest"]["cycles_data"])
             for index, value in enumerate(row["branch_cycle_ns"])
             if cycle != 0 or index >= 2],
            report["guest"]["no_case_branch_cycle_ns"])
        for cycle in report["guest"]["cycles_data"]:
            for branch in cycle["branches"]:
                self.assertNotEqual(branch["mount_namespace"],
                                    cycle["template_mount_namespace"])
                if branch["mode"] == "baseline":
                    self.assertEqual(branch["source_sha256"],
                                     run_qemu.PINNED_SOURCE_SHA)


if __name__ == "__main__":
    unittest.main()
