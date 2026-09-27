"""Offline checks for the published workspace-preparation A/B report contract."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from examples.realworld_boltons26.benchmark_workspace_overlay_ab import _check_guest_report


REPORT = (Path(__file__).resolve().parents[1] / "docs" / "measurements"
          / "workspace_overlay_ab_2026-09-25.json")


class WorkspaceOverlayReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads(REPORT.read_text(encoding="utf-8"))

    def _check(self, guest):
        return _check_guest_report(
            guest, pairs=20,
            seed_digest=self.report["asset_binding"]["seed_workspace_sha256"])

    def test_published_report_has_all_matched_pairs_and_checks(self):
        self.assertEqual(self.report["status"], "passed")
        guest = self._check(self.report["guest"])
        self.assertEqual(len(guest["attempts"]), 42)
        self.assertEqual({row["repaired_source_sha256"] for row in guest["attempts"]},
                         {guest["attempts"][0]["repaired_source_sha256"]})
        self.assertTrue(all(
            next(row["prepare_ns"] for row in guest["attempts"]
                 if row["pair"] == pair and row["arm"] == "overlay")
            < next(row["prepare_ns"] for row in guest["attempts"]
                   if row["pair"] == pair and row["arm"] == "tar_extract")
            for pair in range(20)))

    def test_changed_pair_order_is_rejected(self):
        guest = copy.deepcopy(self.report["guest"])
        guest["attempts"][4], guest["attempts"][5] = (
            guest["attempts"][5], guest["attempts"][4])
        with self.assertRaisesRegex(RuntimeError, "guest_attempt_semantics_invalid"):
            self._check(guest)

    def test_changed_repair_digest_is_rejected(self):
        guest = copy.deepcopy(self.report["guest"])
        guest["attempts"][3]["repaired_source_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "guest_repair_digest_mismatch"):
            self._check(guest)

    def test_fabricated_summary_is_rejected(self):
        guest = copy.deepcopy(self.report["guest"])
        guest["summary"]["overlay"]["median_ms"] = 0.000001
        with self.assertRaisesRegex(RuntimeError, "guest_summary_mismatch"):
            self._check(guest)


if __name__ == "__main__":
    unittest.main()
