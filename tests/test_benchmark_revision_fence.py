"""Offline contract checks for the stable-revision Docker A/B harness."""

import copy
import unittest

from examples.realworld_boltons26.benchmark_revision_fence import (
    _assert_parity, _schedule,
)


class RevisionFenceBenchmarkTests(unittest.TestCase):
    @staticmethod
    def _rows():
        template = {"status": ["graded", "graded"], "reward": [1.0, 0.0],
                    "evidence_sha256": ["repair-proof", "baseline-proof"],
                    "action_sha256s": [["read", "edit", "check", "submit"], ["submit"]],
                    "policy_revisions": [["rev"] * 4, ["rev"]],
                    "freshness": ["current_revision", "current_revision"],
                    "stale_action_rejections": [0, 0]}
        return [dict(copy.deepcopy(template), pair=pair, arm=arm)
                for pair, arm in _schedule(3)]

    def test_alternating_balanced_schedule_and_exact_parity(self):
        rows = self._rows()
        self.assertEqual([(r["pair"], r["arm"]) for r in rows],
                         [(1, "report_only"), (1, "fence"),
                          (2, "fence"), (2, "report_only"),
                          (3, "report_only"), (3, "fence")])
        _assert_parity(rows, 3)

    def test_parity_fails_on_host_evidence_or_applied_action_change(self):
        for key, value in (("evidence_sha256", ["wrong", "baseline-proof"]),
                           ("action_sha256s", [["wrong"], ["submit"]]),
                           ("reward", [0.0, 0.0])):
            rows = self._rows()
            rows[1][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(
                    RuntimeError, "stable_revision_parity_failed"):
                _assert_parity(rows, 3)

    def test_missing_or_reordered_arm_fails_closed(self):
        rows = self._rows()
        for invalid in (rows[:-1], rows[1:] + rows[:1]):
            with self.assertRaisesRegex(RuntimeError, "incomplete_ab_schedule"):
                _assert_parity(invalid, 3)
        with self.assertRaisesRegex(ValueError, "pairs"):
            list(_schedule(1))


if __name__ == "__main__":
    unittest.main()
