"""Branch-local RL preparation over actual RealWorldEnv sibling exports."""

import copy
import unittest

from future_prediction_bench.branch_advantages import prepare_sibling_advantages
from future_prediction_bench.realworld import RealWorldEnv
from test_realworld_branching import BranchAdapter, Clock, task


class BranchAdvantagesTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.parent = RealWorldEnv(task(), BranchAdapter(), clock=self.clock.now,
                                   monotonic_clock=self.clock.monotonic)
        self.parent.reset("revision-001")
        self.parent.step({"action": "set", "value": "shared-prefix"})
        checkpoint = self.parent.create_branch_checkpoint()
        self.records = []
        for name, value in (("good", "good"), ("bad", "bad")):
            child = self.parent.fork_from_checkpoint(
                checkpoint, BranchAdapter(), branch_id=name)
            child.step({"action": "set", "value": value})
            child.step({"action": "submit"})
            child.verify()
            self.records.append(child.export_trajectory())

    def test_real_sibling_exports_prepare_only_suffix_advantages(self):
        result = prepare_sibling_advantages(
            list(reversed(self.records)), expected_siblings=2,
            current_policy_id="revision-001", as_of=self.clock.now())
        members = {item["branch_id"]: item for item in result["members"]}
        self.assertEqual((members["good"]["advantage"], members["bad"]["advantage"]),
                         (1.0, -1.0))
        self.assertEqual(result["loss_scope"], "post_checkpoint_policy_actions_only")
        self.assertFalse(result["trainer_ready"])
        self.assertEqual(len(result["shared_prefix_visible_event_sha256s"]), 3)
        self.assertEqual(len(members["good"]["suffix_action_event_sha256s"]), 2)
        self.assertNotIn(self.records[0]["events"][1]["sha256"],
                         members["good"]["suffix_action_event_sha256s"])
        self.assertEqual(result["checkpoint_id"],
                         self.records[0]["branch_lineage"]["checkpoint_id"])

    def test_incomplete_stale_mixed_or_tampered_cohorts_fail_closed(self):
        for changed in (
            lambda rows: rows.pop(),
            lambda rows: rows[1].update(policy_id="revision-002"),
            lambda rows: rows[1]["branch_lineage"].update(checkpoint_id="other"),
            lambda rows: rows[1]["branch_lineage"].update(branch_id="good"),
            lambda rows: rows[1]["events"][0]["payload"].update(state="changed"),
            lambda rows: rows[1]["branch_lineage"].update(
                prefix_visible_event_count=2),
        ):
            with self.subTest(change=changed):
                rows = copy.deepcopy(self.records)
                changed(rows)
                with self.assertRaises(ValueError):
                    prepare_sibling_advantages(
                        rows, expected_siblings=2,
                        current_policy_id="revision-001", as_of=self.clock.now())

    def test_equal_rewards_have_zero_local_signal(self):
        rows = copy.deepcopy(self.records)
        rows[0]["reward"] = 0.0
        result = prepare_sibling_advantages(
            rows, expected_siblings=2, current_policy_id="revision-001",
            as_of=self.clock.now())
        self.assertEqual([item["advantage"] for item in result["members"]], [0.0, 0.0])

    def test_future_reward_cannot_enter_current_batch(self):
        from datetime import timedelta
        with self.assertRaisesRegex(ValueError, "preparation time"):
            prepare_sibling_advantages(
                self.records, expected_siblings=2,
                current_policy_id="revision-001",
                as_of=self.clock.now() - timedelta(seconds=1))


if __name__ == "__main__":
    unittest.main()
