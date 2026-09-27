import copy
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from future_prediction_bench.demo import FixtureClock, fixture_questions
from future_prediction_bench.store import Store
from future_prediction_bench.training import (assistant_segments, calibration_probe, collect_rollout_group,
                                               group_advantages, prepare_training_groups)
from future_prediction_bench.training_demo import SmokeAnalystModel, SmokeResearchProvider, run_rl_smoke


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.clock = FixtureClock()
        self.store = Store(":memory:", mode="fixture", clock=self.clock)
        self.addCleanup(self.store.close)
        self.question = fixture_questions()[0]
        self.store.add_question(self.question)

    def collect(self):
        return collect_rollout_group(self.store, self.question["question_id"], group_id="group-1", group_size=4,
                                     policy_revision="checkpoint-1", model=SmokeAnalystModel(), provider=SmokeResearchProvider(),
                                     reward_mode="negative_brier")

    def settle(self):
        self.clock.value = datetime(2030, 1, 3, 9, tzinfo=timezone.utc)
        self.store.resolve(self.question["question_id"], outcome="yes", evidence_urls=["https://example.org/fixture/bulletin"],
                           evidence_text="Synthetic result")

    def prepare(self, records, revision="checkpoint-1", **kwargs):
        return prepare_training_groups(records, current_policy_revision=revision, available_at=self.clock().isoformat(),
                                       run_mode="fixture", **kwargs)

    def test_complete_delayed_groups_and_masks(self):
        reports = self.collect()
        self.assertEqual({r["status"] for r in reports}, {"pending_reward"})
        self.assertEqual(self.store.export_training(), [])
        self.settle()
        result = self.prepare(self.store.export_training())
        self.assertEqual(result["summary"]["prepared_samples"], 4)
        self.assertFalse(result["trainer_ready"])
        for sample in result["groups"][0]["samples"]:
            self.assertEqual(sum(segment["loss_mask"] for segment in sample["segments"]), 6)
            for segment in sample["segments"]:
                self.assertEqual(segment["loss_mask"], int(segment["message"]["role"] == "assistant"))
        self.assertAlmostEqual(sum(sample["sequence_advantage"] for sample in result["groups"][0]["samples"]), 0)

    def test_no_retry_or_duplicate_sample(self):
        self.collect()
        reports = self.collect()
        self.assertEqual({r["status"] for r in reports}, {"skipped_existing_assignment"})
        self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM episodes").fetchone()[0], 4)

    def test_incomplete_stale_and_mixed_groups_quarantined(self):
        self.collect()
        self.settle()
        records = self.store.export_training()
        for mutated, revision, reason in [(records[:-1], "checkpoint-1", "incomplete_or_duplicate_group"),
                                         (records, "checkpoint-2", "stale_policy_requires_off_policy_adapter")]:
            result = self.prepare(mutated, revision)
            self.assertEqual(result["groups"], [])
            self.assertEqual(result["quarantined"][0]["reason"], reason)
        for key, value in (("collection_config_hash", "other"), ("reward_mode", "baseline_improvement")):
            changed = copy.deepcopy(records)
            changed[0]["rollout"][key] = value
            self.assertEqual(self.prepare(changed)["groups"], [])

    def test_heldout_and_not_yet_available_outcomes_rejected(self):
        self.collect()
        self.settle()
        records = self.store.export_training()
        changed = copy.deepcopy(records)
        changed[0]["question"]["split"] = "test"
        self.assertEqual(self.prepare(changed)["groups"], [])
        changed = copy.deepcopy(records)
        changed[0]["resolved_at"] = "2031-01-01T00:00:00Z"
        self.assertEqual(self.prepare(changed)["quarantined"][0]["reason"], "outcome_unavailable_at_cutoff")

    def test_host_submit_is_never_policy_supervision(self):
        episode = self.store.create_episode(self.question["question_id"], "policy", track="rl", research_mode="no_search")
        self.store.submit(episode, None, agent_generated=False)
        event = self.store.events(episode)[-1]
        self.assertEqual(event["loss_mask"], 0)
        self.assertEqual(event["origin"], "host")

    def test_registration_precedes_actions_and_freezes_group(self):
        def create():
            return self.store.create_episode(self.question["question_id"], "policy", track="rl", research_mode="no_search")
        context = {"group_id": "g", "group_size": 2, "sample_index": 0, "policy_revision": "p"}
        episode = create()
        self.store.register_rollout(episode, context)
        with self.assertRaises(ValueError):
            self.store.register_rollout(episode, context)
        with self.assertRaises(ValueError):
            self.store.register_rollout(create(), {**context, "sample_index": 1, "policy_revision": "changed"})
        late = create()
        self.store.submit(late, {"yes": .5, "no": .5})
        with self.assertRaises(ValueError):
            self.store.register_rollout(late, {**context, "sample_index": 1})

    def test_mask_adapter_rejects_unattributed_and_rewritten_context(self):
        turns = [{"request": {"messages": [{"role": "user", "content": "Question"}]},
                  "response": {"role": "assistant", "content": "Answer"}}]
        self.assertEqual([s["loss_mask"] for s in assistant_segments(turns)], [0, 1])
        with self.assertRaises(ValueError):
            assistant_segments(turns + [{"request": {"messages": [{"role": "user", "content": "Changed"}]},
                                         "response": {"role": "assistant", "content": "Answer"}}])
        with self.assertRaises(ValueError):
            assistant_segments([])

    def test_baseline_shift_cancels_and_rloo_is_not_std_scaled(self):
        for method in ("rloo", "centered", "standard_grpo"):
            original = group_advantages([-.1, -.2, -.4], method)
            shifted = group_advantages([.2, .1, -.1], method)
            for a, b in zip(original, shifted):
                self.assertAlmostEqual(a, b)
        self.assertEqual(group_advantages([.5, .5]), [0, 0])
        self.assertAlmostEqual(group_advantages([-.1, -.2])[0], .1)
        for rewards in ([1], [True, .3], [float("nan"), .2]):
            with self.assertRaises(ValueError):
                group_advantages(rewards)

    def test_exact_calibration_probe_finds_standardization_bias(self):
        probe = calibration_probe()
        at_truth = next(row for row in probe["rows"] if row["center_probability"] == .7)["expected_upper_advantage"]
        self.assertAlmostEqual(at_truth["rloo"], 0)
        self.assertAlmostEqual(at_truth["centered"], 0)
        self.assertGreater(at_truth["standard_grpo"], .399)
        beyond = next(row for row in probe["rows"] if row["center_probability"] == .9)["expected_upper_advantage"]
        self.assertLess(beyond["rloo"], 0)
        self.assertGreater(beyond["standard_grpo"], 0)

    def test_reproducible_smoke_two_option_counts(self):
        with tempfile.TemporaryDirectory() as root:
            report = run_rl_smoke(Path(root) / "smoke")
            self.assertEqual(report["pending_before_resolution"], 8)
            self.assertEqual(report["prepared"]["prepared_groups"], 2)
            self.assertEqual(report["stale_policy_check"]["prepared_groups"], 0)
            self.assertEqual(report["tool_calls"], 40)
            exported = json.loads((Path(root) / "smoke" / "prepared_groups.json").read_text())
            self.assertNotIn("synthetic-fixture-baseline-only", json.dumps(exported))


if __name__ == "__main__":
    unittest.main()
