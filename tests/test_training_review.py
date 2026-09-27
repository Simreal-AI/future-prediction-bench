"""Regression cases for delayed replay admission and exact request preservation."""

import copy
import unittest
from datetime import datetime, timezone

from future_prediction_bench.demo import FixtureClock, fixture_questions
from future_prediction_bench.runner import run_questions
from future_prediction_bench.store import Store, digest
from future_prediction_bench.training import collect_rollout_group, prepare_training_groups
from future_prediction_bench.training_demo import SmokeAnalystModel, SmokeResearchProvider


class ChangedModel(SmokeAnalystModel):
    def public_config(self):
        return {**super().public_config(), "model_revision": "different"}


class MarketFixture:
    def __init__(self, revision):
        self.revision = revision

    def public_config(self):
        return {"adapter": "offline_market_fixture", "revision": self.revision}


class TrainingReviewTests(unittest.TestCase):
    def setUp(self):
        self.clock = FixtureClock()
        self.store = Store(":memory:", mode="fixture", clock=self.clock)
        self.addCleanup(self.store.close)
        self.question = fixture_questions()[0]
        self.store.add_question(self.question)

    def collect(self, **options):
        arguments = {"group_id": "review-group", "group_size": 2, "policy_revision": "revision-1",
                     "model": SmokeAnalystModel(), "provider": SmokeResearchProvider(),
                     "reward_mode": "negative_brier", **options}
        return collect_rollout_group(self.store, self.question["question_id"], **arguments)

    def records(self, **options):
        self.collect(**options)
        self.clock.value = datetime(2030, 1, 3, 9, tzinfo=timezone.utc)
        self.store.resolve(self.question["question_id"], outcome="yes",
                           evidence_urls=self.question["resolution"]["source_urls"], evidence_text="Fixture result")
        return self.store.export_training()

    def prepare(self, records):
        return prepare_training_groups(records, current_policy_revision="revision-1",
                                       available_at=self.clock().isoformat(), run_mode="fixture")

    def assert_quarantined(self, records, reason=None):
        result = self.prepare(records)
        self.assertEqual(result["groups"], [])
        self.assertEqual(sum(row["count"] for row in result["quarantined"]), len(records))
        if reason is not None:
            self.assertEqual(result["quarantined"][0]["reason"], reason)

    @staticmethod
    def resign(record):
        record["sha256"] = digest({key: value for key, value in record.items() if key != "sha256"})

    def test_prepared_samples_preserve_tool_conditioning_and_audit_records(self):
        records = self.records()
        prepared = self.prepare(records)
        self.assertFalse(prepared["trainer_ready"])
        source_by_id = {row["episode"]["episode_id"]: row for row in records}
        for sample in prepared["groups"][0]["samples"]:
            original = source_by_id[sample["episode_id"]]
            self.assertEqual(sample["model_turns"], original["model_turns"])
            self.assertEqual(sample["events"], original["events"])
            self.assertTrue(sample["events_are_audit_only"])
            self.assertTrue(sample["model_turns"][0]["request"]["tools"])
            self.assertFalse(sample["token_ids_available"])
            self.assertFalse(sample["behavior_logprobs_available"])
            for segment in sample["segments"]:
                self.assertEqual(segment["loss_mask"], int(segment["message"]["role"] == "assistant"))
            sample["model_turns"][0]["request"]["tools"].clear()
            self.assertTrue(original["model_turns"][0]["request"]["tools"])

    def test_host_budget_termination_stays_masked_out(self):
        records = self.records(max_steps=1)
        prepared = self.prepare(records)
        self.assertEqual(prepared["summary"]["prepared_samples"], 2)
        for sample in prepared["groups"][0]["samples"]:
            final = sample["events"][-1]
            self.assertEqual((final["tool"], final["origin"], final["loss_mask"]), ("submit", "host", 0))
            self.assertEqual(sum(segment["loss_mask"] for segment in sample["segments"]), 1)
        altered = copy.deepcopy(records)
        altered[0]["events"][-1]["loss_mask"] = 1
        self.resign(altered[0]["events"][-1])
        self.assert_quarantined(altered, "invalid_event_loss_mask")

    def test_question_metadata_episode_and_run_identity_are_checked(self):
        records = self.records()
        cases = [
            ("question", "prompt", "Changed forecast target", "question_digest_mismatch"),
            ("question", "event_id", "another-event", "mixed_question_identity"),
            ("rollout", "sha256", "bad", "rollout_metadata_digest_mismatch"),
            ("episode", "policy_id", "another-policy", "episode_policy_mismatch"),
            ("episode", "run_mode", "live", "run_mode_mismatch"),
            ("question", "is_fixture", False, "run_mode_mismatch"),
        ]
        for section, key, value, reason in cases:
            with self.subTest(section=section, key=key):
                altered = copy.deepcopy(records)
                altered[0][section][key] = value
                self.assert_quarantined(altered, reason)
        altered = copy.deepcopy(records)
        altered[0]["group_key"][1] = "another-policy"
        self.assert_quarantined(altered, "group_key_mismatch")
        altered = copy.deepcopy(records)
        altered[0]["episode"]["episode_id"] = altered[1]["episode"]["episode_id"]
        self.assert_quarantined(altered, "missing_or_duplicate_episode_id")

    def test_registration_and_model_times_are_bounded_by_submission(self):
        records = self.records()
        for timestamp in ("2029-12-31T12:00:00Z", "2031-01-01T12:00:00Z"):
            with self.subTest(timestamp=timestamp, field="registered_at"):
                altered = copy.deepcopy(records)
                altered[0]["rollout"]["registered_at"] = timestamp
                self.resign(altered[0]["rollout"])
                self.assert_quarantined(altered, "invalid_forecast_timing")
            with self.subTest(timestamp=timestamp, field="model_turn_at"):
                altered = copy.deepcopy(records)
                altered[0]["model_turns"][0]["at"] = timestamp
                self.assert_quarantined(altered, "invalid_model_turn_timing")

    def test_resolution_must_be_available_legal_and_common_to_group(self):
        records = self.records()
        for timestamp, reason in (("2031-01-01T12:00:00Z", "outcome_unavailable_at_cutoff"),
                                  ("2029-12-31T12:00:00Z", "resolution_before_outcome_window"),
                                  ("2030-01-03T08:30:00Z", "mixed_resolution_context")):
            with self.subTest(timestamp=timestamp):
                altered = copy.deepcopy(records)
                altered[0]["resolved_at"] = timestamp
                self.assert_quarantined(altered, reason)
        for outcome, reason in (("unknown", "invalid_resolved_outcome"), ("no", "mixed_resolution_context")):
            altered = copy.deepcopy(records)
            altered[0]["resolution"]["outcome"] = outcome
            self.assert_quarantined(altered, reason)

    def test_malformed_rows_are_quarantined_without_losing_valid_groups(self):
        records = self.records()
        result = self.prepare(records + [[], None, "text", {"rollout": []}])
        self.assertEqual(result["summary"]["prepared_groups"], 1)
        self.assertEqual(sum(row["count"] for row in result["quarantined"]), 4)
        for field, value in (("episode", []), ("question", []), ("resolution", []),
                             ("model_turns", [None]), ("model_turns", {}), ("events", [None])):
            with self.subTest(field=field, value=value):
                altered = copy.deepcopy(records)
                altered[0][field] = value
                self.assert_quarantined(altered)
        altered = copy.deepcopy(records)
        altered[0]["model_turns"][0]["request"] = []
        self.assert_quarantined(altered, "malformed_model_request")

    def test_tool_schemas_cannot_change_within_or_between_samples(self):
        records = self.records()
        altered = copy.deepcopy(records)
        altered[0]["model_turns"][1]["request"]["tools"] = []
        self.assert_quarantined(altered, "unsupported_tool_schema_change")
        altered = copy.deepcopy(records)
        for turn in altered[0]["model_turns"]:
            turn["request"]["tools"] = []
        self.assert_quarantined(altered, "mixed_tool_definitions")

    def test_existing_groups_reject_changed_model_mode_budget_and_evidence(self):
        self.collect()
        for options in ({"model": ChangedModel()}, {"research_mode": "no_search"},
                        {"max_calls": 9}, {"evidence_pack_id": "different-evidence"}):
            with self.subTest(options=options):
                with self.assertRaisesRegex(ValueError, "different collection contract"):
                    self.collect(**options)
                self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM episodes").fetchone()[0], 2)
        self.assertEqual({row["status"] for row in self.collect()}, {"skipped_existing_assignment"})

    def test_existing_partial_group_rejects_config_before_new_assignment(self):
        run_questions(self.store, [self.question["question_id"]], model=SmokeAnalystModel(),
                      provider=SmokeResearchProvider(), track="rl", reward_mode="negative_brier",
                      rollout_context={"group_id": "review-group", "group_size": 2,
                                       "sample_index": 1, "policy_revision": "revision-1"})
        with self.assertRaisesRegex(ValueError, "different collection contract"):
            self.collect(max_calls=9)
        self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM episodes").fetchone()[0], 1)

    def test_market_provider_revision_is_in_retry_contract(self):
        self.collect(market_mode="market_aware", market_provider=MarketFixture("one"))
        with self.assertRaisesRegex(ValueError, "different collection contract"):
            self.collect(market_mode="market_aware", market_provider=MarketFixture("two"))
        result = self.collect(market_mode="market_aware", market_provider=MarketFixture("one"))
        self.assertEqual({row["status"] for row in result}, {"skipped_existing_assignment"})


if __name__ == "__main__":
    unittest.main()
