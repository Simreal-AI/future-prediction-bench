"""Private baseline selection, sealing, export boundaries, and delayed rewards."""

import copy
import json
import unittest
from datetime import datetime, timezone

from future_prediction_bench.baselines import seal_baselines
from future_prediction_bench.cli import public_dataset_question
from future_prediction_bench.demo import FixtureClock, FixtureProvider, fixture_questions
from future_prediction_bench.env import PredictionEnv
from future_prediction_bench.schema import public_question
from future_prediction_bench.store import Store, digest


class FakeMarket:
    def __init__(self, market):
        self.market = market
        self.last_snapshot = {"observed_at": "2030-01-01T12:00:00Z", "sha256": "fixture-snapshot"}
        self.urls = []

    def get_json(self, url):
        self.urls.append(url)
        return copy.deepcopy(self.market)


class FakeInternal:
    def __init__(self, probabilities):
        self.probabilities, self.requests = probabilities, []

    def public_config(self):
        return {"model": "private-baseline-fixture", "temperature": 0}

    def complete(self, messages, tools, **kwargs):
        self.requests.append(copy.deepcopy({"messages": messages, "tools": tools, **kwargs}))
        return {"choices": [{"message": {"role": "assistant", "content": json.dumps({"probabilities": self.probabilities})}}]}


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.clock = FixtureClock()
        self.store = Store(":memory:", mode="fixture", clock=self.clock)
        self.question = fixture_questions()[0]
        self.question["metadata"] = {"private_answer": "private-question-marker", "source_id": "fixture"}
        self.store.add_question(self.question)
        self.qid = self.question["question_id"]
        self.market = {"id": "123", "question": "Equivalent reviewed fixture event", "description": "Exact frozen resolution rules",
                       "endDate": "2030-01-02T23:00:00Z", "active": True, "closed": False, "acceptingOrders": True,
                       "updatedAt": "2030-01-01T11:59:00Z", "outcomes": '["Yes","No"]', "outcomePrices": '["0.65","0.35"]'}
        self.mapping = {"provider": "polymarket", "reviewed": True, "market_id": "123",
                        "question_sha256": digest(public_question(self.question)), "market_question": self.market["question"],
                        "market_description_sha256": digest(self.market["description"]), "market_end_date": self.market["endDate"],
                        "resolution_equivalence_note": "Same event, cutoff, outcomes, and resolution source.",
                        "outcome_map": {"Yes": "yes", "No": "no"}}

    def tearDown(self):
        self.store.close()

    def seal(self, *, mapping=None, market=None, model=None):
        return seal_baselines(self.store, [self.qid], config={"market_mappings": {self.qid: mapping or self.mapping}},
                              client=FakeMarket(market or self.market), internal_model=model)[0]

    def episode(self, qid=None):
        env = PredictionEnv(self.store, FixtureProvider())
        observation = env.reset(qid or self.qid, "tested-policy", track="rl", research_mode="no_search", reward_mode="baseline_improvement")
        return env, observation

    def resolve(self, qid=None, *, outcome="yes", status="resolved"):
        self.clock.value = datetime(2030, 1, 3, 9, tzinfo=timezone.utc)
        self.store.resolve(qid or self.qid, outcome=outcome, status=status,
                           evidence_urls=["https://example.org/fixture/bulletin"], evidence_text="Synthetic resolved evidence")

    def test_reviewed_market_has_priority_over_internal_inference(self):
        model = FakeInternal({"yes": .37, "no": .63})
        report = self.seal(model=model)
        self.assertEqual(report["kind"], "market")
        self.assertEqual(model.requests, [])
        self.assertEqual(self.store.private_baseline(self.qid)["probabilities"], {"yes": .65, "no": .35})
        self.assertNotIn("probabilities", report)

    def test_market_mapping_rejects_changed_question_rules_date_or_outcomes(self):
        cases = [("mapping", "question_sha256", "different-version"), ("mapping", "reviewed", False),
                 ("mapping", "resolution_equivalence_note", ""), ("mapping", "market_id", "../123"),
                 ("mapping", "outcome_map", {"Yes": "yes", "No": "yes"}),
                 ("market", "question", "Changed question"), ("market", "description", "Changed rules"),
                 ("market", "endDate", "2030-01-04T23:00:00Z"), ("market", "outcomes", '["Yes","Maybe"]'),
                 ("market", "outcomePrices", '["0.7","0.5"]'), ("market", "outcomePrices", [True, 0]),
                 ("market", "updatedAt", "2029-12-31T12:00:00Z"), ("market", "closed", True)]
        for target, field, value in cases:
            with self.subTest(target=target, field=field):
                mapping, market = copy.deepcopy(self.mapping), copy.deepcopy(self.market)
                (mapping if target == "mapping" else market)[field] = value
                report = self.seal(mapping=mapping, market=market)
                self.assertEqual(report["status"], "missing")
                self.assertEqual(report["market_match"], "unavailable_or_invalid")
                self.assertIsNone(self.store.private_baseline(self.qid))

    def test_invalid_market_falls_back_to_actual_internal_response(self):
        model = FakeInternal({"yes": .37, "no": .63})
        report = self.seal(mapping={**self.mapping, "reviewed": False}, model=model)
        self.assertEqual(report["kind"], "internal_model")
        self.assertEqual(self.store.private_baseline(self.qid)["probabilities"], {"yes": .37, "no": .63})
        self.assertEqual(len(model.requests), 1)
        self.assertEqual(model.requests[0]["tools"], [])
        self.assertNotIn("private-question-marker", json.dumps(model.requests))
        self.assertIn("private-baseline-fixture", json.dumps(self.store.private_baseline(self.qid)["metadata"]))

    def test_missing_or_invalid_internal_model_never_fabricates_baseline(self):
        for model in (None, FakeInternal({"yes": .3}), FakeInternal({"yes": .7, "no": .6})):
            with self.subTest(model=model):
                report = seal_baselines(self.store, [self.qid], client=FakeMarket(self.market), internal_model=model)[0]
                self.assertEqual(report["status"], "missing")
                self.assertIsNone(self.store.private_baseline(self.qid))

    def test_binary_and_multiclass_reward_improvement(self):
        cases = [(self.question, {"yes": .6, "no": .4}, {"yes": .8, "no": .2}, "yes", .12),
                 (fixture_questions()[1], {"low": .2, "medium": .3, "high": .5},
                  {"low": .1, "medium": .7, "high": .2}, "medium", .32)]
        episodes = []
        for question, baseline, candidate, outcome, expected in cases:
            self.store.add_question(question)
            qid = question["question_id"]
            self.store.seal_baseline(qid, baseline, kind="internal_model", identity="fixed-fixture")
            env, _ = self.episode(qid)
            result = env.step({"action": "submit", "probabilities": candidate})
            self.assertIsNone(result["reward"])
            episodes.append((env, qid, outcome, expected))
        for env, qid, outcome, expected in episodes:
            self.resolve(qid, outcome=outcome)
            self.assertAlmostEqual(env.reward_status()["reward"], expected)

    def test_worse_forecast_receives_negative_baseline_improvement(self):
        self.store.seal_baseline(self.qid, {"yes": .8, "no": .2}, kind="internal_model", identity="fixed-fixture")
        env, _ = self.episode()
        env.step({"action": "submit", "probabilities": {"yes": .6, "no": .4}})
        self.resolve()
        self.assertAlmostEqual(env.reward_status()["reward"], -.12)

    def test_baseline_must_precede_any_episode_and_deadline(self):
        self.store.create_episode(self.qid, "old-policy", research_mode="no_search")
        with self.assertRaisesRegex(ValueError, "before any forecast"):
            self.store.seal_baseline(self.qid, {"yes": .5, "no": .5}, kind="market", identity="fixture")
        question = fixture_questions()[1]
        self.store.add_question(question)
        self.clock.value = datetime(2030, 1, 1, 23, tzinfo=timezone.utc)
        with self.assertRaises(ValueError):
            self.store.seal_baseline(question["question_id"], {"low": .2, "medium": .3, "high": .5}, kind="internal_model", identity="fixture")

    def test_sealed_baseline_is_immutable_and_idempotent_pipeline_skips_it(self):
        self.seal()
        with self.assertRaisesRegex(ValueError, "immutable"):
            self.store.seal_baseline(self.qid, {"yes": .5, "no": .5}, kind="internal_model", identity="replacement")
        model = FakeInternal({"yes": .5, "no": .5})
        self.assertEqual(self.seal(model=model)["status"], "already_sealed")
        self.assertEqual(model.requests, [])

    def test_baseline_private_in_observations_dataset_and_training_exports(self):
        self.store.seal_baseline(self.qid, {"yes": .98765, "no": .01235}, kind="internal_model",
                                 identity="private-baseline-marker", metadata={"private_detail": "hidden-baseline-detail"})
        env, observation = self.episode()
        env.step({"action": "submit", "probabilities": {"yes": .8, "no": .2}})
        self.resolve()
        exports = [observation, public_dataset_question(self.store.question(self.qid)), self.store.export_training()]
        self.assertEqual(len(exports[-1]), 1)
        for exported in exports:
            encoded = json.dumps(exported)
            for private in ("private-baseline-marker", "hidden-baseline-detail", "0.98765", "private-question-marker"):
                self.assertNotIn(private, encoded)

    def test_invalid_penalty_and_void_reward_are_unchanged(self):
        self.seal()
        invalid, _ = self.episode()
        self.assertEqual(invalid.step({"action": "submit", "probabilities": {"yes": 1}})["reward"], -1)
        valid, _ = self.episode()
        valid.step({"action": "submit", "probabilities": {"yes": .5, "no": .5}})
        self.resolve(outcome=None, status="void")
        self.assertEqual(valid.reward_status()["status"], "void")
        self.assertIsNone(valid.reward_status()["reward"])
        self.assertEqual(self.store.export_training(), [])


if __name__ == "__main__":
    unittest.main()
