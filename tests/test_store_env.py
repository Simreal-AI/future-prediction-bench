import tempfile
import unittest
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from future_prediction_bench.demo import FixtureClock, FixtureProvider, fixture_questions, run_demo
from future_prediction_bench.env import PredictionEnv
from future_prediction_bench.store import Store


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.clock = FixtureClock()
        self.store = Store(":memory:", mode="fixture", clock=self.clock)
        self.question = fixture_questions()[0]
        self.store.add_question(self.question)

    def tearDown(self):
        self.store.close()

    def env(self, **kwargs):
        env = PredictionEnv(self.store, FixtureProvider())
        env.reset(self.question["question_id"], "checkpoint-001", **kwargs)
        return env

    def advance(self):
        self.clock.value = datetime(2030, 1, 3, 9, tzinfo=timezone.utc)

    def resolve(self, **kwargs):
        return self.store.resolve(self.question["question_id"],
                                  evidence_urls=["https://example.org/fixture/bulletin"],
                                  evidence_text="fixture evidence", **kwargs)

    def test_search_submit_delayed_reward_and_replay(self):
        env = self.env(track="rl")
        search = env.step({"action": "search", "query": "weather bulletin"})
        self.assertIsNone(search["reward"])
        final = env.step({"action": "submit", "probabilities": {"yes": .7, "no": .3}})
        self.assertTrue(final["terminated"])
        self.assertIsNone(final["reward"])
        self.assertEqual(self.store.export_training(), [])
        self.advance()
        self.resolve(outcome="yes")
        self.assertAlmostEqual(env.reward_status()["reward"], -.09)
        records = self.store.export_training()
        self.assertEqual(len(records), 1)
        self.assertFalse(records[0]["trainer_ready"])
        self.assertEqual(records[0]["events"][1]["loss_mask"], 0)
        self.assertTrue(records[0]["episode"]["receipt_sha256"])

    def test_invalid_immediate_penalty(self):
        env = self.env(track="rl")
        result = env.step({"action": "submit", "probabilities": {"yes": 1}})
        self.assertEqual(result["reward"], -1)
        self.assertFalse(result["info"]["awaiting_outcome"])
        self.assertEqual(len(self.store.export_training()), 1)
        self.assertEqual(self.store.summary()["groups"][0]["n_scored_questions"], 0)
        self.advance()
        self.resolve(outcome="yes")
        self.assertEqual(self.store.summary()["groups"][0]["brier_penalized_question_mean"], 1)

    def test_void_is_not_negative_outcome(self):
        env = self.env(track="rl", research_mode="no_search")
        env.step({"action": "submit", "probabilities": {"yes": .7, "no": .3}})
        self.advance()
        self.resolve(status="void")
        self.assertEqual(env.reward_status()["status"], "void")
        self.assertIsNone(env.reward_status()["reward"])
        self.assertEqual(self.store.export_training(), [])

    def test_no_post_deadline_submission_or_resolution_rewrite(self):
        env = self.env(research_mode="no_search")
        self.clock.value = datetime(2030, 1, 1, 23, tzinfo=timezone.utc)
        with self.assertRaises(ValueError):
            env.step({"action": "submit", "probabilities": {"yes": .7, "no": .3}})
        self.advance()
        first = self.resolve(outcome="yes")
        self.assertEqual(first, self.resolve(outcome="yes"))
        with self.assertRaises(ValueError):
            self.resolve(outcome="no")

    def test_split_isolation_and_immutable_question(self):
        alternative = deepcopy(self.question)
        alternative["question_id"] = "changed"
        alternative["split"] = "test"
        with self.assertRaises(ValueError):
            self.store.add_question(alternative)
        alternative = deepcopy(self.question)
        alternative["prompt"] += "changed wording"
        with self.assertRaises(ValueError):
            self.store.add_question(alternative)
        test = fixture_questions()[2]
        self.store.add_question(test)
        with self.assertRaises(ValueError):
            self.store.create_episode(test["question_id"], "model", track="rl")

    def test_submission_is_immutable(self):
        env = self.env(research_mode="no_search")
        env.step({"action": "submit", "probabilities": {"yes": .5, "no": .5}})
        with self.assertRaises(ValueError):
            self.store.submit(env.episode_id, {"yes": 1, "no": 0})
        with self.assertRaises(ValueError):
            env.step({"action": "search", "query": "new evidence"})

    def test_live_fixture_separation(self):
        with self.assertRaises(ValueError):
            Store(":memory:", clock=self.clock)
        store = Store(":memory:")
        try:
            with self.assertRaises(ValueError):
                store.add_question(self.question)
        finally:
            store.close()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.sqlite"
            Store(path, mode="fixture").close()
            with self.assertRaises(ValueError):
                Store(path, mode="live")

    def test_late_tool_return_is_audited_and_env_can_continue(self):
        clock = self.clock

        class SlowProvider(FixtureProvider):
            def search(self, query):
                clock.value = datetime(2030, 1, 1, 23, tzinfo=timezone.utc)
                return super().search(query)

        env = PredictionEnv(self.store, SlowProvider())
        env.reset(self.question["question_id"], "slow-model")
        result = env.step({"action": "search", "query": "query before deadline"})
        self.assertTrue(result["terminated"])
        self.assertEqual(result["observation"]["results"], [])
        self.assertEqual(env.reward_status()["status"], "missed")
        events = self.store.events(env.episode_id)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[-1]["payload"]["reason"], "forecast_deadline_reached")
        next_question = deepcopy(self.question)
        next_question.update(question_id="tomorrow", event_id="tomorrow", cluster_id="tomorrow",
                             forecast_deadline="2030-01-02T23:00:00Z", outcome_not_before="2030-01-03T00:00:00Z")
        self.store.add_question(next_question)
        observation = env.reset("tomorrow", "slow-model")
        self.assertEqual(observation["question"]["question_id"], "tomorrow")

    def test_benchmark_cannot_retry_same_question_and_config(self):
        self.env(research_mode="no_search")
        with self.assertRaises(ValueError):
            self.env(research_mode="no_search")
        # RL intentionally supports multiple rollouts from one behavior policy.
        self.env(track="rl", research_mode="no_search")
        self.env(track="rl", research_mode="no_search")

    def test_end_to_end_demo(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_demo(directory)
            self.assertEqual(result["pending_before_resolution"], 5)
            self.assertEqual(result["exported_train_trajectories"], 4)
            self.assertEqual(len(result["summary"]["groups"]), 2)
            self.assertTrue((Path(directory) / "training_trajectories.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
