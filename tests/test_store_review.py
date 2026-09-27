"""Independent integration checks for submission, resolution, and aggregation."""

import unittest
from datetime import datetime, timezone

from future_prediction_bench.store import Store


def question(identifier="one", *, split="train"):
    return {
        "schema_version": "0.1", "question_id": identifier,
        "event_id": f"event-{identifier}", "cluster_id": f"cluster-{identifier}",
        "split": split, "kind": "binary", "prompt": "Synthetic event?",
        "options": [{"id": "yes", "text": "Yes"}, {"id": "no", "text": "No"}],
        "issued_at": "2027-01-01T00:00:00Z", "forecast_deadline": "2027-01-02T00:00:00Z",
        "outcome_not_before": "2027-01-03T00:00:00Z", "resolve_after": "2027-01-04T00:00:00Z",
        "resolution": {"criteria": "Use the synthetic daily flag.",
                       "source_urls": ["https://example.org/flags"], "on_ambiguous": "void"},
        "domain": "synthetic", "is_fixture": True,
    }


class StoreReviewTests(unittest.TestCase):
    def setUp(self):
        self.time = datetime(2027, 1, 1, 12, tzinfo=timezone.utc)
        self.store = Store(":memory:", mode="fixture", clock=lambda: self.time)
        self.addCleanup(self.store.close)

    def _create(self, question_id, *, track="rl"):
        return self.store.create_episode(question_id, "review-policy", track=track, research_mode="no_search")

    def _resolve(self, question_id, *, outcome="yes", status="resolved"):
        return self.store.resolve(question_id, outcome=outcome, status=status,
                                  evidence_urls=["https://example.org/flags/result"],
                                  evidence_text="Synthetic fixture result.")

    def test_submission_and_resolution_are_immutable_and_idempotent(self):
        self.store.add_question(question())
        episode_id = self._create("one")
        receipt = self.store.submit(episode_id, {"yes": 0.8, "no": 0.2})
        self.assertEqual(receipt["status"], "pending_reward")
        self.assertIsNone(receipt["reward"])
        with self.assertRaises(ValueError):
            self.store.submit(episode_id, {"yes": 1.0, "no": 0.0})
        with self.assertRaises(ValueError):
            self._resolve("one")
        self.time = datetime(2027, 1, 4, tzinfo=timezone.utc)
        resolution = self._resolve("one")
        graded = self.store.episode(episode_id)
        self.assertAlmostEqual(graded["reward"], -0.04)
        self.assertEqual(self._resolve("one"), resolution)
        self.assertEqual(self.store.episode(episode_id), graded)
        with self.assertRaises(ValueError):
            self._resolve("one", outcome="no")
        self.assertEqual(self.store.episode(episode_id), graded)
        self.assertEqual(len(self.store.export_training()), 1)

    def test_test_split_cannot_be_trained_or_crossed_by_related_questions(self):
        self.store.add_question(question("training"))
        self.store.add_question(question("heldout", split="test"))
        with self.assertRaises(ValueError):
            self._create("heldout", track="rl")
        for field in ("event_id", "cluster_id"):
            related = question("alias", split="test")
            related[field] = question("training")[field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.store.add_question(related)
        training = self._create("training")
        heldout = self._create("heldout", track="benchmark")
        self.store.submit(training, {"yes": 0.6, "no": 0.4})
        self.store.submit(heldout, {"yes": 0.6, "no": 0.4})
        self.time = datetime(2027, 1, 4, tzinfo=timezone.utc)
        self._resolve("training")
        self._resolve("heldout")
        exported = self.store.export_training()
        self.assertEqual([item["episode"]["episode_id"] for item in exported], [training])

    def test_void_does_not_create_outcome_reward_or_training_supervision(self):
        self.store.add_question(question())
        valid = self._create("one")
        invalid = self._create("one")
        self.store.submit(valid, {"yes": 0.6, "no": 0.4})
        malformed = self.store.submit(invalid, {"yes": True, "no": False})
        self.assertEqual(malformed["status"], "invalid")
        self.assertEqual(malformed["reward"], -1.0)
        self.time = datetime(2027, 1, 4, tzinfo=timezone.utc)
        self._resolve("one", status="void", outcome=None)
        self.assertEqual(self.store.episode(valid)["status"], "void")
        self.assertIsNone(self.store.episode(valid)["reward"])
        self.assertEqual(self.store.export_training(), [])
        group = self.store.summary()["groups"][0]
        self.assertEqual(group["n_scored_questions"], 0)
        self.assertIsNone(group["brier_penalized_question_mean"])

    def test_rollouts_are_averaged_within_question_before_overall_mean(self):
        self.store.add_question(question("one"))
        self.store.add_question(question("two"))
        for _ in range(3):
            self.store.submit(self._create("one"), {"yes": 1.0, "no": 0.0})
        self.store.submit(self._create("two"), {"yes": 0.0, "no": 1.0})
        self.time = datetime(2027, 1, 4, tzinfo=timezone.utc)
        self._resolve("one")
        self._resolve("two")
        group = self.store.summary()["groups"][0]
        self.assertEqual(group["n_scored_questions"], 2)
        self.assertEqual(group["n_scored_rollouts"], 4)
        self.assertEqual(group["brier_penalized_question_mean"], 0.5)

    def test_resolved_unanswered_question_cannot_disappear_from_penalized_score(self):
        self.store.add_question(question("answered"))
        self.store.add_question(question("abandoned"))
        self.store.submit(self._create("answered"), {"yes": 1.0, "no": 0.0})
        abandoned = self._create("abandoned")
        self.time = datetime(2027, 1, 4, tzinfo=timezone.utc)
        self._resolve("answered")
        self._resolve("abandoned")
        group = self.store.summary()["groups"][0]
        self.assertEqual(group["n_scored_questions"], 2)
        self.assertEqual(group["brier_penalized_question_mean"], 0.5)
        self.assertNotEqual(self.store.episode(abandoned)["status"], "active")
        self.assertNotIn(abandoned, [item["episode"]["episode_id"]
                                     for item in self.store.export_training()])


if __name__ == "__main__":
    unittest.main()
