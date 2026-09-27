"""Recovery regressions from an independent automation review."""

import copy
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from future_prediction_bench.http import strict_json_loads
from future_prediction_bench.pipeline import Pipeline
from future_prediction_bench.store import Store


class ReviewSource:
    source_id = "mlb"
    calls = 0

    def discover(self, now, config, client):
        return [copy.deepcopy(REVIEW_QUESTION)]

    def resolve(self, question, now, client):
        type(self).calls += 1
        return {
            "status": "resolved", "outcome": "yes" if self.calls == 1 else "no",
            "evidence_urls": ["https://statsapi.mlb.com/review-fixture"],
            "evidence_text": "Synthetic first successful snapshot",
            "source_snapshots": [{"sha256": "a" * 64, "path": "fixture.json", "observed_at": now.isoformat()}],
        }


REVIEW_QUESTION = {
    "schema_version": "0.1", "is_fixture": True,
    "question_id": "recovery-fixture", "event_id": "recovery-fixture", "cluster_id": "recovery-fixture",
    "split": "test", "kind": "binary", "domain": "synthetic",
    "prompt": "Will the synthetic event happen?",
    "options": [{"id": "yes", "text": "Yes"}, {"id": "no", "text": "No"}],
    "issued_at": "2030-01-01T00:00:00Z", "forecast_deadline": "2030-01-02T00:00:00Z",
    "outcome_not_before": "2030-01-03T00:00:00Z", "resolve_after": "2030-01-03T01:00:00Z",
    "resolution": {"criteria": "Use the first valid synthetic source snapshot.", "on_ambiguous": "void",
                   "source_urls": ["https://statsapi.mlb.com/review-fixture"]},
    "metadata": {"source_id": "mlb", "target_at": "2030-01-03T00:00:00Z"},
}


class AutomationRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.time = datetime(2030, 1, 1, 1, tzinfo=timezone.utc)
        self.store = Store(":memory:", mode="fixture", clock=lambda: self.time)
        self.pipeline = Pipeline(self.store, {"sources": [{"id": "mlb"}]}, client=object())
        self.pipeline.sources = [(ReviewSource(), {"id": "mlb"})]
        ReviewSource.calls = 0

    def tearDown(self):
        self.store.close()

    def test_rediscovery_repairs_question_committed_before_job_creation(self):
        # This is the durable state left by a crash between the two commits.
        self.store.add_question(REVIEW_QUESTION)
        self.assertEqual(self.pipeline.collect()["duplicates"], ["recovery-fixture"])
        self.assertIsNotNone(self.store.db.execute(
            "SELECT 1 FROM resolution_jobs WHERE question_id='recovery-fixture'"
        ).fetchone(), "An immutable published question must regain its settlement job")

    def test_retry_reuses_first_valid_proposal_after_storage_failure(self):
        self.pipeline.collect()
        self.time = datetime(2030, 1, 3, 2, tzinfo=timezone.utc)
        with patch("future_prediction_bench.pipeline.SOURCE_REGISTRY", {"mlb": ReviewSource}):
            with patch.object(self.store, "resolve", side_effect=OSError("Synthetic transient write failure")):
                first = self.pipeline.resolve_due()
            self.assertTrue(first["errors"])
            self.time += timedelta(hours=1)
            self.pipeline.resolve_due()
        self.assertEqual(ReviewSource.calls, 1, "Do not fetch a revised label after a valid snapshot is captured")
        row = self.store.db.execute("SELECT payload FROM resolutions WHERE question_id='recovery-fixture'").fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(strict_json_loads(row[0])["outcome"], "yes")

    def test_strict_json_rejects_numeric_overflow_as_well_as_nan_literals(self):
        for text in ['{"value":1e400}', '{"value":-1e400}', '{"nested":[1e400]}', '{"value":NaN}']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                strict_json_loads(text)


if __name__ == "__main__":
    unittest.main()
