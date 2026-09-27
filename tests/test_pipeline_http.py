"""Persistence, retry, and public export tests without external network access."""

import copy
import hashlib
import io
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from future_prediction_bench.cli import database_lock, main, public_dataset_question
from future_prediction_bench.demo import fixture_questions
from future_prediction_bench.http import FetchError, HttpClient
from future_prediction_bench.pipeline import Pipeline
from future_prediction_bench.store import Store


class Response(io.BytesIO):
    def geturl(self):
        return "https://earthquake.usgs.gov/test"


class Opener:
    def __init__(self, raw):
        self.raw = raw
        self.calls = 0

    def open(self, request, timeout):
        self.calls += 1
        return Response(self.raw)


class HttpTests(unittest.TestCase):
    def test_raw_snapshots_are_byte_exact_and_strictly_parsed(self):
        raw = b'{ "count": 0 }\n'
        with tempfile.TemporaryDirectory() as directory:
            client = HttpClient(directory, opener=Opener(raw))
            self.assertEqual(client.get_json("https://earthquake.usgs.gov/test"), {"count": 0})
            snapshot = client.last_snapshot
            self.assertEqual(Path(snapshot["path"]).read_bytes(), raw)
            self.assertEqual(snapshot["sha256"], hashlib.sha256(raw).hexdigest())

    def test_failed_json_does_not_leave_a_success_snapshot(self):
        for raw in (b'{"count":NaN}', b'{"count":1e400}', b'{"count":1,"count":2}', b'not json'):
            with self.subTest(raw=raw), tempfile.TemporaryDirectory() as directory:
                client = HttpClient(directory, opener=Opener(raw))
                with self.assertRaises(FetchError):
                    client.get_json("https://earthquake.usgs.gov/test")
                self.assertIsNone(client.last_snapshot)

    def test_nonallowlisted_urls_and_oversized_responses_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            opener = Opener(b'{}')
            client = HttpClient(directory, opener=opener)
            for url in ("http://earthquake.usgs.gov/test", "https://localhost/test", "https://u:p@earthquake.usgs.gov/test", "https://earthquake.usgs.gov:444/test"):
                with self.subTest(url=url), self.assertRaises(FetchError):
                    client.get_json(url)
            self.assertEqual(opener.calls, 0)
            client = HttpClient(directory, opener=Opener(b' ' * 20), max_bytes=10)
            with self.assertRaises(FetchError):
                client.get_json("https://earthquake.usgs.gov/test")


class Source:
    source_id = "mlb"
    question = None
    response = {"status": "pending", "reason": "not published"}
    calls = 0

    def discover(self, now, config, client):
        return [copy.deepcopy(self.question)]

    def resolve(self, question, now, client):
        type(self).calls += 1
        if isinstance(self.response, Exception):
            raise self.response
        return copy.deepcopy(self.response)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2030, 1, 1, 12, tzinfo=timezone.utc)
        self.store = Store(":memory:", mode="fixture", clock=lambda: self.now)
        self.config = {"sources": [{"id": "mlb"}], "pipeline": {"retry_base_seconds": 60, "max_resolution_attempts": 2}}
        self.pipeline = Pipeline(self.store, self.config, client=object())
        question = fixture_questions()[0]
        question["metadata"] = {"source_id": "mlb", "target_at": "2030-01-02T00:00:00Z", "private_note": "PRIVATE_SENTINEL"}
        Source.question = question
        Source.calls = 0
        Source.response = {"status": "pending", "reason": "not published"}
        self.pipeline.sources = [(Source(), {"id": "mlb"})]
        self.registry = patch("future_prediction_bench.pipeline.SOURCE_REGISTRY", {"mlb": Source})
        self.registry.start()

    def tearDown(self):
        self.registry.stop()
        self.store.close()

    def test_collection_is_immutable_and_source_change_enters_review(self):
        qid = Source.question["question_id"]
        self.assertEqual(self.pipeline.collect()["published"], [qid])
        Source.question["issued_at"] = "2030-01-01T01:00:00Z"
        self.assertEqual(self.pipeline.collect()["duplicates"], [qid])
        Source.question["prompt"] += " CHANGED"
        self.assertEqual(self.pipeline.collect()["review"][0]["question_id"], qid)
        self.assertNotIn("CHANGED", self.store.question(qid)["prompt"])
        self.assertEqual(self.pipeline.status()["questions"], 1)

    def test_no_fetch_before_due_and_retry_limit_requires_review_not_no(self):
        qid = Source.question["question_id"]
        self.pipeline.collect()
        self.pipeline.resolve_due()
        self.assertEqual(Source.calls, 0)
        self.now = datetime(2030, 1, 3, 9, tzinfo=timezone.utc)
        Source.response = TimeoutError("fixture timeout")
        self.assertEqual(self.pipeline.resolve_due()["pending"], [qid])
        self.pipeline.resolve_due()
        self.assertEqual(Source.calls, 1)
        self.now += timedelta(minutes=2)
        self.assertEqual(self.pipeline.resolve_due()["review"], [qid])
        self.assertIsNone(self.store.db.execute("SELECT 1 FROM resolutions").fetchone())
        self.pipeline.retry(qid)
        self.assertEqual(self.pipeline.status()["resolution_jobs"], {"pending": 1})

    def test_startup_repairs_orphan_even_after_discovery_window(self):
        self.store.add_question(Source.question)
        self.now = datetime(2030, 1, 5, tzinfo=timezone.utc)
        restarted = Pipeline(self.store, self.config, client=object())
        self.assertEqual(restarted.status()["resolution_jobs"], {"pending": 1})

    def test_public_question_export_excludes_private_metadata(self):
        exported = public_dataset_question(Source.question)
        self.assertNotIn("PRIVATE_SENTINEL", json.dumps(exported))
        self.assertEqual(exported["metadata"]["source_id"], "mlb")

    def test_collect_reports_source_failure_without_fabricated_questions(self):
        with patch.object(self.pipeline.sources[0][0], "discover", side_effect=TimeoutError("fixture")):
            result = self.pipeline.collect()
        self.assertEqual(result["published"], [])
        self.assertEqual(len(result["source_errors"]), 1)


class CliTests(unittest.TestCase):
    def test_lock_rejects_overlapping_local_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bench.sqlite"
            with database_lock(path):
                with self.assertRaises(ValueError):
                    with database_lock(path):
                        self.fail("Overlapping lock unexpectedly acquired")

    def test_existing_db_required_and_bad_config_rejected(self):
        with tempfile.TemporaryDirectory() as directory, patch("sys.stderr", new_callable=io.StringIO):
            self.assertEqual(main(["status", "--db", str(Path(directory) / "missing.sqlite")]), 2)
            self.assertEqual(main(["forecast", "--db", str(Path(directory) / "missing.sqlite"), "--limit", "0"]), 2)


if __name__ == "__main__":
    unittest.main()
