"""Offline evidence replay tests using ResearchSession's real audit format."""

import copy
import hashlib
import json
import unittest
from datetime import datetime, timedelta, timezone

from future_prediction_bench.evidence_pack import (
    EvidenceNotRecorded, EvidencePack, EvidencePackError, EvidencePackProvider,
    synthetic_replay_microbenchmark,
)
from future_prediction_bench.research import ResearchSession


CAPTURED = datetime(2030, 1, 1, 12, tzinfo=timezone.utc)
DEADLINE = CAPTURED + timedelta(hours=1)
URL = "https://example.org/report"
QUESTION = {"issued_at": (CAPTURED - timedelta(hours=1)).isoformat(),
            "forecast_deadline": DEADLINE.isoformat()}


def signed(value):
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    canonical = json.dumps(unsigned, sort_keys=True, ensure_ascii=False,
                           separators=(",", ":"), allow_nan=False)
    value["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class FixtureProvider:
    def __init__(self):
        self.calls = []
        self.text = "Initial public evidence."
        self.results = None

    def search(self, query):
        self.calls.append(("search", query))
        if self.results is not None:
            return copy.deepcopy(self.results)
        return [{"url": URL, "title": "Recorded report", "text": self.text,
                 "published_at": (CAPTURED - timedelta(days=1)).isoformat()}]

    def open(self, url):
        self.calls.append(("open", url))
        return {"url": URL, "title": "Recorded page", "text": self.text,
                "published_at": (CAPTURED - timedelta(days=1)).isoformat()}


class EvidencePackTests(unittest.TestCase):
    def setUp(self):
        self.live = FixtureProvider()
        self.time = [CAPTURED]
        self.session = ResearchSession(QUESTION, self.live, market_mode="market_aware",
                                       clock=lambda: self.time[0], max_calls=8)

    def pack(self):
        self.session.search("forecast outlook")
        self.session.open(URL)
        return EvidencePack.from_events(self.session.events, DEADLINE)

    def test_roundtrip_exact_replay_and_original_retrieval_time(self):
        pack = self.pack()
        self.assertEqual(EvidencePack.from_dict(pack.to_dict()).sha256, pack.sha256)
        provider = EvidencePackProvider(pack, replay_as_of=CAPTURED)
        self.assertEqual(provider.search("forecast outlook")[0]["text"], "Initial public evidence.")
        self.assertEqual(provider.open(URL)["retrieved_at"], "2030-01-01T12:00:00Z")
        self.assertEqual(provider.public_config()["pack_sha256"], pack.sha256)
        self.assertNotIn("Initial public evidence", json.dumps(provider.public_config()))

        self.live.text = "Later provider mutation must not change the pack."
        document = pack.to_dict()
        document["searches"]["forecast outlook"][0]["snapshots"][0]["text"] = "Caller mutation"
        self.assertEqual(provider.search("forecast outlook")[0]["text"], "Initial public evidence.")

        replay_at = CAPTURED + timedelta(minutes=1)
        replay_session = ResearchSession(QUESTION, provider, market_mode="market_aware",
                                         clock=lambda: replay_at)
        replayed = replay_session.search("forecast outlook")
        opened = replay_session.open(URL)
        self.assertTrue(replayed["success"])
        self.assertTrue(opened["success"])
        self.assertEqual(replayed["results"][0]["retrieved_at"], "2030-01-01T12:00:00Z")
        self.assertEqual(replayed["results"][0]["observed_at"], "2030-01-01T12:01:00Z")
        self.assertEqual(self.live.calls, [("search", "forecast outlook"), ("open", URL)])
        self.assertEqual(provider.metrics()["search_calls"], 3)
        self.assertEqual(provider.metrics()["open_calls"], 2)

    def test_unknown_query_and_url_fail_closed(self):
        provider = self.pack().provider()
        with self.assertRaises(EvidenceNotRecorded):
            provider.search("unrecorded query")
        with self.assertRaises(EvidenceNotRecorded):
            provider.open("https://example.org/unrecorded")
        replay = ResearchSession(QUESTION, provider, clock=lambda: CAPTURED + timedelta(seconds=1))
        observation = replay.search("unrecorded query")
        self.assertEqual(observation["reason"], "provider_error")
        self.assertEqual(observation["error_type"], "EvidenceNotRecorded")
        self.assertEqual(provider.metrics()["misses"], 3)
        self.assertEqual(len(self.live.calls), 2)

    def test_replay_at_earlier_virtual_time_discards_later_retrieval(self):
        pack = self.pack()
        provider = pack.provider(replay_as_of=CAPTURED)
        earlier = ResearchSession(QUESTION, provider, clock=lambda: CAPTURED - timedelta(seconds=1))
        result = earlier.search("forecast outlook")
        self.assertFalse(result["success"])
        self.assertEqual(result["filter_reasons"], {"future_retrieved_at": 1})
        self.assertEqual(earlier.sources, {})

    def test_dated_versions_select_latest_available_record(self):
        self.session.search("forecast outlook")
        self.time[0] += timedelta(minutes=2)
        self.live.text = "Updated evidence at 12:02."
        self.session.search("forecast outlook")
        pack = EvidencePack.from_events(self.session.events, DEADLINE)
        early = pack.provider(replay_as_of=CAPTURED)
        late = pack.provider(replay_as_of=self.time[0])
        self.assertEqual(early.search("forecast outlook")[0]["text"], "Initial public evidence.")
        self.assertEqual(late.search("forecast outlook")[0]["text"], "Updated evidence at 12:02.")
        with self.assertRaises(EvidenceNotRecorded):
            pack.provider(replay_as_of=CAPTURED - timedelta(seconds=1)).search("forecast outlook")

    def test_market_content_is_refiltered_by_replay_session(self):
        self.live.results = [
            {"url": "https://polymarket.com/event/x", "title": "Market", "text": "Market odds are 75%."},
            {"url": URL, "title": "Public bulletin", "text": "Production rose this week."},
        ]
        captured = self.session.search("mixed sources")
        self.assertEqual(len(captured["results"]), 2)
        provider = EvidencePack.from_events(self.session.events, DEADLINE).provider()
        replay = ResearchSession(QUESTION, provider, market_mode="no_consensus",
                                 clock=lambda: CAPTURED + timedelta(seconds=1))
        result = replay.search("mixed sources")
        self.assertTrue(result["success"])
        self.assertEqual(len(result["results"]), 1)
        self.assertEqual(result["filter_reasons"], {"market_source": 1})
        self.assertEqual(result["results"][0]["url"], URL)

    def test_cutoff_future_source_and_tampering_are_rejected(self):
        pack = self.pack()
        events = copy.deepcopy(self.session.events)
        events[-1]["observed_at"] = DEADLINE.isoformat()
        signed(events[-1])
        with self.assertRaisesRegex(EvidencePackError, "post_cutoff"):
            EvidencePack.from_events(events, DEADLINE)

        for key in ("published_at", "source_updated_at", "retrieved_at"):
            with self.subTest(key=key):
                events = copy.deepcopy(self.session.events)
                snapshot = events[1]["payload"]["results"][0]
                snapshot[key] = (CAPTURED + timedelta(seconds=1)).isoformat()
                signed(snapshot)
                signed(events[1])
                with self.assertRaisesRegex(EvidencePackError, "future_" + key):
                    EvidencePack.from_events(events, DEADLINE)

        document = pack.to_dict()
        document["pages"][URL][0]["snapshot"]["text"] = "Tampered text"
        with self.assertRaises(EvidencePackError):
            EvidencePack.from_dict(document)
        document = pack.to_dict()
        document["pack_sha256"] = "0" * 64
        with self.assertRaisesRegex(EvidencePackError, "hash_mismatch"):
            EvidencePack.from_dict(document)

    def test_only_successful_observations_are_packed(self):
        self.session.search("forecast outlook")
        self.session.search("forecast outlook")
        self.assertEqual(len(self.session.events), 4)
        pack = EvidencePack.from_events(self.session.events, DEADLINE)
        self.assertEqual(len(pack.to_dict()["searches"]["forecast outlook"]), 2)
        self.assertEqual(pack.provider().public_config()["recorded_pages"], 0)
        with self.assertRaises(EvidencePackError):
            EvidencePack.from_events(self.session.events[:-1], DEADLINE)

    def test_bounded_synthetic_microbenchmark_reports_scope_and_call_counts(self):
        report = synthetic_replay_microbenchmark(iterations=2, slow_call_seconds=0.001)
        self.assertEqual(report["fixture"], "synthetic_sleep_provider_v1")
        self.assertIn("Offline synthetic", report["scope"])
        self.assertEqual(report["replay_metrics"]["search_calls"], 2)
        self.assertEqual(report["replay_metrics"]["open_calls"], 2)
        self.assertEqual(report["synthetic_provider_calls"], 4)
        for invalid in (0, 101, True):
            with self.assertRaises(ValueError):
                synthetic_replay_microbenchmark(iterations=invalid)


if __name__ == "__main__":
    unittest.main()
