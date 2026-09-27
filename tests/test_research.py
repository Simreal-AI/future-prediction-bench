"""Gateway invariants use offline fixtures and trusted deterministic clocks."""

import hashlib
import json
import unittest
from datetime import datetime, timedelta, timezone

from future_prediction_bench.research import ResearchSession


NOW = datetime(2026, 9, 22, 3, 0, tzinfo=timezone.utc)
QUESTION = {
    "question_id": "gateway-test",
    "issued_at": (NOW - timedelta(days=1)).isoformat(),
    "forecast_deadline": (NOW + timedelta(hours=1)).isoformat(),
}


def document(url="https://example.org/report", text="Observed production increased.", **extra):
    return {"url": url, "title": "Report", "text": text, **extra}


class FixtureProvider:
    def __init__(self, results=None, page=None, on_call=None):
        self.results = results if results is not None else [document()]
        self.page = page if page is not None else document()
        self.on_call = on_call
        self.calls = []

    def search(self, query):
        self.calls.append(("search", query))
        if self.on_call:
            self.on_call()
        return self.results

    def open(self, url):
        self.calls.append(("open", url))
        if self.on_call:
            self.on_call()
        return self.page


class ResearchTests(unittest.TestCase):
    def session(self, provider=None, **kwargs):
        return ResearchSession(QUESTION, provider or FixtureProvider(), clock=lambda: NOW, **kwargs)

    def test_full_snapshots_masks_timestamps_and_hashes(self):
        provider = FixtureProvider()
        session = self.session(provider)
        result = session.search("production outlook")
        session.open("https://example.org/report")
        self.assertEqual(session.successful_searches, 1)
        self.assertEqual(len(session.events), 4)
        self.assertEqual([event["loss_mask"] for event in session.events], [1, 0, 1, 0])
        for event in session.events:
            self.assertEqual(event["at"], "2026-09-22T03:00:00Z")
            self.assertEqual(event["at"], event["observed_at"])
            unsigned = {key: value for key, value in event.items() if key != "sha256"}
            canonical = json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
            self.assertEqual(hashlib.sha256(canonical.encode()).hexdigest(), event["sha256"])
        result["results"][0]["text"] = "Model-side mutation"
        provider.results[0]["text"] = "Provider-side mutation"
        self.assertEqual(session.events[1]["payload"]["results"][0]["text"], "Observed production increased.")

    def test_filters_market_domains_subdomains_and_consensus_snippets(self):
        provider = FixtureProvider(results=[
            document("https://api.polymarket.com/x"),
            document("https://kalshi.com/x"),
            document("https://www.metaculus.com/x"),
            document("https://manifold.markets/x"),
            document("https://www.predictit.org/x"),
            document("https://oddsportal.com/x"),
            document(text="The market probability is now 63%."),
            document(text="The consensus forecast is 0.71."),
            document(text="预测市场概率为 65%。"),
            document(text="Prediction markets put the chance at 60%."),
            document(text="Rainfall increased 20% year on year."),
        ])
        session = self.session(provider)
        result = session.search("outlook")
        self.assertEqual(len(result["results"]), 1)
        self.assertEqual(result["filtered_count"], 10)
        self.assertEqual(result["filter_reasons"], {"market_source": 6, "consensus_text": 4})
        self.assertNotIn("63%", json.dumps(session.events))
        self.assertTrue(result["success"])

    def test_empty_or_fully_filtered_search_is_not_successful(self):
        session = self.session(FixtureProvider(results=[document("https://kalshi.com/x")]))
        self.assertFalse(session.search("outlook")["success"])
        session.provider.results = []
        self.assertFalse(session.search("again")["success"])
        self.assertEqual(session.successful_searches, 0)

    def test_market_aware_preserves_and_marks_exposure(self):
        session = self.session(FixtureProvider(results=[document("https://kalshi.com/x", "Market odds are 75%.")]),
                               market_mode="market_aware")
        result = session.search("outlook")
        self.assertTrue(result["consensus_exposure"])
        self.assertTrue(result["results"][0]["is_market_source"])
        self.assertTrue(result["results"][0]["contains_consensus_text"])
        self.assertEqual(result["filtered_count"], 0)

    def test_open_blocks_market_before_provider_and_checks_redirect(self):
        provider = FixtureProvider(page=document("https://www.polymarket.com/redirected"))
        session = self.session(provider)
        self.assertEqual(session.open("https://kalshi.com/x")["reason"], "market_source")
        self.assertEqual(provider.calls, [])
        self.assertEqual(session.open("https://example.org/redirect")["reason"], "market_source")
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(len(session.events), 4)

    def test_open_does_not_accept_ambiguous_or_non_web_urls(self):
        session = self.session(max_calls=20)
        for url in ["file:///private/secret", "https://kalshi%2ecom/x", "https://kalshi.com\\@example.org", "https://user@kalshi.com/x"]:
            with self.subTest(url=url):
                self.assertEqual(session.open(url)["reason"], "invalid_url")
        self.assertEqual(session.provider.calls, [])

    def test_temporal_boundaries_are_host_enforced(self):
        for at, expected in [
            (NOW - timedelta(days=2), "question_not_issued"),
            (NOW + timedelta(hours=1), "forecast_deadline_reached"),
        ]:
            with self.subTest(at=at):
                provider = FixtureProvider()
                session = ResearchSession(QUESTION, provider, clock=lambda: at)
                self.assertEqual(session.search("outlook")["reason"], expected)
                self.assertEqual(session.open("https://example.org/report")["reason"], expected)
                self.assertEqual(provider.calls, [])
                self.assertEqual(len(session.events), 4)

    def test_provider_finishing_at_deadline_cannot_deliver_late_data(self):
        for tool in ("search", "open"):
            with self.subTest(tool=tool):
                time = [NOW]
                provider = FixtureProvider(on_call=lambda: time.__setitem__(0, NOW + timedelta(hours=1)))
                session = ResearchSession(QUESTION, provider, clock=lambda: time[0])
                result = getattr(session, tool)("outlook" if tool == "search" else "https://example.org/report")
                self.assertEqual(result["reason"], "forecast_deadline_reached")
                self.assertFalse(result["success"])
                self.assertNotIn("Observed production", json.dumps(session.events))
                self.assertEqual(session.successful_searches, 0)
                self.assertEqual(session.events[-1]["observed_at"], "2026-09-22T04:00:00Z")

    def test_publication_dates_future_invalid_and_unknown(self):
        provider = FixtureProvider(results=[
            document(published_at=(NOW + timedelta(seconds=1)).isoformat()),
            document(published_at="2026-09-21T12:00:00"),
            document(published_at="invalid"),
            document(published_at=(NOW - timedelta(hours=1)).isoformat()),
            document(),
        ])
        result = self.session(provider).search("outlook")
        self.assertEqual(result["filtered_count"], 3)
        self.assertEqual(result["filter_reasons"], {"future_published_at": 1, "invalid_published_at": 2})
        self.assertEqual([item["published_at_status"] for item in result["results"]], ["known", "unknown"])

    def test_call_budget_counts_locally_blocked_attempts(self):
        session = self.session(max_calls=1)
        session.open("https://kalshi.com/x")
        result = session.search("outlook")
        self.assertEqual(result["reason"], "call_budget_exhausted")
        self.assertEqual(session.calls_used, 1)
        self.assertEqual(session.provider.calls, [])
        self.assertEqual(len(session.events), 4)

    def test_provider_error_is_logged_without_exposing_error_content(self):
        def fail():
            raise RuntimeError("untrusted source content")
        session = self.session(FixtureProvider(on_call=fail))
        result = session.search("outlook")
        self.assertEqual(result["reason"], "provider_error")
        self.assertEqual(result["error_type"], "RuntimeError")
        self.assertNotIn("untrusted source content", json.dumps(session.events))
        self.assertEqual(session.events[-1]["loss_mask"], 0)

    def test_invalid_session_configuration_fails_early(self):
        for options in [{"market_mode": "clean"}, {"max_calls": 0}, {"max_calls": True}]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.session(**options)


if __name__ == "__main__":
    unittest.main()
