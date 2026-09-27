"""Analyst tool integration with deterministic, public-data fixtures only."""

import copy
import json
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from future_prediction_bench.analyst import (PolymarketPublicProvider, TOOLKIT_VERSION,
                                            calculate, tool_definitions)
from future_prediction_bench.demo import FixtureClock, FixtureProvider, fixture_questions
from future_prediction_bench.env import PredictionEnv
from future_prediction_bench.providers import ProviderError
from future_prediction_bench.research import ResearchSession
from future_prediction_bench.store import Store, digest


def market(identifier="123", **changes):
    return {"id": identifier, "question": "Will the specified event happen?", "description": "Frozen market rules.",
            "outcomes": '["Yes", "No"]', "outcomePrices": '["0.63", "0.36"]',
            "active": True, "closed": False, "updatedAt": "2030-01-01T08:00:00Z", **changes}


class MarketFixture:
    def __init__(self, on_call=None):
        self.calls = []
        self.on_call = on_call

    def snapshot(self, market_id, **kwargs):
        self.calls.append(("snapshot", market_id))
        if self.on_call:
            self.on_call()
        return PolymarketPublicProvider._market(market(market_id))

    def search(self, query, **kwargs):
        self.calls.append(("search", query))
        if self.on_call:
            self.on_call()
        return [PolymarketPublicProvider._market(market())]

    def public_config(self):
        return {"adapter": "offline_market_test"}


class CalculatorTests(unittest.TestCase):
    def test_arithmetic_and_rejected_code_or_resource_exhaustion(self):
        self.assertAlmostEqual(calculate("(0.6 * 0.8) / (0.6 * 0.8 + 0.4 * 0.2)"), 6 / 7)
        self.assertEqual(calculate("-(2 ** 3) + 10"), 2)
        for expression in ["__import__('os').system('true')", "open('/tmp/x')", "(1).__class__", "[1, 2]",
                           "True", "1 / 0", "9 ** 9999999", "1e309", "'abc' * 5", "(-1) ** 0.5", "+" * 1000 + "1"]:
            with self.subTest(expression=expression), self.assertRaises(ValueError):
                calculate(expression)


class AnalystSessionTests(unittest.TestCase):
    def setUp(self):
        self.clock = FixtureClock()
        self.question = fixture_questions()[0]

    def session(self, **kwargs):
        return ResearchSession(self.question, FixtureProvider(), clock=self.clock, **kwargs)

    def test_shared_budget_strict_arguments_and_audit_hashes(self):
        session = self.session(max_calls=3)
        self.assertTrue(session.dispatch("search", {"query": "weather"})["success"])
        self.assertEqual(session.dispatch("calculator", {"expression": "2 + 3"})["value"], 5)
        malformed = session.dispatch("calculator", {"expression": "2", "unexpected": True})
        self.assertEqual(malformed["reason"], "invalid_tool_arguments")
        exhausted = session.dispatch("open", {"url": "https://example.org/report"})
        self.assertEqual(exhausted["reason"], "call_budget_exhausted")
        self.assertEqual(exhausted["calls_remaining"], 0)
        self.assertEqual(session.calls_used, 3)
        for index, event in enumerate(session.events):
            self.assertEqual(event["loss_mask"], 1 if index % 2 == 0 else 0)
            self.assertEqual(event["toolkit_version"], TOOLKIT_VERSION)
            self.assertEqual(event["sha256"], digest({key: value for key, value in event.items() if key != "sha256"}))

    def test_notebook_and_revisions_require_sources_from_this_session(self):
        session = self.session(max_calls=12)
        source = session.dispatch("search", {"query": "weather"})["results"][0]
        note = session.dispatch("notebook", {"claim": "The bulletin supports this estimate.", "source_hashes": [source["sha256"]]})
        self.assertFalse(note["note"]["support_verified"])
        self.assertEqual(note["note"]["sources"][0]["url"], source["url"])
        note["note"]["claim"] = "mutated"
        self.assertNotEqual(session.notebook.notes[0]["claim"], "mutated")
        arguments = {"probabilities": {"yes": .6, "no": .4}, "rationale": "Initial estimate.", "source_hashes": [source["sha256"]]}
        first = session.dispatch("draft", arguments)
        arguments["probabilities"] = {"yes": .7, "no": .3}
        second = session.dispatch("draft", arguments)
        self.assertEqual(second["draft"]["revision"], 2)
        self.assertEqual(second["draft"]["previous_probabilities"], first["draft"]["probabilities"])
        self.assertFalse(second["draft"]["submitted"])
        self.assertNotIn("reward", second)
        self.assertFalse(self.session().dispatch("notebook", {"claim": "Unknown", "source_hashes": [source["sha256"]]})["success"])
        self.assertFalse(session.dispatch("draft", {**arguments, "probabilities": {"yes": 1}})["success"])
        self.assertEqual(len(session.notebook.drafts), 2)

    def test_market_mode_gate_prevents_provider_call(self):
        provider = MarketFixture()
        session = self.session(market_provider=provider)
        result = session.dispatch("market_snapshot", {"market_id": "123"})
        self.assertEqual(result["reason"], "market_tools_require_market_aware")
        self.assertEqual(provider.calls, [])
        names = lambda mode: {entry["function"]["name"] for entry in tool_definitions(self.question, market_mode=mode)}
        self.assertNotIn("market_search", names("no_consensus"))
        self.assertIn("market_snapshot", names("market_aware"))
        self.assertEqual([entry["function"]["name"] for entry in tool_definitions(self.question, "no_search", "market_aware")], ["submit"])

    def test_market_observations_are_hashed_sources_and_no_matched_baseline(self):
        session = self.session(market_mode="market_aware", market_provider=MarketFixture())
        observation = session.dispatch("market_search", {"query": "the specified event"})
        source = observation["results"][0]
        self.assertTrue(source["consensus_exposure"])
        self.assertFalse(observation["match_verified"])
        self.assertEqual(json.loads(source["text"])["outcomePrices"], [.63, .36])
        self.assertIn(source["sha256"], session.sources)
        self.assertEqual(session.successful_searches, 0)

    def test_late_market_return_is_discarded_and_not_usable_as_evidence(self):
        def advance():
            self.clock.value = datetime.fromisoformat(self.question["forecast_deadline"].replace("Z", "+00:00"))
        provider = MarketFixture(on_call=advance)
        session = self.session(market_mode="market_aware", market_provider=provider)
        result = session.dispatch("market_snapshot", {"market_id": "123"})
        self.assertEqual(result["reason"], "forecast_deadline_reached")
        self.assertNotIn("results", result)
        self.assertEqual(session.sources, {})
        self.assertEqual(session.events[-1]["loss_mask"], 0)

    def test_market_update_timestamp_cannot_be_in_the_future(self):
        class FutureMarket(MarketFixture):
            def snapshot(self, market_id, **kwargs):
                return PolymarketPublicProvider._market(market(market_id, updatedAt="2099-01-01T00:00:00Z"))
        session = self.session(market_mode="market_aware", market_provider=FutureMarket())
        result = session.dispatch("market_snapshot", {"market_id": "123"})
        self.assertFalse(result["success"])
        self.assertEqual(result["filter_reasons"], {"future_source_updated_at": 1})
        self.assertEqual(session.sources, {})

    def test_environment_draft_does_not_submit_or_reward_and_baseline_stays_private(self):
        with_store = Store(":memory:", mode="fixture", clock=self.clock)
        try:
            with_store.add_question(self.question)
            with_store.seal_baseline(self.question["question_id"], {"yes": .98765, "no": .01235},
                                     kind="internal_model", identity="hidden-private-model")
            env = PredictionEnv(with_store, FixtureProvider(), market_provider=MarketFixture())
            reset = env.reset(self.question["question_id"], "test-agent", track="rl", market_mode="market_aware", reward_mode="baseline_improvement")
            env.step({"action": "search", "query": "weather"})
            result = env.step({"action": "draft", "probabilities": {"yes": .6, "no": .4}, "rationale": "Tentative", "source_hashes": []})
            self.assertFalse(result["terminated"])
            self.assertIsNone(result["reward"])
            self.assertEqual(with_store.episode(env.episode_id)["status"], "active")
            market_result = env.step({"action": "market_snapshot", "market_id": "123"})
            transcript = json.dumps([reset, result, market_result, with_store.events(env.episode_id)])
            self.assertNotIn("hidden-private-model", transcript)
            self.assertNotIn("0.98765", transcript)
            env.step({"action": "submit", "probabilities": {"yes": .7, "no": .3}})
            self.assertEqual(with_store.episode(env.episode_id)["status"], "pending_reward")
        finally:
            with_store.close()


class MarketProviderTests(unittest.TestCase):
    def test_fixed_public_get_requests_and_bounded_results(self):
        calls = []
        def request(url, **kwargs):
            calls.append((url, kwargs))
            return {"events": [{"markets": [market(str(index)) for index in range(10)]}]}
        provider = PolymarketPublicProvider(request=request)
        results = provider.search("event & detail")
        self.assertEqual(len(results), 5)
        self.assertTrue(calls[0][0].startswith("https://gamma-api.polymarket.com/public-search?"))
        self.assertIn("q=event+%26+detail", calls[0][0])
        self.assertFalse(calls[0][1]["public_only"])
        self.assertNotIn("payload", calls[0][1])
        self.assertNotIn("headers", calls[0][1])
        for bad in ["https://internal/", "../secrets", "123?x=1", "", "1" * 33]:
            with self.subTest(market_id=bad), self.assertRaises(ValueError):
                provider.snapshot(bad)
        self.assertEqual(len(calls), 1)

    def test_trusted_market_transport_rejects_arbitrary_destinations_routes_and_queries(self):
        calls = []
        provider = PolymarketPublicProvider(request=lambda url, **kwargs: calls.append(url))
        for url in ["http://gamma-api.polymarket.com/markets/123", "https://localhost/markets/123",
                    "https://127.0.0.1/markets/123", "https://gamma-api.polymarket.com.evil.example/markets/123",
                    "https://gamma-api.polymarket.com@localhost/markets/123",
                    "https://gamma-api.polymarket.com:443/markets/123", "https://gamma-api.polymarket.com/../admin",
                    "https://gamma-api.polymarket.com/orders", "https://gamma-api.polymarket.com/markets/123?url=http://localhost",
                    "https://gamma-api.polymarket.com/markets/123#fragment", "https://gamma-api.polymarket.com/markets/%31",
                    "https://gamma-api.polymarket.com/public-search?q=x", "https://gamma-api.polymarket.com/public-search?q=x&q=y&limit_per_type=5&search_profiles=false&search_tags=false"]:
            with self.subTest(url=url), self.assertRaises(ProviderError):
                provider._get(url)
        self.assertEqual(calls, [])
        self.assertEqual(provider.public_config()["transport"], "fixed_gamma_https_get_v1")

    def test_fixed_market_transport_never_follows_redirects(self):
        provider = PolymarketPublicProvider()
        with patch("future_prediction_bench.providers._request_bytes", return_value={
                "status": 302, "headers": {"location": "https://localhost/private"}, "body": b"", "url": "https://gamma-api.polymarket.com/markets/123"}) as request:
            with self.assertRaises(ProviderError):
                provider.snapshot("123")
        self.assertEqual(request.call_count, 1)
        self.assertFalse(request.call_args.kwargs["public_only"])

    def test_rejects_wrong_market_identity_bad_prices_and_mismatched_outcomes(self):
        provider = PolymarketPublicProvider(request=lambda *args, **kwargs: market("456"))
        with self.assertRaises(ProviderError):
            provider.snapshot("123")
        for changes in [{"outcomePrices": '["NaN", "0.3"]'}, {"outcomePrices": '[0.5]'},
                        {"outcomePrices": '[true, 0.3]'}, {"outcomes": '["Yes", "No", "Other"]'},
                        {"outcomePrices": '["1.2", "0.3"]'}]:
            with self.subTest(changes=changes), self.assertRaises(ProviderError):
                PolymarketPublicProvider._market(market(**changes))


if __name__ == "__main__":
    unittest.main()
