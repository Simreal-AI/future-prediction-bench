"""Source-boundary tests for preregistered timing and conservative settlement."""

import copy
import unittest
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlparse

from future_prediction_bench.schema import parse_timestamp, validate_question
from future_prediction_bench.sources import MLBSource, USGSSource, build_sources


NOW = datetime(2026, 9, 22, 0, 0, tzinfo=timezone.utc)


class FakeClient:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.urls = []
        self.last_snapshot = {}

    def get_json(self, url):
        self.urls.append(url)
        self.last_snapshot = {
            "url": url, "observed_at": NOW.isoformat(),
            "sha256": "a" * 64, "path": "synthetic-snapshot.json",
        }
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return copy.deepcopy(response)


def game(**updates):
    value = {
        "gamePk": 123, "gameDate": "2026-09-23T12:00:00Z", "ifNecessary": "N",
        "status": {"abstractGameState": "Preview", "detailedState": "Scheduled", "startTimeTBD": False},
        "teams": {"home": {"team": {"id": 1, "name": "Synthetic Home"}},
                  "away": {"team": {"id": 2, "name": "Synthetic Away"}}},
    }
    value.update(updates)
    return value


def schedule(*games):
    return {"totalGames": len(games), "dates": [{"date": "2026-09-23", "games": list(games)}]}


def final_game(home=2, away=1, **updates):
    value = game(status={"abstractGameState": "Final", "detailedState": "Final"})
    value["teams"]["home"]["score"] = home
    value["teams"]["away"]["score"] = away
    value.update(updates)
    return value


class MLBSourceTests(unittest.TestCase):
    def setUp(self):
        self.source = MLBSource()
        self.question = self.source.discover(NOW, {}, FakeClient(schedule(game())))[0]
        self.settlement_time = parse_timestamp(self.question["resolve_after"])

    def test_discovery_preserves_future_chronology_and_stable_identity(self):
        validate_question(self.question)
        other = self.source.discover(NOW + timedelta(hours=1), {"split": "train"}, FakeClient(schedule(game())))[0]
        for key in ("question_id", "event_id", "cluster_id"):
            self.assertEqual(self.question[key], other[key])
        self.assertEqual(self.question["split"], "test")
        self.assertEqual(self.question["forecast_deadline"], "2026-09-23T11:00:00Z")
        self.assertEqual(self.question["outcome_not_before"], "2026-09-23T12:00:00Z")
        self.assertEqual(self.question["metadata"]["target_at"], "2026-09-23T18:00:00Z")
        self.assertFalse(self.question["is_fixture"])
        self.assertEqual(self.question["metadata"]["discovery_provenance"]["sha256"], "a" * 64)

    def test_skips_known_started_unknown_time_conditional_resumed_and_too_late_games(self):
        excluded = [
            game(gameDate="2026-09-22T01:00:00Z"),
            game(gameDate="2026-09-29T00:00:00Z"),
            game(status={"abstractGameState": "Live", "detailedState": "In Progress", "startTimeTBD": False}),
            game(status={"abstractGameState": "Preview", "detailedState": "Postponed", "startTimeTBD": False}),
            game(status={"abstractGameState": "Preview", "detailedState": "Scheduled", "startTimeTBD": True}),
            game(ifNecessary="Y"), game(resumeDate="2026-09-24"), game(gamePk=True),
        ]
        self.assertEqual(self.source.discover(NOW, {}, FakeClient(schedule(*excluded))), [])

    def test_does_not_fetch_results_before_settlement_window(self):
        client = FakeClient()
        result = self.source.resolve(self.question, self.settlement_time - timedelta(seconds=1), client)
        self.assertEqual(result["status"], "pending")
        self.assertEqual(client.urls, [])

    def test_final_winners_map_to_both_binary_options(self):
        for home, away, outcome in [(6, 3, "yes"), (1, 9, "no")]:
            with self.subTest(home=home, away=away):
                result = self.source.resolve(self.question, self.settlement_time, FakeClient(schedule(final_game(home, away))))
                self.assertEqual((result["status"], result["outcome"]), ("resolved", outcome))
                self.assertEqual(len(result["source_snapshots"]), 1)

    def test_final_ties_missing_scores_wrong_teams_and_late_first_observation_need_review(self):
        changed = final_game()
        changed["teams"]["home"]["team"]["id"] = 99
        for value in (final_game(1, 1), final_game(None, 1), final_game(True, 0), changed,
                      final_game(resumedFrom="2026-09-23T12:00:00Z")):
            result = self.source.resolve(self.question, self.settlement_time, FakeClient(schedule(value)))
            self.assertEqual(result["status"], "needs_review")
            self.assertIsNone(result["outcome"])
        result = self.source.resolve(self.question, self.settlement_time + timedelta(days=1), FakeClient(schedule(final_game())))
        self.assertEqual(result["status"], "needs_review")

    def test_postponement_cancellation_and_unsafe_reschedule_void(self):
        for value in [
            game(status={"detailedState": "Postponed"}),
            game(status={"detailedState": "Cancelled"}),
            final_game(gameDate="2026-09-24T12:00:00Z"),
            final_game(gameDate=self.question["forecast_deadline"]),
        ]:
            self.assertEqual(self.source.resolve(self.question, self.settlement_time, FakeClient(schedule(value)))["status"], "void")

    def test_live_game_waits_then_voids_without_inventing_loser(self):
        live = game(status={"abstractGameState": "Live", "detailedState": "In Progress"})
        pending = self.source.resolve(self.question, self.settlement_time, FakeClient(schedule(live)))
        self.assertEqual(pending["status"], "pending")
        void = self.source.resolve(self.question, parse_timestamp(self.question["metadata"]["resolution_parameters"]["void_after"]), FakeClient(schedule(live)))
        self.assertEqual(void["status"], "void")
        self.assertIsNone(void["outcome"])

    def test_missing_game_requires_review_and_incomplete_payload_raises(self):
        result = self.source.resolve(self.question, self.settlement_time, FakeClient({"dates": [], "totalGames": 0}))
        self.assertEqual(result["status"], "needs_review")
        for value in ({}, {"dates": [], "totalGames": 1}, {"dates": [{"games": None}], "totalGames": 0}):
            with self.assertRaises(ValueError):
                self.source.discover(NOW, {}, FakeClient(value))

    def test_network_error_propagates_for_pipeline_retry(self):
        with self.assertRaises(TimeoutError):
            self.source.resolve(self.question, self.settlement_time, FakeClient(TimeoutError("source timeout")))


class USGSSourceTests(unittest.TestCase):
    def setUp(self):
        self.source = USGSSource()
        self.capability = {"eventtypes": ["earthquake", "explosion"]}
        self.questions = self.source.discover(NOW, {}, FakeClient(self.capability))
        self.question = self.questions[0]
        self.settlement_time = parse_timestamp(self.question["resolve_after"])

    def test_discovery_preregisters_five_future_days_and_immutable_snapshot_rule(self):
        self.assertEqual(len(self.questions), 5)
        for question in self.questions:
            validate_question(question)
            self.assertEqual([option["id"] for option in question["options"]], ["zero", "one", "two_or_more"])
            self.assertLessEqual(parse_timestamp(question["metadata"]["target_at"]), NOW + timedelta(days=7))
            self.assertIn("reviewstatus=all", question["resolution"]["criteria"])
        self.assertEqual(self.question["forecast_deadline"], "2026-09-22T23:00:00Z")
        self.assertEqual(self.question["outcome_not_before"], "2026-09-23T00:00:00Z")
        self.assertEqual(self.question["resolve_after"], "2026-09-25T00:00:00Z")

    def test_question_ids_are_stable_across_split_time_but_cluster_threshold_variants(self):
        repeated = self.source.discover(NOW + timedelta(hours=1), {"split": "train"}, FakeClient(self.capability))[0]
        different_threshold = self.source.discover(NOW, {"min_magnitude": 6}, FakeClient(self.capability))[0]
        self.assertEqual(self.question["question_id"], repeated["question_id"])
        self.assertNotEqual(self.question["question_id"], different_threshold["question_id"])
        self.assertEqual(self.question["cluster_id"], different_threshold["cluster_id"])

    def test_inclusive_api_endtime_excludes_next_midnight(self):
        query = parse_qs(urlparse(self.question["resolution"]["source_urls"][0]).query)
        self.assertEqual(query["starttime"], ["2026-09-23T00:00:00Z"])
        self.assertEqual(query["endtime"], ["2026-09-23T23:59:59.999000Z"])
        self.assertEqual(query["eventtype"], ["earthquake"])
        self.assertNotIn("reviewstatus", query)

    def test_does_not_fetch_early_even_if_positive_bin_could_already_be_known(self):
        for now in [NOW, parse_timestamp(self.question["outcome_not_before"]) + timedelta(hours=2), self.settlement_time - timedelta(seconds=1)]:
            client = FakeClient()
            result = self.source.resolve(self.question, now, client)
            self.assertEqual(result["status"], "pending")
            self.assertEqual(client.urls, [])

    def test_valid_zero_positive_and_large_aggregate_counts(self):
        for count, expected in [(0, "zero"), (1, "one"), (2, "two_or_more"), (25000, "two_or_more")]:
            with self.subTest(count=count):
                result = self.source.resolve(self.question, self.settlement_time, FakeClient({"count": count, "maxAllowed": 20000}))
                self.assertEqual((result["status"], result["outcome"]), ("resolved", expected))
                self.assertIn(f"count={count}", result["evidence_text"])
                self.assertEqual(len(result["source_snapshots"]), 1)

    def test_malformed_counts_never_resolve_as_zero(self):
        for payload in [{}, [], {"count": 0}, {"count": True, "maxAllowed": 20000},
                        {"count": -1, "maxAllowed": 20000}, {"count": "0", "maxAllowed": 20000},
                        {"count": 0, "maxAllowed": 0}, {"features": [], "metadata": {"count": 0}}]:
            with self.subTest(payload=payload):
                result = self.source.resolve(self.question, self.settlement_time, FakeClient(payload))
                self.assertEqual(result["status"], "needs_review")
                self.assertIsNone(result["outcome"])

    def test_timeout_propagates_instead_of_becoming_zero(self):
        with self.assertRaises(TimeoutError):
            self.source.resolve(self.question, self.settlement_time, FakeClient(TimeoutError("upstream unavailable")))

    def test_invalid_configuration_and_missing_capability_rejected(self):
        for config in [{"region": "unknown"}, {"min_magnitude": float("nan")}, {"lookahead_days": 2.5},
                       {"settle_hours": 0}, {"horizon_hours": 169}, {"min_lead_minutes": True}]:
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.source.discover(NOW, config, FakeClient(self.capability))
        with self.assertRaises(ValueError):
            self.source.discover(NOW, {}, FakeClient({"eventtypes": []}))
        with self.assertRaises(ValueError):
            self.source.discover(NOW.replace(tzinfo=None), {}, FakeClient(self.capability))


class SourceRegistryTests(unittest.TestCase):
    def test_enabled_sources_return_independent_configuration(self):
        config = {"sources": [{"id": "mlb", "enabled": True}, {"id": "usgs", "enabled": False}]}
        sources = build_sources(config)
        self.assertEqual(len(sources), 1)
        self.assertIsInstance(sources[0][0], MLBSource)
        sources[0][1]["split"] = "train"
        self.assertNotIn("split", config["sources"][0])

    def test_rejects_malformed_duplicate_and_unknown_sources(self):
        for config in [{}, {"sources": {}}, {"sources": ["mlb"]}, {"sources": [{"id": "unknown"}]},
                       {"sources": [{"id": "mlb"}, {"id": "mlb"}]},
                       {"sources": [{"id": "mlb", "enabled": "yes"}]},
                       {"sources": [{"id": "mlb", "split": "held-out"}]}]:
            with self.subTest(config=config), self.assertRaises(ValueError):
                build_sources(config)


if __name__ == "__main__":
    unittest.main()
