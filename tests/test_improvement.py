"""Offline behavioral checks for candidate identity, chronology, and comparison."""

import copy
import unittest

from future_prediction_bench.improvement import (
    artifact_sha256, create_candidate_manifest, create_evaluation_contract,
    evaluate_promotion, seal_dev_outcomes, validate_candidate_manifest,
)


def comparison_fixture(count=30, candidate_brier=0.1):
    questions = [{
        "question_id": f"q-{index}", "event_id": f"event-{index}",
        "cluster_id": f"cluster-{index}",
        "forecast_deadline": "2027-02-02T00:00:00Z",
        "outcome_not_before": "2027-02-03T00:00:00Z",
        "resolve_after": "2027-02-04T00:00:00Z",
    } for index in range(count)]
    contract = create_evaluation_contract(
        frozen_at="2027-01-01T00:00:00Z", dev_start="2027-02-01T00:00:00Z",
        dev_end="2027-03-01T00:00:00Z", questions=questions,
    )
    incumbent = create_candidate_manifest(
        artifacts={"prompt": "incumbent", "config": '{"model":"offline"}'},
        created_at="2027-01-02T00:00:00Z", training_data_cutoff="2026-12-31T00:00:00Z",
        contract=contract,
    )
    candidate = create_candidate_manifest(
        artifacts={"prompt": "candidate", "config": '{"model":"offline"}'},
        created_at="2027-01-03T00:00:00Z", training_data_cutoff="2026-12-31T00:00:00Z",
        contract=contract, parent=incumbent,
    )
    rows = [{
        "question_id": f"q-{index}", "incumbent_submitted_at": "2027-02-01T12:00:00Z",
        "candidate_submitted_at": "2027-02-01T12:00:00Z",
        "resolved_at": "2027-02-04T00:00:00Z",
        "incumbent_brier": 0.3, "candidate_brier": candidate_brier,
    } for index in range(count)]
    return contract, incumbent, candidate, rows


def seal(contract, incumbent, candidate, rows, **kwargs):
    return seal_dev_outcomes(contract=contract, incumbent=incumbent, candidate=candidate,
                             rows=rows, sealed_at="2027-03-01T00:00:00Z", **kwargs)


def evaluate(contract, incumbent, candidate, sealed):
    return evaluate_promotion(contract=contract, incumbent=incumbent, candidate=candidate,
                              sealed_outcomes=sealed, expected_contract_sha256=contract["sha256"])


class ImprovementTests(unittest.TestCase):
    def test_artifacts_are_hashed_by_content_and_manifest_has_parent_lineage(self):
        contract, incumbent, candidate, _ = comparison_fixture()
        self.assertEqual(artifact_sha256("abc"), artifact_sha256(b"abc"))
        self.assertNotEqual(artifact_sha256("abc"), artifact_sha256("abcd"))
        self.assertEqual(candidate["version"], 2)
        self.assertEqual(candidate["parent_sha256"], incumbent["sha256"])
        self.assertEqual(candidate["contract_sha256"], contract["sha256"])
        self.assertNotIn("prompt", candidate)
        self.assertNotEqual(candidate["sha256"], incumbent["sha256"])
        tampered = copy.deepcopy(candidate)
        tampered["artifact_sha256"]["prompt"] = artifact_sha256("changed")
        with self.assertRaisesRegex(ValueError, "digest"):
            validate_candidate_manifest(tampered)

    def test_manifest_rejects_future_training_and_post_dev_creation(self):
        contract, incumbent, _, _ = comparison_fixture()
        for created_at, cutoff in [
            ("2027-01-03T00:00:00Z", "2027-01-04T00:00:00Z"),
            ("2027-02-01T00:00:00Z", "2026-12-31T00:00:00Z"),
            ("2027-01-01T00:00:00Z", "2026-12-31T00:00:00Z"),
            ("2027-01-03T00:00:00Z", "2026-12-30T00:00:00Z"),
        ]:
            with self.subTest(created_at=created_at, cutoff=cutoff), self.assertRaises(ValueError):
                create_candidate_manifest(artifacts={"prompt": "x"}, created_at=created_at,
                                          training_data_cutoff=cutoff, contract=contract, parent=incumbent)

    def test_clear_improvement_is_eligible_and_small_cohort_is_not(self):
        for count, eligible in [(30, True), (29, False), (1, False)]:
            args = comparison_fixture(count)
            original = copy.deepcopy(args)
            result = evaluate(*args[:3], seal(*args))
            self.assertEqual(args, original)
            self.assertEqual(result["eligible"], eligible)
            self.assertAlmostEqual(result["mean_brier_improvement"], 0.2)
            self.assertEqual(result["cluster_count"], count)
            if not eligible:
                self.assertIsNone(result["confidence_interval"])
                self.assertEqual(result["reason"], "insufficient_clusters")

    def test_regression_tie_and_mixed_uncertainty_are_not_promoted(self):
        for brier in [0.3, 0.4]:
            args = comparison_fixture(candidate_brier=brier)
            self.assertFalse(evaluate(*args[:3], seal(*args))["eligible"])
        contract, incumbent, candidate, rows = comparison_fixture()
        for index, row in enumerate(rows):
            row["candidate_brier"] = 0.1 if index % 2 else 0.5
        result = evaluate(contract, incumbent, candidate, seal(contract, incumbent, candidate, rows))
        self.assertFalse(result["eligible"])
        self.assertLess(result["confidence_interval"][0], 0)
        self.assertGreater(result["confidence_interval"][1], 0)

    def test_rows_are_order_independent_and_bootstrap_is_deterministic(self):
        contract, incumbent, candidate, rows = comparison_fixture()
        for index, row in enumerate(rows):
            row["candidate_brier"] = 0.05 + index / 100
        first = evaluate(contract, incumbent, candidate, seal(contract, incumbent, candidate, rows))
        second = evaluate(contract, incumbent, candidate,
                          seal(contract, incumbent, candidate, list(reversed(rows))))
        self.assertEqual(first["mean_brier_improvement"], second["mean_brier_improvement"])
        self.assertEqual(first["confidence_interval"], second["confidence_interval"])

    def test_incomplete_duplicate_unknown_and_private_rows_fail(self):
        contract, incumbent, candidate, rows = comparison_fixture()
        private = copy.deepcopy(rows)
        private[0]["private_baseline"] = 0.25
        unknown = copy.deepcopy(rows)
        unknown[0]["question_id"] = "test-question"
        for bad in [rows[:-1], rows + [rows[0]], private, unknown]:
            with self.subTest(bad=bad[0]), self.assertRaises(ValueError):
                seal(contract, incumbent, candidate, bad)

    def test_bad_scores_and_late_or_premature_timestamps_fail(self):
        contract, incumbent, candidate, rows = comparison_fixture()
        for field, value in [
            ("candidate_brier", float("nan")), ("candidate_brier", float("inf")),
            ("candidate_brier", True), ("candidate_brier", -0.1), ("candidate_brier", 1.1),
            ("candidate_brier", 10**1000),
            ("candidate_submitted_at", "2027-02-02T00:00:00Z"),
            ("candidate_submitted_at", "2027-02-02T00:00:01Z"),
            ("candidate_submitted_at", "2027-01-31T23:59:59Z"),
            ("incumbent_submitted_at", "2027-01-01T00:00:00Z"),
            ("resolved_at", "2027-02-03T00:00:00Z"),
            ("resolved_at", "2027-04-01T00:00:00Z"),
        ]:
            bad = copy.deepcopy(rows)
            bad[0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                seal(contract, incumbent, candidate, bad)
        with self.assertRaises(ValueError):
            seal_dev_outcomes(contract=contract, incumbent=incumbent, candidate=candidate,
                              rows=rows, sealed_at="2027-02-28T00:00:00Z")
        on_start = copy.deepcopy(rows)
        on_start[0]["candidate_submitted_at"] = contract["dev_start"]
        self.assertIsInstance(seal(contract, incumbent, candidate, on_start), dict)

    def test_contract_and_outcome_integrity_and_comparison_budget(self):
        contract, incumbent, candidate, rows = comparison_fixture()
        sealed = seal(contract, incumbent, candidate, rows)
        modified = copy.deepcopy(sealed)
        modified["split"] = "test"
        with self.assertRaises(ValueError):
            evaluate(contract, incumbent, candidate, modified)
        with self.assertRaises(ValueError):
            evaluate_promotion(contract=contract, incumbent=incumbent, candidate=candidate,
                               sealed_outcomes=sealed, expected_contract_sha256=artifact_sha256("other"))
        with self.assertRaises(ValueError):
            seal(contract, incumbent, candidate, rows, comparison_index=2)
        with self.assertRaises(ValueError):
            seal(contract, candidate, incumbent, rows)

    def test_contract_rejects_posthoc_cohorts_cluster_splitting_and_weak_guards(self):
        contract, _, _, _ = comparison_fixture()
        arguments = {key: copy.deepcopy(contract[key]) for key in (
            "frozen_at", "dev_start", "dev_end", "questions", "min_clusters",
            "min_improvement", "confidence", "max_comparisons", "bootstrap_samples", "seed",
        )}
        for patch in [
            {"frozen_at": "2027-02-01T00:00:00Z"}, {"min_clusters": 19},
            {"min_clusters": True}, {"min_improvement": -0.01}, {"confidence": 0.5},
            {"seed": -1}, {"max_comparisons": 0},
            {"confidence": 0.999, "max_comparisons": 20, "bootstrap_samples": 2000},
        ]:
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                create_evaluation_contract(**(arguments | patch))
        for field, value in [
            ("event_id", "event-1"),  # One event would be split across independent clusters.
            ("forecast_deadline", "2027-01-31T00:00:00Z"),
            ("forecast_deadline", "2027-02-01T00:00:00Z"),
            ("outcome_not_before", "2027-02-02T00:00:00Z"),
            ("resolve_after", "2027-03-02T00:00:00Z"),
            ("private_baseline", 0.25),
        ]:
            bad_questions = copy.deepcopy(arguments["questions"])
            bad_questions[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                create_evaluation_contract(**(arguments | {"questions": bad_questions}))

    def test_event_and_cluster_averaging_prevents_replication_weighting(self):
        _, _, _, rows = comparison_fixture(3)
        questions = [{
            "question_id": f"q-{i}", "event_id": "event-a" if i < 2 else "event-b",
            "cluster_id": "one-cluster", "forecast_deadline": "2027-02-02T00:00:00Z",
            "outcome_not_before": "2027-02-03T00:00:00Z", "resolve_after": "2027-02-04T00:00:00Z",
        } for i in range(3)]
        contract = create_evaluation_contract(frozen_at="2027-01-01T00:00:00Z",
                                              dev_start="2027-02-01T00:00:00Z",
                                              dev_end="2027-03-01T00:00:00Z", questions=questions)
        incumbent = create_candidate_manifest(artifacts={"prompt": "a"},
            created_at="2027-01-02T00:00:00Z", training_data_cutoff="2026-12-31T00:00:00Z", contract=contract)
        candidate = create_candidate_manifest(artifacts={"prompt": "b"},
            created_at="2027-01-03T00:00:00Z", training_data_cutoff="2026-12-31T00:00:00Z",
            contract=contract, parent=incumbent)
        rows[2]["candidate_brier"] = 0.5
        result = evaluate(contract, incumbent, candidate, seal(contract, incumbent, candidate, rows))
        self.assertAlmostEqual(result["mean_brier_improvement"], 0.0)
        self.assertEqual(result["event_count"], 2)
        self.assertEqual(result["cluster_count"], 1)
        self.assertFalse(result["eligible"])


if __name__ == "__main__":
    unittest.main()
