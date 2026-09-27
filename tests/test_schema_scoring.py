"""Behavioral checks for future-event chronology and proper-score conventions."""

import copy
import math
import unittest
from datetime import datetime, timezone

from future_prediction_bench.schema import (
    option_ids,
    parse_timestamp,
    validate_probabilities,
    validate_question,
)
from future_prediction_bench.scoring import clipped_log_loss, normalized_brier, score_prediction


def binary_question():
    return {
        "schema_version": "0.1",
        "question_id": "synthetic-weather-001",
        "event_id": "synthetic-weather-event-001",
        "cluster_id": "synthetic-weather",
        "split": "train",
        "kind": "binary",
        "prompt": "Will the synthetic sensor report rain on 2027-01-03?",
        "options": [{"id": "yes", "text": "Rain"}, {"id": "no", "text": "No rain"}],
        "issued_at": "2027-01-01T00:00:00Z",
        "forecast_deadline": "2027-01-02T00:00:00Z",
        "outcome_not_before": "2027-01-03T00:00:00Z",
        "resolve_after": "2027-01-04T00:00:00Z",
        "resolution": {
            "criteria": "Use the synthetic sensor's daily rain flag; missing data voids the event.",
            "source_urls": ["https://example.org/synthetic/sensor"],
            "on_ambiguous": "void",
        },
        "domain": "synthetic_weather",
        "is_fixture": True,
    }


class QuestionValidationTests(unittest.TestCase):
    def test_valid_binary_and_categorical_questions_are_not_mutated(self):
        binary = binary_question()
        original = copy.deepcopy(binary)
        self.assertIsNone(validate_question(binary))
        self.assertEqual(binary, original)
        self.assertEqual(option_ids(binary), ["yes", "no"])
        categorical = binary_question()
        categorical["kind"] = "categorical"
        categorical["options"] = [
            {"id": "low", "text": "Below 10"},
            {"id": "medium", "text": "At least 10 and below 20"},
            {"id": "high", "text": "At least 20"},
        ]
        validate_question(categorical)

    def test_every_required_field_is_required(self):
        for field in binary_question().keys() - {"is_fixture"}:
            with self.subTest(field=field):
                question = binary_question()
                del question[field]
                with self.assertRaises(ValueError):
                    validate_question(question)

    def test_blank_fields_duplicate_ids_and_wrong_option_counts_fail(self):
        mutations = [
            {"question_id": "  "}, {"event_id": ""}, {"cluster_id": None},
            {"prompt": "\n"}, {"domain": False}, {"kind": "multiple"},
            {"split": "validation"}, {"schema_version": 0.1},
            {"kind": "categorical"}, {"options": []},
            {"options": [{"id": "yes", "text": "Y"}, {"id": "yes", "text": "N"}]},
            {"options": [{"id": "yes", "text": "Y"}, {"id": "no", "text": " "}]},
            {"metadata": []}, {"is_fixture": 1},
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                question = binary_question()
                question.update(mutation)
                with self.assertRaises(ValueError):
                    validate_question(question)
        question = binary_question()
        question["options"].append({"id": "other", "text": "Other"})
        with self.assertRaises(ValueError):
            validate_question(question)

    def test_resolution_requires_criteria_sources_and_void_policy(self):
        for field, value in [
            ("criteria", " "), ("source_urls", []), ("source_urls", "https://example.org"),
            ("source_urls", ["file:///tmp/outcome"]), ("source_urls", ["https:///outcome"]),
            ("source_urls", ["https://example.org:invalid/outcome"]),
            ("source_urls", ["https://example.org/has space"]), ("on_ambiguous", "guess"),
        ]:
            with self.subTest(field=field, value=value):
                question = binary_question()
                question["resolution"][field] = value
                with self.assertRaises(ValueError):
                    validate_question(question)

    def test_timestamp_is_explicit_and_converted_to_utc(self):
        expected = datetime(2027, 1, 1, tzinfo=timezone.utc)
        self.assertEqual(parse_timestamp("2027-01-01T08:00:00+08:00"), expected)
        self.assertEqual(parse_timestamp("2027-01-01T00:00:00Z"), expected)
        for timestamp in ["2027-01-01T00:00:00", "2027-01-01", "", None, 123, "invalid"]:
            with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                parse_timestamp(timestamp)

    def test_forecasts_must_precede_earliest_possible_outcome(self):
        for field, value in [
            ("issued_at", "2027-01-02T00:00:00Z"),
            ("issued_at", "2027-01-05T00:00:00Z"),
            ("forecast_deadline", "2027-01-03T00:00:00Z"),
            ("forecast_deadline", "2027-01-04T00:00:00Z"),
            ("resolve_after", "2027-01-02T00:00:00Z"),
            # These textual dates look ordered, but their actual UTC instants are not.
            ("forecast_deadline", "2027-01-01T01:00:00+08:00"),
        ]:
            with self.subTest(field=field, value=value):
                question = binary_question()
                question[field] = value
                with self.assertRaises(ValueError):
                    validate_question(question)
        question = binary_question()
        question["resolve_after"] = question["outcome_not_before"]
        validate_question(question)


class ProbabilityValidationTests(unittest.TestCase):
    def test_accepts_complete_distribution_and_preserves_values(self):
        original = {"yes": 0.6, "no": 0.4000005}
        validated = validate_probabilities(original, iter(["yes", "no"]))
        self.assertEqual(validated, original)
        self.assertIsNot(validated, original)
        self.assertNotEqual(sum(validated.values()), 1.0)
        self.assertEqual(validate_probabilities({"yes": 1, "no": 0}, ["yes", "no"]),
                         {"yes": 1.0, "no": 0.0})

    def test_rejects_incomplete_extra_or_invalid_probabilities(self):
        invalid = [
            {}, {"yes": 1}, {"yes": 0.5, "no": 0.5, "other": 0},
            {"yes": 0.6, "no": 0.5}, {"yes": -0.1, "no": 1.1},
            {"yes": True, "no": False}, {"yes": "0.5", "no": 0.5},
            {"yes": float("nan"), "no": 0.5}, {"yes": float("inf"), "no": 0.5},
            {"yes": float("-inf"), "no": 0.5}, {"yes": 0.6, "no": 0.400002},
            {"yes": 10 ** 1000, "no": 0}, [0.5, 0.5],
        ]
        for probabilities in invalid:
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                validate_probabilities(probabilities, ["yes", "no"])

    def test_rejects_malformed_option_id_iterables(self):
        for ids in [[], ["yes"], ["yes", "yes"], ["yes", " "], "yes", None, ["yes", 1]]:
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                validate_probabilities({"yes": 0.5, "no": 0.5}, ids)


class ScoringTests(unittest.TestCase):
    def test_binary_brier_equals_squared_error_for_both_outcomes(self):
        for p_yes in [0.0, 0.01, 0.2, 0.5, 0.87, 1.0]:
            for outcome, y in [("yes", 1), ("no", 0)]:
                with self.subTest(p_yes=p_yes, outcome=outcome):
                    actual = normalized_brier({"yes": p_yes, "no": 1 - p_yes}, outcome)
                    self.assertAlmostEqual(actual, (p_yes - y) ** 2)

    def test_multiclass_score_reward_and_uniform_baseline(self):
        result = score_prediction({"a": 0.2, "b": 0.3, "c": 0.5}, "c")
        self.assertAlmostEqual(result["brier"], 0.19)
        self.assertAlmostEqual(result["clipped_log_loss"], math.log(2))
        self.assertEqual(result["log_loss_epsilon"], 1e-15)
        self.assertAlmostEqual(result["reward"], -0.19)
        self.assertAlmostEqual(result["uniform_brier"], 1 / 3)
        self.assertAlmostEqual(result["brier_skill_vs_uniform"], 0.43)

    def test_bounds_perfect_and_certainly_wrong_predictions(self):
        for n_options in [2, 3, 10]:
            ids = [str(index) for index in range(n_options)]
            certain = {identifier: float(identifier == "0") for identifier in ids}
            self.assertEqual(normalized_brier(certain, "0"), 0.0)
            self.assertEqual(normalized_brier(certain, "1"), 1.0)
            self.assertEqual(score_prediction(certain, "0")["brier_skill_vs_uniform"], 1.0)
            uniform = {identifier: 1 / n_options for identifier in ids}
            self.assertAlmostEqual(score_prediction(uniform, "0")["brier_skill_vs_uniform"], 0.0)
            for outcome in ids:
                self.assertLessEqual(normalized_brier(uniform, outcome), 1.0)
                self.assertGreaterEqual(normalized_brier(uniform, outcome), 0.0)

    def test_expected_brier_is_minimized_by_truthful_distribution(self):
        truth = {"a": 0.2, "b": 0.3, "c": 0.5}
        expected_truthful = sum(weight * normalized_brier(truth, outcome)
                                for outcome, weight in truth.items())
        for candidate in [{"a": 0.1, "b": 0.2, "c": 0.7},
                          {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3},
                          {"a": 0.0, "b": 0.0, "c": 1.0}]:
            expected_candidate = sum(weight * normalized_brier(candidate, outcome)
                                     for outcome, weight in truth.items())
            self.assertGreater(expected_candidate, expected_truthful)
            self.assertAlmostEqual(expected_candidate - expected_truthful,
                                   0.5 * sum((candidate[key] - truth[key]) ** 2 for key in truth))

    def test_log_loss_clips_zero_and_validates_epsilon(self):
        certain = {"yes": 1.0, "no": 0.0}
        self.assertEqual(clipped_log_loss(certain, "yes"), 0.0)
        self.assertAlmostEqual(clipped_log_loss(certain, "no"), -math.log(1e-15))
        self.assertAlmostEqual(clipped_log_loss(certain, "no", epsilon=0.01), math.log(100))
        for epsilon in [0, -1, 1, 2, True, "0.1", float("nan"), float("inf"), 10 ** 1000]:
            with self.subTest(epsilon=epsilon), self.assertRaises(ValueError):
                clipped_log_loss(certain, "yes", epsilon=epsilon)

    def test_all_scorers_reject_unknown_outcomes_and_invalid_distributions(self):
        for scorer in [normalized_brier, clipped_log_loss, score_prediction]:
            for probabilities, outcome in [
                ({"yes": 0.5, "no": 0.5}, "unknown"),
                ({"yes": 0.5, "no": 0.5}, None),
                ({"yes": True, "no": False}, "yes"),
                ({"yes": 0.9, "no": 0.9}, "yes"), ({"yes": 1}, "yes"),
                ([], "yes"),
            ]:
                with self.subTest(scorer=scorer.__name__, probabilities=probabilities,
                                  outcome=outcome), self.assertRaises(ValueError):
                    scorer(probabilities, outcome)


if __name__ == "__main__":
    unittest.main()
