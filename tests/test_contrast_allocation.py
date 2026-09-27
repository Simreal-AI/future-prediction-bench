from itertools import product
import math
import unittest

from future_prediction_bench.contrast_allocation import ContrastAnchor, allocate_contrast


class ContrastAllocationTests(unittest.TestCase):
    def test_root_matches_exhaustive_budget_search(self):
        probabilities = [0.01, 0.3, 0.5, 0.99]
        anchors = [ContrastAnchor(str(i), p) for i, p in enumerate(probabilities)]
        for budget in (0, 2, 3, 4, 7, 8):
            with self.subTest(budget=budget):
                result = allocate_contrast(anchors, budget=budget, max_per_anchor=4)
                expected = max(sum(0 if n == 0 else 1-p**n-(1-p)**n for p, n in zip(probabilities, counts))
                    for counts in product((0, 2, 3, 4), repeat=4) if sum(counts) == budget)
                self.assertAlmostEqual(result["predicted_contrast_objective"], expected)
                self.assertEqual(sum(result["allocation"].values()), budget)
                self.assertNotIn(1, result["allocation"].values())

    def test_prefix_matches_exhaustive_search(self):
        anchors = [ContrastAnchor("A", 0.9, 1), ContrastAnchor("B", 0.2, 1), ContrastAnchor("C", 0.6, 0)]
        for budget in (0, 1, 3, 5, 8):
            with self.subTest(budget=budget):
                result = allocate_contrast(anchors, budget=budget, max_per_anchor=3, stage="prefix")
                expected = max(sum(1-(a.success_probability if a.factual_reward else 1-a.success_probability)**n
                                   for a, n in zip(anchors, counts))
                               for counts in product(range(4), repeat=3) if sum(counts) == budget)
                self.assertAlmostEqual(result["predicted_contrast_objective"], expected)

    def test_input_order_does_not_change_plan(self):
        anchors = [ContrastAnchor("A", 0.5), ContrastAnchor("B", 0.5)]
        a = allocate_contrast(anchors, budget=4, max_per_anchor=4)
        b = allocate_contrast(reversed(anchors), budget=4, max_per_anchor=4)
        self.assertEqual(a, b)

    def test_fixed_budget_can_skip_deterministic_roots(self):
        result = allocate_contrast([ContrastAnchor("solved", 1), ContrastAnchor("failed", 0),
                                    ContrastAnchor("contrast", 0.5)], budget=8, max_per_anchor=8)
        self.assertEqual(result["allocation"], {"contrast": 8, "failed": 0, "solved": 0})

    def test_terminal_and_unknown_reward_cannot_be_expanded(self):
        for anchor in (ContrastAnchor("A", 0.5, 1, True), ContrastAnchor("A", 0.5),
                       ContrastAnchor("A", 0.5, True), ContrastAnchor("A", 0.5, 0.4)):
            with self.subTest(anchor=anchor), self.assertRaises(ValueError):
                allocate_contrast([anchor], budget=1, max_per_anchor=2, stage="prefix")

    def test_invalid_budget_and_probabilities_rejected(self):
        for p in (True, math.nan, math.inf, -0.1, 1.1):
            with self.subTest(p=p), self.assertRaises(ValueError):
                allocate_contrast([ContrastAnchor("A", p)], budget=2, max_per_anchor=2)
        for budget in (True, 1, 5):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                allocate_contrast([ContrastAnchor("A", 0.5)], budget=budget, max_per_anchor=4)


if __name__ == "__main__":
    unittest.main()
