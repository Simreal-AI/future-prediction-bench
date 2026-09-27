"""Optional checks executing actual pinned upstream functions with real torch.

Set FPB_TREE_GRPO_CHECKOUT to the clean author checkout to enable these.
No trainer/backend mocks are used. Ordinary dependency-free CI skips them.
"""
import os
from pathlib import Path
import unittest

from future_prediction_bench.official_tree_advantages import OfficialTreeAdvantageBridge


class OfficialTreeAdvantagesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        checkout = os.environ.get("FPB_TREE_GRPO_CHECKOUT")
        if not checkout:
            raise unittest.SkipTest("Pinned Tree-GRPO checkout not configured")
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("Real PyTorch not installed")
        cls.bridge = OfficialTreeAdvantageBridge(Path(checkout))
        cls.torch = torch

    def inputs(self):
        torch = self.torch
        rewards = torch.tensor([[0., 0., 0.], [0., 0., 1.], [0., 0., 0.], [0., 0., 0.]])
        responses = torch.ones(4, 3)
        policy = responses.clone()
        policy[:, 0] = 0
        return rewards, responses, policy, ["question"] * 4, ["A", "A", "B", "B"]

    def test_two_modes_have_distinct_credit_and_context_masks(self):
        tree = self.bridge.prepare(*self.inputs(), mode="tree")
        norm = self.bridge.prepare(*self.inputs(), mode="tree_2norm")
        self.assertFalse(tree["trainer_ready"])
        self.assertTrue(self.torch.all(tree["loss_advantages"][:, 0] == 0))
        self.assertLess(tree["advantages"][2, 1].item(), 0)
        self.assertEqual(norm["advantages"][2, 1].item(), 0)
        self.assertLess(norm["advantages"][0, 1].item(), 0)
        self.assertGreater(norm["advantages"][1, 1].item(), 0)

    def test_equal_rewards_have_zero_signal(self):
        values = list(self.inputs())
        values[0][:] = 0
        for mode in ("tree", "tree_2norm"):
            with self.subTest(mode=mode):
                result = self.bridge.prepare(*values, mode=mode)
                self.assertTrue(self.torch.all(result["advantages"] == 0))

    def test_padding_reward_is_rejected(self):
        values = list(self.inputs())
        values[1][1, 2] = 0
        values[2][1, 2] = 0
        with self.assertRaisesRegex(ValueError, "padded"):
            self.bridge.prepare(*values)

    def test_policy_mask_cannot_cover_padding(self):
        values = list(self.inputs())
        values[1][0, 1] = 0
        with self.assertRaisesRegex(ValueError, "subset"):
            self.bridge.prepare(*values)

    def test_no_policy_target_rejected(self):
        values = list(self.inputs())
        values[2][0, :] = 0
        with self.assertRaisesRegex(ValueError, "loss target"):
            self.bridge.prepare(*values)

    def test_nonfinite_reward_rejected(self):
        values = list(self.inputs())
        values[0][0, 1] = float("nan")
        with self.assertRaisesRegex(ValueError, "Finite"):
            self.bridge.prepare(*values)

    def test_tree_cannot_mix_questions(self):
        values = list(self.inputs())
        values[3][1] = "other"
        with self.assertRaisesRegex(ValueError, "unrelated"):
            self.bridge.prepare(*values)

    def test_singleton_trees_rejected(self):
        values = list(self.inputs())
        values[4] = ["A", "B", "C", "D"]
        with self.assertRaisesRegex(ValueError, "two scored"):
            self.bridge.prepare(*values)


if __name__ == "__main__":
    unittest.main()
