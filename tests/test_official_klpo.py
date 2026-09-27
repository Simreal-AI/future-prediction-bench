"""Optional real-PyTorch checks against pinned author KLPO code.

Enable with FPB_KLPO_CHECKOUT. These verify actual loss gradients and record
contracts, not GPU training, LLM trajectories or distributed execution.
"""
import os
import unittest

from examples.official_klpo.check_loss import check_gradients, check_guards, load_upstream


class OfficialKLPOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        checkout = os.environ.get("FPB_KLPO_CHECKOUT")
        if not checkout:
            raise unittest.SkipTest("Pinned KLPO checkout not configured")
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("Real PyTorch not installed")
        cls.torch = torch
        cls.module, cls.adapter_module, cls.provenance = load_upstream(checkout)

    def test_two_routes_recover_exact_auxiliary_expectation_and_native_adapter_gradient(self):
        rows = check_gradients(self.module, self.adapter_module)
        self.assertEqual([row["exhaustively_enumerated_auxiliary_banks"] for row in rows], [4, 16])
        for row in rows:
            self.assertLess(row["maximum_absolute_gradient_difference"], 1e-12)
            self.assertTrue(row["native_loss_adapter_gradient_matches_direct_loss"])
            self.assertTrue(row["native_ignores_ppo_old_logprobs_and_group_advantages"])

    def test_actual_upstream_rejects_missing_and_invalid_records(self):
        outcomes = check_guards(self.module, self.adapter_module)
        self.assertEqual(len(outcomes), 8)
        self.assertTrue(all(outcome["rejected"] for outcome in outcomes.values()))

    def test_context_tokens_and_historical_sampler_receive_no_gradient(self):
        torch = self.torch
        logits = torch.tensor([[[.3, -.2], [.5, .7], [-.5, .4]]], dtype=torch.float64, requires_grad=True)
        current = logits.log_softmax(-1)
        historical = torch.tensor([.2, .8], dtype=logits.dtype).log().expand_as(current).clone().requires_grad_()
        actions = torch.tensor([[0, 1, 1]])
        auxiliary = torch.tensor([[[1], [0], [0]]])
        mask = torch.tensor([[True, False, True]])
        reward = torch.tensor([.5], dtype=logits.dtype, requires_grad=True)
        loss, _ = self.module.klpo_token_loss(
            current.gather(-1, actions[..., None]).squeeze(-1).masked_fill(~mask, float("nan")),
            historical.gather(-1, actions[..., None]).squeeze(-1).masked_fill(~mask, float("inf")),
            reward, mask,
            mc_log_probs=current.gather(-1, auxiliary).masked_fill(~mask[..., None], float("nan")),
            behavior_mc_log_probs=historical.gather(-1, auxiliary).masked_fill(~mask[..., None], float("inf")))
        loss.backward()
        self.assertTrue(torch.all(logits.grad[~mask] == 0))
        self.assertGreater(logits.grad[mask].abs().sum().item(), 0)
        self.assertIsNone(historical.grad)
        self.assertIsNone(reward.grad)

    def test_one_complete_response_has_nonzero_update_without_reward_group(self):
        torch = self.torch
        logits = torch.tensor([[[.4, -.1]]], dtype=torch.float64, requires_grad=True)
        current = logits.log_softmax(-1)
        historical = torch.tensor([[[.25, .75]]], dtype=logits.dtype).log()
        loss, _ = self.module.klpo_token_loss(current[..., 0], historical[..., 0],
            torch.tensor([1.], dtype=logits.dtype), torch.ones(1, 1, dtype=torch.bool),
            mc_log_probs=current[..., 1:2], behavior_mc_log_probs=historical[..., 1:2])
        gradient, = torch.autograd.grad(loss, logits)
        self.assertGreater(gradient.norm().item(), 0)


if __name__ == "__main__":
    unittest.main()
