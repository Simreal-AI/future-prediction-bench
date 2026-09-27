"""Real CPU PyTorch check of pinned upstream Tree-GRPO advantage code."""
import argparse
import json
from pathlib import Path

from future_prediction_bench.official_tree_advantages import OfficialTreeAdvantageBridge


def check(checkout):
    bridge = OfficialTreeAdvantageBridge(checkout)
    torch = bridge.torch
    rewards = torch.tensor([[0., 0., 0., value] for value in (0., 1., 0., 0.)])
    response_mask = torch.tensor([[1., 1., 1., 1.], [1., 1., 1., 1.],
                                  [1., 1., 1., 0.], [1., 1., 1., 0.]])
    # Two prefix context/observation tokens are not policy loss targets.
    policy_mask = response_mask.clone()
    policy_mask[:, :2] = 0
    questions = ["q1"] * 4
    trees = ["tree-A", "tree-A", "tree-B", "tree-B"]
    result = {"schema_version": "official-tree-advantages-cpu-v1", "upstream": bridge.provenance,
              "torch_version": str(torch.__version__), "device": "cpu",
              "synthetic_scored_tensors": True, "model_rollout": False,
              "optimizer": False, "cases": []}
    for mode in ("tree", "tree_2norm"):
        prepared = bridge.prepare(rewards, response_mask, policy_mask, questions, trees, mode=mode)
        advantages, loss = prepared["advantages"], prepared["loss_advantages"]
        if torch.any(loss[:, :2] != 0).item() or torch.any(advantages[response_mask == 0] != 0).item():
            raise RuntimeError("context_or_padding_loss_leak")
        if not loss[0, 2] < 0 < loss[1, 2]:
            raise RuntimeError("scored_sibling_order_not_preserved")
        result["cases"].append({"mode": mode, "advantages": advantages.tolist(),
            "policy_loss_advantages": loss.tolist(), "context_and_padding_masked": True})
    # Equal-reward trees carry no within-tree contrast under sequential norm.
    if result["cases"][1]["advantages"][2] != [0.] * 4:
        raise RuntimeError("equal_reward_tree_has_spurious_local_contrast")
    for bad_questions, bad_trees in ((["q1", "q2", "q1", "q1"], trees),
                                     (questions, ["A", "B", "C", "D"])):
        try:
            bridge.prepare(rewards, response_mask, policy_mask, bad_questions, bad_trees)
        except ValueError:
            pass
        else:
            raise RuntimeError("invalid_cohort_accepted")
    result["invalid_cohorts_rejected"] = 2
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists():
        raise ValueError("New output required")
    result = check(args.checkout)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
