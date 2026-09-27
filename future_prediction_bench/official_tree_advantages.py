"""Optional direct execution of pinned Tree-GRPO advantage function bodies.

This imports no trainer or distributed backend. The exact reviewed upstream
function AST is executed with real PyTorch and its declared defaultdict
dependency. No upstream source is vendored. These are outcome advantages,
not a complete loss, tokenization bridge, rollout collector, or optimizer.
"""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import subprocess

COMMIT = "19bf3fa0c74d8b9ba70619a09a2fd480c31b9d59"
SOURCE = "verl/trainer/ppo/core_algos.py"
FUNCTIONS = {"compute_grpo_outcome_advantage", "compute_tree_grpo_outcome_advantage"}


class OfficialTreeAdvantageBridge:
    def __init__(self, checkout):
        import torch
        self.torch = torch
        checkout = Path(checkout).resolve()
        head = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(checkout), "status", "--porcelain"], text=True)
        if head != COMMIT or dirty:
            raise ValueError("A clean pinned Tree-GRPO checkout is required")
        path = checkout / SOURCE
        if path.is_symlink():
            raise ValueError("Upstream source cannot be a symlink")
        source = path.read_bytes()
        nodes = [node for node in ast.parse(source, filename=str(path)).body
                 if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS]
        if {node.name for node in nodes} != FUNCTIONS or len(nodes) != 2:
            raise ValueError("Expected upstream advantage functions are missing")
        namespace = {"torch": torch, "defaultdict": defaultdict}
        # Function bodies, defaults and annotations remain unchanged. Other
        # module imports are omitted, rather than replaced with fake modules.
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
        self.functions = {name: namespace[name] for name in FUNCTIONS}
        self.provenance = {"repository": "https://github.com/AMAP-ML/Tree-GRPO",
            "commit": head, "source_path": SOURCE,
            "source_sha256": hashlib.sha256(source).hexdigest(),
            "execution_scope": "two unmodified function ASTs with real PyTorch",
            "dependency_doubles": [], "modified_upstream_function_bodies": False}

    def prepare(self, token_rewards, response_mask, policy_loss_mask,
                question_uids, tree_uids, *, mode="tree"):
        """Execute upstream inter+inner (`tree`) or sequential (`tree_2norm`).

        Caller supplies frozen on-policy tensors and globally distinct tree
        identities. Returns still require behavior logprobs and a supported
        optimizer; no advantage is an authorization to update a model.
        """
        torch = self.torch
        tensors = (token_rewards, response_mask, policy_loss_mask)
        if (any(not isinstance(t, torch.Tensor) or t.ndim != 2 for t in tensors)
                or any(t.shape != token_rewards.shape or t.device != token_rewards.device for t in tensors)
                or any(not torch.isfinite(t).all().item() for t in tensors)):
            raise ValueError("Finite tensors with matching 2D shapes and devices are required")
        if not token_rewards.is_floating_point() or not token_rewards.numel():
            raise ValueError("Nonempty floating point rewards are required")
        for mask in (response_mask, policy_loss_mask):
            if not torch.all((mask == 0) | (mask == 1)).item():
                raise ValueError("Masks must be binary")
        if torch.any(policy_loss_mask > response_mask).item():
            raise ValueError("Policy loss mask must be a subset of the response mask")
        if torch.any(token_rewards[response_mask == 0] != 0).item():
            raise ValueError("Rewards cannot occupy padded response positions")
        if torch.any(policy_loss_mask.sum(dim=-1) == 0).item():
            raise ValueError("Each response needs at least one policy loss target")
        batch = token_rewards.shape[0]
        if (not isinstance(question_uids, (list, tuple)) or not isinstance(tree_uids, (list, tuple))
                or len(question_uids) != batch or len(tree_uids) != batch
                or any(not isinstance(uid, str) or not uid for uid in (*question_uids, *tree_uids))):
            raise ValueError("One nonempty question and tree ID per response is required")
        owners = {}
        for question, tree in zip(question_uids, tree_uids):
            if tree in owners and owners[tree] != question:
                raise ValueError("A tree cannot combine unrelated questions")
            owners[tree] = question
        if any(count < 2 for count in Counter(tree_uids).values()):
            raise ValueError("At least two scored responses per tree are required")
        if mode == "tree":
            function = self.functions["compute_grpo_outcome_advantage"]
            inter, _ = function(token_rewards.clone(), response_mask, question_uids)
            inner, _ = function(token_rewards.clone(), response_mask, tree_uids)
            advantages = inter + inner
        elif mode == "tree_2norm":
            function = self.functions["compute_tree_grpo_outcome_advantage"]
            advantages, _ = function(token_rewards.clone(), response_mask, question_uids, tree_uids)
        else:
            raise ValueError("mode must be tree or tree_2norm")
        if not torch.isfinite(advantages).all().item():
            raise ValueError("Upstream returned nonfinite advantages")
        return {"advantages": advantages, "loss_advantages": advantages * policy_loss_mask,
                "mode": mode, "provenance": dict(self.provenance), "trainer_ready": False}
