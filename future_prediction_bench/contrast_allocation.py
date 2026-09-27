"""Bounded exact allocation from TRACE root/prefix contrast objectives.

Independent implementation of equations 12--16, not official TRACE code.
Predicted success probabilities are inputs, not a learned predictor. Budgets
count continuations, not tokens, GPU time, or a measured learning gain.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ContrastAnchor:
    anchor_id: str
    success_probability: float
    factual_reward: int | None = None
    terminal: bool = False


def allocate_contrast(anchors, *, budget, max_per_anchor, stage="root"):
    if stage not in {"root", "prefix"}:
        raise ValueError("stage must be root or prefix")
    if (type(budget) is not int or not 0 <= budget <= 4096 or
            type(max_per_anchor) is not int or not 1 <= max_per_anchor <= 64):
        raise ValueError("bounded integer budget and per-anchor cap required")
    anchors = tuple(anchors)
    if not 1 <= len(anchors) <= 512 or any(not isinstance(a, ContrastAnchor) for a in anchors):
        raise ValueError("one to 512 typed anchors required")
    if len({a.anchor_id for a in anchors}) != len(anchors):
        raise ValueError("duplicate anchor IDs")
    if len(anchors) * (budget + 1) * (max_per_anchor + 1) > 2_000_000:
        raise ValueError("allocation exceeds the bounded CPU work limit")
    for anchor in anchors:
        p = anchor.success_probability
        if (not isinstance(anchor.anchor_id, str) or not anchor.anchor_id or
                isinstance(p, bool) or not isinstance(p, (int, float)) or
                not math.isfinite(p) or not 0 <= p <= 1 or
                type(anchor.terminal) is not bool or anchor.terminal):
            raise ValueError("nonterminal named anchors with finite probabilities required")
        if stage == "root" and anchor.factual_reward is not None:
            raise ValueError("root anchors have no factual terminal reward")
        if stage == "prefix" and (type(anchor.factual_reward) is not int or anchor.factual_reward not in (0, 1)):
            raise ValueError("prefix anchors require a resolved binary factual reward")
    anchors = tuple(sorted(anchors, key=lambda a: a.anchor_id))
    choices = [0] + list(range(2 if stage == "root" else 1, max_per_anchor + 1))

    def utility(anchor, count):
        if count == 0:
            return 0.0
        p = anchor.success_probability
        if stage == "root":
            return 1.0 - p ** count - (1.0 - p) ** count
        same = p if anchor.factual_reward == 1 else 1.0 - p
        return 1.0 - same ** count

    # Each layer keeps only achievable exact budgets. Backpointers avoid
    # copying a complete allocation vector at every candidate transition.
    scores, backpointers = {0: 0.0}, []
    for anchor in anchors:
        next_scores, parent = {}, {}
        values = {count: utility(anchor, count) for count in choices}
        for used in sorted(scores):
            for count in choices:
                total = used + count
                if total > budget:
                    break
                candidate = scores[used] + values[count]
                if total not in next_scores or candidate > next_scores[total]:
                    next_scores[total] = candidate
                    parent[total] = (used, count)
        scores = next_scores
        backpointers.append(parent)
    if budget not in scores:
        raise ValueError("exact budget is infeasible under stage and caps")
    counts, remaining = {}, budget
    for anchor, parents in zip(reversed(anchors), reversed(backpointers)):
        remaining, count = parents[remaining]
        counts[anchor.anchor_id] = count
    return {"kind": "trace_contrast_allocation_independent_v1", "stage": stage,
            "allocation": {a.anchor_id: counts[a.anchor_id] for a in anchors},
            "total_continuations": budget, "predicted_contrast_objective": scores[budget],
            "probability_predictor_trained": False, "training_speedup_measured": False}
