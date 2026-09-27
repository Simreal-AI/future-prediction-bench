"""Proper probability scores shared by benchmark evaluation and RL rewards."""

from __future__ import annotations

import math
from numbers import Real

from .schema import validate_probabilities


def _validated_distribution(probabilities: dict, outcome: str) -> dict[str, float]:
    if not isinstance(probabilities, dict):
        raise ValueError("probabilities must be an object keyed by option ID")
    distribution = validate_probabilities(probabilities, probabilities.keys())
    if not isinstance(outcome, str) or outcome not in distribution:
        raise ValueError("outcome must be one of the submitted option IDs")
    return distribution


def normalized_brier(probabilities: dict, outcome: str) -> float:
    """Return 0.5 * sum((p_k - 1[k == outcome])**2), with smaller being better.

    This convention ranges from 0 to 1 for normalized distributions and reduces
    to (p_yes - y)**2 for binary questions. It is half the unscaled multiclass
    Brier score; report the convention when comparing external leaderboards.
    """
    distribution = _validated_distribution(probabilities, outcome)
    return 0.5 * math.fsum(
        (probability - float(identifier == outcome)) ** 2
        for identifier, probability in distribution.items()
    )


def clipped_log_loss(probabilities: dict, outcome: str, epsilon: float = 1e-15) -> float:
    """Return -ln(max(p_outcome, epsilon)); a perfect forecast has zero loss.

    Clipping only the lower end keeps impossible-outcome submissions finite.
    Brier is the primary score because finite log clipping alters strict propriety
    near the clipping threshold. Natural logarithms give this score in nats.
    """
    distribution = _validated_distribution(probabilities, outcome)
    if isinstance(epsilon, bool) or not isinstance(epsilon, Real):
        raise ValueError("epsilon must be a finite real number in (0, 1)")
    try:
        epsilon = float(epsilon)
    except (ValueError, OverflowError) as exc:
        raise ValueError("epsilon must be a finite real number in (0, 1)") from exc
    if not math.isfinite(epsilon) or not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must be a finite real number in (0, 1)")
    return -math.log(max(distribution[outcome], epsilon))


def score_prediction(probabilities: dict, outcome: str) -> dict[str, float]:
    """Score one resolved question, including reward and an option-count baseline.

    Uniform Brier is (K-1)/(2K). Skill is 1 - Brier / uniform Brier, so uniform
    forecasts score zero skill and a perfect forecast scores one. Aggregate raw
    Brier and uniform Brier first to compute dataset-level Brier skill.
    """
    distribution = _validated_distribution(probabilities, outcome)
    brier = normalized_brier(distribution, outcome)
    uniform_brier = (len(distribution) - 1) / (2.0 * len(distribution))
    return {
        "brier": brier,
        "clipped_log_loss": clipped_log_loss(distribution, outcome),
        "log_loss_epsilon": 1e-15,
        "reward": -brier,
        "uniform_brier": uniform_brier,
        "brier_skill_vs_uniform": 1.0 - brier / uniform_brier,
    }
