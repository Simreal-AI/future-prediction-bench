"""Fail-closed result guard for the pinned official ROLL GEMRunner.

The stock runner returns ``Finished, score=0`` after a max-step or inference
response-error exit even when no candidate was submitted. These helpers
require a graded RealWorldEnv before its EpisodeResult can reach a manager.
They do not modify ROLL source or implement deferred rollout scheduling.
"""

from __future__ import annotations

import math

from .gem_bridge import RealWorldGemBridge, RewardNotVerified


def require_verified_roll_result(result, bridge: RealWorldGemBridge):
    """Return a ROLL EpisodeResult only if it matches a trusted final grade."""
    if not isinstance(bridge, RealWorldGemBridge) or bridge.env is None:
        raise RewardNotVerified(status="unverified",
                                reason="runner_environment_missing",
                                episode_id=None)
    env = bridge.env
    if env.status != "graded" or env.reward is None:
        raise RewardNotVerified(status=env.status,
                                reason=env.reason or "episode_not_graded",
                                episode_id=env.episode_id,
                                next_verify_at=(env.next_verify_at.isoformat()
                                                if env.next_verify_at else None))
    score = getattr(result, "score", None)
    steps = getattr(result, "step_scores", None)
    if (getattr(result, "status", None) != "Finished"
            or isinstance(score, bool) or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or not isinstance(steps, list) or len(steps) != env.actions_used
            or not steps or any(type(value) not in (int, float) or not math.isfinite(value)
                                for value in steps)
            or any(value != 0.0 for value in steps[:-1])
            or float(steps[-1]) != env.reward
            or float(score) != env.reward):
        raise RewardNotVerified(status="inconsistent_result",
                                reason="roll_result_differs_from_verified_episode",
                                episode_id=env.episode_id)
    return result


def guarded_gem_runner_class(upstream_gem_runner_cls):
    """Build a local subclass around a pinned ROLL GEMRunner class.

    The caller supplies ``env_factory(seed)`` at construction. ROLL's default
    YAML class loader does not provide that callable, so this class factory is
    a source-level integration example, not a ready-to-select training class.
    """
    if not isinstance(upstream_gem_runner_cls, type):
        raise TypeError("upstream_gem_runner_cls must be a class")

    class GuardedRealWorldGEMRunner(upstream_gem_runner_cls):
        def __init__(self, *args, env_factory, policy_prefix="roll", **kwargs):
            self._fpb_env_factory = env_factory
            self._fpb_policy_prefix = policy_prefix
            super().__init__(*args, **kwargs)

        def setup(self):
            self.env = RealWorldGemBridge(self._fpb_env_factory,
                                          policy_prefix=self._fpb_policy_prefix)

        def run_job(self, seed):
            result = super().run_job(seed)
            return require_verified_roll_result(result, self.env)

    return GuardedRealWorldGEMRunner
