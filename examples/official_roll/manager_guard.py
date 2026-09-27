"""Fail-closed boundary before ROLL ProxyEnvManager constructs training data.

This module imports no ROLL dependency. The operator's ROLL installation
supplies the pinned ProxyEnvManager and ``ray.get`` to the class factory. The
guard is intentionally separate from deferred reward scheduling: a pending
episode is *not* silently turned into a zero-score training sample.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace

from .gem_bridge import RealWorldGemBridge, RewardNotVerified
from .guarded_runner import require_verified_roll_result
from .sample_guard import (require_behavior_sample_lineage,
                           require_recorded_behavior_logprobs,
                           require_trainable_roll_sample)


def require_verified_harbor_result(result, bridge):
    """Check the exact result dict that ROLL passes to sample construction."""
    if not isinstance(result, Mapping):
        raise RewardNotVerified(status="inconsistent_result",
                                reason="roll_result_is_not_a_mapping",
                                episode_id=None)
    candidate = SimpleNamespace(
        status=result.get("status"), score=result.get("score"),
        step_scores=result.get("step_scores"),
    )
    require_verified_roll_result(candidate, bridge)
    return result


def guarded_proxy_env_manager_class(upstream_proxy_cls, *, ray_get):
    """Build a subclass whose final sample-construction gate checks the grade.

    ROLL's original ``run_rollout_loop`` remains in control of scheduling.
    If either the runner or this gate raises ``RewardNotVerified``, the wrapper
    posts ROLL's documented ``None`` completion marker for the claimed
    episode, then re-raises. The error is intentionally visible: no delayed
    reward queue is configured by this boundary alone.
    """
    if not isinstance(upstream_proxy_cls, type):
        raise TypeError("upstream_proxy_cls must be a class")
    if not callable(ray_get):
        raise TypeError("ray_get must be callable")

    class VerifiedProxyEnvManager(upstream_proxy_cls):
        def formulate_rollouts(self, harbor_result):
            bridge = getattr(getattr(self, "agent_runner", None), "env", None)
            if not isinstance(bridge, RealWorldGemBridge):
                raise RewardNotVerified(status="unverified",
                                        reason="roll_runner_bridge_missing",
                                        episode_id=None)
            require_verified_harbor_result(harbor_result, bridge)
            require_recorded_behavior_logprobs(
                getattr(self, "history", ()), episode_id=bridge.env.episode_id,
            )
            sample = super().formulate_rollouts(harbor_result)
            require_trainable_roll_sample(
                sample, expected_reward=bridge.env.reward,
                episode_id=bridge.env.episode_id,
            )
            return require_behavior_sample_lineage(
                sample, self.history, episode_id=bridge.env.episode_id,
            )

        def run_rollout_loop(self, data):
            try:
                return super().run_rollout_loop(data)
            except RewardNotVerified as error:
                self.running = False
                self.last_unverified_status = error.status
                episode_id = getattr(self, "episode_id", None)
                if episode_id is not None:
                    ray_get(self.output_queue.put.remote(
                        self.env_config["group_id"], episode_id,
                        self.current_step, None, self.env_config["env_id"],
                    ))
                raise

    return VerifiedProxyEnvManager
