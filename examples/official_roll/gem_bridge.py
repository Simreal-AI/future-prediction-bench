"""A narrow RealWorldEnv adapter for ROLL's pinned GEMRunner text protocol.

No ROLL source is vendored or imported here. This bridge only implements the
five-value environment protocol; it does not construct ROLL training samples.
"""

from __future__ import annotations

import json

from future_prediction_bench.realworld import RealWorldEnv


class RewardNotVerified(RuntimeError):
    """The episode has no trusted terminal scalar reward to give a trainer."""

    def __init__(self, *, status: str, reason: str, episode_id: str | None,
                 next_verify_at: str | None = None):
        self.status = status
        self.reason = reason
        self.episode_id = episode_id
        self.next_verify_at = next_verify_at
        # The reason may originate in a private verifier; keep it out of logs.
        super().__init__(f"No verified reward: {status}")


class RealWorldGemBridge:
    """Match the ``reset(seed)``/``step(action_text)`` calls in ROLL GEMRunner.

    ``env_factory(seed)`` must construct a fresh RealWorldEnv with a frozen
    task and trusted adapter. The caller owns seed-to-task mapping and outcome
    scheduling. This class deliberately raises on every ungraded terminal
    state because GEMRunner unconditionally appends and sums returned rewards.
    """

    def __init__(self, env_factory, *, policy_prefix: str = "roll"):
        if not callable(env_factory):
            raise TypeError("env_factory must be callable")
        if not isinstance(policy_prefix, str) or not policy_prefix.strip():
            raise ValueError("policy_prefix must be nonblank")
        self.env_factory = env_factory
        self.policy_prefix = policy_prefix
        self.env: RealWorldEnv | None = None

    @staticmethod
    def _text(value) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False)

    @staticmethod
    def _action(action_text: str) -> dict:
        if not isinstance(action_text, str):
            raise TypeError("ROLL GEMRunner must pass action text")
        try:
            action = json.loads(action_text) if len(action_text) <= 65536 else None
            json.dumps(action, allow_nan=False)
        except (ValueError, RecursionError, UnicodeError):
            action = None
        # This reserved action is never a valid task tool name. Malformed model
        # output still consumes an action through RealWorldEnv's policy budget.
        return action if isinstance(action, dict) else {"action": "_invalid_roll_json"}

    def reset(self, seed: int):
        if type(seed) is not int:
            raise TypeError("seed must be an integer")
        if self.env is not None:
            if self.env.status != "graded":
                raise RewardNotVerified(status=self.env.status,
                                        reason=self.env.reason or "episode_not_graded",
                                        episode_id=self.env.episode_id)
            self.close()
        env = self.env_factory(seed)
        if not isinstance(env, RealWorldEnv):
            raise TypeError("env_factory must return RealWorldEnv")
        try:
            opening = env.reset(f"{self.policy_prefix}-seed-{seed}")
            if opening["status"] != "active":
                raise RewardNotVerified(status=opening["status"],
                                        reason=opening["observation"].get("reason", "reset_not_active"),
                                        episode_id=opening["episode_id"])
        except Exception:
            close = getattr(env.adapter, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
            raise
        self.env = env
        info = {"env_instruction": "Return exactly one JSON object naming an available task action.",
                "episode_id": opening["episode_id"],
                "task_id": opening["task"]["task_id"]}
        return self._text(opening), info

    def _grade_or_raise(self, result):
        assert self.env is not None
        if result["status"] == "graded":
            return float(result["reward"])
        raise RewardNotVerified(status=result["status"],
                                reason=result.get("reason") or "ungraded_terminal_episode",
                                episode_id=self.env.episode_id,
                                next_verify_at=result.get("next_verify_at"))

    def step(self, action_text: str):
        if self.env is None or self.env.status != "active":
            raise ValueError("step requires an active reset episode")
        transition = self.env.step(self._action(action_text))
        info = {"episode_id": transition["info"]["episode_id"],
                "status": transition["info"]["status"]}
        observation = self._text(transition["observation"])
        if not transition["terminated"]:
            # There is no verified terminal outcome yet. Zero here is only an
            # intermediate step reward, never a substitute for a pending grade.
            return observation, 0.0, False, False, info
        if self.env.status != "pending":
            raise RewardNotVerified(status=self.env.status,
                                    reason=self.env.reason or "episode_not_submitted",
                                    episode_id=self.env.episode_id)
        reward = self._grade_or_raise(self.env.verify())
        info["status"] = "graded"
        return observation, reward, True, False, info

    def collect_verified_reward(self) -> float:
        """Trusted-host polling helper; ROLL GEMRunner does not call this."""
        if self.env is None or self.env.status not in {"pending", "graded", "void"}:
            raise ValueError("No submitted episode is available for verification")
        return self._grade_or_raise(self.env.verify())

    def close(self):
        if self.env is None:
            return
        close = getattr(self.env.adapter, "close", None)
        try:
            if callable(close):
                close()
        finally:
            self.env = None
