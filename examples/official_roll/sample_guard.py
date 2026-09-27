"""Validate a ROLL ProxyEnvManager sample before it enters a training queue.

The pinned upstream formatter can create a zero-score placeholder when its
MessageTracker has no generated tokens. A verified environment grade alone is
therefore insufficient to establish that a training sample is usable.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from .gem_bridge import RewardNotVerified


def _reject(reason: str, episode_id=None):
    raise RewardNotVerified(
        status="untrainable_sample", reason=reason, episode_id=episode_id,
    )


def require_recorded_behavior_logprobs(history, *, episode_id=None):
    """Reject inference steps that ROLL would silently mask or fill with zero.

    ROLL's message tracker treats a missing logprob list differently in
    trajectory and step mode. Both silently discard behavior-policy evidence
    for that response, so a configured training adapter requires it here.
    """
    for step in history:
        if not isinstance(step, Mapping):
            _reject("malformed_inference_history", episode_id)
        tokens = step.get("response_ids")
        logprobs = step.get("logprobs")
        if not isinstance(tokens, (list, tuple)) or not tokens:
            _reject("response_ids_missing", episode_id)
        if any(type(token) is not int or token < 0 for token in tokens):
            _reject("malformed_response_token", episode_id)
        if not isinstance(logprobs, (list, tuple)) or len(logprobs) != len(tokens):
            _reject("behavior_logprobs_missing", episode_id)
        try:
            if not all(math.isfinite(float(value)) for value in logprobs):
                _reject("nonfinite_behavior_logprob", episode_id)
        except (TypeError, ValueError, OverflowError):
            _reject("malformed_behavior_logprob", episode_id)
    return history


def require_behavior_sample_lineage(sample, history, *, episode_id=None):
    """Check that queued policy tokens/logprobs are exactly the recorded ones.

    This supports the pinned manager's ordinary single-branch trajectory and
    ordered step modes. Forked/duplicated branches fail closed until their
    lineage is represented explicitly rather than inferred from row order.
    """
    require_recorded_behavior_logprobs(history, episode_id=episode_id)
    batch = getattr(sample, "batch", None)
    if batch is None:
        _reject("sample_batch_missing", episode_id)
    try:
        token_rows = batch["input_ids"].tolist()
        mask_rows = batch["response_mask"].tolist()
        logprob_rows = batch["infer_logprobs"].tolist()
        sampled = []
        for tokens, masks, logprobs in zip(token_rows, mask_rows, logprob_rows):
            for position, is_response in enumerate(masks):
                if is_response:
                    if position == 0:
                        _reject("response_at_unshiftable_position", episode_id)
                    sampled.append((tokens[position], logprobs[position - 1]))
        recorded = [(token, logprob)
                    for step in history
                    for token, logprob in zip(step["response_ids"], step["logprobs"])]
        if len(sampled) != len(recorded) or not recorded:
            _reject("response_lineage_length_mismatch", episode_id)
        for (sample_token, sample_prob), (record_token, record_prob) in zip(
                sampled, recorded):
            if (sample_token != record_token
                    or not math.isclose(float(sample_prob), float(record_prob),
                                        rel_tol=0.0, abs_tol=1e-5)):
                _reject("response_lineage_mismatch", episode_id)
    except (TypeError, ValueError, IndexError, KeyError, AttributeError) as error:
        _reject(f"malformed_response_lineage:{type(error).__name__}", episode_id)
    return sample


def require_trainable_roll_sample(sample, *, expected_reward: float,
                                  episode_id=None):
    """Fail closed on empty/misaligned token masks, logprobs, or reward.

    This checks the actual tensors that ``ProxyEnvManager.formulate_rollouts``
    returns. It deliberately makes no optimizer or throughput claim.
    """
    batch = getattr(sample, "batch", None)
    metadata = getattr(sample, "non_tensor_batch", None)
    if batch is None or not hasattr(batch, "keys") or not hasattr(batch, "__getitem__") \
            or not isinstance(metadata, Mapping):
        _reject("sample_batch_or_metadata_missing", episode_id)
    keys = ("input_ids", "attention_mask", "response_mask", "prompt_mask",
            "scores", "infer_logprobs")
    if any(key not in batch for key in keys):
        _reject("required_token_tensor_missing", episode_id)
    if "episode_scores" not in metadata or "step_scores" not in metadata:
        _reject("reward_metadata_missing", episode_id)

    try:
        shapes = {key: tuple(batch[key].shape) for key in keys}
        n_rows, n_tokens = shapes["input_ids"]
        if n_rows < 1 or n_tokens < 2:
            _reject("empty_token_batch", episode_id)
        for key in keys[:-1]:
            if shapes[key] != (n_rows, n_tokens):
                _reject("misaligned_token_tensor", episode_id)
        if shapes["infer_logprobs"] != (n_rows, n_tokens - 1):
            _reject("misaligned_infer_logprobs", episode_id)
        if len(metadata["episode_scores"]) != n_rows or len(metadata["step_scores"]) != n_rows:
            _reject("misaligned_reward_metadata", episode_id)

        attn = batch["attention_mask"].tolist()
        response = batch["response_mask"].tolist()
        prompt = batch["prompt_mask"].tolist()
        scores = batch["scores"].tolist()
        logprobs = batch["infer_logprobs"].tolist()
        expected = float(expected_reward)
        if not math.isfinite(expected):
            _reject("nonfinite_verified_reward", episode_id)

        for index in range(n_rows):
            if not math.isclose(float(metadata["episode_scores"][index]), expected,
                                rel_tol=0.0, abs_tol=1e-6):
                _reject("episode_reward_mismatch", episode_id)
            if not math.isclose(sum(float(value) for value in scores[index]),
                                float(metadata["step_scores"][index]),
                                rel_tol=0.0, abs_tol=1e-6):
                _reject("token_reward_mismatch", episode_id)
            if not all(math.isfinite(float(value)) for value in scores[index]):
                _reject("nonfinite_token_reward", episode_id)
            if any(bool(response[index][position]) and not bool(attn[index][position])
                   for position in range(n_tokens)):
                _reject("response_on_padding", episode_id)
            if any(bool(prompt[index][position]) and not bool(attn[index][position])
                   for position in range(n_tokens)):
                _reject("prompt_on_padding", episode_id)
            if any(bool(prompt[index][position]) and bool(response[index][position])
                   for position in range(n_tokens)):
                _reject("overlapping_prompt_response", episode_id)
            if bool(response[index][0]) or not any(bool(x) for x in response[index][1:]):
                _reject("no_trainable_response_token", episode_id)
            if not any(bool(x) for x in prompt[index]):
                _reject("no_prompt_token", episode_id)
            for position in range(1, n_tokens):
                if bool(response[index][position]) and not math.isfinite(
                    float(logprobs[index][position - 1])
                ):
                    _reject("nonfinite_policy_logprob", episode_id)
    except (TypeError, ValueError, IndexError, KeyError, AttributeError) as error:
        _reject(f"malformed_sample:{type(error).__name__}", episode_id)
    return sample
