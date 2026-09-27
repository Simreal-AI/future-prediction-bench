"""Algorithm-neutral RL collection and text-batch preparation.

This module does not tokenize text, recover behavior log probabilities, or run a
policy optimizer. A delayed trajectory must not silently become on-policy data.
"""

from __future__ import annotations

import copy
import math
from collections import Counter, defaultdict

from .schema import option_ids, parse_timestamp, public_question, validate_question
from .store import digest


ADVANTAGE_METHODS = ("rloo", "centered", "standard_grpo")


def validate_rollout_context(context):
    required = {"group_id", "sample_index", "group_size", "policy_revision"}
    optional = {"collection_config_hash", "evidence_pack_id"}
    if not isinstance(context, dict) or not required <= context.keys() or context.keys() - required - optional:
        raise ValueError("Invalid rollout context fields")
    for key in ("group_id", "policy_revision", *sorted(optional & context.keys())):
        if not isinstance(context[key], str) or not context[key].strip() or len(context[key]) > 256:
            raise ValueError(f"{key} must be a nonempty string of at most 256 characters")
    size, index = context["group_size"], context["sample_index"]
    if isinstance(size, bool) or not isinstance(size, int) or not 2 <= size <= 64:
        raise ValueError("group_size must be between 2 and 64")
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < size:
        raise ValueError("sample_index must be an integer within the group")
    return copy.deepcopy(context)


def group_advantages(rewards, method="rloo"):
    """Sequence advantages; standard_grpo is an explicit calibration ablation.

    No batch/group standard deviation is applied by the default estimator.
    Equal-reward groups return zero; they are reported rather than resampled.
    """
    if method not in ADVANTAGE_METHODS:
        raise ValueError("Unknown advantage method")
    rewards = list(rewards)
    if len(rewards) < 2 or any(isinstance(x, bool) or not isinstance(x, (int, float))
                               or not math.isfinite(x) or abs(x) > 1 for x in rewards):
        raise ValueError("At least two finite rewards in [-1, 1] are required")
    mean = math.fsum(rewards) / len(rewards)
    centered = [reward - mean for reward in rewards]
    if method == "rloo":
        return [value * len(rewards) / (len(rewards) - 1) for value in centered]
    if method == "centered":
        return centered
    std = math.sqrt(math.fsum(value * value for value in centered) / len(rewards))
    return [value / (std + 1e-8) for value in centered]


def assistant_segments(model_turns):
    """Loss masks over exact message objects, never over duplicated tool events.

    The trainer must apply its exact chat template and derive token masks from
    these spans. Observation tokens stay in context but have no policy loss.
    Context rewrites require a separate explicit adapter and are rejected here.
    """
    if not isinstance(model_turns, list) or not model_turns:
        raise ValueError("missing_model_trajectory")
    segments, history = [], []
    for index, turn in enumerate(model_turns):
        if not isinstance(turn, dict) or turn.get("error") or not isinstance(turn.get("response"), dict):
            raise ValueError("incomplete_model_trajectory")
        request = turn.get("request", {})
        if not isinstance(request, dict):
            raise ValueError("malformed_model_request")
        messages = request.get("messages")
        if not isinstance(messages, list) or messages[:len(history)] != history:
            raise ValueError("unsupported_context_rewrite")
        added = messages[len(history):]
        if any(not isinstance(message, dict) or message.get("role") not in {"system", "user", "tool"}
               for message in added):
            raise ValueError("unattributed_assistant_tokens")
        if index == 0 and (not added or not any(message["role"] == "user" for message in added)):
            raise ValueError("missing_question_context")
        segments.extend({"message": copy.deepcopy(message), "loss_mask": 0, "origin": "context"}
                        for message in added)
        response = turn["response"]
        if response.get("role") != "assistant":
            raise ValueError("invalid_assistant_response")
        segments.append({"message": copy.deepcopy(response), "loss_mask": 1, "origin": "policy"})
        history = copy.deepcopy(messages) + [copy.deepcopy(response)]
    return segments


def _validate_metadata(meta):
    context_fields = {"group_id", "group_size", "sample_index", "policy_revision",
                      "collection_config_hash", "evidence_pack_id"}
    validate_rollout_context({key: meta[key] for key in context_fields if key in meta})
    if not meta.get("collection_config_hash"):
        raise ValueError("missing_collection_config_hash")
    if meta.get("sha256") != digest({key: value for key, value in meta.items() if key != "sha256"}):
        raise ValueError("rollout_metadata_digest_mismatch")


def _validate_model_turns(turns, *, registered_at, submitted_at, config_hash):
    segments = assistant_segments(turns)
    previous_at, tools = registered_at, None
    for turn in turns:
        at = parse_timestamp(turn.get("at"))
        if not previous_at <= at <= submitted_at:
            raise ValueError("invalid_model_turn_timing")
        previous_at = at
        if turn.get("config_hash") != config_hash:
            raise ValueError("model_config_mismatch")
        request_tools = turn["request"].get("tools")
        if not isinstance(request_tools, list) or any(not isinstance(tool, dict) for tool in request_tools):
            raise ValueError("missing_or_malformed_tool_definitions")
        if tools is not None and request_tools != tools:
            raise ValueError("unsupported_tool_schema_change")
        tools = request_tools
    return segments


def _validate_events(events, *, registered_at, submitted_at):
    """Check audit events without converting host actions into policy targets."""
    if not isinstance(events, list) or not events:
        raise ValueError("missing_submission_event")
    previous_at = registered_at
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError("malformed_trajectory_event")
        at = parse_timestamp(event.get("at"))
        if not previous_at <= at <= submitted_at:
            raise ValueError("invalid_event_timing")
        previous_at = at
        if event.get("sha256") != digest({key: value for key, value in event.items() if key != "sha256"}):
            raise ValueError("event_digest_mismatch")
        is_submit = event.get("tool") == "submit"
        if is_submit:
            if index != len(events) - 1 or at != submitted_at or event.get("type") != "action":
                raise ValueError("invalid_submission_event")
            if event.get("origin") not in {"policy", "host"}:
                raise ValueError("invalid_submission_origin")
            expected_mask = int(event["origin"] == "policy")
        else:
            if event.get("type") not in {"action", "observation"}:
                raise ValueError("malformed_trajectory_event")
            expected_mask = int(event["type"] == "action")
        if type(event.get("loss_mask")) is not int or event["loss_mask"] != expected_mask:
            raise ValueError("invalid_event_loss_mask")
    if events[-1].get("tool") != "submit":
        raise ValueError("missing_submission_event")


def prepare_training_groups(records, *, current_policy_revision, available_at, method="rloo", run_mode="live"):
    """Require complete, settled, same-revision groups; quarantine everything else.

    Exact revision matching is conservative. Async/off-policy learners may use
    older rollouts only through an adapter with actual token-level policy ratios.
    Nothing here fabricates those ratios or treats unknown revisions as current.
    """
    if method not in ADVANTAGE_METHODS or run_mode not in {"live", "fixture"}:
        raise ValueError("Invalid training preparation mode")
    if not isinstance(current_policy_revision, str) or not current_policy_revision.strip():
        raise ValueError("current_policy_revision is required")
    cutoff = parse_timestamp(available_at)
    buckets, rejected = defaultdict(list), []
    for record in records:
        if not isinstance(record, dict):
            rejected.append({"group_id": None, "reason": "malformed_training_record", "count": 1})
            continue
        rollout = record.get("rollout")
        if not isinstance(rollout, dict) or not isinstance(rollout.get("group_id"), str):
            rejected.append({"group_id": None, "reason": "unregistered_rollout", "count": 1})
        else:
            buckets[rollout["group_id"]].append(record)
    groups = []
    for group_id, items in sorted(buckets.items()):
        try:
            first = items[0]["rollout"]
            for item in items:
                _validate_metadata(item["rollout"])
            size = first["group_size"]
            if len(items) != size or sorted(item["rollout"]["sample_index"] for item in items) != list(range(size)):
                raise ValueError("incomplete_or_duplicate_group")
            if first["policy_revision"] != current_policy_revision:
                raise ValueError("stale_policy_requires_off_policy_adapter")
            scope = ("question_id", "question_sha256", "policy_revision", "collection_config_hash",
                     "group_size", "market_mode", "research_mode", "reward_mode", "evidence_pack_id")
            if any(any(item["rollout"].get(key) != first.get(key) for key in scope) for item in items):
                raise ValueError("mixed_collection_context")
            prepared, episode_ids = [], set()
            common_question, common_resolution, common_resolved_at, common_tools = None, None, None, None
            for item in sorted(items, key=lambda value: value["rollout"]["sample_index"]):
                episode, question = item["episode"], item["question"]
                if not isinstance(episode, dict) or not isinstance(question, dict):
                    raise ValueError("malformed_training_record")
                if item.get("split") != "train" or question.get("split") != "train" or episode.get("track") != "rl":
                    raise ValueError("heldout_or_non_rl_data")
                if (item.get("run_mode") != run_mode or episode.get("run_mode") != run_mode
                        or type(question.get("is_fixture", False)) is not bool
                        or question.get("is_fixture", False) != (run_mode == "fixture")):
                    raise ValueError("run_mode_mismatch")
                validate_question(question)
                if digest(public_question(question)) != first["question_sha256"]:
                    raise ValueError("question_digest_mismatch")
                if common_question is not None and question != common_question:
                    raise ValueError("mixed_question_identity")
                common_question = question
                if any(episode.get(key) != first.get(key) for key in ("question_id", "market_mode", "research_mode", "reward_mode")):
                    raise ValueError("episode_context_mismatch")
                if episode.get("policy_id") != "agent-" + first["collection_config_hash"]:
                    raise ValueError("episode_policy_mismatch")
                episode_id = episode.get("episode_id")
                if not isinstance(episode_id, str) or not episode_id.strip() or episode_id in episode_ids:
                    raise ValueError("missing_or_duplicate_episode_id")
                episode_ids.add(episode_id)
                if item.get("group_key") != [episode[key] for key in
                                             ("question_id", "policy_id", "market_mode", "research_mode", "reward_mode")]:
                    raise ValueError("group_key_mismatch")
                if question.get("question_id") != first["question_id"]:
                    raise ValueError("question_context_mismatch")
                if episode.get("status") not in {"graded", "invalid"}:
                    raise ValueError("unsettled_episode")
                resolution = item.get("resolution")
                if not isinstance(resolution, dict) or resolution.get("status") != "resolved" or not item.get("resolved_at"):
                    raise ValueError("unsettled_or_void_question")
                resolved_at = parse_timestamp(item["resolved_at"])
                if resolved_at > cutoff:
                    raise ValueError("outcome_unavailable_at_cutoff")
                if resolved_at < parse_timestamp(question["outcome_not_before"]):
                    raise ValueError("resolution_before_outcome_window")
                if resolution.get("outcome") not in option_ids(question):
                    raise ValueError("invalid_resolved_outcome")
                if (common_resolution is not None and
                        (resolution != common_resolution or resolved_at != common_resolved_at)):
                    raise ValueError("mixed_resolution_context")
                common_resolution, common_resolved_at = resolution, resolved_at
                created_at = parse_timestamp(episode["created_at"])
                submitted_at = parse_timestamp(episode["submitted_at"])
                registered_at = parse_timestamp(item["rollout"]["registered_at"])
                if not (parse_timestamp(question["issued_at"]) <= created_at <= registered_at
                        <= submitted_at < parse_timestamp(question["forecast_deadline"])):
                    raise ValueError("invalid_forecast_timing")
                segments = _validate_model_turns(item["model_turns"], registered_at=registered_at,
                                                submitted_at=submitted_at, config_hash=first["collection_config_hash"])
                request_tools = item["model_turns"][0]["request"]["tools"]
                if common_tools is not None and request_tools != common_tools:
                    raise ValueError("mixed_tool_definitions")
                common_tools = request_tools
                _validate_events(item["events"], registered_at=registered_at, submitted_at=submitted_at)
                prepared.append({"episode_id": episode["episode_id"], "sample_index": item["rollout"]["sample_index"],
                                 "reward": episode["reward"], "segments": segments,
                                 "model_turns": copy.deepcopy(item["model_turns"]),
                                 "events": copy.deepcopy(item["events"]), "events_are_audit_only": True,
                                 "behavior_policy_revision": first["policy_revision"],
                                 "behavior_logprobs_available": False, "token_ids_available": False})
            advantages = group_advantages([item["reward"] for item in prepared], method)
            for item, advantage in zip(prepared, advantages):
                item["sequence_advantage"] = advantage
            groups.append({"group_id": group_id, "question_id": first["question_id"],
                           "cluster_id": items[0]["question"]["cluster_id"], "method": method,
                           "zero_variance": all(abs(value) < 1e-12 for value in advantages),
                           "collection": copy.deepcopy({key: first.get(key) for key in scope}), "samples": prepared})
        except (ValueError, KeyError, TypeError) as exc:
            reason = str(exc) if isinstance(exc, ValueError) else "malformed_training_record"
            rejected.append({"group_id": group_id, "reason": reason, "count": len(items)})
    counts = Counter()
    for item in rejected:
        counts[item["reason"]] += item["count"]
    return {"schema_version": "0.3", "record_type": "prepared_text_groups", "trainer_ready": False,
            "run_mode": run_mode, "advantage_method": method, "available_at": available_at,
            "current_policy_revision": current_policy_revision,
            "requirements": ["exact_chat_template_tokenization", "assistant_token_loss_masks",
                             "behavior_token_logprobs", "trainer_optimizer_and_policy_ratio_checks"],
            "groups": groups, "quarantined": rejected,
            "summary": {"prepared_groups": len(groups), "prepared_samples": sum(len(g["samples"]) for g in groups),
                        "zero_variance_groups": sum(g["zero_variance"] for g in groups),
                        "quarantined_samples_by_reason": dict(counts)}}


def collect_rollout_group(store, question_id, *, group_id, group_size, policy_revision, model, provider=None,
                          evidence_pack_id=None, market_provider=None, **run_options):
    """Collect a fixed group before deadline; repeated calls skip assigned indices.

    The model adapter controls stochastic decoding. A group is not a license to
    retry failed samples or select only high-reward trajectories after settlement.
    """
    from .runner import RunLimits, run_questions, runner_config
    context = {"group_id": group_id, "sample_index": 0, "group_size": group_size, "policy_revision": policy_revision}
    if evidence_pack_id is not None:
        context["evidence_pack_id"] = evidence_pack_id
    validate_rollout_context(context)
    if run_options.pop("track", "rl") != "rl":
        raise ValueError("Grouped collection is restricted to the RL track")
    limit_fields = set(RunLimits.__dataclass_fields__)
    modes = {"market_mode": "no_consensus", "research_mode": "self_research", "reward_mode": "baseline_improvement"}
    if run_options.keys() - limit_fields - modes.keys():
        raise ValueError("Unsupported grouped collection options")
    modes.update({key: run_options[key] for key in modes.keys() & run_options.keys()})
    limits = RunLimits(**{key: run_options[key] for key in limit_fields & run_options.keys()})
    config = runner_config(model, provider, limits, market_provider=market_provider, track="rl", **modes)
    expected = {**context, **modes, "question_id": question_id,
                "question_sha256": digest(public_question(store.question(question_id))),
                "collection_config_hash": digest(config)}
    # Validate the entire existing contract before skipping or adding any sample.
    for row in store.db.execute("SELECT episode_id FROM rollout_metadata WHERE group_id=?", (group_id,)):
        metadata = store.rollout(row[0])
        _validate_metadata(metadata)
        if any(metadata.get(key) != value for key, value in expected.items() if key != "sample_index") or metadata.get("evidence_pack_id") != evidence_pack_id:
            raise ValueError("Existing group has a different collection contract")
    reports = []
    for index in range(group_size):
        existing = store.db.execute("SELECT episode_id FROM rollout_metadata WHERE group_id=? AND sample_index=?",
                                    (group_id, index)).fetchone()
        if existing:
            reports.append({"episode_id": existing[0], "sample_index": index, "status": "skipped_existing_assignment"})
            continue
        reports.extend(run_questions(store, [question_id], model=model, provider=provider, market_provider=market_provider, track="rl",
                                     rollout_context={**context, "sample_index": index}, **run_options))
    return reports


def calibration_probe(true_probability=0.7, delta=0.05):
    """Exact two-outcome toy expectation, not a model-training experiment.

    Compare two candidate reports p-delta and p+delta with binary Brier rewards.
    A positive expected advantage favors the upper report. At p=q, symmetric
    reports have equal expected Brier; a systematic positive preference is bias.
    """
    if not 0 < true_probability < 1 or not 0 < delta < .1:
        raise ValueError("Use an interior true probability and delta in (0, .1)")
    rows = []
    for center in sorted({.2, .5, true_probability, .9}):
        if not delta < center < 1 - delta:
            continue
        expected = {}
        for method in ADVANTAGE_METHODS:
            value = 0.0
            for outcome, weight in ((1, true_probability), (0, 1 - true_probability)):
                rewards = [-(probability - outcome) ** 2 for probability in (center - delta, center + delta)]
                value += weight * group_advantages(rewards, method)[1]
            expected[method] = value
        rows.append({"center_probability": center, "expected_upper_advantage": expected})
    return {"experiment": "exact_binary_brier_advantage_probe", "is_model_training": False,
            "true_probability": true_probability, "candidate_delta": delta, "rows": rows}
