"""Host-side sibling-return preparation for real-world branch trajectories.

This is a branch-local RLOO estimator, not a policy optimizer. Shared-prefix
actions receive no local sibling advantage: every sibling begins from the same
frozen state, and this contrast says nothing about which action reached it.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone

from .realworld import SCHEMA_VERSION, _digest, _timestamp


_SHA = re.compile(r"[0-9a-f]{64}\Z")


def prepare_sibling_advantages(records, *, expected_siblings: int,
                               current_policy_id: str, as_of: datetime) -> dict:
    """Validate a complete host-owned sibling cohort and prepare suffix data.

    ``expected_siblings`` comes from the frozen collection plan. Incomplete,
    mixed-policy, mixed-checkpoint, malformed, or ungraded cohorts fail closed.
    Output remains text-level and explicitly ``trainer_ready=False``.
    """
    if (type(expected_siblings) is not int or not 2 <= expected_siblings <= 64
            or not isinstance(current_policy_id, str) or not current_policy_id):
        raise ValueError("A planned sibling count and current policy ID are required")
    if (not isinstance(as_of, datetime) or as_of.tzinfo is None
            or as_of.utcoffset() is None):
        raise ValueError("A timezone-aware preparation time is required")
    as_of = as_of.astimezone(timezone.utc)
    if not isinstance(records, (list, tuple)) or len(records) != expected_siblings:
        raise ValueError("Sibling cohort is incomplete or oversized")

    prepared = []
    common = None
    prefix_hashes = None
    episode_ids = set()
    branch_ids = set()
    for record in records:
        if (not isinstance(record, dict)
                or record.get("schema_version") != SCHEMA_VERSION
                or record.get("record_type") != "realworld_text_trajectory"
                or record.get("trainer_ready") is not False
                or record.get("policy_id") != current_policy_id
                or not isinstance(record.get("is_fixture"), bool)):
            raise ValueError("Invalid or stale real-world branch trajectory")
        lineage = record.get("branch_lineage")
        if not isinstance(lineage, dict):
            raise ValueError("A branch lineage is required")
        key_fields = ("task_id", "task_sha256", "reward_contract_sha256",
                      "policy_id", "is_fixture")
        lineage_fields = ("parent_episode_id", "checkpoint_id",
                          "prefix_event_sha256", "checkpoint_kind",
                          "prefix_visible_event_count")
        group = tuple(record.get(field) for field in key_fields) + tuple(
            lineage.get(field) for field in lineage_fields)
        if common is None:
            common = group
        elif group != common:
            raise ValueError("Sibling trajectories have different frozen parents or tasks")
        if any(not isinstance(record.get(field), str) or not record[field]
               for field in ("task_id", "policy_id", "episode_id")):
            raise ValueError("Task, policy, and episode IDs are required")
        if any(not isinstance(record.get(field), str) or not _SHA.fullmatch(record[field])
               for field in ("task_sha256", "reward_contract_sha256")):
            raise ValueError("Frozen task and reward hashes are required")
        if (any(not isinstance(lineage.get(field), str) or not lineage[field]
                for field in ("parent_episode_id", "checkpoint_id",
                              "checkpoint_kind", "branch_id"))
                or not isinstance(lineage.get("prefix_event_sha256"), str)
                or not _SHA.fullmatch(lineage["prefix_event_sha256"])):
            raise ValueError("Invalid branch lineage")
        episode_id, branch_id = record["episode_id"], lineage["branch_id"]
        if episode_id in episode_ids or branch_id in branch_ids:
            raise ValueError("Duplicate sibling episode or branch ID")
        episode_ids.add(episode_id)
        branch_ids.add(branch_id)

        reward = record.get("reward")
        if (isinstance(reward, bool) or not isinstance(reward, (int, float))
                or not math.isfinite(reward) or not -1 <= reward <= 1):
            raise ValueError("Sibling reward must be finite and within the task range")
        submitted = _timestamp(record.get("submitted_at"), "submitted_at")
        available = _timestamp(record.get("available_at"), "available_at")
        resolved = _timestamp(record.get("resolved_at"), "resolved_at")
        if not submitted <= available <= resolved <= as_of:
            raise ValueError("Sibling reward was not resolved by preparation time")
        events = record.get("events")
        count = lineage.get("prefix_visible_event_count")
        if (not isinstance(events, list) or type(count) is not int
                or not 1 <= count < len(events)):
            raise ValueError("Visible branch prefix boundary is missing")
        for event in events:
            if (not isinstance(event, dict) or event.get("visible_to_policy") is not True
                    or not isinstance(event.get("sha256"), str)
                    or not _SHA.fullmatch(event["sha256"])
                    or _digest({field: value for field, value in event.items()
                                if field != "sha256"}) != event["sha256"]):
                raise ValueError("Visible event digest or provenance differs")
        current_hashes = tuple(event["sha256"] for event in events[:count])
        if prefix_hashes is None:
            prefix_hashes = current_hashes
        elif current_hashes != prefix_hashes:
            raise ValueError("Sibling policy-visible prefixes differ")
        suffix_actions = [event["sha256"] for event in events[count:]
                          if event.get("origin") == "policy"
                          and event.get("kind") == "action"
                          and event.get("loss_mask") == 1]
        if not suffix_actions:
            raise ValueError("Sibling has no policy action after its checkpoint")
        prepared.append({"episode_id": episode_id, "branch_id": branch_id,
                         "reward": float(reward),
                         "suffix_action_event_sha256s": suffix_actions})

    rewards = [member["reward"] for member in prepared]
    total = math.fsum(rewards)
    for index, member in enumerate(prepared):
        # The other siblings form an action-independent local baseline.
        member["advantage"] = member["reward"] - (
            total - member["reward"]) / (expected_siblings - 1)
    prepared.sort(key=lambda member: member["branch_id"])
    return {"kind": "realworld_sibling_rloo_suffix_v1",
            "trainer_ready": False,
            "task_id": records[0]["task_id"],
            "task_sha256": records[0]["task_sha256"],
            "reward_contract_sha256": records[0]["reward_contract_sha256"],
            "policy_id": current_policy_id,
            "prepared_as_of": as_of.isoformat(),
            "is_fixture": records[0]["is_fixture"],
            "parent_episode_id": records[0]["branch_lineage"]["parent_episode_id"],
            "checkpoint_id": records[0]["branch_lineage"]["checkpoint_id"],
            "checkpoint_kind": records[0]["branch_lineage"]["checkpoint_kind"],
            "prefix_event_sha256": records[0]["branch_lineage"]["prefix_event_sha256"],
            "shared_prefix_visible_event_sha256s": list(prefix_hashes),
            "members": prepared,
            "loss_scope": "post_checkpoint_policy_actions_only",
            "requires_before_optimizer": ["token_ids", "behavior_token_logprobs",
                                           "token_level_observation_masks", "policy_update"]}
