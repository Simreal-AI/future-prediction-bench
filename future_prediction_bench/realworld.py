"""Task-agnostic, delayed-verification environment for real-world agent work.

The trusted adapter owns task effects and outcome verification. This module only
freezes the task contract, meters interaction, and keeps an auditable boundary
between policy actions, visible observations, and private verifier results.
It does not execute code, browse the web, or infer success from tool output.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import sqlite3
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path


SCHEMA_VERSION = "realworld-0.1"
_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class AdapterInfrastructureError(RuntimeError):
    """Trusted infrastructure failed; do not assign a policy failure reward."""


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _timestamp(value, field):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a timezone-aware ISO 8601 string")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError as exc:
        raise ValueError(f"{field} must be a timezone-aware ISO 8601 string") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include an explicit timezone")
    return parsed.astimezone(timezone.utc)


def _positive_number(value, field, *, maximum):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite number")
    if not 0 < value <= maximum:
        raise ValueError(f"{field} must be in (0, {maximum}]")
    return float(value)


def _nonblank(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonblank string")
    return value


def validate_task(task):
    """Return a detached, digest-stamped task; no model controls its contract."""
    required = {"schema_version", "task_id", "event_id", "cluster_id", "split", "prompt",
                "issued_at", "action_deadline", "outcome_not_before", "verify_after",
                "tool_manifest", "reward_contract", "budgets", "is_fixture"}
    optional = {"adapter_id", "adapter_version", "metadata"}
    if not isinstance(task, dict) or not required <= task.keys() or task.keys() - required - optional:
        raise ValueError("Invalid real-world task fields")
    if task["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Invalid real-world task schema version")
    for field in ("task_id", "event_id", "cluster_id", "prompt"):
        _nonblank(task[field], field)
    for field in ("adapter_id", "adapter_version"):
        if field in task:
            _nonblank(task[field], field)
    if task["split"] not in {"train", "dev", "test"}:
        raise ValueError("split must be train, dev, or test")
    if type(task["is_fixture"]) is not bool:
        raise ValueError("is_fixture must be boolean")
    issued = _timestamp(task["issued_at"], "issued_at")
    deadline = _timestamp(task["action_deadline"], "action_deadline")
    outcome = _timestamp(task["outcome_not_before"], "outcome_not_before")
    verify = _timestamp(task["verify_after"], "verify_after")
    if not (issued < deadline and issued <= outcome <= verify):
        raise ValueError("Expected issued_at < action_deadline and issued_at <= outcome_not_before <= verify_after")
    tools = task["tool_manifest"]
    if not isinstance(tools, list) or not tools:
        raise ValueError("tool_manifest must be a nonempty list")
    names = []
    for tool in tools:
        if not isinstance(tool, dict) or not {"name", "description"} <= tool.keys() or tool.keys() - {"name", "description", "parameters"}:
            raise ValueError("Invalid tool manifest entry")
        if not isinstance(tool["name"], str) or not _NAME.fullmatch(tool["name"]):
            raise ValueError("Invalid tool name")
        _nonblank(tool["description"], "tool description")
        if "parameters" in tool and not isinstance(tool["parameters"], dict):
            raise ValueError("tool parameters must be an object")
        names.append(tool["name"])
    if len(names) != len(set(names)):
        raise ValueError("Tool names must be unique")
    contract = task["reward_contract"]
    if not isinstance(contract, dict) or not {"id", "description", "min_reward", "max_reward"} <= contract.keys():
        raise ValueError("Invalid reward contract")
    _nonblank(contract["id"], "reward contract id")
    _nonblank(contract["description"], "reward contract description")
    lower, upper = contract["min_reward"], contract["max_reward"]
    if (any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in (lower, upper)) or not -1 <= lower <= upper <= 1):
        raise ValueError("Reward bounds must be finite and within [-1, 1]")
    budgets = task["budgets"]
    if not isinstance(budgets, dict) or not {"max_actions", "max_wall_seconds"} <= budgets.keys() or budgets.keys() - {"max_actions", "max_wall_seconds", "verification_cooldown_seconds", "max_verifications"}:
        raise ValueError("Invalid resource budgets")
    if type(budgets["max_actions"]) is not int or not 1 <= budgets["max_actions"] <= 1000:
        raise ValueError("max_actions must be an integer in [1, 1000]")
    _positive_number(budgets["max_wall_seconds"], "max_wall_seconds", maximum=86400)
    cooldown = budgets.get("verification_cooldown_seconds", 0)
    if isinstance(cooldown, bool) or not isinstance(cooldown, (int, float)) or not math.isfinite(cooldown) or not 0 <= cooldown <= 86400:
        raise ValueError("verification_cooldown_seconds must be in [0, 86400]")
    count = budgets.get("max_verifications", 100)
    if type(count) is not int or not 1 <= count <= 1000:
        raise ValueError("max_verifications must be an integer in [1, 1000]")
    if "metadata" in task and not isinstance(task["metadata"], dict):
        raise ValueError("metadata must be an object")
    frozen = copy.deepcopy(task)
    try:
        _canonical(frozen)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Task must be finite JSON data") from exc
    frozen["reward_contract_sha256"] = _digest(frozen["reward_contract"])
    frozen["task_sha256"] = _digest(frozen)
    return frozen


class RealWorldTaskRegistry:
    """Persist frozen tasks and keep related work out of different splits."""

    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=30)
        self.db.execute("""CREATE TABLE IF NOT EXISTS realworld_tasks (
            task_id TEXT PRIMARY KEY, event_id TEXT NOT NULL, cluster_id TEXT NOT NULL,
            split TEXT NOT NULL, task_sha256 TEXT NOT NULL, task_json TEXT NOT NULL
        )""")
        self.db.execute("CREATE INDEX IF NOT EXISTS realworld_event_idx ON realworld_tasks(event_id)")
        self.db.execute("CREATE INDEX IF NOT EXISTS realworld_cluster_idx ON realworld_tasks(cluster_id)")
        self.db.commit()

    def register(self, task):
        frozen = validate_task(task)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute("SELECT task_sha256 FROM realworld_tasks WHERE task_id=?",
                                  (frozen["task_id"],)).fetchone()
            if row:
                if row[0] != frozen["task_sha256"]:
                    raise ValueError("A registered real-world task cannot change")
                self.db.commit()
                return frozen["task_sha256"]
            conflicts = self.db.execute(
                "SELECT task_id, split FROM realworld_tasks WHERE event_id=? OR cluster_id=?",
                (frozen["event_id"], frozen["cluster_id"])).fetchall()
            if any(split != frozen["split"] for _, split in conflicts):
                raise ValueError("Related real-world tasks cannot cross train/dev/test splits")
            self.db.execute("INSERT INTO realworld_tasks VALUES (?,?,?,?,?,?)",
                            (frozen["task_id"], frozen["event_id"], frozen["cluster_id"],
                             frozen["split"], frozen["task_sha256"], _canonical(frozen)))
            self.db.commit()
            return frozen["task_sha256"]
        except Exception:
            self.db.rollback()
            raise

    def get(self, task_id):
        row = self.db.execute("SELECT task_json FROM realworld_tasks WHERE task_id=?", (task_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def close(self):
        self.db.close()


class RealWorldEnv:
    """In-memory host boundary; persistence and scheduling remain host concerns.

    Adapter methods are `reset(task, *, now)`, `step(action, *, now)`,
    `verify(*, now)`, and `get_state()`. Adapter `step` returns an object with
    `observation` (dict) and `terminated` (bool). The adapter must mark its own
    final action before returning `terminated=True`. Its verifier returns
    pending/resolved/void. Only the trusted host may call `verify`/`get_state`.
    """

    def __init__(self, task, adapter, *, clock=None, monotonic_clock=None):
        self._task = validate_task(task)
        if any(not callable(getattr(adapter, name, None)) for name in ("reset", "step", "verify", "get_state")):
            raise ValueError("Adapter must implement reset, step, verify, and get_state")
        self.adapter = adapter
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.monotonic_clock = monotonic_clock or time.monotonic
        self.status = "created"
        self.policy_id = None
        self.episode_id = None
        self.events = []
        self.reward = None
        self.evidence = None
        self.submitted_at = None
        self.outcome_available_at = None
        self.resolved_at = None
        self.next_verify_at = None
        self.started_at = None
        self._started_mono = None
        self.actions_used = 0
        self.verifications_used = 0
        self.pending_verifications = 0
        self.reset_seconds = 0.0
        self.action_seconds = 0.0
        self.verifier_seconds = 0.0
        self.reason = None
        self.branch_lineage = None
        self._opening_observation = None
        self._branch_checkpoints = {}

    @property
    def task(self):
        """Return a detached copy; the registered contract cannot be edited."""
        return copy.deepcopy(self._task)

    def _now(self):
        value = self.clock()
        if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("clock must return a timezone-aware datetime")
        return value.astimezone(timezone.utc)

    def _record(self, kind, payload, at, *, origin, visible, loss_mask):
        if self.events and at < _timestamp(self.events[-1]["at"], "prior event time"):
            raise ValueError("Audit events must be chronological")
        event = {"sequence": len(self.events), "kind": kind, "origin": origin,
                 "visible_to_policy": visible, "loss_mask": loss_mask,
                 "at": at.isoformat(), "payload": copy.deepcopy(payload)}
        event["sha256"] = _digest(event)
        self.events.append(event)

    def _public_task(self):
        fields = ("schema_version", "task_id", "prompt", "issued_at", "action_deadline",
                  "tool_manifest", "reward_contract", "budgets", "is_fixture",
                  "task_sha256", "reward_contract_sha256")
        return {key: copy.deepcopy(self._task[key]) for key in fields}

    def _elapsed(self):
        return max(0.0, self.monotonic_clock() - self._started_mono)

    def _action_expiry(self, now):
        if now >= _timestamp(self._task["action_deadline"], "action_deadline"):
            return "action_deadline_reached"
        if self._elapsed() >= self._task["budgets"]["max_wall_seconds"]:
            return "wall_time_budget_exhausted"
        return None

    def reset(self, policy_id):
        if self.status != "created":
            raise ValueError("An episode can only be reset once")
        if not isinstance(policy_id, str) or not policy_id.strip():
            raise ValueError("policy_id is required")
        now = self._now()
        if not _timestamp(self._task["issued_at"], "issued_at") <= now < _timestamp(self._task["action_deadline"], "action_deadline"):
            raise ValueError("Outside the action window")
        self.policy_id = policy_id
        self.episode_id = uuid.uuid4().hex
        self.started_at = now.isoformat()
        self._started_mono = self.monotonic_clock()
        before = self.monotonic_clock()
        try:
            observation = self.adapter.reset(copy.deepcopy(self._task), now=now)
            if not isinstance(observation, dict):
                raise ValueError("Adapter reset must return an observation object")
            _canonical(observation)
        except Exception as exc:
            self.status, self.reason = "setup_error", "adapter_reset_error"
            self._record("setup_error", {"error_type": type(exc).__name__}, self._now(),
                         origin="trusted_host", visible=False, loss_mask=0)
            raise ValueError("Adapter reset failed") from exc
        finally:
            self.reset_seconds += max(0.0, self.monotonic_clock() - before)
        if self._action_expiry(self._now()):
            self.status, self.reason = "interrupted", "reset_exceeded_action_window"
            self._record("observation", {"status": "interrupted", "reason": self.reason}, self._now(),
                         origin="adapter", visible=True, loss_mask=0)
            return {"task": self._public_task(), "observation": {"status": "interrupted", "reason": self.reason},
                    "episode_id": self.episode_id, "status": self.status}
        self.status = "active"
        self._record("observation", observation, self._now(), origin="adapter", visible=True, loss_mask=0)
        self._opening_observation = copy.deepcopy(observation)
        return self.opening_observation()

    def opening_observation(self):
        """Return the normal policy-visible opening, without trusted branch metadata."""
        if self._opening_observation is None:
            raise ValueError("Episode has no active opening observation")
        return {"task": self._public_task(), "observation": copy.deepcopy(self._opening_observation),
                "episode_id": self.episode_id, "status": self.status}

    def create_branch_checkpoint(self):
        """Trusted-host operation; freeze an adapter state at an action boundary.

        The adapter is responsible for publishing a verified filesystem or
        full-VM snapshot. This core binds it to the frozen task and audit
        prefix. It never offers a checkpoint action to the policy.
        """
        if self.status != "active" or self._action_expiry(self._now()):
            raise ValueError("Checkpoint requires an active episode within its budget")
        create = getattr(self.adapter, "create_branch_checkpoint", None)
        if not callable(create):
            raise ValueError("Adapter does not support branch checkpoints")
        snapshot = create()
        if not isinstance(snapshot, dict) or not isinstance(snapshot.get("artifact_binding"), dict):
            raise ValueError("Adapter returned an invalid branch checkpoint")
        kind = snapshot.get("checkpoint_kind", "filesystem_snapshot_v1")
        if kind == "filesystem_snapshot_v1":
            if (not {"snapshot_path", "workspace_sha256", "image_sha256"} <= snapshot.keys()
                    or not isinstance(snapshot["snapshot_path"], str)
                    or not snapshot["snapshot_path"]
                    or not isinstance(snapshot["workspace_sha256"], str)
                    or not re.fullmatch(r"[0-9a-f]{64}", snapshot["workspace_sha256"])
                    or not isinstance(snapshot["image_sha256"], str)
                    or not re.fullmatch(r"sha256:[0-9a-f]{64}", snapshot["image_sha256"])
                    or snapshot["image_sha256"] != snapshot["artifact_binding"].get("image_sha256")):
                raise ValueError("Adapter returned an invalid filesystem checkpoint")
        elif kind == "full_vm_state_qcow2_v1":
            if (not {"snapshot_tag", "snapshot_disk_sha256"} <= snapshot.keys()
                    or not isinstance(snapshot["snapshot_tag"], str)
                    or not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", snapshot["snapshot_tag"])
                    or not isinstance(snapshot["snapshot_disk_sha256"], str)
                    or not re.fullmatch(r"[0-9a-f]{64}", snapshot["snapshot_disk_sha256"])
                    or snapshot["artifact_binding"].get("runtime_kind") != "qemu_hvf_full_vm_qcow2_v1"):
                raise ValueError("Adapter returned an invalid full-VM checkpoint")
        else:
            raise ValueError("Adapter returned an unknown branch checkpoint kind")
        try:
            _canonical(snapshot)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Branch checkpoint must be finite JSON data") from exc
        if (self._task.get("metadata", {}).get("artifact_binding") is not None
                and snapshot["artifact_binding"] != self._task["metadata"]["artifact_binding"]):
            raise ValueError("Branch checkpoint differs from frozen task artifacts")
        if self._action_expiry(self._now()):
            raise ValueError("Checkpoint exceeded the episode action window")
        checkpoint_id = uuid.uuid4().hex
        ref = {**copy.deepcopy(snapshot), "checkpoint_id": checkpoint_id,
               "task_sha256": self._task["task_sha256"],
               "parent_episode_id": self.episode_id,
               "prefix_event_sha256": _digest(self.events),
               "actions_used": self.actions_used,
               "elapsed_seconds": self._elapsed(),
               "policy_id": self.policy_id}
        self._branch_checkpoints[checkpoint_id] = {
            "ref": copy.deepcopy(ref), "events": copy.deepcopy(self.events),
            "started_at": self.started_at, "started_mono": self._started_mono,
            "reset_seconds": self.reset_seconds,
            "action_seconds": self.action_seconds,
            "lineage": copy.deepcopy(self.branch_lineage)}
        return copy.deepcopy(ref)

    def fork_from_checkpoint(self, checkpoint_ref, branch_adapter, *, branch_id):
        """Trusted-host branch with the parent's budget and audit prefix.

        The adapter defines whether this is filesystem-only or full-VM state;
        it must start an independent sandbox from the frozen snapshot.
        """
        if self.status != "active" or self._action_expiry(self._now()):
            raise ValueError("Fork requires an active parent within its budget")
        if (not isinstance(branch_id, str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", branch_id)):
            raise ValueError("Invalid branch_id")
        if not isinstance(checkpoint_ref, dict):
            raise ValueError("Invalid branch checkpoint reference")
        checkpoint_id = checkpoint_ref.get("checkpoint_id")
        if not isinstance(checkpoint_id, str):
            raise ValueError("Invalid branch checkpoint reference")
        saved = self._branch_checkpoints.get(checkpoint_id)
        if saved is None or checkpoint_ref != saved["ref"]:
            raise ValueError("Branch checkpoint was not issued by this episode")
        ref = saved["ref"]
        if (ref["task_sha256"] != self._task["task_sha256"]
                or ref["parent_episode_id"] != self.episode_id
                or ref["policy_id"] != self.policy_id
                or ref["prefix_event_sha256"] != _digest(saved["events"])
                or ref["actions_used"] > self._task["budgets"]["max_actions"]):
            raise ValueError("Branch checkpoint lineage is invalid")
        reset_from_checkpoint = getattr(branch_adapter, "reset_from_checkpoint", None)
        if not callable(reset_from_checkpoint):
            raise ValueError("Branch adapter cannot reset from checkpoint")
        raw_task = {key: copy.deepcopy(value) for key, value in self._task.items()
                    if key not in {"task_sha256", "reward_contract_sha256"}}
        branch = RealWorldEnv(raw_task, branch_adapter,
                              clock=self.clock, monotonic_clock=self.monotonic_clock)
        if branch._task["task_sha256"] != self._task["task_sha256"]:
            raise ValueError("Branch task binding changed")
        branch.policy_id = self.policy_id
        branch.episode_id = uuid.uuid4().hex
        branch.started_at = saved["started_at"]
        branch._started_mono = saved["started_mono"]
        branch.actions_used = ref["actions_used"]
        branch.reset_seconds = saved["reset_seconds"]
        branch.action_seconds = saved["action_seconds"]
        branch.events = copy.deepcopy(saved["events"])
        now = branch._now()
        if branch.events and now < _timestamp(branch.events[-1]["at"], "prior event time"):
            raise ValueError("Clock moved before the branch checkpoint")
        branch.branch_lineage = {"branch_id": branch_id,
                                 "parent_episode_id": self.episode_id,
                                 "checkpoint_id": ref["checkpoint_id"],
                                 "prefix_event_sha256": ref["prefix_event_sha256"],
                                 "prefix_visible_event_count": sum(
                                     event["visible_to_policy"] for event in saved["events"]),
                                 "checkpoint_kind": ref.get("checkpoint_kind", "filesystem_snapshot_v1"),
                                 "parent_lineage": copy.deepcopy(saved["lineage"])}
        if "workspace_sha256" in ref:
            branch.branch_lineage["workspace_sha256"] = ref["workspace_sha256"]
        if "snapshot_disk_sha256" in ref:
            branch.branch_lineage["snapshot_disk_sha256"] = ref["snapshot_disk_sha256"]
        before = branch.monotonic_clock()
        try:
            observation = reset_from_checkpoint(copy.deepcopy(branch._task), copy.deepcopy(ref), now=now)
            if not isinstance(observation, dict):
                raise ValueError("Branch adapter reset must return an observation object")
            _canonical(observation)
            if branch._action_expiry(branch._now()):
                raise ValueError("Branch setup exceeded the action window")
        except Exception as exc:
            close = getattr(branch_adapter, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
            raise ValueError("Branch adapter reset failed") from exc
        finally:
            branch.reset_seconds += max(0.0, branch.monotonic_clock() - before)
        branch.status = "active"
        branch._record("branch", copy.deepcopy(branch.branch_lineage), branch._now(),
                       origin="trusted_host", visible=False, loss_mask=0)
        branch._record("observation", observation, branch._now(),
                       origin="adapter", visible=True, loss_mask=0)
        branch._opening_observation = copy.deepcopy(observation)
        return branch

    def step(self, action):
        if self.status != "active":
            raise ValueError("step requires an active episode")
        now = self._now()
        expiry = self._action_expiry(now)
        if expiry:
            self.status, self.reason = "missed", expiry
            return {"observation": {"status": "missed", "reason": expiry}, "reward": None,
                    "terminated": True, "info": {"episode_id": self.episode_id, "status": self.status}}
        if not isinstance(action, dict):
            raise ValueError("action must be an object")
        try:
            _canonical(action)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("action must be finite JSON data") from exc
        self.actions_used += 1
        self._record("action", action, now, origin="policy", visible=True, loss_mask=1)
        if action.get("action") not in {item["name"] for item in self._task["tool_manifest"]}:
            result = {"status": "error", "reason": "tool_not_available"}
            terminal = False
        else:
            before = self.monotonic_clock()
            try:
                transition = self.adapter.step(copy.deepcopy(action), now=now)
                if (not isinstance(transition, dict) or set(transition) != {"observation", "terminated"}
                        or not isinstance(transition["observation"], dict)
                        or type(transition["terminated"]) is not bool):
                    raise ValueError("Invalid adapter transition")
                _canonical(transition["observation"])
                result = copy.deepcopy(transition["observation"])
                terminal = transition["terminated"]
            except AdapterInfrastructureError as exc:
                result = {"status": "interrupted", "reason": "adapter_infrastructure_error",
                          "error_type": type(exc).__name__}
                self.status, self.reason = "interrupted", "adapter_infrastructure_error"
                terminal = False
            except Exception as exc:
                result = {"status": "error", "reason": "adapter_error", "error_type": type(exc).__name__}
                terminal = False
            finally:
                self.action_seconds += max(0.0, self.monotonic_clock() - before)
        observed = self._now()
        expiry = self._action_expiry(observed)
        if self.status == "interrupted":
            pass
        elif expiry:
            # A late tool result cannot be used as a policy observation.
            result = {"status": "missed", "reason": expiry, "late_result_discarded": True}
            self.status, self.reason = "missed", expiry
        elif terminal:
            self.status = "pending"
            self.submitted_at = observed.isoformat()
        elif self.actions_used >= self._task["budgets"]["max_actions"]:
            self.status, self.reason = "missed", "action_budget_exhausted"
        self._record("observation", result, observed, origin="adapter", visible=True, loss_mask=0)
        return {"observation": result, "reward": None, "terminated": self.status != "active",
                "info": {"episode_id": self.episode_id, "status": self.status,
                         "actions_used": self.actions_used, "actions_remaining": max(0, self._task["budgets"]["max_actions"] - self.actions_used),
                         "awaiting_verification": self.status == "pending"}}

    def _pending(self, reason):
        return {"status": "pending", "reward": None, "reason": reason,
                "next_verify_at": self.next_verify_at.isoformat() if self.next_verify_at else None}

    def verify(self):
        """Trusted host call. No verifier output ever becomes a policy observation."""
        if self.status not in {"pending", "graded", "void"}:
            raise ValueError("Verification requires a submitted episode")
        if self.status in {"graded", "void"}:
            result = {"status": self.status, "reward": self.reward,
                      "evidence": copy.deepcopy(self.evidence), "available_at": self.outcome_available_at}
            if self.status == "void":
                result["reason"] = self.reason
            return result
        now = self._now()
        due = max(_timestamp(self._task["verify_after"], "verify_after"),
                  _timestamp(self.submitted_at, "submitted_at"))
        if now < due:
            self.next_verify_at = due
            return self._pending("not_due")
        if self.next_verify_at and now < self.next_verify_at:
            return self._pending("cooldown")
        if self.verifications_used >= self._task["budgets"].get("max_verifications", 100):
            return self._pending("verification_attempt_budget_exhausted")
        self.verifications_used += 1
        before = self.monotonic_clock()
        try:
            proposal = self.adapter.verify(now=now)
            if not isinstance(proposal, dict) or proposal.get("status") not in {"pending", "resolved", "void"}:
                raise ValueError("Invalid verifier response")
            _canonical(proposal)
            if proposal["status"] == "resolved":
                if set(proposal) != {"status", "reward", "evidence", "available_at"}:
                    raise ValueError("Invalid resolved verifier fields")
                reward = proposal["reward"]
                if (isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward)
                        or not self._task["reward_contract"]["min_reward"] <= reward <= self._task["reward_contract"]["max_reward"]):
                    raise ValueError("Verified reward is outside the frozen contract")
                available = _timestamp(proposal["available_at"], "available_at")
                if not max(_timestamp(self._task["outcome_not_before"], "outcome_not_before"),
                           _timestamp(self.submitted_at, "submitted_at")) <= available <= self._now():
                    raise ValueError("Verifier outcome was not available at the claimed time")
                if not isinstance(proposal["evidence"], dict) or not proposal["evidence"]:
                    raise ValueError("Resolved verifier evidence is required")
                self.status, self.reward, self.evidence = "graded", float(reward), copy.deepcopy(proposal["evidence"])
                self.outcome_available_at = available.isoformat()
                self.resolved_at = self._now().isoformat()
                self.next_verify_at = None
            elif proposal["status"] == "void":
                if (set(proposal) != {"status", "reason", "evidence"} or not isinstance(proposal["evidence"], dict)
                        or not proposal["evidence"]):
                    raise ValueError("Void requires a reason and evidence")
                self.reason = _nonblank(proposal["reason"], "void reason")
                self.status, self.evidence = "void", copy.deepcopy(proposal["evidence"])
                self.outcome_available_at = self._now().isoformat()
                self.resolved_at = self._now().isoformat()
                self.next_verify_at = None
            else:
                if set(proposal) - {"status", "reason", "retry_after"}:
                    raise ValueError("Pending verification cannot contain outcomes or reward")
                self.pending_verifications += 1
                cooldown = float(self._task["budgets"].get("verification_cooldown_seconds", 0))
                retry = now + timedelta(seconds=cooldown)
                if "retry_after" in proposal:
                    retry = max(retry, _timestamp(proposal["retry_after"], "retry_after"))
                self.next_verify_at = retry
                self.reason = _nonblank(proposal.get("reason", "not_yet_available"), "pending reason")
            audit_payload = copy.deepcopy(proposal)
        except Exception as exc:
            self.pending_verifications += 1
            self.next_verify_at = now + timedelta(seconds=float(self._task["budgets"].get("verification_cooldown_seconds", 0)))
            self.reason = "verifier_error"
            audit_payload = {"status": "error", "error_type": type(exc).__name__}
        finally:
            self.verifier_seconds += max(0.0, self.monotonic_clock() - before)
        self._record("verification", audit_payload, self._now(), origin="trusted_verifier", visible=False, loss_mask=0)
        if self.status == "graded":
            return {"status": "graded", "reward": self.reward, "evidence": copy.deepcopy(self.evidence),
                    "available_at": self.outcome_available_at}
        if self.status == "void":
            return {"status": "void", "reward": None, "reason": self.reason,
                    "evidence": copy.deepcopy(self.evidence), "available_at": self.outcome_available_at}
        return self._pending(self.reason)

    def get_state(self):
        """Trusted diagnostics. Never forward this object to the policy."""
        return {"episode_id": self.episode_id, "policy_id": self.policy_id,
                "task": copy.deepcopy(self._task), "status": self.status, "reason": self.reason,
                "reward": self.reward, "evidence": copy.deepcopy(self.evidence),
                "started_at": self.started_at, "submitted_at": self.submitted_at,
                "outcome_available_at": self.outcome_available_at, "resolved_at": self.resolved_at,
                "next_verify_at": self.next_verify_at.isoformat() if self.next_verify_at else None,
                "branch_lineage": copy.deepcopy(self.branch_lineage),
                "metrics": {"actions_used": self.actions_used, "verifications_used": self.verifications_used,
                            "pending_verifications": self.pending_verifications,
                            "reset_seconds": self.reset_seconds, "action_seconds": self.action_seconds,
                            "verifier_seconds": self.verifier_seconds},
                "events": copy.deepcopy(self.events),
                "adapter_state": copy.deepcopy(self.adapter.get_state())}

    def export_trajectory(self):
        """Training audit only, never a tokenized or optimizer-ready batch."""
        if self.status != "graded" or self._task["split"] != "train":
            return None
        return {"schema_version": SCHEMA_VERSION, "record_type": "realworld_text_trajectory",
                "trainer_ready": False, "is_fixture": self._task["is_fixture"],
                "task_id": self._task["task_id"], "event_id": self._task["event_id"],
                "cluster_id": self._task["cluster_id"], "task_sha256": self._task["task_sha256"],
                "reward_contract_sha256": self._task["reward_contract_sha256"],
                "policy_id": self.policy_id, "episode_id": self.episode_id,
                "branch_lineage": copy.deepcopy(self.branch_lineage),
                "submitted_at": self.submitted_at, "available_at": self.outcome_available_at,
                "resolved_at": self.resolved_at,
                "reward": self.reward,
                "events": [copy.deepcopy(event) for event in self.events if event["visible_to_policy"]]}
