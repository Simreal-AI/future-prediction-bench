"""Durable, same-host verification handoff for submitted Docker episodes.

The actor is responsible for a normal ``RealWorldEnv`` reset/actions/submit.
Only a trusted coordinator may enqueue its frozen submission. SQLite leases
survive verifier process death; a lease token fences publication of reward.
Verification can run more than once after a crash, but one job can publish at
most one final reward. No model inference, token logprobs, trainer or remote
service is implied by this queue.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .coding_env import DockerCodingAdapter, _workspace_digest
from .realworld import RealWorldEnv, RealWorldTaskRegistry, _timestamp, validate_task


class LeaseLost(RuntimeError):
    """A dead/stale worker must never publish reward after its lease expired."""


class QueueFull(RuntimeError):
    """Bounded submitted-work backlog requires producer backpressure."""


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _sha(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _code_binding():
    """Refuse to resume a submission under changed trusted verifier code."""
    sources = {"queue": Path(__file__),
               "docker_adapter": Path(DockerCodingAdapter.verify.__code__.co_filename),
               "realworld_contract": Path(validate_task.__code__.co_filename)}
    if any(path.is_symlink() or not path.is_file() for path in sources.values()):
        raise ValueError("Trusted verifier source identity is unavailable")
    return {name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in sources.items()}


def _plain_task(frozen):
    task = copy.deepcopy(frozen)
    task_sha = task.pop("task_sha256", None)
    reward_sha = task.pop("reward_contract_sha256", None)
    checked = validate_task(task)
    if checked["task_sha256"] != task_sha or checked["reward_contract_sha256"] != reward_sha:
        raise ValueError("Frozen task digest differs from its contract")
    return task, checked


def _separate(*paths):
    roots = [Path(path).resolve() for path in paths]
    for index, left in enumerate(roots):
        if any(left.is_relative_to(right) or right.is_relative_to(left)
               for right in roots[index + 1:]):
            raise ValueError("Trusted queue, verifier, seed and submission roots must be disjoint")


def _restore_submitted_env(task, adapter, state):
    """Restore the trusted state needed by RealWorldEnv.verify, not actor runtime.

    The candidate container was already stopped at submit. This reconstructs
    the host's metering and audit state from its durable submitted record so
    verification uses the canonical task clock/reward contract again.
    """
    env = RealWorldEnv(task, adapter)
    metrics = state["metrics"]
    if (state["task"] != env.task or state["status"] != "pending"
            or state["reward"] is not None or state["evidence"] is not None
            or state["adapter_state"]["submitted"] is not True
            or not isinstance(state["events"], list)
            or not isinstance(state["episode_id"], str)
            or not isinstance(state["policy_id"], str)
            or not isinstance(state["submitted_at"], str)):
        raise ValueError("Invalid submitted environment state")
    for key in ("actions_used", "verifications_used", "pending_verifications"):
        if type(metrics.get(key)) is not int or metrics[key] < 0:
            raise ValueError("Invalid submitted environment metrics")
    for key in ("reset_seconds", "action_seconds", "verifier_seconds"):
        number = metrics.get(key)
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or number < 0:
            raise ValueError("Invalid submitted environment metrics")
    env.status = "pending"
    env.policy_id = state["policy_id"]
    env.episode_id = state["episode_id"]
    env.started_at = state["started_at"]
    env.submitted_at = state["submitted_at"]
    env.events = copy.deepcopy(state["events"])
    env.actions_used = metrics["actions_used"]
    env.verifications_used = metrics["verifications_used"]
    env.pending_verifications = metrics["pending_verifications"]
    env.reset_seconds = float(metrics["reset_seconds"])
    env.action_seconds = float(metrics["action_seconds"])
    env.verifier_seconds = float(metrics["verifier_seconds"])
    env.reason = state["reason"]
    env.branch_lineage = copy.deepcopy(state["branch_lineage"])
    env.next_verify_at = (_timestamp(state["next_verify_at"], "next_verify_at")
                          if state["next_verify_at"] is not None else None)
    env._started_mono = time.monotonic()
    return env


@dataclass(frozen=True)
class VerificationLease:
    job_id: str
    token: str
    attempt: int
    expires_at: float
    payload: dict
    progress_state: dict | None = None


class DurableDockerRolloutQueue:
    """SQLite WAL queue for *already submitted*, frozen Docker coding work.

    ``enqueue_submitted`` is called by the trusted actor coordinator after
    ``submit``. The queue path is host-owned and disjoint from the candidate
    and hidden verifier trees. Each worker opens its own connection. A live
    worker renews its lease during blocking Docker cases; a dead worker's
    lease becomes claimable after expiry. CAS on the random token plus expiry
    makes reward publication at most once, even if cases are re-executed.
    The optional queue-local revision fence serializes trusted revision
    updates and final reward publication in this same SQLite database.
    """

    def __init__(self, path, *, max_outstanding=128):
        if type(max_outstanding) is not int or not 1 <= max_outstanding <= 10000:
            raise ValueError("max_outstanding must be an integer in [1, 10000]")
        self.max_outstanding = max_outstanding
        self.path = Path(path).absolute()
        if self.path.is_symlink() or self.path.parent.is_symlink():
            raise ValueError("Queue path and parent must not be symlinks")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path = self.path.resolve()
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def _initialize(self):
        db = self._connect()
        try:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=FULL")
            db.execute("""CREATE TABLE IF NOT EXISTS rollout_jobs (
                job_id TEXT PRIMARY KEY, episode_id TEXT NOT NULL UNIQUE,
                submission_sha256 TEXT NOT NULL UNIQUE, payload_json TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL, policy_revision TEXT NOT NULL,
                task_sha256 TEXT NOT NULL, snapshot_sha256 TEXT NOT NULL,
                progress_json TEXT NOT NULL, progress_sha256 TEXT NOT NULL,
                priority INTEGER NOT NULL, created_at REAL NOT NULL,
                due_at REAL NOT NULL, state TEXT NOT NULL,
                attempt INTEGER NOT NULL DEFAULT 0,
                lease_token TEXT, lease_owner TEXT, lease_until REAL,
                result_json TEXT, reward REAL, last_pending_reason TEXT,
                finished_at REAL,
                CHECK(state IN ('queued','leased','graded','void','exhausted','stale')),
                CHECK((state IN ('graded','void')) = (result_json IS NOT NULL)),
                CHECK((state='graded') = (reward IS NOT NULL))
            )""")
            fields = {row[1] for row in db.execute("PRAGMA table_info(rollout_jobs)")}
            schema = db.execute("""SELECT sql FROM sqlite_master
                WHERE type='table' AND name='rollout_jobs'""").fetchone()[0]
            if not {"episode_id", "submission_sha256", "progress_json",
                    "progress_sha256", "lease_token", "result_json"} <= fields or "'stale'" not in schema or "'void'" not in schema:
                raise ValueError("Incompatible durable rollout schema; create a new queue")
            db.execute("CREATE INDEX IF NOT EXISTS rollout_ready_idx ON rollout_jobs(state,due_at,priority,created_at)")
            db.execute("""CREATE TABLE IF NOT EXISTS rollout_revisions (
                policy_id TEXT PRIMARY KEY, revision TEXT NOT NULL,
                generation INTEGER NOT NULL, updated_at REAL NOT NULL
            )""")
        finally:
            db.close()

    def set_current_revision(self, policy_id, revision):
        """Trusted host declaration of current policy weights, in queue order.

        Every update and an opt-in final reward commit use ``BEGIN IMMEDIATE``
        against this database. This fences the queue's declared revision, not
        an out-of-band inference server or the actual actor weight files.
        """
        if not isinstance(policy_id, str) or not policy_id.strip() or len(policy_id) > 256:
            raise ValueError("policy_id must be a nonblank bounded string")
        if not isinstance(revision, str) or not revision.strip() or len(revision) > 256:
            raise ValueError("revision must be a nonblank bounded string")
        db = self._connect()
        try:
            db.execute("BEGIN IMMEDIATE")
            prior = db.execute("SELECT revision,generation FROM rollout_revisions WHERE policy_id=?",
                               (policy_id,)).fetchone()
            if prior is None:
                generation = 1
                db.execute("""INSERT INTO rollout_revisions
                    (policy_id,revision,generation,updated_at) VALUES (?,?,?,?)""",
                    (policy_id, revision, generation, time.time()))
            elif prior["revision"] == revision:
                generation = prior["generation"]
            else:
                generation = prior["generation"] + 1
                db.execute("""UPDATE rollout_revisions SET revision=?,generation=?,updated_at=?
                    WHERE policy_id=?""", (revision, generation, time.time(), policy_id))
            db.commit()
            return {"policy_id": policy_id, "revision": revision,
                    "generation": generation}
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def current_revision(self, policy_id):
        """Read the queue-local declaration; only a same-transaction commit fences it."""
        db = self._connect()
        try:
            row = db.execute("""SELECT revision,generation FROM rollout_revisions
                WHERE policy_id=?""", (policy_id,)).fetchone()
            return ({"policy_id": policy_id, "revision": row["revision"],
                     "generation": row["generation"]} if row is not None else None)
        finally:
            db.close()

    def enqueue_submitted(self, job_id, env, *, policy_revision, priority=5,
                          registry_path=None, require_queue_revision=False):
        """Durably hand off the host's submitted RealWorldEnv without reward.

        The coordinator remains responsible for closing the actor adapter.
        An identical repeated enqueue is idempotent; a changed payload is
        rejected. The actor-to-queue handoff is only durable after this call
        returns. A crash before it cannot be recovered by this component.
        """
        if not isinstance(job_id, str) or not job_id or len(job_id) > 128:
            raise ValueError("job_id must be a nonempty bounded string")
        if not isinstance(policy_revision, str) or not policy_revision.strip() or len(policy_revision) > 256:
            raise ValueError("policy_revision must be an explicit nonblank weight revision")
        if type(priority) is not int or not 0 <= priority <= 9:
            raise ValueError("priority must be an integer in [0, 9]")
        if type(require_queue_revision) is not bool:
            raise ValueError("require_queue_revision must be boolean")
        adapter = getattr(env, "adapter", None)
        if not isinstance(adapter, DockerCodingAdapter):
            raise ValueError("Only DockerCodingAdapter submissions are supported")
        state = env.get_state()
        if (state.get("status") != "pending" or state.get("reward") is not None
                or not state.get("submitted_at") or not adapter.submitted
                or adapter.container_name is not None):
            raise ValueError("A frozen submitted episode without reward is required")
        task, checked = _plain_task(state["task"])
        if state["task"] != checked or state["policy_id"] != env.policy_id:
            raise ValueError("Submitted state differs from frozen episode")
        binding = checked.get("metadata", {}).get("artifact_binding")
        if binding is None or binding != adapter.expected_binding or binding != adapter.artifact_binding():
            raise ValueError("Submission artifacts differ from frozen task")
        snapshot = Path(adapter.submitted_workspace)
        if snapshot.is_symlink() or not snapshot.is_dir():
            raise ValueError("Submitted snapshot is missing or linked")
        snapshot = snapshot.resolve()
        _separate(self.path.parent, adapter.seed_dir, adapter.verifier_dir,
                  adapter.workspace, snapshot)
        snapshot_sha = _workspace_digest(snapshot)
        if snapshot.name != snapshot_sha:
            raise ValueError("Submitted snapshot differs from frozen checkpoint")
        if not checked["is_fixture"]:
            if registry_path is None:
                raise ValueError("Non-fixture submissions require a persistent task registry")
            registry = Path(registry_path).resolve()
            _separate(registry.parent, self.path.parent, adapter.seed_dir,
                      adapter.verifier_dir, adapter.workspace, snapshot)
            store = RealWorldTaskRegistry(registry)
            try:
                if store.get(checked["task_id"]) != checked:
                    raise ValueError("Submission differs from registered real-world task")
            finally:
                store.close()
        due = max(_timestamp(checked["verify_after"], "verify_after"),
                  _timestamp(state["submitted_at"], "submitted_at")).timestamp()
        payload = {"schema": "submitted-docker-rollout-v1", "task": task,
                   "task_sha256": checked["task_sha256"], "submission_state": state,
                   "submission_sha256": _sha(state), "policy_revision": policy_revision,
                   "queue_revision_required": require_queue_revision,
                   "queue_revision_generation": None,
                   "snapshot_path": str(snapshot), "snapshot_sha256": snapshot_sha,
                   "actor_workspace_path": str(adapter.workspace),
                   "seed_dir": str(adapter.seed_dir), "verifier_dir": str(adapter.verifier_dir),
                   "image": adapter.image, "binding": binding,
                   "code_binding": _code_binding(),
                   "visible_check": list(adapter.visible_check),
                   "verifier_workers": adapter.verifier_workers}
        db = self._connect()
        try:
            db.execute("BEGIN IMMEDIATE")
            prior = db.execute("""SELECT payload_json,payload_sha256
                FROM rollout_jobs WHERE job_id=?""", (job_id,)).fetchone()
            if prior is None:
                if require_queue_revision:
                    current = db.execute("""SELECT revision,generation FROM rollout_revisions
                        WHERE policy_id=?""", (state["policy_id"],)).fetchone()
                    if current is None or current["revision"] != policy_revision:
                        raise ValueError("Queue current revision must match the submitted policy")
                    payload["queue_revision_generation"] = current["generation"]
                duplicate = db.execute("""SELECT job_id FROM rollout_jobs
                    WHERE episode_id=? OR submission_sha256=?""",
                    (state["episode_id"], payload["submission_sha256"])).fetchone()
                if duplicate is not None:
                    raise ValueError("Submitted episode is already assigned to another job_id")
                backlog = db.execute("""SELECT COUNT(*) FROM rollout_jobs
                    WHERE state IN ('queued','leased')""").fetchone()[0]
                if backlog >= self.max_outstanding:
                    raise QueueFull("Submitted verifier backlog is full")
                encoded, digest = _canonical(payload), _sha(payload)
                if len(encoded.encode("utf-8")) > 4_000_000:
                    raise ValueError("Submitted episode audit exceeds queue limit")
                db.execute("""INSERT INTO rollout_jobs
                    (job_id,episode_id,submission_sha256,payload_json,payload_sha256,policy_revision,task_sha256,
                     snapshot_sha256,progress_json,progress_sha256,priority,created_at,due_at,state)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,'queued')""",
                    (job_id, state["episode_id"], payload["submission_sha256"],
                     encoded, digest, policy_revision, checked["task_sha256"],
                     snapshot_sha, _canonical(state), _sha(state),
                     priority, time.time(), due))
            else:
                old = json.loads(prior["payload_json"])
                if _sha(old) != prior["payload_sha256"]:
                    raise ValueError("Durable queue payload digest mismatch")
                if old.get("queue_revision_required") != require_queue_revision:
                    raise ValueError("job_id already binds a different submission")
                payload["queue_revision_generation"] = old.get("queue_revision_generation")
                if prior["payload_sha256"] != _sha(payload):
                    raise ValueError("job_id already binds a different submission")
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
        return {"job_id": job_id, "task_sha256": checked["task_sha256"],
                "policy_revision": policy_revision, "submission_sha256": payload["submission_sha256"],
                "snapshot_sha256": snapshot_sha, "due_at": due,
                "queue_revision_generation": payload["queue_revision_generation"]}

    def claim(self, worker_id, *, lease_seconds=30.0):
        if not isinstance(worker_id, str) or not worker_id or len(worker_id) > 128:
            raise ValueError("worker_id must be a nonempty bounded string")
        if isinstance(lease_seconds, bool) or not isinstance(lease_seconds, (int, float)) or not 0.2 <= lease_seconds <= 3600:
            raise ValueError("lease_seconds must be in [0.2, 3600]")
        now, token = time.time(), uuid.uuid4().hex
        db = self._connect()
        try:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("""SELECT * FROM rollout_jobs WHERE due_at <= ? AND
                (state='queued' OR (state='leased' AND lease_until <= ?))
                ORDER BY priority,created_at,job_id LIMIT 1""", (now, now)).fetchone()
            if row is None:
                db.commit()
                return None
            payload = json.loads(row["payload_json"])
            if (_sha(payload) != row["payload_sha256"]
                    or payload.get("submission_sha256") != row["submission_sha256"]
                    or payload.get("submission_state", {}).get("episode_id") != row["episode_id"]
                    or payload.get("policy_revision") != row["policy_revision"]
                    or payload.get("task_sha256") != row["task_sha256"]
                    or payload.get("snapshot_sha256") != row["snapshot_sha256"]):
                raise ValueError("Durable queue payload binding mismatch")
            progress = json.loads(row["progress_json"])
            if _sha(progress) != row["progress_sha256"]:
                raise ValueError("Durable verifier progress digest mismatch")
            expiry = now + float(lease_seconds)
            db.execute("""UPDATE rollout_jobs SET state='leased',attempt=attempt+1,
                lease_token=?,lease_owner=?,lease_until=? WHERE job_id=?""",
                (token, worker_id, expiry, row["job_id"]))
            db.commit()
            return VerificationLease(row["job_id"], token, row["attempt"] + 1,
                                     expiry, payload, progress)
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def renew(self, lease, *, lease_seconds=30.0):
        if isinstance(lease_seconds, bool) or not isinstance(lease_seconds, (int, float)) or not 0.2 <= lease_seconds <= 3600:
            raise ValueError("lease_seconds must be in [0.2, 3600]")
        now = time.time()
        db = self._connect()
        try:
            db.execute("BEGIN IMMEDIATE")
            changed = db.execute("""UPDATE rollout_jobs SET lease_until=?
                WHERE job_id=? AND state='leased' AND lease_token=? AND lease_until>?""",
                (now + lease_seconds, lease.job_id, lease.token, now)).rowcount
            db.commit()
            return changed == 1
        finally:
            db.close()

    def _finish(self, lease, *, result=None, pending_reason=None, retry_seconds=0.0,
                progress_state=None, atomic_revision_fence=False):
        if (result is None) == (pending_reason is None):
            raise ValueError("Exactly one final result or pending reason is required")
        if type(atomic_revision_fence) is not bool:
            raise ValueError("atomic_revision_fence must be boolean")
        # The enqueue-time contract cannot be weakened by a verifier that
        # omits its optional worker flag. Keep this check at the SQL commit
        # boundary as well as in process_one's pre-verification gate.
        atomic_revision_fence = (atomic_revision_fence
                                 or lease.payload.get("queue_revision_required") is True)
        if isinstance(retry_seconds, bool) or not isinstance(retry_seconds, (int, float)) or not 0 <= retry_seconds <= 86400:
            raise ValueError("retry_seconds must be in [0, 86400]")
        progress = lease.progress_state if progress_state is None else progress_state
        if progress is None:
            progress = {}
        progress_json, progress_sha = _canonical(progress), _sha(progress)
        db = self._connect()
        try:
            db.execute("BEGIN IMMEDIATE")
            now = time.time()
            if result is not None:
                if result.get("status") == "graded":
                    state, reward = "graded", result["reward"]
                    if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
                        raise ValueError("Graded durable result requires a finite numeric reward")
                elif result.get("status") == "void":
                    state, reward = "void", None
                    if result.get("reward") is not None:
                        raise ValueError("Void durable result cannot carry a reward")
                else:
                    raise ValueError("Durable terminal result must be graded or void")
                reason, due = None, now
                committed_result = result
                if atomic_revision_fence:
                    policy_id = lease.payload["submission_state"]["policy_id"]
                    generation = lease.payload.get("queue_revision_generation")
                    current = db.execute("""SELECT revision,generation FROM rollout_revisions
                        WHERE policy_id=?""", (policy_id,)).fetchone()
                    if (lease.payload.get("queue_revision_required") is not True
                            or type(generation) is not int or generation < 1
                            or current is None
                            or current["revision"] != lease.payload["policy_revision"]
                            or current["generation"] != generation):
                        state, reward, committed_result = "stale", None, None
                        reason = "queue_revision_mismatch_at_commit"
                    else:
                        committed_result = {**result, "queue_revision_fence": {
                            "policy_id": policy_id, "revision": current["revision"],
                            "generation": current["generation"]}}
                result_json = _canonical(committed_result) if committed_result is not None else None
            else:
                state = ("exhausted" if pending_reason == "verification_attempt_budget_exhausted"
                         else "stale" if pending_reason == "queue_revision_mismatch_before_verification"
                         else "queued")
                result_json, reward = None, None
                reason, due = str(pending_reason)[:256], now + retry_seconds
            changed = db.execute("""UPDATE rollout_jobs SET state=?,result_json=?,reward=?,
                last_pending_reason=?,due_at=?,finished_at=?,progress_json=?,progress_sha256=?,lease_token=NULL,
                lease_owner=NULL,lease_until=NULL
                WHERE job_id=? AND state='leased' AND lease_token=? AND lease_until>?""",
                (state, result_json, reward, reason, due,
                 now if state in {"graded", "void", "stale", "exhausted"} else None,
                 progress_json, progress_sha, lease.job_id, lease.token, now)).rowcount
            if changed != 1:
                raise LeaseLost("Verifier lease expired or was reassigned")
            db.commit()
            return state
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def _verify(self, lease):
        payload = lease.payload
        if (payload.get("schema") != "submitted-docker-rollout-v1"
                or payload.get("task_sha256") is None
                or payload.get("policy_revision") is None):
            raise ValueError("Invalid durable submission")
        if payload.get("code_binding") != _code_binding():
            raise ValueError("trusted_verifier_code_changed")
        task, frozen = _plain_task(payload["submission_state"]["task"])
        progress = lease.progress_state
        if (task != payload["task"] or frozen["task_sha256"] != payload["task_sha256"]
                or _sha(payload["submission_state"]) != payload["submission_sha256"]
                or payload["submission_state"]["status"] != "pending"
                or payload["submission_state"]["reward"] is not None
                or not isinstance(progress, dict)
                or progress.get("task") != frozen
                or progress.get("episode_id") != payload["submission_state"]["episode_id"]
                or progress.get("policy_id") != payload["submission_state"]["policy_id"]
                or progress.get("submitted_at") != payload["submission_state"]["submitted_at"]):
            raise ValueError("Durable submission differs from its frozen state")
        snapshot = Path(payload["snapshot_path"])
        if snapshot.is_symlink() or not snapshot.is_dir():
            raise ValueError("submitted_snapshot_missing_or_linked")
        snapshot = snapshot.resolve()
        _separate(self.path.parent, payload["seed_dir"], payload["verifier_dir"],
                  payload["actor_workspace_path"], snapshot)
        if snapshot.name != payload["snapshot_sha256"] or _workspace_digest(snapshot) != payload["snapshot_sha256"]:
            raise ValueError("submitted_snapshot_changed")
        adapter = DockerCodingAdapter(
            seed_dir=payload["seed_dir"], verifier_dir=payload["verifier_dir"],
            image=payload["image"], output_root=self.path.parent / "verifier-reopen" / lease.token,
            visible_check=payload["visible_check"], verifier_workers=payload["verifier_workers"])
        try:
            binding = adapter.artifact_binding()
            if binding != payload["binding"] or binding != frozen["metadata"]["artifact_binding"]:
                raise ValueError("submitted_artifact_binding_changed")
            adapter.expected_binding = binding
            adapter.image_id = binding["image_sha256"]
            adapter.task_sha256 = frozen["task_sha256"]
            adapter.submitted_workspace = snapshot
            adapter.submitted = True
            env = _restore_submitted_env(task, adapter, progress)
            verification = env.verify()
            final_state = env.get_state()
        finally:
            adapter.close()
        if verification.get("status") == "void":
            available = _timestamp(verification.get("available_at"), "available_at")
            if (final_state.get("status") != "void"
                    or verification.get("reward") is not None
                    or not isinstance(verification.get("reason"), str)
                    or not verification["reason"].strip()
                    or not isinstance(verification.get("evidence"), dict)
                    or not verification["evidence"]
                    or available < max(_timestamp(frozen["outcome_not_before"], "outcome_not_before"),
                                       _timestamp(payload["submission_state"]["submitted_at"], "submitted_at"))
                    or available > datetime.now(timezone.utc)):
                raise ValueError("verifier_void_violates_frozen_contract")
            return {"status": "void", "reward": None,
                    "reason": verification["reason"],
                    "evidence": verification["evidence"],
                    "available_at": verification["available_at"],
                    "final_state": final_state,
                    "episode_id": payload["submission_state"]["episode_id"],
                    "policy_id": payload["submission_state"]["policy_id"],
                    "policy_revision": payload["policy_revision"],
                    "task_sha256": frozen["task_sha256"],
                    "submission_sha256": payload["submission_sha256"],
                    "snapshot_sha256": payload["snapshot_sha256"],
                    "trainer_ready": False}
        if verification.get("status") != "graded":
            return {"status": "pending", "reason": verification.get("reason", "unresolved"),
                    "next_verify_at": verification.get("next_verify_at"),
                    "progress_state": final_state}
        reward = verification.get("reward")
        evidence = verification.get("evidence")
        available = _timestamp(verification.get("available_at"), "available_at")
        if (isinstance(reward, bool) or not isinstance(reward, (int, float))
                or not math.isfinite(reward)
                or not frozen["reward_contract"]["min_reward"] <= reward <= frozen["reward_contract"]["max_reward"]
                or available < max(_timestamp(frozen["outcome_not_before"], "outcome_not_before"),
                                   _timestamp(payload["submission_state"]["submitted_at"], "submitted_at"))
                or available > datetime.now(timezone.utc)
                or not isinstance(evidence, dict)
                or evidence.get("workspace_sha256") != payload["snapshot_sha256"]
                or evidence.get("verifier_sha256") != binding["verifier_sha256"]
                or evidence.get("image_sha256") != binding["image_sha256"]):
            raise ValueError("verifier_resolution_violates_frozen_contract")
        return {"status": "graded", "reward": float(reward), "evidence": evidence,
                "available_at": verification["available_at"],
                "final_state": final_state,
                "text_trajectory": env.export_trajectory(),
                "episode_id": payload["submission_state"]["episode_id"],
                "policy_id": payload["submission_state"]["policy_id"],
                "policy_revision": payload["policy_revision"],
                "task_sha256": frozen["task_sha256"],
                "submission_sha256": payload["submission_sha256"],
                "snapshot_sha256": payload["snapshot_sha256"],
                "trainer_ready": False}

    def process_one(self, worker_id, *, revision_provider, lease_seconds=30.0,
                    retry_seconds=2.0, after_verify=None,
                    atomic_revision_fence=False):
        """Claim one ready submission; heartbeat and verify outside SQL locks.

        ``after_verify`` is a trusted fault-injection hook used by crash
        experiments. A pending result or infrastructure failure is requeued
        without reward. The caller may invoke this repeatedly from several
        independent processes.
        """
        if not callable(revision_provider):
            raise ValueError("A trusted current weight revision provider is required")
        if type(atomic_revision_fence) is not bool:
            raise ValueError("atomic_revision_fence must be boolean")
        lease = self.claim(worker_id, lease_seconds=lease_seconds)
        if lease is None:
            return None
        # A strict submission carries its own immutable fence requirement.
        # Worker configuration can opt additional jobs in, but cannot opt a
        # bound job out of the queue-local revision check.
        atomic_revision_fence = (atomic_revision_fence
                                 or lease.payload.get("queue_revision_required") is True)
        stop = threading.Event()
        lost = threading.Event()

        def keep_alive():
            while not stop.wait(max(0.05, lease_seconds / 3)):
                try:
                    if not self.renew(lease, lease_seconds=lease_seconds):
                        lost.set()
                        return
                except Exception:
                    lost.set()
                    return

        heartbeat = threading.Thread(target=keep_alive, daemon=True)
        heartbeat.start()
        try:
            result = None
            try:
                if atomic_revision_fence:
                    generation = lease.payload.get("queue_revision_generation")
                    current = self.current_revision(
                        lease.payload["submission_state"]["policy_id"])
                    if (lease.payload.get("queue_revision_required") is not True
                            or type(generation) is not int or generation < 1
                            or current is None
                            or current["revision"] != lease.payload["policy_revision"]
                            or current["generation"] != generation):
                        self._finish(lease, pending_reason=
                                     "queue_revision_mismatch_before_verification")
                        return {"job_id": lease.job_id, "status": "stale",
                                "reason": "queue_revision_mismatch_before_verification",
                                "reward": None, "attempt": lease.attempt}
                if revision_provider() != lease.payload["policy_revision"]:
                    raise ValueError("policy_revision_changed_before_verification")
                result = self._verify(lease)
                if after_verify is not None:
                    after_verify(lease, copy.deepcopy(result))
                if revision_provider() != lease.payload["policy_revision"]:
                    raise ValueError("policy_revision_changed_before_reward_commit")
                if lost.is_set():
                    raise LeaseLost("Verifier lost its lease during execution")
                if result["status"] in {"graded", "void"}:
                    final_state = self._finish(
                        lease, result=result,
                        atomic_revision_fence=atomic_revision_fence)
                    if final_state == "stale":
                        return {"job_id": lease.job_id, "status": "stale",
                                "reason": "queue_revision_mismatch_at_commit",
                                "reward": None, "attempt": lease.attempt}
                    return {"job_id": lease.job_id, "status": final_state,
                            "reward": result["reward"], "attempt": lease.attempt}
                reason = result.get("reason", "unresolved")
                next_verify_at = result.get("next_verify_at")
                if next_verify_at is not None:
                    retry_seconds = max(retry_seconds, _timestamp(
                        next_verify_at, "next_verify_at").timestamp() - time.time())
                retry_seconds = max(retry_seconds, float(lease.payload["task"]["budgets"].get(
                    "verification_cooldown_seconds", 0)))
            except LeaseLost:
                raise
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
            if lost.is_set():
                raise LeaseLost("Verifier lost its lease during execution")
            final_state = self._finish(
                lease, pending_reason=reason, retry_seconds=retry_seconds,
                progress_state=result.get("progress_state")
                if isinstance(result, dict) else None)
            return {"job_id": lease.job_id, "status": (
                        "exhausted" if final_state == "exhausted" else "pending"),
                    "reason": reason, "attempt": lease.attempt, "reward": None}
        finally:
            stop.set()
            heartbeat.join(timeout=max(1.0, min(lease_seconds, 5.0)))

    def get(self, job_id):
        """Return a trusted-host audit row; never forward it to the policy."""
        db = self._connect()
        try:
            db.execute("BEGIN")
            row = db.execute("SELECT * FROM rollout_jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                db.commit()
                return None
            payload = json.loads(row["payload_json"])
            if _sha(payload) != row["payload_sha256"]:
                raise ValueError("Durable queue payload digest mismatch")
            policy_id = payload["submission_state"]["policy_id"]
            current = db.execute("""SELECT revision,generation FROM rollout_revisions
                WHERE policy_id=?""", (policy_id,)).fetchone()
            bound_generation = payload.get("queue_revision_generation")
            strict = payload.get("queue_revision_required") is True
            generation_matches = (current is not None
                                  and type(bound_generation) is int
                                  and current["generation"] == bound_generation)
            audit = {"job_id": job_id, "state": row["state"],
                    "attempt": row["attempt"], "policy_revision": row["policy_revision"],
                    "task_sha256": row["task_sha256"],
                    "snapshot_sha256": row["snapshot_sha256"],
                    "submission_sha256": payload["submission_sha256"],
                    "queue_current_revision": current["revision"] if current else None,
                    "queue_revision_generation": current["generation"] if current else None,
                    "queue_revision_matches_job": (
                        current is not None and current["revision"] == row["policy_revision"]),
                    "queue_revision_generation_bound": bound_generation,
                    "queue_revision_generation_matches_job": (
                        generation_matches if strict else None),
                    "queue_revision_fence_current": (
                        generation_matches and current["revision"] == row["policy_revision"]
                        if strict else None),
                    "lease_owner": row["lease_owner"], "lease_until": row["lease_until"],
                    "last_pending_reason": row["last_pending_reason"],
                    "reward": row["reward"],
                    "result": json.loads(row["result_json"]) if row["result_json"] else None}
            db.commit()
            return audit
        finally:
            db.close()
