"""Bounded, resumable same-host trajectory control for RealWorldEnv.

Generation, environment interaction, and trusted reward verification run in
separate worker pools. The scheduler owns all stage transitions and is the only
producer of bounded worker queues, avoiding cyclic worker-to-worker deadlocks.
It retains pending verifier leases in memory for explicit retry. This module
does not create behavior log probabilities or perform a gradient update.
"""

from __future__ import annotations

import copy
import heapq
import hashlib
import json
import math
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from typing import Callable

from .coding_env import DockerCodingAdapter
from .realworld import RealWorldEnv, validate_task


_STALE_QUEUED_ACTION = object()
_STALE_BEFORE_VERIFY = object()


def _json_digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _positive_int(name, value, maximum=128):
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [1, {maximum}]")


def _p95(values):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


@dataclass(frozen=True)
class EpisodeLease:
    """Trusted host-owned environment and its exactly-once cleanup callback."""

    env: RealWorldEnv
    close: Callable[[], None]


@dataclass(frozen=True)
class TrajectoryJob:
    job_id: str
    policy_id: str
    open_episode: Callable[[], EpisodeLease]
    generate_action: Callable[[dict, tuple[dict, ...], str], dict | None]


@dataclass
class _Context:
    index: int
    job: TrajectoryJob
    lease: EpisodeLease | None = None
    observation: dict | None = None
    visible_history: list[dict] = field(default_factory=list)
    actions: list[dict] = field(default_factory=list)
    pending_action: dict | None = None
    stale_action_rejections: int = 0
    rejected_action_sha256s: list[str] = field(default_factory=list)
    phases: list[dict] = field(default_factory=list)
    status: str = "queued"
    reward: float | None = None
    error: str | None = None
    episode_id: str | None = None
    verification: dict | None = None
    text_trajectory: dict | None = None
    closed: bool = False
    completion_offset_seconds: float | None = None
    verification_ready_at: float | None = None
    verification_wait_seconds: float = 0.0
    # A stage may still be opening or using a lease after another worker dies.
    # Guard the handoff so cleanup waits for that stage, including a late open.
    _lease_guard: object = field(default_factory=Lock, repr=False)
    _lease_in_use: bool = False
    _cleanup_requested: bool = False


class TrajectoryControlPlane:
    """Keep multi-turn episodes alive across stage handoffs and pending retries.

    ``revision_provider`` returns the current immutable model revision ID.
    ``generate_action`` must actually use the revision passed to it (or fail
    if those weights are unavailable); a string label cannot enforce model
    weight provenance on its own. The exact-revision gate is a conservative
    *freshness* check, not a claim that text trajectories are trainable.

    ``verification_delay_seconds`` is a host-side, bounded short-wait hint
    applied once after submission. With the default scheduler placement, the
    submitted lease still counts toward ``max_in_flight`` but its timer does
    not consume a reward worker. The worker placement is a controlled A/B
    reference using the same readiness timestamp. Long outcome waits should
    return ``pending`` and use ``resume_pending`` later.

    ``stale_revision_policy='fence'`` rejects a generated action if its
    revision changed before the environment step begins. Before the first
    accepted action it can regenerate up to ``max_stale_regenerations`` times.
    After an accepted action, a revision change ends the episode ungraded:
    restarting under a new policy would mix action provenance in one rollout.
    This is a same-process dispatch fence, not atomic weight synchronization
    with a remote inference server.
    """

    _STAGES = ("actor", "environment", "reward")

    def __init__(self, jobs, *, revision_provider, actor_workers=1,
                 environment_workers=1, reward_workers=1,
                 actor_queue_capacity=2, environment_queue_capacity=2,
                 reward_queue_capacity=2, max_in_flight=4,
                 before_verify=None, verification_delay_seconds=None,
                 verification_wait_placement="scheduler",
                 stale_revision_policy="report_only", max_stale_regenerations=2):
        self.jobs = list(jobs)
        if not self.jobs or any(not isinstance(job, TrajectoryJob) for job in self.jobs):
            raise ValueError("jobs must be a nonempty sequence of TrajectoryJob")
        if len({job.job_id for job in self.jobs}) != len(self.jobs):
            raise ValueError("job_id values must be unique")
        for job in self.jobs:
            if not job.job_id or not job.policy_id or not all(
                    callable(callback) for callback in (job.open_episode, job.generate_action)):
                raise ValueError("Each job requires IDs and callable episode/action functions")
        for name, value in (("actor_workers", actor_workers),
                            ("environment_workers", environment_workers),
                            ("reward_workers", reward_workers),
                            ("actor_queue_capacity", actor_queue_capacity),
                            ("environment_queue_capacity", environment_queue_capacity),
                            ("reward_queue_capacity", reward_queue_capacity),
                            ("max_in_flight", max_in_flight)):
            _positive_int(name, value)
        if not callable(revision_provider):
            raise ValueError("revision_provider must be callable")
        if before_verify is not None and not callable(before_verify):
            raise ValueError("before_verify must be callable")
        if verification_delay_seconds is not None and not callable(verification_delay_seconds):
            raise ValueError("verification_delay_seconds must be callable")
        if (not isinstance(verification_wait_placement, str)
                or verification_wait_placement not in {"scheduler", "worker"}):
            raise ValueError("verification_wait_placement must be scheduler or worker")
        if stale_revision_policy not in {"report_only", "fence"}:
            raise ValueError("stale_revision_policy must be report_only or fence")
        if (type(max_stale_regenerations) is not int
                or not 0 <= max_stale_regenerations <= 16):
            raise ValueError("max_stale_regenerations must be an integer in [0, 16]")
        self.revision_provider = revision_provider
        self.before_verify = before_verify
        self.verification_delay_seconds = verification_delay_seconds
        self.verification_wait_placement = verification_wait_placement
        self.stale_revision_policy = stale_revision_policy
        self.max_stale_regenerations = max_stale_regenerations
        self.worker_counts = {"actor": actor_workers,
                              "environment": environment_workers,
                              "reward": reward_workers}
        self.capacities = {"actor": actor_queue_capacity,
                           "environment": environment_queue_capacity,
                           "reward": reward_queue_capacity}
        self.max_in_flight = max_in_flight
        self.contexts = [_Context(index, job) for index, job in enumerate(self.jobs)]
        self._started = False
        self._closed = False
        self._run_count = 0
        self._last_wall = None

    def _revision(self):
        revision = self.revision_provider()
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError("revision_provider must return a nonblank revision ID")
        return revision

    def run(self):
        """Run all not-yet-started jobs, retaining pending verifier leases."""
        if self._started or self._closed:
            raise ValueError("Fresh jobs can only run once on an open control plane")
        self._started = True
        return self._drive(self.contexts, fresh=True)

    def resume_pending(self, job_ids=None):
        """Retry selected in-memory pending verifiers; no outcome is invented.

        The caller owns scheduling around ``next_verify_at`` and may call this
        after new evidence arrives. A process restart needs an adapter-specific
        durable lease protocol, which this in-memory manager does not provide.
        """
        if not self._started or self._closed:
            raise ValueError("There is no open run to resume")
        if job_ids is None:
            selected = [ctx for ctx in self.contexts if ctx.status == "pending"]
        else:
            ids = set(job_ids)
            known = {ctx.job.job_id for ctx in self.contexts if ctx.status == "pending"}
            if not ids or ids - known:
                raise ValueError("Can only resume known pending job IDs")
            selected = [ctx for ctx in self.contexts if ctx.job.job_id in ids]
        if not selected:
            raise ValueError("No pending verifier leases")
        return self._drive(selected, fresh=False)

    def close(self):
        """Release pending leases after ``run``/``resume_pending`` returns.

        External callers must not invoke this concurrently with an active
        drive. Internal failure cleanup can defer a close until an already
        running environment or verifier operation returns.
        """
        if self._closed:
            return
        for ctx in self.contexts:
            self._request_cleanup(ctx)
        self._closed = True

    def text_trajectories(self):
        """Return detached visible-event audits for graded train episodes.

        RealWorldEnv exports no token-level behavior likelihoods; consumers
        must not pass these directly to an optimizer. In opt-in fence mode,
        only episodes whose action revision is still current are returned.
        """
        current = self._revision() if self.stale_revision_policy == "fence" else None
        return {ctx.job.job_id: copy.deepcopy(ctx.text_trajectory)
                for ctx in self.contexts if ctx.text_trajectory is not None
                and (current is None or ctx.actions and all(
                    action["policy_revision"] == current for action in ctx.actions))}

    def _finish(self, ctx, *, start):
        if ctx.status == "graded" and ctx.text_trajectory is None:
            try:
                ctx.text_trajectory = ctx.lease.env.export_trajectory()
            except Exception as exc:
                ctx.error = f"trajectory_export: {type(exc).__name__}: {exc}"
                ctx.status, ctx.reward = "infrastructure_error", None
        if ctx.status != "pending":
            self._request_cleanup(ctx)
        ctx.completion_offset_seconds = max(0.0, time.monotonic() - start)

    @staticmethod
    def _close_lease_locked(ctx):
        """Close once while holding the per-episode guard, after its last use."""
        if ctx.lease is not None and not ctx.closed:
            try:
                ctx.lease.close()
            except Exception as exc:
                ctx.error = f"cleanup: {type(exc).__name__}: {exc}"
                ctx.status, ctx.reward = "infrastructure_error", None
            finally:
                ctx.closed = True

    def _request_cleanup(self, ctx):
        with ctx._lease_guard:
            ctx._cleanup_requested = True
            if not ctx._lease_in_use:
                self._close_lease_locked(ctx)

    @staticmethod
    def _begin_lease_use(ctx):
        with ctx._lease_guard:
            if ctx._cleanup_requested:
                raise RuntimeError("episode_cleanup_requested")
            if ctx._lease_in_use:
                raise RuntimeError("concurrent_episode_lease_use")
            ctx._lease_in_use = True

    def _end_lease_use(self, ctx):
        with ctx._lease_guard:
            ctx._lease_in_use = False
            if ctx._cleanup_requested:
                self._close_lease_locked(ctx)
            return ctx._cleanup_requested

    def _drive(self, selected, *, fresh):
        start = time.monotonic()
        phase_counts_before = {ctx.index: len(ctx.phases) for ctx in selected}
        queues = {stage: Queue(maxsize=self.capacities[stage]) for stage in self._STAGES}
        # At most one stage result exists per in-flight episode.
        completions = Queue(maxsize=self.max_in_flight)
        waiting = {stage: deque() for stage in self._STAGES}
        # A delayed verifier still owns one bounded in-flight lease, but does
        # not occupy a reward worker or its queue until its deadline is due.
        delayed_rewards = []
        queue_high_water = {stage: 0 for stage in self._STAGES}
        stopping = Event()
        worker_error = []

        def worker(stage):
            queue = queues[stage]
            while not stopping.is_set():
                try:
                    payload = queue.get(timeout=0.1)
                except Empty:
                    continue
                try:
                    ctx, operation, revision, queued_at = payload
                    began = time.monotonic()
                    result, error = None, None
                    try:
                        if stage == "actor":
                            result = ctx.job.generate_action(
                                copy.deepcopy(ctx.observation),
                                tuple(copy.deepcopy(ctx.visible_history)), revision)
                        else:
                            self._begin_lease_use(ctx)
                            try:
                                if stage == "environment":
                                    if operation == "reset":
                                        lease = ctx.job.open_episode()
                                        if not isinstance(lease, EpisodeLease):
                                            raise ValueError("open_episode must return EpisodeLease")
                                        ctx.lease = lease
                                        if ctx._cleanup_requested:
                                            raise RuntimeError("episode_cleanup_requested")
                                        result = lease.env.reset(ctx.job.policy_id)
                                    else:
                                        if (self.stale_revision_policy == "fence"
                                                and revision != self._revision()):
                                            result = _STALE_QUEUED_ACTION
                                        else:
                                            result = ctx.lease.env.step(operation)
                                else:
                                    if self.verification_wait_placement == "worker" and ctx.verification_ready_at is not None:
                                        remaining = ctx.verification_ready_at - time.monotonic()
                                        if remaining > 0:
                                            time.sleep(remaining)
                                    if (self.stale_revision_policy == "fence" and ctx.actions
                                            and ctx.actions[0]["policy_revision"] != self._revision()):
                                        result = _STALE_BEFORE_VERIFY
                                    else:
                                        if self.before_verify is not None:
                                            self.before_verify(ctx.job.job_id)
                                        if ctx._cleanup_requested:
                                            raise RuntimeError("episode_cleanup_requested")
                                        if (self.stale_revision_policy == "fence" and ctx.actions
                                                and ctx.actions[0]["policy_revision"] != self._revision()):
                                            result = _STALE_BEFORE_VERIFY
                                        else:
                                            result = ctx.lease.env.verify()
                            finally:
                                if self._end_lease_use(ctx):
                                    raise RuntimeError("episode_cleanup_requested")
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                    ended = time.monotonic()
                    completion = (ctx, stage, operation, revision, result, error,
                                  queued_at, began, ended)
                    while not stopping.is_set():
                        try:
                            completions.put(completion, timeout=0.1)
                            break
                        except Full:
                            continue
                except BaseException as exc:  # A worker must not disappear silently.
                    worker_error.append(exc)
                    return
                finally:
                    queue.task_done()

        workers = [Thread(target=worker, args=(stage,), daemon=True,
                          name=f"trajectory-{stage}-{n}")
                   for stage in self._STAGES for n in range(self.worker_counts[stage])]
        for thread in workers:
            thread.start()

        def schedule(ctx, stage, operation, revision=None):
            waiting[stage].append((ctx, operation, revision, time.monotonic()))

        def schedule_verification(ctx, *, initial):
            ctx.verification_ready_at = None
            ctx.verification_wait_seconds = 0.0
            if initial and self.verification_delay_seconds is not None:
                delay = self.verification_delay_seconds(ctx.job.job_id)
                if (isinstance(delay, bool) or not isinstance(delay, (int, float))
                        or not math.isfinite(delay) or not 0 <= delay <= 30):
                    raise ValueError("verification_delay_seconds must return finite seconds in [0, 30]")
                ctx.verification_wait_seconds = float(delay)
                ctx.verification_ready_at = time.monotonic() + delay
            if (self.verification_wait_placement == "scheduler"
                    and ctx.verification_ready_at is not None
                    and ctx.verification_ready_at > time.monotonic()):
                heapq.heappush(delayed_rewards, (ctx.verification_ready_at, ctx.index, ctx))
            else:
                schedule(ctx, "reward", "verify")

        def reject_stale_action(ctx, action, current):
            ctx.stale_action_rejections += 1
            ctx.rejected_action_sha256s.append(action["action_sha256"])
            if ctx.actions or ctx.stale_action_rejections > self.max_stale_regenerations:
                ctx.status, ctx.reward = "stale_policy_revision", None
            else:
                schedule(ctx, "actor", "generate", current)

        waiting_jobs = deque(selected)
        active = 0
        completed = 0
        try:
            while completed < len(selected):
                while waiting_jobs and active < self.max_in_flight:
                    ctx = waiting_jobs.popleft()
                    schedule(ctx, "environment", "reset") if fresh else schedule_verification(ctx, initial=False)
                    active += 1
                now = time.monotonic()
                while delayed_rewards and delayed_rewards[0][0] <= now:
                    _, _, ctx = heapq.heappop(delayed_rewards)
                    schedule(ctx, "reward", "verify")
                for stage in self._STAGES:
                    while waiting[stage]:
                        try:
                            queues[stage].put_nowait(waiting[stage][0])
                        except Full:
                            break
                        waiting[stage].popleft()
                        queue_high_water[stage] = max(queue_high_water[stage], queues[stage].qsize())
                try:
                    timeout = min(0.1, max(0.0, delayed_rewards[0][0] - time.monotonic())) if delayed_rewards else 0.1
                    (ctx, stage, operation, revision, result, error,
                     queued_at, began, ended) = completions.get(timeout=timeout)
                except Empty:
                    if worker_error or not any(thread.is_alive() for thread in workers):
                        raise RuntimeError("Trajectory worker died without a completion")
                    continue
                ctx.phases.append({"stage": stage,
                                   "operation": operation if stage != "environment" or operation == "reset" else "step",
                                   "queue_seconds": max(0.0, began - queued_at),
                                   "work_seconds": max(0.0, ended - began),
                                   "started_offset_seconds": max(0.0, began - start),
                                   "finished_offset_seconds": max(0.0, ended - start)})
                if error is not None:
                    ctx.status, ctx.reward, ctx.error = "infrastructure_error", None, error
                elif stage == "environment" and operation == "reset":
                    ctx.episode_id = result.get("episode_id")
                    if ctx.lease.env.status == "active":
                        ctx.observation = copy.deepcopy(result)
                        ctx.visible_history.append({"opening": copy.deepcopy(result)})
                        schedule(ctx, "actor", "generate", self._revision())
                    else:
                        ctx.status = ctx.lease.env.status
                elif stage == "actor":
                    if result is None:
                        ctx.status = "policy_stopped_ungraded"
                    elif not isinstance(result, dict):
                        ctx.status, ctx.error = "infrastructure_error", "generate_action must return an action object or None"
                    else:
                        try:
                            action_hash = _json_digest(result)
                        except (TypeError, ValueError, OverflowError) as exc:
                            ctx.status, ctx.error = "infrastructure_error", f"invalid action: {exc}"
                        else:
                            action_record = {"policy_revision": revision,
                                             "action_sha256": action_hash,
                                             "generated_offset_seconds": max(0.0, ended - start)}
                            if (self.stale_revision_policy == "fence"
                                    and revision != (current := self._revision())):
                                reject_stale_action(ctx, action_record, current)
                            else:
                                if self.stale_revision_policy == "fence":
                                    ctx.pending_action = action_record
                                else:
                                    ctx.actions.append(action_record)
                                schedule(ctx, "environment", copy.deepcopy(result), revision)
                elif stage == "environment":
                    if result is _STALE_QUEUED_ACTION:
                        action_record = ctx.pending_action
                        ctx.pending_action = None
                        if action_record is None:
                            ctx.status, ctx.error = "infrastructure_error", "stale action record missing"
                        else:
                            reject_stale_action(ctx, action_record, self._revision())
                    else:
                        if ctx.pending_action is not None:
                            ctx.actions.append(ctx.pending_action)
                            ctx.pending_action = None
                        ctx.observation = copy.deepcopy(result)
                        ctx.visible_history.append({"action": copy.deepcopy(operation),
                                                    "transition": copy.deepcopy(result)})
                        if (self.stale_revision_policy == "fence" and ctx.actions
                                and ctx.actions[0]["policy_revision"] != self._revision()):
                            ctx.status, ctx.reward = "stale_policy_revision", None
                        elif ctx.lease.env.status == "active":
                            schedule(ctx, "actor", "generate", self._revision())
                        elif ctx.lease.env.status == "pending":
                            schedule_verification(ctx, initial=True)
                        else:
                            ctx.status = ctx.lease.env.status
                else:
                    if result is _STALE_BEFORE_VERIFY:
                        ctx.status, ctx.reward = "stale_policy_revision", None
                    else:
                        ctx.verification = copy.deepcopy(result)
                        if (self.stale_revision_policy == "fence" and ctx.actions
                                and ctx.actions[0]["policy_revision"] != self._revision()):
                            ctx.status, ctx.reward = "stale_policy_revision", None
                        else:
                            ctx.status = ctx.lease.env.status
                            if ctx.status == "graded":
                                ctx.reward = ctx.lease.env.reward
                            elif ctx.status not in {"pending", "void"}:
                                ctx.status = "infrastructure_error"
                                ctx.error = "Verifier returned an unexpected environment status"
                if ctx.status in {"queued", "pending"} and stage != "reward":
                    continue
                if ctx.status == "queued":  # stage still active
                    continue
                self._finish(ctx, start=start)
                completed += 1
                active -= 1
        finally:
            # Stop workers before releasing leases. A dead worker or a full
            # queue must never block cleanup while a reward is unresolved.
            stopping.set()
            for queue in queues.values():
                while True:
                    try:
                        queue.get_nowait()
                    except Empty:
                        break
                    else:
                        queue.task_done()
            for thread in workers:
                thread.join(timeout=1)
            if completed < len(selected):
                for ctx in selected:
                    if ctx.status == "queued":
                        ctx.status, ctx.reward = "infrastructure_error", None
                    self._finish(ctx, start=start)
        self._run_count += 1
        wall = max(0.0, time.monotonic() - start)
        self._last_wall = wall
        return self.report(wall_seconds=wall, queue_high_water=queue_high_water,
                           selected_count=len(selected),
                           selected_graded_count=sum(ctx.status == "graded" for ctx in selected),
                           phase_scope=[phase for ctx in selected
                                        for phase in ctx.phases[phase_counts_before[ctx.index]:]])

    def report(self, *, wall_seconds=None, queue_high_water=None, selected_count=None,
               selected_graded_count=None, phase_scope=None):
        current = self._revision()
        episodes = []
        for ctx in self.contexts:
            revisions = [action["policy_revision"] for action in ctx.actions]
            freshness = ("not_graded" if ctx.status != "graded" else
                         "no_actions" if not revisions else
                         "mixed_policy_revisions" if len(set(revisions)) != 1 else
                         "current_revision" if revisions[0] == current else
                         "stale_revision")
            episodes.append({"index": ctx.index, "job_id": ctx.job.job_id,
                             "episode_id": ctx.episode_id, "status": ctx.status,
                             "reward": ctx.reward, "error": ctx.error,
                             "verification_status": ctx.verification.get("status")
                             if isinstance(ctx.verification, dict) else None,
                             "verification_evidence_sha256": _json_digest(ctx.verification["evidence"])
                             if isinstance(ctx.verification, dict)
                             and isinstance(ctx.verification.get("evidence"), dict) else None,
                             "text_trajectory_sha256": _json_digest(ctx.text_trajectory)
                             if ctx.text_trajectory is not None else None,
                             "policy_id": ctx.job.policy_id,
                             "actions": copy.deepcopy(ctx.actions),
                             "stale_action_rejections": ctx.stale_action_rejections,
                             "rejected_action_sha256s": list(ctx.rejected_action_sha256s),
                             "phase_timestamps": copy.deepcopy(ctx.phases),
                             "completion_offset_seconds": ctx.completion_offset_seconds,
                             "verification_wait_seconds": ctx.verification_wait_seconds,
                             "freshness_gate": freshness,
                             "trainer_ready": False})
        phases = phase_scope if phase_scope is not None else [
            phase for ctx in self.contexts for phase in ctx.phases]
        waits = {stage: [phase["queue_seconds"] for phase in phases if phase["stage"] == stage]
                 for stage in self._STAGES}
        graded = sum(ctx.status == "graded" for ctx in self.contexts)
        return {"schema_version": "realworld-trajectory-control-0.1",
                "architecture": "same_host_bounded_three_stage_control_plane",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "current_policy_revision": current, "trainer_ready": False,
                "verification_wait_placement": self.verification_wait_placement,
                "stale_revision_policy": self.stale_revision_policy,
                "max_stale_regenerations": self.max_stale_regenerations,
                "run_count": self._run_count,
                "wall_seconds": wall_seconds,
                "selected_count": selected_count,
                "selected_graded_count": selected_graded_count,
                "graded_count": graded,
                "valid_graded_episodes_per_hour": selected_graded_count * 3600 / wall_seconds
                if wall_seconds and selected_graded_count is not None else None,
                "pending_count": sum(ctx.status == "pending" for ctx in self.contexts),
                "infrastructure_error_count": sum(ctx.status == "infrastructure_error" for ctx in self.contexts),
                "fresh_graded_count": sum(item["freshness_gate"] == "current_revision" for item in episodes),
                "queue_high_water": queue_high_water,
                "queue_wait_p95_seconds": {stage: _p95(waits[stage]) for stage in self._STAGES},
                "queue_wait_median_seconds": {stage: statistics.median(waits[stage]) if waits[stage] else None
                                              for stage in self._STAGES},
                "episodes": episodes}


def coding_episode_factory(*, task, seed_dir, verifier_dir, image, output_root,
                           visible_check,
                           verifier_workers=1):
    """Bind a real Docker coding adapter to the generic trajectory manager."""
    frozen = validate_task(task)
    if not frozen["is_fixture"]:
        raise ValueError("Non-fixture tasks require a registered factory")
    root = Path(output_root).resolve()

    def open_episode(index):
        output = root / f"episode-{index:06d}"
        adapter = DockerCodingAdapter(seed_dir=seed_dir, verifier_dir=verifier_dir,
                                      image=image, output_root=output / "instances",
                                      visible_check=visible_check,
                                      verifier_workers=verifier_workers)
        try:
            bound = copy.deepcopy(task)
            metadata = bound.setdefault("metadata", {})
            binding = adapter.artifact_binding()
            if "artifact_binding" in metadata and metadata["artifact_binding"] != binding:
                raise ValueError("Coding artifacts differ from task's frozen binding")
            metadata["artifact_binding"] = binding
            env = RealWorldEnv(bound, adapter)
            return EpisodeLease(env=env, close=adapter.close)
        except Exception:
            adapter.close()
            raise

    return open_episode
