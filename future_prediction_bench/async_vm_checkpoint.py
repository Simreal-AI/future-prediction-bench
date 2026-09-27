"""Turn-boundary overlap of a real VM checkpoint and a policy wait.

This is a narrow, trusted-host experiment inspired by Crab's completion gate.
The callback sees a detached, immutable observation and has no VM handle. The
caller must not access the adapter/runtime out of band while a turn runs.
Each turn still uses the full QEMU snapshot and durable recovery journal; no
guest command or next action is issued until its commit and the policy boundary
record have completed. A failed step, checkpoint, policy, or publication halts
the coordinator and cannot be interpreted as a zero reward.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .semantic_vm_recovery import CodingVMRecoveryJournal, RecoveryError


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


@dataclass(frozen=True)
class PolicyPrompt:
    turn: int
    observation_json: str
    observation_sha256: str


@dataclass(frozen=True)
class PolicyChoice:
    observation_sha256: str
    next_action: dict


class AsyncVMCheckpointCoordinator:
    """One exclusive QEMU episode with a durable policy-response release gate.

    ``mode='serial'`` is the A/B control. ``mode='overlap'`` starts the journal
    checkpoint on a worker, then runs the observation-only policy callback on
    the caller. Both arms commit precisely the same full VM state per turn.
    The policy callback must be pure with respect to the VM; the caller owns
    the adapter exclusively for the lifetime of this coordinator.
    """

    def __init__(self, adapter, directory, *, mode="overlap", inspector=None):
        if mode not in {"serial", "overlap"}:
            raise ValueError("mode must be serial or overlap")
        self.adapter = adapter
        self.journal = CodingVMRecoveryJournal(
            adapter, directory, mode="every_turn", inspector=inspector)
        self.mode = mode
        self._executor = ThreadPoolExecutor(max_workers=1,
                                            thread_name_prefix="fpb-vm-checkpoint")
        self._gate = threading.Lock()
        self._gate_owner = None
        self._next_action = None
        self._last_boundary = None
        self._halted = False
        self._closed = False
        self.turn_metrics = []

    def _enter(self):
        if not self._gate.acquire(blocking=False):
            raise RecoveryError("checkpoint_boundary_in_flight")
        self._gate_owner = threading.get_ident()
        if self._closed or self._halted:
            self._leave()
            raise RecoveryError("coordinator_closed_or_halted")

    def _leave(self):
        self._gate_owner = None
        self._gate.release()

    def begin(self):
        self._enter()
        try:
            if self._last_boundary is not None or self.journal.manifest is not None:
                raise RecoveryError("coordinator_already_started")
            return self.journal.begin()
        finally:
            self._leave()

    def _boundary_path(self, turn):
        return self.journal.directory / f"policy-turn-{turn:06d}.json"

    def _publish_boundary(self, boundary):
        path = self._boundary_path(boundary["turn"])
        if path.exists() or path.is_symlink():
            raise RecoveryError("policy_boundary_already_exists")
        descriptor, staging = tempfile.mkstemp(
            prefix=".policy-boundary-", dir=self.journal.directory)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(_canonical(boundary) + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
            # Link rather than replace: a previously published turn may not
            # silently be overwritten by a retry or another coordinator.
            os.link(staging, path)
            os.unlink(staging)
            directory_fd = os.open(self.journal.directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            Path(staging).unlink(missing_ok=True)

    def _check_prior_boundary(self):
        if self._last_boundary is None:
            return
        path = self._boundary_path(self._last_boundary["turn"])
        if path.is_symlink() or not path.is_file():
            raise RecoveryError("policy_boundary_missing_or_changed")
        try:
            recorded = json.loads(path.read_bytes())
        except (ValueError, OSError) as exc:
            raise RecoveryError("policy_boundary_missing_or_changed") from exc
        manifest_bytes = self.journal.path.read_bytes()
        if (recorded != self._last_boundary
                or recorded["journal_sha256"] != hashlib.sha256(manifest_bytes).hexdigest()
                or recorded["checkpoint_tag"] != self.journal.manifest["checkpoint_tag"]
                or recorded["turn"] != self.journal.manifest["turn"]
                or recorded["next_action_sha256"] != _digest(self._next_action)):
            raise RecoveryError("policy_boundary_missing_or_changed")

    def run_turn(self, action, policy: Callable[[PolicyPrompt], PolicyChoice], *,
                 now, inject_after_save=False):
        """Execute one bounded tool turn and return only after both commits.

        The callback may inspect ``PolicyPrompt`` and compute the next action;
        it must not touch the VM. ``inject_after_save`` is fault testing only.
        """
        self._enter()
        action_in_flight = False
        try:
            self.journal._state()
            self._check_prior_boundary()
            if (not isinstance(action, dict)
                    or action.get("action") not in self.journal._ACTIONS):
                raise ValueError("unsupported_turn_action")
            # The caller may retain or mutate its action object while the
            # checkpoint worker runs. Freeze the exact command before any VM
            # side effect so the journal and metrics describe what executed.
            try:
                frozen_action = json.loads(_canonical(action))
            except (TypeError, ValueError, UnicodeError) as exc:
                raise ValueError("turn_action_not_canonical_json") from exc
            if self._next_action is not None and frozen_action != self._next_action:
                raise RecoveryError("action_differs_from_committed_policy_choice")
            started = time.monotonic()
            action_in_flight = True
            try:
                result = self.adapter.step(frozen_action, now=now)
            except BaseException:
                self.journal.needs_recovery = True
                self._halted = True
                raise
            action_seconds = time.monotonic() - started
            if not isinstance(result, dict) or "observation" not in result:
                self.journal.needs_recovery = True
                self._halted = True
                raise RecoveryError("tool_result_missing_observation")
            try:
                observation_json = _canonical(result["observation"]).decode("utf-8")
                observation_sha256 = _digest(result["observation"])
            except (TypeError, ValueError, UnicodeError) as exc:
                self.journal.needs_recovery = True
                self._halted = True
                raise RecoveryError("tool_observation_not_canonical_json") from exc
            prompt = PolicyPrompt(self.journal.manifest["turn"] + 1,
                                  observation_json, observation_sha256)
            checkpoint_clock = {}

            def commit():
                checkpoint_clock["start"] = time.monotonic()
                try:
                    return self.journal._commit(
                        frozen_action, result, safe=False,
                        opaque=frozen_action["action"] == "run_visible_checks",
                        inject_after_save=inject_after_save)
                finally:
                    checkpoint_clock["end"] = time.monotonic()

            policy_error = None
            choice = None
            if self.mode == "overlap":
                future = self._executor.submit(commit)
                policy_start = time.monotonic()
                try:
                    choice = policy(prompt)
                except BaseException as exc:
                    policy_error = exc
                policy_end = time.monotonic()
                gate_start = time.monotonic()
                try:
                    commit_result = future.result()  # Always await before release/close.
                except BaseException:
                    self._halted = True
                    raise
                gate_end = time.monotonic()
            else:
                try:
                    commit_result = commit()
                except BaseException:
                    self._halted = True
                    raise
                policy_start = time.monotonic()
                try:
                    choice = policy(prompt)
                except BaseException as exc:
                    policy_error = exc
                policy_end = time.monotonic()
                gate_start = gate_end = policy_end
            if policy_error is not None:
                self._halted = True
                raise policy_error
            if (not isinstance(choice, PolicyChoice)
                    or choice.observation_sha256 != prompt.observation_sha256
                    or not isinstance(choice.next_action, dict)
                    or choice.next_action.get("action") not in
                    (self.journal._ACTIONS | {"submit"})):
                self._halted = True
                raise RecoveryError("policy_choice_not_bound_to_observation")
            try:
                next_action = json.loads(_canonical(choice.next_action))
            except (TypeError, ValueError, UnicodeError) as exc:
                self._halted = True
                raise RecoveryError("policy_choice_not_canonical_json") from exc
            journal_bytes = self.journal.path.read_bytes()
            boundary = {
                "schema": "fpb-async-vm-policy-boundary-v1",
                "turn": commit_result["turn"],
                "checkpoint_tag": self.journal.manifest["checkpoint_tag"],
                "journal_sha256": hashlib.sha256(journal_bytes).hexdigest(),
                "observation_sha256": prompt.observation_sha256,
                "next_action_sha256": _digest(next_action),
                "coordinator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            }
            try:
                self._publish_boundary(boundary)
            except BaseException:
                self._halted = True
                raise
            self._last_boundary = boundary
            self._next_action = next_action
            checkpoint_seconds = checkpoint_clock["end"] - checkpoint_clock["start"]
            overlap_seconds = max(0.0, min(checkpoint_clock["end"], policy_end)
                                  - max(checkpoint_clock["start"], policy_start))
            measurements = {
                "turn": commit_result["turn"], "action": frozen_action["action"],
                "action_seconds": action_seconds,
                "checkpoint_seconds": checkpoint_seconds,
                "policy_wait_seconds": policy_end - policy_start,
                "checkpoint_policy_overlap_seconds": overlap_seconds,
                "exposed_checkpoint_gate_seconds": (gate_end - gate_start
                    if self.mode == "overlap" else checkpoint_seconds),
                "boundary_seconds": time.monotonic() - gate_end
                    if self.mode == "overlap" else time.monotonic() - policy_end,
                "turn_seconds": time.monotonic() - started,
            }
            self.turn_metrics.append(measurements)
            return {"result": result, "next_action": next_action,
                    "observation_sha256": prompt.observation_sha256,
                    "measurements": measurements}
        except BaseException:
            if action_in_flight:
                # Any cancellation after issuing a guest action may leave an
                # uncommitted state. Join a possibly submitted worker before
                # releasing the exclusive gate, then fail closed.
                self._halted = True
                self.journal.needs_recovery = True
                self._executor.shutdown(wait=True, cancel_futures=False)
            raise
        finally:
            self._leave()

    def submit_and_verify(self, *, now):
        self._enter()
        try:
            self.journal._state()
            self._check_prior_boundary()
            if self._next_action != {"action": "submit"}:
                raise RecoveryError("submit_not_committed_policy_choice")
            return self.journal.submit_and_verify(now=now)
        finally:
            self._leave()

    def close(self):
        if self._gate_owner == threading.get_ident():
            raise RecoveryError("cannot_close_within_policy_callback")
        with self._gate:
            if not self._closed:
                self._executor.shutdown(wait=True, cancel_futures=False)
                self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
