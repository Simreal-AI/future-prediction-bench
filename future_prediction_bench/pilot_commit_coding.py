"""Connect budgeted dispatch to actual host-verified coding outcomes.

The producer owns the repository adapter and verifier invocation. An injected
policy provider receives visible observations and returns tool actions only;
it never supplies a reward or verifier callback. The operator-owned SQLite
database is the authority, not a portable certificate against its owner.

Budget units remain dispatched episodes. Nanosecond wall costs are measured
separately; no token throughput, GPU speed, or optimizer execution is implied.

The inherited Docker verifier grades a candidate command timeout/output limit
as binary failure only after its owned container cleanup is confirmed. Docker
infrastructure exceptions or unavailable verification remain unresolved; they
are never converted to reward zero by this producer.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Mapping, Protocol

from . import coding_env, realworld, pilot_commit
from .coding_env import DockerCodingAdapter, _workspace_digest
from .pilot_commit import (CommitAllocation, CommitPlan, PilotBudgetLedger,
    PilotReceipt, Reservation, SelectionResult, TaskRevision, plan_pilot_commit)
from .realworld import RealWorldEnv, RealWorldTaskRegistry, validate_task


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _hash(value, field):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(field + "_requires_SHA256")


def _identifier(value, field):
    if type(value) is not str or not value.strip() or len(value) > 256 or any(ord(c) < 32 for c in value):
        raise ValueError(field + "_requires_bounded_identifier")


def _directory(path):
    lexical = Path(path).absolute()
    if any(p.is_symlink() for p in (lexical, *lexical.parents)):
        raise ValueError("nonsymlink_owned_path_required")
    return lexical.resolve()


@dataclass(frozen=True)
class CodingPolicyBinding:
    """Opaque operator-pinned policy inputs; raw provider identities stay private.

    These are operator declarations, not attestation of remote model weights.
    Providers must expose the matching policy_revision before and after calls.
    """
    controller_sha256: str
    prompt_sha256: str
    provider_config_sha256: str

    def __post_init__(self):
        for name, value in asdict(self).items():
            _hash(value, name)

    @property
    def revision(self):
        return _digest(asdict(self))


class CodingActionProvider(Protocol):
    policy_revision: str

    def next_action(self, observation: Mapping, *, action_index: int,
                    reservation_id: str) -> Mapping:
        """Return an action from the visible task's tool manifest only."""
        ...


@dataclass(frozen=True)
class CodingDispatch:
    dispatched: bool
    receipt: PilotReceipt | None = None
    outcome_sha256: str | None = None


def _case_reward(verification, *, specification, verifier_sha256, image_sha256):
    """Read actual adapter case evidence, never a policy-provided pass flag."""
    if type(verification) is not dict or verification.get("status") != "graded":
        raise ValueError("terminal_repository_verification_required")
    evidence = verification.get("evidence")
    if (type(evidence) is not dict or evidence.get("kind") != "host_checked_command_cases_v1" or
            evidence.get("verifier_sha256") != verifier_sha256 or
            evidence.get("image_sha256") != image_sha256):
        raise ValueError("exact_host_checked_repository_evidence_required")
    cases = evidence.get("case_results")
    expected_cases = specification["cases"]
    if type(cases) is not list or len(cases) != len(expected_cases) or not cases:
        raise ValueError("all_actual_verifier_cases_required")
    for case, expected in zip(cases, expected_cases):
        if type(case) is not dict or type(case.get("passed")) is not bool:
            raise ValueError("typed_actual_case_result_required")
        if case.get("limit_exceeded"):
            if (case["passed"] is not False or case.get("return_code") is not None or
                    case.get("stdout_sha256") is not None or case["limit_exceeded"] not in
                    ("docker_cli_timed_out", "docker_cli_output_limit_exceeded")):
                raise ValueError("invalid_verifier_limit_result")
        else:
            if (type(case.get("return_code")) is not int or not 0 <= case["return_code"] <= 255 or
                    case["return_code"] == 125):
                raise ValueError("actual_case_exit_code_required")
            _hash(case.get("stdout_sha256"), "actual_case_stdout")
            actual_passed = (case["return_code"] == expected["expected_returncode"] and
                case["stdout_sha256"] == hashlib.sha256(expected["expected_stdout"].encode()).hexdigest())
            if case["passed"] is not actual_passed:
                raise ValueError("actual_stdout_exit_code_and_case_flag_disagree")
    reward = int(all(case["passed"] for case in cases))
    stated = verification.get("reward")
    if type(stated) not in (int, float) or stated != reward:
        raise ValueError("case_outcomes_and_terminal_reward_disagree")
    return reward


class CodingOutcomeProducer:
    """One exact repository/task/policy binding, with a fresh adapter per claim.

    Only the shipped DockerCodingAdapter is constructed; arbitrary adapter
    factories and reward callbacks are deliberately absent. The inference
    provider is caller-owned and its observations contain no hidden grading
    data. No credential, environment model setting, or provider name is read.
    """

    def __init__(self, ledger, *, task, seed_dir, verifier_dir, image, output,
                 policy_binding, visible_check, command_timeout=20.0,
                 verifier_workers=1):
        if type(ledger) is not PilotBudgetLedger or type(policy_binding) is not CodingPolicyBinding:
            raise ValueError("typed_persistent_ledger_and_policy_binding_required")
        self.ledger, self.policy_binding = ledger, policy_binding
        self.seed_dir, self.verifier_dir = _directory(seed_dir), _directory(verifier_dir)
        self.output = _directory(output)
        ledger_path = _directory(ledger.path)
        for root in (self.seed_dir, self.verifier_dir):
            if (self.output.is_relative_to(root) or root.is_relative_to(self.output) or
                    ledger_path.is_relative_to(root)):
                raise ValueError("ledger_output_and_trusted_inputs_must_be_disjoint")
        if self.output == ledger_path or ledger_path.is_relative_to(self.output):
            raise ValueError("budget_database_must_stay_outside_episode_output")
        self._options = dict(seed_dir=self.seed_dir, verifier_dir=self.verifier_dir,
            image=image, visible_check=visible_check, command_timeout=command_timeout,
            verifier_workers=verifier_workers)
        admission = DockerCodingAdapter(output_root=self.output / "admission", **self._options)
        # Image inspection and input hashing are read-only admission, not a rollout.
        self.binding = admission.artifact_binding()
        bound = copy.deepcopy(task)
        validate_task(bound)
        if bound["reward_contract"]["min_reward"] != 0 or bound["reward_contract"]["max_reward"] != 1:
            raise ValueError("binary_repository_reward_contract_required")
        if not bound["is_fixture"] and bound.get("adapter_id") != "docker_coding":
            raise ValueError("nonfixture_Docker_repository_adapter_binding_required")
        bound["adapter_id"], bound["adapter_version"] = "docker_coding", "0.1"
        metadata = bound.setdefault("metadata", {})
        if metadata.get("artifact_binding", self.binding) != self.binding:
            raise ValueError("frozen_repository_artifacts_changed")
        metadata["artifact_binding"] = self.binding
        self._task = bound
        self.task_revision = validate_task(bound)["task_sha256"]
        self.policy_revision = policy_binding.revision
        self._source_paths = {"pilot_commit_coding.py": Path(__file__),
            "coding_env.py": Path(coding_env.__file__), "realworld.py": Path(realworld.__file__),
            "pilot_commit.py": Path(pilot_commit.__file__)}
        self.source_sha256 = {name: _sha(path) for name, path in self._source_paths.items()}
        self._spec = json.loads((self.verifier_dir / "verify.json").read_bytes())
        if (type(self._spec) is not dict or self._spec.get("kind") != "command_cases_v1" or
                type(self._spec.get("cases")) is not list or not 1 <= len(self._spec["cases"]) <= 32):
            raise ValueError("bounded_actual_repository_verifier_specification_required")
        for case in self._spec["cases"]:
            if (type(case) is not dict or set(case) != {"argv", "expected_stdout", "expected_returncode"} or
                    type(case["argv"]) is not list or not 1 <= len(case["argv"]) <= 32 or
                    any(type(s) is not str or not s or len(s) > 2048 or "\x00" in s for s in case["argv"]) or
                    type(case["expected_stdout"]) is not str or len(case["expected_stdout"].encode()) > 12000 or
                    type(case["expected_returncode"]) is not int or not 0 <= case["expected_returncode"] <= 124):
                raise ValueError("actual_repository_case_contract_required")
        self.output.mkdir(parents=True, exist_ok=True)
        manifest = {"schema_version": "coding-pilot-producer-binding-v1", "epoch_id": ledger.epoch_id,
            "task_id": bound["task_id"], "task_revision": self.task_revision,
            "policy_revision": self.policy_revision, "artifact_binding": self.binding,
            "source_sha256": self.source_sha256}
        manifest_file = self.output / "producer-manifest.json"
        if manifest_file.exists():
            if manifest_file.is_symlink() or json.loads(manifest_file.read_bytes()) != manifest:
                raise ValueError("producer_output_binding_conflict")
        else:
            if any(self.output.iterdir()):
                raise ValueError("new_or_matching_owned_producer_output_required")
            source = self.output / "source"
            source.mkdir()
            for name, path in self._source_paths.items():
                (source / name).write_bytes(path.read_bytes())
                if _sha(source / name) != self.source_sha256[name]:
                    raise ValueError("producer_source_changed_before_freeze")
            manifest_file.write_text(_json(manifest) + "\n")
        self._manifest_sha256 = _sha(manifest_file)
        with ledger._transaction() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS coding_pilot_outcomes (
                epoch_id TEXT NOT NULL, reservation_id TEXT NOT NULL, receipt_id TEXT NOT NULL,
                outcome_sha256 TEXT NOT NULL, payload TEXT NOT NULL,
                PRIMARY KEY(epoch_id,reservation_id), UNIQUE(epoch_id,receipt_id))""")

    def task(self, *, commit_floor=2, commit_cap=8):
        return TaskRevision(self._task["task_id"], self.task_revision, commit_floor, commit_cap)

    def _guard(self):
        current = {name: _sha(path) for name, path in self._source_paths.items()}
        frozen = {name: _sha(self.output / "source" / name) for name in self.source_sha256}
        if current != self.source_sha256 or frozen != current:
            raise ValueError("producer_or_verifier_source_changed")
        if _sha(self.output / "producer-manifest.json") != self._manifest_sha256:
            raise ValueError("producer_manifest_changed")
        admission = DockerCodingAdapter(output_root=self.output / "admission", **self._options)
        if admission.artifact_binding() != self.binding:
            raise ValueError("repository_or_verifier_artifact_changed")
        if validate_task(self._task)["task_sha256"] != self.task_revision:
            raise ValueError("exact_task_revision_changed")

    def reserve_pilots(self, *, request_id, count=2):
        """Reserve initial samples before outcomes exist, without selecting rewards.

        Uses the existing transactional reservation primitive. This bootstrap
        plan is explicitly unselected; later allocation uses real receipts.
        """
        _identifier(request_id, "request_id")
        if type(count) is not int or not 2 <= count <= 1024:
            raise ValueError("bounded_initial_pilot_count_required")
        self._guard()
        task = self.task()
        request_sha = _digest(["coding-initial-pilot-v1", task.task_id, self.task_revision,
                               self.policy_revision, count])

        def build(snapshot):
            if snapshot["remaining"] < count:
                raise ValueError("initial_pilots_exceed_remaining_budget")
            reservations = tuple(Reservation(_digest([self.ledger.epoch_id, request_id,
                "initial-pilot", self.task_revision, self.policy_revision, index]), task.task_id,
                self.task_revision, self.policy_revision) for index in range(count))
            return CommitPlan(self.ledger.epoch_id, request_id, self.policy_revision,
                "unselected-coding-initial-pilots-v1", "actual-coding-verifier-v1",
                snapshot["total_budget"], snapshot["spent"], snapshot["reserved"], count,
                snapshot["remaining"] - count, (), SelectionResult((), (), (), ()),
                (CommitAllocation(task.task_id, self.task_revision, count, ()),), reservations, ())

        return self.ledger._reserve(request_id, request_sha, build)

    def dispatch(self, reservation, *, owner_id, provider):
        if type(reservation) is not Reservation:
            raise ValueError("typed_existing_reservation_required")
        _hash(reservation.reservation_id, "reservation_id")
        _identifier(owner_id, "owner_id")
        if (reservation.task_id, reservation.task_revision, reservation.policy_revision) != (
                self._task["task_id"], self.task_revision, self.policy_revision):
            raise ValueError("dispatch_binding_does_not_match_exact_task_policy")
        if getattr(provider, "policy_revision", None) != self.policy_revision or not callable(getattr(provider, "next_action", None)):
            raise ValueError("generic_provider_exact_policy_binding_required")
        self._guard()
        with self.ledger._transaction() as db:
            actual = self.ledger._reservation(db, reservation.reservation_id)
            if tuple(actual[name] for name in ("task_id", "task_revision", "policy_revision")) != (
                    reservation.task_id, reservation.task_revision, reservation.policy_revision):
                raise ValueError("exact_database_reservation_required")
        granted = self.ledger.claim(reservation.reservation_id, owner_id=owner_id,
            current_policy_revision=self.policy_revision, current_task_revision=self.task_revision)
        if not granted:
            return CodingDispatch(False)
        started = time.perf_counter_ns()
        started_at = datetime.now(timezone.utc).isoformat()
        costs = {"provider_ns": 0, "provider_calls": 0, "reset_ns": 0,
                 "actions_ns": 0, "verification_ns": 0, "cleanup_ns": 0}
        root = self.output / reservation.reservation_id
        adapter, env, verification = None, None, None
        root_created = False
        failure, phase = None, "setup"
        status, reward = "unresolved", None
        try:
            root.mkdir()
            root_created = True
            adapter = DockerCodingAdapter(output_root=root / "runtime", **self._options)
            registry = RealWorldTaskRegistry(self.ledger.path)
            try:
                registry.register(self._task)
            finally:
                registry.close()
            env = RealWorldEnv(self._task, adapter)
            before = time.perf_counter_ns()
            try:
                observation = env.reset(self.policy_revision)
            finally:
                costs["reset_ns"] += time.perf_counter_ns() - before
            phase = "policy_actions"
            for index in range(self._task["budgets"]["max_actions"]):
                if env.status != "active":
                    break
                before = time.perf_counter_ns()
                try:
                    action = provider.next_action(copy.deepcopy(observation),
                        action_index=index, reservation_id=reservation.reservation_id)
                finally:
                    costs["provider_ns"] += time.perf_counter_ns() - before
                    costs["provider_calls"] += 1
                if (type(action) is not dict or any(name in action for name in
                        ("reward", "passed", "verification", "terminal_reward"))):
                    raise ValueError("provider_may_supply_only_repository_tool_actions")
                if getattr(provider, "policy_revision", None) != self.policy_revision:
                    raise ValueError("provider_policy_changed_during_dispatch")
                before = time.perf_counter_ns()
                try:
                    transition = env.step(action)
                finally:
                    costs["actions_ns"] += time.perf_counter_ns() - before
                observation = transition["observation"]
            phase = "repository_verification"
            if env.status == "pending":
                before = time.perf_counter_ns()
                try:
                    verification = env.verify()
                finally:
                    costs["verification_ns"] += time.perf_counter_ns() - before
            if env.status == "graded":
                # The actor's submitted bytes and trusted verifier must still
                # match the adapter's genuine immutable grading inputs.
                if (type(adapter) is not DockerCodingAdapter or adapter.verified is None or
                        verification["evidence"] != adapter.verified["evidence"] or
                        _workspace_digest(adapter.submitted_workspace) != adapter.submitted_workspace.name or
                        _workspace_digest(self.verifier_dir) != self.binding["verifier_sha256"]):
                    raise ValueError("actual_adapter_verified_workspace_required")
                reward = _case_reward(verification, specification=self._spec,
                    verifier_sha256=self.binding["verifier_sha256"], image_sha256=self.binding["image_sha256"])
                if verification["evidence"].get("workspace_sha256") != adapter.submitted_workspace.name:
                    raise ValueError("actual_submitted_workspace_digest_required")
                status = "resolved_binary"
            elif env.status == "interrupted":
                status = "unresolved"
            else:
                status = "unresolved"
        except Exception as exc:
            # Provider exception messages can contain private backend details.
            # Record type/phase only; never serialize the provider or its repr.
            failure = {"phase": phase, "error_type": type(exc).__name__}
            status, reward = "unresolved", None
        finally:
            if adapter is not None:
                before = time.perf_counter_ns()
                try:
                    adapter.close()
                except Exception as exc:
                    failure = {"phase": "cleanup", "error_type": type(exc).__name__}
                    status, reward = "unresolved", None
                finally:
                    costs["cleanup_ns"] = time.perf_counter_ns() - before
            inputs_unchanged = False
            try:
                self._guard()
                if getattr(provider, "policy_revision", None) != self.policy_revision:
                    raise ValueError("provider_policy_changed")
                inputs_unchanged = True
            except Exception as exc:
                failure = {"phase": "final_revision_guard", "error_type": type(exc).__name__}
                status, reward = "unresolved", None
            costs["wall_ns"] = time.perf_counter_ns() - started
        costs["actions_used"] = env.actions_used if env is not None else 0
        measured = {"schema_version": "actual-coding-verification-v1", "epoch_id": self.ledger.epoch_id,
            "reservation": asdict(reservation), "owner_id": owner_id, "started_at": started_at,
            "task_revision": self.task_revision, "policy_revision": self.policy_revision,
            "artifact_binding": self.binding, "source_sha256": self.source_sha256,
            "inputs_unchanged_after_execution": inputs_unchanged,
            "terminal_status": env.status if env is not None else None,
            "verification": verification, "case_count": len(self._spec["cases"]),
            "cost": costs, "cost_scope": "wall_includes_reset_provider_actions_verifier_cleanup_final_readonly_guards; excludes_admission_claim_and_persistence",
            "budget_units": "one_dispatched_episode", "provider_identity_recorded": False,
            "optimizer_executed": False, "failure": failure}
        verification_id = "coding:" + _digest(measured) if status == "resolved_binary" else None
        receipt = PilotReceipt(_digest([self.ledger.epoch_id, reservation.reservation_id,
            self.task_revision, self.policy_revision]), self._task["task_id"], self.task_revision,
            self.policy_revision, status, units=1, reward=reward, verification_id=verification_id)
        record = {"schema_version": "coding-pilot-outcome-v1", "receipt": asdict(receipt), "execution": measured}
        payload, record_sha = _json(record), _digest(record)
        # Persist genuine outcome before completion; a crash in between can be
        # completed explicitly without rerunning inference or the repository.
        with self.ledger._transaction() as db:
            db.execute("INSERT INTO coding_pilot_outcomes VALUES (?,?,?,?,?)", (
                self.ledger.epoch_id, reservation.reservation_id, receipt.receipt_id, record_sha, payload))
        self.ledger.complete(reservation.reservation_id, owner_id=owner_id, receipt=receipt)
        if root_created:
            (root / "outcome.json").write_text(json.dumps(record, indent=2) + "\n")
            if env is not None:
                trajectory = env.export_trajectory()
                if trajectory is not None:
                    (root / "trajectory.json").write_text(json.dumps(trajectory, indent=2) + "\n")
        return CodingDispatch(True, receipt, record_sha)

    def _record(self, *, receipt_id=None, reservation_id=None):
        if (receipt_id is None) == (reservation_id is None):
            raise ValueError("one_outcome_identity_required")
        key, value = ("receipt_id", receipt_id) if receipt_id is not None else ("reservation_id", reservation_id)
        with self.ledger._transaction() as db:
            row = db.execute("SELECT * FROM coding_pilot_outcomes WHERE epoch_id=? AND " + key + "=?",
                             (self.ledger.epoch_id, value)).fetchone()
        if row is None:
            return None
        record = json.loads(row["payload"])
        if _digest(record) != row["outcome_sha256"] or record.get("schema_version") != "coding-pilot-outcome-v1":
            raise ValueError("authoritative_coding_outcome_corrupted")
        return record

    def verify_receipt(self, receipt):
        """Planner verifier backed by producer evidence, not an agent assertion."""
        if type(receipt) is not PilotReceipt or receipt.status != "resolved_binary":
            return False
        try:
            record = self._record(receipt_id=receipt.receipt_id)
            if not self._record_matches_receipt(record, receipt):
                return False
            return self._spent_receipt_matches(record, receipt)
        except (ValueError, KeyError, TypeError):
            return False

    def _record_matches_receipt(self, record, receipt):
        if record is None or record["receipt"] != asdict(receipt):
            return False
        execution = record["execution"]
        reservation = execution["reservation"]
        common = (execution.get("schema_version") == "actual-coding-verification-v1" and
            execution["epoch_id"] == self.ledger.epoch_id and
            execution["task_revision"] == receipt.task_revision == self.task_revision and
            execution["policy_revision"] == receipt.policy_revision == self.policy_revision and
            execution["artifact_binding"] == self.binding and execution["source_sha256"] == self.source_sha256 and
            execution["case_count"] == len(self._spec["cases"]) and receipt.units == 1 and
            (reservation["task_id"], reservation["task_revision"], reservation["policy_revision"]) ==
            (receipt.task_id, receipt.task_revision, receipt.policy_revision) and
            all(type(value) is int and value >= 0 for value in execution["cost"].values()) and
            execution["cost"].get("wall_ns", 0) > 0)
        if not common:
            return False
        if receipt.status != "resolved_binary":
            return receipt.status in ("failed", "unresolved") and receipt.reward is None and receipt.verification_id is None
        return (execution["inputs_unchanged_after_execution"] is True and
            execution["terminal_status"] == "graded" and execution["failure"] is None and
            receipt.verification_id == "coding:" + _digest(execution) and
            _case_reward(execution["verification"], specification=self._spec,
                verifier_sha256=self.binding["verifier_sha256"], image_sha256=self.binding["image_sha256"]) == receipt.reward)

    def _spent_receipt_matches(self, record, receipt):
        with self.ledger._transaction() as db:
            row = db.execute("SELECT p.payload,p.reservation_id,r.state,r.receipt_id,r.owner_id FROM pilot_receipts p "
                "JOIN pilot_reservations r ON p.epoch_id=r.epoch_id AND p.reservation_id=r.reservation_id "
                "WHERE p.epoch_id=? AND p.receipt_id=?", (self.ledger.epoch_id, receipt.receipt_id)).fetchone()
        return (row is not None and row["state"] == "spent" and row["receipt_id"] == receipt.receipt_id and
            row["payload"] == _json(asdict(receipt)) and
            row["reservation_id"] == record["execution"]["reservation"]["reservation_id"] and
            row["owner_id"] == record["execution"]["owner_id"])

    def recover_completion(self, reservation, *, owner_id):
        """Finish a persisted claimed outcome; never rerun the episode."""
        if type(reservation) is not Reservation:
            raise ValueError("typed_reservation_required")
        record = self._record(reservation_id=reservation.reservation_id)
        if record is None or record["execution"]["reservation"] != asdict(reservation) or record["execution"]["owner_id"] != owner_id:
            raise ValueError("actual_persisted_owned_outcome_required")
        receipt = PilotReceipt(**record["receipt"])
        if not self._record_matches_receipt(record, receipt):
            raise ValueError("persisted_repository_verification_invalid")
        return self.ledger.complete(reservation.reservation_id, owner_id=owner_id, receipt=receipt)

    def plan_commits(self, *, request_id, pilots, selector_backend, selector_id,
                     commit_floor=2, commit_cap=8, **selection_options):
        """Select already charged genuine pilots through the existing planner.

        Core record_pilots treats reserved completions as conflicting new past
        cost records. This narrow facade validates their spent reservation and
        exact persisted outcome instead; it neither refunds nor double-charges.
        """
        self._guard()
        producer = self

        class PrechargedLedger:
            epoch_id = producer.ledger.epoch_id

            def record_pilots(self, receipts):
                for receipt in receipts:
                    record = producer._record(receipt_id=receipt.receipt_id)
                    if not producer._record_matches_receipt(record, receipt):
                        raise ValueError("actual_coding_pilot_outcome_required")
                    if not producer._spent_receipt_matches(record, receipt):
                        raise ValueError("same_epoch_precharged_pilot_required")
                return producer.ledger.snapshot()

            def _reserve(self, *args):
                return producer.ledger._reserve(*args)

        return plan_pilot_commit(PrechargedLedger(), request_id=request_id,
            current_policy_revision=self.policy_revision,
            tasks=(self.task(commit_floor=commit_floor, commit_cap=commit_cap),), pilots=pilots,
            selector_backend=selector_backend, evidence_verifier=self.verify_receipt,
            selector_id=selector_id, evidence_verifier_id="actual-coding-outcome-SQLite-v1", **selection_options)
