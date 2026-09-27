"""Bounded pilot/commit allocation with explicit trusted evidence and callbacks.

This is a control plane. It does not generate model tokens, compute policy
gradients, or certify agent-provided rewards. Callers own the authoritative
verifier and must claim a persistent reservation before dispatching work.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from typing import Callable, Mapping, Sequence

MAX_PLAN_RESERVATIONS = 65_536


def _text(value, field):
    if type(value) is not str or not value.strip() or len(value) > 256 or any(ord(c) < 32 for c in value):
        raise ValueError(f"{field}_must_be_a_bounded_nonblank_identifier")


def _integer(value, field, minimum=0):
    if type(value) is not int or not minimum <= value <= 1_000_000_000:
        raise ValueError(f"{field}_must_be_a_bounded_integer")


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value):
    return hashlib.sha256(_encoded(value).encode()).hexdigest()


@dataclass(frozen=True)
class TaskRevision:
    task_id: str
    task_revision: str
    commit_floor: int = 2
    commit_cap: int = 8

    def __post_init__(self):
        _text(self.task_id, "task_id")
        _text(self.task_revision, "task_revision")
        _integer(self.commit_floor, "commit_floor")
        _integer(self.commit_cap, "commit_cap")
        if self.commit_cap < self.commit_floor:
            raise ValueError("commit_cap_below_floor")


@dataclass(frozen=True)
class PilotReceipt:
    receipt_id: str
    task_id: str
    task_revision: str
    policy_revision: str
    status: str
    units: int = 1
    reward: int | None = None
    verification_id: str | None = None

    def __post_init__(self):
        for field in ("receipt_id", "task_id", "task_revision", "policy_revision"):
            _text(getattr(self, field), field)
        _integer(self.units, "spent_units", 1)
        if type(self.status) is not str or self.status not in ("resolved_binary", "failed", "unresolved", "excluded"):
            raise ValueError("unknown_receipt_status")
        if self.status == "resolved_binary":
            if type(self.reward) is not int or self.reward not in (0, 1):
                raise ValueError("resolved_reward_requires_explicit_integer_zero_or_one")
            _text(self.verification_id, "verification_id")
        elif self.reward is not None or self.verification_id is not None:
            raise ValueError("nonresolved_receipt_cannot_supply_a_reward")


@dataclass(frozen=True)
class SelectionThresholds:
    lower: float = .125
    upper: float = .25
    exclude: float | None = 1.0

    def __post_init__(self):
        for field in ("lower", "upper", "exclude"):
            value = getattr(self, field)
            if field == "exclude" and value is None:
                continue
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("finite_probability_threshold_required")
            object.__setattr__(self, field, float(value))
        if self.lower > 1 - self.upper:
            raise ValueError("selection_thresholds_have_no_keep_interval")


@dataclass(frozen=True)
class SelectorInput:
    task_ids: tuple[str, ...]
    prompt_indices: tuple[int, ...]
    rewards: tuple[int, ...]
    thresholds: SelectionThresholds


@dataclass(frozen=True)
class SelectionResult:
    keep: tuple[str, ...]
    too_correct: tuple[str, ...]
    too_incorrect: tuple[str, ...]
    exclude_too_easy: tuple[str, ...]


@dataclass(frozen=True)
class ReceiptExclusion:
    receipt_id: str
    reason: str


@dataclass(frozen=True)
class CommitAllocation:
    task_id: str
    task_revision: str
    count: int
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class Reservation:
    reservation_id: str
    task_id: str
    task_revision: str
    policy_revision: str


@dataclass(frozen=True)
class CommitPlan:
    epoch_id: str
    request_id: str
    policy_revision: str
    selector_id: str
    evidence_verifier_id: str
    total_budget: int
    spent_at_creation: int
    reserved_before_plan: int
    reserved_by_plan: int
    unallocated_at_creation: int
    pilot_receipt_ids: tuple[str, ...]
    selection: SelectionResult
    allocations: tuple[CommitAllocation, ...]
    reservations: tuple[Reservation, ...]
    ineligible_receipts: tuple[ReceiptExclusion, ...]


def _decode_plan(payload):
    value = json.loads(payload)
    value["selection"] = SelectionResult(**{key: tuple(ids) for key, ids in value["selection"].items()})
    value["allocations"] = tuple(CommitAllocation(**{**item, "evidence_ids": tuple(item["evidence_ids"])}) for item in value["allocations"])
    value["reservations"] = tuple(Reservation(**item) for item in value["reservations"])
    value["ineligible_receipts"] = tuple(ReceiptExclusion(**item) for item in value["ineligible_receipts"])
    value["pilot_receipt_ids"] = tuple(value["pilot_receipt_ids"])
    return CommitPlan(**value)


class PilotBudgetLedger:
    """Transactional budget shared by clients using this SQLite file and epoch.

    A successful claim is required before work is dispatched. Repeated claims
    return False; an uncertain claimed operation is not automatically refunded
    or retried. This ledger does not prevent a caller from bypassing it.
    """

    def __init__(self, path: str | Path, *, epoch_id: str, total_budget: int):
        _text(epoch_id, "epoch_id")
        _integer(total_budget, "total_budget")
        if str(path) == ":memory:":
            raise ValueError("persistent_sqlite_path_required")
        self._path = Path(path)
        self._epoch_id = epoch_id
        self._total_budget = total_budget
        with self._transaction() as db:
            db.execute("CREATE TABLE IF NOT EXISTS pilot_epochs (epoch_id TEXT PRIMARY KEY, total_budget INTEGER NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS pilot_receipts (epoch_id TEXT, receipt_id TEXT, payload TEXT NOT NULL, units INTEGER NOT NULL, reservation_id TEXT, PRIMARY KEY(epoch_id, receipt_id))")
            db.execute("CREATE TABLE IF NOT EXISTS pilot_plans (epoch_id TEXT, request_id TEXT, request_sha256 TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(epoch_id, request_id))")
            db.execute("CREATE TABLE IF NOT EXISTS pilot_reservations (epoch_id TEXT, reservation_id TEXT, task_id TEXT NOT NULL, task_revision TEXT NOT NULL, policy_revision TEXT NOT NULL, state TEXT NOT NULL CHECK(state IN ('pending','claimed','spent','cancelled')), owner_id TEXT, receipt_id TEXT, PRIMARY KEY(epoch_id, reservation_id))")
            row = db.execute("SELECT total_budget FROM pilot_epochs WHERE epoch_id=?", (epoch_id,)).fetchone()
            if row is not None and row[0] != total_budget:
                raise ValueError("budget_epoch_configuration_conflict")
            db.execute("INSERT OR IGNORE INTO pilot_epochs VALUES (?,?)", (epoch_id, total_budget))

    @property
    def path(self):
        return self._path

    @property
    def epoch_id(self):
        return self._epoch_id

    @property
    def total_budget(self):
        return self._total_budget

    @contextmanager
    def _transaction(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def _snapshot(self, db):
        budget = db.execute("SELECT total_budget FROM pilot_epochs WHERE epoch_id=?", (self.epoch_id,)).fetchone()[0]
        spent = db.execute("SELECT COALESCE(SUM(units),0) FROM pilot_receipts WHERE epoch_id=?", (self.epoch_id,)).fetchone()[0]
        reserved = db.execute("SELECT COUNT(*) FROM pilot_reservations WHERE epoch_id=? AND state IN ('pending','claimed')", (self.epoch_id,)).fetchone()[0]
        return {"total_budget": budget, "spent": spent, "reserved": reserved, "remaining": budget - spent - reserved}

    def snapshot(self):
        with self._transaction() as db:
            return self._snapshot(db)

    def record_pilots(self, receipts: Sequence[PilotReceipt]):
        """Record actual past costs once, including failed and stale pilots."""
        checked = _unique_receipts(receipts)
        with self._transaction() as db:
            for receipt in checked:
                payload = _encoded(asdict(receipt))
                old = db.execute("SELECT payload,reservation_id FROM pilot_receipts WHERE epoch_id=? AND receipt_id=?", (self.epoch_id, receipt.receipt_id)).fetchone()
                if old is not None:
                    if old["payload"] != payload or old["reservation_id"] is not None:
                        raise ValueError("pilot_receipt_identity_conflict")
                    continue
                db.execute("INSERT INTO pilot_receipts VALUES (?,?,?,?,NULL)", (self.epoch_id, receipt.receipt_id, payload, receipt.units))
            if self._snapshot(db)["remaining"] < 0:
                raise ValueError("pilot_costs_exceed_remaining_epoch_budget")
        return self.snapshot()

    def _reserve(self, request_id, request_sha256, builder):
        with self._transaction() as db:
            previous = db.execute("SELECT request_sha256,payload FROM pilot_plans WHERE epoch_id=? AND request_id=?", (self.epoch_id, request_id)).fetchone()
            if previous is not None:
                if previous["request_sha256"] != request_sha256:
                    raise ValueError("plan_request_identity_conflict")
                return _decode_plan(previous["payload"])
            snapshot = self._snapshot(db)
            plan = builder(snapshot)
            if plan.reserved_by_plan > snapshot["remaining"]:
                raise ValueError("commit_reservations_exceed_remaining_budget")
            for item in plan.reservations:
                db.execute("INSERT INTO pilot_reservations VALUES (?,?,?,?,?,'pending',NULL,NULL)", (self.epoch_id, item.reservation_id, item.task_id, item.task_revision, item.policy_revision))
            db.execute("INSERT INTO pilot_plans VALUES (?,?,?,?)", (self.epoch_id, request_id, request_sha256, _encoded(asdict(plan))))
            if self._snapshot(db)["remaining"] < 0:
                raise ValueError("epoch_budget_conservation_failed")
            return plan

    def _reservation(self, db, reservation_id):
        _text(reservation_id, "reservation_id")
        row = db.execute("SELECT * FROM pilot_reservations WHERE epoch_id=? AND reservation_id=?", (self.epoch_id, reservation_id)).fetchone()
        if row is None:
            raise ValueError("unknown_reservation")
        return row

    def claim(self, reservation_id: str, *, owner_id: str, current_policy_revision: str, current_task_revision: str) -> bool:
        """Atomically authorize one dispatch; False means do not dispatch again."""
        for name, value in (("owner_id", owner_id), ("current_policy_revision", current_policy_revision), ("current_task_revision", current_task_revision)):
            _text(value, name)
        with self._transaction() as db:
            row = self._reservation(db, reservation_id)
            if row["policy_revision"] != current_policy_revision or row["task_revision"] != current_task_revision:
                raise ValueError("reservation_revision_changed_before_dispatch")
            if row["state"] == "cancelled":
                raise ValueError("reservation_cancelled")
            if row["state"] != "pending":
                if row["owner_id"] != owner_id:
                    raise ValueError("reservation_owned_by_another_worker")
                return False
            db.execute("UPDATE pilot_reservations SET state='claimed',owner_id=? WHERE epoch_id=? AND reservation_id=?", (owner_id, self.epoch_id, reservation_id))
            return True

    def complete(self, reservation_id: str, *, owner_id: str, receipt: PilotReceipt) -> bool:
        """Charge one claimed dispatch, including failures, without double spend."""
        _text(owner_id, "owner_id")
        if type(receipt) is not PilotReceipt:
            raise ValueError("typed_pilot_receipt_required")
        with self._transaction() as db:
            row = self._reservation(db, reservation_id)
            if row["owner_id"] != owner_id or row["state"] not in ("claimed", "spent"):
                raise ValueError("claimed_reservation_and_owner_required")
            if receipt.units != 1 or (receipt.task_id, receipt.task_revision, receipt.policy_revision) != (row["task_id"], row["task_revision"], row["policy_revision"]):
                raise ValueError("completion_receipt_does_not_match_reservation")
            payload = _encoded(asdict(receipt))
            old = db.execute("SELECT payload,reservation_id FROM pilot_receipts WHERE epoch_id=? AND receipt_id=?", (self.epoch_id, receipt.receipt_id)).fetchone()
            if row["state"] == "spent":
                if row["receipt_id"] != receipt.receipt_id or old is None or old["payload"] != payload or old["reservation_id"] != reservation_id:
                    raise ValueError("completed_reservation_receipt_conflict")
                return False
            if old is not None:
                raise ValueError("completion_receipt_already_spent")
            db.execute("INSERT INTO pilot_receipts VALUES (?,?,?,?,?)", (self.epoch_id, receipt.receipt_id, payload, 1, reservation_id))
            db.execute("UPDATE pilot_reservations SET state='spent',receipt_id=? WHERE epoch_id=? AND reservation_id=?", (receipt.receipt_id, self.epoch_id, reservation_id))
            return True

    def cancel_pending(self, reservation_id: str) -> bool:
        """Release only a reservation that has never authorized a dispatch."""
        with self._transaction() as db:
            row = self._reservation(db, reservation_id)
            if row["state"] == "cancelled":
                return False
            if row["state"] != "pending":
                raise ValueError("claimed_or_spent_work_cannot_be_refunded")
            db.execute("UPDATE pilot_reservations SET state='cancelled' WHERE epoch_id=? AND reservation_id=?", (self.epoch_id, reservation_id))
            return True


def _unique_receipts(receipts):
    unique = {}
    for receipt in receipts:
        if type(receipt) is not PilotReceipt:
            raise ValueError("typed_pilot_receipt_required")
        if receipt.receipt_id in unique and unique[receipt.receipt_id] != receipt:
            raise ValueError("conflicting_duplicate_pilot_receipt")
        unique[receipt.receipt_id] = receipt
    return tuple(unique[key] for key in sorted(unique))


def _selection(input: SelectorInput, output: Mapping[str, Sequence[int]]) -> SelectionResult:
    keys = ("keep", "too_correct", "too_incorrect", "exclude_too_easy")
    if not isinstance(output, Mapping) or set(output) != set(keys):
        raise ValueError("selector_categories_malformed")
    actual = {}
    for key in keys:
        ids = output[key]
        if not isinstance(ids, (tuple, list)) or any(type(index) is not int or not 0 <= index < len(input.task_ids) for index in ids) or len(set(ids)) != len(ids):
            raise ValueError("selector_prompt_ids_malformed")
        actual[key] = set(ids)
    expected = {key: set() for key in keys}
    for index in range(len(input.task_ids)):
        values = [reward for group, reward in zip(input.prompt_indices, input.rewards) if group == index]
        mean = sum(values) / len(values)
        category = "too_correct" if mean > 1-input.thresholds.upper else "too_incorrect" if mean < input.thresholds.lower else "keep"
        expected[category].add(index)
        if input.thresholds.exclude is not None and mean >= input.thresholds.exclude:
            expected["exclude_too_easy"].add(index)
    if actual != expected:
        raise ValueError("selector_result_does_not_match_trusted_evidence")
    return SelectionResult(**{key: tuple(input.task_ids[index] for index in sorted(actual[key])) for key in keys})


def _allocate(tasks, remaining):
    counts = {task.task_id: 0 for task in tasks}
    active = []
    # Stable order gives feasible floors to a deterministic subset, never
    # exceeding the budget merely to satisfy all configured minimums.
    for task in tasks:
        if task.commit_floor <= remaining:
            counts[task.task_id] = task.commit_floor
            remaining -= task.commit_floor
            active.append(task)
    while remaining and active:
        active = [task for task in active if counts[task.task_id] < task.commit_cap]
        if not active:
            break
        per_task = remaining // len(active)
        if not per_task:
            for task in active[:remaining]:
                counts[task.task_id] += 1
            remaining = 0
            break
        for task in active:
            increment = min(per_task, task.commit_cap-counts[task.task_id])
            counts[task.task_id] += increment
            remaining -= increment
    return counts


def plan_pilot_commit(
    ledger: PilotBudgetLedger, *, request_id: str, current_policy_revision: str,
    tasks: Sequence[TaskRevision], pilots: Sequence[PilotReceipt],
    selector_backend: Callable[[SelectorInput], Mapping[str, Sequence[int]]],
    evidence_verifier: Callable[[PilotReceipt], bool], selector_id: str,
    evidence_verifier_id: str, thresholds: SelectionThresholds = SelectionThresholds(),
    minimum_pilot_count: int = 2,
) -> CommitPlan:
    """Charge pilots, select trusted current evidence and reserve future commits.

    Trusted cost receipts are persisted before selector calls, so a failed
    selector cannot refund already spent work. Reward trust comes exclusively
    from the explicit caller-owned verifier, not from a field supplied by an
    agent. Selection and reservation do not generate the reserved rollouts.
    """
    for name, value in (("request_id", request_id), ("current_policy_revision", current_policy_revision), ("selector_id", selector_id), ("evidence_verifier_id", evidence_verifier_id)):
        _text(value, name)
    _integer(minimum_pilot_count, "minimum_pilot_count", 2)
    if type(thresholds) is not SelectionThresholds or not callable(selector_backend) or not callable(evidence_verifier):
        raise ValueError("explicit_selector_verifier_and_typed_thresholds_required")
    task_map = {}
    for task in tasks:
        if type(task) is not TaskRevision or task.task_id in task_map:
            raise ValueError("unique_typed_current_task_revisions_required")
        task_map[task.task_id] = task
    ordered_tasks = tuple(task_map[key] for key in sorted(task_map))
    receipts = _unique_receipts(pilots)
    ledger.record_pilots(receipts)
    grouped = {task.task_id: [] for task in ordered_tasks}
    exclusions = []
    for receipt in receipts:
        task = task_map.get(receipt.task_id)
        reason = ("task_not_in_epoch" if task is None else "task_revision_changed" if task.task_revision != receipt.task_revision else
                  "policy_revision_changed" if receipt.policy_revision != current_policy_revision else "outcome_not_resolved_binary" if receipt.status != "resolved_binary" else None)
        if reason is None:
            trusted = evidence_verifier(receipt)
            if type(trusted) is not bool:
                raise ValueError("evidence_verifier_must_return_explicit_bool")
            if not trusted:
                reason = "untrusted_reward_evidence"
        if reason is not None:
            exclusions.append(ReceiptExclusion(receipt.receipt_id, reason))
        else:
            grouped[receipt.task_id].append(receipt)
    eligible_ids = tuple(key for key in sorted(grouped) if len(grouped[key]) >= minimum_pilot_count)
    for key in sorted(grouped):
        if key not in eligible_ids:
            exclusions.extend(ReceiptExclusion(receipt.receipt_id, "insufficient_current_trusted_pilots") for receipt in grouped[key])
    selector_input = SelectorInput(eligible_ids,
        tuple(index for index, key in enumerate(eligible_ids) for _ in grouped[key]),
        tuple(receipt.reward for key in eligible_ids for receipt in grouped[key]), thresholds)
    selection = _selection(selector_input, selector_backend(selector_input)) if eligible_ids else SelectionResult((), (), (), ())
    selected_tasks = tuple(task_map[key] for key in selection.keep if key not in selection.exclude_too_easy)
    request_sha256 = _sha({"policy_revision": current_policy_revision, "tasks": [asdict(task) for task in ordered_tasks],
        "pilots": [asdict(receipt) for receipt in receipts], "thresholds": asdict(thresholds), "minimum_pilot_count": minimum_pilot_count,
        "selector_id": selector_id, "evidence_verifier_id": evidence_verifier_id,
        "selection": asdict(selection), "ineligible_receipts": [asdict(item) for item in exclusions]})

    def build(snapshot):
        counts = _allocate(selected_tasks, snapshot["remaining"])
        if sum(counts.values()) > MAX_PLAN_RESERVATIONS:
            raise ValueError("commit_plan_exceeds_bounded_batch_limit")
        allocations = tuple(CommitAllocation(task.task_id, task.task_revision, counts[task.task_id], tuple(item.receipt_id for item in grouped[task.task_id])) for task in selected_tasks)
        reservations = tuple(Reservation(_sha([ledger.epoch_id, request_id, task.task_id, task.task_revision, current_policy_revision, ordinal]),
                                        task.task_id, task.task_revision, current_policy_revision)
                             for task in selected_tasks for ordinal in range(counts[task.task_id]))
        return CommitPlan(ledger.epoch_id, request_id, current_policy_revision, selector_id, evidence_verifier_id,
            snapshot["total_budget"], snapshot["spent"], snapshot["reserved"], len(reservations), snapshot["remaining"]-len(reservations),
            tuple(item.receipt_id for item in receipts), selection, allocations, reservations, tuple(exclusions))

    return ledger._reserve(request_id, request_sha256, build)
