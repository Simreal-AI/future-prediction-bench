"""Host-owned external-operation fence for checkpointed QEMU episodes.

This is a deliberately narrow transfer of execution-edit safety: a stable
operation ID survives a VM restore, while an unresolved external request
prevents snapshot/restore. It does not implement the exact workflow checker,
fork/merge rules, or arbitrary guest egress mediation in Zheng et al.

The journal must live outside every restorable guest disk. The only supported
external-effect path is ``dispatch``; a guest with an alternate network or
device path would void this guarantee. SQLite FULL/WAL protects process-crash
recovery on the same host; power-loss durability depends on the filesystem.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path

PROVIDER_KEY_FORMAT = "fpb-op-v1-json-pair-sha256"


class ExternalEffectError(RuntimeError):
    """An external effect or execution edit could not be proved safe."""


class ExternalEffectPending(ExternalEffectError):
    """The provider outcome is unknown; a blind retry is forbidden."""


def _json_bytes(value):
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("operation_payload_must_be_json") from exc
    if len(encoded) > 1_000_000:
        raise ValueError("operation_payload_too_large")
    return encoded


def _identity(value, label):
    if (not isinstance(value, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", value) is None):
        raise ValueError(f"invalid_{label}")
    return value


def _provider_key(episode_id, call_id):
    # Canonical pair encoding avoids delimiter ambiguity when either
    # identity contains ':'. The format prefix versions provider-facing IDs;
    # SHA256 keeps them bounded without exposing the original identifiers.
    digest = hashlib.sha256(_json_bytes([episode_id, call_id])).hexdigest()
    return "fpb-op-v1:" + digest


class HostOperationJournal:
    """One same-host, restore-domain-external operation ledger per episode.

    The caller must supply a stable call ID for the same logical operation on
    replay. Different payloads or routes under that ID are conflicts. The
    effect callback receives a versioned hash of the canonical episode/call
    identity pair to pass to
    a provider supporting identity-based status lookup. If a call fails after
    dispatch, its status stays pending until an authoritative provider lookup
    supplies the committed result. Neither this class nor restore retries an
    uncertain effect.
    """

    def __init__(self, path, episode_id):
        self.episode_id = _identity(episode_id, "episode_id")
        raw_path = Path(path)
        if raw_path.is_symlink() or raw_path.parent.is_symlink():
            raise ValueError("operation_journal_symlink_forbidden")
        if not raw_path.parent.is_dir():
            raise ValueError("operation_journal_parent_required")
        self.path = raw_path.resolve()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("""CREATE TABLE IF NOT EXISTS journal_scope (
                singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                episode_id TEXT NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS operations (
                call_id TEXT PRIMARY KEY,
                route TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL,
                state TEXT NOT NULL CHECK(state IN ('pending','completed')),
                result_json TEXT,
                CHECK ((state='completed') = (result_json IS NOT NULL)))""")
            db.execute("""CREATE TABLE IF NOT EXISTS operation_protocol (
                singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                provider_key_format TEXT NOT NULL)""")
            row = db.execute("SELECT episode_id FROM journal_scope WHERE singleton=1").fetchone()
            if row is None:
                db.execute("INSERT INTO journal_scope VALUES (1,?)", (self.episode_id,))
            elif row[0] != self.episode_id:
                raise ExternalEffectError("operation_journal_episode_mismatch")
            protocol = db.execute("SELECT provider_key_format FROM operation_protocol WHERE singleton=1").fetchone()
            if protocol is None:
                # Legacy ledgers did not persist provider-key encoding. A
                # pending effect might already exist under its original key;
                # changing that key for lookup/retry would be unsafe.
                if db.execute("SELECT 1 FROM operations WHERE state='pending' LIMIT 1").fetchone():
                    raise ExternalEffectPending("legacy_pending_provider_key_requires_manual_reconciliation")
                db.execute("INSERT INTO operation_protocol VALUES (1,?)", (PROVIDER_KEY_FORMAT,))
            elif protocol[0] != PROVIDER_KEY_FORMAT:
                raise ExternalEffectError("unsupported_provider_key_format")
            db.execute("COMMIT")
        os.chmod(self.path, 0o600)

    @contextmanager
    def _connect(self):
        if self.path.is_symlink():
            raise ExternalEffectError("operation_journal_symlink_forbidden")
        db = sqlite3.connect(self.path, isolation_level=None, timeout=30.0)
        try:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=FULL")
            db.execute("PRAGMA busy_timeout=30000")
        except BaseException:
            db.close()
            raise
        try:
            yield db
        finally:
            db.close()

    def _row(self, db, call_id):
        return db.execute("""SELECT route,payload_sha256,state,result_json
                             FROM operations WHERE call_id=?""", (call_id,)).fetchone()

    @staticmethod
    def _check_row(row, route, payload_sha256):
        if row[0] != route or row[1] != payload_sha256:
            raise ExternalEffectError("stable_call_id_conflicts_with_prior_operation")

    def inspect(self, call_id):
        """Trusted diagnostic; the returned state is never guest-visible."""
        call_id = _identity(call_id, "call_id")
        with self._connect() as db:
            row = self._row(db, call_id)
        if row is None:
            return None
        return {"route": row[0], "payload_sha256": row[1], "state": row[2],
                "result": None if row[3] is None else json.loads(row[3])}

    def dispatch(self, call_id, route, payload, effect):
        """Record intent before invoking ``effect``; never blindly redispatch.

        ``effect(stable_key, route, payload)`` must be the *sole* path to the
        provider. A completed record returns its exact stored JSON result.
        A pending record, including one left by a lost response, blocks retry.
        """
        call_id = _identity(call_id, "call_id")
        route = _identity(route, "route")
        if not callable(effect):
            raise TypeError("effect must be callable")
        frozen_payload_bytes = _json_bytes(payload)
        frozen_payload = json.loads(frozen_payload_bytes)
        payload_sha256 = hashlib.sha256(frozen_payload_bytes).hexdigest()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = self._row(db, call_id)
            if row is not None:
                self._check_row(row, route, payload_sha256)
                if row[2] != "completed":
                    raise ExternalEffectPending("external_operation_outcome_unknown")
                return json.loads(row[3])
            db.execute("""INSERT INTO operations
                (call_id,route,payload_sha256,state,result_json)
                VALUES (?,?,?,'pending',NULL)""",
                (call_id, route, payload_sha256))
            db.execute("COMMIT")

        # This callback is deliberately outside the SQLite transaction. A
        # process death or lost response leaves a durable pending record.
        result = effect(_provider_key(self.episode_id, call_id), route,
                        copy.deepcopy(frozen_payload))
        encoded_result = _json_bytes(result).decode("utf-8")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = self._row(db, call_id)
            if row is None:
                raise ExternalEffectError("operation_record_missing_after_dispatch")
            self._check_row(row, route, payload_sha256)
            if row[2] == "completed":
                if row[3] != encoded_result:
                    raise ExternalEffectError("provider_receipt_conflicts_with_record")
            else:
                db.execute("""UPDATE operations SET state='completed',result_json=?
                              WHERE call_id=? AND state='pending'""",
                           (encoded_result, call_id))
            db.execute("COMMIT")
        return json.loads(encoded_result)

    def reconcile(self, call_id, lookup):
        """Settle a pending call from an authoritative, read-only lookup.

        ``lookup(stable_key, route, payload_sha256)`` must return either
        ``{'status': 'unknown'}`` or ``{'status': 'committed', 'result': ...}``.
        An unknown outcome stays pending. Lookup never dispatches the effect.
        The operator must trust the provider's query and receipt semantics.
        """
        call_id = _identity(call_id, "call_id")
        if not callable(lookup):
            raise TypeError("lookup must be callable")
        with self._connect() as db:
            row = self._row(db, call_id)
        if row is None:
            raise ExternalEffectError("operation_not_recorded")
        if row[2] == "completed":
            return json.loads(row[3])
        answer = lookup(_provider_key(self.episode_id, call_id), row[0], row[1])
        if not isinstance(answer, dict) or answer.get("status") not in {
                "unknown", "committed"} or set(answer) != (
                {"status"} if answer.get("status") == "unknown"
                else {"status", "result"}):
            raise ExternalEffectError("invalid_provider_lookup_receipt")
        if answer["status"] == "unknown":
            raise ExternalEffectPending("external_operation_outcome_unknown")
        encoded_result = _json_bytes(answer["result"]).decode("utf-8")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            current = self._row(db, call_id)
            if current is None or current[:2] != row[:2]:
                raise ExternalEffectError("operation_record_changed_during_reconcile")
            if current[2] == "completed":
                if current[3] != encoded_result:
                    raise ExternalEffectError("provider_receipt_conflicts_with_record")
            else:
                db.execute("""UPDATE operations SET state='completed',result_json=?
                              WHERE call_id=? AND state='pending'""",
                           (encoded_result, call_id))
            db.execute("COMMIT")
        return json.loads(encoded_result)

    def bounded_vm_edit(self, callback):
        """Atomically exclude new dispatches while a VM snapshot/edit runs.

        This is conservative: an unresolved effect refuses checkpoint/restore.
        The lock is SQLite's cross-process writer lock, held until the VM
        operation finishes; the callback must not itself dispatch an effect.
        """
        if not callable(callback):
            raise TypeError("callback must be callable")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT call_id FROM operations WHERE state='pending' LIMIT 1").fetchone()
            if row is not None:
                raise ExternalEffectPending("external_operation_outcome_unknown")
            result = callback()
            db.execute("COMMIT")
            return result


class FencedMicroVMRuntime:
    """Opt-in facade for a trusted MicroVMRuntime checkpoint/restore path.

    Keep the raw runtime private to the host coordinator. Direct calls to the
    underlying runtime or unmediated guest egress bypass this Python facade.
    Fork and merge are refused: this narrowed implementation has no safe
    branch-identity rule for an external operation.
    """

    def __init__(self, runtime, journal: HostOperationJournal):
        if not isinstance(journal, HostOperationJournal):
            raise TypeError("journal must be HostOperationJournal")
        if journal.path == Path(runtime.disk_path).resolve():
            raise ValueError("operation_journal_must_be_outside_vm_disk")
        self._runtime = runtime
        self.journal = journal

    def __getattr__(self, name):
        return getattr(self._runtime, name)

    def save_snapshot(self, *args, **kwargs):
        return self.journal.bounded_vm_edit(
            lambda: self._runtime.save_snapshot(*args, **kwargs))

    def load_snapshot(self, *args, **kwargs):
        return self.journal.bounded_vm_edit(
            lambda: self._runtime.load_snapshot(*args, **kwargs))

    def fork_snapshot(self, *args, **kwargs):
        raise ExternalEffectError("external_operation_fork_not_supported")

    def external_operation(self, call_id, route, payload, effect):
        return self.journal.dispatch(call_id, route, payload, effect)
