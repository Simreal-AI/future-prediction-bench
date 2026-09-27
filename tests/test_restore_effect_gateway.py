"""Fault and concurrency checks for the restore-domain-external effect fence."""

from __future__ import annotations

import tempfile
import threading
import sqlite3
import unittest
from pathlib import Path

from future_prediction_bench.restore_effect_gateway import (
    ExternalEffectError, ExternalEffectPending, FencedMicroVMRuntime,
    HostOperationJournal,
)


class FakeVM:
    def __init__(self, disk_path):
        self.disk_path = disk_path
        self.value = "before"
        self.saved = {}
        self.load_calls = 0
        self.save_calls = 0

    def save_snapshot(self, tag):
        self.saved[tag] = self.value
        self.save_calls += 1
        return {"kind": "full_vm_state_qcow2_v1", "tag": tag}

    def load_snapshot(self, tag, *, resume=False):
        self.value = self.saved[tag]
        self.load_calls += 1


class RestoreEffectGatewayTest(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.path = self.root / "host-operations.sqlite3"
        self.vm = FakeVM(self.root / "guest.qcow2")
        self.journal = HostOperationJournal(self.path, "episode-1")
        self.fenced = FencedMicroVMRuntime(self.vm, self.journal)

    def test_completed_effect_replayed_after_restore_without_second_dispatch(self):
        self.fenced.save_snapshot("before")
        deliveries = []

        def provider(key, route, payload):
            deliveries.append((key, route, payload))
            return {"receipt": "provider-7", "value": 42}

        original = self.fenced.external_operation(
            "tool-call-7", "payment", {"amount": 42}, provider)
        self.vm.value = "after"
        # Simulate a new host coordinator opening the same journal after a
        # process restart, while the VM restore rolls back guest state only.
        reopened = FencedMicroVMRuntime(
            self.vm, HostOperationJournal(self.path, "episode-1"))
        reopened.load_snapshot("before", resume=True)
        replay = reopened.external_operation(
            "tool-call-7", "payment", {"amount": 42}, provider)
        self.assertEqual(original, replay)
        self.assertEqual(self.vm.value, "before")
        self.assertEqual(self.vm.load_calls, 1)
        self.assertEqual(len(deliveries), 1)
        self.assertRegex(deliveries[0][0], r"^fpb-op-v1:[0-9a-f]{64}$")
        self.assertEqual(deliveries[0][1:], ("payment", {"amount": 42}))

    def test_lost_response_blocks_restore_until_authoritative_lookup(self):
        self.fenced.save_snapshot("before")
        deliveries = []

        def committed_then_lost(key, route, payload):
            deliveries.append(key)
            raise ConnectionError("response lost after provider commit")

        with self.assertRaises(ConnectionError):
            self.fenced.external_operation(
                "call-a", "payment", {"amount": 17}, committed_then_lost)
        self.assertEqual(self.journal.inspect("call-a")["state"], "pending")
        with self.assertRaises(ExternalEffectPending):
            self.fenced.load_snapshot("before")
        with self.assertRaises(ExternalEffectPending):
            self.fenced.save_snapshot("another")
        with self.assertRaises(ExternalEffectPending):
            self.journal.reconcile("call-a", lambda *_: {"status": "unknown"})
        self.assertEqual(self.vm.load_calls, 0)
        self.assertEqual(self.vm.save_calls, 1)

        reopened = HostOperationJournal(self.path, "episode-1")
        result = reopened.reconcile(
            "call-a", lambda key, route, digest: {
                "status": "committed", "result": {"receipt": key, "status": "paid"}})
        FencedMicroVMRuntime(self.vm, reopened).load_snapshot("before")
        replay = reopened.dispatch(
            "call-a", "payment", {"amount": 17}, committed_then_lost)
        self.assertEqual(result, replay)
        self.assertEqual(len(deliveries), 1)
        self.assertEqual(result["receipt"], deliveries[0])
        self.assertEqual(self.vm.load_calls, 1)

    def test_episode_call_pair_cannot_collide_at_colon_boundary(self):
        observed = []
        def provider(key, route, payload):
            observed.append(key)
            return {"key": key}
        first = HostOperationJournal(self.root / "colon-a.sqlite3", "a:b")
        second = HostOperationJournal(self.root / "colon-b.sqlite3", "a")
        first.dispatch("c", "route", {"x": 1}, provider)
        second.dispatch("b:c", "route", {"x": 2}, provider)
        self.assertEqual(len(observed), 2)
        self.assertNotEqual(observed[0], observed[1])
        self.assertRegex(observed[0], r"^fpb-op-v1:[0-9a-f]{64}$")
        self.assertRegex(observed[1], r"^fpb-op-v1:[0-9a-f]{64}$")
        # Reopening and replay retain each original key/result without a
        # new effect, independently of which identity contains the colon.
        replay = HostOperationJournal(self.root / "colon-a.sqlite3", "a:b").dispatch(
            "c", "route", {"x": 1}, provider)
        self.assertEqual(replay, {"key": observed[0]})
        self.assertEqual(len(observed), 2)

    def test_legacy_pending_key_format_cannot_be_silently_migrated(self):
        def lost(*_):
            raise ConnectionError("provider result unknown")
        with self.assertRaises(ConnectionError):
            self.journal.dispatch("pending", "route", {"x": 1}, lost)
        with sqlite3.connect(self.path) as db:
            db.execute("DROP TABLE operation_protocol")
        with self.assertRaisesRegex(ExternalEffectPending, "legacy_pending_provider_key"):
            HostOperationJournal(self.path, "episode-1")
        with sqlite3.connect(self.path) as db:
            self.assertEqual(db.execute("SELECT state FROM operations WHERE call_id='pending'").fetchone(),
                             ("pending",))

    def test_legacy_completed_calls_replay_without_provider_key_migration_effect(self):
        recorded = self.journal.dispatch("done", "route", {"x": 1}, lambda *_: {"old_receipt": "legacy-key"})
        with sqlite3.connect(self.path) as db:
            db.execute("DROP TABLE operation_protocol")
        reopened = HostOperationJournal(self.path, "episode-1")
        deliveries = []
        result = reopened.dispatch("done", "route", {"x": 1},
                                   lambda *args: deliveries.append(args))
        self.assertEqual(result, recorded)
        self.assertEqual(deliveries, [])
        with sqlite3.connect(self.path) as db:
            self.assertEqual(db.execute("SELECT provider_key_format FROM operation_protocol").fetchone(),
                             ("fpb-op-v1-json-pair-sha256",))

    def test_changed_payload_or_route_cannot_reuse_stable_id(self):
        callback = lambda *_: {"ok": True}
        self.journal.dispatch("one", "route-a", {"x": 1}, callback)
        with self.assertRaisesRegex(ExternalEffectError, "conflicts"):
            self.journal.dispatch("one", "route-a", {"x": 2}, callback)
        with self.assertRaisesRegex(ExternalEffectError, "conflicts"):
            self.journal.dispatch("one", "route-b", {"x": 1}, callback)
        with self.assertRaisesRegex(ExternalEffectError, "episode_mismatch"):
            HostOperationJournal(self.path, "episode-2")

    def test_in_flight_dispatch_refuses_restore(self):
        entered = threading.Event()
        release = threading.Event()
        errors = []

        def provider(*_):
            entered.set()
            if not release.wait(3):
                raise TimeoutError("test provider wait expired")
            return {"ok": True}

        def invoke():
            try:
                self.fenced.external_operation("one", "payment", {"x": 1}, provider)
            except BaseException as exc:
                errors.append(exc)

        self.fenced.save_snapshot("before")
        worker = threading.Thread(target=invoke)
        worker.start()
        self.assertTrue(entered.wait(3))
        try:
            with self.assertRaises(ExternalEffectPending):
                self.fenced.load_snapshot("before")
            self.assertEqual(self.vm.load_calls, 0)
        finally:
            release.set()
            worker.join(3)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.fenced.load_snapshot("before")
        self.assertEqual(self.vm.load_calls, 1)

    def test_restore_transaction_fences_concurrent_new_dispatch(self):
        entered = threading.Event()
        release = threading.Event()
        provider_called = threading.Event()
        errors = []

        def slow_restore(tag, *, resume=False):
            entered.set()
            if not release.wait(3):
                raise TimeoutError("test restore wait expired")
            self.vm.load_calls += 1

        self.vm.load_snapshot = slow_restore
        restore = threading.Thread(target=lambda: self.fenced.load_snapshot("before"))

        def provider(*_):
            provider_called.set()
            return {"ok": True}

        def invoke():
            try:
                self.fenced.external_operation("one", "payment", {"x": 1}, provider)
            except BaseException as exc:
                errors.append(exc)

        restore.start()
        self.assertTrue(entered.wait(3))
        dispatch = threading.Thread(target=invoke)
        dispatch.start()
        try:
            self.assertFalse(provider_called.wait(0.15))
            self.assertEqual(self.vm.load_calls, 0)
        finally:
            release.set()
            restore.join(3)
            dispatch.join(3)
        self.assertFalse(restore.is_alive())
        self.assertFalse(dispatch.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(provider_called.is_set())
        self.assertEqual(self.vm.load_calls, 1)

    def test_fork_refused_and_journal_cannot_be_guest_disk(self):
        with self.assertRaisesRegex(ExternalEffectError, "fork_not_supported"):
            self.fenced.fork_snapshot("before", [self.root / "child.qcow2"])
        with self.assertRaisesRegex(ValueError, "outside_vm_disk"):
            FencedMicroVMRuntime(FakeVM(self.path), self.journal)


if __name__ == "__main__":
    unittest.main()
