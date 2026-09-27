"""Offline contract and fault tests for the proof-only full-VM case pool."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from examples.realworld_boltons26.fullvm_verifier_pool import (
    TwoChildFullVMVerifierPool, _probe_independent_state,
)
from examples.realworld_boltons26.benchmark_fullvm_verifier_pool import (
    _assert_parity, _check_host_pressure, _schedule,
)
from future_prediction_bench.microvm_coding import MicroVMCodingAdapter


VISIBLE = ("python3", "-B", "-c", "print('visible')")


class FakeVM:
    kernel_sha256 = "a" * 64
    initramfs_sha256 = "b" * 64
    readonly_disk_paths = ()
    enable_action_port = False

    def __init__(self, disk_path, *, fail_case=False, fail_fork=False):
        self.disk_path = Path(disk_path)
        self.fail_case = fail_case
        self.fail_fork = fail_fork
        self.restores = 0
        self.closed = False
        self.children = []
        self.state = {}
        self.metrics = {"fork_disk_clone_seconds": 0.0,
                        "fork_child_start_restore_seconds": 0.0}

    def load_snapshot(self, tag):
        assert tag == "submitted"
        if self.fail_case:
            raise RuntimeError("injected_child_restore_failure")
        self.restores += 1

    def fork_snapshot(self, tag, paths, *, resume_parent, parallel_children):
        assert (tag, resume_parent, parallel_children) == ("submitted", True, True)
        if self.fail_fork:
            raise RuntimeError("injected_fork_failure")
        self.children = [FakeVM(path, fail_case=(self.fail_case and index == 1))
                         for index, path in enumerate(paths)]
        for child in self.children:
            child.disk_path.write_bytes(b"fake-qcow2")
        self.metrics["fork_disk_clone_seconds"] += 0.01
        self.metrics["fork_child_start_restore_seconds"] += 0.02
        return self.children

    def get_state(self):
        return {"metrics": dict(self.metrics)}

    def run_shell(self, command, *, timeout):
        if command.startswith("printf '") and "' > " in command:
            value, path = command.removeprefix("printf '").split("' > ", 1)
            self.state[path] = value
            return {"return_code": 0, "stdout": ""}
        if command.startswith("cat "):
            path = command.removeprefix("cat ")
            return {"return_code": 0 if path in self.state else 1,
                    "stdout": self.state.get(path, "")}
        if command.startswith("test ! -e "):
            path = command.removeprefix("test ! -e ")
            return {"return_code": 0 if path not in self.state else 1,
                    "stdout": ""}
        raise AssertionError("Unexpected fake shell command")

    def close(self):
        self.closed = True


class FakeRunner:
    command_timeout = 30.0
    _validate_python_argv = staticmethod(MicroVMCodingAdapter._validate_python_argv)

    def __init__(self, owner, vm):
        self.owner = owner
        self.vm = vm

    def _run_python_case(self, argv, *, timeout, unprivileged):
        self.owner.calls.append((self.vm, argv[3], timeout, unprivileged))
        if self.owner.mutate_verifier and argv[3] == "print(0)":
            (self.owner.verifier_dir / "verify.json").write_text("{}", encoding="utf-8")
        data = (argv[3].removeprefix("print(").removesuffix(")") + "\n").encode()
        return {"return_code": 0, "stdout_bytes": data,
                "stdout": data.decode(), "truncated": False}


class PoolHarness(TwoChildFullVMVerifierPool):
    def _make_case_runner(self, child):
        return FakeRunner(self, child)


class PoolContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.verifier = self.root / "verifier"
        self.pool = self.root / "pool"
        self.verifier.mkdir()
        self.pool.mkdir()
        self.disk = self.root / "parent.qcow2"
        self.disk.write_bytes(b"parent-qcow2")
        self.cases = [{"argv": ["python3", "-B", "-c", f"print({index})"],
                       "expected_stdout": f"{index}\n", "expected_returncode": 0}
                      for index in range(14)]
        self._write_cases()

    def _write_cases(self):
        (self.verifier / "verify.json").write_text(
            json.dumps({"kind": "command_cases_v1", "cases": self.cases}),
            encoding="utf-8")

    def _adapter(self, cls=PoolHarness, **runtime_flags):
        runtime = FakeVM(self.disk, **runtime_flags)
        if cls is PoolHarness:
            adapter = cls(runtime, verifier_dir=self.verifier, visible_check=VISIBLE,
                          pool_dir=self.pool)
        else:
            adapter = cls(runtime, verifier_dir=self.verifier, visible_check=VISIBLE)
        adapter.submitted = True
        adapter.snapshot = {"kind": "full_vm_state_qcow2_v1", "tag": "submitted"}
        adapter.expected_binding = adapter.artifact_binding()
        adapter.calls = []
        adapter.mutate_verifier = False
        if cls is MicroVMCodingAdapter:
            adapter._run_python_case = lambda argv, **kwargs: FakeRunner(
                adapter, runtime)._run_python_case(argv, **kwargs)
        return adapter

    def test_14_case_evidence_reward_binding_and_order_match_serial(self):
        serial = self._adapter(MicroVMCodingAdapter)
        pool = self._adapter()
        self.assertEqual(serial.artifact_binding(), pool.artifact_binding())
        serial_out = serial.verify(now=datetime.now(timezone.utc))
        pool_out = pool.verify(now=datetime.now(timezone.utc))
        for key in ("status", "reward", "evidence"):
            self.assertEqual(serial_out[key], pool_out[key])
        self.assertEqual(pool_out["reward"], 1.0)
        self.assertEqual(serial.runtime.restores, 14)
        self.assertEqual([child.restores for child in pool.runtime.children], [7, 7])
        for index, child in enumerate(pool.runtime.children):
            self.assertEqual({code for vm, code, _, _ in pool.calls if vm is child},
                             {f"print({case})" for case in range(index, 14, 2)})
        self.assertEqual(pool.metrics["full_vm_restores"], 14)
        self.assertEqual(pool.metrics["hidden_cases"], 14)
        self.assertTrue(all(call[2:] == (30.0, True) for call in pool.calls))
        self.assertTrue(all(child.closed for child in pool.runtime.children))
        self.assertFalse(list(self.pool.iterdir()))
        self.assertEqual(pool.verify(now=datetime.now(timezone.utc)), pool_out)
        self.assertEqual(len(pool.calls), 14)

    def test_failure_reward_zero_matches_serial(self):
        self.cases[5]["expected_stdout"] = "wrong\n"
        self._write_cases()
        serial = self._adapter(MicroVMCodingAdapter)
        pool = self._adapter()
        expected = serial.verify(now=datetime.now(timezone.utc))
        result = pool.verify(now=datetime.now(timezone.utc))
        self.assertEqual(result["reward"], 0.0)
        self.assertEqual(result["evidence"], expected["evidence"])

    def test_child_failure_is_pending_and_cleans_all_disks(self):
        pool = self._adapter(fail_case=True)
        result = pool.verify(now=datetime.now(timezone.utc))
        self.assertEqual(result, {"status": "pending",
                                  "reason": "vm_verifier_infrastructure_error"})
        self.assertIsNone(pool.verified)
        self.assertEqual(pool.metrics["hidden_cases"], 0)
        self.assertTrue(all(child.closed for child in pool.runtime.children))
        self.assertFalse(list(self.pool.iterdir()))
        self.assertEqual(pool.active_children, ())

    def test_fork_failure_is_pending(self):
        pool = self._adapter(fail_fork=True)
        self.assertEqual(pool.verify(now=datetime.now(timezone.utc))["reason"],
                         "vm_verifier_infrastructure_error")
        self.assertFalse(list(self.pool.iterdir()))

    def test_verifier_change_is_pending_after_all_cases(self):
        pool = self._adapter()
        pool.mutate_verifier = True
        result = pool.verify(now=datetime.now(timezone.utc))
        self.assertEqual(result, {"status": "pending",
                                  "reason": "verifier_changed_during_execution"})
        self.assertEqual(len(pool.calls), 14)
        self.assertFalse(list(self.pool.iterdir()))

    def test_invalid_verifier_and_virtio_rejected(self):
        pool = self._adapter()
        self.cases[2]["argv"] = ["sh", "-c", "bad"]
        self._write_cases()
        self.assertEqual(pool.verify(now=datetime.now(timezone.utc))["reason"],
                         "verifier_differs_from_frozen_task")
        vm = FakeVM(self.disk)
        vm.enable_action_port = True
        with self.assertRaises(ValueError):
            PoolHarness(vm, verifier_dir=self.verifier, visible_check=VISIBLE,
                        pool_dir=self.pool)

    def test_index_corruption_cannot_score(self):
        class BadIndexPool(PoolHarness):
            def _worker(self, child, indexed_cases):
                rows, seconds = super()._worker(child, indexed_cases)
                if rows:
                    rows[0] = (0, rows[0][1])
                return rows, seconds

        runtime = FakeVM(self.disk)
        pool = BadIndexPool(runtime, verifier_dir=self.verifier,
                            visible_check=VISIBLE, pool_dir=self.pool)
        pool.submitted = True
        pool.snapshot = {"kind": "full_vm_state_qcow2_v1", "tag": "submitted"}
        pool.expected_binding = pool.artifact_binding()
        pool.calls = []
        pool.mutate_verifier = False
        self.assertEqual(pool.verify(now=datetime.now(timezone.utc))["reason"],
                         "vm_verifier_infrastructure_error")

    def test_five_pair_schedule_alternates_and_rejects_short_run(self):
        schedule = list(_schedule(5))
        self.assertEqual(len(schedule), 20)
        self.assertEqual(schedule[:4], [
            (1, "repair", "serial"), (1, "repair", "pool"),
            (1, "baseline", "serial"), (1, "baseline", "pool")])
        self.assertEqual(schedule[4:8], [
            (2, "baseline", "pool"), (2, "baseline", "serial"),
            (2, "repair", "pool"), (2, "repair", "serial")])
        with self.assertRaises(ValueError):
            list(_schedule(4))

    def test_probe_checks_independent_ram_and_writable_disk(self):
        parent = FakeVM(self.disk)
        children = [FakeVM(self.root / f"child-{index}.qcow2") for index in range(2)]
        _probe_independent_state(parent, children)
        self.assertEqual(len(children[0].state), 2)
        self.assertEqual(len(children[1].state), 2)
        self.assertFalse(parent.state)
        self.assertTrue(any(path.startswith("/tmp/") for path in children[0].state))
        self.assertTrue(any(path.startswith("/mnt/root/") for path in children[0].state))
        shared = {}
        children[0].state = children[1].state = shared
        with self.assertRaisesRegex(RuntimeError, "isolation_probe_failed"):
            _probe_independent_state(parent, children)

    def test_ab_parity_catches_changed_evidence(self):
        rows = []
        for pair, branch, arm in _schedule(5):
            rows.append({"pair": pair, "branch": branch, "arm": arm,
                         "task_sha256": "task", "artifact_binding_sha256": "binding",
                         "candidate_python_identity": "guest_uid_gid_65534_v2",
                         "opening_observation_sha256": "opening",
                         "action_observation_sha256s": [branch],
                         "reward": 1.0 if branch == "repair" else 0.0,
                         "evidence": {"kind": "host_checked_qemu_full_vm_cases_v1",
                                      "case_results": [{"passed": branch == "repair"}] * 14},
                         "case_results": [{"passed": branch == "repair"}] * 14,
                         "final_source_sha256": branch,
                         "adapter_metrics": {"full_vm_restores": 14,
                                             "hidden_cases": 14}})
        _assert_parity(rows, 5)
        rows[11]["case_results"] = [{"passed": True}] * 14
        with self.assertRaisesRegex(RuntimeError, "same_contract_parity_failed"):
            _assert_parity(rows, 5)

    def test_host_pressure_guard_requires_samples_and_free_memory(self):
        row = {"peak_qemu_rss_kib_sampled": None, "qemu_rss_samples": 0}
        with self.assertRaisesRegex(RuntimeError, "sampling_unavailable"):
            _check_host_pressure(row)
        row.update(peak_qemu_rss_kib_sampled=100_000, qemu_rss_samples=3)
        with mock.patch(
                "examples.realworld_boltons26.benchmark_fullvm_verifier_pool."
                "_host_memory_free_percent", return_value=9):
            with self.assertRaisesRegex(RuntimeError, "memory_pressure_guard"):
                _check_host_pressure(row)
        with mock.patch(
                "examples.realworld_boltons26.benchmark_fullvm_verifier_pool."
                "_host_memory_free_percent", return_value=26):
            _check_host_pressure(row)
        self.assertEqual(row["host_memory_free_percent_after"], 26)


if __name__ == "__main__":
    unittest.main()
