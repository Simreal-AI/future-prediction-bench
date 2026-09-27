"""Portable offline checks; creates an unrelated two-case repository fixture."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import resource
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                       "examples" / "resident_guest_candidate"))

from future_prediction_bench.coding_env import _workspace_digest
from future_prediction_bench.realworld import RealWorldEnv
from candidate_api import load_fixture, make_env, verify_assets
import case_runner
from guest_supervisor import Supervisor, _bound_batch_case_result
from host_client import GuestSerialTransport, ResidentHostAdapter, ResidentInterrupted
from smoke_qemu import _outer_workspace_sha
from wire import (MAX_BATCH_REQUEST, MAX_REQUEST, CASE_REQUEST_TIMEOUT,
                  BATCH_REQUEST_TIMEOUT, ProtocolError, frame, unframe,
                  validate_request)


ORIGINAL = 'def answer():\n    return "old"\n'
FIXED = 'def answer():\n    return "fixed"\n'


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def make_fixture(root, *, source_path="pkg/logic.py", task_id="synthetic-repair-v1",
                 case_count=2, long_expected=False, long_codes=False):
    root = Path(root)
    task_dir = root / "task"
    seed = task_dir / "seed" / source_path
    seed.parent.mkdir(parents=True)
    seed.write_text(ORIGINAL, encoding="utf-8")
    task = {
        "schema_version": "realworld-0.1", "task_id": task_id,
        "event_id": task_id, "cluster_id": task_id, "split": "train",
        "prompt": "Change answer() to return fixed.",
        "issued_at": "2026-09-25T00:00:00+00:00",
        "action_deadline": "2026-09-25T00:15:00+00:00",
        "outcome_not_before": "2026-09-25T00:00:00+00:00",
        "verify_after": "2026-09-25T00:00:00+00:00",
        "tool_manifest": [{"name": name, "description": name} for name in
                          ("read_file", "replace_text", "submit")],
        "reward_contract": {"id": task_id, "description": "Two private cases.",
                            "min_reward": 0, "max_reward": 1},
        "budgets": {"max_actions": 5, "max_wall_seconds": 60},
        "is_fixture": True, "adapter_id": "synthetic", "adapter_version": "1",
        "metadata": {},
    }
    task_path = task_dir / "task.json"
    _write_json(task_path, task)
    module = source_path.removesuffix(".py").replace("/", ".")
    verifier = {"kind": "command_cases_v1", "cases": [
        {"argv": ["python3", "-B", "-c", f"from {module} import answer; print(answer())"],
         "expected_stdout": "fixed\n", "expected_returncode": 0},
        {"argv": ["python3", "-B", "-c", "print('constant')"],
         "expected_stdout": "constant\n", "expected_returncode": 0},
    ]}
    if long_expected:
        verifier["cases"][0]["expected_stdout"] = "X" * 65
    while len(verifier["cases"]) < case_count:
        index = len(verifier["cases"])
        code = "print('constant')"
        if long_codes:
            code += "\n#" + "x" * (512 - len(code.encode("utf-8")) - 2)
        verifier["cases"].append({"argv": ["python3", "-B", "-c", code],
                                   "expected_stdout": "constant\n",
                                   "expected_returncode": 0})
    verifier["cases"] = verifier["cases"][:case_count]
    verifier_path = task_dir / "verifier" / "verify.json"
    _write_json(verifier_path, verifier)
    contract = {
        "kind": "stateless_python_cases_resident_v1", "task_id": task_id,
        "verifier_sha256": _sha(verifier_path.read_bytes()),
        "source_path": source_path, "seed_file_sha256": _sha(seed.read_bytes()),
        "case_count": case_count,
        "requires_live_background_process_state": False,
        "requires_shared_case_filesystem_state": False,
        "requires_quiescent_submitted_state": True,
        "allow_unprivileged_case_execution": True,
    }
    contract_path = root / "contract.json"
    _write_json(contract_path, contract)
    assets = root / "assets"
    assets.mkdir()
    binaries = {
        "rootfs.qcow2": b"fake-rootfs",
        "vmlinuz-virt": b"fake-kernel",
        "initramfs-virt": b"fake-initramfs",
        "modloop-virt-padded.raw": b"fake-modloop",
    }
    for name, data in binaries.items():
        (assets / name).write_bytes(data)
    manifest = {
        "schema_version": "synthetic-microvm-assets-v1", "task_id": task_id,
        "architecture": "linux/arm64",
        "seed_workspace_sha256": _workspace_digest(task_dir / "seed"),
        "rootfs_qcow2_sha256": _sha(binaries["rootfs.qcow2"]),
        "alpine_sha256": {"vmlinuz-virt": _sha(binaries["vmlinuz-virt"]),
                           "initramfs-virt": _sha(binaries["initramfs-virt"])},
        "modloop_disk_sha256": _sha(binaries["modloop-virt-padded.raw"]),
    }
    manifest_path = assets / "manifest.json"
    _write_json(manifest_path, manifest)
    return task_path, verifier_path, contract_path, manifest_path, assets


class FakeSession:
    def __init__(self, backend):
        self.backend = backend
        self.text = ORIGINAL
        self.case_count = 0
        number = len(backend.sessions) + 1
        self.namespaces = {"pid": f"pid:[{number}]", "mount": f"mnt:[{number}]",
                           "pid_one": True, "process_nonce": f"{number:032x}"}
        self.guest_reset_ns = 1000
        self.namespace_setup_ns = 500

    def call(self, op, args):
        if op == "action":
            name, value = args["name"], args["input"]
            if name == "read_file":
                if self.backend.fail_submit_read and self.backend.read_count == 1:
                    return {"accepted": False, "reason": "invalid_file"}
                self.backend.read_count += 1
                return {"accepted": True, "text": self.text,
                        "sha256": _sha(self.text.encode()), "bytes": len(self.text),
                        "truncated": False}
            if _sha(self.text.encode()) != value["expected_sha256"]:
                return {"accepted": False, "reason": "sha256_mismatch"}
            if self.text.count(value["old"]) != 1:
                return {"accepted": False, "reason": "old_text_not_unique"}
            self.text = self.text.replace(value["old"], value["new"])
            return {"accepted": True, "sha256": _sha(self.text.encode()),
                    "bytes": len(self.text)}
        if op == "submit":
            return {"status": "pending", "reward": None}
        if op == "case_batch":
            results = []
            for code in args["codes"]:
                result = self.call("case", {"code": code})
                result.pop("branch_sha256")
                results.append(_bound_batch_case_result(result))
            return {"branch_sha256": _sha(self.text.encode()), "cases": results}
        if op == "case":
            self.case_count += 1
            if self.backend.fail_case_at == self.case_count:
                raise RuntimeError("simulated_case_failure")
            code = args["code"]
            output = ("fixed\n" if self.text == FIXED else "old\n") if "answer()" in code else "constant\n"
            if self.backend.overlong and self.case_count == 1:
                output = "X" * 65
            return {"return_code": (self.backend.forced_returncode
                                    if self.backend.forced_returncode is not None
                                    and self.case_count == 1 else 0),
                    "stdout_b64": base64.b64encode(output.encode()).decode(),
                    "truncated": False, "output_over_batch_cap": False,
                    "guest_case_ns": 1000,
                    "branch_sha256": _sha(self.text.encode())}
        raise AssertionError(op)

    def close(self, *, completed=False):
        if self.backend.fail_close:
            raise RuntimeError("simulated_cleanup_failure")
        if completed and self.case_count != self.backend.expected_case_count:
            raise RuntimeError("simulated_incomplete_cases")
        self.backend.closed.append(self.text)
        return 700

    def abort(self):
        self.backend.aborted += 1


class FakeBackend:
    def __init__(self, *, overlong=False, fail_case_at=None, fail_close=False,
                 fail_submit_read=False, expected_case_count=2,
                 forced_returncode=None):
        self.sessions = []
        self.closed = []
        self.aborted = 0
        self.overlong = overlong
        self.fail_case_at = fail_case_at
        self.fail_close = fail_close
        self.fail_submit_read = fail_submit_read
        self.forced_returncode = forced_returncode
        self.read_count = 0
        self.expected_case_count = expected_case_count

    def start(self, episode_id, mode):
        session = FakeSession(self)
        self.sessions.append(session)
        return session


class FakeRuntime:
    def __init__(self, supervisor):
        self.supervisor = supervisor
        self.requests = []
        self.timeouts = []

    def run_shell(self, command, *, timeout):
        args = shlex.split(command)
        assert args[-2] == "--call"
        request = unframe(base64.b64decode(args[-1], validate=True),
                          limit=MAX_BATCH_REQUEST)
        self.requests.append(request)
        self.timeouts.append(timeout)
        answer = self.supervisor.handle(request)
        return {"return_code": 0,
                    "stdout": "FPB_RESIDENT_V0=" + base64.b64encode(
                    frame(answer, limit=16384)).decode()}


class LocalDigestRuntime:
    def run_shell(self, command, *, timeout):
        parts = shlex.split(command)
        assert parts[-2] == "-c"
        result = subprocess.run([sys.executable, "-I", "-B", "-c", parts[-1]],
                                capture_output=True, text=True, timeout=timeout)
        return {"return_code": result.returncode, "stdout": result.stdout}


class PortableCandidateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.paths = make_fixture(self.temp.name)
        self.private = load_fixture(task_path=self.paths[0], verifier_path=self.paths[1],
                                    contract_path=self.paths[2],
                                    assets_manifest_path=self.paths[3])

    def connected(self, *, overlong=False, fail_case_at=None, fail_close=False,
                  fail_submit_read=False, private=None, forced_returncode=None):
        private = private or self.private
        backend = FakeBackend(overlong=overlong, fail_case_at=fail_case_at,
                              fail_close=fail_close,
                              fail_submit_read=fail_submit_read,
                              expected_case_count=len(private.cases),
                              forced_returncode=forced_returncode)
        supervisor = Supervisor(backend, source_path=private.source_path,
                                seed_file_sha256=private.seed_file_sha256,
                                case_count=len(private.cases))
        runtime = FakeRuntime(supervisor)
        client = ResidentHostAdapter(GuestSerialTransport(runtime))
        client.connect()
        return client, backend, runtime

    @staticmethod
    def submit(env, *, repair):
        env.reset("synthetic-policy")
        path = env.adapter.private_cases.source_path
        read = env.step({"action": "read_file", "path": path})
        if repair:
            edit = env.step({"action": "replace_text", "path": path,
                             "expected_file_sha256": read["observation"]["sha256"],
                             "old_text": '"old"', "new_text": '"fixed"'})
            assert edit["observation"].get("sha256") == _sha(FIXED.encode())
        submitted = env.step({"action": "submit"})
        assert submitted["reward"] is None

    def test_two_case_sequential_repair_and_baseline(self):
        client, backend, runtime = self.connected()
        for mode, reward, passes in (("repair", 1.0, 2), ("baseline", 0.0, 1)):
            env = make_env(client, self.private, episode_mode=mode)
            self.submit(env, repair=mode == "repair")
            result = env.verify()
            self.assertEqual((result["status"], result["reward"]), ("graded", reward))
            self.assertEqual(result["evidence"]["passed_count"], passes)
        self.assertEqual(backend.closed, [FIXED, ORIGINAL])
        self.assertEqual(len([r for r in runtime.requests if r["op"] == "case"]), 4)
        self.assertTrue(all("expected_stdout" not in str(r) for r in runtime.requests))
        self.assertTrue(all(timeout >= CASE_REQUEST_TIMEOUT
                            for request, timeout in zip(runtime.requests, runtime.timeouts)
                            if request["op"] == "case"))

    def test_batch_two_cases_and_overlong_candidate_is_definite_mismatch(self):
        for overlong in (False, True):
            client, _, runtime = self.connected(overlong=overlong)
            env = make_env(client, self.private, case_transport="batch")
            self.submit(env, repair=True)
            result = env.verify()
            self.assertEqual(result["status"], "graded")
            self.assertEqual(result["reward"], 0.0 if overlong else 1.0)
            self.assertEqual(result["evidence"]["passed_count"], 1 if overlong else 2)
            self.assertEqual(len([r for r in runtime.requests if r["op"] == "case_batch"]), 1)
            self.assertEqual(len([r for r in runtime.requests if r["op"] == "case"]), 0)
            self.assertTrue(all(timeout >= BATCH_REQUEST_TIMEOUT
                                for request, timeout in zip(runtime.requests, runtime.timeouts)
                                if request["op"] == "case_batch"))

    def test_stdout_close_then_sleep_is_scored_as_candidate_timeout(self):
        candidate = subprocess.Popen(
            [sys.executable, "-c", "import os,time; os.close(1); time.sleep(1)"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            start_new_session=True)
        with mock.patch.object(case_runner, "CASE_SECONDS", 0.05):
            result = case_runner._collect(candidate)
        self.assertEqual(result["return_code"], 124)
        self.assertFalse(result["truncated"])
        self.assertEqual(candidate.poll(), -9)

    def test_case_resource_caps_are_hard_and_respect_stricter_inherited_limits(self):
        with (mock.patch.object(case_runner.resource, "getrlimit",
                                return_value=(1024, 2048)),
              mock.patch.object(case_runner.resource, "setrlimit") as set_limit):
            case_runner._set_case_limits(((resource.RLIMIT_FSIZE, 8192),))
        set_limit.assert_called_once_with(resource.RLIMIT_FSIZE, (1024, 1024))
        self.assertEqual({kind for kind, _ in case_runner.CASE_RESOURCE_LIMITS},
                         {resource.RLIMIT_CPU, resource.RLIMIT_AS,
                          resource.RLIMIT_NPROC, resource.RLIMIT_FSIZE,
                          resource.RLIMIT_NOFILE, resource.RLIMIT_CORE})

    def test_file_size_limit_terminates_candidate_as_scored_result(self):
        target = Path(self.temp.name) / "limited-output.bin"
        candidate = subprocess.Popen(
            [sys.executable, "-c",
             "import pathlib,sys; pathlib.Path(sys.argv[1]).write_bytes(b'x'*4096)",
             str(target)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL, start_new_session=True,
            preexec_fn=lambda: case_runner._set_case_limits(
                ((resource.RLIMIT_FSIZE, 1024),)))
        result = case_runner._collect(candidate)
        self.assertNotEqual(result["return_code"], 0)
        self.assertNotEqual(result["return_code"], 124)
        self.assertLessEqual(target.stat().st_size, 1024)

    def test_resource_exits_are_scored_failures_in_both_transports(self):
        for transport in ("sequential", "batch"):
            for returncode in (-25, 124, 125):
                client, _, _ = self.connected(forced_returncode=returncode)
                env = make_env(client, self.private, case_transport=transport)
                self.submit(env, repair=True)
                grade = env.verify()
                self.assertEqual((grade["status"], grade["reward"],
                                  grade["evidence"]["passed_count"]),
                                 ("graded", 0.0, 1))

    def test_overlong_candidate_reward_matches_sequential(self):
        for transport in ("sequential", "batch"):
            client, _, _ = self.connected(overlong=True)
            env = make_env(client, self.private, case_transport=transport)
            self.submit(env, repair=True)
            grade = env.verify()
            self.assertEqual((grade["status"], grade["reward"],
                              grade["evidence"]["passed_count"]),
                             ("graded", 0.0, 1))

    def test_long_expected_output_falls_back_to_sequential(self):
        paths = make_fixture(Path(self.temp.name) / "long-expected",
                             task_id="long-expected-v1", long_expected=True)
        private = load_fixture(task_path=paths[0], verifier_path=paths[1],
                               contract_path=paths[2], assets_manifest_path=paths[3])
        client, _, runtime = self.connected(private=private)
        env = make_env(client, private, case_transport="batch")
        self.submit(env, repair=True)
        grade = env.verify()
        self.assertEqual((grade["status"], grade["reward"]), ("graded", 0.0))
        self.assertEqual(env.adapter.metrics["effective_case_transport"], "sequential")
        self.assertEqual(env.adapter.metrics["batch_fallback_reason"],
                         "expected_stdout_over_batch_cap")
        self.assertFalse(any(r["op"] == "case_batch" for r in runtime.requests))

    def test_oversized_batch_request_falls_back_before_guest_call(self):
        paths = make_fixture(Path(self.temp.name) / "long-codes",
                             task_id="long-codes-v1", case_count=14, long_codes=True)
        private = load_fixture(task_path=paths[0], verifier_path=paths[1],
                               contract_path=paths[2], assets_manifest_path=paths[3])
        client, _, runtime = self.connected(private=private)
        env = make_env(client, private, case_transport="batch")
        self.submit(env, repair=True)
        grade = env.verify()
        self.assertEqual((grade["status"], grade["reward"],
                          grade["evidence"]["passed_count"]), ("graded", 1.0, 14))
        self.assertEqual(env.adapter.metrics["batch_fallback_reason"],
                         "batch_request_exceeds_bound")
        self.assertEqual(sum(r["op"] == "case" for r in runtime.requests), 14)
        self.assertFalse(any(r["op"] == "case_batch" for r in runtime.requests))

    def test_another_source_path_is_accepted(self):
        paths = make_fixture(Path(self.temp.name) / "second", source_path="lib/other.py",
                             task_id="second-repair-v1")
        private = load_fixture(task_path=paths[0], verifier_path=paths[1],
                               contract_path=paths[2], assets_manifest_path=paths[3])
        self.assertEqual(private.source_path, "lib/other.py")
        self.assertEqual(private.task_id, "second-repair-v1")

    def test_contract_seed_case_count_and_path_are_enforced(self):
        for key, bad in (("seed_file_sha256", "0" * 64), ("case_count", 3),
                         ("source_path", "../escape")):
            contract = json.loads(self.paths[2].read_text())
            contract[key] = bad
            _write_json(self.paths[2], contract)
            with self.assertRaises(ValueError):
                load_fixture(task_path=self.paths[0], verifier_path=self.paths[1],
                             contract_path=self.paths[2], assets_manifest_path=self.paths[3])
            contract[key] = self.private.binding["seed_file_sha256"] if key == "seed_file_sha256" else (2 if key == "case_count" else "pkg/logic.py")
            _write_json(self.paths[2], contract)

    def test_edited_verifier_withholds_reward(self):
        client, _, runtime = self.connected()
        env = make_env(client, self.private)
        self.submit(env, repair=True)
        self.paths[1].write_bytes(self.paths[1].read_bytes() + b"\n")
        result = env.verify()
        self.assertEqual(result["status"], "pending")
        self.assertIsNone(result["reward"])
        self.assertFalse(any(r["op"] == "case" for r in runtime.requests))

    def test_wrong_running_guest_binding_is_rejected(self):
        client, _, _ = self.connected()
        client.source_path = "other.py"
        with self.assertRaisesRegex(ValueError, "guest_helper_binding_mismatch"):
            make_env(client, self.private)

    def test_wrong_guest_case_count_is_rejected(self):
        client, _, _ = self.connected()
        client.case_count += 1
        with self.assertRaisesRegex(ValueError, "guest_helper_binding_mismatch"):
            make_env(client, self.private)

    def test_guest_requires_exact_case_count_before_completed_close(self):
        client, _, _ = self.connected()
        env = make_env(client, self.private)
        self.submit(env, repair=True)
        client.run_case(self.private.cases[0]["argv"][3],
                        expected_branch_sha256=_sha(FIXED.encode()))
        with self.assertRaises(ResidentInterrupted):
            client.close(completed=True)
        self.assertEqual(client.state, "interrupted")

    def test_guest_rejects_wrong_batch_count(self):
        client, _, _ = self.connected()
        env = make_env(client, self.private)
        self.submit(env, repair=True)
        with self.assertRaises(ResidentInterrupted):
            client.run_case_batch([self.private.cases[0]["argv"][3]],
                                  expected_branch_sha256=_sha(FIXED.encode()))
        self.assertEqual(client.state, "interrupted")

    def test_guest_rejects_case_beyond_frozen_count(self):
        client, _, _ = self.connected()
        env = make_env(client, self.private)
        self.submit(env, repair=True)
        for case in self.private.cases:
            client.run_case(case["argv"][3],
                            expected_branch_sha256=_sha(FIXED.encode()))
        with self.assertRaises(ResidentInterrupted):
            client.run_case(self.private.cases[0]["argv"][3],
                            expected_branch_sha256=_sha(FIXED.encode()))
        self.assertEqual(client.state, "interrupted")

    def test_unreadable_submit_source_interrupts_without_reward(self):
        client, _, runtime = self.connected(fail_submit_read=True)
        env = make_env(client, self.private)
        env.reset("synthetic-policy")
        env.step({"action": "read_file", "path": self.private.source_path})
        result = env.step({"action": "submit"})
        self.assertEqual(result["info"]["status"], "interrupted")
        self.assertIsNone(result["reward"])
        self.assertFalse(any(r["op"] == "submit" for r in runtime.requests))

    def test_mutated_host_asset_at_submit_interrupts_without_reward(self):
        client, _, runtime = self.connected()
        env = make_env(client, self.private)
        env.reset("synthetic-policy")
        env.step({"action": "read_file", "path": self.private.source_path})
        self.paths[0].write_bytes(self.paths[0].read_bytes() + b"\n")
        result = env.step({"action": "submit"})
        self.assertEqual(result["info"]["status"], "interrupted")
        self.assertIsNone(result["reward"])
        self.assertFalse(any(r["op"] == "submit" for r in runtime.requests))

    def test_case_or_cleanup_fault_keeps_reward_pending(self):
        for options in ({"fail_case_at": 1}, {"fail_close": True}):
            client, _, _ = self.connected(**options)
            env = make_env(client, self.private)
            self.submit(env, repair=True)
            result = env.verify()
            self.assertEqual(result["status"], "pending")
            self.assertIsNone(result["reward"])

    def test_helper_digest_mismatch_blocks_episode(self):
        client, _, _ = self.connected()
        client.guest_helper_sha256["wire.py"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "guest_helper_binding_mismatch"):
            make_env(client, self.private)

    def test_asset_preflight_and_digest_mutation(self):
        self.assertEqual(verify_assets(self.private, self.paths[4])["case_count"], 2)
        (self.paths[4] / "rootfs.qcow2").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "asset_digest_mismatch"):
            verify_assets(self.private, self.paths[4])

    def test_cli_preflight_from_operator_paths(self):
        project_root = Path(__file__).resolve().parents[1]
        command = [sys.executable,
                   str(project_root / "examples" /
                       "resident_guest_candidate" / "candidate_api.py"),
                   "--task-path", str(self.paths[0]), "--verifier-path", str(self.paths[1]),
                   "--contract-path", str(self.paths[2]), "--assets-dir", str(self.paths[4]),
                   "--check-assets"]
        env = dict(os.environ, PYTHONPATH=str(project_root) + os.pathsep
                   + os.environ.get("PYTHONPATH", ""))
        result = subprocess.run(command, capture_output=True, text=True,
                                check=True, env=env)
        self.assertEqual(json.loads(result.stdout)["status"], "assets_valid")

    def test_guest_workspace_digest_mirrors_pinned_host_digest(self):
        seed = self.paths[0].parent / "seed"
        self.assertEqual(_outer_workspace_sha(LocalDigestRuntime(), root=seed),
                         _workspace_digest(seed))
        (seed / "pkg" / "extra.txt").write_text("changed", encoding="utf-8")
        self.assertEqual(_outer_workspace_sha(LocalDigestRuntime(), root=seed),
                         _workspace_digest(seed))

    def test_batch_bounds_remain_hard(self):
        request = {"v": 1, "seq": 0, "op": "case_batch", "args": {"codes": ["x"] * 15}}
        with self.assertRaises(ProtocolError):
            validate_request(request)
        request["args"]["codes"] = ["x", "y"]
        validate_request(request)


if __name__ == "__main__":
    unittest.main()
