"""Integration contracts for the isolated coding adapter without a Docker daemon."""

import hashlib
import errno
import json
import stat
import subprocess
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from future_prediction_bench.coding_env import DockerCodingAdapter, _workspace_digest
from future_prediction_bench.microvm_coding import replace_text_helper_binding
from future_prediction_bench.realworld import RealWorldEnv
from future_prediction_bench.replace_text import apply as apply_replace_text


class FakeDockerAdapter(DockerCodingAdapter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.commands = []
        self.fail_actor = False

    def _docker(self, arguments, *, timeout=None):
        self.commands.append(list(arguments))
        if arguments[:2] == ["image", "inspect"]:
            return subprocess.CompletedProcess(arguments, 0, b"sha256:" + b"a" * 64 + b"\n", b"")
        if arguments[0] == "inspect":
            return subprocess.CompletedProcess(arguments, 0, b"true\n", b"")
        if arguments[0] == "exec":
            if self.fail_actor:
                from future_prediction_bench.coding_env import CodingRuntimeError
                raise CodingRuntimeError("docker_cli_timed_out")
            return subprocess.CompletedProcess(arguments, 0, b"", b"")
        if arguments[0] == "run" and "--detach" not in arguments:
            source = " ".join(arguments)
            stdout = b"5\n" if "add(2, 3)" in source else b"0\n"
            return subprocess.CompletedProcess(arguments, 0, stdout, b"")
        return subprocess.CompletedProcess(arguments, 0, b"container\n", b"")


class CodingAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.seed = self.root / "seed"
        self.verifier = self.root / "verifier"
        self.seed.mkdir()
        self.verifier.mkdir()
        (self.seed / "math_utils.py").write_text("def add(a, b): return a-b\n", encoding="utf-8")
        (self.verifier / "verify.json").write_text(json.dumps({
            "kind": "command_cases_v1", "cases": [
                {"argv": ["python3", "-B", "-c", "from math_utils import add; print(add(2, 3))"],
                 "expected_stdout": "5\n", "expected_returncode": 0},
                {"argv": ["python3", "-B", "-c", "from math_utils import add; print(add(-2, 2))"],
                 "expected_stdout": "0\n", "expected_returncode": 0},
            ]}), encoding="utf-8")
        self.now = datetime.now(timezone.utc)
        self.task = {
            "schema_version": "realworld-0.1", "task_id": "fixture-add", "event_id": "fixture-add",
            "cluster_id": "fixture-add", "split": "train", "prompt": "Fix add.",
            "issued_at": (self.now - timedelta(minutes=1)).isoformat(),
            "action_deadline": (self.now + timedelta(minutes=20)).isoformat(),
            "outcome_not_before": (self.now - timedelta(minutes=1)).isoformat(),
            "verify_after": (self.now - timedelta(minutes=1)).isoformat(),
            "tool_manifest": [{"name": name, "description": name} for name in
                              ("list_files", "read_file", "write_file", "run_visible_checks", "submit")],
            "reward_contract": {"id": "tests", "description": "Hidden cases.",
                                "min_reward": 0, "max_reward": 1},
            "budgets": {"max_actions": 8, "max_wall_seconds": 900}, "is_fixture": True,
        }

    def adapter(self):
        return FakeDockerAdapter(seed_dir=self.seed, verifier_dir=self.verifier,
                                 image="fixture:local", output_root=self.root / "output")

    def test_separate_host_checked_verifier_and_changed_only_checkpoints(self):
        adapter = self.adapter()
        env = RealWorldEnv(self.task, adapter)
        try:
            env.reset("scripted-policy")
            self.assertEqual(adapter.get_state()["metrics"]["checkpoints_created"], 1)
            env.step({"action": "list_files"})
            env.step({"action": "read_file", "path": "math_utils.py"})
            self.assertEqual(adapter.get_state()["metrics"]["checkpoints_created"], 1)
            env.step({"action": "write_file", "path": "math_utils.py",
                      "content": "def add(a, b): return a+b\n"})
            self.assertEqual(adapter.get_state()["metrics"]["checkpoints_created"], 2)
            env.step({"action": "submit"})
            self.assertIsNone(env.get_state()["reward"])
            self.assertEqual(env.verify()["reward"], 1)
            grading = [cmd for cmd in adapter.commands if cmd[0] == "run" and "--detach" not in cmd]
            self.assertEqual(len(grading), 2)
            self.assertTrue(all("/verifier" not in " ".join(cmd) for cmd in grading))
            self.assertEqual(env.export_trajectory()["reward"], 1)
        finally:
            adapter.close()

    def test_replace_text_atomic_success_and_policy_visible_conflicts(self):
        adapter = self.adapter()
        task = dict(self.task)
        task["tool_manifest"] = [*self.task["tool_manifest"],
                                 {"name": "replace_text", "description": "Exact edit."}]
        task["metadata"] = {"replace_text_helper_binding": replace_text_helper_binding()}
        env = RealWorldEnv(task, adapter)
        try:
            env.reset("scripted-policy")
            source = adapter.workspace / "math_utils.py"
            source.chmod(0o640)
            before_stat = source.stat()
            original = source.read_bytes()
            action = {"action": "replace_text", "path": "math_utils.py",
                      "expected_file_sha256": hashlib.sha256(original).hexdigest(),
                      "old_text": "a-b", "new_text": "a+b"}
            result = env.step(action)["observation"]
            self.assertEqual(result["sha256"], hashlib.sha256(
                b"def add(a, b): return a+b\n").hexdigest())
            self.assertEqual(result["workspace_sha256"], _workspace_digest(adapter.workspace))
            self.assertEqual(source.read_bytes(), b"def add(a, b): return a+b\n")
            self.assertEqual(stat.S_IMODE(source.stat().st_mode), 0o640)
            self.assertEqual((source.stat().st_uid, source.stat().st_gid),
                             (before_stat.st_uid, before_stat.st_gid))
            self.assertEqual(adapter.get_state()["metrics"]["checkpoints_created"], 2)
            changed = source.read_bytes()
            stale = env.step(action)["observation"]
            self.assertEqual((stale["status"], stale["reason"]),
                             ("conflict", "sha256_mismatch"))
            no_match = dict(action, expected_file_sha256=hashlib.sha256(changed).hexdigest(),
                            old_text="no match")
            self.assertEqual(env.step(no_match)["observation"]["reason"],
                             "old_text_not_unique")
            self.assertEqual(source.read_bytes(), changed)
            self.assertEqual(adapter.get_state()["metrics"]["checkpoints_created"], 2)
            self.assertEqual(list(adapter.workspace.glob(".fpb-replace-*")), [])
        finally:
            adapter.close()

    def test_replace_text_rejects_symlinks_hardlinks_and_duplicate_match(self):
        with tempfile.TemporaryDirectory() as root:
            workspace = Path(root) / "workspace"
            outside = Path(root) / "outside.txt"
            workspace.mkdir()
            outside.write_text("private", encoding="utf-8")
            (workspace / "dupe.txt").write_text("a a", encoding="utf-8")
            duplicate = {"action": "replace_text", "path": "dupe.txt",
                         "expected_file_sha256": hashlib.sha256(b"a a").hexdigest(),
                         "old_text": "a", "new_text": "b"}
            self.assertEqual(apply_replace_text(workspace, duplicate)["reason"],
                             "old_text_not_unique")
            self.assertEqual((workspace / "dupe.txt").read_text(), "a a")
            (workspace / "link.txt").symlink_to(outside)
            link = dict(duplicate, path="link.txt",
                        expected_file_sha256=hashlib.sha256(b"private").hexdigest(),
                        old_text="private", new_text="changed")
            with self.assertRaises(ValueError):
                apply_replace_text(workspace, link)
            self.assertEqual(outside.read_text(), "private")
            (workspace / "parent_link").symlink_to(Path(root), target_is_directory=True)
            parent_link = dict(link, path="parent_link/outside.txt")
            with self.assertRaises(ValueError):
                apply_replace_text(workspace, parent_link)
            self.assertEqual(outside.read_text(), "private")
            (workspace / "alias.txt").hardlink_to(workspace / "dupe.txt")
            with self.assertRaises(ValueError):
                apply_replace_text(workspace, duplicate)
            self.assertEqual((workspace / "alias.txt").read_text(), "a a")
            self.assertEqual(list(workspace.glob(".fpb-replace-*")), [])

    def test_replace_text_failed_atomic_rename_leaves_original_and_no_temp(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "source.txt"
            source.write_bytes(b"old\n")
            action = {"action": "replace_text", "path": "source.txt",
                      "expected_file_sha256": hashlib.sha256(b"old\n").hexdigest(),
                      "old_text": "old", "new_text": "new"}
            with mock.patch("future_prediction_bench.replace_text.os.replace",
                            side_effect=OSError("rename failed")):
                with self.assertRaises(OSError):
                    apply_replace_text(root, action)
            self.assertEqual(source.read_bytes(), b"old\n")
            self.assertEqual(list(Path(root).glob(".fpb-replace-*")), [])

    def test_replace_text_overlapping_match_and_chmod_race_conflict(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "source.txt"
            source.write_bytes(b"aaa")
            action = {"action": "replace_text", "path": "source.txt",
                      "expected_file_sha256": hashlib.sha256(b"aaa").hexdigest(),
                      "old_text": "aa", "new_text": "b"}
            self.assertEqual(apply_replace_text(root, action)["reason"],
                             "old_text_not_unique")
            self.assertEqual(source.read_bytes(), b"aaa")
            source.write_bytes(b"old\n")
            source.chmod(0o644)
            action.update(expected_file_sha256=hashlib.sha256(b"old\n").hexdigest(),
                          old_text="old", new_text="new")
            from future_prediction_bench import replace_text as module
            original_stat = module.os.stat
            count = 0

            def change_mode_before_final_stat(path, *args, **kwargs):
                nonlocal count
                if path == "source.txt" and kwargs.get("dir_fd") is not None:
                    count += 1
                    if count == 2:
                        source.chmod(0o600)
                return original_stat(path, *args, **kwargs)

            with mock.patch("future_prediction_bench.replace_text.os.stat",
                            side_effect=change_mode_before_final_stat):
                result = apply_replace_text(root, action)
            self.assertEqual(result["reason"], "file_changed")
            self.assertEqual(source.read_bytes(), b"old\n")
            self.assertEqual(stat.S_IMODE(source.stat().st_mode), 0o600)
            self.assertEqual(list(Path(root).glob(".fpb-replace-*")), [])
            with mock.patch("future_prediction_bench.replace_text.os.stat",
                            side_effect=FileNotFoundError("concurrent deletion")):
                result = apply_replace_text(root, action)
            self.assertEqual(result["reason"], "file_changed")
            self.assertEqual(source.read_bytes(), b"old\n")

    def test_replace_text_requires_frozen_helper_identity(self):
        for forged in ({"source_sha256": "0" * 64},
                       {"guest_program_sha256": "1" * 64}):
            with self.subTest(forged=forged):
                adapter = self.adapter()
                task = dict(self.task)
                task["tool_manifest"] = [*self.task["tool_manifest"],
                                         {"name": "replace_text", "description": "Exact edit."}]
                task["metadata"] = {"replace_text_helper_binding": {
                    **replace_text_helper_binding(), **forged}}
                env = RealWorldEnv(task, adapter)
                with self.assertRaises(ValueError):
                    env.reset("scripted-policy")
                self.assertIsNone(adapter.workspace)

    def test_replace_text_action_transport_bound_matches_vm(self):
        adapter = self.adapter()
        task = dict(self.task)
        task["tool_manifest"] = [*self.task["tool_manifest"],
                                 {"name": "replace_text", "description": "Exact edit."}]
        task["metadata"] = {"replace_text_helper_binding": replace_text_helper_binding()}
        env = RealWorldEnv(task, adapter)
        try:
            env.reset("scripted-policy")
            original = (adapter.workspace / "math_utils.py").read_bytes()
            action = {"action": "replace_text", "path": "math_utils.py",
                      "expected_file_sha256": hashlib.sha256(original).hexdigest(),
                      "old_text": "x" * 400, "new_text": ""}
            self.assertEqual(env.step(action)["observation"]["reason"],
                             "old_text_not_unique")
            for invalid in (dict(action, old_text="x" * 512),
                            dict(action, old_text="x" * 800),
                            dict(action, old_text="x" * 1024),
                            dict(action, path="界" * 200)):
                self.assertEqual(env.step(invalid)["observation"]["status"], "error")
            self.assertEqual((adapter.workspace / "math_utils.py").read_bytes(), original)
            self.assertEqual(adapter.get_state()["metrics"]["checkpoints_created"], 1)
        finally:
            adapter.close()

    def test_replace_text_hardlink_is_policy_error_without_mutation(self):
        adapter = self.adapter()
        task = dict(self.task)
        task["tool_manifest"] = [*self.task["tool_manifest"],
                                 {"name": "replace_text", "description": "Exact edit."}]
        task["metadata"] = {"replace_text_helper_binding": replace_text_helper_binding()}
        env = RealWorldEnv(task, adapter)
        try:
            env.reset("scripted-policy")
            source = adapter.workspace / "math_utils.py"
            original = source.read_bytes()
            (adapter.workspace / "alias.py").hardlink_to(source)
            action = {"action": "replace_text", "path": "math_utils.py",
                      "expected_file_sha256": hashlib.sha256(original).hexdigest(),
                      "old_text": "a-b", "new_text": "a+b"}
            self.assertEqual(env.step(action)["observation"],
                             {"status": "error", "reason": "adapter_error",
                              "error_type": "ValueError"})
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual((adapter.workspace / "alias.py").read_bytes(), original)
            self.assertEqual(adapter.get_state()["metrics"]["checkpoints_created"], 1)
        finally:
            adapter.close()

    def test_replace_text_io_failure_interrupts_without_policy_reward(self):
        adapter = self.adapter()
        task = dict(self.task)
        task["tool_manifest"] = [*self.task["tool_manifest"],
                                 {"name": "replace_text", "description": "Exact edit."}]
        task["metadata"] = {"replace_text_helper_binding": replace_text_helper_binding()}
        env = RealWorldEnv(task, adapter)
        try:
            env.reset("scripted-policy")
            source = adapter.workspace / "math_utils.py"
            original = source.read_bytes()
            action = {"action": "replace_text", "path": "math_utils.py",
                      "expected_file_sha256": hashlib.sha256(original).hexdigest(),
                      "old_text": "a-b", "new_text": "a+b"}
            with mock.patch("future_prediction_bench.replace_text.apply",
                            side_effect=OSError(errno.EIO, "disk I/O")):
                result = env.step(action)
            self.assertEqual(result["info"]["status"], "interrupted")
            self.assertIsNone(result["reward"])
            self.assertEqual(source.read_bytes(), original)
        finally:
            adapter.close()

    def test_actor_runtime_failure_interrupts_instead_of_policy_miss(self):
        adapter = self.adapter()
        env = RealWorldEnv(self.task, adapter)
        try:
            env.reset("scripted-policy")
            adapter.fail_actor = True
            result = env.step({"action": "run_visible_checks"})
            self.assertEqual(result["info"]["status"], "interrupted")
            self.assertIsNone(result["reward"])
            self.assertIsNone(env.export_trajectory())
            with self.assertRaises(ValueError):
                env.step({"action": "submit"})
        finally:
            adapter.close()

    def test_uncorrected_submission_is_graded_zero(self):
        adapter = self.adapter()
        env = RealWorldEnv(self.task, adapter)
        try:
            env.reset("scripted-policy")
            env.step({"action": "submit"})
            # A trusted host comparison is the source of the grade, not a
            # policy-controlled JSON document printed from the container.
            original = adapter._docker
            def wrong_first_case(arguments, *, timeout=None):
                result = original(arguments, timeout=timeout)
                if arguments[0] == "run" and "--detach" not in arguments and "add(2, 3)" in " ".join(arguments):
                    return subprocess.CompletedProcess(arguments, 0, b"-1\n", b"")
                return result
            adapter._docker = wrong_first_case
            self.assertEqual(env.verify()["reward"], 0.0)
        finally:
            adapter.close()

    def test_hidden_verifier_must_be_disjoint_from_seed(self):
        with self.assertRaises(ValueError):
            FakeDockerAdapter(seed_dir=self.seed, verifier_dir=self.seed,
                              image="fixture:local", output_root=self.root / "output")

    def test_frozen_artifacts_and_filesystem_state_are_checked(self):
        adapter = self.adapter()
        binding = adapter.artifact_binding()
        nonfixture = {**self.task, "is_fixture": False,
                      "metadata": {"artifact_binding": binding}}
        env = RealWorldEnv(nonfixture, adapter)
        try:
            env.reset("scripted-policy")
            self.assertEqual(adapter.expected_binding, binding)
        finally:
            adapter.close()
        before = _workspace_digest(self.seed)
        (self.seed / "empty-directory").mkdir()
        self.assertNotEqual(_workspace_digest(self.seed), before)
        changed = self.adapter()
        with self.assertRaises(ValueError):
            RealWorldEnv(nonfixture, changed).reset("scripted-policy")
        self.assertIsNone(changed.container_name)

    def test_branch_restores_frozen_files_into_independent_workspaces(self):
        adapter = self.adapter()
        env = RealWorldEnv(self.task, adapter)
        branches = []
        try:
            env.reset("scripted-policy")
            env.step({"action": "write_file", "path": "math_utils.py",
                      "content": "def add(a, b): return a+b\n"})
            ref = env.create_branch_checkpoint()
            self.assertEqual(adapter.get_state()["metrics"]["docker_starts"], 2)
            self.assertEqual(adapter.get_state()["metrics"]["branch_checkpoints_created"], 1)
            for branch_id in ("one", "two"):
                branch_adapter = self.adapter()
                branch = env.fork_from_checkpoint(ref, branch_adapter, branch_id=branch_id)
                branches.append(branch_adapter)
                observed = branch.step({"action": "read_file", "path": "math_utils.py"})
                self.assertEqual(observed["observation"]["text"], "def add(a, b): return a+b\n")
            branches[0].workspace.joinpath("math_utils.py").write_text("first branch\n")
            self.assertEqual(branches[1].workspace.joinpath("math_utils.py").read_text(),
                             "def add(a, b): return a+b\n")
            self.assertEqual(adapter.workspace.joinpath("math_utils.py").read_text(),
                             "def add(a, b): return a+b\n")
            self.assertEqual(_workspace_digest(Path(ref["snapshot_path"])), ref["workspace_sha256"])
        finally:
            for branch_adapter in branches:
                branch_adapter.close()
            adapter.close()

    def test_tampered_branch_snapshot_cannot_be_restored(self):
        adapter = self.adapter()
        env = RealWorldEnv(self.task, adapter)
        branch_adapter = self.adapter()
        try:
            env.reset("scripted-policy")
            ref = env.create_branch_checkpoint()
            Path(ref["snapshot_path"]).joinpath("math_utils.py").write_text("tampered\n")
            with self.assertRaisesRegex(ValueError, "Branch adapter reset failed"):
                env.fork_from_checkpoint(ref, branch_adapter, branch_id="tampered")
            self.assertIsNone(branch_adapter.container_name)
        finally:
            branch_adapter.close()
            adapter.close()


if __name__ == "__main__":
    unittest.main()
