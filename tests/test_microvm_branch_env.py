"""Full-VM checkpoint lineage and independent RealWorldEnv branch contracts.

The fake serial guest exercises the public host boundary without QEMU; the
separate pinned Boltons integration uses an actual QEMU/HVF guest.
"""

import base64
import hashlib
import json
import re
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from future_prediction_bench.microvm_coding import MicroVMCodingAdapter
from future_prediction_bench.microvm_runtime import MicroVMRuntimeError
from future_prediction_bench.realworld import RealWorldEnv


FILE = "/mnt/root/workspace/bug.py"


class ForkingRuntime:
    kernel_sha256 = "a" * 64
    initramfs_sha256 = "b" * 64
    readonly_disk_paths = ()

    def __init__(self, disk_path, *, files=None):
        self.disk_path = Path(disk_path)
        self.files = dict(files or {FILE: b"seed\n"})
        self.snapshots = {}
        self.children = []
        self.closed = False

    def start(self):
        pass

    def wait_for_serial(self, marker, *, timeout):
        pass

    def run_shell(self, command, *, timeout):
        if command.startswith("for d in /dev/vd?"):
            return {"return_code": 0, "stdout": "/dev/vda:68737173:0000\n/dev/vdb:00000000:53ef\n"}
        if "find /mnt/root/workspace -type l -print" in command:
            return {"return_code": 0, "stdout": ""}
        if "find /mnt/root/workspace -type f -print0 | base64" in command:
            names = b"\0".join(name.encode() for name in self.files) + b"\0"
            return {"return_code": 0, "stdout": base64.b64encode(names).decode() + "\n"}
        if command.startswith("if [ -f "):
            data = self.files.get(FILE)
            if data is None:
                return {"return_code": 0, "stdout": "MISSING\n"}
            digest = hashlib.sha256(data).hexdigest()
            return {"return_code": 0, "stdout":
                    f"{digest}  {FILE}\n{len(data)}\n{base64.b64encode(data).decode()}\n"}
        if command.startswith("mkdir -p ") and " && : > " in command:
            path = command.split(" && : > ", 1)[1]
            self.files[path] = b""
        elif command.startswith("printf '%s' ") and " | base64 -d >> " in command:
            encoded = command.split("'", 3)[3]
            path = command.rsplit(" >> ", 1)[1]
            self.files[path] += base64.b64decode(encoded)
        elif command.startswith("mv -f "):
            _, _, old, new = command.split()
            self.files[new] = self.files.pop(old)
        elif "chroot /mnt/root /usr/local/bin/python3.12" in command:
            # The exact private case remains on the host; the fake guest only
            # returns deterministic candidate output from its own file state.
            raw = self.files[FILE].strip() + b"\n"
            payload = {"return_code": 0,
                       "stdout_b64": base64.b64encode(raw).decode(),
                       "truncated": False}
            encoded = base64.b64encode(json.dumps(payload).encode()).decode()
            return {"return_code": 0, "stdout": f"FPB_CASE_RESULT={encoded}\n"}
        return {"return_code": 0, "stdout": ""}

    def save_snapshot(self, tag):
        self.snapshots[tag] = dict(self.files)
        return {"kind": "full_vm_state_qcow2_v1", "tag": tag}

    def load_snapshot(self, tag):
        self.files = dict(self.snapshots[tag])

    def fork_snapshot(self, tag, child_disk_paths):
        if tag not in self.snapshots:
            raise MicroVMRuntimeError("snapshot_not_found")
        result = []
        for path in child_disk_paths:
            child_path = Path(path)
            child_path.write_bytes(self.disk_path.read_bytes())
            child = ForkingRuntime(child_path, files=self.snapshots[tag])
            child.snapshots[tag] = dict(self.snapshots[tag])
            self.children.append(child)
            result.append(child)
        # QEMU may update live qcow2 container bytes on stop/cont even though
        # the named internal snapshot remains the same logical guest state.
        self.disk_path.write_bytes(self.disk_path.read_bytes() + b"|metadata-update")
        return result

    def get_state(self):
        return {"running": not self.closed}

    def close(self):
        self.closed = True


def fixture_task(binding):
    now = datetime.now(timezone.utc)
    return {"schema_version": "realworld-0.1", "task_id": "vm-branch-fixture",
            "event_id": "vm-branch-fixture", "cluster_id": "vm-branch-fixture",
            "split": "train", "prompt": "Repair the file.",
            "issued_at": (now - timedelta(minutes=1)).isoformat(),
            "action_deadline": (now + timedelta(minutes=10)).isoformat(),
            "outcome_not_before": (now - timedelta(minutes=1)).isoformat(),
            "verify_after": (now - timedelta(minutes=1)).isoformat(),
            "tool_manifest": [{"name": name, "description": name} for name in
                              ("read_file", "write_file", "submit")],
            "reward_contract": {"id": "host_cases", "description": "Private case",
                                "min_reward": 0, "max_reward": 1},
            "budgets": {"max_actions": 8, "max_wall_seconds": 300},
            "is_fixture": True, "metadata": {"artifact_binding": binding}}


class MicroVMBranchEnvTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.disk = self.root / "parent.qcow2"
        self.disk.write_bytes(b"fake-qcow2")
        verifier = self.root / "verifier"
        verifier.mkdir()
        (verifier / "verify.json").write_text(json.dumps({
            "kind": "command_cases_v1", "cases": [{
                "argv": ["python3", "-B", "-c", "print(open('bug.py').read().strip())"],
                "expected_stdout": "good\n", "expected_returncode": 0}]}), encoding="utf-8")
        self.runtime = ForkingRuntime(self.disk)
        self.adapter = MicroVMCodingAdapter(
            self.runtime, verifier_dir=verifier,
            visible_check=["python3", "-B", "-c", "print('visible')"])
        self.env = RealWorldEnv(fixture_task(self.adapter.artifact_binding()), self.adapter)
        self.env.reset("fixture-policy")
        self.addCleanup(self.adapter.close)

    def test_two_child_rewards_state_and_cleanup(self):
        prefix = self.env.get_state()["events"]
        ref = self.env.create_branch_checkpoint()
        self.assertEqual(ref["checkpoint_kind"], "full_vm_state_qcow2_v1")
        self.assertRegex(ref["snapshot_disk_sha256"], r"^[a-f0-9]{64}$")
        first_adapter = self.adapter.branch_adapter(self.root / "first.qcow2")
        second_adapter = self.adapter.branch_adapter(self.root / "second.qcow2")
        first = self.env.fork_from_checkpoint(ref, first_adapter, branch_id="good")
        self.assertNotEqual(hashlib.sha256(self.disk.read_bytes()).hexdigest(),
                            ref["snapshot_disk_sha256"])
        second = self.env.fork_from_checkpoint(ref, second_adapter, branch_id="bad")
        try:
            self.assertEqual(first.events[:len(prefix)], prefix)
            self.assertEqual(first.branch_lineage["checkpoint_kind"], ref["checkpoint_kind"])
            self.assertEqual(first.branch_lineage["snapshot_disk_sha256"], ref["snapshot_disk_sha256"])
            first.step({"action": "write_file", "path": "bug.py", "content": "good\n"})
            second.step({"action": "write_file", "path": "bug.py", "content": "bad\n"})
            self.assertEqual(first.step({"action": "read_file", "path": "bug.py"})["observation"]["text"], "good\n")
            self.assertEqual(second.step({"action": "read_file", "path": "bug.py"})["observation"]["text"], "bad\n")
            self.assertEqual(self.env.step({"action": "read_file", "path": "bug.py"})["observation"]["text"], "seed\n")
            first.step({"action": "submit"})
            second.step({"action": "submit"})
            self.assertEqual(first.verify()["reward"], 1.0)
            self.assertEqual(second.verify()["reward"], 0.0)
            self.assertEqual(first.export_trajectory()["branch_lineage"]["branch_id"], "good")
        finally:
            first_adapter.close()
            second_adapter.close()
        self.assertTrue(all(child.closed for child in self.runtime.children))
        self.assertFalse(self.runtime.closed)

    def test_checkpoint_reference_tampering_rejected(self):
        ref = self.env.create_branch_checkpoint()
        changed = {**ref, "snapshot_disk_sha256": "0" * 64}
        with self.assertRaises(ValueError):
            self.env.fork_from_checkpoint(
                changed, self.adapter.branch_adapter(self.root / "tampered.qcow2"),
                branch_id="tampered")
        self.assertFalse((self.root / "tampered.qcow2").exists())
        self.assertEqual(len(self.runtime.children), 0)

    def test_adapter_rejects_different_frozen_artifact_binding(self):
        ref = self.env.create_branch_checkpoint()
        wrong_task = fixture_task({**self.adapter.expected_binding, "verifier_sha256": "0" * 64})
        child = self.adapter.branch_adapter(self.root / "wrong.qcow2")
        with self.assertRaises(ValueError):
            child.reset_from_checkpoint(wrong_task, ref, now=datetime.now(timezone.utc))
        self.assertFalse((self.root / "wrong.qcow2").exists())


if __name__ == "__main__":
    unittest.main()
