"""Offline contract, isolation, timeout and reward gates for Humanize opt-in."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from examples.cooperative_realworld.host_runtime import (
    CooperativePrivateCases, CooperativeTransport,
)
from examples.cooperative_realworld_humanize.benchmark import _equivalence
from examples.cooperative_realworld_humanize.make_task_v2 import (
    NEW_TEXT, OLD_TEXT, TASK_ID,
)
from examples.realworld_humanize.make_task import VISIBLE_CHECK, _repair
from future_prediction_bench.coding_env import _workspace_digest
from future_prediction_bench.prepared_microvm import _clean_task, _HUMANIZE_PROFILE
from future_prediction_bench.microvm_coding import replace_text_helper_binding


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _row(condition, *, mode):
    original, repaired = "1" * 64, "2" * 64
    passing = 14 if mode == "repair" else 5
    actions = [{"path": "src/humanize/filesize.py", "sha256": original,
                "text": "public preview", "truncated": False}]
    if mode == "repair":
        actions.append({"path": "src/humanize/filesize.py", "sha256": repaired})
    actions.append({"status": "submitted", "snapshot_kind": (
        "quiescent_process_fork_frozen_overlay_v1" if condition == "cooperative"
        else "full_vm_state_qcow2_v1")})
    return {"condition": condition, "mode": mode,
            "task_sha256": ("a" if condition == "cooperative" else "b") * 64,
            "reward": 1.0 if mode == "repair" else 0.0,
            "passed_cases": passing,
            "case_results": [{"return_code": 0, "stdout_sha256": "c" * 64,
                              "passed": index < passing} for index in range(14)],
            "opening_observation": {
                "task_id": TASK_ID, "workspace_root": "/workspace",
                "visible_check": list(VISIBLE_CHECK),
                "tools": (["read_file", "replace_text", "submit"]
                          if condition == "cooperative" else
                          ["read_file", "write_file", "replace_text",
                           "run_visible_checks", "submit"])},
            "action_observations": actions}


class HumanizeCooperativeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        task_dir = self.root / "task"
        seed = task_dir / "seed/src/humanize/filesize.py"
        seed.parent.mkdir(parents=True)
        seed.write_text("def naturalsize(value): return str(value)\n", encoding="utf-8")
        now = datetime.now(timezone.utc)
        task = {
            "schema_version": "realworld-0.1", "task_id": TASK_ID,
            "event_id": TASK_ID, "cluster_id": "humanize-filesize-naturalsize",
            "split": "train", "prompt": "Repair pinned Humanize.",
            "issued_at": (now - timedelta(minutes=1)).isoformat(),
            "action_deadline": (now + timedelta(minutes=10)).isoformat(),
            "outcome_not_before": (now - timedelta(minutes=1)).isoformat(),
            "verify_after": (now - timedelta(minutes=1)).isoformat(),
            "tool_manifest": [{"name": name, "description": name} for name in
                              ("read_file", "write_file", "replace_text",
                               "run_visible_checks", "submit")],
            "reward_contract": {"id": "humanize4150-naturalsize-host-cases-v1",
                                "description": "All 14 host cases pass.",
                                "min_reward": 0, "max_reward": 1},
            "budgets": {"max_actions": 8, "max_wall_seconds": 900},
            "is_fixture": True, "adapter_id": "docker_coding", "adapter_version": "0.1",
            "metadata": {"source_sdist_sha256":
                         _HUMANIZE_PROFILE["source_sdist_sha256"],
                         "replace_text_helper_binding": replace_text_helper_binding()},
        }
        self.task_path = task_dir / "task.json"
        _json(self.task_path, task)
        cases = [{"argv": ["python3", "-B", "-c", "print(1)"],
                  "expected_stdout": "1\n", "expected_returncode": 0}
                 for _ in range(14)]
        self.verifier_path = task_dir / "verifier/verify.json"
        _json(self.verifier_path, {"kind": "command_cases_v1", "cases": cases})
        self.resident_path = self.root / "resident.json"
        _json(self.resident_path, {
            "kind": "stateless_python_cases_resident_v1", "task_id": TASK_ID,
            "verifier_sha256": _sha(self.verifier_path.read_bytes()),
            "source_path": "src/humanize/filesize.py",
            "seed_file_sha256": _sha(seed.read_bytes()), "case_count": 14,
            "requires_live_background_process_state": False,
            "requires_shared_case_filesystem_state": False,
            "requires_quiescent_submitted_state": True,
            "allow_unprivileged_case_execution": True,
        })
        self.cooperative_path = self.root / "cooperative.json"
        _json(self.cooperative_path, {
            "kind": "quiescent_process_fork_frozen_overlay_humanize_v1",
            "task_id": TASK_ID, "source_path": "src/humanize/filesize.py",
            "case_count": 14,
            "requires_live_background_process_state": False,
            "requires_shared_case_filesystem_state": False,
            "requires_quiescent_submitted_state": True,
            "resident_case_contract_sha256": _sha(self.resident_path.read_bytes()),
        })
        self.assets_path = self.root / "assets/manifest.json"
        _json(self.assets_path, {
            "schema_version": "synthetic-humanize-assets-v1", "task_id": TASK_ID,
            "architecture": "linux/arm64",
            "seed_workspace_sha256": _workspace_digest(task_dir / "seed"),
            "rootfs_qcow2_sha256": "d" * 64,
        })

    def private(self):
        return CooperativePrivateCases(
            self.task_path, self.verifier_path, self.assets_path,
            contract_path=self.cooperative_path,
            resident_contract_path=self.resident_path)

    def test_humanize_isolation_contract_is_exact_and_host_only(self):
        private = self.private()
        self.assertEqual(private.source_path, "src/humanize/filesize.py")
        self.assertEqual(len(private.cases), 14)
        self.assertEqual(private.binding["checkpoint_kind"],
                         "quiescent_process_fork_frozen_overlay_humanize_v1")
        changed = json.loads(self.cooperative_path.read_text())
        changed["requires_shared_case_filesystem_state"] = True
        _json(self.cooperative_path, changed)
        with self.assertRaisesRegex(ValueError, "cooperative_checkpoint_contract_not_pinned"):
            self.private()

    def test_prepared_profile_binds_humanize_and_rejects_wrong_verifier(self):
        task = json.loads(self.task_path.read_text())
        task["metadata"]["artifact_binding"] = {
            "runtime_kind": "qemu_hvf_full_vm_qcow2_v1",
            "verifier_sha256": _HUMANIZE_PROFILE["verifier_sha256"]}
        self.assertEqual(_clean_task(task)[0]["task_id"], TASK_ID)
        task["metadata"]["artifact_binding"]["verifier_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "pinned full-VM artifact binding"):
            _clean_task(task)

    def test_exact_repair_and_reward_case_parity(self):
        source = "header\n" + OLD_TEXT + "footer\n"
        self.assertEqual(_repair(source), source.replace(OLD_TEXT, NEW_TEXT, 1))
        for mode in ("repair", "baseline"):
            cooperative = _row("cooperative", mode=mode)
            full = _row("prepared_full_vm", mode=mode)
            parity = _equivalence(cooperative, full, mode=mode,
                                  original_sha="1" * 64, repaired_sha="2" * 64)
            self.assertTrue(parity["equal_per_case_results"])
            full["case_results"][0]["stdout_sha256"] = "f" * 64
            with self.assertRaisesRegex(RuntimeError, "host_private_case_result"):
                _equivalence(cooperative, full, mode=mode,
                             original_sha="1" * 64, repaired_sha="2" * 64)
            full = _row("prepared_full_vm", mode=mode)
            full["reward"] = 1.0 - full["reward"]
            with self.assertRaisesRegex(RuntimeError, "host_private_case_result_or_reward"):
                _equivalence(cooperative, full, mode=mode,
                             original_sha="1" * 64, repaired_sha="2" * 64)

    def test_batch_transport_has_finite_timeout_and_keeps_expected_outputs_private(self):
        class Runtime:
            def run_shell(self, command, *, timeout):
                self.timeout = timeout
                self.command = command
                return {"return_code": 1, "stdout": ""}
        runtime = Runtime()
        request = {"v": 1, "seq": 1, "op": "case_batch", "args": {"codes": ["print(1)"]}}
        with self.assertRaisesRegex(Exception, "cooperative_serial_call_failed"):
            CooperativeTransport(runtime).exchange(request)
        self.assertEqual(runtime.timeout, 290.0)
        self.assertNotIn("expected_stdout", runtime.command)

    def test_published_report_has_all_ten_graded_pairs_and_no_case_text(self):
        path = (Path(__file__).resolve().parents[1] / "docs/measurements"
                / "cooperative_humanize_ab_fivepair_2026-09-25.json")
        raw = path.read_text(encoding="utf-8")
        for marker in ("/Users/", "/private/tmp/", "expected_stdout", "stdout_b64",
                       "stdout_sha256", "case_results", "action_observations"):
            self.assertNotIn(marker, raw)
        report = json.loads(raw)
        self.assertEqual((report["status"], report["repetitions"],
                          len(report["pairs"])), ("passed", 5, 10))
        for pair in report["pairs"]:
            self.assertTrue(all(pair["parity"].values()))
            self.assertEqual({row["condition"] for row in pair["episodes"]},
                             {"cooperative", "prepared_full_vm"})
            expected = (14, 1.0) if pair["mode"] == "repair" else (5, 0.0)
            for row in pair["episodes"]:
                self.assertEqual((row["passed_cases"], row["reward"]), expected)


if __name__ == "__main__":
    unittest.main()
