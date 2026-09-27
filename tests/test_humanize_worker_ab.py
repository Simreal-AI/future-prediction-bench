"""Offline contracts for the controlled Humanize Docker verifier-worker A/B."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from examples.realworld_humanize import benchmark_verifier_workers as bench


SOURCE_SHA = "a" * 64
FIXED_SHA = "b" * 64
VERIFIER_SHA = "c" * 64
TASK_SHA = "d" * 64
IMAGE = "sha256:" + "e" * 64


def fake_raw(mode, workers, *, complete=5.0, verify=3.0):
    pass_count = bench.EXPECTED_PASS[mode]
    submitted = SOURCE_SHA if mode == "baseline" else FIXED_SHA
    transitions = [{"observation": {"value": f"{mode}-{i}"}}
                   for i in range(bench.EXPECTED_ACTIONS[mode] - 1)]
    transitions.append({"observation": {"status": "submitted",
                                         "workspace_sha256": submitted}})
    cases = [{"return_code": 0, "stdout_sha256": f"{i:064x}",
              "passed": i < pass_count} for i in range(14)]
    return {
        "task_id": bench.TASK_ID, "status": "graded",
        "reward": bench.EXPECTED_REWARD[mode], "elapsed_seconds": complete,
        "image_sha256": IMAGE,
        "opening": {"task": {"task_sha256": TASK_SHA},
                    "observation": {"workspace_sha256": SOURCE_SHA,
                                    "image_sha256": IMAGE, "tools": ["submit"]}},
        "transitions": transitions,
        "verification": {"evidence": {"case_results": cases,
                                      "workspace_sha256": submitted,
                                      "verifier_sha256": VERIFIER_SHA,
                                      "image_sha256": IMAGE}},
        "environment_metrics": {"actions_used": bench.EXPECTED_ACTIONS[mode]},
        "adapter_metrics": {"checkpoints_created": bench.EXPECTED_CHECKPOINTS[mode],
                            "verifier_workers": workers, "verifier_cases": 14,
                            "verify_seconds": verify},
    }


class HumanizeWorkerABTests(unittest.TestCase):
    def test_three_pairs_alternate_order_and_balance_both_modes(self):
        schedule = bench._schedule()
        self.assertEqual(len(schedule), 12)
        self.assertEqual(schedule[:4], [(0, "baseline", 1), (0, "baseline", 4),
                                        (0, "solution", 1), (0, "solution", 4)])
        self.assertEqual(schedule[4:8], [(1, "solution", 4), (1, "solution", 1),
                                         (1, "baseline", 4), (1, "baseline", 1)])
        for mode in ("baseline", "solution"):
            self.assertEqual(sorted((pair, worker) for pair, m, worker in schedule
                                    if m == mode),
                             [(pair, worker) for pair in range(3) for worker in (1, 4)])

    def test_episode_rejects_wrong_reward_or_verifier_worker(self):
        raw = fake_raw("baseline", 1)
        record = bench._episode(raw, mode="baseline", workers=1,
                                source_sha=SOURCE_SHA, action_sha={"baseline": "x"})
        self.assertEqual(record["passed_cases"], 5)
        self.assertEqual(record["complete_episode_seconds"], 5.0)
        raw["reward"] = 1.0
        with self.assertRaisesRegex(RuntimeError, "contract_failed"):
            bench._episode(raw, mode="baseline", workers=1,
                           source_sha=SOURCE_SHA, action_sha={"baseline": "x"})
        raw = fake_raw("solution", 4)
        raw["adapter_metrics"]["verifier_workers"] = 1
        with self.assertRaisesRegex(RuntimeError, "contract_failed"):
            bench._episode(raw, mode="solution", workers=4,
                           source_sha=SOURCE_SHA, action_sha={"solution": "y"})

    def test_parity_rejects_changed_visible_observation_or_case_vector(self):
        records = []
        action_sha = {"baseline": "x", "solution": "y"}
        for pair, mode, worker in bench._schedule():
            record = bench._episode(fake_raw(mode, worker,
                                             complete=5.0 if worker == 1 else 3.0,
                                             verify=3.0 if worker == 1 else 1.0),
                                    mode=mode, workers=worker, source_sha=SOURCE_SHA,
                                    action_sha=action_sha)
            record["pair"] = pair
            records.append(record)
        report = bench._finalize(records, source_sha=SOURCE_SHA,
                                 action_sha=action_sha)
        self.assertEqual(report["total_episodes"], 12)
        self.assertEqual(report["summary"]["solution"]["4"]["verifier_stage_seconds"]["median"], 1.0)
        self.assertEqual(report["summary"]["baseline"]["paired_differences_seconds_1_minus_4"][0]["complete_episode"], 2.0)
        broken = copy.deepcopy(records)
        broken[1]["visible_observations_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "visible_observations_sha256"):
            bench._finalize(broken, source_sha=SOURCE_SHA, action_sha=action_sha)
        broken = copy.deepcopy(records)
        broken[1]["case_vector_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "case_vector_sha256"):
            bench._finalize(broken, source_sha=SOURCE_SHA, action_sha=action_sha)

    def test_sanitized_report_rejects_raw_case_detail_or_local_path(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "summary.json"
            bench._write_public(path, {"status": "passed", "digest": "a" * 64})
            self.assertEqual(json.loads(path.read_text())["status"], "passed")
            with self.assertRaisesRegex(RuntimeError, "private_detail"):
                bench._write_public(path, {"case_results": [{"passed": True}]})
            with self.assertRaisesRegex(RuntimeError, "private_detail"):
                # Assemble a synthetic home path at runtime so source scans do
                # not mistake this fixture for an actual private machine path.
                local_path = "/" + "Users" + "/someone/work"
                bench._write_public(path, {"local_path": local_path})

    def test_orchestrator_runs_12_episodes_without_docker(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            task_dir = root / "task"
            task_dir.mkdir()
            output = root / "run"
            calls = []

            def fake_runner(**kwargs):
                mode = "baseline" if len(kwargs["actions"]) == 1 else "solution"
                worker = kwargs["verifier_workers"]
                calls.append((mode, worker))
                directory = Path(kwargs["output"])
                directory.mkdir(parents=True)
                (directory / "report.json").write_text(json.dumps(fake_raw(mode, worker)))

            task = {"task_id": bench.TASK_ID,
                    "metadata": {"source_workspace_sha256": SOURCE_SHA}}
            action_sha = {"baseline": "x", "solution": "y"}
            with patch.object(bench, "_preflight", return_value=(
                    task, {"baseline": [{"action": "submit"}],
                           "solution": [{"action": name} for name in
                                        ("read_file", "write_file", "run_visible_checks", "submit")]},
                    action_sha)):
                result = bench.benchmark(image=IMAGE, output=output, task_dir=task_dir,
                                         runner=fake_runner)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(calls, [(mode, worker) for _, mode, worker in bench._schedule()])
            self.assertEqual(json.loads((output / "summary.json").read_text())["total_episodes"], 12)


if __name__ == "__main__":
    unittest.main()
