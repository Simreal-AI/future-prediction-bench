"""State, reward, and timing-contract tests for the simulated CI task world."""

from __future__ import annotations

import tempfile
import unittest

from examples.collinear_ci_triage.benchmark import benchmark
from examples.collinear_ci_triage.world import (
    TARGET_RUN, TARGET_TIMEOUT_MS, HostVerifier, SeedArtifact, solved_trace, step,
)


class CITriageWorldTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.seed = SeedArtifact(self.directory.name)
        self.verifier = HostVerifier(self.seed)

    def test_discovery_and_step_hide_reward_and_expected_state(self):
        with self.seed.new_world("seed_copy") as world:
            tools = world.handle("GET", "/tools")
            self.assertEqual({tool["name"] for tool in tools["tools"]}, {
                "inspect_failure", "update_job_timeout", "queue_rerun", "post_triage_note"})
            self.assertNotIn("reward", repr(tools).lower())
            self.assertNotIn(str(TARGET_TIMEOUT_MS), repr(tools))
            inspected = step(world, "inspect_failure", run_id=TARGET_RUN)
            self.assertEqual(inspected["observation"]["job_config"]["timeout_ms"], 60_000)
            self.assertNotIn("reward", inspected)
            self.assertEqual(world.state_digest(), self.seed.initial_digest)
            self.assertEqual(world.trace[0]["state_diff"], [])

    def test_seed_copy_and_sql_reseed_are_deterministic_and_independent(self):
        with (self.seed.new_world("seed_copy") as first,
              self.seed.new_world("seed_copy") as second,
              self.seed.new_world("sql_reseed") as baseline):
            self.assertEqual(len({first.database_path, second.database_path,
                                  baseline.database_path}), 3)
            self.assertEqual({first.initial_digest, second.initial_digest,
                              baseline.initial_digest}, {self.seed.initial_digest})
            step(first, "update_job_timeout", run_id=TARGET_RUN,
                 timeout_ms=TARGET_TIMEOUT_MS)
            self.assertNotEqual(first.state_digest(), second.state_digest())
            self.assertEqual(second.state_digest(), baseline.state_digest())

    def test_programmatic_grade_requires_exact_application_change(self):
        with self.seed.new_world("seed_copy") as world:
            observations = solved_trace(world)
            self.assertEqual(observations[-1]["observation"]["status"], "passed")
            grade = self.verifier.verify(world)
            self.assertEqual(grade["reward"], 1.0)
            self.assertEqual(len(world.trace), 3)
            self.assertEqual(world.trace[0]["state_diff"], [])
            self.assertEqual({change["table"] for change in world.trace[1]["state_diff"]},
                             {"job_config", "meta"})
            self.assertEqual({change["table"] for change in world.trace[2]["state_diff"]},
                             {"reruns", "meta"})
            self.assertEqual(world.trace[1]["after_digest"],
                             world.trace[2]["before_digest"])

    def test_text_only_claim_and_forged_trace_cannot_earn_reward(self):
        with self.seed.new_world("seed_copy") as world:
            note = step(world, "post_triage_note", run_id=TARGET_RUN,
                        body="Timeout fixed and rerun passed.")
            self.assertFalse(note.get("is_error", False))
            self.assertEqual(self.verifier.verify(world)["reward"], 0.0)
        with self.seed.new_world("seed_copy") as world:
            initial = world.state_digest()
            world.trace.append({"action": {"tool_name": "queue_rerun", "parameters": {
                "run_id": TARGET_RUN}}, "response": {"observation": "passed"},
                "before_digest": initial, "after_digest": initial,
                "state_diff": [], "step_seconds": 0.0})
            self.assertEqual(self.verifier.verify(world)["reward"], 0.0)

    def test_wrong_target_overbroad_fix_and_wrong_order_are_rejected(self):
        with self.seed.new_world("seed_copy") as world:
            step(world, "queue_rerun", run_id=TARGET_RUN)
            self.assertEqual(self.verifier.verify(world)["reward"], 0.0)
        with self.seed.new_world("seed_copy") as world:
            step(world, "update_job_timeout", run_id=TARGET_RUN, timeout_ms=300_000)
            step(world, "queue_rerun", run_id=TARGET_RUN)
            self.assertEqual(self.verifier.verify(world)["reward"], 0.0)
        with self.seed.new_world("seed_copy") as world:
            step(world, "update_job_timeout", run_id=1843,
                 timeout_ms=TARGET_TIMEOUT_MS)
            step(world, "queue_rerun", run_id=1843)
            self.assertEqual(self.verifier.verify(world)["reward"], 0.0)

    def test_invalid_action_is_bounded_and_does_not_mutate_state(self):
        with self.seed.new_world("seed_copy") as world:
            start = world.state_digest()
            self.assertTrue(step(world, "update_job_timeout", run_id=TARGET_RUN,
                                 timeout_ms="120000")["is_error"])
            self.assertTrue(step(world, "inspect_failure", run_id="1842 OR 1=1")["is_error"])
            self.assertTrue(world.handle("POST", "/step", {"action": {
                "tool_name": "queue_rerun", "parameters": {"run_id": object()}}})["is_error"])
            self.assertEqual(world.state_digest(), start)
            self.assertEqual(world.state()["meta"][0]["value"], 0)

    def test_step_budget_prevents_unbounded_trace_growth(self):
        with self.seed.new_world("seed_copy") as world:
            for _ in range(world.MAX_STEPS):
                self.assertNotIn("is_error", step(world, "inspect_failure", run_id=TARGET_RUN))
            self.assertTrue(step(world, "inspect_failure", run_id=TARGET_RUN)["is_error"])
            self.assertEqual(len(world.trace), world.MAX_STEPS)
            self.assertEqual(world.state_digest(), self.seed.initial_digest)

    def test_malformed_step_attempts_consume_budget_and_have_empty_diffs(self):
        with self.seed.new_world("seed_copy") as world:
            for _ in range(world.MAX_STEPS):
                response = world.handle("POST", "/step", {"not_action": {}})
                self.assertTrue(response["is_error"])
            self.assertEqual(len(world.trace), world.MAX_STEPS)
            self.assertTrue(all(entry["state_diff"] == [] for entry in world.trace))
            self.assertTrue(all(entry["before_digest"] == entry["after_digest"]
                                for entry in world.trace))
            self.assertIn("budget exceeded", step(
                world, "inspect_failure", run_id=TARGET_RUN)["observation"])
            self.assertEqual(world.state_digest(), self.seed.initial_digest)

    def test_paired_benchmark_reports_exact_parity_and_no_false_rewards(self):
        report = benchmark(repetitions=5, warmup=1)
        self.assertEqual(len(report["pairs"]), 5)
        self.assertEqual(report["correctness"]["false_rewards"], 0)
        self.assertTrue(report["correctness"]["independent_siblings"])
        for pair in report["pairs"]:
            self.assertEqual(pair["sql_reseed"]["final_digest"],
                             pair["seed_copy"]["final_digest"])
            self.assertEqual(pair["sql_reseed"]["reward"], 1.0)
            self.assertEqual(pair["seed_copy"]["reward"], 1.0)
            self.assertEqual(len(pair["seed_copy"]["step_seconds"]), 3)


if __name__ == "__main__":
    unittest.main()
