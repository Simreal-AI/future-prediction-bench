"""Offline contract tests for the optional pinned ROLL GEMRunner bridge."""

import json
import importlib.util
import logging
import os
import subprocess
import sys
import types
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from examples.official_roll.gem_bridge import RealWorldGemBridge, RewardNotVerified
from examples.official_roll.guarded_runner import (
    guarded_gem_runner_class, require_verified_roll_result,
)
from future_prediction_bench.realworld import RealWorldEnv


START = datetime(2030, 1, 1, 10, tzinfo=timezone.utc)


class Clock:
    def __init__(self):
        self.now_value = START
        self.seconds = 0.0

    def now(self):
        return self.now_value

    def monotonic(self):
        return self.seconds

    def advance(self, seconds):
        self.now_value += timedelta(seconds=seconds)
        self.seconds += seconds


def task(*, seed, verify_delay=0, max_actions=3):
    return {
        "schema_version": "realworld-0.1", "task_id": f"roll-fixture-{seed}",
        "event_id": f"roll-event-{seed}", "cluster_id": f"roll-cluster-{seed}",
        "split": "train", "prompt": "Read a file, then submit for hidden grading.",
        "issued_at": START.isoformat(),
        "action_deadline": (START + timedelta(hours=2)).isoformat(),
        "outcome_not_before": START.isoformat(),
        "verify_after": (START + timedelta(seconds=verify_delay)).isoformat(),
        "tool_manifest": [
            {"name": "read_file", "description": "Read a file."},
            {"name": "submit", "description": "Submit the candidate."},
        ],
        "reward_contract": {"id": "fixture", "description": "Hidden fixture grade.",
                            "min_reward": 0.0, "max_reward": 1.0},
        "budgets": {"max_actions": max_actions, "max_wall_seconds": 7200,
                    "verification_cooldown_seconds": 10, "max_verifications": 3},
        "is_fixture": True,
        "metadata": {"private_fixture_key": "never-show-me"},
    }


class Adapter:
    def __init__(self, proposals=None):
        self.proposals = list(proposals or [])
        self.actions = []
        self.verify_calls = 0
        self.closed = False

    def reset(self, specification, *, now):
        return {"workspace": "visible fixture"}

    def step(self, action, *, now):
        self.actions.append(action)
        if action["action"] == "submit":
            return {"observation": {"status": "submitted"}, "terminated": True}
        return {"observation": {"status": "tool_result"}, "terminated": False}

    def verify(self, *, now):
        self.verify_calls += 1
        if self.proposals:
            return self.proposals.pop(0)
        return {"status": "resolved", "reward": 1.0,
                "evidence": {"hidden_checks": "private-verifier"},
                "available_at": now.isoformat()}

    def get_state(self):
        return {"actions": len(self.actions)}

    def close(self):
        self.closed = True


class RollBridgeTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.adapters = []
        self.seeds = []

    def bridge(self, *, proposals=None, verify_delay=0, max_actions=3):
        def factory(seed):
            self.seeds.append(seed)
            adapter = Adapter(proposals)
            self.adapters.append(adapter)
            return RealWorldEnv(task(seed=seed, verify_delay=verify_delay,
                                     max_actions=max_actions), adapter,
                                clock=self.clock.now,
                                monotonic_clock=self.clock.monotonic)
        return RealWorldGemBridge(factory)

    def test_graded_episode_matches_gem_tuple_and_hides_verifier(self):
        bridge = self.bridge()
        opening, info = bridge.reset(seed=41)
        self.assertEqual(self.seeds, [41])
        self.assertEqual(json.loads(opening)["task"]["task_id"], "roll-fixture-41")
        self.assertIn("env_instruction", info)
        self.assertNotIn("never-show-me", opening + json.dumps(info))
        intermediate = bridge.step('{"action":"read_file","path":"main.py"}')
        self.assertEqual(len(intermediate), 5)
        self.assertEqual(intermediate[1:4], (0.0, False, False))
        terminal = bridge.step('{"action":"submit"}')
        self.assertEqual(len(terminal), 5)
        self.assertEqual(terminal[1:4], (1.0, True, False))
        self.assertEqual(terminal[4]["status"], "graded")
        self.assertEqual(self.adapters[0].verify_calls, 1)
        self.assertNotIn("private-verifier", json.dumps(terminal))
        self.assertEqual(bridge.env.status, "graded")
        bridge.close()
        self.assertTrue(self.adapters[0].closed)

    def test_pending_not_due_has_no_roll_reward_or_second_submit(self):
        bridge = self.bridge(verify_delay=60)
        bridge.reset(7)
        with self.assertRaises(RewardNotVerified) as context:
            bridge.step('{"action":"submit"}')
        error = context.exception
        self.assertEqual((error.status, error.reason), ("pending", "not_due"))
        self.assertIsNotNone(error.next_verify_at)
        self.assertEqual(self.adapters[0].verify_calls, 0)
        self.assertEqual(len(self.adapters[0].actions), 1)
        with self.assertRaises(RewardNotVerified):
            bridge.reset(8)
        with self.assertRaises(ValueError):
            bridge.step('{"action":"submit"}')
        self.assertEqual(self.seeds, [7])
        self.clock.advance(60)
        self.assertEqual(bridge.collect_verified_reward(), 1.0)
        self.assertEqual(self.adapters[0].verify_calls, 1)

    def test_verifier_pending_and_void_never_become_zero(self):
        pending = self.bridge(proposals=[{"status": "pending", "reason": "source_not_published"}])
        pending.reset(3)
        with self.assertRaises(RewardNotVerified) as context:
            pending.step('{"action":"submit"}')
        self.assertEqual(context.exception.reason, "source_not_published")
        with self.assertRaises(RewardNotVerified) as cooldown:
            pending.collect_verified_reward()
        self.assertEqual(cooldown.exception.reason, "cooldown")
        self.clock.advance(10)
        self.assertEqual(pending.collect_verified_reward(), 1.0)

        voided = self.bridge(proposals=[{"status": "void", "reason": "source_removed",
                                        "evidence": {"secret": "not-for-policy"}}])
        voided.reset(4)
        with self.assertRaises(RewardNotVerified) as context:
            voided.step('{"action":"submit"}')
        self.assertEqual((context.exception.status, context.exception.reason),
                         ("void", "source_removed"))
        self.assertNotIn("not-for-policy", str(context.exception))

    def test_graded_zero_is_distinct_from_ungraded_terminal(self):
        proposals = [{"status": "resolved", "reward": 0.0,
                      "evidence": {"checks": "failed"},
                      "available_at": START.isoformat()}]
        graded_zero = self.bridge(proposals=proposals)
        graded_zero.reset(5)
        self.assertEqual(graded_zero.step('{"action":"submit"}')[1:4], (0.0, True, False))
        self.assertEqual(graded_zero.env.status, "graded")

        missed = self.bridge(max_actions=1)
        missed.reset(6)
        with self.assertRaises(RewardNotVerified) as context:
            missed.step('{"action":"read_file"}')
        self.assertEqual(context.exception.status, "missed")
        self.assertIsNone(missed.env.reward)

    def test_malformed_json_consumes_action_budget(self):
        bridge = self.bridge()
        bridge.reset(9)
        for malformed in ("not JSON", '{"action":"submit","value":NaN}'):
            observation, reward, terminated, truncated, _ = bridge.step(malformed)
            self.assertEqual((reward, terminated, truncated), (0.0, False, False))
            self.assertEqual(json.loads(observation)["reason"], "tool_not_available")
        self.assertEqual(bridge.env.actions_used, 2)
        self.assertEqual(self.adapters[0].actions, [])

    def test_failed_reset_closes_adapter(self):
        clock = self.clock

        class SlowAdapter(Adapter):
            def reset(self, specification, *, now):
                clock.advance(7201)
                return super().reset(specification, now=now)

        adapter = SlowAdapter()
        bridge = RealWorldGemBridge(lambda seed: RealWorldEnv(
            task(seed=seed), adapter, clock=clock.now,
            monotonic_clock=clock.monotonic))
        with self.assertRaises(RewardNotVerified):
            bridge.reset(10)
        self.assertTrue(adapter.closed)
        self.assertIsNone(bridge.env)

    def test_result_guard_rejects_unsubmitted_zero_but_accepts_verified_zero(self):
        unresolved = self.bridge()
        unresolved.reset(11)
        unresolved.step('{"action":"read_file"}')
        stock_zero = types.SimpleNamespace(status="Finished", score=0.0,
                                           step_scores=[0.0])
        with self.assertRaises(RewardNotVerified) as context:
            require_verified_roll_result(stock_zero, unresolved)
        self.assertEqual(context.exception.status, "active")

        proposal = {"status": "resolved", "reward": 0.0,
                    "evidence": {"checks": "failed"},
                    "available_at": START.isoformat()}
        verified_zero = self.bridge(proposals=[proposal])
        verified_zero.reset(12)
        verified_zero.step('{"action":"submit"}')
        self.assertIs(require_verified_roll_result(stock_zero, verified_zero), stock_zero)
        mismatched = types.SimpleNamespace(status="Finished", score=1.0,
                                           step_scores=[0.0])
        with self.assertRaises(RewardNotVerified) as context:
            require_verified_roll_result(mismatched, verified_zero)
        self.assertEqual(context.exception.status, "inconsistent_result")

    def test_optional_pinned_upstream_gem_runner_execution(self):
        """Execute unmodified upstream runner methods with only imports/LLM stubbed.

        The official checkout is optional for public offline tests. When set,
        FPB_UPSTREAM_ROLL must point at the exact audited source commit.
        """
        checkout_arg = os.environ.get("FPB_UPSTREAM_ROLL")
        if not checkout_arg:
            self.skipTest("Set FPB_UPSTREAM_ROLL to a pinned official ROLL checkout")
        checkout = Path(checkout_arg).resolve()
        actual_sha = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
        self.assertEqual(actual_sha, "192b1a01ea61c113b2deb543f7b115783038dff8")
        runner_dir = checkout / "roll" / "pipeline" / "agentic" / "agent_runner"
        dirty_runner = subprocess.check_output(
            ["git", "-C", str(checkout), "status", "--porcelain", "--",
             "roll/pipeline/agentic/agent_runner/base.py",
             "roll/pipeline/agentic/agent_runner/gem_runner.py",
             "roll/utils/str_utils.py"], text=True)
        self.assertEqual(dirty_runner, "", "Audited upstream runner files must be clean")

        modules = {}
        for name in ("roll", "roll.pipeline", "roll.pipeline.agentic",
                     "roll.pipeline.agentic.agent_runner", "roll.pipeline.agentic.env",
                     "roll.utils"):
            module = types.ModuleType(name)
            module.__path__ = []
            modules[name] = module
        modules["omegaconf"] = types.ModuleType("omegaconf")
        modules["omegaconf"].DictConfig = dict
        modules["omegaconf"].OmegaConf = object
        modules["roll.utils.logging"] = types.ModuleType("roll.utils.logging")
        modules["roll.utils.logging"].get_logger = lambda: logging.getLogger("roll-pinned-smoke")
        modules["roll.pipeline.agentic.env"].gem = types.ModuleType("gem")
        selected = {}
        modules["roll.pipeline.agentic.env"].gem.make = lambda **kwargs: selected["bridge"]

        def load_official(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            self.assertIsNotNone(spec)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module

        with patch.dict(sys.modules, modules):
            load_official("roll.utils.str_utils", checkout / "roll" / "utils" / "str_utils.py")
            load_official("roll.pipeline.agentic.agent_runner.base", runner_dir / "base.py")
            official = load_official("roll.pipeline.agentic.agent_runner.gem_runner",
                                     runner_dir / "gem_runner.py")
            GuardedGEMRunner = guarded_gem_runner_class(official.GEMRunner)

            def run_with_actions(bridge, actions, *, max_steps):
                selected["bridge"] = bridge
                runner = official.GEMRunner(
                    base_url="http://unused.local", env_id=0,
                    env_config={"env_type": "fixture", "max_steps": max_steps,
                                "agent_system_template": "Fixture agent",
                                "agent_template": "{observation}"})
                queued = iter(actions)
                runner._llm_request = lambda client, messages: {
                    "choices": [{"message": {"content": next(queued)}}]}
                return runner

            def guarded_with_actions(bridge, actions, *, max_steps):
                runner = GuardedGEMRunner(
                    base_url="http://unused.local", env_id=0,
                    env_config={"env_type": "fixture", "max_steps": max_steps,
                                "agent_system_template": "Fixture agent",
                                "agent_template": "{observation}"},
                    env_factory=bridge.env_factory)
                queued = iter(actions)
                runner._llm_request = lambda client, messages: {
                    "choices": [{"message": {"content": next(queued)}}]}
                return runner

            verified = self.bridge()
            runner = run_with_actions(verified,
                                      ['{"action":"read_file"}', '{"action":"submit"}'],
                                      max_steps=2)
            try:
                result = runner.run_job(101)
                self.assertIsInstance(result, official.EpisodeResult)
                self.assertEqual((result.status, result.score, result.step_scores),
                                 ("Finished", 1.0, [0.0, 1.0]))
                self.assertEqual(verified.env.status, "graded")
            finally:
                runner.teardown()

            # The local subclass blocks exactly the stock max-step false zero.
            unsubmitted_guarded = self.bridge()
            runner = guarded_with_actions(unsubmitted_guarded,
                                          ['{"action":"read_file"}'], max_steps=1)
            try:
                with self.assertRaises(RewardNotVerified) as context:
                    runner.run_job(104)
                self.assertEqual(context.exception.status, "active")
                self.assertIsNone(runner.env.env.reward)
            finally:
                runner.teardown()

            zero_proposal = {"status": "resolved", "reward": 0.0,
                             "evidence": {"checks": "failed"},
                             "available_at": START.isoformat()}
            verified_zero_guarded = self.bridge(proposals=[zero_proposal])
            runner = guarded_with_actions(verified_zero_guarded,
                                          ['{"action":"submit"}'], max_steps=1)
            try:
                result = runner.run_job(105)
                self.assertIsInstance(result, official.EpisodeResult)
                self.assertEqual((result.status, result.score, result.step_scores),
                                 ("Finished", 0.0, [0.0]))
                self.assertEqual(runner.env.env.status, "graded")
            finally:
                runner.teardown()

            # Stock treats this inference error as Finished/0; guard rejects it.
            error_guarded = self.bridge()
            runner = guarded_with_actions(error_guarded, [], max_steps=1)
            runner._llm_request = lambda client, messages: {"error": "fake_inference_error"}
            try:
                with self.assertRaises(RewardNotVerified) as context:
                    runner.run_job(106)
                self.assertEqual(context.exception.status, "active")
            finally:
                runner.teardown()

            delayed = self.bridge(verify_delay=60)
            runner = run_with_actions(delayed, ['{"action":"submit"}'], max_steps=1)
            try:
                with self.assertRaises(RewardNotVerified) as context:
                    runner.run_job(102)
                self.assertEqual((context.exception.status, context.exception.reason),
                                 ("pending", "not_due"))
                self.assertIsNone(delayed.env.reward)
                self.assertEqual(self.adapters[-1].verify_calls, 0)
            finally:
                runner.teardown()

            # This negative control demonstrates why the stock upstream runner
            # cannot be used as our trainer without a terminal-grade guard.
            unsubmitted = self.bridge()
            runner = run_with_actions(unsubmitted, ['{"action":"read_file"}'], max_steps=1)
            try:
                result = runner.run_job(103)
                self.assertEqual((result.status, result.score), ("Finished", 0.0))
                self.assertEqual(unsubmitted.env.status, "active")
                self.assertIsNone(unsubmitted.env.reward)
            finally:
                runner.teardown()


if __name__ == "__main__":
    unittest.main()
