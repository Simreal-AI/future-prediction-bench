"""Offline fixture tests for the separate real-world RL environment contract."""

import copy
import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from future_prediction_bench.realworld import RealWorldEnv, RealWorldTaskRegistry, validate_task


BASE = datetime(2030, 1, 1, 10, tzinfo=timezone.utc)


class Clock:
    def __init__(self):
        self.value = BASE
        self.seconds = 0.0

    def now(self):
        return self.value

    def monotonic(self):
        return self.seconds

    def advance(self, seconds):
        self.value += timedelta(seconds=seconds)
        self.seconds += seconds


def task(**changes):
    value = {
        "schema_version": "realworld-0.1", "task_id": "fixture-code-1",
        "event_id": "fixture-project-1", "cluster_id": "fixture-project-1", "split": "train",
        "prompt": "Repair the fixture project and submit it for hidden verification.",
        "issued_at": BASE.isoformat(),
        "action_deadline": (BASE + timedelta(minutes=20)).isoformat(),
        "outcome_not_before": BASE.isoformat(), "verify_after": BASE.isoformat(),
        "tool_manifest": [{"name": "read_file", "description": "Read a fixture file."},
                          {"name": "submit", "description": "Finish the fixture task."}],
        "reward_contract": {"id": "fixture-hidden-check-v1", "description": "Pass hidden fixture checks.",
                            "min_reward": 0.0, "max_reward": 1.0},
        "budgets": {"max_actions": 4, "max_wall_seconds": 1200,
                    "verification_cooldown_seconds": 60, "max_verifications": 3},
        "is_fixture": True,
        "metadata": {"private_fixture_key": "do-not-expose"},
    }
    value.update(changes)
    return value


class FixtureAdapter:
    def __init__(self, clock, proposals=None):
        self.clock = clock
        self.proposals = list(proposals or [])
        self.submitted = False
        self.reset_calls = 0
        self.step_calls = 0
        self.verify_calls = 0

    def reset(self, spec, *, now):
        self.reset_calls += 1
        assert spec["is_fixture"] is True
        return {"workspace": "fixture", "visible_checks": 0}

    def step(self, action, *, now):
        self.step_calls += 1
        if action["action"] == "submit":
            self.submitted = True
            return {"observation": {"status": "submitted"}, "terminated": True}
        return {"observation": {"status": "ok", "file": "fixture content"}, "terminated": False}

    def verify(self, *, now):
        self.verify_calls += 1
        assert self.submitted
        return self.proposals.pop(0) if self.proposals else {
            "status": "resolved", "reward": 1.0,
            "evidence": {"verifier": "fixture-hidden-checks", "checks_sha256": "a" * 64},
            "available_at": now.isoformat(),
        }

    def get_state(self):
        return {"submitted": self.submitted, "verify_calls": self.verify_calls}


class RealWorldEnvTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.adapter = FixtureAdapter(self.clock)

    def env(self, spec=None, adapter=None):
        return RealWorldEnv(spec or task(), adapter or self.adapter,
                            clock=self.clock.now, monotonic_clock=self.clock.monotonic)

    def test_fast_feedback_submission_and_masked_audit(self):
        original = task()
        env = self.env(original)
        original["reward_contract"]["max_reward"] = 0
        start = env.reset("fixture-policy-1")
        self.assertEqual(start["status"], "active")
        self.assertNotIn("metadata", start["task"])
        self.assertNotIn("private_fixture_key", json.dumps(start))
        self.assertEqual(start["task"]["reward_contract"]["max_reward"], 1.0)
        detached = env.task
        detached["reward_contract"]["max_reward"] = 0
        self.assertEqual(env.task["reward_contract"]["max_reward"], 1.0)
        self.assertEqual(start["task"]["reward_contract_sha256"], env.task["reward_contract_sha256"])
        self.assertIsNone(env.step({"action": "read_file", "path": "main.py"})["reward"])
        submitted = env.step({"action": "submit"})
        self.assertTrue(submitted["terminated"])
        self.assertEqual(submitted["info"]["status"], "pending")
        self.assertIsNone(submitted["reward"])
        resolved = env.verify()
        self.assertEqual((resolved["status"], resolved["reward"]), ("graded", 1.0))
        self.assertEqual(env.verify(), resolved)
        self.assertEqual(self.adapter.verify_calls, 1)
        state = env.get_state()
        self.assertEqual(state["metrics"]["actions_used"], 2)
        self.assertEqual(state["metrics"]["verifications_used"], 1)
        self.assertEqual([event["loss_mask"] for event in state["events"]], [0, 1, 0, 1, 0, 0])
        self.assertEqual([event["visible_to_policy"] for event in state["events"]],
                         [True, True, True, True, True, False])
        for event in state["events"]:
            payload = {key: value for key, value in event.items() if key != "sha256"}
            encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
            self.assertEqual(event["sha256"], hashlib.sha256(encoded.encode()).hexdigest())
        trajectory = env.export_trajectory()
        self.assertFalse(trajectory["trainer_ready"])
        self.assertTrue(trajectory["is_fixture"])
        self.assertEqual(len(trajectory["events"]), 5)
        self.assertNotIn("fixture-hidden-checks", json.dumps(trajectory))

    def test_delayed_verification_pending_and_cooldown(self):
        delayed = task(verify_after=(BASE + timedelta(minutes=30)).isoformat())
        adapter = FixtureAdapter(self.clock, proposals=[{"status": "pending", "reason": "not published"}])
        env = self.env(delayed, adapter)
        env.reset("fixture-policy")
        env.step({"action": "submit"})
        self.assertEqual(env.verify()["reason"], "not_due")
        self.assertEqual(adapter.verify_calls, 0)
        self.clock.advance(30 * 60)
        self.assertEqual(env.verify()["reason"], "not published")
        self.assertEqual(adapter.verify_calls, 1)
        self.assertEqual(env.verify()["reason"], "cooldown")
        self.clock.advance(60)
        self.assertEqual(env.verify()["status"], "graded")
        self.assertEqual(env.get_state()["metrics"]["pending_verifications"], 1)

    def test_verifier_cannot_fabricate_reward_or_early_availability(self):
        bad = [{"status": "resolved", "reward": 2.0, "evidence": {"source": "fixture"},
                "available_at": BASE.isoformat()},
               {"status": "resolved", "reward": 1.0, "evidence": {"source": "fixture"},
                "available_at": (BASE + timedelta(days=1)).isoformat()}]
        adapter = FixtureAdapter(self.clock, proposals=bad)
        env = self.env(adapter=adapter)
        env.reset("fixture-policy")
        env.step({"action": "submit"})
        self.assertEqual(env.verify()["reason"], "verifier_error")
        self.assertIsNone(env.get_state()["reward"])
        self.clock.advance(60)
        self.assertEqual(env.verify()["reason"], "verifier_error")
        self.assertEqual(env.get_state()["status"], "pending")
        self.assertIsNone(env.export_trajectory())

    def test_void_is_not_failure_reward_or_training_data(self):
        adapter = FixtureAdapter(self.clock, proposals=[{
            "status": "void", "reason": "source unavailable", "evidence": {"source": "fixture"}}])
        env = self.env(adapter=adapter)
        env.reset("fixture-policy")
        env.step({"action": "submit"})
        self.assertEqual(env.verify()["status"], "void")
        self.assertEqual(env.verify()["reason"], "source unavailable")
        self.assertIsNone(env.get_state()["reward"])
        self.assertIsNone(env.export_trajectory())

    def test_adapter_setup_error_does_not_create_retryable_policy_episode(self):
        class BrokenAdapter(FixtureAdapter):
            def reset(self, spec, *, now):
                raise RuntimeError("fixture unavailable")

        env = self.env(adapter=BrokenAdapter(self.clock))
        with self.assertRaises(ValueError):
            env.reset("fixture-policy")
        self.assertEqual(env.get_state()["status"], "setup_error")
        self.assertIsNone(env.export_trajectory())
        with self.assertRaises(ValueError):
            env.reset("fixture-policy")

    def test_action_budget_and_late_result(self):
        limited = self.env(task(budgets={"max_actions": 1, "max_wall_seconds": 1200}))
        limited.reset("fixture-policy")
        result = limited.step({"action": "read_file", "path": "main.py"})
        self.assertEqual(result["info"]["status"], "missed")
        self.assertIsNone(limited.export_trajectory())
        with self.assertRaises(ValueError):
            limited.step({"action": "submit"})

        class SlowAdapter(FixtureAdapter):
            def step(self, action, *, now):
                result = super().step(action, now=now)
                self.clock.advance(21 * 60)
                return result

        slow = self.env(adapter=SlowAdapter(self.clock))
        slow.reset("fixture-policy")
        result = slow.step({"action": "read_file", "path": "main.py"})
        self.assertEqual(result["observation"]["reason"], "action_deadline_reached")
        self.assertNotIn("fixture content", json.dumps(result))
        self.assertEqual(slow.get_state()["status"], "missed")

    def test_manifest_split_and_chronology_validation(self):
        with self.assertRaises(ValueError):
            validate_task(task(tool_manifest=[{"name": "submit", "description": "One"},
                                              {"name": "submit", "description": "Two"}]))
        with self.assertRaises(ValueError):
            validate_task(task(verify_after=(BASE - timedelta(seconds=1)).isoformat()))
        with self.assertRaises(ValueError):
            validate_task(task(reward_contract={"id": "x", "description": "x",
                                                "min_reward": 0, "max_reward": float("nan")}))
        env = self.env(task(split="test"))
        env.reset("fixture-policy")
        env.step({"action": "submit"})
        self.assertEqual(env.verify()["status"], "graded")
        self.assertIsNone(env.export_trajectory())

    def test_persistent_registry_freezes_tasks_and_separates_related_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.sqlite"
            registry = RealWorldTaskRegistry(path)
            try:
                first = task()
                sealed = registry.register(first)
                self.assertEqual(registry.register(first), sealed)
                self.assertEqual(registry.get(first["task_id"])["task_sha256"], sealed)
                with self.assertRaises(ValueError):
                    registry.register(task(prompt="Altered after registration"))
                with self.assertRaises(ValueError):
                    registry.register(task(task_id="fixture-code-2", split="test"))
                with self.assertRaises(ValueError):
                    registry.register(task(task_id="fixture-code-3", event_id="another",
                                           split="dev"))
            finally:
                registry.close()
            reopened = RealWorldTaskRegistry(path)
            try:
                self.assertEqual(reopened.get(first["task_id"])["task_sha256"], sealed)
            finally:
                reopened.close()


if __name__ == "__main__":
    unittest.main()
