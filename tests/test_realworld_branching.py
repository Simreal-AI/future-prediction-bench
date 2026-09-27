"""Trusted filesystem-branch contracts; no Docker daemon is needed here."""

import hashlib
import unittest
from datetime import datetime, timedelta, timezone

from future_prediction_bench.realworld import RealWorldEnv


BASE = datetime(2026, 9, 24, 12, tzinfo=timezone.utc)
IMAGE = "sha256:" + "a" * 64


class Clock:
    def __init__(self):
        self.at = BASE
        self.mono = 100.0

    def now(self):
        return self.at

    def monotonic(self):
        return self.mono

    def advance(self, seconds):
        self.at += timedelta(seconds=seconds)
        self.mono += seconds


class BranchAdapter:
    snapshots = {}

    def __init__(self):
        self.state = None
        self.submitted = False
        self.closed = False

    def reset(self, task, *, now):
        self.state = "seed"
        return {"state": self.state}

    def step(self, action, *, now):
        if action["action"] == "set":
            self.state = action["value"]
            return {"observation": {"state": self.state}, "terminated": False}
        if action["action"] == "submit":
            self.submitted = True
            return {"observation": {"status": "submitted"}, "terminated": True}
        raise ValueError("Unsupported action")

    def verify(self, *, now):
        return {"status": "resolved", "reward": float(self.state == "good"),
                "evidence": {"state_sha256": self._sha(self.state)},
                "available_at": now.isoformat()}

    def get_state(self):
        return {"state": self.state, "submitted": self.submitted}

    @staticmethod
    def _sha(value):
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def create_branch_checkpoint(self):
        digest = self._sha(self.state)
        self.snapshots[digest] = self.state
        return {"snapshot_path": "/trusted/snapshots/" + digest,
                "workspace_sha256": digest,
                "artifact_binding": {"image_sha256": IMAGE},
                "image_sha256": IMAGE}

    def reset_from_checkpoint(self, task, checkpoint_ref, *, now):
        if checkpoint_ref["task_sha256"] != task["task_sha256"]:
            raise ValueError("Task binding changed")
        self.state = self.snapshots[checkpoint_ref["workspace_sha256"]]
        return {"state": self.state}

    def close(self):
        self.closed = True


def task(*, actions=5, seconds=60):
    return {"schema_version": "realworld-0.1", "task_id": "branch-fixture",
            "event_id": "branch-fixture", "cluster_id": "branch-fixture",
            "split": "train", "prompt": "Set the correct value.",
            "issued_at": (BASE - timedelta(minutes=1)).isoformat(),
            "action_deadline": (BASE + timedelta(minutes=5)).isoformat(),
            "outcome_not_before": (BASE - timedelta(minutes=1)).isoformat(),
            "verify_after": (BASE - timedelta(minutes=1)).isoformat(),
            "tool_manifest": [{"name": name, "description": name}
                              for name in ("set", "submit")],
            "reward_contract": {"id": "branch-fixture", "description": "Exact value",
                                "min_reward": 0, "max_reward": 1},
            "budgets": {"max_actions": actions, "max_wall_seconds": seconds},
            "is_fixture": True}


class RealWorldBranchingTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.parent = RealWorldEnv(task(), BranchAdapter(), clock=self.clock.now,
                                   monotonic_clock=self.clock.monotonic)
        self.parent.reset("revision-001")

    def test_forks_share_frozen_prefix_but_not_mutable_state_or_reward(self):
        self.parent.step({"action": "set", "value": "prefix"})
        prefix_events = self.parent.get_state()["events"]
        ref = self.parent.create_branch_checkpoint()
        self.parent.step({"action": "set", "value": "later-parent-state"})
        good = self.parent.fork_from_checkpoint(ref, BranchAdapter(), branch_id="good-1")
        bad = self.parent.fork_from_checkpoint(ref, BranchAdapter(), branch_id="bad-1")
        self.assertEqual(good.opening_observation()["observation"], {"state": "prefix"})
        self.assertEqual(bad.opening_observation()["observation"], {"state": "prefix"})
        self.assertEqual(good.events[:len(prefix_events)], prefix_events)
        self.assertEqual(good.actions_used, 1)
        self.assertEqual(good.policy_id, self.parent.policy_id)
        self.assertEqual(good.task["task_sha256"], self.parent.task["task_sha256"])
        self.assertNotEqual(good.episode_id, bad.episode_id)
        self.assertNotEqual(good.episode_id, self.parent.episode_id)
        self.assertFalse(good.events[len(prefix_events)]["visible_to_policy"])
        self.assertNotIn("branch_lineage", good.opening_observation())
        self.assertEqual(good.branch_lineage["prefix_event_sha256"], ref["prefix_event_sha256"])
        good.step({"action": "set", "value": "good"})
        bad.step({"action": "set", "value": "bad"})
        good.step({"action": "submit"})
        bad.step({"action": "submit"})
        self.assertEqual(good.verify()["reward"], 1.0)
        self.assertEqual(bad.verify()["reward"], 0.0)
        self.assertEqual(self.parent.adapter.state, "later-parent-state")
        self.assertEqual(good.export_trajectory()["branch_lineage"]["branch_id"], "good-1")

    def test_reference_tampering_and_foreign_parent_rejected(self):
        ref = self.parent.create_branch_checkpoint()
        changed = {**ref, "actions_used": ref["actions_used"] + 1}
        with self.assertRaises(ValueError):
            self.parent.fork_from_checkpoint(changed, BranchAdapter(), branch_id="changed")
        foreign = RealWorldEnv(task(), BranchAdapter(), clock=self.clock.now,
                               monotonic_clock=self.clock.monotonic)
        foreign.reset("revision-001")
        with self.assertRaises(ValueError):
            foreign.fork_from_checkpoint(ref, BranchAdapter(), branch_id="foreign")

    def test_prefix_actions_and_wall_time_are_not_reset(self):
        parent = RealWorldEnv(task(actions=2, seconds=10), BranchAdapter(), clock=self.clock.now,
                              monotonic_clock=self.clock.monotonic)
        parent.reset("revision-001")
        parent.step({"action": "set", "value": "prefix"})
        ref = parent.create_branch_checkpoint()
        self.clock.advance(8)
        child = parent.fork_from_checkpoint(ref, BranchAdapter(), branch_id="timed")
        self.assertEqual(child.step({"action": "set", "value": "good"})["info"]["status"], "missed")
        self.assertEqual(child.get_state()["metrics"]["actions_used"], 2)
        self.assertIsNone(child.export_trajectory())
        self.clock.advance(3)
        with self.assertRaises(ValueError):
            parent.fork_from_checkpoint(ref, BranchAdapter(), branch_id="too-late")

    def test_checkpoint_is_host_only_and_requires_active_episode(self):
        self.assertNotIn("create_branch_checkpoint", [tool["name"]
                          for tool in self.parent.opening_observation()["task"]["tool_manifest"]])
        self.parent.step({"action": "submit"})
        with self.assertRaises(ValueError):
            self.parent.create_branch_checkpoint()


if __name__ == "__main__":
    unittest.main()
