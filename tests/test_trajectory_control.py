"""Control-plane invariants with the actual RealWorldEnv state machine."""

from __future__ import annotations

import threading
import time
import unittest
from datetime import datetime, timedelta, timezone

from future_prediction_bench.realworld import RealWorldEnv
from future_prediction_bench.trajectory_control import (
    EpisodeLease, TrajectoryControlPlane, TrajectoryJob,
)


def _task(index):
    now = datetime.now(timezone.utc)
    return {"schema_version": "realworld-0.1", "task_id": f"control-{index}",
            "event_id": f"control-{index}", "cluster_id": f"control-{index}",
            "split": "train", "prompt": "Read and submit the task.",
            "issued_at": (now - timedelta(minutes=1)).isoformat(),
            "action_deadline": (now + timedelta(minutes=5)).isoformat(),
            "outcome_not_before": (now - timedelta(minutes=1)).isoformat(),
            "verify_after": (now - timedelta(minutes=1)).isoformat(),
            "tool_manifest": [{"name": "read_file", "description": "Read."},
                              {"name": "submit", "description": "Submit."}],
            "reward_contract": {"id": "test", "description": "Test.",
                                "min_reward": 0, "max_reward": 1},
            "budgets": {"max_actions": 3, "max_wall_seconds": 30},
            "is_fixture": True}


class _Adapter:
    def __init__(self, index, recorder, *, pending_once=False):
        self.index = index
        self.recorder = recorder
        self.pending_once = pending_once
        self.verify_count = 0
        self.submitted = False

    def reset(self, task, *, now):
        self.recorder.append((self.index, "reset"))
        return {"status": "ready"}

    def step(self, action, *, now):
        self.recorder.append((self.index, action["action"]))
        if action["action"] == "submit":
            self.submitted = True
            return {"observation": {"status": "submitted"}, "terminated": True}
        return {"observation": {"status": "read", "content": "public"}, "terminated": False}

    def verify(self, *, now):
        self.verify_count += 1
        self.recorder.append((self.index, "verify"))
        if self.pending_once and self.verify_count == 1:
            return {"status": "pending", "reason": "external_feedback_pending"}
        return {"status": "resolved", "reward": float(self.index % 2),
                "evidence": {"kind": "trusted-test"},
                "available_at": datetime.now(timezone.utc).isoformat()}

    def get_state(self):
        return {"submitted": self.submitted}

    def close(self):
        self.recorder.append((self.index, "close"))


def _job(index, recorder, *, pending_once=False, on_generate=None):
    def open_episode():
        adapter = _Adapter(index, recorder, pending_once=pending_once)
        return EpisodeLease(RealWorldEnv(_task(index), adapter), adapter.close)

    def generate(observation, history, revision):
        assert not any("verification" in item for item in history)
        if len(history) == 2:
            assert history[1]["action"] == {"action": "read_file"}
            assert history[1]["transition"]["observation"]["status"] == "read"
        if on_generate:
            on_generate(index, len(history), revision)
        return {"action": "read_file" if len(history) == 1 else "submit"}

    return TrajectoryJob(f"job-{index}", "test-policy", open_episode, generate)


class TrajectoryControlTests(unittest.TestCase):
    def test_multiturn_bounded_pipeline_reward_and_freshness(self):
        recorder = []
        jobs = [_job(index, recorder) for index in range(4)]
        manager = TrajectoryControlPlane(jobs, revision_provider=lambda: "revision-1",
                                         actor_workers=2, environment_workers=2,
                                         reward_workers=1, actor_queue_capacity=1,
                                         environment_queue_capacity=1,
                                         reward_queue_capacity=1, max_in_flight=3)
        report = manager.run()
        self.assertEqual(report["graded_count"], 4)
        self.assertEqual(report["pending_count"], 0)
        self.assertEqual(report["infrastructure_error_count"], 0)
        self.assertEqual(report["fresh_graded_count"], 4)
        self.assertEqual([item["reward"] for item in report["episodes"]], [0, 1, 0, 1])
        self.assertTrue(all(len(item["actions"]) == 2 for item in report["episodes"]))
        self.assertTrue(all(len(item["phase_timestamps"]) == 6 for item in report["episodes"]))
        self.assertTrue(all(value <= 1 for value in report["queue_high_water"].values()))
        self.assertEqual(sum(name == "close" for _, name in recorder), 4)
        self.assertFalse(any(item["trainer_ready"] for item in report["episodes"]))
        self.assertEqual(set(manager.text_trajectories()), {f"job-{index}" for index in range(4)})
        self.assertTrue(all(item["text_trajectory_sha256"] for item in report["episodes"]))
        manager.close()

    def test_pending_verifier_is_resumable_and_never_reward_zero(self):
        recorder = []
        manager = TrajectoryControlPlane(
            [_job(1, recorder, pending_once=True)],
            revision_provider=lambda: "rev-a")
        first = manager.run()
        self.assertEqual(first["episodes"][0]["status"], "pending")
        self.assertIsNone(first["episodes"][0]["reward"])
        self.assertEqual(first["pending_count"], 1)
        self.assertNotIn((1, "close"), recorder)
        self.assertEqual(manager.text_trajectories(), {})
        second = manager.resume_pending()
        self.assertEqual(second["episodes"][0]["status"], "graded")
        self.assertEqual(second["episodes"][0]["reward"], 1)
        self.assertEqual(second["selected_graded_count"], 1)
        self.assertEqual(sum(name == "reset" for _, name in recorder), 1)
        self.assertEqual(sum(name == "verify" for _, name in recorder), 2)
        self.assertEqual(sum(name == "close" for _, name in recorder), 1)
        manager.close()

    def test_mixed_revision_is_rejected_and_later_update_rechecks(self):
        recorder = []
        current = {"value": "rev-a"}

        def update_on_first_action(index, turn, revision):
            if turn == 1:
                current["value"] = "rev-b"

        mixed = TrajectoryControlPlane(
            [_job(0, recorder, on_generate=update_on_first_action)],
            revision_provider=lambda: current["value"])
        report = mixed.run()
        self.assertEqual(report["episodes"][0]["status"], "graded")
        self.assertEqual([item["policy_revision"] for item in report["episodes"][0]["actions"]],
                         ["rev-a", "rev-b"])
        self.assertEqual(report["episodes"][0]["freshness_gate"], "mixed_policy_revisions")
        mixed.close()

        current["value"] = "rev-a"
        stable = TrajectoryControlPlane([_job(1, recorder)],
                                        revision_provider=lambda: current["value"])
        self.assertEqual(stable.run()["episodes"][0]["freshness_gate"], "current_revision")
        current["value"] = "rev-b"
        self.assertEqual(stable.report()["episodes"][0]["freshness_gate"], "stale_revision")
        stable.close()

    def test_opt_in_fence_regenerates_stale_first_action_without_guest_step(self):
        recorder = []
        current = {"value": "rev-a"}
        generations = []

        def rotate_on_first_generation(index, turn, revision):
            generations.append(revision)
            if turn == 1 and revision == "rev-a":
                current["value"] = "rev-b"

        manager = TrajectoryControlPlane(
            [_job(1, recorder, on_generate=rotate_on_first_generation)],
            revision_provider=lambda: current["value"],
            stale_revision_policy="fence", max_stale_regenerations=1)
        report = manager.run()
        episode = report["episodes"][0]
        self.assertEqual(episode["status"], "graded")
        self.assertEqual(episode["reward"], 1.0)
        self.assertEqual(episode["freshness_gate"], "current_revision")
        self.assertEqual(episode["stale_action_rejections"], 1)
        self.assertEqual(generations, ["rev-a", "rev-b", "rev-b"])
        self.assertEqual([action["policy_revision"] for action in episode["actions"]],
                         ["rev-b", "rev-b"])
        self.assertEqual(sum(name == "read_file" for _, name in recorder), 1)
        self.assertEqual(sum(name == "verify" for _, name in recorder), 1)
        manager.close()

    def test_opt_in_fence_checks_again_at_environment_dispatch(self):
        recorder = []
        current = {"value": "rev-a"}
        flipped = {"value": False}

        def revision():
            if (threading.current_thread().name.startswith("trajectory-environment-")
                    and not flipped["value"]):
                current["value"] = "rev-b"
                flipped["value"] = True
            return current["value"]

        manager = TrajectoryControlPlane(
            [_job(1, recorder)], revision_provider=revision,
            stale_revision_policy="fence", max_stale_regenerations=1)
        report = manager.run()
        episode = report["episodes"][0]
        self.assertEqual(episode["status"], "graded")
        self.assertEqual(episode["reward"], 1.0)
        self.assertEqual(episode["stale_action_rejections"], 1)
        self.assertEqual([item["policy_revision"] for item in episode["actions"]],
                         ["rev-b", "rev-b"])
        self.assertEqual(sum(name == "read_file" for _, name in recorder), 1)
        self.assertEqual(sum(name == "submit" for _, name in recorder), 1)
        manager.close()

    def test_opt_in_fence_stops_mid_episode_revision_change_ungraded(self):
        recorder = []
        current = {"value": "rev-a"}

        def rotate_after_first_step(index, turn, revision):
            if turn == 2:
                current["value"] = "rev-b"

        manager = TrajectoryControlPlane(
            [_job(1, recorder, on_generate=rotate_after_first_step)],
            revision_provider=lambda: current["value"],
            stale_revision_policy="fence", max_stale_regenerations=2)
        report = manager.run()
        episode = report["episodes"][0]
        self.assertEqual(episode["status"], "stale_policy_revision")
        self.assertIsNone(episode["reward"])
        self.assertEqual(episode["stale_action_rejections"], 1)
        self.assertEqual(len(episode["actions"]), 1)
        self.assertEqual(sum(name == "submit" for _, name in recorder), 0)
        self.assertEqual(sum(name == "verify" for _, name in recorder), 0)
        self.assertEqual(sum(name == "close" for _, name in recorder), 1)
        self.assertEqual(manager.text_trajectories(), {})
        manager.close()

    def test_opt_in_fence_budget_exhaustion_keeps_reward_none(self):
        recorder = []
        current = {"value": "rev-a"}

        def flip(index, turn, revision):
            current["value"] = "rev-b"

        manager = TrajectoryControlPlane(
            [_job(1, recorder, on_generate=flip)],
            revision_provider=lambda: current["value"],
            stale_revision_policy="fence", max_stale_regenerations=0)
        episode = manager.run()["episodes"][0]
        self.assertEqual(episode["status"], "stale_policy_revision")
        self.assertIsNone(episode["reward"])
        self.assertEqual(episode["actions"], [])
        self.assertEqual(episode["stale_action_rejections"], 1)
        self.assertNotIn((1, "read_file"), recorder)
        manager.close()

    def test_stable_revision_fence_preserves_reward_and_evidence(self):
        records = []
        outputs = []
        for policy in ("report_only", "fence"):
            manager = TrajectoryControlPlane(
                [_job(1, records)], revision_provider=lambda: "stable-revision",
                stale_revision_policy=policy)
            outputs.append(manager.run()["episodes"][0])
            manager.close()
        first, second = outputs
        for key in ("status", "reward", "verification_evidence_sha256",
                    "freshness_gate"):
            self.assertEqual(first[key], second[key])
        self.assertEqual([a["action_sha256"] for a in first["actions"]],
                         [a["action_sha256"] for a in second["actions"]])
        self.assertEqual(second["stale_action_rejections"], 0)

    def test_opt_in_fence_skips_verifier_after_revision_change(self):
        recorder = []
        current = {"value": "rev-a"}

        def rotate_before_verify(job_id):
            current["value"] = "rev-b"

        manager = TrajectoryControlPlane(
            [_job(1, recorder)], revision_provider=lambda: current["value"],
            stale_revision_policy="fence", before_verify=rotate_before_verify)
        episode = manager.run()["episodes"][0]
        self.assertEqual(episode["status"], "stale_policy_revision")
        self.assertIsNone(episode["reward"])
        self.assertEqual(episode["verification_status"], None)
        self.assertEqual(sum(name == "verify" for _, name in recorder), 0)
        self.assertEqual(sum(name == "close" for _, name in recorder), 1)
        manager.close()

    def test_opt_in_fence_discards_reward_if_revision_changes_during_verification(self):
        recorder = []
        current = {"value": "rev-a"}

        class RotatingVerifier(_Adapter):
            def verify(self, *, now):
                result = super().verify(now=now)
                current["value"] = "rev-b"
                return result

        def open_episode():
            adapter = RotatingVerifier(1, recorder)
            return EpisodeLease(RealWorldEnv(_task(1), adapter), adapter.close)

        scripted = _job(1, recorder)
        manager = TrajectoryControlPlane(
            [TrajectoryJob(scripted.job_id, scripted.policy_id,
                           open_episode, scripted.generate_action)],
            revision_provider=lambda: current["value"],
            stale_revision_policy="fence")
        episode = manager.run()["episodes"][0]
        self.assertEqual(episode["status"], "stale_policy_revision")
        self.assertIsNone(episode["reward"])
        self.assertEqual(episode["verification_status"], "graded")
        self.assertEqual(sum(name == "verify" for _, name in recorder), 1)
        self.assertEqual(manager.text_trajectories(), {})
        self.assertEqual(sum(name == "close" for _, name in recorder), 1)
        manager.close()

    def test_opt_in_fence_excludes_stale_graded_audit_from_export(self):
        recorder = []
        current = {"value": "rev-a"}
        manager = TrajectoryControlPlane(
            [_job(1, recorder)], revision_provider=lambda: current["value"],
            stale_revision_policy="fence")
        report = manager.run()
        self.assertEqual(report["episodes"][0]["status"], "graded")
        self.assertEqual(set(manager.text_trajectories()), {"job-1"})
        current["value"] = "rev-b"
        self.assertEqual(manager.report()["episodes"][0]["freshness_gate"],
                         "stale_revision")
        self.assertEqual(manager.text_trajectories(), {})
        manager.close()

    def test_verifier_exception_is_infrastructure_error_without_reward(self):
        recorder = []
        manager = TrajectoryControlPlane(
            [_job(1, recorder)], revision_provider=lambda: "rev-a",
            before_verify=lambda _: (_ for _ in ()).throw(ConnectionError("verifier lost")))
        report = manager.run()
        self.assertEqual(report["episodes"][0]["status"], "infrastructure_error")
        self.assertIsNone(report["episodes"][0]["reward"])
        self.assertEqual(report["infrastructure_error_count"], 1)
        self.assertIn((1, "close"), recorder)
        manager.close()

    def test_reward_straggler_does_not_block_next_actor(self):
        recorder = []
        verifier_started = threading.Event()
        release = threading.Event()
        overlapped = threading.Event()

        def before_verify(job_id):
            if job_id == "job-0":
                verifier_started.set()
                release.wait(timeout=3)

        def on_generate(index, turn, revision):
            if index == 1 and verifier_started.is_set() and not release.is_set():
                overlapped.set()

        first, second = _job(0, recorder), _job(1, recorder, on_generate=on_generate)

        def delayed_open():
            verifier_started.wait(timeout=3)
            return second.open_episode()

        jobs = [first, TrajectoryJob(second.job_id, second.policy_id, delayed_open,
                                     second.generate_action)]
        manager = TrajectoryControlPlane(jobs, revision_provider=lambda: "rev-a",
                                         actor_workers=1, environment_workers=2,
                                         reward_workers=1, max_in_flight=2,
                                         before_verify=before_verify)
        result = {}
        thread = threading.Thread(target=lambda: result.setdefault("report", manager.run()), daemon=True)
        thread.start()
        try:
            self.assertTrue(verifier_started.wait(timeout=3))
            self.assertTrue(overlapped.wait(timeout=3))
        finally:
            release.set()
            thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result["report"]["graded_count"], 2)
        manager.close()

    def test_invalid_settings_fail_closed(self):
        recorder = []
        with self.assertRaisesRegex(ValueError, "actor_workers"):
            TrajectoryControlPlane([_job(0, recorder)], revision_provider=lambda: "x",
                                   actor_workers=True)
        with self.assertRaisesRegex(ValueError, "unique"):
            TrajectoryControlPlane([_job(0, recorder), _job(0, recorder)],
                                   revision_provider=lambda: "x")
        with self.assertRaisesRegex(ValueError, "verification_wait_placement"):
            TrajectoryControlPlane([_job(0, recorder)], revision_provider=lambda: "x",
                                   verification_wait_placement="unknown")
        with self.assertRaisesRegex(ValueError, "stale_revision_policy"):
            TrajectoryControlPlane([_job(0, recorder)], revision_provider=lambda: "x",
                                   stale_revision_policy="unknown")
        with self.assertRaisesRegex(ValueError, "max_stale_regenerations"):
            TrajectoryControlPlane([_job(0, recorder)], revision_provider=lambda: "x",
                                   max_stale_regenerations=True)

    def test_short_verification_wait_does_not_occupy_reward_worker(self):
        def run(placement):
            recorder = []
            first_submitted = threading.Event()

            class SubmissionAdapter(_Adapter):
                def step(self, action, *, now):
                    result = super().step(action, now=now)
                    if self.index == 0 and action["action"] == "submit":
                        first_submitted.set()
                    return result

            def make_job(index):
                def open_episode():
                    adapter = SubmissionAdapter(index, recorder)
                    return EpisodeLease(RealWorldEnv(_task(index), adapter), adapter.close)

                def generate(observation, history, revision):
                    if index and not first_submitted.wait(timeout=2):
                        raise TimeoutError("first episode did not submit")
                    return {"action": "submit"}

                return TrajectoryJob(f"job-{index}", "test-policy", open_episode, generate)

            def before_verify(job_id):
                recorder.append((int(job_id.split("-")[1]), "verify_started"))
                time.sleep(0.045)

            manager = TrajectoryControlPlane(
                [make_job(index) for index in range(3)],
                revision_provider=lambda: "rev-a", actor_workers=3,
                environment_workers=1, reward_workers=1, max_in_flight=3,
                before_verify=before_verify,
                verification_delay_seconds=lambda job_id: 0.3 if job_id == "job-0" else 0,
                verification_wait_placement=placement)
            try:
                report = manager.run()
            finally:
                manager.close()
            self.assertEqual(report["graded_count"], 3)
            self.assertEqual(report["fresh_graded_count"], 3)
            self.assertEqual([episode["reward"] for episode in report["episodes"]], [0, 1, 0])
            self.assertEqual([episode["verification_wait_seconds"] for episode in report["episodes"]],
                             [0.3, 0.0, 0.0])
            self.assertEqual(sum(name == "close" for _, name in recorder), 3)
            return report, [index for index, name in recorder if name == "verify_started"]

        worker, worker_order = run("worker")
        scheduler, scheduler_order = run("scheduler")
        self.assertEqual(worker_order[0], 0)
        self.assertEqual(scheduler_order[-1], 0)
        self.assertLess(scheduler["wall_seconds"], worker["wall_seconds"])
        for left, right in zip(worker["episodes"], scheduler["episodes"]):
            self.assertEqual(left["verification_evidence_sha256"],
                             right["verification_evidence_sha256"])
            self.assertEqual([action["action_sha256"] for action in left["actions"]],
                             [action["action_sha256"] for action in right["actions"]])

    def test_invalid_readiness_delay_fails_closed(self):
        recorder = []
        manager = TrajectoryControlPlane(
            [_job(0, recorder)], revision_provider=lambda: "rev-a",
            verification_delay_seconds=lambda _: float("nan"))
        with self.assertRaisesRegex(ValueError, "finite seconds"):
            manager.run()
        self.assertIn((0, "close"), recorder)
        self.assertFalse(any(ctx.reward is not None for ctx in manager.contexts))
        manager.close()

    def test_dead_actor_with_bounded_queue_cannot_hang_shutdown(self):
        recorder = []
        first = _job(0, recorder)
        def crash_after_queue_fills(observation, history, revision):
            time.sleep(0.15)  # Let the next actor request fill its one-slot queue.
            raise SystemExit("injected actor death")
        first = TrajectoryJob(first.job_id, first.policy_id,
                              first.open_episode, crash_after_queue_fills)
        manager = TrajectoryControlPlane(
            [first, _job(1, recorder), _job(2, recorder)],
            revision_provider=lambda: "rev-a", actor_workers=1,
            environment_workers=1, reward_workers=1,
            actor_queue_capacity=1, environment_queue_capacity=1,
            reward_queue_capacity=1, max_in_flight=3)
        result = {}
        def run():
            try:
                manager.run()
            except BaseException as exc:
                result["error"] = exc
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        thread.join(timeout=3)
        self.assertFalse(thread.is_alive(), "worker death deadlocked queue cleanup")
        self.assertIsInstance(result.get("error"), RuntimeError)
        self.assertFalse(any(ctx.reward is not None for ctx in manager.contexts))
        self.assertTrue(all(ctx.closed for ctx in manager.contexts if ctx.lease is not None))
        manager.close()

    def test_worker_death_defers_close_until_active_verifier_returns(self):
        recorder = []
        verifier_started = threading.Event()
        verifier_release = threading.Event()
        verifier_active = threading.Event()
        closed = threading.Event()
        concurrent_close = threading.Event()

        class BlockingAdapter(_Adapter):
            def verify(self, *, now):
                verifier_active.set()
                verifier_started.set()
                try:
                    if not verifier_release.wait(timeout=5):
                        raise TimeoutError("test verifier was not released")
                    return super().verify(now=now)
                finally:
                    verifier_active.clear()

            def close(self):
                if verifier_active.is_set():
                    concurrent_close.set()
                super().close()
                closed.set()

        def open_blocking():
            adapter = BlockingAdapter(0, recorder)
            return EpisodeLease(RealWorldEnv(_task(0), adapter), adapter.close)

        ordinary = _job(0, recorder)
        first = TrajectoryJob(ordinary.job_id, ordinary.policy_id,
                              open_blocking, ordinary.generate_action)

        def open_after_verifier():
            if not verifier_started.wait(timeout=5):
                raise TimeoutError("verifier never started")
            return _job(1, recorder).open_episode()

        def die_in_actor(observation, history, revision):
            raise SystemExit("injected actor death")

        second = TrajectoryJob("job-1", "test-policy", open_after_verifier,
                               die_in_actor)
        manager = TrajectoryControlPlane(
            [first, second], revision_provider=lambda: "rev-a",
            actor_workers=1, environment_workers=2, reward_workers=1,
            max_in_flight=2)
        result = {}

        def run():
            try:
                manager.run()
            except BaseException as exc:
                result["error"] = exc

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        try:
            self.assertTrue(verifier_started.wait(timeout=5))
            thread.join(timeout=3)
            self.assertFalse(thread.is_alive(), "shutdown waited on active verifier")
            self.assertIsInstance(result.get("error"), RuntimeError)
            self.assertFalse(closed.is_set(), "lease closed while verifier was active")
            self.assertFalse(concurrent_close.is_set())
        finally:
            verifier_release.set()
            thread.join(timeout=5)
        self.assertTrue(closed.wait(timeout=3), "deferred lease cleanup did not run")
        self.assertFalse(concurrent_close.is_set())
        self.assertEqual(recorder.count((0, "close")), 1)
        manager.close()
        self.assertEqual(recorder.count((0, "close")), 1)

    def test_worker_death_closes_lease_returned_by_late_open(self):
        recorder = []
        open_started = threading.Event()
        open_release = threading.Event()
        late_closed = threading.Event()

        def late_open():
            open_started.set()
            if not open_release.wait(timeout=5):
                raise TimeoutError("test opener was not released")
            adapter = _Adapter(1, recorder)
            def close():
                adapter.close()
                late_closed.set()
            return EpisodeLease(RealWorldEnv(_task(1), adapter), close)

        def die_in_actor(observation, history, revision):
            if not open_started.wait(timeout=5):
                raise TimeoutError("late opener never started")
            raise SystemExit("injected actor death")

        first = _job(0, recorder)
        first = TrajectoryJob(first.job_id, first.policy_id,
                              first.open_episode, die_in_actor)
        second = TrajectoryJob("job-1", "test-policy", late_open,
                               lambda observation, history, revision: None)
        manager = TrajectoryControlPlane(
            [first, second], revision_provider=lambda: "rev-a",
            actor_workers=1, environment_workers=2, reward_workers=1,
            max_in_flight=2)
        result = {}

        def run():
            try:
                manager.run()
            except BaseException as exc:
                result["error"] = exc

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        try:
            self.assertTrue(open_started.wait(timeout=5))
            thread.join(timeout=3)
            self.assertFalse(thread.is_alive(), "shutdown waited on late opener")
            self.assertIsInstance(result.get("error"), RuntimeError)
            self.assertNotIn((1, "close"), recorder)
        finally:
            open_release.set()
            thread.join(timeout=5)
        self.assertTrue(late_closed.wait(timeout=3), "late-returned lease leaked")
        self.assertNotIn((1, "reset"), recorder)
        self.assertEqual(recorder.count((1, "close")), 1)
        manager.close()
        self.assertEqual(recorder.count((1, "close")), 1)


if __name__ == "__main__":
    unittest.main()
