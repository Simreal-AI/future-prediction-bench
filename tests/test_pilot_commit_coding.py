"""Real verifier/control-plane logic with mocked Docker transport, not rollouts.

No Docker/QEMU, remote inference, repository speed or training claim is made.
The adapter itself still freezes submissions and checks host-only cases.
"""
import copy
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from future_prediction_bench.coding_env import DockerCodingAdapter
from future_prediction_bench.pilot_commit import PilotBudgetLedger, PilotReceipt
from future_prediction_bench.pilot_commit_coding import (
    CodingOutcomeProducer, CodingPolicyBinding, _case_reward, _digest)


class ScriptedUnitProvider:
    def __init__(self, revision, actions):
        self.policy_revision = revision
        self.actions = copy.deepcopy(actions)
        self.observations = []

    def next_action(self, observation, *, action_index, reservation_id):
        self.observations.append(copy.deepcopy(observation))
        return self.actions[action_index]


def unit_selector(data):
    result = {key: [] for key in ("keep", "too_correct", "too_incorrect", "exclude_too_easy")}
    for index in range(len(data.task_ids)):
        values = [reward for group, reward in zip(data.prompt_indices, data.rewards) if group == index]
        mean = sum(values) / len(values)
        key = "too_correct" if mean > 1 - data.thresholds.upper else "too_incorrect" if mean < data.thresholds.lower else "keep"
        result[key].append(index)
        if data.thresholds.exclude is not None and mean >= data.thresholds.exclude:
            result["exclude_too_easy"].append(index)
    return result


class CodingProducerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.seed, self.verifier = self.root / "seed", self.root / "verifier"
        self.seed.mkdir()
        self.verifier.mkdir()
        (self.seed / "math_utils.py").write_text("def add(a, b): return a-b\n")
        self.spec = {"kind": "command_cases_v1", "cases": [
            {"argv": ["python3", "-B", "-c", "from math_utils import add; print(add(2, 3))"],
             "expected_stdout": "5\n", "expected_returncode": 0},
            {"argv": ["python3", "-B", "-c", "from math_utils import add; print(add(-2, 2))"],
             "expected_stdout": "0\n", "expected_returncode": 0}]}
        (self.verifier / "verify.json").write_text(json.dumps(self.spec))
        now = datetime.now(timezone.utc)
        self.task = {"schema_version": "realworld-0.1", "task_id": "unit-add", "event_id": "unit-add",
            "cluster_id": "unit-add", "split": "train", "prompt": "Fix add.",
            "issued_at": (now - timedelta(minutes=1)).isoformat(),
            "action_deadline": (now + timedelta(minutes=30)).isoformat(),
            "outcome_not_before": (now - timedelta(minutes=1)).isoformat(),
            "verify_after": (now - timedelta(minutes=1)).isoformat(),
            "tool_manifest": [{"name": name, "description": name} for name in
                ("read_file", "write_file", "run_visible_checks", "submit")],
            "reward_contract": {"id": "unit-cases", "description": "Hidden host checks.",
                                "min_reward": 0, "max_reward": 1},
            "budgets": {"max_actions": 6, "max_wall_seconds": 600}, "is_fixture": True}
        self.ledger = PilotBudgetLedger(self.root / "budget.sqlite3", epoch_id="unit-epoch", total_budget=6)
        self.policy = CodingPolicyBinding(*(_digest(value) for value in ("unit-controller", "unit-prompt", "unit-config")))
        self.commands = []
        self.transport = patch.object(DockerCodingAdapter, "_docker", self.fake_docker)
        self.transport.start()
        self.addCleanup(self.transport.stop)
        self.producer = CodingOutcomeProducer(self.ledger, task=self.task, seed_dir=self.seed,
            verifier_dir=self.verifier, image="unit-image-not-real", output=self.root / "producer",
            policy_binding=self.policy, visible_check=("python3", "-m", "py_compile", "math_utils.py"))

    def fake_docker(self, arguments, *, timeout=None):
        self.commands.append(arguments)
        if arguments[:2] == ["image", "inspect"]:
            return subprocess.CompletedProcess(arguments, 0, b"sha256:" + b"a" * 64 + b"\n", b"")
        if arguments[0] == "inspect":
            return subprocess.CompletedProcess(arguments, 0, b"true\n", b"")
        if arguments[0] in ("run", "exec"):
            # Every simulated actor or grading command must already have a
            # genuine persistent claimed reservation; admission is read-only.
            with self.ledger._transaction() as db:
                claimed = db.execute("SELECT COUNT(*) FROM pilot_reservations WHERE epoch_id=? AND state='claimed'",
                                     (self.ledger.epoch_id,)).fetchone()[0]
            self.assertGreater(claimed, 0)
        if arguments[0] == "run" and "--detach" not in arguments:
            mount = arguments[arguments.index("--mount") + 1]
            workspace = Path(mount.split("src=", 1)[1].split(",dst=", 1)[0])
            corrected = "return a+b" in (workspace / "math_utils.py").read_text()
            first = "add(2, 3)" in " ".join(arguments)
            stdout = (b"5\n" if corrected else b"-1\n") if first else (b"0\n" if corrected else b"-4\n")
            return subprocess.CompletedProcess(arguments, 0, stdout, b"")
        return subprocess.CompletedProcess(arguments, 0, b"unit-container\n", b"")

    def provider(self, *, fixed=False):
        actions = ([{"action": "write_file", "path": "math_utils.py",
                     "content": "def add(a, b): return a+b\n"}] if fixed else []) + [{"action": "submit"}]
        return ScriptedUnitProvider(self.policy.revision, actions)

    def run_pilots(self):
        bootstrap = self.producer.reserve_pilots(request_id="initial")
        receipts = []
        for index, reservation in enumerate(bootstrap.reservations):
            result = self.producer.dispatch(reservation, owner_id="unit-worker", provider=self.provider(fixed=bool(index)))
            self.assertTrue(result.dispatched)
            receipts.append(result.receipt)
        return bootstrap, receipts

    def test_actual_adapter_case_outcomes_charge_then_drive_commit_reservations(self):
        bootstrap, pilots = self.run_pilots()
        self.assertEqual([item.reward for item in pilots], [0, 1])
        self.assertTrue(all(self.producer.verify_receipt(item) for item in pilots))
        self.assertEqual(self.ledger.snapshot(), {"total_budget": 6, "spent": 2, "reserved": 0, "remaining": 4})
        plan = self.producer.plan_commits(request_id="selected", pilots=pilots,
            selector_backend=unit_selector, selector_id="unit-selector", commit_cap=2)
        self.assertEqual(plan.selection.keep, (self.task["task_id"],))
        self.assertEqual(plan.spent_at_creation, 2)
        self.assertEqual(plan.reserved_by_plan, 2)
        self.assertEqual(self.ledger.snapshot(), {"total_budget": 6, "spent": 2, "reserved": 2, "remaining": 2})
        calls_before = len(self.commands)
        duplicate = self.producer.dispatch(bootstrap.reservations[0], owner_id="unit-worker", provider=self.provider())
        self.assertFalse(duplicate.dispatched)
        self.assertIsNone(duplicate.receipt)
        self.assertFalse(any(command[0] in ("run", "exec") for command in self.commands[calls_before:]))
        committed = self.producer.dispatch(plan.reservations[0], owner_id="commit-worker", provider=self.provider(fixed=True))
        self.assertEqual(committed.receipt.reward, 1)
        self.assertEqual(self.ledger.snapshot(), {"total_budget": 6, "spent": 3, "reserved": 1, "remaining": 2})

    def test_provider_only_receives_visible_data_and_cost_is_measured_separately(self):
        reservation = self.producer.reserve_pilots(request_id="initial").reservations[0]
        provider = self.provider(fixed=True)
        result = self.producer.dispatch(reservation, owner_id="worker", provider=provider)
        serialized = json.dumps(provider.observations)
        for hidden in ("expected_stdout", "verify.json", str(self.verifier), "case_results"):
            self.assertNotIn(hidden, serialized)
        record = self.producer._record(receipt_id=result.receipt.receipt_id)
        costs = record["execution"]["cost"]
        self.assertEqual(result.receipt.units, 1)
        self.assertEqual(costs["provider_calls"], 2)
        self.assertGreater(costs["wall_ns"], 0)
        self.assertGreater(costs["verification_ns"], 0)
        self.assertTrue(all(type(value) is int and value >= 0 for value in costs.values()))

    def test_model_claimed_pass_and_provider_exception_remain_unresolved_and_charged(self):
        bootstrap = self.producer.reserve_pilots(request_id="initial")
        claimed = ScriptedUnitProvider(self.policy.revision, [{"action": "submit", "passed": True, "reward": 1}])
        throwing = ScriptedUnitProvider(self.policy.revision, [])
        throwing.next_action = Mock(side_effect=RuntimeError("private-backend-sentinel-must-not-be-written"))
        receipts = []
        for reservation, provider in zip(bootstrap.reservations, (claimed, throwing)):
            result = self.producer.dispatch(reservation, owner_id="worker", provider=provider)
            receipt = result.receipt
            self.assertEqual(receipt.status, "unresolved")
            self.assertIsNone(receipt.reward)
            self.assertFalse(self.producer.verify_receipt(receipt))
            self.assertNotIn("private-backend-sentinel", json.dumps(self.producer._record(receipt_id=receipt.receipt_id)))
            receipts.append(receipt)
        self.assertEqual(self.ledger.snapshot()["spent"], 2)
        selector = Mock(side_effect=AssertionError("unresolved outcomes must not be selected"))
        plan = self.producer.plan_commits(request_id="none", pilots=receipts,
            selector_backend=selector, selector_id="unit-selector")
        self.assertEqual(plan.reserved_by_plan, 0)
        self.assertEqual(plan.spent_at_creation, 2)
        selector.assert_not_called()

    def test_changed_policy_task_and_verifier_reject_before_inference_or_actor_start(self):
        reservation = self.producer.reserve_pilots(request_id="initial").reservations[0]
        for field in ("policy", "task", "verifier"):
            provider = self.provider()
            if field == "policy":
                provider.policy_revision = "b" * 64
            elif field == "task":
                self.producer._task["prompt"] += " changed"
            else:
                (self.verifier / "verify.json").write_text("{}")
            before = len(self.commands)
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.producer.dispatch(reservation, owner_id="worker", provider=provider)
            self.assertEqual(provider.observations, [])
            self.assertFalse(any(command[0] in ("run", "exec") for command in self.commands[before:]))
            self.producer._task["prompt"] = self.task["prompt"]
            (self.verifier / "verify.json").write_text(json.dumps(self.spec))
        self.assertEqual(self.ledger.snapshot()["spent"], 0)

    def test_receipts_cannot_be_forged_or_replayed_under_other_bindings(self):
        _, pilots = self.run_pilots()
        real = pilots[1]
        for forged in (replace(real, reward=0), replace(real, policy_revision="b" * 64),
                       replace(real, task_revision="b" * 64), replace(real, verification_id="agent-claimed")):
            self.assertFalse(self.producer.verify_receipt(forged))
        unknown = replace(real, receipt_id="not-written-by-producer")
        with self.assertRaises(ValueError):
            self.producer.plan_commits(request_id="forged", pilots=(pilots[0], unknown),
                selector_backend=unit_selector, selector_id="unit-selector")

    def test_stored_outcome_without_completed_ledger_is_not_trusted_and_can_recover(self):
        reservation = self.producer.reserve_pilots(request_id="initial").reservations[0]
        with patch.object(self.ledger, "complete", side_effect=RuntimeError("simulated-ledger-write-failure")):
            with self.assertRaisesRegex(RuntimeError, "ledger-write-failure"):
                self.producer.dispatch(reservation, owner_id="worker", provider=self.provider(fixed=True))
        record = self.producer._record(reservation_id=reservation.reservation_id)
        receipt = PilotReceipt(**record["receipt"])
        self.assertFalse(self.producer.verify_receipt(receipt))
        before = len(self.commands)
        self.assertTrue(self.producer.recover_completion(reservation, owner_id="worker"))
        self.assertFalse(self.producer.recover_completion(reservation, owner_id="worker"))
        self.assertEqual(len(self.commands), before)
        self.assertTrue(self.producer.verify_receipt(receipt))
        self.assertEqual(self.ledger.snapshot()["spent"], 1)

    def test_corrupt_outcome_digest_and_budget_overflow_fail_closed(self):
        _, pilots = self.run_pilots()
        with self.ledger._transaction() as db:
            db.execute("UPDATE coding_pilot_outcomes SET outcome_sha256=? WHERE receipt_id=?",
                       ("0" * 64, pilots[0].receipt_id))
        self.assertFalse(self.producer.verify_receipt(pilots[0]))
        with self.assertRaisesRegex(ValueError, "remaining_budget"):
            self.producer.reserve_pilots(request_id="overflow", count=5)

    def test_artifact_change_after_pilots_rejects_before_selection_or_reservation(self):
        _, pilots = self.run_pilots()
        original = (self.verifier / "verify.json").read_bytes()
        snapshot = self.ledger.snapshot()
        (self.verifier / "verify.json").write_bytes(original + b"\n")
        selector = Mock(side_effect=AssertionError("stale artifact must not reach selector"))
        with self.assertRaisesRegex(ValueError, "artifact_changed"):
            self.producer.plan_commits(request_id="stale", pilots=pilots,
                selector_backend=selector, selector_id="unit-selector", commit_cap=2)
        selector.assert_not_called()
        self.assertEqual(self.ledger.snapshot(), snapshot)
        (self.verifier / "verify.json").write_bytes(original)
        plan = self.producer.plan_commits(request_id="unchanged", pilots=pilots,
            selector_backend=unit_selector, selector_id="unit-selector", commit_cap=2)
        self.assertEqual(plan.reserved_by_plan, 2)
        self.assertEqual(plan.spent_at_creation, 2)


class CaseEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.spec = {"cases": [{"expected_stdout": "yes\n", "expected_returncode": 0}]}
        self.value = {"status": "graded", "reward": 1, "evidence": {
            "kind": "host_checked_command_cases_v1", "verifier_sha256": "a" * 64,
            "image_sha256": "sha256:" + "b" * 64, "case_results": [{"return_code": 0,
                "stdout_sha256": hashlib.sha256(b"yes\n").hexdigest(), "passed": True}]}}

    def derive(self, value):
        return _case_reward(value, specification=self.spec, verifier_sha256="a" * 64,
                            image_sha256="sha256:" + "b" * 64)

    def test_case_pass_flags_cannot_override_actual_stdout_or_exit_code(self):
        self.assertEqual(self.derive(self.value), 1)
        for alteration in ({"stdout_sha256": "c" * 64}, {"return_code": 1}, {"passed": 1},
                           {"return_code": None}, {"return_code": 125}):
            changed = copy.deepcopy(self.value)
            changed["evidence"]["case_results"][0].update(alteration)
            with self.subTest(alteration=alteration), self.assertRaises(ValueError):
                self.derive(changed)

    def test_missing_cases_wrong_bindings_or_boolean_reward_are_rejected(self):
        for alteration in ([], self.value["evidence"]["case_results"] * 2):
            changed = copy.deepcopy(self.value)
            changed["evidence"]["case_results"] = alteration
            with self.assertRaises(ValueError):
                self.derive(changed)
        changed = copy.deepcopy(self.value)
        changed["reward"] = True
        with self.assertRaises(ValueError):
            self.derive(changed)
        changed = copy.deepcopy(self.value)
        changed["evidence"]["verifier_sha256"] = "c" * 64
        with self.assertRaises(ValueError):
            self.derive(changed)


if __name__ == "__main__":
    unittest.main()
