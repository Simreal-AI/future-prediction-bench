import dataclasses
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from examples.official_pilot_commit import check_components as probe
from future_prediction_bench.pilot_commit import (
    PilotBudgetLedger, PilotReceipt, SelectionThresholds, TaskRevision,
    plan_pilot_commit,
)


def strict_selector(data):
    """Owned backend fixture; original upstream execution is separate."""
    result = {key: [] for key in ("keep", "too_correct", "too_incorrect", "exclude_too_easy")}
    for index in range(len(data.task_ids)):
        rewards = [reward for group, reward in zip(data.prompt_indices, data.rewards) if group == index]
        mean = sum(rewards)/len(rewards)
        category = "too_correct" if mean > 1-data.thresholds.upper else "too_incorrect" if mean < data.thresholds.lower else "keep"
        result[category].append(index)
        if data.thresholds.exclude is not None and mean >= data.thresholds.exclude:
            result["exclude_too_easy"].append(index)
    return result


class PilotCommitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)/"budget.sqlite3"
        self.ledger = PilotBudgetLedger(self.path, epoch_id="epoch-7", total_budget=16)

    def tearDown(self):
        self.tmp.cleanup()

    def pilots(self, outcomes=None, *, policy="policy-7", revision="task-v1"):
        outcomes = outcomes or {"a": [0,0], "b": [0,1], "c": [1,0], "d": [1,1]}
        return tuple(PilotReceipt(f"{task}-{index}", task, revision, policy, "resolved_binary", reward=reward,
                                  verification_id=f"verifier:{task}-{index}")
                     for task, rewards in outcomes.items() for index, reward in enumerate(rewards))

    def plan(self, *, ledger=None, pilots=None, tasks=None, **overrides):
        params = {"request_id": "plan-1", "current_policy_revision": "policy-7",
                  "tasks": tasks if tasks is not None else tuple(TaskRevision(key,"task-v1") for key in ("a","b","c","d")),
                  "pilots": pilots if pilots is not None else self.pilots(),
                  "selector_backend": strict_selector,
                  "evidence_verifier": lambda receipt: receipt.verification_id == f"verifier:{receipt.receipt_id}",
                  "selector_id": "owned-selector-fixture", "evidence_verifier_id": "owned-verifier-v1"}
        params.update(overrides)
        return plan_pilot_commit(ledger or self.ledger, **params)

    def test_real_costs_and_exact_selected_task_ids(self):
        result = self.plan()
        self.assertEqual(result.selection.keep, ("b","c"))
        self.assertEqual(result.selection.too_correct, ("d",))
        self.assertEqual(result.selection.too_incorrect, ("a",))
        self.assertEqual(result.selection.exclude_too_easy, ("d",))
        self.assertEqual([(item.task_id,item.count,item.evidence_ids) for item in result.allocations],
                         [("b",4,("b-0","b-1")),("c",4,("c-0","c-1"))])
        self.assertEqual(self.ledger.snapshot(), {"total_budget":16,"spent":8,"reserved":8,"remaining":0})
        self.assertEqual(len({item.reservation_id for item in result.reservations}),8)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.request_id = "modified"

    def test_failed_stale_changed_excluded_untrusted_and_unknown_pilots_all_cost(self):
        extras = (
            PilotReceipt("failed","a","task-v1","policy-7","failed"),
            PilotReceipt("stale","a","task-v1","policy-6","resolved_binary",reward=1,verification_id="verifier:stale"),
            PilotReceipt("changed","a","task-v0","policy-7","resolved_binary",reward=1,verification_id="verifier:changed"),
            PilotReceipt("unresolved","a","task-v1","policy-7","unresolved"),
            PilotReceipt("excluded","a","task-v1","policy-7","excluded"),
            PilotReceipt("agent-claimed","a","task-v1","policy-7","resolved_binary",reward=1,verification_id="agent"),
            PilotReceipt("unknown","other","task-v1","policy-7","failed"),
        )
        result = self.plan(pilots=self.pilots()+extras)
        self.assertEqual(result.spent_at_creation,15)
        self.assertEqual(result.reserved_by_plan,0)  # one unit cannot satisfy a two-unit floor
        self.assertEqual(result.unallocated_at_creation,1)
        reasons = {item.receipt_id:item.reason for item in result.ineligible_receipts}
        self.assertEqual(reasons["stale"],"policy_revision_changed")
        self.assertEqual(reasons["changed"],"task_revision_changed")
        self.assertEqual(reasons["agent-claimed"],"untrusted_reward_evidence")
        self.assertEqual(reasons["unknown"],"task_not_in_epoch")

    def test_duplicate_receipt_does_not_add_cost_or_fake_a_second_pilot(self):
        receipt = self.pilots({"b":[1]})[0]
        result = self.plan(pilots=(receipt,receipt),tasks=(TaskRevision("b","task-v1"),))
        self.assertEqual(self.ledger.snapshot()["spent"],1)
        self.assertEqual(result.reserved_by_plan,0)
        self.assertEqual(result.ineligible_receipts[0].reason,"insufficient_current_trusted_pilots")

    def test_conflicting_duplicate_receipt_fails_before_cost_record(self):
        receipt = self.pilots({"b":[1]})[0]
        with self.assertRaisesRegex(ValueError,"conflicting_duplicate"):
            self.plan(pilots=(receipt,dataclasses.replace(receipt,reward=0)))
        self.assertEqual(self.ledger.snapshot()["spent"],0)

    def test_probability_nonfinite_and_boolean_rewards_are_rejected(self):
        for reward in (.5, 1.0, float("nan"), float("inf"), True, "1"):
            with self.subTest(reward=reward),self.assertRaises(ValueError):
                PilotReceipt("x","a","task-v1","policy-7","resolved_binary",reward=reward,verification_id="v")
        with self.assertRaises(ValueError):
            PilotReceipt("x","a","task-v1","policy-7","unresolved",reward=0)

    def test_finite_thresholds_and_nonempty_keep_interval_required(self):
        for kwargs in ({"lower":float("nan")},{"upper":1.1},{"lower":.6,"upper":.6},{"exclude":True}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                SelectionThresholds(**kwargs)

    def test_exclusion_is_separate_subset_and_overlapping_keep_is_not_allocated(self):
        result = self.plan(thresholds=SelectionThresholds(exclude=.5))
        self.assertEqual(result.selection.keep,("b","c"))
        self.assertEqual(result.selection.exclude_too_easy,("b","c","d"))
        self.assertEqual(result.allocations,())
        self.assertEqual(result.reserved_by_plan,0)

    def test_inclusive_boundaries_and_per_task_ceilings(self):
        outcomes = {"a":[0,1,0,0],"b":[1,1,1,0]}
        result = self.plan(pilots=self.pilots(outcomes),tasks=(TaskRevision("b","task-v1",2,3),TaskRevision("a","task-v1",1,2)),
                           thresholds=SelectionThresholds(lower=.25,upper=.25))
        self.assertEqual(result.selection.keep,("a","b"))
        self.assertEqual([(item.task_id,item.count) for item in result.allocations],[("a",2),("b",3)])
        self.assertEqual(result.unallocated_at_creation,3)

    def test_floor_deferral_is_deterministic_and_never_overspends(self):
        self.ledger = PilotBudgetLedger(self.path,epoch_id="tiny",total_budget=7)
        result = self.plan(pilots=self.pilots({"b":[0,1],"c":[0,1]}),tasks=(TaskRevision("c","task-v1",2,8),TaskRevision("b","task-v1",2,8)))
        self.assertEqual([(item.task_id,item.count) for item in result.allocations],[("b",3),("c",0)])
        self.assertEqual(self.ledger.snapshot()["remaining"],0)

    def test_pilot_overspend_rolls_back_the_whole_batch(self):
        self.ledger = PilotBudgetLedger(self.path,epoch_id="tiny",total_budget=2)
        with self.assertRaisesRegex(ValueError,"pilot_costs_exceed"):
            self.plan()
        self.assertEqual(self.ledger.snapshot()["spent"],0)

    def test_prior_reservations_are_charged_before_new_pilots(self):
        self.plan()
        with self.assertRaisesRegex(ValueError,"pilot_costs_exceed"):
            self.ledger.record_pilots((PilotReceipt("extra","a","task-v1","policy-7","failed"),))
        self.assertEqual(self.ledger.snapshot(),{"total_budget":16,"spent":8,"reserved":8,"remaining":0})

    def test_plan_retry_across_new_handle_is_idempotent(self):
        first = self.plan()
        second_ledger = PilotBudgetLedger(self.path,epoch_id="epoch-7",total_budget=16)
        self.assertEqual(self.plan(ledger=second_ledger),first)
        self.assertEqual(second_ledger.snapshot()["reserved"],8)
        with self.assertRaisesRegex(ValueError,"plan_request_identity_conflict"):
            self.plan(ledger=second_ledger,thresholds=SelectionThresholds(lower=.2))

    def test_two_database_handles_racing_cannot_double_reserve(self):
        barrier = threading.Barrier(2)
        plans,errors = [],[]
        lock = threading.Lock()
        def worker(index):
            try:
                ledger = PilotBudgetLedger(self.path,epoch_id="epoch-7",total_budget=16)
                barrier.wait(timeout=5)
                plan = self.plan(ledger=ledger,request_id=f"race-{index}")
                with lock:
                    plans.append(plan)
            except Exception as exc:
                with lock:
                    errors.append(exc)
        threads = [threading.Thread(target=worker,args=(index,)) for index in range(2)]
        for thread in threads: thread.start()
        for thread in threads: thread.join(timeout=10)
        self.assertFalse(errors)
        self.assertEqual(len(plans),2)
        self.assertEqual(sorted(plan.reserved_by_plan for plan in plans),[0,8])
        self.assertEqual(self.ledger.snapshot()["reserved"],8)

    def test_claim_checks_current_policy_and_task_then_authorizes_exactly_once(self):
        reservation = self.plan().reservations[0]
        for policy,revision in (("policy-8","task-v1"),("policy-7","task-v2")):
            with self.assertRaisesRegex(ValueError,"revision_changed"):
                self.ledger.claim(reservation.reservation_id,owner_id="worker-1",current_policy_revision=policy,current_task_revision=revision)
        self.assertTrue(self.ledger.claim(reservation.reservation_id,owner_id="worker-1",current_policy_revision="policy-7",current_task_revision="task-v1"))
        self.assertFalse(self.ledger.claim(reservation.reservation_id,owner_id="worker-1",current_policy_revision="policy-7",current_task_revision="task-v1"))
        with self.assertRaisesRegex(ValueError,"another_worker"):
            self.ledger.claim(reservation.reservation_id,owner_id="worker-2",current_policy_revision="policy-7",current_task_revision="task-v1")

    def test_two_handles_racing_to_claim_authorize_only_one_worker(self):
        reservation = self.plan().reservations[0]
        barrier = threading.Barrier(2)
        granted,errors = [],[]
        lock = threading.Lock()
        def worker(index):
            try:
                ledger = PilotBudgetLedger(self.path,epoch_id="epoch-7",total_budget=16)
                barrier.wait(timeout=5)
                result = ledger.claim(reservation.reservation_id,owner_id=f"worker-{index}",
                    current_policy_revision="policy-7",current_task_revision="task-v1")
                with lock: granted.append(result)
            except Exception as exc:
                with lock: errors.append(exc)
        threads = [threading.Thread(target=worker,args=(index,)) for index in range(2)]
        for thread in threads: thread.start()
        for thread in threads: thread.join(timeout=10)
        self.assertEqual(granted,[True])
        self.assertEqual(len(errors),1)
        self.assertIsInstance(errors[0],ValueError)
        self.assertIn("another_worker",str(errors[0]))
        self.assertEqual(self.ledger.snapshot()["reserved"],8)

    def test_failed_completion_costs_once_and_cannot_be_refunded(self):
        reservation = self.plan().reservations[0]
        self.ledger.claim(reservation.reservation_id,owner_id="worker",current_policy_revision="policy-7",current_task_revision="task-v1")
        receipt = PilotReceipt("commit-failed",reservation.task_id,"task-v1","policy-7","failed")
        self.assertTrue(self.ledger.complete(reservation.reservation_id,owner_id="worker",receipt=receipt))
        self.assertFalse(self.ledger.complete(reservation.reservation_id,owner_id="worker",receipt=receipt))
        self.assertEqual(self.ledger.snapshot(),{"total_budget":16,"spent":9,"reserved":7,"remaining":0})
        with self.assertRaisesRegex(ValueError,"cannot_be_refunded"):
            self.ledger.cancel_pending(reservation.reservation_id)
        with self.assertRaisesRegex(ValueError,"receipt_conflict"):
            self.ledger.complete(reservation.reservation_id,owner_id="worker",receipt=dataclasses.replace(receipt,receipt_id="different"))

    def test_cancellation_only_refunds_never_claimed_work(self):
        result = self.plan()
        pending,claimed = result.reservations[:2]
        self.assertTrue(self.ledger.cancel_pending(pending.reservation_id))
        self.assertFalse(self.ledger.cancel_pending(pending.reservation_id))
        self.ledger.claim(claimed.reservation_id,owner_id="worker",current_policy_revision="policy-7",current_task_revision="task-v1")
        with self.assertRaisesRegex(ValueError,"cannot_be_refunded"):
            self.ledger.cancel_pending(claimed.reservation_id)
        self.assertEqual(self.ledger.snapshot()["remaining"],1)
        self.assertEqual(self.plan(),result)  # cancelled reservation is not recreated
        self.assertEqual(self.ledger.snapshot()["remaining"],1)

    def test_completion_requires_claim_owner_exact_revision_and_fresh_receipt(self):
        result = self.plan()
        reservation = result.reservations[0]
        receipt = PilotReceipt("done",reservation.task_id,"task-v1","policy-7","failed")
        with self.assertRaises(ValueError): self.ledger.complete(reservation.reservation_id,owner_id="worker",receipt=receipt)
        self.ledger.claim(reservation.reservation_id,owner_id="worker",current_policy_revision="policy-7",current_task_revision="task-v1")
        for invalid in (dataclasses.replace(receipt,policy_revision="policy-8"),dataclasses.replace(receipt,task_revision="task-v2"),dataclasses.replace(receipt,units=2)):
            with self.assertRaises(ValueError): self.ledger.complete(reservation.reservation_id,owner_id="worker",receipt=invalid)
        with self.assertRaises(ValueError): self.ledger.complete(reservation.reservation_id,owner_id="other",receipt=receipt)
        self.assertEqual(self.ledger.snapshot()["spent"],8)

    def test_malformed_unknown_duplicate_or_false_semantic_callback_results_fail_closed(self):
        mutations = (
            lambda value: {**value,"extra":[]},
            lambda value: {**value,"keep":[999]},
            lambda value: {**value,"keep":[1,1]},
            lambda value: {**value,"keep":[True,2]},
            lambda value: {**value,"keep":[0,1,2]},
            lambda value: {**value,"exclude_too_easy":[]},
            lambda value: {**value,"keep":"1,2"},
        )
        for index,mutation in enumerate(mutations):
            with self.subTest(index=index),self.assertRaises(ValueError):
                self.plan(request_id=f"bad-{index}",selector_backend=lambda data: mutation(strict_selector(data)))
        self.assertEqual(self.ledger.snapshot()["spent"],8)  # real pilots are still charged
        self.assertEqual(self.ledger.snapshot()["reserved"],0)

    def test_nonboolean_verifier_and_empty_eligible_set(self):
        with self.assertRaisesRegex(ValueError,"explicit_bool"):
            self.plan(evidence_verifier=lambda receipt: 1)
        def forbidden(data):
            self.fail("selector must not see untrusted outcomes")
        result = self.plan(request_id="untrusted",evidence_verifier=lambda receipt:False,selector_backend=forbidden)
        self.assertEqual(result.reserved_by_plan,0)
        self.assertEqual(len(result.ineligible_receipts),8)

    def test_epoch_budget_and_receipt_identity_are_immutable(self):
        self.ledger.record_pilots(self.pilots())
        with self.assertRaisesRegex(ValueError,"configuration_conflict"):
            PilotBudgetLedger(self.path,epoch_id="epoch-7",total_budget=17)
        with self.assertRaisesRegex(ValueError,"receipt_identity_conflict"):
            self.ledger.record_pilots((dataclasses.replace(self.pilots()[0],reward=1),))

    def test_extreme_valid_integer_plan_is_bounded_before_reservation_materialization(self):
        ledger = PilotBudgetLedger(self.path,epoch_id="large",total_budget=1_000_000_000)
        with patch("future_prediction_bench.pilot_commit.Reservation") as reservation_type:
            with self.assertRaisesRegex(ValueError,"bounded_batch_limit"):
                self.plan(ledger=ledger,pilots=self.pilots({"b":[0,1]}),tasks=(TaskRevision("b","task-v1",0,1_000_000_000),))
            reservation_type.assert_not_called()
        self.assertEqual(ledger.snapshot()["reserved"],0)
        self.assertEqual(ledger.snapshot()["spent"],2)

    def test_public_epoch_budget_configuration_is_read_only(self):
        with self.assertRaises(AttributeError):
            self.ledger.total_budget = 32
        self.assertEqual(self.ledger.snapshot()["total_budget"],16)

    def test_previously_spent_pilot_cannot_complete_another_reserved_rollout(self):
        reservation = self.plan().reservations[0]
        self.ledger.claim(reservation.reservation_id,owner_id="worker",current_policy_revision="policy-7",current_task_revision="task-v1")
        receipt = next(item for item in self.pilots() if item.task_id == reservation.task_id)
        with self.assertRaisesRegex(ValueError,"already_spent"):
            self.ledger.complete(reservation.reservation_id,owner_id="worker",receipt=receipt)
        self.assertEqual(self.ledger.snapshot(),{"total_budget":16,"spent":8,"reserved":8,"remaining":0})


class PreparationGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(tempfile.gettempdir()).resolve())
        self.root = Path(self.tmp.name)
        self.source = self.root/"source"
        self.dep1 = self.root/"deps1"
        self.dep2 = self.root/"deps2"
        for path in (self.source,self.dep1,self.dep2):
            path.mkdir()
        self.output = self.root/"output"

    def tearDown(self):
        self.tmp.cleanup()

    def reject_before_inputs(self,source=None,output=None,deps=None):
        with patch.object(probe,"source_state") as source_check,patch.object(probe,"dependency_state") as dependency_check:
            with self.assertRaises(ValueError):
                probe.checked_inputs(source or self.source,output or self.output,deps or [self.dep1,self.dep2])
            source_check.assert_not_called()
            dependency_check.assert_not_called()
            self.assertFalse(self.output.exists())

    def test_fresh_distinct_paths_do_not_create_output(self):
        with patch.object(probe,"source_state",return_value={}),patch.object(probe,"dependency_state",return_value={}) as dependency_check:
            result = probe.checked_inputs(self.source,self.output,[self.dep1,self.dep2])
            self.assertEqual(result[:3],(self.source,self.output,[self.dep1,self.dep2]))
            dependency_check.assert_called_once()
            self.assertFalse(self.output.exists())

    def test_output_inside_source(self):
        self.reject_before_inputs(output=self.source/"bad")

    def test_source_inside_output(self):
        self.reject_before_inputs(output=self.root)

    def test_output_inside_dependency(self):
        self.reject_before_inputs(output=self.dep1/"bad")

    def test_dependency_inside_output(self):
        self.reject_before_inputs(output=self.root)

    def test_source_dependency_overlap(self):
        self.reject_before_inputs(deps=[self.source,self.dep2])

    def test_duplicate_dependencies(self):
        self.reject_before_inputs(deps=[self.dep1,self.dep1])

    def test_existing_output_directory(self):
        existing = self.root/"existing"
        existing.mkdir()
        self.reject_before_inputs(output=existing)

    def test_existing_output_file(self):
        existing = self.root/"existing"
        existing.write_text("preserve")
        self.reject_before_inputs(output=existing)
        self.assertEqual(existing.read_text(),"preserve")

    def test_output_leaf_symlink(self):
        link = self.root/"link"
        link.symlink_to(self.dep1,target_is_directory=True)
        self.reject_before_inputs(output=link)

    def test_output_ancestor_symlink(self):
        link = self.root/"link"
        link.symlink_to(self.dep1,target_is_directory=True)
        self.reject_before_inputs(output=link/"new")

    def test_dangling_output_symlink(self):
        link = self.root/"link"
        link.symlink_to(self.root/"missing",target_is_directory=True)
        self.reject_before_inputs(output=link)

    def test_source_symlink(self):
        link = self.root/"link"
        link.symlink_to(self.source,target_is_directory=True)
        self.reject_before_inputs(source=link)

    def test_dependency_symlink(self):
        link = self.root/"link"
        link.symlink_to(self.dep1,target_is_directory=True)
        self.reject_before_inputs(deps=[link,self.dep2])

    def test_source_pin_failure_prevents_output_creation(self):
        with patch.object(probe,"source_state",side_effect=ValueError("pin mismatch")),patch.object(probe,"dependency_state") as dependency_check:
            with self.assertRaisesRegex(ValueError,"pin mismatch"):
                probe.checked_inputs(self.source,self.output,[self.dep1,self.dep2])
            dependency_check.assert_not_called()
            self.assertFalse(self.output.exists())

    def test_dependency_pin_failure_prevents_output_creation(self):
        with patch.object(probe,"source_state",return_value={}),patch.object(probe,"dependency_state",side_effect=ValueError("pin mismatch")):
            with self.assertRaisesRegex(ValueError,"pin mismatch"):
                probe.checked_inputs(self.source,self.output,[self.dep1,self.dep2])
            self.assertFalse(self.output.exists())

    def test_regular_file_hash_rejects_symlink(self):
        target = self.root/"target"
        target.write_bytes(b"preserve")
        link = self.root/"link"
        link.symlink_to(target)
        with self.assertRaises(OSError):
            probe.sha_file(link)
        self.assertEqual(target.read_bytes(),b"preserve")

    def test_regular_file_hash_rejects_directory(self):
        with self.assertRaises(ValueError):
            probe.sha_file(self.dep1)


if __name__ == "__main__":
    unittest.main()
