"""Run the unchanged original selector through the product budget planner.

The binary outcomes are explicitly owned CPU fixtures. SQLite and original
selection code are real; no model, sandbox rollout or optimizer executes.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import traceback

from examples.official_pilot_commit.check_components import (
    checked_inputs, dependency_state, run as run_components, source_state,
)
from future_prediction_bench.pilot_commit import (
    PilotBudgetLedger, PilotReceipt, SelectionThresholds, TaskRevision,
    plan_pilot_commit,
)


def integration(output: Path):
    import numpy as np
    from recipe.pc.utils import select_prompts
    calls = []

    def original_selector(data):
        result = select_prompts(np.asarray(data.prompt_indices,dtype=np.int64),
            np.asarray(data.rewards,dtype=np.float64),
            diversity_threshold_upper=data.thresholds.upper,
            diversity_threshold_lower=data.thresholds.lower,
            exclude_threshold_upper=data.thresholds.exclude)
        normalized = {key:[int(index) for index in value] for key,value in result.items()}
        calls.append({"input":asdict(data),"actual_original_output":normalized})
        return normalized

    registry = {}
    def receipts(outcomes, *, policy="policy-7"):
        result = []
        for task,values in outcomes.items():
            for index,reward in enumerate(values):
                receipt = PilotReceipt(f"{task}-{index}",task,"task-v1",policy,"resolved_binary",reward=reward,verification_id=f"owned-fixture:{task}-{index}")
                registry[receipt.verification_id] = asdict(receipt)
                result.append(receipt)
        return tuple(result)

    def verified(receipt):
        return registry.get(receipt.verification_id) == asdict(receipt)

    outcomes = {"always-fails":[0,0],"mixed-a":[0,1],"mixed-b":[1,0],"always-passes":[1,1]}
    pilots = receipts(outcomes)
    tasks = tuple(TaskRevision(key,"task-v1") for key in outcomes)
    params = {"request_id":"original-selector-plan", "current_policy_revision":"policy-7", "tasks":tasks,
        "pilots":pilots, "selector_backend":original_selector,"evidence_verifier":verified,
        "selector_id":"databricks-pilot-commit-6def20ea-selector","evidence_verifier_id":"owned-CPU-fixture-registry-v1"}
    ledger = PilotBudgetLedger(output/"budget.sqlite3",epoch_id="primary",total_budget=16)
    first = plan_pilot_commit(ledger,**params)
    assert first.selection.keep == ("mixed-a","mixed-b")
    assert [(item.task_id,item.count) for item in first.allocations] == [("mixed-a",4),("mixed-b",4)]
    assert ledger.snapshot() == {"total_budget":16,"spent":8,"reserved":8,"remaining":0}
    reopened = PilotBudgetLedger(output/"budget.sqlite3",epoch_id="primary",total_budget=16)
    retry = plan_pilot_commit(reopened,**{**params,"tasks":tuple(reversed(tasks)),"pilots":tuple(reversed(pilots))})
    assert retry == first and reopened.snapshot()["reserved"] == 8
    rejects = []
    reservation = first.reservations[0]
    for policy,revision in (("policy-8","task-v1"),("policy-7","task-v2")):
        try:
            reopened.claim(reservation.reservation_id,owner_id="dispatcher",current_policy_revision=policy,current_task_revision=revision)
        except ValueError as exc:
            rejects.append({"policy":policy,"task_revision":revision,"error":str(exc)})
        else:
            raise AssertionError("stale_dispatch_was_not_rejected")
    granted = reopened.claim(reservation.reservation_id,owner_id="dispatcher",current_policy_revision="policy-7",current_task_revision="task-v1")
    repeated_claim = reopened.claim(reservation.reservation_id,owner_id="dispatcher",current_policy_revision="policy-7",current_task_revision="task-v1")
    assert granted is True and repeated_claim is False
    failed = PilotReceipt("failed-commit",reservation.task_id,"task-v1","policy-7","failed")
    charged = ledger.complete(reservation.reservation_id,owner_id="dispatcher",receipt=failed)
    repeated_completion = reopened.complete(reservation.reservation_id,owner_id="dispatcher",receipt=failed)
    assert charged is True and repeated_completion is False
    assert ledger.snapshot() == {"total_budget":16,"spent":9,"reserved":7,"remaining":0}
    exhausted = plan_pilot_commit(reopened,**{**params,"request_id":"overlapping-next-plan"})
    assert exhausted.reserved_by_plan == 0

    overlap_ledger = PilotBudgetLedger(output/"budget.sqlite3",epoch_id="overlapping-exclusion",total_budget=16)
    overlap = plan_pilot_commit(overlap_ledger,**params,thresholds=SelectionThresholds(exclude=.5))
    assert overlap.selection.keep == ("mixed-a","mixed-b")
    assert set(overlap.selection.exclude_too_easy) == {"mixed-a","mixed-b","always-passes"}
    assert overlap.reserved_by_plan == 0

    small_ledger = PilotBudgetLedger(output/"budget.sqlite3",epoch_id="small-floor",total_budget=7)
    small = plan_pilot_commit(small_ledger,**{**params,"tasks":tuple(task for task in tasks if task.task_id.startswith("mixed")),
        "pilots":tuple(item for item in pilots if item.task_id.startswith("mixed"))})
    assert [(item.task_id,item.count) for item in small.allocations] == [("mixed-a",3),("mixed-b",0)]

    extras = (PilotReceipt("past-failed","mixed-a","task-v1","policy-7","failed"),
              PilotReceipt("past-unresolved","mixed-a","task-v1","policy-7","unresolved"),
              PilotReceipt("past-stale","mixed-a","task-v1","policy-6","resolved_binary",reward=1,verification_id="older-policy-verifier"))
    cost_ledger = PilotBudgetLedger(output/"budget.sqlite3",epoch_id="all-past-costs",total_budget=17)
    cost = plan_pilot_commit(cost_ledger,**{**params,"pilots":pilots+extras})
    assert cost.spent_at_creation == 11 and cost.reserved_by_plan == 6
    assert {item.receipt_id for item in cost.ineligible_receipts} == {item.receipt_id for item in extras}

    before_untrusted = len(calls)
    untrusted_ledger = PilotBudgetLedger(output/"budget.sqlite3",epoch_id="untrusted",total_budget=16)
    untrusted = plan_pilot_commit(untrusted_ledger,**{**params,"evidence_verifier":lambda receipt:False,"evidence_verifier_id":"deny-all-fixture"})
    assert len(calls) == before_untrusted and untrusted.reserved_by_plan == 0 and untrusted.spent_at_creation == 8
    return {"scope":"real_original_selector_and_product_SQLite_control_plane_with_owned_binary_fixtures",
        "environment_rollout_executed":False,"model_inference_executed":False,"optimizer_update_executed":False,
        "original_selector_call_count":len(calls),"original_selector_calls":calls,
        "primary_plan":asdict(first),"retry_equals_original":retry == first,
        "stale_dispatch_rejections":rejects,"first_claim_granted":granted,"repeated_claim_granted":repeated_claim,
        "failed_completion_charged":charged,"repeated_completion_charged":repeated_completion,
        "primary_final_budget":ledger.snapshot(),"overlapping_next_plan":asdict(exhausted),
        "independent_exclusion_overlap_plan":asdict(overlap),"small_floor_plan":asdict(small),
        "failed_unresolved_stale_cost_plan":asdict(cost),"untrusted_plan":asdict(untrusted),"passed":True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--dependency",type=Path,action="append",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--official-tests",action="store_true")
    args = parser.parse_args()
    source,output,dependencies,pins = checked_inputs(args.source,args.output,args.dependency)
    output.mkdir(parents=True,exist_ok=False)
    report = {"passed":False,"scope":"original_CPU_components_plus_guarded_product_allocation"}
    try:
        report = run_components(source,output,dependencies,args.official_tests,pins)
        report["planner_integration"] = integration(output)
        core = Path(__file__).resolve().parents[2]/"future_prediction_bench/pilot_commit.py"
        report["core_source_sha256"] = hashlib.sha256(core.read_bytes()).hexdigest()
        report["planner_probe_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        report["source_preservation"]["after_planner"] = source_state(source)
        report["dependency_preservation"]["after_planner"] = dependency_state(dependencies)
        report["passed"] = True
    except Exception as exc:
        report["passed"] = False
        report["error"] = {"type":type(exc).__name__,"message":str(exc)[-2000:],"traceback_tail":traceback.format_exc()[-8000:]}
    (output/"result.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"passed":report["passed"],"original_selector_call_count":report.get("planner_integration",{}).get("original_selector_call_count"),"official_tests":report.get("official_tests",{}).get("counts")},indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception as exc:
        print(json.dumps({"passed":False,"error":str(exc)[-2000:]}))
        exit_code = 2
    raise SystemExit(exit_code)
