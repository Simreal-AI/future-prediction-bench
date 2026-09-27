"""Use genuine Boltons case outcomes with the unchanged original selector.

This runs offline Docker repository episodes with explicit scripted controls.
It does not call a model, produce policy gradients or measure training speed.
The generic production provider interface lives in pilot_commit_coding.py.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import sys

from examples.official_pilot_commit.check_components import (
    COMMIT, DEPENDENCY_PINS, checked_inputs, dependency_state, source_state)
from examples.realworld_boltons26.make_task import ARCHIVE_SHA256, SOURCE_COMMIT
from future_prediction_bench.pilot_commit import PilotBudgetLedger
from future_prediction_bench.pilot_commit_coding import CodingOutcomeProducer, CodingPolicyBinding
from future_prediction_bench.realworld import validate_task
from future_prediction_bench.realworld_demo import load_coding_actions


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ScriptedControlProvider:
    """A fixed two-control policy with an explicit per-episode sample index."""

    def __init__(self, revision, controls, sample_index):
        self.policy_revision = revision
        self.controls, self.sample_index = controls, sample_index
        self.calls = 0

    def next_action(self, observation, *, action_index, reservation_id):
        self.calls += 1
        # Expected outputs and grader state are not part of observation.
        return copy.deepcopy(self.controls[self.sample_index % len(self.controls)][action_index])


def check(*, task_dir, image, source, dependencies, output):
    source, output, dependencies, dependency_pins = checked_inputs(
        Path(source), Path(output), [Path(p) for p in dependencies])
    task_dir = Path(task_dir).resolve(strict=True)
    if (output.is_relative_to(task_dir) or task_dir.is_relative_to(output)):
        raise ValueError("task_inputs_and_output_must_be_disjoint")
    inputs = {name: task_dir / name for name in
              ("task.json", "actions.baseline.jsonl", "actions.solution.jsonl", "boltons-26.0.0.tar.gz")}
    if any(path.is_symlink() or not path.is_file() for path in inputs.values()):
        raise ValueError("regular_public_task_and_action_inputs_required")
    input_pins = {name: sha(path) for name, path in inputs.items()}
    if input_pins["boltons-26.0.0.tar.gz"] != ARCHIVE_SHA256:
        raise ValueError("genuine_pinned_Boltons_distribution_required")
    task = json.loads(inputs["task.json"].read_bytes())
    validate_task(task)
    if (task["is_fixture"] is not True or task.get("metadata", {}).get("source_sdist_sha256") != ARCHIVE_SHA256 or
            task.get("metadata", {}).get("source_commit") != SOURCE_COMMIT):
        raise ValueError("explicit_public_upstream_repair_fixture_required")
    deadline = datetime.fromisoformat(task["action_deadline"].replace("Z", "+00:00"))
    if deadline <= datetime.now(timezone.utc):
        raise ValueError("generate_a_fresh_task_window_from_the_pinned_cached_distribution")
    if any(name == "verl" or name.startswith(("verl.", "recipe.pc")) for name in sys.modules):
        raise ValueError("fresh_process_without_preloaded_author_selector_required")
    before_source = source_state(source)
    sys.path[:0] = [str(source), *[str(path) for path in dependencies]]
    sys.dont_write_bytecode = True
    import numpy as np
    from recipe.pc.utils import select_prompts
    if Path(inspect.getfile(select_prompts)).resolve() != source / "recipe/pc/utils.py":
        raise ValueError("normal_original_selector_import_required")
    for name in DEPENDENCY_PINS:
        if not any(Path(sys.modules[name].__file__).resolve().is_relative_to(root) for root in dependencies):
            raise ValueError("original_selector_dependency_outside_pinned_inputs")
    controls = [load_coding_actions(inputs["actions.baseline.jsonl"]),
                load_coding_actions(inputs["actions.solution.jsonl"])]
    config = {"kind": "scripted_two_control_alternating_samples_v1",
              "controls_sha256": [input_pins["actions.baseline.jsonl"], input_pins["actions.solution.jsonl"]]}
    binding = CodingPolicyBinding(sha(__file__), hashlib.sha256(task["prompt"].encode()).hexdigest(),
        hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
    output.mkdir(parents=True)
    report = {"schema_version": "official-pilot-coding-outcomes-v1", "passed": False,
        "scope": "actual_scripted_public_repository_episodes_and_original_selector_not_model_rollouts",
        "model_inference_executed": False, "optimizer_update_executed": False,
        "gpu_speedup_measured": False, "input_sha256": input_pins,
        "driver_sha256": sha(__file__), "scripted_policy_revision": binding.revision,
        "source_before": before_source, "dependencies_before": dependency_pins,
        "original_selector_calls": [], "dispatches": []}
    (output / "driver-source.py").write_bytes(Path(__file__).read_bytes())
    ledger = PilotBudgetLedger(output / "budget.sqlite3", epoch_id="actual-repository-control", total_budget=4)
    try:
        producer = CodingOutcomeProducer(ledger, task=task, seed_dir=task_dir / "seed",
            verifier_dir=task_dir / "verifier", image=image, output=output / "producer",
            policy_binding=binding, visible_check=("python3", "-B", "-c", "import boltons.strutils"))
        bootstrap = producer.reserve_pilots(request_id="initial-two-controls", count=2)
        report["initial_plan"] = asdict(bootstrap)
        report["budget_after_initial_reservation"] = ledger.snapshot()
        pilots = []
        for index, reservation in enumerate(bootstrap.reservations):
            provider = ScriptedControlProvider(binding.revision, controls, index)
            result = producer.dispatch(reservation, owner_id="scripted-control-worker", provider=provider)
            report["dispatches"].append({"phase": "pilot", "sample_index": index,
                "reservation_id": reservation.reservation_id, "dispatched": result.dispatched,
                "receipt": asdict(result.receipt) if result.receipt else None,
                "outcome_sha256": result.outcome_sha256, "provider_calls": provider.calls})
            if result.receipt is None or not producer.verify_receipt(result.receipt):
                raise RuntimeError("genuine_terminal_repository_pilot_required")
            pilots.append(result.receipt)
        report["budget_after_actual_pilots"] = ledger.snapshot()

        def original_selector(data):
            actual = select_prompts(np.asarray(data.prompt_indices, dtype=np.int64),
                np.asarray(data.rewards, dtype=np.float64),
                diversity_threshold_upper=data.thresholds.upper,
                diversity_threshold_lower=data.thresholds.lower,
                exclude_threshold_upper=data.thresholds.exclude)
            normalized = {key: [int(index) for index in value] for key, value in actual.items()}
            report["original_selector_calls"].append({"input": asdict(data), "actual_output": normalized})
            return normalized

        plan = producer.plan_commits(request_id="allocate-from-actual-cases", pilots=pilots,
            selector_backend=original_selector, selector_id="databricks-pilot-commit-" + COMMIT,
            commit_floor=2, commit_cap=2)
        report["actual_commit_plan"] = asdict(plan)
        report["budget_after_actual_selection"] = ledger.snapshot()
        if not plan.reservations:
            raise RuntimeError("actual_control_outcomes_did_not_select_any_commit_job")
        for index, reservation in enumerate(plan.reservations, start=2):
            provider = ScriptedControlProvider(binding.revision, controls, index)
            result = producer.dispatch(reservation, owner_id="scripted-commit-worker", provider=provider)
            report["dispatches"].append({"phase": "commit", "sample_index": index,
                "reservation_id": reservation.reservation_id, "dispatched": result.dispatched,
                "receipt": asdict(result.receipt) if result.receipt else None,
                "outcome_sha256": result.outcome_sha256, "provider_calls": provider.calls})
            if result.receipt is None or not producer.verify_receipt(result.receipt):
                raise RuntimeError("genuine_terminal_repository_commit_required")
        retry_provider = ScriptedControlProvider(binding.revision, controls, 0)
        retry = producer.dispatch(bootstrap.reservations[0], owner_id="scripted-control-worker", provider=retry_provider)
        report["duplicate_dispatch"] = {"dispatched": retry.dispatched,
            "provider_calls": retry_provider.calls, "receipt": asdict(retry.receipt) if retry.receipt else None}
        report["final_budget"] = ledger.snapshot()
        if retry.dispatched or retry_provider.calls or report["final_budget"] != {
                "total_budget": 4, "spent": 4, "reserved": 0, "remaining": 0}:
            raise RuntimeError("preclaimed_budget_or_no_redispatch_witness_failed")
        report["passed"] = True
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        try:
            report["source_after"] = source_state(source)
            report["dependencies_after"] = dependency_state(dependencies)
            report["input_sha256_after"] = {name: sha(path) for name, path in inputs.items()}
            if (report["source_after"] != before_source or report["dependencies_after"] != dependency_pins or
                    report["input_sha256_after"] != input_pins or sha(__file__) != report["driver_sha256"] or
                    sha(output / "driver-source.py") != report["driver_sha256"]):
                raise ValueError("original_source_or_actual_task_inputs_changed")
        except Exception as exc:
            report["preservation_error"] = str(exc)
            report["passed"] = False
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--dependency", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = check(task_dir=args.task_dir, image=args.image, source=args.source,
                   dependencies=args.dependency, output=args.output)
    print(json.dumps({"passed": result["passed"], "error": result.get("error"),
                      "final_budget": result.get("final_budget")}, indent=2))
    raise SystemExit(0 if result["passed"] else 2)
