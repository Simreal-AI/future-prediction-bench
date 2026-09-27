"""Measure fresh simulated CI worlds against SQL reseeding on this host.

Both arms run the same scripted tools and host verifier. The baseline creates
and inserts the complete seed into a new SQLite file for every rollout; the
candidate copies one prepared seed SQLite file into a new rollout directory.
This is not a Collinear, Docker, VM, policy-model, or RL-training benchmark.
"""

from __future__ import annotations

import argparse
import json
import platform
import sqlite3
import statistics
import tempfile
import time
from pathlib import Path

from .world import (HISTORY_ROWS, TARGET_RUN, TARGET_TIMEOUT_MS,
                    HostVerifier, SeedArtifact, solved_trace, step)


def _summary(values):
    ordered = sorted(values)
    return {"count": len(ordered),
            "median_ms": round(statistics.median(ordered) * 1000, 6),
            "p95_ms": round(ordered[(95 * len(ordered) + 99) // 100 - 1] * 1000, 6),
            "min_ms": round(ordered[0] * 1000, 6),
            "max_ms": round(ordered[-1] * 1000, 6)}


def _episode(seed, verifier, mode):
    started = time.perf_counter()
    with seed.new_world(mode) as world:
        discovery_started = time.perf_counter()
        specs = world.handle("GET", "/tools")["tools"]
        discovery_seconds = time.perf_counter() - discovery_started
        if len(specs) != 4:
            raise RuntimeError("tool schema changed")
        solved_trace(world, discovered=specs)
        verification_started = time.perf_counter()
        grade = verifier.verify(world)
        verification_seconds = time.perf_counter() - verification_started
        if grade["reward"] != 1.0 or not all(grade["checks"].values()):
            raise RuntimeError("reference trace did not pass the host verifier")
        result = {"mode": mode, "reward": grade["reward"],
                  "initial_digest": world.initial_digest,
                  "final_digest": grade["final_digest"],
                  "application_diff_digest": grade["application_diff_digest"],
                  "setup_seconds": world.setup_seconds,
                  "tool_discovery_seconds": discovery_seconds,
                  "step_seconds": [entry["step_seconds"] for entry in world.trace],
                  "verification_seconds": verification_seconds,
                  "state_diff_count": sum(len(entry["state_diff"]) for entry in world.trace)}
    result["graded_episode_seconds"] = time.perf_counter() - started
    return result


def _sibling_and_adversarial_checks(seed, verifier):
    with seed.new_world("seed_copy") as first, seed.new_world("seed_copy") as second:
        if first.database_path == second.database_path:
            raise RuntimeError("sibling rollouts share a writable database")
        if not first.state_digest() == second.state_digest() == seed.initial_digest:
            raise RuntimeError("sibling seeds differ")
        solved_trace(first)
        if second.state_digest() != seed.initial_digest:
            raise RuntimeError("first sibling mutated the second")
        solved_trace(second)
        first_grade, second_grade = verifier.verify(first), verifier.verify(second)
        if (first_grade["reward"] != 1.0 or second_grade["reward"] != 1.0
                or first_grade["final_digest"] != second_grade["final_digest"]):
            raise RuntimeError("independent siblings diverged")
        sibling_digest = first_grade["final_digest"]

    attacks = {}
    with seed.new_world("seed_copy") as world:
        step(world, "post_triage_note", run_id=TARGET_RUN,
             body="Fixed release/2.4 linux-integration timeout and rerun passed.")
        attacks["text_only_note"] = verifier.verify(world)["reward"]
    with seed.new_world("seed_copy") as world:
        unchanged = world.state_digest()
        world.trace.append({"action": {"tool_name": "queue_rerun", "parameters": {
            "run_id": TARGET_RUN}}, "response": {"observation": "passed"},
            "before_digest": unchanged, "after_digest": unchanged,
            "state_diff": [], "step_seconds": 0.0})
        attacks["forged_text_trace"] = verifier.verify(world)["reward"]
    with seed.new_world("seed_copy") as world:
        step(world, "queue_rerun", run_id=TARGET_RUN)
        attacks["rerun_without_fix"] = verifier.verify(world)["reward"]
    with seed.new_world("seed_copy") as world:
        step(world, "update_job_timeout", run_id=TARGET_RUN, timeout_ms=300_000)
        step(world, "queue_rerun", run_id=TARGET_RUN)
        attacks["overbroad_timeout"] = verifier.verify(world)["reward"]
    with seed.new_world("seed_copy") as world:
        step(world, "update_job_timeout", run_id=1843, timeout_ms=TARGET_TIMEOUT_MS)
        step(world, "queue_rerun", run_id=1843)
        attacks["wrong_branch"] = verifier.verify(world)["reward"]
    if any(value != 0.0 for value in attacks.values()):
        raise RuntimeError("an adversarial trace obtained a false reward")
    return {"sibling_final_digest": sibling_digest,
            "sibling_initial_digest": seed.initial_digest,
            "independent_siblings": True,
            "adversarial_rewards": attacks,
            "false_rewards": sum(value > 0 for value in attacks.values())}


def benchmark(*, repetitions=30, warmup=3):
    if type(repetitions) is not int or not 5 <= repetitions <= 100:
        raise ValueError("repetitions must be 5..100")
    if type(warmup) is not int or not 0 <= warmup <= 20:
        raise ValueError("warmup must be 0..20")
    with tempfile.TemporaryDirectory(prefix="fpb-ci-seed-") as directory:
        seed = SeedArtifact(directory)
        verifier = HostVerifier(seed)
        checks = _sibling_and_adversarial_checks(seed, verifier)
        pairs = []
        for index in range(warmup + repetitions):
            order = ("sql_reseed", "seed_copy") if index % 2 == 0 else (
                "seed_copy", "sql_reseed")
            values = {mode: _episode(seed, verifier, mode) for mode in order}
            if values["sql_reseed"]["initial_digest"] != values["seed_copy"]["initial_digest"]:
                raise RuntimeError("baseline and prepared starts differ")
            if (values["sql_reseed"]["final_digest"] != values["seed_copy"]["final_digest"]
                    or values["sql_reseed"]["application_diff_digest"]
                    != values["seed_copy"]["application_diff_digest"]):
                raise RuntimeError("baseline and prepared final states differ")
            if index >= warmup:
                pairs.append({"order": list(order), "sql_reseed": values["sql_reseed"],
                              "seed_copy": values["seed_copy"]})
        modes = {}
        for mode in ("sql_reseed", "seed_copy"):
            values = [pair[mode] for pair in pairs]
            modes[mode] = {
                "setup": _summary([value["setup_seconds"] for value in values]),
                "tool_discovery": _summary([value["tool_discovery_seconds"] for value in values]),
                "step": _summary([step_time for value in values
                                  for step_time in value["step_seconds"]]),
                "verification": _summary([value["verification_seconds"] for value in values]),
                "graded_episode": _summary([value["graded_episode_seconds"] for value in values]),
            }
        baseline_total = sum(pair["sql_reseed"]["graded_episode_seconds"] for pair in pairs)
        prepared_total = sum(pair["seed_copy"]["graded_episode_seconds"] for pair in pairs)
        report = {"kind": "simulated_ci_triage_collinear_transfer_v1",
                  "scope": "stdlib SQLite CI simulation; no Docker, VM, service, model, or trainer",
                  "method": {"baseline": "fresh SQLite file with schema and seed SQL per rollout",
                             "candidate": "fresh private SQLite file copied from a prepared seed artifact",
                             "protocol": "in-process GET /tools and POST /step shapes",
                             "pair_order": "alternated on one host",
                             "repetitions": repetitions, "warmup": warmup,
                             "history_rows": HISTORY_ROWS},
                  "host": {"platform": platform.platform(),
                           "python": platform.python_version(), "sqlite": sqlite3.sqlite_version},
                  "seed": {"initial_state_digest": seed.initial_digest,
                           "template_build_ms": round(seed.build_seconds * 1000, 6)},
                  "correctness": checks,
                  "timing": {"modes": modes,
                             "baseline_total_ms": round(baseline_total * 1000, 6),
                             "prepared_total_ms": round(prepared_total * 1000, 6),
                             "prepared_plus_template_ms": round(
                                 (prepared_total + seed.build_seconds) * 1000, 6),
                             "episode_time_ratio_baseline_over_prepared": round(
                                 baseline_total / prepared_total, 6),
                             "amortized_time_ratio_baseline_over_prepared": round(
                                 baseline_total / (prepared_total + seed.build_seconds), 6)},
                  "pairs": pairs,
                  "limitations": [
                      "synthetic application state and deterministic scripted tool trace",
                      "logical per-rollout files only; no container or VM security isolation",
                      "no inference, optimizer update, or real CI service",
                      "one host and a small SQLite seed; ratios are not Collinear results"]}
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    report = benchmark(repetitions=args.repetitions, warmup=args.warmup)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"correctness": report["correctness"],
                      "timing": report["timing"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
