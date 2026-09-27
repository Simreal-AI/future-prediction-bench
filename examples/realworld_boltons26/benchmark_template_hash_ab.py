"""Reproduce the template-source-hashing A/B on the public Boltons v2 fixture.

The legacy arm restores the old hash schedule around the current spawn code:
one source SHA before cloning, one after constructing its sole child and before
boot, and one after boot. Both arms retain the same per-child pinned SHA check.
All arms use one fixed task deadline and one child per prepared episode.
Invoke as ``python3 -m examples.realworld_boltons26.benchmark_template_hash_ab``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
import time
from pathlib import Path
from unittest.mock import patch

import future_prediction_bench.microvm_template as template_module
from . import benchmark_small_edit_v2 as bench


_SOURCE_HASH_PHASES = []
_TEMPLATE_SIZES = []


def _checked_source_hash(template, expected, *, phase, mismatch_reason):
    started = time.monotonic()
    actual = template_module._sha256_file(template.disk_path)
    _SOURCE_HASH_PHASES.append({
        "phase": phase,
        "seconds": time.monotonic() - started,
        "template_disk_bytes": template.disk_path.stat().st_size,
    })
    if actual != expected:
        raise template_module.MicroVMRuntimeError(mismatch_reason)


def _legacy_spawn(self, child_disk_paths, *, max_workers=4, popen_factory=None):
    _TEMPLATE_SIZES.append(self.disk_path.stat().st_size)
    original_verified = template_module.MicroVMTemplate._verified_manifest
    original_runtime = template_module.MicroVMRuntime
    original_spawn = _legacy_spawn.candidate_spawn
    expected = None

    def verified_with_source_sha(template):
        nonlocal expected
        manifest = original_verified(template)
        expected = manifest["disk_sha256"]
        _checked_source_hash(template, expected, phase="before_clone",
                             mismatch_reason="template_disk_digest_mismatch")
        return manifest

    def runtime_then_source_sha(*args, **kwargs):
        child = original_runtime(*args, **kwargs)
        _checked_source_hash(self, expected, phase="after_clone_before_boot",
                             mismatch_reason="template_disk_changed_during_spawn")
        return child

    with patch.object(template_module.MicroVMTemplate, "_verified_manifest",
                      verified_with_source_sha), \
         patch.object(template_module, "MicroVMRuntime", runtime_then_source_sha):
        children = original_spawn(self, child_disk_paths, max_workers=max_workers,
                                  popen_factory=popen_factory)
    try:
        _checked_source_hash(self, expected, phase="after_boot",
                             mismatch_reason="template_disk_changed_during_spawn")
    except template_module.MicroVMRuntimeError:
        for child in children:
            child.close()
        for path in child_disk_paths:
            Path(path).unlink(missing_ok=True)
        raise
    return children


_legacy_spawn.candidate_spawn = template_module.MicroVMTemplate.spawn


def _candidate_spawn_with_size(self, child_disk_paths, *, max_workers=4,
                               popen_factory=None):
    _TEMPLATE_SIZES.append(self.disk_path.stat().st_size)
    return _legacy_spawn.candidate_spawn(
        self, child_disk_paths, max_workers=max_workers,
        popen_factory=popen_factory)


def _same_semantics(left, right):
    left_rows = {(r["method"], r["branch"], r["condition"]): r
                 for r in left["runs"]}
    right_rows = {(r["method"], r["branch"], r["condition"]): r
                  for r in right["runs"]}
    if (len(left["runs"]) != 8 or len(right["runs"]) != 8
            or len(left_rows) != 8 or len(right_rows) != 8
            or left_rows.keys() != right_rows.keys()):
        raise RuntimeError("A/B arm identities differ")
    for key in left_rows:
        for field in ("task_sha256", "opening_observation",
                      "action_observation_sha256s", "case_results", "reward",
                      "passed_cases", "final_source_sha256"):
            if left_rows[key][field] != right_rows[key][field]:
                raise RuntimeError(f"A/B semantic mismatch for {key}: {field}")
        for row in (left_rows[key], right_rows[key]):
            if (row["adapter_metrics"]["full_vm_restores"] != 1
                    or row["adapter_metrics"]["stateless_batches"] != 1):
                raise RuntimeError(f"A/B restore/batch count differs for {key}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stateless-contract", required=True)
    parser.add_argument("--pairs", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.pairs <= 5:
        parser.error("pairs must be in [1, 5]")
    _SOURCE_HASH_PHASES.clear()
    _TEMPLATE_SIZES.clear()
    output = Path(args.output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    task_path = Path(args.task_dir) / "task.json"
    source_task = json.loads(task_path.read_text(encoding="utf-8"))
    fixed_task = bench._fixture_task(source_task)
    original_fixture_task = bench._fixture_task

    def same_fixed_task(task):
        if task["task_id"] != fixed_task["task_id"]:
            raise RuntimeError("A/B task identity changed")
        return copy.deepcopy(fixed_task)

    bench._fixture_task = same_fixed_task
    reports = []
    sizes_by_pair = []
    order_by_pair = []
    try:
        for pair_index in range(args.pairs):
            order = ("legacy_hash_schedule", "child_hash_only") if pair_index % 2 == 0 \
                    else ("child_hash_only", "legacy_hash_schedule")
            pair_reports = {}
            pair_sizes = {}
            for variant in order:
                run_output = output / f"pair-{pair_index}-{variant}"
                phase_start = len(_SOURCE_HASH_PHASES)
                size_start = len(_TEMPLATE_SIZES)
                if variant == "legacy_hash_schedule":
                    with patch.object(template_module.MicroVMTemplate, "spawn",
                                      _legacy_spawn):
                        report = bench.benchmark(
                            args.task_dir, args.assets_dir, run_output,
                            repetitions=1, stateless_contract=args.stateless_contract,
                            preinstall_stateless_helper=True)
                else:
                    with patch.object(template_module.MicroVMTemplate, "spawn",
                                      _candidate_spawn_with_size):
                        report = bench.benchmark(
                            args.task_dir, args.assets_dir, run_output,
                            repetitions=1, stateless_contract=args.stateless_contract,
                            preinstall_stateless_helper=True)
                pair_reports[variant] = report
                sizes = _TEMPLATE_SIZES[size_start:]
                if len(sizes) != 4 or len(set(sizes)) != 1:
                    raise RuntimeError("Expected four prepared child spawns from one template size")
                pair_sizes[variant] = sizes[0]
                if variant == "legacy_hash_schedule":
                    phases = _SOURCE_HASH_PHASES[phase_start:]
                    if (len(phases) != 12
                            or {phase["phase"] for phase in phases}
                               != {"before_clone", "after_clone_before_boot", "after_boot"}):
                        raise RuntimeError("Legacy arm did not execute exactly three source hashes per prepared episode")
                print(f"pair={pair_index} variant={variant} completed", flush=True)
            _same_semantics(pair_reports["legacy_hash_schedule"],
                            pair_reports["child_hash_only"])
            if (reports and pair_reports["child_hash_only"]["task_sha256"]
                    != reports[0]["child_hash_only"]["task_sha256"]):
                raise RuntimeError("A/B task changed between pairs")
            reports.append(pair_reports)
            sizes_by_pair.append(pair_sizes)
            order_by_pair.append(list(order))
            print(f"pair={pair_index} cross-variant semantics PASS", flush=True)
    finally:
        bench._fixture_task = original_fixture_task

    legacy = [r for pair in reports for r in pair["legacy_hash_schedule"]["runs"]
              if r["condition"] == "prepared"]
    candidate = [r for pair in reports for r in pair["child_hash_only"]["runs"]
                 if r["condition"] == "prepared"]
    summary = {"kind": "template_source_hash_ab_v1",
               "task_sha256": reports[0]["child_hash_only"]["task_sha256"],
               "candidate_source_sha256": hashlib.sha256(
                   Path(template_module.__file__).read_bytes()).hexdigest(),
               "pairs": args.pairs, "arms_per_variant": len(legacy) * 2,
               "all_cross_variant_semantics_passed": True,
               "variant_order_by_pair": order_by_pair,
               "template_disk_bytes_by_pair": sizes_by_pair,
               "legacy_source_hash_phases": _SOURCE_HASH_PHASES,
               "legacy_source_hash_median_seconds": statistics.median(
                   phase["seconds"] for phase in _SOURCE_HASH_PHASES),
               "legacy_source_hash_total_seconds": sum(
                   phase["seconds"] for phase in _SOURCE_HASH_PHASES),
               "prepared_median_legacy_seconds": statistics.median(
                   r["wall_seconds"] for r in legacy),
               "prepared_median_candidate_seconds": statistics.median(
                   r["wall_seconds"] for r in candidate),
               "prepared_total_legacy_seconds": sum(r["wall_seconds"] for r in legacy),
               "prepared_total_candidate_seconds": sum(r["wall_seconds"] for r in candidate),
               "by_method": {}}
    for method in ("full_write", "replace_text"):
        old = [r["wall_seconds"] for r in legacy if r["method"] == method]
        new = [r["wall_seconds"] for r in candidate if r["method"] == method]
        summary["by_method"][method] = {
            "prepared_median_legacy_seconds": statistics.median(old),
            "prepared_median_candidate_seconds": statistics.median(new),
            "prepared_total_legacy_seconds": sum(old),
            "prepared_total_candidate_seconds": sum(new),
            "repair_only_legacy_median_seconds": statistics.median(
                r["wall_seconds"] for r in legacy
                if r["method"] == method and r["branch"] == "repair"),
            "repair_only_candidate_median_seconds": statistics.median(
                r["wall_seconds"] for r in candidate
                if r["method"] == method and r["branch"] == "repair"),
        }
    (output / "comparison.json").write_text(json.dumps(summary, indent=2,
                                                        sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
