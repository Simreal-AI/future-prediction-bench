"""A/B real-QEMU turn-boundary checkpoint overlap with a simulated policy wait.

The wait models policy inference latency; no model runs in this benchmark. The
serial and overlap arms execute identical pinned Boltons actions, full QEMU
checkpoints, and 14 host-private grading cases. Each arm owns a separate qcow2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from future_prediction_bench.async_vm_checkpoint import (
    AsyncVMCheckpointCoordinator, PolicyChoice,
)
from future_prediction_bench.microvm_coding import MicroVMCodingAdapter
from future_prediction_bench.microvm_runtime import _clone_or_copy_qcow2
from future_prediction_bench.realworld import validate_task

from .benchmark_semantic_recovery import _assets, _runtime, _source_hash
from .check_microvm_branch_env import _fixture_task


def _episode(name, mode, task_source, manifest, actions, task_dir, assets,
             output, memory_mib, wait_seconds):
    disk = output / f"{name}.qcow2"
    clone_mode = _clone_or_copy_qcow2(assets / "rootfs.qcow2", disk)
    runtime = _runtime(assets, manifest, disk, memory_mib)
    adapter = MicroVMCodingAdapter(
        runtime, verifier_dir=task_dir / "verifier",
        visible_check=("python3", "-B", "-c", "import boltons.strutils"))
    task = _fixture_task(task_source)
    task.setdefault("metadata", {})["artifact_binding"] = adapter.artifact_binding()
    validate_task(task)
    coordinator = AsyncVMCheckpointCoordinator(
        adapter, output / f"{name}-journal", mode=mode)
    observations = []
    turns = []
    started = time.monotonic()
    try:
        adapter.reset(task, now=datetime.now(timezone.utc))
        coordinator.begin()
        for action, next_action in zip(actions[:-1], actions[1:]):
            def policy(prompt, next_action=next_action):
                # The callback receives only immutable, detached JSON. The
                # digest is echoed so the release gate binds the chosen next
                # action to the exact observation of this completed turn.
                json.loads(prompt.observation_json)
                time.sleep(wait_seconds)
                return PolicyChoice(prompt.observation_sha256, next_action)

            outcome = coordinator.run_turn(action, policy,
                                           now=datetime.now(timezone.utc))
            observations.append(outcome["observation_sha256"])
            turns.append(outcome["measurements"])
        before_submit = time.monotonic() - started
        terminal = coordinator.submit_and_verify(now=datetime.now(timezone.utc))
        elapsed = time.monotonic() - started
        graded = terminal["grading"]
        if (terminal["submission"]["observation"].get("status") != "submitted"
                or graded.get("status") != "resolved"):
            raise RuntimeError("Pinned QEMU grading did not resolve")
        cases = graded["evidence"]["case_results"]
        if len(cases) != 14:
            raise RuntimeError("Pinned hidden case count changed")
        return {
            "name": name, "mode": mode, "clone_mode": clone_mode,
            "simulated_policy_wait_seconds_per_turn": wait_seconds,
            "whole_graded_episode_seconds": elapsed,
            "through_committed_turns_seconds": before_submit,
            "reward": graded["reward"],
            "passed_cases": sum(case["passed"] for case in cases),
            "case_count": len(cases), "case_results": cases,
            "observations": observations,
            "final_source_sha256": _source_hash(runtime),
            "turns": turns,
            "journal_snapshot_saves": coordinator.journal.metrics["snapshot_saves"],
            "qemu_snapshot_saves": runtime.metrics["snapshot_saves"],
        }
    finally:
        # The worker must have completed before the VM lease is released.
        coordinator.close()
        adapter.close()
        disk.unlink(missing_ok=True)


def benchmark(task_dir, assets_dir, output_dir, *, wait_ms=100.0,
              pairs=2, memory_mib=128):
    task_dir, assets, output = (Path(path).resolve() for path in
                                (task_dir, assets_dir, output_dir))
    if (isinstance(wait_ms, bool) or not isinstance(wait_ms, (int, float))
            or not 0 <= wait_ms <= 5000):
        raise ValueError("wait_ms must be 0..5000")
    if type(pairs) is not int or not 1 <= pairs <= 10:
        raise ValueError("pairs must be 1..10")
    if type(memory_mib) is not int or not 128 <= memory_mib <= 8192:
        raise ValueError("memory_mib must be 128..8192")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output directory must be empty")
    if (output.is_relative_to(task_dir) or output.is_relative_to(assets)
            or task_dir.is_relative_to(output) or assets.is_relative_to(output)):
        raise ValueError("Output must be disjoint from task and assets")
    task, manifest, actions = _assets(task_dir, assets)
    output.mkdir(parents=True, exist_ok=True)
    runs = []
    for pair in range(pairs):
        order = ("serial", "overlap") if pair % 2 == 0 else ("overlap", "serial")
        for mode in order:
            name = f"pair-{pair + 1}-{mode}"
            run = _episode(name, mode, task, manifest, actions, task_dir, assets,
                           output, memory_mib, wait_ms / 1000)
            runs.append(run)
            print(f"{name}: graded={run['whole_graded_episode_seconds']:.3f}s "
                  f"reward={run['reward']}", flush=True)
    first = runs[0]
    if (first["reward"] != 1.0 or first["passed_cases"] != 14
            or any(run["reward"] != first["reward"]
                   or run["case_results"] != first["case_results"]
                   or run["observations"] != first["observations"]
                   or run["final_source_sha256"] != first["final_source_sha256"]
                   or run["journal_snapshot_saves"] != first["journal_snapshot_saves"]
                   or run["qemu_snapshot_saves"] != first["qemu_snapshot_saves"]
                   for run in runs)):
        raise RuntimeError("Serial/overlap action, checkpoint, or grade parity failed")
    serial = [run for run in runs if run["mode"] == "serial"]
    overlap = [run for run in runs if run["mode"] == "overlap"]
    report = {
        "scope": "pinned public Boltons v2 scripted real-QEMU/HVF checkpoint overlap",
        "uses_actual_qemu_savevm": True,
        "policy_wait_kind": "time.sleep surrogate; no model, inference, GPU, or optimizer",
        "memory_mib": memory_mib, "pairs": pairs, "wait_ms": wait_ms,
        "all_observation_parity": True, "all_hidden_case_parity": True,
        "all_reward_parity": True, "all_checkpoint_count_parity": True,
        "serial_median_graded_seconds": statistics.median(
            run["whole_graded_episode_seconds"] for run in serial),
        "overlap_median_graded_seconds": statistics.median(
            run["whole_graded_episode_seconds"] for run in overlap),
        "paired_graded_deltas_seconds": [
            serial[index]["whole_graded_episode_seconds"]
            - overlap[index]["whole_graded_episode_seconds"]
            for index in range(pairs)],
        "serial_checkpoint_critical_path_seconds": sum(
            turn["exposed_checkpoint_gate_seconds"]
            for run in serial for turn in run["turns"]),
        "overlap_exposed_gate_seconds": sum(
            turn["exposed_checkpoint_gate_seconds"]
            for run in overlap for turn in run["turns"]),
        "overlap_hidden_checkpoint_seconds": sum(
            turn["checkpoint_policy_overlap_seconds"]
            for run in overlap for turn in run["turns"]),
        "runs": [{key: value for key, value in run.items()
                  if key not in {"case_results", "observations"}}
                 for run in runs],
        "limitations": [
            "One trusted, exclusively owned QEMU adapter per arm",
            "Full QEMU checkpoint per turn, not Crab's eBPF/ZFS/CRIU classifier/backend",
            "Injected policy wait is not evidence of model-training throughput",
            "Initial VM boot, initial checkpoint, and host-private grading remain serial",
            "No host power-loss atomicity or agent-process/event-log recovery",
        ],
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["report_sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wait-ms", type=float, default=100.0)
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--memory-mib", type=int, default=128)
    args = parser.parse_args()
    summary = benchmark(args.task_dir, args.assets_dir, args.output,
                        wait_ms=args.wait_ms, pairs=args.pairs,
                        memory_mib=args.memory_mib)
    print(json.dumps({key: summary[key] for key in (
        "serial_median_graded_seconds", "overlap_median_graded_seconds",
        "paired_graded_deltas_seconds", "overlap_exposed_gate_seconds",
        "report_sha256")}, indent=2))
