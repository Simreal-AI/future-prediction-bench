"""Real-QEMU correctness probe for terminal verifier VM-process recovery.

The same pinned Boltons repair runs in a control VM and a crash VM. In the
crash arm, the first hidden Python case completes, then a trusted host hook
writes a guest /tmp marker, kills QEMU, and interrupts grading before a case
result can be published. A matching replacement VM restores the submitted
snapshot and reruns all 14 host-private cases. No model inference or optimizer
is involved, and the run is not a throughput comparison.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import future_prediction_bench.microvm_coding as coding_module
import future_prediction_bench.microvm_runtime as runtime_module
import future_prediction_bench.semantic_vm_recovery as recovery_module
from future_prediction_bench.microvm_coding import MicroVMCodingAdapter
from future_prediction_bench.microvm_runtime import MicroVMRuntime, _clone_or_copy_qcow2
from future_prediction_bench.realworld import validate_task
from future_prediction_bench.semantic_vm_recovery import CodingVMRecoveryJournal, RecoveryError

from .benchmark_semantic_recovery import _assets, _required, _runtime, _source_hash
from .check_microvm_branch_env import _fixture_task


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _episode(name, task_source, manifest, actions, task_dir, assets, output,
             memory_mib, *, crash_during_case):
    disk = output / f"{name}.qcow2"
    clone_mode = _clone_or_copy_qcow2(assets / "rootfs.qcow2", disk)
    runtime = _runtime(assets, manifest, disk, memory_mib)
    adapter = MicroVMCodingAdapter(
        runtime, verifier_dir=task_dir / "verifier",
        visible_check=("python3", "-B", "-c", "import boltons.strutils"))
    task = _fixture_task(task_source)
    task.setdefault("metadata", {})["artifact_binding"] = adapter.artifact_binding()
    validate_task(task)
    started = time.monotonic()
    submit_calls = 0
    original_step = adapter.step

    def counted_step(action, *, now):
        nonlocal submit_calls
        if action.get("action") == "submit":
            submit_calls += 1
        return original_step(action, now=now)

    adapter.step = counted_step
    try:
        adapter.reset(task, now=datetime.now(timezone.utc))
        journal = CodingVMRecoveryJournal(adapter, output / f"{name}-journal",
                                          mode="every_turn")
        journal.begin()
        observations = []
        for action in actions[:-1]:
            outcome = journal.apply(action, now=datetime.now(timezone.utc))
            observations.append(_digest(outcome["result"]["observation"]))
        if not crash_during_case:
            terminal = journal.submit_and_verify(now=datetime.now(timezone.utc))
            negative_control = None
            crashed_after_case = False
            marker_removed = None
        else:
            original_case = adapter._run_python_case
            completed_cases = 0

            def kill_after_first_case(argv, *, timeout=None, unprivileged=False):
                nonlocal completed_cases
                result = original_case(argv, timeout=timeout,
                                       unprivileged=unprivileged)
                completed_cases += 1
                if completed_cases == 1:
                    _required(runtime, "printf interrupted > /tmp/fpb-terminal-vm-crash")
                    runtime._process.kill()
                    runtime._process.wait(timeout=5)
                    raise KeyboardInterrupt("injected_qemu_crash_after_first_hidden_case")
                return result

            adapter._run_python_case = kill_after_first_case
            try:
                try:
                    journal.submit_and_verify(now=datetime.now(timezone.utc))
                except KeyboardInterrupt as exc:
                    if str(exc) != "injected_qemu_crash_after_first_hidden_case":
                        raise
                else:
                    raise RuntimeError("VM crash injection did not interrupt grading")
            finally:
                adapter._run_python_case = original_case
            if (completed_cases != 1 or runtime._process.poll() is None
                    or not journal.terminal_attempt_path.is_file()
                    or not journal.terminal_submit_path.is_file()
                    or journal.terminal_result_path.exists()
                    or adapter.verified is not None or submit_calls != 1):
                raise RuntimeError("Crashed verifier released reward or lost submit boundary")
            crashed_after_case = True

            wrong = MicroVMRuntime(
                assets / "vmlinuz-virt", assets / "initramfs-virt", disk,
                kernel_sha256="0" * 64,
                initramfs_sha256=manifest["alpine_sha256"]["initramfs-virt"],
                readonly_disk_paths=(assets / "modloop-virt-padded.raw",),
                memory_mib=memory_mib, command_timeout=30)
            try:
                try:
                    journal.resume_verification_after_vm_crash(
                        wrong, now=datetime.now(timezone.utc))
                except RecoveryError as exc:
                    if str(exc) != "replacement_vm_differs_from_frozen_runtime":
                        raise
                else:
                    raise RuntimeError("Mismatched kernel identity was accepted")
                if wrong._process is not None or journal.terminal_result_path.exists():
                    raise RuntimeError("Mismatched replacement booted or released reward")
            finally:
                wrong.close()
            negative_control = "changed_kernel_sha256_rejected_before_boot"

            replacement = _runtime(assets, manifest, disk, memory_mib)
            terminal = journal.resume_verification_after_vm_crash(
                replacement, now=datetime.now(timezone.utc))
            marker_removed = _required(
                replacement, "test ! -e /tmp/fpb-terminal-vm-crash && printf clean"
            ) == "clean"
            if not marker_removed:
                raise RuntimeError("Interrupted hidden case RAM marker survived restore")
            if journal.submit_and_verify(now=datetime.now(timezone.utc)) != terminal:
                raise RuntimeError("Second terminal call differed from durable result")

        if (submit_calls != 1 or not journal.terminal_result_path.is_file()
                or terminal["grading"].get("status") != "resolved"):
            raise RuntimeError("Terminal reward was not durably resolved once")
        evidence = terminal["grading"].get("evidence", {})
        cases = evidence.get("case_results")
        if (not isinstance(cases, list) or len(cases) != 14
                or not all(case.get("passed") is True for case in cases)
                or terminal["grading"].get("reward") != 1.0):
            raise RuntimeError("Pinned 14-case Boltons verifier changed")
        return {"arm": name, "elapsed_seconds": round(time.monotonic() - started, 6),
                "clone_mode": clone_mode, "submit_calls": submit_calls,
                "terminal_id": terminal["terminal_id"],
                "observations": observations,
                "case_results": copy.deepcopy(cases),
                "case_results_sha256": _digest(cases),
                "reward": terminal["grading"]["reward"],
                "passed_cases": sum(case["passed"] for case in cases),
                "source_sha256": _source_hash(adapter.runtime),
                "crashed_after_first_hidden_case": crashed_after_case,
                "partial_case_marker_removed": marker_removed,
                "negative_control": negative_control}
    finally:
        adapter.close()
        runtime.close()
        disk.unlink(missing_ok=True)


def check(task_dir, assets_dir, output_dir, *, memory_mib=128):
    task_dir, assets, output = (Path(path).resolve() for path in
                                (task_dir, assets_dir, output_dir))
    if type(memory_mib) is not int or not 128 <= memory_mib <= 8192:
        raise ValueError("memory_mib must be 128..8192")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output directory must be empty")
    if (output.is_relative_to(task_dir) or output.is_relative_to(assets)
            or task_dir.is_relative_to(output) or assets.is_relative_to(output)):
        raise ValueError("Output must be disjoint from task and assets")
    task, manifest, actions = _assets(task_dir, assets)
    output.mkdir(parents=True, exist_ok=True)
    control = _episode("control", task, manifest, actions, task_dir, assets,
                       output, memory_mib, crash_during_case=False)
    recovered = _episode("vm-crash", task, manifest, actions, task_dir, assets,
                         output, memory_mib, crash_during_case=True)
    if (control["observations"] != recovered["observations"]
            or control["case_results"] != recovered["case_results"]
            or control["source_sha256"] != recovered["source_sha256"]
            or control["reward"] != recovered["reward"]
            or recovered["submit_calls"] != 1
            or not recovered["crashed_after_first_hidden_case"]
            or not recovered["partial_case_marker_removed"]
            or recovered["negative_control"] != "changed_kernel_sha256_rejected_before_boot"):
        raise RuntimeError("Terminal VM crash changed hidden evidence or reward")
    report = {
        "kind": "terminal_vm_crash_recovery_qemu_correctness_v1",
        "scope": "one pinned public Boltons v2 repair; real QEMU/HVF control and crash arms",
        "uses_actual_qemu": True, "model_or_optimizer_included": False,
        "control": {key: value for key, value in control.items()
                    if key not in {"observations", "case_results"}},
        "recovered": {key: value for key, value in recovered.items()
                      if key not in {"observations", "case_results"}},
        "action_observation_parity": True,
        "all_14_case_evidence_equal": True,
        "reward_equal": True,
        "source_sha256_equal": True,
        "task_sha256": _sha(task_dir / "task.json"),
        "asset_manifest_sha256": _sha(assets / "manifest.json"),
        "source_hashes": {
            "recovery": _sha(recovery_module.__file__),
            "runtime": _sha(runtime_module.__file__),
            "adapter": _sha(coding_module.__file__),
            "benchmark": _sha(Path(__file__)),
        },
        "limitations": [
            "host adapter and verifier survive; only the QEMU process is killed",
            "full-VM verifier mode and one pinned solved fixture",
            "no host power-loss or multi-tenant recovery guarantee",
            "not a matched throughput A/B or RL training measurement",
        ],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n",
                                        encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--memory-mib", type=int, default=128)
    args = parser.parse_args()
    result = check(args.task_dir, args.assets_dir, args.output,
                   memory_mib=args.memory_mib)
    print(json.dumps({"control": {"reward": result["control"]["reward"],
                                  "passed_cases": result["control"]["passed_cases"]},
                      "recovered": {"reward": result["recovered"]["reward"],
                                    "passed_cases": result["recovered"]["passed_cases"]},
                      "all_14_case_evidence_equal": result["all_14_case_evidence_equal"]},
                     indent=2))
