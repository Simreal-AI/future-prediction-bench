"""Real-QEMU host-coordinator process restart after a committed terminal submit.

The parent starts separate Python coordinator processes. The crash coordinator
uses os._exit after its first hidden case, leaving its QEMU child orphaned.
The parent verifies that QEMU owns the expected disk, kills that orphan, then
starts a *new* Python process with a new runtime, adapter, and journal. A
control process runs the same pinned Boltons repair without a crash. This is
a correctness probe, not a throughput or learned-policy measurement.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import future_prediction_bench.microvm_coding as coding_module
import future_prediction_bench.microvm_runtime as runtime_module
import future_prediction_bench.semantic_vm_recovery as recovery_module
from future_prediction_bench.microvm_coding import MicroVMCodingAdapter
from future_prediction_bench.microvm_runtime import _clone_or_copy_qcow2
from future_prediction_bench.realworld import validate_task
from future_prediction_bench.semantic_vm_recovery import (
    CodingVMRecoveryJournal, RecoveryError,
)

from .benchmark_semantic_recovery import _assets, _required, _runtime, _source_hash
from .check_microvm_branch_env import _fixture_task


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load(output, name):
    return json.loads((output / name).read_text(encoding="utf-8"))


def _adapter(task_dir, assets, manifest, disk):
    runtime = _runtime(assets, manifest, disk, 128)
    adapter = MicroVMCodingAdapter(
        runtime, verifier_dir=task_dir / "verifier",
        visible_check=("python3", "-B", "-c", "import boltons.strutils"))
    return runtime, adapter


def _worker(role, task_dir, assets, output):
    manifest = _load(assets, "manifest.json")
    task = _load(output, "frozen-task.json")
    actions = _load(output, "actions.json")
    arm = "control" if role == "control" else "crash"
    disk = output / f"{arm}.qcow2"
    runtime, adapter = _adapter(task_dir, assets, manifest, disk)
    journal = CodingVMRecoveryJournal(adapter, output / f"{arm}-journal",
                                      mode="every_turn")
    started = time.monotonic()
    try:
        if role in {"control", "crash"}:
            adapter.reset(task, now=datetime.now(timezone.utc))
            journal.begin()
            journal.enable_host_restart(task)
            observations = []
            for action in actions[:-1]:
                result = journal.apply(action, now=datetime.now(timezone.utc))
                observations.append(_digest(result["result"]["observation"]))
            _write_json(output / f"{arm}-preterminal.json",
                        {"observations": observations,
                         "source_sha256": _source_hash(runtime)})
            original_step = adapter.step
            submit_calls = 0

            def counted_step(action, *, now):
                nonlocal submit_calls
                if action.get("action") == "submit":
                    submit_calls += 1
                    _write_json(output / f"{arm}-submit-count.json",
                                {"submit_calls": submit_calls})
                return original_step(action, now=now)

            adapter.step = counted_step
            if role == "crash":
                original_case = adapter._run_python_case
                completed = 0

                def die_after_first_case(argv, *, timeout=None, unprivileged=False):
                    nonlocal completed
                    result = original_case(argv, timeout=timeout,
                                           unprivileged=unprivileged)
                    completed += 1
                    if completed == 1:
                        _required(runtime, "printf interrupted > /tmp/fpb-host-crash-case")
                        os._exit(71)
                    return result

                adapter._run_python_case = die_after_first_case
            terminal = journal.submit_and_verify(now=datetime.now(timezone.utc))
            _write_json(output / f"{arm}-result.json",
                        {"reward": terminal["grading"]["reward"],
                         "cases": terminal["grading"]["evidence"]["case_results"],
                         "source_sha256": _source_hash(runtime),
                         "submit_calls": submit_calls,
                         "elapsed_seconds": round(time.monotonic() - started, 6)})
            return 0
        if role == "wrong-kernel":
            runtime.kernel_sha256 = "0" * 64
            try:
                journal.resume_verification_after_host_restart(
                    task, now=datetime.now(timezone.utc))
            except RecoveryError as exc:
                if str(exc) != "host_restart_frozen_identity_mismatch":
                    raise
                if runtime._process is not None or journal.terminal_result_path.exists():
                    raise RuntimeError("Mismatched host restart booted or released reward")
                _write_json(output / "wrong-kernel-result.json",
                            {"rejected_before_boot": True, "reason": str(exc)})
                return 0
            raise RuntimeError("Mismatched kernel was accepted")
        if role == "live-orphan":
            try:
                journal.resume_verification_after_host_restart(
                    task, now=datetime.now(timezone.utc))
            except RecoveryError as exc:
                if str(exc) != "old_host_qemu_still_running":
                    raise
                if runtime._process is not None or journal.terminal_result_path.exists():
                    raise RuntimeError("Live orphan check booted or released reward")
                _write_json(output / "live-orphan-result.json",
                            {"rejected_before_boot": True, "reason": str(exc)})
                return 0
            raise RuntimeError("A second coordinator accepted a live orphan")
        if role == "resume":
            original_step = adapter.step
            submit_calls = 0

            def counted_step(action, *, now):
                nonlocal submit_calls
                if action.get("action") == "submit":
                    submit_calls += 1
                return original_step(action, now=now)

            adapter.step = counted_step
            terminal = journal.resume_verification_after_host_restart(
                task, now=datetime.now(timezone.utc))
            marker_removed = _required(
                runtime, "test ! -e /tmp/fpb-host-crash-case && printf clean") == "clean"
            if not marker_removed or submit_calls != 0:
                raise RuntimeError("Restart repeated submit or kept interrupted case state")
            if journal.submit_and_verify(now=datetime.now(timezone.utc)) != terminal:
                raise RuntimeError("Durable terminal result was not idempotent")
            _write_json(output / "crash-result.json",
                        {"reward": terminal["grading"]["reward"],
                         "cases": terminal["grading"]["evidence"]["case_results"],
                         "source_sha256": _source_hash(runtime),
                         "new_adapter_submit_calls": submit_calls,
                         "partial_case_marker_removed": marker_removed,
                         "elapsed_seconds": round(time.monotonic() - started, 6)})
            return 0
        raise ValueError("Unknown worker role")
    finally:
        # os._exit in the crash role intentionally bypasses this block.
        adapter.close()


def _run_worker(role, task_dir, assets, output, *, expected=0):
    command = [sys.executable, "-m",
               "examples.realworld_boltons26.check_terminal_host_restart_recovery",
               "--role", role,
               "--task-dir", str(task_dir), "--assets-dir", str(assets),
               "--output", str(output)]
    worker = subprocess.Popen(command, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True,
                              start_new_session=True)
    try:
        stdout, stderr = worker.communicate(timeout=240)
    except subprocess.TimeoutExpired:
        # The worker and its QEMU child share a process group. Do not leave
        # an untracked VM holding a writable qcow2 after a probe timeout.
        os.killpg(worker.pid, signal.SIGKILL)
        worker.communicate()
        raise RuntimeError(f"{role} worker timed out; process group killed")
    if worker.returncode != expected:
        raise RuntimeError(f"{role} worker exited {worker.returncode}; "
                           f"stdout={stdout[-2000:]}; "
                           f"stderr={stderr[-4000:]}")


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _kill_verified_orphan(pid, disk):
    if not _pid_alive(pid):
        return False
    command = subprocess.run(["ps", "-ww", "-p", str(pid), "-o", "command="],
                             capture_output=True, text=True, check=False)
    if (command.returncode != 0 or "qemu-system-aarch64" not in command.stdout
            or str(disk) not in command.stdout):
        raise RuntimeError("Recorded orphan PID is not the expected QEMU/disk")
    os.kill(pid, signal.SIGKILL)
    for _ in range(100):
        if not _pid_alive(pid):
            return True
        time.sleep(0.05)
    raise RuntimeError("Orphan QEMU did not exit after SIGKILL")


def check(task_dir, assets_dir, output_dir):
    task_dir, assets, output = (Path(path).resolve() for path in
                                (task_dir, assets_dir, output_dir))
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output directory must be new or empty")
    if (output.is_relative_to(task_dir) or output.is_relative_to(assets)
            or task_dir.is_relative_to(output) or assets.is_relative_to(output)):
        raise ValueError("Task, assets, and output must be disjoint")
    task_source, manifest, actions = _assets(task_dir, assets)
    output.mkdir(parents=True, exist_ok=True)
    clone_modes = {}
    for arm in ("control", "crash"):
        clone_modes[arm] = _clone_or_copy_qcow2(
            assets / "rootfs.qcow2", output / f"{arm}.qcow2")
    _, adapter = _adapter(task_dir, assets, manifest, output / "control.qcow2")
    task = _fixture_task(task_source)
    task.setdefault("metadata", {})["artifact_binding"] = adapter.artifact_binding()
    validate_task(task)
    _write_json(output / "frozen-task.json", task)
    _write_json(output / "actions.json", actions)
    orphan_pid = None
    try:
        _run_worker("control", task_dir, assets, output)
        _run_worker("crash", task_dir, assets, output, expected=71)
        crash_journal = output / "crash-journal"
        context = _load(crash_journal, "host-restart-context.json")
        orphan_pid = context["old_qemu_pid"]
        if (not (crash_journal / "terminal-submit.json").is_file()
                or (crash_journal / "terminal-result.json").exists()
                or _load(output, "crash-submit-count.json")["submit_calls"] != 1):
            raise RuntimeError("Crash occurred outside committed submit boundary")
        # A new host coordinator must refuse to touch a disk still held by
        # the orphan. This negative control runs in another Python process.
        _run_worker("live-orphan", task_dir, assets, output)
        orphan_killed = _kill_verified_orphan(orphan_pid, output / "crash.qcow2")
        _run_worker("wrong-kernel", task_dir, assets, output)
        _run_worker("resume", task_dir, assets, output)
        control, recovered = (_load(output, f"{arm}-result.json")
                              for arm in ("control", "crash"))
        control_pre, crash_pre = (_load(output, f"{arm}-preterminal.json")
                                  for arm in ("control", "crash"))
        if (control["reward"] != recovered["reward"]
                or control["cases"] != recovered["cases"]
                or control["source_sha256"] != recovered["source_sha256"]
                or control_pre != crash_pre
                or control["submit_calls"] != 1
                or recovered["new_adapter_submit_calls"] != 0
                or not recovered["partial_case_marker_removed"]
                or len(control["cases"]) != 14
                or not all(case["passed"] for case in control["cases"])):
            raise RuntimeError("Host restart changed observations, hidden cases, or reward")
        report = {
            "kind": "terminal_host_process_restart_qemu_correctness_v1",
            "scope": "two pinned Boltons v2 arms; separate Python coordinator processes",
            "uses_actual_qemu": True, "model_or_optimizer_included": False,
            "host_process_exited_after_first_hidden_case": True,
            "orphan_qemu_killed_by_supervisor": orphan_killed,
            "live_orphan_rejected_before_boot": True,
            "wrong_kernel_rejected_before_boot": True,
            "durable_submit_marker_before_restart": True,
            "no_result_record_before_restart": True,
            "new_adapter_submit_calls": recovered["new_adapter_submit_calls"],
            "original_adapter_submit_calls": 1,
            "action_observation_parity": True,
            "all_14_case_evidence_equal": True,
            "reward_equal": True,
            "source_sha256_equal": True,
            "partial_case_marker_removed": recovered["partial_case_marker_removed"],
            "reward": recovered["reward"],
            "passed_cases": sum(case["passed"] for case in recovered["cases"]),
            "clone_modes": clone_modes,
            "task_sha256": _sha(task_dir / "task.json"),
            "asset_manifest_sha256": _sha(assets / "manifest.json"),
            "source_hashes": {
                "recovery": _sha(recovery_module.__file__),
                "runtime": _sha(runtime_module.__file__),
                "adapter": _sha(coding_module.__file__),
                "probe": _sha(Path(__file__))},
            "limitations": [
                "one pinned solved fixture and full-VM serial verifier only",
                "a trusted supervisor kills the orphan QEMU before reopening the disk",
                "host power loss, concurrent coordinators, and multi-tenant recovery unproven",
                "not a throughput, millisecond checkpoint, or RL training measurement"],
        }
        _write_json(output / "report.json", report)
        return report
    finally:
        if orphan_pid is not None and _pid_alive(orphan_pid):
            _kill_verified_orphan(orphan_pid, output / "crash.qcow2")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--role", choices=("parent", "control", "crash", "live-orphan", "wrong-kernel", "resume"),
                        default="parent")
    args = parser.parse_args()
    if args.role == "parent":
        result = check(args.task_dir, args.assets_dir, args.output)
        print(json.dumps({key: result[key] for key in (
            "all_14_case_evidence_equal", "reward_equal", "passed_cases",
            "orphan_qemu_killed_by_supervisor")}, indent=2))
    else:
        raise SystemExit(_worker(args.role, Path(args.task_dir).resolve(),
                                 Path(args.assets_dir).resolve(), Path(args.output).resolve()))
