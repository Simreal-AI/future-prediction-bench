"""Compare every-turn and conservative Crab-inspired QEMU recovery.

Runs a pinned, already-solved Boltons repair in real ARM64 Linux VMs.  It
checks per-action observations, hidden-case scores, full-VM restoration of a
guest RAM marker and live process, and a coordinator failure injected after
`savevm` but before the durable manifest commit.  No model or optimizer runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from future_prediction_bench.coding_env import _workspace_digest
from future_prediction_bench.microvm_coding import MicroVMCodingAdapter
from future_prediction_bench.microvm_runtime import (
    MicroVMRuntime, _clone_or_copy_qcow2, _sha256_file,
)
from future_prediction_bench.realworld import validate_task
from future_prediction_bench.semantic_vm_recovery import (
    CodingVMRecoveryJournal, InjectedAfterSave, RecoveryError,
)

from .check_microvm_branch_env import _fixture_task


def _required(runtime, command):
    result = runtime.run_shell(command, timeout=30)
    if result["return_code"]:
        raise RuntimeError("trusted guest state check failed")
    return result["stdout"]


def _assets(task_dir, assets):
    task = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    manifest = json.loads((assets / "manifest.json").read_text(encoding="utf-8"))
    if (not task.get("is_fixture")
            or manifest.get("schema_version") != "boltons-microvm-assets-v2"
            or manifest.get("task_id") != task.get("task_id")
            or manifest.get("source_sdist_sha256") != task.get("metadata", {}).get("source_sdist_sha256")
            or manifest.get("seed_workspace_sha256") != _workspace_digest(task_dir / "seed")
            or _sha256_file(assets / "rootfs.qcow2") != manifest.get("rootfs_qcow2_sha256")
            or _sha256_file(assets / "modloop-virt-padded.raw") != manifest.get("modloop_disk_sha256")):
        raise ValueError("Expected exact pinned v2 Boltons guest assets")
    actions = [json.loads(line) for line in
               (task_dir / "actions.solution.replace_text.jsonl").read_text(
                   encoding="utf-8").splitlines() if line]
    if ([action.get("action") for action in actions]
            != ["read_file", "replace_text", "run_visible_checks", "submit"]):
        raise ValueError("Pinned exact-edit actions differ")
    return task, manifest, actions


def _runtime(assets, manifest, disk, memory_mib):
    return MicroVMRuntime(
        assets / "vmlinuz-virt", assets / "initramfs-virt", disk,
        kernel_sha256=manifest["alpine_sha256"]["vmlinuz-virt"],
        initramfs_sha256=manifest["alpine_sha256"]["initramfs-virt"],
        readonly_disk_paths=(assets / "modloop-virt-padded.raw",),
        memory_mib=memory_mib, command_timeout=30)


def _source_hash(runtime):
    output = _required(runtime, "sha256sum /mnt/root/workspace/boltons/strutils.py")
    return output.split()[0]


def _episode(name, mode, task_source, manifest, actions, task_dir, assets,
             output, memory_mib, *, inject_fault=False, check_process=False,
             interrupt_terminal=False):
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
    original_sha = hashlib.sha256((task_dir / "seed" / "boltons" / "strutils.py").read_bytes()).hexdigest()
    observations = []
    decisions = []
    fault_verified = False
    process_restored = False
    ram_restored = False
    vm_process_crash_restored = False
    terminal_interruption_verified = False
    prior_journal_metrics = {}
    prior_vm_snapshot_saves = 0
    prior_vm_snapshot_loads = 0
    try:
        adapter.reset(task, now=datetime.now(timezone.utc))
        journal = CodingVMRecoveryJournal(adapter, output / f"{name}-journal", mode=mode)
        journal.begin()
        # Repeated explicit read/list turns provide a meaningful no-effect
        # opportunity.  A later source repair forces full CPU/RAM/disk save.
        sequence = [actions[0], {"action": "list_files"}, actions[0],
                    actions[1], actions[0], {"action": "list_files"}, actions[2]]
        for index, action in enumerate(sequence):
            if inject_fault and index == 3:
                manifest_before = journal.path.read_bytes()
                try:
                    journal.apply(action, now=datetime.now(timezone.utc),
                                  inject_after_save=True)
                except InjectedAfterSave:
                    pass
                else:
                    raise RuntimeError("Failure injection failed to fire")
                if journal.path.read_bytes() != manifest_before:
                    raise RuntimeError("Uncommitted checkpoint became visible")
                try:
                    journal.submit_and_verify(now=datetime.now(timezone.utc))
                except RecoveryError as exc:
                    if str(exc) != "recover_before_next_turn":
                        raise
                else:
                    raise RuntimeError("Uncommitted state was allowed to receive reward")
                # A fresh coordinator instance must discover only the old
                # committed tag and replay its earlier safe reads.
                prior_journal_metrics = dict(journal.metrics)
                journal = CodingVMRecoveryJournal(adapter, journal.directory, mode=mode)
                journal.recover(now=datetime.now(timezone.utc))
                if _source_hash(runtime) != original_sha:
                    raise RuntimeError("Orphaned edit survived recovery")
                fault_verified = True
            outcome = journal.apply(action, now=datetime.now(timezone.utc))
            observations.append(hashlib.sha256(json.dumps(
                outcome["result"]["observation"], sort_keys=True, separators=(",", ":"),
                ensure_ascii=False).encode()).hexdigest())
            decisions.append({"action": action["action"],
                              "checkpointed": outcome["checkpointed"]})
        # Rewind a committed checkpoint and replay only the skipped reads.
        source_before = _source_hash(runtime)
        recovery = journal.recover(now=datetime.now(timezone.utc))
        if _source_hash(runtime) != source_before:
            raise RuntimeError("Source changed across recovery")
        if check_process:
            journal.apply_opaque(lambda: _required(runtime,
                "printf 'checkpointed' > /tmp/fpb-recovery-ram; "
                "sleep 120 </dev/null >/dev/null 2>&1 & true"))
            _required(runtime, "pidof sleep")
            _required(runtime, "printf 'wrong' > /tmp/fpb-recovery-ram; killall sleep")
            # Kill the QEMU process itself, keeping the committed qcow2 on
            # disk.  A fresh QEMU starts paused and loads the committed tag.
            runtime._process.kill()
            runtime._process.wait(timeout=5)
            prior_vm_snapshot_saves += runtime.metrics["snapshot_saves"]
            prior_vm_snapshot_loads += runtime.metrics["snapshot_loads"]
            replacement = _runtime(assets, manifest, disk, memory_mib)
            journal.recover_after_vm_crash(replacement, now=datetime.now(timezone.utc))
            runtime = replacement
            vm_process_crash_restored = True
            ram_restored = _required(runtime, "cat /tmp/fpb-recovery-ram") == "checkpointed"
            process_restored = bool(_required(runtime, "pidof sleep").strip())
            if not ram_restored or not process_restored:
                raise RuntimeError("Full VM did not restore RAM/process state")
        if interrupt_terminal:
            original_case_runner = adapter._run_python_case
            hidden_calls = 0

            def interrupted_case_runner(argv, *, timeout=None, unprivileged=False):
                nonlocal hidden_calls
                result = original_case_runner(argv, timeout=timeout,
                                              unprivileged=unprivileged)
                hidden_calls += 1
                if hidden_calls == 1:
                    _required(runtime, "printf interrupted > /tmp/fpb-terminal-interrupted")
                    raise KeyboardInterrupt("injected_after_first_real_hidden_case")
                return result

            adapter._run_python_case = interrupted_case_runner
            try:
                try:
                    journal.submit_and_verify(now=datetime.now(timezone.utc))
                except KeyboardInterrupt as exc:
                    if str(exc) != "injected_after_first_real_hidden_case":
                        raise
                else:
                    raise RuntimeError("Terminal verifier interruption did not fire")
            finally:
                adapter._run_python_case = original_case_runner
            if (hidden_calls != 1 or not journal.terminal_attempt_path.is_file()
                    or not journal.terminal_submit_path.is_file()
                    or journal.terminal_result_path.exists() or adapter.verified is not None):
                raise RuntimeError("Interrupted verifier released terminal reward")
            _required(runtime, "test -e /tmp/fpb-terminal-interrupted")
            terminal = journal.resume_verification(now=datetime.now(timezone.utc))
            _required(runtime, "test ! -e /tmp/fpb-terminal-interrupted")
            if journal.submit_and_verify(now=datetime.now(timezone.utc)) != terminal:
                raise RuntimeError("Duplicate terminal call changed the resolved grade")
            terminal_interruption_verified = True
        else:
            terminal = journal.submit_and_verify(now=datetime.now(timezone.utc))
        submit = terminal["submission"]
        if submit["observation"].get("status") != "submitted":
            raise RuntimeError("Repair submission failed")
        graded = terminal["grading"]
        if graded.get("status") != "resolved":
            raise RuntimeError("Hidden grading remained pending")
        cases = graded["evidence"]["case_results"]
        if len(cases) != 14:
            raise RuntimeError("Pinned task no longer has 14 cases")
        return {"name": name, "mode": mode, "fault_injected": inject_fault,
                "clone_mode": clone_mode, "elapsed_seconds": time.monotonic() - started,
                "reward": graded["reward"], "passed_cases": sum(item["passed"] for item in cases),
                "case_count": len(cases), "case_results": cases,
                "observation_sha256s": observations,
                "final_source_sha256": _source_hash(runtime), "decisions": decisions,
                "recovery": recovery, "fault_recovery_verified": fault_verified,
                "process_restored": process_restored, "ram_restored": ram_restored,
                "vm_process_crash_restored": vm_process_crash_restored,
                "terminal_interruption_verified": terminal_interruption_verified,
                "journal_metrics": {key: prior_journal_metrics.get(key, 0) + value
                                    for key, value in journal.metrics.items()},
                "vm_snapshot_saves": (prior_vm_snapshot_saves
                                      + runtime.metrics["snapshot_saves"]),
                "vm_snapshot_loads": (prior_vm_snapshot_loads
                                      + runtime.metrics["snapshot_loads"])}
    finally:
        adapter.close()
        disk.unlink(missing_ok=True)


def benchmark(task_dir, assets_dir, output_dir, *, memory_mib=128,
              terminal_interruption=False):
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
    runs = []
    arms = [
        ("every-turn", "every_turn", False, False),
        ("selective", "selective", False, False),
        ("selective-fault", "selective", True, True),
    ]
    if terminal_interruption:
        arms.append(("selective-terminal-interrupted", "selective", False, False))
    for name, mode, fault, process in arms:
        run = _episode(name, mode, task, manifest, actions, task_dir, assets,
                       output, memory_mib, inject_fault=fault, check_process=process,
                       interrupt_terminal=name == "selective-terminal-interrupted")
        runs.append(run)
        print(f"{name}: reward={run['reward']}, snapshots={run['journal_metrics']['snapshot_saves']}", flush=True)
    control, selective, fault = runs[:3]
    interrupted = runs[3] if terminal_interruption else None
    if len({control["reward"], selective["reward"], fault["reward"]}) != 1:
        raise RuntimeError("Checkpoint strategies gave different rewards")
    if (control["reward"] != 1.0 or any(run["passed_cases"] != 14 for run in runs)
            or control["observation_sha256s"] != selective["observation_sha256s"]
            or control["observation_sha256s"] != fault["observation_sha256s"]
            or control["case_results"] != selective["case_results"]
            or control["case_results"] != fault["case_results"]
            or len({run["final_source_sha256"] for run in runs}) != 1
            or not fault["fault_recovery_verified"]
            or not fault["process_restored"] or not fault["ram_restored"]
            or not fault["vm_process_crash_restored"]):
        raise RuntimeError("Recovery parity or injected failure invariant failed")
    if interrupted is not None and (
            not interrupted["terminal_interruption_verified"]
            or interrupted["reward"] != control["reward"]
            or interrupted["passed_cases"] != 14
            or interrupted["observation_sha256s"] != control["observation_sha256s"]
            or interrupted["case_results"] != control["case_results"]
            or interrupted["final_source_sha256"] != control["final_source_sha256"]):
        raise RuntimeError("Terminal interruption changed hidden-case or reward parity")
    summary = {"scope": "pinned public Boltons v2 scripted QEMU/HVF recovery experiment",
               "memory_mib": memory_mib, "uses_actual_qemu": True,
               "model_or_optimizer_included": False,
               "arms": [{k: v for k, v in run.items()
                         if k not in {"observation_sha256s", "case_results"}}
                        for run in runs],
               "observation_parity": True, "all_hidden_case_parity": True,
               "terminal_interruption_parity": (
                   interrupted["terminal_interruption_verified"]
                   if interrupted is not None else None),
               "full_vm_checkpoint_saves_avoided": (
                   control["journal_metrics"]["snapshot_saves"]
                   - selective["journal_metrics"]["snapshot_saves"]),
               "elapsed_ratio_control_over_selective": (
                   control["elapsed_seconds"] / selective["elapsed_seconds"]),
               "recovery_scope": "sandbox snapshot plus replayed read observations; not RealWorldEnv event-log recovery",
               "limitations": ["trusted bounded tools and exclusive coordinator only",
                               "not eBPF, ZFS, CRIU, or arbitrary process diff",
                               "no host power-loss durability guarantee"]}
    (output / "report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--memory-mib", type=int, default=128)
    parser.add_argument("--terminal-interruption", action="store_true")
    args = parser.parse_args()
    result = benchmark(args.task_dir, args.assets_dir, args.output,
                       memory_mib=args.memory_mib,
                       terminal_interruption=args.terminal_interruption)
    print(json.dumps({"saves_avoided": result["full_vm_checkpoint_saves_avoided"],
                      "elapsed_ratio": result["elapsed_ratio_control_over_selective"]}, indent=2))
