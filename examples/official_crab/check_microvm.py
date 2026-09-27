"""Execute pinned upstream Crab inspection/policy against a real QEMU guest.

No CRIU/ZFS or eBPF backend is substituted or claimed. Component decisions
are promoted to a full QEMU snapshot, with that promotion in the evidence.
"""
from __future__ import annotations

import argparse
import base64
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys
import time

from future_prediction_bench.microvm_runtime import _clone_or_copy_qcow2
from examples.realworld_boltons26.microvm_benchmark import _boot, _required, _runtime
from examples.realworld_boltons26.microvm_template_benchmark import _fixture

COMMIT = "9607d61a41dc44358cf078c4b438bfd971c8ee9d"
MONITOR = "crab/host_inspector/process_monitor.py"
OPERATIONS = [
    {"op": "state"}, {"op": "state"}, {"op": "state"},
    {"op": "transient"}, {"op": "file", "value": "changed"},
    {"op": "state"}, {"op": "memory", "value": 456}, {"op": "state"},
]
EXPECTED_SELECTIVE = [True, False, False, False, True, False, True, False]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_upstream(checkout):
    checkout = Path(checkout).resolve()
    head = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(checkout), "status", "--porcelain"], text=True)
    if head != COMMIT or dirty:
        raise ValueError("Crab requires the pinned, clean upstream checkout")
    if "crab" in sys.modules:
        raise RuntimeError("Load in a fresh Python process to avoid module shadowing")
    sys.path.insert(0, str(checkout))
    scheduler = importlib.import_module("crab.scheduler")
    config = importlib.import_module("crab.config")
    models = importlib.import_module("crab.models")
    if Path(scheduler.__file__).resolve() != checkout / "crab/scheduler.py":
        raise RuntimeError("Crab import origin mismatch")
    policy = scheduler.FaultToleranceCheckpointingPolicy(config.SchedulerConfig(
        min_checkpoint_interval_seconds=0,
        force_checkpoint_after_seconds=0,
        prefer_checkpoint_during_llm_request=False,
    ))
    return policy, models.SandboxSnapshot, {
        "repository": "https://github.com/open-agent-infra/crab",
        "commit": head, "process_monitor_sha256": sha(checkout / MONITOR),
        "scheduler_sha256": sha(checkout / "crab/scheduler.py"),
        "modified_upstream_files": [], "dependency_doubles": [],
    }


def upload(vm, source, target):
    data = Path(source).read_bytes()
    if not 0 < len(data) <= 65536:
        raise ValueError("helper_size_invalid")
    _required(vm, ": > " + shlex.quote(target))
    for offset in range(0, len(data), 2700):
        block = base64.b64encode(data[offset:offset + 2700]).decode()
        _required(vm, "printf '%s' '" + block + "' | base64 -d >> " + shlex.quote(target))
    output = _required(vm, "sha256sum " + shlex.quote(target))
    if output.split()[0] != sha(source):
        raise RuntimeError("uploaded_helper_digest_mismatch")


def call(vm, request):
    encoded = base64.b64encode(json.dumps(request, allow_nan=False).encode()).decode()
    result = _required(vm, "LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/aarch64-linux-gnu:/usr/lib/x86_64-linux-gnu "
        "chroot /mnt/root /usr/local/bin/python3.12 -I -B /fpb_crab_probe.py --call " + encoded)
    matches = re.findall(r"FPB_CRAB_RESULT:([A-Za-z0-9+/=]+)", result)
    if len(matches) != 1:
        raise RuntimeError("crab_probe_result_framing_failed")
    payload = json.loads(base64.b64decode(matches[0], validate=True))
    if "error" in payload and payload.get("known") is not False:
        raise RuntimeError("crab_probe_failed: " + str(payload["error"]))
    return payload


def install(vm, checkout):
    _required(vm, "mkdir -p /mnt/root/proc /mnt/root/dev && "
        "mount -t proc proc /mnt/root/proc && "
        "(test -c /mnt/root/dev/null || mknod -m 666 /mnt/root/dev/null c 1 3)")
    upload(vm, Path(checkout) / MONITOR, "/mnt/root/fpb_crab_process_monitor.py")
    upload(vm, Path(__file__).with_name("guest_probe.py"), "/mnt/root/fpb_crab_probe.py")
    _required(vm, "LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/aarch64-linux-gnu:/usr/lib/x86_64-linux-gnu "
        "chroot /mnt/root /usr/local/bin/python3.12 -I -B /fpb_crab_probe.py --serve "
        "</dev/null >/mnt/root/fpb-crab-probe.log 2>&1 & true")
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if vm.run_shell("test -S /mnt/root/fpb-crab-probe.sock", timeout=5)["return_code"] == 0:
            return call(vm, {"op": "state"})
        time.sleep(0.02)
    raise RuntimeError("crab_probe_not_ready: " + _required(vm, "cat /mnt/root/fpb-crab-probe.log"))


def trial(vm, *, condition, repetition, policy, snapshot_type, packed):
    vm.load_snapshot("crabstart", resume=True)
    records, snapshots = [], []
    last_checkpoint_at = None
    started = time.perf_counter()
    for turn, operation in enumerate(OPERATIONS):
        action_started = time.perf_counter()
        prepared = call(vm, {"op": "step", "action": operation,
                            "inspect": condition == "selective"}) if packed else None
        if not packed:
            call(vm, operation)
        action_seconds = time.perf_counter() - action_started
        inspection_seconds, inspection, decision = 0.0, None, None
        if condition == "selective":
            if packed:
                inspection = prepared["inspection"]
                # Included in packed RPC wall time; no invented host split.
            else:
                inspected = time.perf_counter()
                inspection = call(vm, {"op": "inspect"})
                inspection_seconds = time.perf_counter() - inspected
            decision = policy.evaluate(snapshot_type(
                sandbox_id="microvm-probe", runtime_name="qemu-full-vm",
                is_running=True, process_changed=inspection["process_changed"],
                filesystem_changed=inspection["filesystem_changed"],
                observed_at=datetime.now(timezone.utc), last_checkpoint_at=last_checkpoint_at,
            ))
            should_save = decision.should_checkpoint
            if packed and not should_save and (not inspection["known"] or
                    inspection["process_changed"] or inspection["filesystem_changed"]):
                raise RuntimeError("speculative_baseline_requires_save_or_known_clean_skip")
        else:
            should_save = True
        save_seconds, baseline_seconds = 0.0, 0.0
        tag = None
        if should_save:
            # Reset only at a commit boundary. Any failed save aborts the run;
            # its cleared dirty bits can never authorize a subsequent skip.
            if not packed:
                baseline_started = time.perf_counter()
                call(vm, {"op": "baseline"})
                baseline_seconds = time.perf_counter() - baseline_started
            tag = f"{condition[0]}{repetition}t{turn}"
            saving = time.perf_counter()
            vm.save_snapshot(tag)
            save_seconds = time.perf_counter() - saving
            last_checkpoint_at = datetime.now(timezone.utc)
        state = prepared["state"] if packed else call(vm, {"op": "state"})
        if tag:
            snapshots.append((tag, state))
        records.append({"turn": turn, "operation": operation, "saved": should_save,
            "action_seconds": action_seconds, "inspection_seconds": inspection_seconds,
            "packed_rpc_includes_action_inspection_state_and_baseline": packed,
            "baseline_seconds": baseline_seconds, "save_seconds": save_seconds,
            "inspection": inspection, "decision": asdict(decision) if decision else None,
            "effective_backend": "full_vm_state_qcow2_v1" if should_save else None,
            "partial_decision_promoted_to_full_vm": bool(decision and should_save and
                not (decision.checkpoint_process and decision.checkpoint_filesystem)),
            "state": state})
    elapsed = time.perf_counter() - started
    if condition == "selective" and [r["saved"] for r in records] != EXPECTED_SELECTIVE:
        raise RuntimeError("unexpected_official_checkpoint_decisions: " + json.dumps([
            {"turn": r["turn"], "saved": r["saved"], "inspection": r["inspection"]}
            for r in records]))
    # Test each committed state by corrupting BOTH private RAM and disk first.
    # These correctness challenges are outside the timed capture workload.
    restored = []
    for tag, expected in snapshots:
        call(vm, {"op": "memory", "value": 999})
        call(vm, {"op": "file", "value": "corrupted"})
        restore_started = time.perf_counter()
        vm.load_snapshot(tag, resume=True)
        restore_seconds = time.perf_counter() - restore_started
        actual = call(vm, {"op": "state"})
        if actual != expected:
            raise RuntimeError("checkpoint_ram_disk_restore_mismatch")
        restored.append({"tag": tag, "restore_seconds": restore_seconds,
                         "ram_and_disk_match": True})
    # A missing workload must force conservative inspection, not a clean skip.
    call(vm, {"op": "kill_worker"})
    missing = call(vm, {"op": "inspect"})
    if missing["known"] or not missing["process_changed"] or not missing["filesystem_changed"]:
        raise RuntimeError("missing_worker_did_not_fail_closed")
    vm.load_snapshot(snapshots[-1][0], resume=True)
    if call(vm, {"op": "state"}) != snapshots[-1][1]:
        raise RuntimeError("killed_worker_restore_mismatch")
    # Remove measured tags outside timing to avoid growing qcow2 metadata
    # across alternating conditions. The start snapshot remains available.
    for tag, _ in snapshots:
        vm._hmp("delvm " + tag)
        if vm._snapshot_listed(tag, vm._hmp("info snapshots")):
            raise RuntimeError("trial_snapshot_cleanup_failed")
        vm._snapshots.discard(tag)
    return {"condition": condition, "repetition": repetition,
        "workload_seconds": elapsed, "checkpoint_count": len(snapshots),
        "save_seconds": sum(r["save_seconds"] for r in records),
        "inspection_seconds": sum(r["inspection_seconds"] for r in records),
        "guest_inspection_ns": [r["inspection"]["inspection_ns"] for r in records if r["inspection"]],
        "records": records, "restore_checks": restored,
        "missing_worker_fail_closed": True, "killed_worker_restored": True}


def check(checkout, task_dir, assets_dir, output, *, repetitions=3, x86_tcg=False, packed=True):
    if type(repetitions) is not int or not 1 <= repetitions <= 10:
        raise ValueError("repetitions must be in [1,10]")
    checkout, task_dir, assets, output = [Path(p).resolve() for p in
        (checkout, task_dir, assets_dir, output)]
    if output.exists() or any(output.is_relative_to(p) or p.is_relative_to(output)
        for p in (checkout, task_dir, assets)):
        raise ValueError("output must be new and disjoint from all inputs")
    policy, snapshot_type, upstream = load_upstream(checkout)
    base, _, _, _ = _fixture(task_dir, assets)
    output.mkdir(parents=True)
    disk = output / "probe.qcow2"
    _clone_or_copy_qcow2(base, disk)
    runs = []
    setup = time.perf_counter()
    if x86_tcg:
        from .x86_runtime import runtime, boot
        vm_context, boot_guest = runtime(assets, disk), boot
    else:
        vm_context, boot_guest = _runtime(assets, disk), _boot
    with vm_context as vm:
        boot_guest(vm)
        initial = install(vm, checkout)
        capabilities = call(vm, {"op": "capabilities"})
        if not capabilities["soft_dirty_known_write_detected"]:
            failure = {"schema_version": "official-crab-capability-probe-v1",
                "upstream": upstream, "capabilities": capabilities,
                "benchmark_started": False, "reason": "soft_dirty_calibration_failed"}
            (output / "capability_failure.json").write_text(json.dumps(failure, indent=2) + "\n")
            raise RuntimeError("guest_soft_dirty_unsupported: benchmark aborted before any skip")
        if initial["value"] != 123 or initial["file"] != "seed":
            raise RuntimeError("initial_state_invalid")
        vm.save_snapshot("crabstart")
        setup_seconds = time.perf_counter() - setup
        for repetition in range(repetitions):
            conditions = ["every_turn", "selective"] if repetition % 2 == 0 else ["selective", "every_turn"]
            for condition in conditions:
                runs.append(trial(vm, condition=condition, repetition=repetition,
                                  policy=policy, snapshot_type=snapshot_type, packed=packed))
    summary = {condition: {
        "checkpoint_count_per_run": [r["checkpoint_count"] for r in runs if r["condition"] == condition],
        "median_workload_seconds": statistics.median(r["workload_seconds"] for r in runs if r["condition"] == condition),
        "median_save_seconds": statistics.median(r["save_seconds"] for r in runs if r["condition"] == condition),
    } for condition in ("every_turn", "selective")}
    result = {"schema_version": "official-crab-microvm-probe-v2", "upstream": upstream,
        "guest_probe_sha256": sha(Path(__file__).with_name("guest_probe.py")),
        "scope": "eight-action controlled stopped-worker RAM and single-file state probe",
        "graded_software_episode": False, "model_inference": False, "optimizer": False,
        "criu_zfs_backend": False, "ebpf_filesystem_monitor": False,
        "incremental_process_checkpoint": False, "setup_seconds_excluded": setup_seconds,
        "capabilities": capabilities, "qemu_acceleration": "tcg" if x86_tcg else "hvf",
        "packed_rpc": packed, "packed_transport_identical_in_both_conditions": True,
        "repetitions": repetitions, "summary": summary, "runs": runs}
    (output / "measurement.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    # Trial disk is disposable; retain evidence, never embed it in releases.
    disk.unlink()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--x86-tcg", action="store_true")
    parser.add_argument("--legacy-rpc", action="store_true")
    args = parser.parse_args()
    result = check(args.checkout, args.task_dir, args.assets, args.output,
                   repetitions=args.repetitions, x86_tcg=args.x86_tcg, packed=not args.legacy_rpc)
    print(json.dumps({"summary": result["summary"], "upstream": result["upstream"]}, indent=2))
