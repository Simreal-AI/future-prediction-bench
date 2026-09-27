"""Compare opt-in guest stateless cases against exact full-VM case restore.

The same pinned Boltons code and 14 host-private outputs are used in both
conditions. This is a scripted integration fixture, not learned-policy RL.
The fast path is available only through the explicit task-bound contract.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import statistics
import time
from pathlib import Path

from future_prediction_bench.coding_env import _workspace_digest
from future_prediction_bench.microvm_runtime import _clone_or_copy_qcow2
from future_prediction_bench.http import strict_json_loads
from future_prediction_bench.stateless_verifier import GUEST_ENV, GUEST_PROGRAM, GUEST_PYTHON, StatelessCaseVerifier

from .microvm_benchmark import _boot, _grade, _patch, _required, _runtime, _sha


def _probe_codes():
    first_code = (
        "from pathlib import Path\n"
        "Path('.fpb-case-overlay-marker').write_text('first')\n"
        "Path('/tmp/fpb-case-tmp-marker').write_text('first')\n"
        "blocked=[]\n"
        "for name in ('/workspace/fpb-case-direct-marker', "
        "'/var/tmp/fpb-case-var-marker', '/dev/shm/fpb-case-shm-marker'):\n"
        "    try:\n"
        "        Path(name).write_text('first')\n"
        "    except OSError:\n"
        "        blocked.append(name)\n"
        "print('DIRECT_DENIED' if len(blocked)==3 else 'DIRECT_WRITABLE')\n")
    second_code = (
        "from pathlib import Path\n"
        "print(Path('.fpb-case-overlay-marker').exists(), "
        "Path('/tmp/fpb-case-tmp-marker').exists(), "
        "Path('/workspace/fpb-case-direct-marker').exists(), "
        "Path('/var/tmp/fpb-case-var-marker').exists(), "
        "Path('/dev/shm/fpb-case-shm-marker').exists(), "
        "Path('/proc/self').exists())\n")
    sleeper_code = (
        "import subprocess,sys\n"
        "p=subprocess.Popen([sys.executable,'-B','-c',"
        "'import time; time.sleep(30)'],stdin=subprocess.DEVNULL,"
        "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)\n"
        "print('SPAWNED' if p.poll() is None else 'FAILED')\n")
    return first_code, second_code, sleeper_code


def _check_probe_outputs(outputs, vm):
    expected = (b"DIRECT_DENIED\n", b"False False False False False False\n", b"SPAWNED\n")
    if (len(outputs) != 3 or any(item["return_code"] != 0 or item["truncated"]
                                 or item["stdout_bytes"] != wanted
                                 for item, wanted in zip(outputs, expected, strict=True))):
        raise RuntimeError("Stateless contamination or detached-process probe failed")
    _required(vm, "test ! -e /mnt/root/workspace/.fpb-case-overlay-marker && "
              "test ! -e /mnt/root/workspace/fpb-case-direct-marker && "
              "test ! -e /mnt/root/tmp/fpb-case-tmp-marker && "
              "test ! -e /mnt/root/var/tmp/fpb-case-var-marker && "
              "test ! -e /mnt/root/dev/shm/fpb-case-shm-marker")
    # The namespace PID 1 exits after each case; Linux must kill even a
    # detached candidate grandchild before the next host command completes.
    remaining = _required(vm, "for p in /proc/[0-9]*/comm; do "
                          "read -r c < \"$p\" || continue; "
                          "[ \"$c\" = python3.12 ] && printf 'LIVE\\n'; "
                          "done; true")
    if remaining.strip():
        raise RuntimeError("Detached candidate process survived stateless case")
    return {"overlay_marker_absent_next_case": True,
            "private_tmp_marker_absent_next_case": True,
            "direct_lower_workspace_write_denied": True,
            "other_global_writable_paths_denied": True,
            "candidate_procfs_not_mounted": True,
            "lower_workspace_unchanged_after_cases": True,
            "detached_candidate_process_cleaned": True}


def _contamination_probe(verifier, vm, *, batch=False):
    codes = _probe_codes()
    if batch:
        verifier.prepare_batch_codes(codes)
        outputs = verifier.run_batch(expected_count=3)
    else:
        outputs = [verifier.run_case(["python3", "-B", "-c", code]) for code in codes]
    return _check_probe_outputs(outputs, vm)


def _assert_parity(full, stateless, expected_reward):
    if (full["reward"] != expected_reward or stateless["reward"] != expected_reward
            or full["case_count"] != 14 or stateless["case_count"] != 14
            or full["case_results"] != stateless["case_results"]):
        raise RuntimeError("Full-VM and stateless verifier outputs differ")


def _diagnose_guest_cases(verifier, vm, full_repaired):
    """Run after timing; no expected output is copied into the guest."""
    source = GUEST_PROGRAM.with_name("guest_stateless_profile.py").read_bytes()
    if not 0 < len(source) <= 30_000:
        raise ValueError("Trusted guest diagnostic exceeds upload bound")
    target = "/mnt/root/fpb_guest_stateless_profile.py"
    verifier._required(f": > {target} && chmod 600 {target}")
    for offset in range(0, len(source), 1500):
        encoded = base64.b64encode(source[offset:offset + 1500]).decode("ascii")
        verifier._required(f"printf '%s' '{encoded}' | base64 -d >> {target}")
    digest = hashlib.sha256(source).hexdigest()
    if verifier._required(f"sha256sum {target}").split()[0] != digest:
        raise RuntimeError("Guest diagnostic digest differs")
    started = time.monotonic()
    output = verifier._required(
        f"{GUEST_ENV} chroot /mnt/root {GUEST_PYTHON} -I -B "
        "/fpb_guest_stateless_profile.py " + verifier.batch_code_sha256,
        timeout=min(300, len(verifier.cases) * 15 + 10))
    host_call_seconds = time.monotonic() - started
    match = re.fullmatch(r"FPB_STATELESS_PROFILE=([A-Za-z0-9+/=]+)\n?", output)
    if not match:
        raise RuntimeError("Guest diagnostic framing failed")
    guest = strict_json_loads(base64.b64decode(match.group(1), validate=True).decode())
    if (not isinstance(guest, dict) or guest.get("kind") != "guest_stateless_case_profile_v1"
            or guest.get("case_count") != len(verifier.cases)
            or not isinstance(guest.get("per_case"), dict)
            or not isinstance(guest.get("outputs"), list)
            or len(guest["outputs"]) != len(verifier.cases)):
        raise RuntimeError("Guest diagnostic report differs")
    for expected, actual in zip(full_repaired["case_results"], guest["outputs"], strict=True):
        if (actual["returncode"] != expected["returncode"]
                or actual["stdout_sha256"] != expected["stdout_sha256"]
                or actual["truncated"]):
            raise RuntimeError("Guest diagnostic case output differs")
    guest["host_call_seconds"] = round(host_call_seconds, 6)
    guest["guest_program_sha256"] = digest
    guest["measured_after_timed_conditions"] = True
    return guest


def benchmark(task_dir, assets_dir, output, *, repetitions=3, contract_path=None):
    task_dir, assets, output = (Path(path).resolve() for path in
                                (task_dir, assets_dir, output))
    if type(repetitions) is not int or not 1 <= repetitions <= 10:
        raise ValueError("repetitions must be 1..10")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output must be new or empty")
    if contract_path is None:
        contract_path = Path(__file__).with_name("stateless_contract.json")
    source = assets / "rootfs.qcow2"
    manifest = json.loads((assets / "manifest.json").read_text(encoding="utf-8"))
    task = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != "boltons-microvm-assets-v2"
            or manifest.get("task_id") != "boltons-26-singularize-ss-v1"
            or task.get("task_id") != manifest["task_id"]
            or manifest.get("seed_workspace_sha256") != _workspace_digest(task_dir / "seed")
            or _sha(source) != manifest.get("rootfs_qcow2_sha256")):
        raise ValueError("Prepared image differs from pinned Boltons fixture")
    output.mkdir(parents=True, exist_ok=True)
    disk = output / "stateless-benchmark.qcow2"
    clone_mode = _clone_or_copy_qcow2(source, disk)
    verifier = None
    started = time.monotonic()
    with _runtime(assets, disk) as vm:
        _boot(vm)
        verifier = StatelessCaseVerifier(vm, task_dir, contract_path)
        guest_program_sha256 = verifier.install()
        batch_code_sha256 = verifier.prepare_batch()
        vm.save_snapshot("pristine")
        setup_seconds = time.monotonic() - started

        full_baseline = _grade(vm, verifier.cases, "pristine")
        vm.load_snapshot("pristine")
        stateless_baseline = verifier.grade()
        _assert_parity(full_baseline, stateless_baseline, 0.0)
        batch_baseline = verifier.grade_batch()
        _assert_parity(full_baseline, batch_baseline, 0.0)

        vm.load_snapshot("pristine")
        _patch(vm)
        vm.save_snapshot("repaired")
        full_repaired = _grade(vm, verifier.cases, "repaired")
        vm.load_snapshot("repaired")
        stateless_repaired = verifier.grade()
        _assert_parity(full_repaired, stateless_repaired, 1.0)
        batch_repaired = verifier.grade_batch()
        _assert_parity(full_repaired, batch_repaired, 1.0)
        serial_contamination = _contamination_probe(verifier, vm)
        batch_contamination = _contamination_probe(verifier, vm, batch=True)
        vm.load_snapshot("repaired")
        verifier.batch_code_sha256 = batch_code_sha256

        runs = []
        for iteration in range(repetitions):
            conditions = ("full_vm_restore_each_case", "stateless_serial_cases",
                          "stateless_batch_cases")
            conditions = conditions[iteration % 3:] + conditions[:iteration % 3]
            for condition in conditions:
                # Identical repaired source state at each condition start.
                # This common starting load is excluded from both timers.
                vm.load_snapshot("repaired")
                case_started = time.monotonic()
                if condition == "full_vm_restore_each_case":
                    grade = _grade(vm, verifier.cases, "repaired")
                elif condition == "stateless_batch_cases":
                    grade = verifier.grade_batch()
                else:
                    grade = verifier.grade()
                elapsed = time.monotonic() - case_started
                _assert_parity(full_repaired, grade, 1.0)
                runs.append({"iteration": iteration, "condition": condition,
                             "wall_seconds": round(elapsed, 6),
                             "reward": grade["reward"], "passed_cases": grade["passed_cases"]})
                print(f"{condition} {iteration} {elapsed:.3f}s", flush=True)
        vm.load_snapshot("repaired")
        diagnostic = _diagnose_guest_cases(verifier, vm, full_repaired)
        metrics = vm.get_state()["metrics"]
    disk.unlink(missing_ok=True)
    medians = {condition: statistics.median(
        item["wall_seconds"] for item in runs if item["condition"] == condition)
        for condition in ("full_vm_restore_each_case", "stateless_serial_cases",
                          "stateless_batch_cases")}
    report = {"kind": "boltons_microvm_stateless_verifier_comparison_v2",
              "scope": "one pinned solved public Boltons 26.0.0 fixture; ARM64 QEMU/HVF on this host",
              "task_id": verifier.task["task_id"],
              "host_private_verifier_sha256": verifier.contract["verifier_sha256"],
              "guest_program_sha256": guest_program_sha256,
              "guest_case_code_sha256": batch_code_sha256,
              "rootfs_seed_sha256": manifest["rootfs_qcow2_sha256"],
              "disk_clone_mode": clone_mode,
              "contract": {"requires_live_background_process_state": False,
                           "requires_shared_case_filesystem_state": False,
                           "requires_quiescent_submitted_state": True,
                           "allow_unprivileged_case_execution": True},
              "validation": {"full_baseline": full_baseline,
                             "stateless_baseline": stateless_baseline,
                             "batch_baseline": batch_baseline,
                             "full_repaired": full_repaired,
                             "stateless_repaired": stateless_repaired,
                             "batch_repaired": batch_repaired,
                             "serial_contamination": serial_contamination,
                             "batch_contamination": batch_contamination},
              "setup_seconds": round(setup_seconds, 6),
              "repetitions": repetitions, "order": "three-condition rotation within repetitions",
              "timed_scope": "14 host-checked case executions; full VM condition loads the same snapshot before every case; serial stateless uses 14 host serial calls; batch stateless uses one call but each case has a fresh mount and PID namespace; common initial load excluded from all conditions",
              "runs": runs, "median_seconds": medians,
              "graded_case_speedups": {
                  "serial_vs_full_vm": round(medians["full_vm_restore_each_case"] /
                                             medians["stateless_serial_cases"], 6),
                  "batch_vs_full_vm": round(medians["full_vm_restore_each_case"] /
                                            medians["stateless_batch_cases"], 6)},
              "untimed_guest_diagnostic": diagnostic,
              "vm_metrics": metrics,
              "does_not_measure": ["model inference", "gradient updates", "RL sample efficiency"],
              "limits": ["opt-in stateless verifier only; not equivalent to full-VM CPU/RAM/device restoration",
                         "one guest kernel shared across cases; not a hardened adversarial sandbox",
                         "candidate runs unprivileged but no comprehensive filesystem or syscall confinement audit",
                         "public solved fixture; no generalization claim"]}
    (output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--contract")
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    result = benchmark(args.task_dir, args.assets_dir, args.output,
                       repetitions=args.repetitions, contract_path=args.contract)
    print(json.dumps({"median_seconds": result["median_seconds"],
                      "graded_case_speedups": result["graded_case_speedups"]}, indent=2))
