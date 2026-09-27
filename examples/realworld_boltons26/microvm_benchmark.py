"""Run a real Boltons repair, host-private grading, and VM C/R timing.

This is a scripted, already-public repair integration fixture. Timings cover
the sandbox and grading path, not policy inference or optimizer updates.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import shlex
import shutil
import statistics
import time
from pathlib import Path

from future_prediction_bench.coding_env import _workspace_digest
from future_prediction_bench.microvm_runtime import MicroVMRuntime


PYTHON = "/usr/local/bin/python3.12"
GUEST_ENV = "LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/aarch64-linux-gnu PYTHONPATH=/workspace"
OLD = "    else:\n        singular = word[:-1]\n    return _match_case(orig_word, singular)"
NEW = "    elif word.endswith('ss'):\n        singular = word\n" + OLD


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required(vm, command):
    result = vm.run_shell(command, timeout=30)
    if result["return_code"]:
        raise RuntimeError(f"Guest setup failed: {command}: {result['stdout'][-1000:]}")
    return result["stdout"]


def _boot(vm):
    vm.start()
    vm.wait_for_serial("Launching initramfs emergency recovery shell", timeout=30)
    # QEMU can enumerate virtio disks in a different order than its arguments.
    # Assert the expected pinned sample layout before mounting anything.
    if "68 73 71 73" not in _required(vm, "hexdump -C -n 4 /dev/vda"):
        raise RuntimeError("Expected pinned SquashFS module disk at /dev/vda")
    if "53 ef" not in _required(vm, "dd if=/dev/vdb bs=1 skip=1080 count=2 2>/dev/null | hexdump -C"):
        raise RuntimeError("Expected ext4 workspace disk at /dev/vdb")
    for command in ("mkdir -p /media/modloop /mnt/root",
                    "mount -t squashfs -o ro /dev/vda /media/modloop",
                    "mount --bind /media/modloop/modules /lib/modules",
                    "modprobe ext4", "mount -t ext4 /dev/vdb /mnt/root"):
        _required(vm, command)
    if "Python 3.12" not in _required(vm, f"{GUEST_ENV} chroot /mnt/root {PYTHON} --version"):
        raise RuntimeError("Python 3.12 is missing from the guest workspace image")


def _python(vm, source, *, timeout=30):
    blob = base64.b64encode(source.encode("utf-8")).decode("ascii")
    expression = f"exec(__import__('base64').b64decode('{blob}'))"
    command = f"{GUEST_ENV} chroot /mnt/root {PYTHON} -B -c {shlex.quote(expression)}"
    return vm.run_shell(command, timeout=timeout)


def _patch(vm):
    source = ("from pathlib import Path\n"
              "p=Path('/workspace/boltons/strutils.py')\n"
              "content=p.read_text(encoding='utf-8')\n"
              f"old={OLD!r}\nnew={NEW!r}\n"
              "assert content.count(old)==1\n"
              "p.write_text(content.replace(old,new),encoding='utf-8')\n"
              "print('FPB_PATCHED')")
    result = _python(vm, source)
    if result["return_code"] != 0 or "FPB_PATCHED" not in result["stdout"]:
        raise RuntimeError("Pinned guest repair did not apply")


def _case(vm, argv):
    if (not isinstance(argv, list) or len(argv) != 4 or argv[:3] != ["python3", "-B", "-c"]
            or not isinstance(argv[3], str) or len(argv[3]) > 2048):
        raise ValueError("This pinned microVM sample supports Python -B -c cases only")
    encoded_case = base64.b64encode(argv[3].encode("utf-8")).decode("ascii")
    source = ("import base64,json,subprocess,sys\n"
              f"code=base64.b64decode('{encoded_case}')\n"
              "result=subprocess.run([sys.executable,'-B','-c',code.decode('utf-8')],"
              "cwd='/workspace',capture_output=True,timeout=10,check=False)\n"
              "payload={'returncode':result.returncode,'stdout':base64.b64encode(result.stdout).decode('ascii')}\n"
              "print('FPB_CASE_RESULT:'+base64.b64encode(json.dumps(payload,separators=(',',':')).encode()).decode())")
    result = _python(vm, source, timeout=20)
    if result["return_code"]:
        raise RuntimeError("Guest verifier command failed at infrastructure layer")
    matches = re.findall(r"FPB_CASE_RESULT:([A-Za-z0-9+/=]+)", result["stdout"])
    if len(matches) != 1:
        raise RuntimeError("Guest verifier result framing failed")
    payload = json.loads(base64.b64decode(matches[0], validate=True))
    return payload["returncode"], base64.b64decode(payload["stdout"], validate=True)


def _grade(vm, cases, tag):
    results = []
    for case in cases:
        vm.load_snapshot(tag)
        actual_returncode, actual_stdout = _case(vm, case["argv"])
        results.append({"passed": actual_returncode == case["expected_returncode"]
                        and actual_stdout == case["expected_stdout"].encode("utf-8"),
                        "returncode": actual_returncode,
                        "stdout_sha256": hashlib.sha256(actual_stdout).hexdigest()})
    return {"reward": 1.0 if all(result["passed"] for result in results) else 0.0,
            "passed_cases": sum(result["passed"] for result in results),
            "case_count": len(cases), "case_results": results}


def _runtime(assets, disk):
    manifest = json.loads((assets / "manifest.json").read_text(encoding="utf-8"))
    kernel, initramfs = assets / "vmlinuz-virt", assets / "initramfs-virt"
    modloop = assets / "modloop-virt-padded.raw"
    if _sha(modloop) != manifest["modloop_disk_sha256"]:
        raise ValueError("Pinned module disk changed")
    return MicroVMRuntime(kernel, initramfs, disk,
                          kernel_sha256=manifest["alpine_sha256"]["vmlinuz-virt"],
                          initramfs_sha256=manifest["alpine_sha256"]["initramfs-virt"],
                          readonly_disk_paths=(modloop,), command_timeout=30)


def benchmark(task_dir, assets_dir, output, *, repetitions=3):
    task_dir, assets, output = map(lambda p: Path(p).resolve(), (task_dir, assets_dir, output))
    if not 1 <= repetitions <= 10:
        raise ValueError("repetitions must be 1..10")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be new or empty")
    specification = json.loads((task_dir / "verifier" / "verify.json").read_text(encoding="utf-8"))
    if specification.get("kind") != "command_cases_v1" or not 1 <= len(specification.get("cases", [])) <= 32:
        raise ValueError("Expected pinned host-side command cases")
    cases = specification["cases"]
    seed_source = (task_dir / "seed" / "boltons" / "strutils.py").read_text(encoding="utf-8")
    if seed_source.count(OLD) != 1:
        raise ValueError("Pinned Boltons source differs from scripted fixture")
    base = assets / "rootfs.qcow2"
    manifest = json.loads((assets / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != "boltons-microvm-assets-v2"
            or manifest.get("task_id") != "boltons-26-singularize-ss-v1"
            or manifest.get("seed_workspace_sha256") != _workspace_digest(task_dir / "seed")):
        raise ValueError("Guest image is not bound to this pinned repository seed")
    if _sha(base) != manifest["rootfs_qcow2_sha256"]:
        raise ValueError("Pristine VM disk changed")
    output.mkdir(parents=True, exist_ok=True)
    runs = []

    # Validate the baseline and exact VM restore once before timing conditions.
    validation_disk = output / "validation.qcow2"
    shutil.copy2(base, validation_disk)
    with _runtime(assets, validation_disk) as vm:
        _boot(vm)
        _required(vm, "printf 'before' > /tmp/fpb-ram-state; sleep 120 </dev/null >/dev/null 2>&1 & echo $! > /tmp/fpb-process.pid")
        _required(vm, "kill -0 $(cat /tmp/fpb-process.pid)")
        vm.save_snapshot("pristine")
        _required(vm, "kill $(cat /tmp/fpb-process.pid); printf 'after' > /tmp/fpb-ram-state")
        vm.load_snapshot("pristine")
        restored_process_and_ram = (_required(vm, "cat /tmp/fpb-ram-state") == "before"
                                    and vm.run_shell("kill -0 $(cat /tmp/fpb-process.pid)")["return_code"] == 0)
        if not restored_process_and_ram:
            raise RuntimeError("Full VM checkpoint failed to restore a process or RAM state")
        baseline = _grade(vm, cases, "pristine")
        vm.load_snapshot("pristine")
        _patch(vm)
        vm.save_snapshot("repaired")
        solution = _grade(vm, cases, "repaired")
        vm.load_snapshot("pristine")
        after_restore = _python(vm, "from boltons.strutils import singularize; print('FPB_VALUE='+singularize('glass'))")
        restored_pristine = "FPB_VALUE=glas" in after_restore["stdout"]
        if (baseline["reward"], solution["reward"], restored_pristine) != (0.0, 1.0, True):
            raise RuntimeError("Baseline, repair, or state restoration failed")
        validation_metrics = vm.get_state()["metrics"]
    print("microVM baseline=0 repair=1 disk/RAM/process-restore=ok", flush=True)

    warm_setup_started = time.monotonic()
    warm_disk = output / "warm.qcow2"
    shutil.copy2(base, warm_disk)
    with _runtime(assets, warm_disk) as vm:
        _boot(vm)
        vm.save_snapshot("warm")
        warm_setup_seconds = time.monotonic() - warm_setup_started
        # Alternate condition order within each pair to reduce warm-cache and
        # host-load bias. The one persistent warm VM stays idle during cold runs.
        for index in range(repetitions):
            conditions = ("cold_full_vm", "warm_full_vm_restore")
            if index % 2:
                conditions = tuple(reversed(conditions))
            for condition in conditions:
                started = time.monotonic()
                if condition == "cold_full_vm":
                    disk = output / f"cold-{index}.qcow2"
                    shutil.copy2(base, disk)
                    with _runtime(assets, disk) as cold_vm:
                        _boot(cold_vm)
                        _patch(cold_vm)
                        cold_vm.save_snapshot("submitted")
                        grade = _grade(cold_vm, cases, "submitted")
                        metrics = cold_vm.get_state()["metrics"]
                else:
                    vm.load_snapshot("warm")
                    _patch(vm)
                    tag = f"submitted{index}"
                    vm.save_snapshot(tag)
                    grade = _grade(vm, cases, tag)
                    metrics = vm.get_state()["metrics"]
                elapsed = time.monotonic() - started
                if grade["reward"] != 1.0:
                    raise RuntimeError(f"{condition} repair grade differed")
                runs.append({"condition": condition, "iteration": index,
                             "wall_seconds": elapsed, "reward": grade["reward"],
                             "vm_metrics": metrics})
                print(f"{condition} {index} {elapsed:.3f}s", flush=True)
    cold_median = statistics.median(item["wall_seconds"] for item in runs if item["condition"] == "cold_full_vm")
    warm_median = statistics.median(item["wall_seconds"] for item in runs if item["condition"] == "warm_full_vm_restore")
    cold_total = sum(item["wall_seconds"] for item in runs if item["condition"] == "cold_full_vm")
    warm_total = warm_setup_seconds + sum(item["wall_seconds"] for item in runs
                                           if item["condition"] == "warm_full_vm_restore")
    report = {"scope": "single public solved Boltons repository fixture, ARM64 QEMU/HVF on this host",
              "measures": "environment setup, scripted repair, full VM C/R, isolated host-checked case grading",
              "does_not_measure": ["policy inference", "gradient updates", "RL sample efficiency"],
              "network_interface": "none", "host_workspace_mount": False,
              "baseline": baseline, "solution": solution,
              "restored_pristine_after_repair": restored_pristine,
              "restored_process_and_ram": restored_process_and_ram,
              "validation_vm_metrics": validation_metrics,
              "repetitions": repetitions, "condition_order": "alternated within pairs",
              "warm_setup_seconds": warm_setup_seconds,
              "runs": runs, "median_cold_seconds": cold_median,
              "median_warm_seconds": warm_median, "steady_state_speedup": cold_median / warm_median,
              "warm_total_seconds": warm_total, "cold_total_seconds": cold_total,
              "amortized_speedup_including_warm_setup": cold_total / warm_total}
    (output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    result = benchmark(args.task_dir, args.assets_dir, args.output,
                       repetitions=args.repetitions)
    print(json.dumps({key: result[key] for key in ("median_cold_seconds", "median_warm_seconds",
                                             "steady_state_speedup", "warm_setup_seconds",
                                             "amortized_speedup_including_warm_setup")}, indent=2))
