"""Five-pair same-contract serial vs two-child full-VM verification proof.

Uses the pinned public Boltons v2 fixture and scripted baseline/repair, not a
learned policy. Every measured row includes disk provisioning, RealWorldEnv
reset/actions/host-private grading, source attestation, and VM teardown. A
separate unmeasured pool episode proves sibling/parent RAM and ext4 isolation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import statistics
import subprocess
import threading
import time
from pathlib import Path

from future_prediction_bench import microvm_coding, microvm_runtime
from future_prediction_bench.microvm_coding import MicroVMCodingAdapter
from future_prediction_bench.microvm_runtime import _clone_or_copy_qcow2, _sha256_file
from future_prediction_bench.realworld import RealWorldEnv, validate_task

from .benchmark_prepared_env import VISIBLE
from .benchmark_semantic_recovery import _assets, _runtime, _source_hash
from .check_microvm_branch_env import _fixture_task
from .fullvm_verifier_pool import TwoChildFullVMVerifierPool


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _proof_source_binding():
    return {
        "fullvm_verifier_pool.py": _sha256_file(Path(__file__).with_name(
            "fullvm_verifier_pool.py")),
        "benchmark_fullvm_verifier_pool.py": _sha256_file(Path(__file__)),
        "microvm_coding.py": _sha256_file(Path(microvm_coding.__file__)),
        "microvm_runtime.py": _sha256_file(Path(microvm_runtime.__file__)),
    }


def _asset_binding(assets, manifest):
    return {
        "manifest_sha256": _sha256_file(assets / "manifest.json"),
        "manifest_schema_version": manifest["schema_version"],
        "source_sdist_sha256": manifest["source_sdist_sha256"],
        "seed_workspace_sha256": manifest["seed_workspace_sha256"],
        "rootfs_qcow2_sha256": manifest["rootfs_qcow2_sha256"],
        "kernel_sha256": manifest["alpine_sha256"]["vmlinuz-virt"],
        "initramfs_sha256": manifest["alpine_sha256"]["initramfs-virt"],
        "modloop_disk_sha256": manifest["modloop_disk_sha256"],
    }


def _host_memory_free_percent():
    """Best-effort macOS pressure signal; absent on other hosts."""
    try:
        output = subprocess.run(["memory_pressure", "-Q"], text=True,
                                capture_output=True, check=True, timeout=3).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"System-wide memory free percentage:\s*(\d+)%", output)
    return int(match.group(1)) if match else None


def _check_host_pressure(row):
    rss = row["peak_qemu_rss_kib_sampled"]
    if rss is None or row["qemu_rss_samples"] == 0:
        raise RuntimeError("qemu_rss_sampling_unavailable")
    if rss > 4 * 1024 * 1024:
        raise RuntimeError("host_rss_guard_exceeded")
    free = _host_memory_free_percent()
    row["host_memory_free_percent_after"] = free
    if free is not None and free < 10:
        raise RuntimeError("host_memory_pressure_guard_exceeded")


def _schedule(pairs):
    if type(pairs) is not int or not 5 <= pairs <= 8:
        raise ValueError("Proof requires five to eight alternating pairs")
    for pair in range(pairs):
        branches = ("repair", "baseline") if pair % 2 == 0 else ("baseline", "repair")
        arms = ("serial", "pool") if pair % 2 == 0 else ("pool", "serial")
        for branch in branches:
            for arm in arms:
                yield pair + 1, branch, arm


class _QemuRssSampler:
    """Best-effort 100 ms sampled sum of live QEMU RSS, in KiB."""

    def __init__(self, runtime, adapter):
        self.runtime = runtime
        self.adapter = adapter
        self.samples = 0
        self.peak_kib = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            processes = [self.runtime] + list(getattr(self.adapter, "active_children", ()))
            pids = sorted({process._process.pid for process in processes
                           if getattr(process, "_process", None) is not None
                           and process._process.poll() is None})
            if pids:
                try:
                    output = subprocess.run(
                        ["ps", "-o", "rss=", "-p", ",".join(map(str, pids))],
                        text=True, capture_output=True, check=True, timeout=1).stdout
                    rss = sum(int(line.strip()) for line in output.splitlines() if line.strip())
                    self.peak_kib = max(self.peak_kib or 0, rss)
                    self.samples += 1
                except (OSError, ValueError, subprocess.SubprocessError):
                    pass
            self._stop.wait(0.1)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join(timeout=2)


def _episode(name, arm, branch, task, manifest, actions, task_dir, assets,
             output, memory_mib, *, isolation_probe=False):
    disk = output / f"{name}.qcow2"
    started = time.monotonic()
    clone_mode = _clone_or_copy_qcow2(assets / "rootfs.qcow2", disk)
    clone_seconds = time.monotonic() - started
    runtime = None
    adapter = None
    try:
        runtime = _runtime(assets, manifest, disk, memory_mib)
        adapter = (TwoChildFullVMVerifierPool(
            runtime, verifier_dir=task_dir / "verifier", visible_check=VISIBLE,
            pool_dir=output / "pool-disks", isolation_probe=isolation_probe)
            if arm == "pool" else MicroVMCodingAdapter(
                runtime, verifier_dir=task_dir / "verifier", visible_check=VISIBLE))
        frozen = copy.deepcopy(task)
        frozen.setdefault("metadata", {})["artifact_binding"] = adapter.artifact_binding()
        validate_task(frozen)
        env = RealWorldEnv(frozen, adapter)
    except Exception:
        if adapter is not None:
            adapter.close()
        elif runtime is not None:
            runtime.close()
        disk.unlink(missing_ok=True)
        raise
    row = None
    try:
        with _QemuRssSampler(runtime, adapter) as sampler:
            reset_start = time.monotonic()
            opening = env.reset("scripted-fullvm-verifier-pool-proof")
            reset_seconds = time.monotonic() - reset_start
            observations = []
            action_times = []
            for action in actions:
                action_start = time.monotonic()
                result = env.step(action)
                action_times.append(time.monotonic() - action_start)
                if (result["info"]["status"] == "interrupted"
                        or result["observation"].get("status") in
                           {"error", "missed", "interrupted", "conflict"}):
                    raise RuntimeError("scripted_policy_action_failed")
                observations.append(_digest(result["observation"]))
            if env.status != "pending":
                raise RuntimeError("scripted_policy_did_not_submit")
            verify_start = time.monotonic()
            graded = env.verify()
            verify_seconds = time.monotonic() - verify_start
            if graded["status"] != "graded":
                raise RuntimeError(f"{arm}_verification_not_graded: {graded!r}")
            source_start = time.monotonic()
            source_sha = _source_hash(runtime)
            source_seconds = time.monotonic() - source_start
            state = env.get_state()
            row = {
                "arm": arm, "branch": branch, "clone_mode": clone_mode,
                "task_sha256": opening["task"]["task_sha256"],
                "artifact_binding_sha256": _digest(frozen["metadata"]["artifact_binding"]),
                "candidate_python_identity": frozen["metadata"]["artifact_binding"][
                    "candidate_python_identity"],
                "opening_observation_sha256": _digest(opening["observation"]),
                "action_observation_sha256s": observations,
                "reward": graded["reward"], "evidence": graded["evidence"],
                "case_results": graded["evidence"]["case_results"],
                "final_source_sha256": source_sha,
                "clone_seconds": clone_seconds,
                "setup_before_reset_seconds": reset_start - started - clone_seconds,
                "reset_seconds": reset_seconds,
                "action_seconds": action_times, "verify_seconds": verify_seconds,
                "source_attestation_seconds": source_seconds,
                "adapter_metrics": state["adapter_state"]["metrics"],
                "vm_metrics": state["adapter_state"]["runtime"]["metrics"],
                "pool_timings": (dict(adapter.pool_timings) if arm == "pool" else None),
            }
            adapter.close()
            disk.unlink(missing_ok=True)
            row["episode_wall_seconds"] = time.monotonic() - started
        row["peak_qemu_rss_kib_sampled"] = sampler.peak_kib
        row["qemu_rss_samples"] = sampler.samples
        return row
    finally:
        adapter.close()
        disk.unlink(missing_ok=True)


def _assert_parity(rows, pairs):
    if len(rows) != 4 * pairs:
        raise RuntimeError("ab_episode_count_incomplete")
    for branch, expected in (("repair", 1.0), ("baseline", 0.0)):
        subset = [row for row in rows if row["branch"] == branch]
        if sorted((row["pair"], row["arm"]) for row in subset) != [
                (pair, arm) for pair in range(1, pairs + 1)
                for arm in ("pool", "serial")]:
            raise RuntimeError("ab_schedule_incomplete")
        first = subset[0]
        for row in subset:
            if (row["reward"] != expected or len(row["case_results"]) != 14
                    or row["evidence"]["kind"] != "host_checked_qemu_full_vm_cases_v1"
                    or row["candidate_python_identity"] != "guest_uid_gid_65534_v2"
                    or row["adapter_metrics"]["full_vm_restores"] != 14
                    or row["adapter_metrics"]["hidden_cases"] != 14):
                raise RuntimeError("same_contract_grade_or_restore_count_failed")
            for key in ("task_sha256", "artifact_binding_sha256",
                        "candidate_python_identity",
                        "opening_observation_sha256", "action_observation_sha256s",
                        "reward", "evidence", "case_results", "final_source_sha256"):
                if row[key] != first[key]:
                    raise RuntimeError("same_contract_parity_failed: " + key)
    if len({row["task_sha256"] for row in rows}) != 1:
        raise RuntimeError("task_sha_differs_across_branches")


def benchmark(task_dir, assets_dir, output_dir, *, pairs=5, memory_mib=128):
    if type(memory_mib) is not int or not 128 <= memory_mib <= 256:
        raise ValueError("Proof caps each of three VMs to 128..256 MiB")
    schedule = list(_schedule(pairs))
    task_dir, assets, output = (Path(path).resolve() for path in
                                (task_dir, assets_dir, output_dir))
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output directory must be new or empty")
    if (output.is_relative_to(task_dir) or task_dir.is_relative_to(output)
            or output.is_relative_to(assets) or assets.is_relative_to(output)):
        raise ValueError("Output, task, and immutable assets must be disjoint")
    task_source, manifest, repair = _assets(task_dir, assets)
    source_binding = _proof_source_binding()
    asset_binding = _asset_binding(assets, manifest)
    baseline = [json.loads(line) for line in
                (task_dir / "actions.baseline.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()]
    if baseline != [{"action": "submit"}]:
        raise ValueError("Pinned baseline actions differ")
    task = _fixture_task(task_source)
    output.mkdir(parents=True, exist_ok=True)
    (output / "pool-disks").mkdir()
    initial_free_percent = _host_memory_free_percent()
    if initial_free_percent is not None and initial_free_percent < 10:
        raise RuntimeError("host_memory_pressure_guard_exceeded_before_start")
    # One unpaired, unmeasured episode exercises VM RAM and writable ext4
    # isolation; it is excluded from every speed comparison below.
    isolation = _episode("isolation-preflight", "pool", "baseline", task,
                         manifest, baseline, task_dir, assets, output,
                         memory_mib, isolation_probe=True)
    if isolation["reward"] != 0.0 or isolation["pool_timings"]["failed"]:
        raise RuntimeError("full_vm_pool_isolation_preflight_failed")
    _check_host_pressure(isolation)
    rows = []
    for pair, branch, arm in schedule:
        actions = repair if branch == "repair" else baseline
        row = _episode(f"pair-{pair}-{branch}-{arm}", arm, branch, task,
                       manifest, actions, task_dir, assets, output, memory_mib)
        row["pair"] = pair
        rows.append(row)
        if (row["reward"] != (1.0 if branch == "repair" else 0.0)
                or len(row["case_results"]) != 14
                or any(case["return_code"] in (124, 125)
                       for case in row["case_results"])):
            raise RuntimeError("unexpected_reward_case_count_or_timeout")
        _check_host_pressure(row)
        print(f"pair {pair} {branch} {arm}: "
              f"episode={row['episode_wall_seconds']:.3f}s "
              f"verify={row['verify_seconds']:.3f}s "
              f"reward={row['reward']}", flush=True)
    _assert_parity(rows, pairs)
    summary = {
        arm: {"episode_median_seconds": statistics.median(
                  row["episode_wall_seconds"] for row in rows if row["arm"] == arm),
              "verify_median_seconds": statistics.median(
                  row["verify_seconds"] for row in rows if row["arm"] == arm),
              "peak_qemu_rss_kib_sampled_max": max(
                  (row["peak_qemu_rss_kib_sampled"] for row in rows
                   if row["arm"] == arm
                   and row["peak_qemu_rss_kib_sampled"] is not None),
                  default=None)}
        for arm in ("serial", "pool")}
    branch_summary = {}
    for branch in ("repair", "baseline"):
        branch_summary[branch] = {}
        for arm in ("serial", "pool"):
            subset = [row for row in rows if row["branch"] == branch
                      and row["arm"] == arm]
            branch_summary[branch][arm] = {
                "episode_median_seconds": statistics.median(
                    row["episode_wall_seconds"] for row in subset),
                "verify_median_seconds": statistics.median(
                    row["verify_seconds"] for row in subset),
            }
        differences = []
        for pair in range(1, pairs + 1):
            matched = {row["arm"]: row for row in rows
                       if row["pair"] == pair and row["branch"] == branch}
            differences.append(matched["serial"]["episode_wall_seconds"]
                               - matched["pool"]["episode_wall_seconds"])
        branch_summary[branch]["paired_serial_minus_pool_median_seconds"] = (
            statistics.median(differences))
    pool_stage_medians = {
        key: statistics.median(row["pool_timings"][key] for row in rows
                               if row["arm"] == "pool")
        for key in ("fork_seconds", "disk_clone_seconds",
                    "child_start_restore_seconds", "parallel_case_wall_seconds",
                    "cleanup_seconds")}
    if (_proof_source_binding() != source_binding
            or _asset_binding(assets, manifest) != asset_binding):
        raise RuntimeError("proof_code_or_asset_manifest_changed_during_ab")
    report = {
        "kind": "full_vm_two_child_verifier_pool_proof_ab_v1",
        "scope": "pinned public solved Boltons v2 scripted RealWorldEnv fixture",
        "does_not_measure": ["policy inference", "gradient updates", "RL sample efficiency"],
        "qemu_rss_note": "100 ms sampled sum of QEMU process RSS; shared pages may be double counted",
        "host_guard": "Abort after an episode if sampled QEMU RSS exceeds 4 GiB, RSS samples are unavailable, or macOS memory_pressure reports below 10% free; this is not a continuous physical-memory guarantee.",
        "host_memory_free_percent_before": initial_free_percent,
        "pool_parent_note": "The submitted parent stays live but runs no hidden cases; its RSS is sampled with the two children. The proof does not establish multi-tenant sandbox security.",
        "memory_mib_per_vm": memory_mib, "maximum_live_vms": 3,
        "pairs": pairs, "schedule": [list(item) for item in schedule],
        "task_id": task["task_id"], "source_sdist_sha256": manifest["source_sdist_sha256"],
        "rootfs_seed_sha256": manifest["rootfs_qcow2_sha256"],
        "asset_binding": asset_binding,
        "proof_source_sha256": source_binding,
        "isolation_preflight": isolation,
        "runs": rows, "summary": summary,
        "by_branch": branch_summary, "pool_stage_medians": pool_stage_medians,
        "episode_median_serial_over_pool_ratio": (
            summary["serial"]["episode_median_seconds"]
            / summary["pool"]["episode_median_seconds"]),
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--memory-mib", type=int, default=128)
    args = parser.parse_args()
    report = benchmark(args.task_dir, args.assets_dir, args.output,
                       pairs=args.pairs, memory_mib=args.memory_mib)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
