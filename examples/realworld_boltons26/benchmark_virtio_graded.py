"""Actual graded RealWorldEnv A/B for opt-in read-only virtio RPC.

The only policy action moved to the persistent guest service is read_file.
Writes, visible checks, submit, and host-private full-VM grading retain the
existing serial adapter implementation. The service is retired before savevm.
This fixture uses a scripted known repair; no model or optimizer runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from future_prediction_bench import guest_action_rpc
from future_prediction_bench.microvm_coding import MicroVMCodingAdapter
from future_prediction_bench.microvm_runtime import (MicroVMRuntime,
                                                      MicroVMRuntimeError,
                                                      _clone_or_copy_qcow2)
from future_prediction_bench.realworld import RealWorldEnv, validate_task
from future_prediction_bench.virtio_action import VirtioActionError

from .benchmark_semantic_recovery import _assets, _source_hash
from .check_microvm_branch_env import _fixture_task


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _runtime(assets, manifest, disk, *, virtio, memory_mib):
    return MicroVMRuntime(
        assets / "vmlinuz-virt", assets / "initramfs-virt", disk,
        kernel_sha256=manifest["alpine_sha256"]["vmlinuz-virt"],
        initramfs_sha256=manifest["alpine_sha256"]["initramfs-virt"],
        readonly_disk_paths=(assets / "modloop-virt-padded.raw",),
        memory_mib=memory_mib, command_timeout=30,
        enable_action_port=virtio)


def _episode(name, mode, task_source, manifest, actions, task_dir, assets,
             output, memory_mib, *, preinstalled_agent=False):
    disk = output / f"{name}.qcow2"
    clone_mode = _clone_or_copy_qcow2(assets / "rootfs.qcow2", disk)
    virtio = mode == "virtio_read"
    adapter = MicroVMCodingAdapter(
        _runtime(assets, manifest, disk, virtio=virtio, memory_mib=memory_mib),
        verifier_dir=task_dir / "verifier",
        visible_check=("python3", "-B", "-c", "import boltons.strutils"),
        read_transport=("virtio_serial_readonly_v1" if virtio else "serial_shell"),
        preinstalled_read_agent=virtio and preinstalled_agent)
    task = _fixture_task(task_source)
    task.setdefault("metadata", {})["artifact_binding"] = adapter.artifact_binding()
    validate_task(task)
    env = RealWorldEnv(task, adapter)
    started = time.monotonic()
    try:
        opening = env.reset("pinned-known-repair")
        reset_seconds = time.monotonic() - started
        action_results, action_seconds = [], []
        for action in actions:
            action_started = time.monotonic()
            transition = env.step(action)
            action_seconds.append(time.monotonic() - action_started)
            action_results.append(transition["observation"])
            if transition["info"]["status"] == "interrupted":
                raise RuntimeError(f"{mode} adapter interrupted on {action['action']}")
        verify_started = time.monotonic()
        graded = env.verify()
        verify_seconds = time.monotonic() - verify_started
        graded_episode_seconds = time.monotonic() - started
        if graded.get("reward") != 1.0 or graded["status"] != "graded":
            raise RuntimeError(f"{mode} pinned repaired task did not resolve")
        cases = graded["evidence"]["case_results"]
        if len(cases) != 14 or not all(case["passed"] for case in cases):
            raise RuntimeError(f"{mode} pinned case result changed")
        state = adapter.get_state()
        if virtio and (state["runtime"]["action_transport"]
                       != "virtio_serial_readonly_retired_v1"
                       or not state["runtime"]["checkpoint_operations_supported"]
                       or not state["runtime"]["action_port_qemu_disconnected_attested"]
                       or state["metrics"]["virtio_reads"] != sum(
                           action["action"] == "read_file" for action in actions)):
            raise RuntimeError("Read port was not retired before graded VM restore")
        retirement = None
        if virtio:
            client = adapter._virtio_client
            if client is None or not client.stopped or client.socket.fileno() != -1:
                raise RuntimeError("Stopped action client retained an open host socket")
            try:
                client.read_file("boltons/strutils.py")
            except VirtioActionError as exc:
                if str(exc) != "action_port_session_unavailable":
                    raise
            else:
                raise RuntimeError("Old action client remained callable")
            try:
                adapter.runtime.action_port_socket()
            except MicroVMRuntimeError as exc:
                if str(exc) != "vm_action_port_not_enabled":
                    raise
            else:
                raise RuntimeError("Retired runtime action socket remained available")
            try:
                adapter.runtime.fork_snapshot("submitted", [output / "forbidden-child.qcow2"])
            except MicroVMRuntimeError as exc:
                if str(exc) != "vm_action_port_fork_not_supported":
                    raise
            else:
                raise RuntimeError("Opt-in runtime unexpectedly allowed snapshot fork")
            retirement = {"stop_acknowledged": True,
                          "host_socket_closed": True,
                          "qemu_disconnect_attested": True,
                          "old_client_unusable": True,
                          "runtime_socket_unavailable": True,
                          "fork_snapshot_rejected": True,
                          "save_snapshot_count": state["runtime"]["metrics"]["snapshot_saves"],
                          "load_snapshot_count": state["runtime"]["metrics"]["snapshot_loads"]}
        return {
            "mode": mode, "clone_mode": clone_mode,
            "opening_observation_sha256": _digest(opening["observation"]),
            "action_observation_sha256s": [_digest(value) for value in action_results],
            "read_observation": action_results[0],
            "reward": graded["reward"], "case_count": len(cases),
            "case_results": cases,
            "final_source_sha256": _source_hash(adapter.runtime),
            "reset_seconds": reset_seconds,
            "action_seconds": dict(zip((a["action"] for a in actions), action_seconds)),
            "action_timings": [
                {"action": action["action"], "seconds": seconds}
                for action, seconds in zip(actions, action_seconds)],
            "read_file_total_seconds": sum(seconds for action, seconds in
                                           zip(actions, action_seconds)
                                           if action["action"] == "read_file"),
            "verify_seconds": verify_seconds,
            "whole_graded_episode_seconds": graded_episode_seconds,
            "retirement": retirement,
            "adapter_metrics": state["metrics"],
            "runtime_metrics": state["runtime"]["metrics"],
        }
    finally:
        adapter.close()
        disk.unlink(missing_ok=True)


def benchmark(task_dir, assets_dir, output_dir, *, pairs=2, memory_mib=128,
              read_heavy=False, preinstalled_agent=False):
    task_dir, assets, output = (Path(path).resolve() for path in
                                (task_dir, assets_dir, output_dir))
    if type(pairs) is not int or not 1 <= pairs <= 5:
        raise ValueError("pairs must be 1..5")
    if type(memory_mib) is not int or not 128 <= memory_mib <= 8192:
        raise ValueError("memory_mib must be 128..8192")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output directory must be empty")
    if (output.is_relative_to(task_dir) or output.is_relative_to(assets)
            or task_dir.is_relative_to(output) or assets.is_relative_to(output)):
        raise ValueError("Output must be disjoint from task and assets")
    task, manifest, actions = _assets(task_dir, assets)
    if preinstalled_agent:
        if (manifest.get("image_variant") != "preinstalled_virtio_read_agent_v1"
                or manifest.get("preinstalled_guest_action_agent_path")
                   != "/fpb_guest_action_rpc.py"
                or manifest.get("preinstalled_guest_action_agent_sha256")
                   != hashlib.sha256(Path(guest_action_rpc.__file__).read_bytes()).hexdigest()):
            raise ValueError("Pinned preinstalled guest agent image differs")
    if read_heavy:
        paths = ("boltons/strutils.py", "tests/test_strutils.py", "README.md",
                 "pyproject.toml", "boltons/__init__.py")
        if task["budgets"]["max_actions"] < 8 or any(
                not (task_dir / "seed" / path).is_file()
                or (task_dir / "seed" / path).is_symlink() for path in paths):
            raise ValueError("Pinned read-heavy workload or action budget differs")
        actions = ([{"action": "read_file", "path": path} for path in paths]
                   + actions[1:])
    output.mkdir(parents=True, exist_ok=True)
    runs = []
    for pair in range(pairs):
        order = (("serial", "virtio_read") if pair % 2 == 0
                 else ("virtio_read", "serial"))
        for mode in order:
            row = _episode(f"pair-{pair + 1}-{mode}", mode, task, manifest,
                           actions, task_dir, assets, output, memory_mib,
                           preinstalled_agent=preinstalled_agent)
            row["pair"] = pair + 1
            runs.append(row)
            print(f"pair {pair + 1} {mode}: {row['whole_graded_episode_seconds']:.3f}s, "
                  f"{sum(a['action'] == 'read_file' for a in actions)} reads "
                  f"{row['read_file_total_seconds'] * 1000:.3f}ms, "
                  f"reward {row['reward']}", flush=True)
    first = runs[0]
    for row in runs[1:]:
        for key in ("opening_observation_sha256", "action_observation_sha256s",
                    "reward", "case_results", "final_source_sha256"):
            if row[key] != first[key]:
                raise RuntimeError(f"A/B {key} differs")
    report = {
        "kind": ("graded_realworld_virtio_preinstalled_read_heavy_ab_v1"
                 if preinstalled_agent and read_heavy
                 else "graded_realworld_virtio_read_heavy_ab_v1" if read_heavy
                 else "graded_realworld_virtio_read_ab_v1"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task_id": task["task_id"], "pairs": pairs,
        "actions": [action["action"] for action in actions],
        "read_paths": [action["path"] for action in actions
                       if action["action"] == "read_file"],
        "rootfs_seed_sha256": manifest["rootfs_qcow2_sha256"],
        "preinstalled_agent": preinstalled_agent,
        "all_action_observations_equal": True,
        "all_case_results_equal": True,
        "all_rewards_equal": True,
        "case_count_per_episode": first["case_count"],
        "read_transport_scope": "policy_read_file_only",
        "snapshot_precondition": "agent_stopped_exited_port_unmounted_host_fd_closed_qemu_guest_off_host_off_chardev_disconnected",
        "runs": runs,
        "medians": {
            mode: {
                "read_file_total_ms": round(statistics.median(
                    r["read_file_total_seconds"] * 1000 for r in runs
                    if r["mode"] == mode), 6),
                "whole_graded_episode_seconds": round(statistics.median(
                    r["whole_graded_episode_seconds"] for r in runs
                    if r["mode"] == mode), 6),
            } for mode in ("serial", "virtio_read")
        },
    }
    (output / "report.json").write_text(json.dumps(report, indent=2,
                                                   ensure_ascii=False) + "\n",
                                        encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--memory-mib", type=int, default=128)
    parser.add_argument("--read-heavy", action="store_true",
                        help="Use five pinned repository reads within the eight-action budget")
    parser.add_argument("--preinstalled-agent", action="store_true",
                        help="Require a separately built pinned image with the exact read RPC preinstalled")
    args = parser.parse_args()
    report = benchmark(args.task_dir, args.assets_dir, args.output,
                       pairs=args.pairs, memory_mib=args.memory_mib,
                       read_heavy=args.read_heavy,
                       preinstalled_agent=args.preinstalled_agent)
    print(json.dumps(report["medians"], indent=2))


if __name__ == "__main__":
    main()
