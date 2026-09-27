"""Record exact 14-case graded parity through two-stage QEMU retirement.

This records the production runtime's QEMU state observations without
changing retirement behavior. Both arms use clones of one pinned
preinstalled-agent image and the same scripted five-read repair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from future_prediction_bench.microvm_coding import MicroVMCodingAdapter
from future_prediction_bench.microvm_runtime import (
    MicroVMRuntime, _clone_or_copy_qcow2,
)
from future_prediction_bench.realworld import RealWorldEnv, validate_task

from .benchmark_semantic_recovery import _assets, _source_hash
from .benchmark_virtio_graded import _fixture_task


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


class RecordingTwoStageRetirementRuntime(MicroVMRuntime):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.qemu_port_states = []

    def action_port_qemu_state(self):
        state = super().action_port_qemu_state()
        self.qemu_port_states.append(dict(state))
        return state


def _episode(mode, task_source, manifest, actions, task_dir, assets, output):
    virtio = mode == "virtio_read"
    disk = output / f"{mode}.qcow2"
    clone_mode = _clone_or_copy_qcow2(assets / "rootfs.qcow2", disk)
    runtime_type = RecordingTwoStageRetirementRuntime if virtio else MicroVMRuntime
    runtime = runtime_type(
        assets / "vmlinuz-virt", assets / "initramfs-virt", disk,
        kernel_sha256=manifest["alpine_sha256"]["vmlinuz-virt"],
        initramfs_sha256=manifest["alpine_sha256"]["initramfs-virt"],
        readonly_disk_paths=(assets / "modloop-virt-padded.raw",),
        memory_mib=128, command_timeout=30, enable_action_port=virtio)
    adapter = MicroVMCodingAdapter(
        runtime, verifier_dir=task_dir / "verifier",
        visible_check=("python3", "-B", "-c", "import boltons.strutils"),
        read_transport=("virtio_serial_readonly_v1" if virtio else "serial_shell"),
        preinstalled_read_agent=virtio)
    task = _fixture_task(task_source)
    task.setdefault("metadata", {})["artifact_binding"] = adapter.artifact_binding()
    validate_task(task)
    env = RealWorldEnv(task, adapter)
    started = time.monotonic()
    try:
        opening = env.reset("proof-known-repair")
        observations = []
        action_times = []
        for action in actions:
            action_started = time.monotonic()
            transition = env.step(action)
            action_times.append(time.monotonic() - action_started)
            if transition["info"]["status"] == "interrupted":
                raise RuntimeError("proof_graded_episode_interrupted")
            observations.append(transition["observation"])
        result = env.verify()
        if result["status"] != "graded" or result["reward"] != 1.0:
            raise RuntimeError("proof_graded_reward_unexpected")
        cases = result["evidence"]["case_results"]
        if len(cases) != 14 or not all(case["passed"] for case in cases):
            raise RuntimeError("proof_hidden_case_result_unexpected")
        state = runtime.get_state()
        if virtio and (not runtime.qemu_port_states
                       or runtime.qemu_port_states[-1] != {
                           "guest": "off", "host": "off",
                           "chardev_disconnected": True}
                       or not state["checkpoint_operations_supported"]
                       or runtime.metrics["snapshot_saves"] != 1
                       or runtime.metrics["snapshot_loads"] != 14):
            raise RuntimeError("proof_virtio_snapshot_or_reopen_missing")
        return {"mode": mode, "clone_mode": clone_mode,
                "opening_observation_sha256": _digest(opening["observation"]),
                "action_observation_sha256s": [_digest(value) for value in observations],
                "case_results_sha256": _digest(cases),
                "case_pass_vector": [case["passed"] for case in cases],
                "reward": result["reward"], "case_count": len(cases),
                "final_source_sha256": _source_hash(runtime),
                "whole_graded_episode_seconds": time.monotonic() - started,
                "read_action_total_seconds": sum(value for action, value in
                                                 zip(actions, action_times)
                                                 if action["action"] == "read_file"),
                "submit_seconds": action_times[-1],
                "snapshot_saves": runtime.metrics["snapshot_saves"],
                "snapshot_loads": runtime.metrics["snapshot_loads"],
                "qemu_port_states": runtime.qemu_port_states if virtio else None,
                "trusted_reopen_seconds": adapter.metrics[
                    "virtio_agent_disconnect_probe_seconds"] if virtio else 0.0}
    finally:
        adapter.close()
        disk.unlink(missing_ok=True)


def probe(task_dir, assets_dir, output_dir):
    task_dir, assets, output = (Path(value).resolve() for value in
                                (task_dir, assets_dir, output_dir))
    if ((output.exists() and (not output.is_dir() or any(output.iterdir())))
            or output.is_relative_to(task_dir) or output.is_relative_to(assets)
            or task_dir.is_relative_to(output) or assets.is_relative_to(output)):
        raise ValueError("Output must be new, empty, and disjoint")
    task_source, manifest, actions = _assets(task_dir, assets)
    if (manifest.get("image_variant") != "preinstalled_virtio_read_agent_v1"
            or manifest.get("preinstalled_guest_action_agent_path")
               != "/fpb_guest_action_rpc.py"):
        raise ValueError("Expected pinned preinstalled virtio agent image")
    paths = ("boltons/strutils.py", "tests/test_strutils.py", "README.md",
             "pyproject.toml", "boltons/__init__.py")
    if task_source["budgets"]["max_actions"] < 8 or any(
            not (task_dir / "seed" / path).is_file()
            or (task_dir / "seed" / path).is_symlink() for path in paths):
        raise ValueError("Pinned five-read workload changed")
    actions = ([{"action": "read_file", "path": path} for path in paths]
               + actions[1:])
    output.mkdir(parents=True, exist_ok=True)
    report = {"kind": "recorded_two_stage_graded_pair_v1",
              "task_id": manifest["task_id"],
              "rootfs_seed_sha256": manifest["rootfs_qcow2_sha256"],
              "read_paths": list(paths), "status": "incomplete",
              "production_runtime_modified_by_probe": False,
              "runs": []}
    try:
        for mode in ("serial", "virtio_read"):
            report["runs"].append(_episode(mode, task_source, manifest, actions,
                                           task_dir, assets, output))
        first, second = report["runs"]
        keys = ("opening_observation_sha256", "action_observation_sha256s",
                "case_results_sha256", "case_pass_vector", "reward",
                "final_source_sha256")
        report["parity"] = {key: first[key] == second[key] for key in keys}
        if not all(report["parity"].values()):
            raise RuntimeError("proof_graded_parity_mismatch")
        report["status"] = "recorded_two_stage_14_case_parity_passed"
    except Exception as exc:
        report["status"] = "failed"
        report["failure_type"] = type(exc).__name__
        report["failure_reason"] = str(exc)[:150]
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n",
                                            encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = probe(args.task_dir, args.assets_dir, args.output)
    print(json.dumps({"status": report["status"], "parity": report["parity"],
                      "runs": [{"mode": run["mode"],
                                "reward": run["reward"],
                                "case_count": run["case_count"],
                                "snapshot_saves": run["snapshot_saves"],
                                "snapshot_loads": run["snapshot_loads"],
                                "whole_graded_episode_seconds": run["whole_graded_episode_seconds"],
                                "qemu_port_state_first": (
                                    run["qemu_port_states"][0]
                                    if run["qemu_port_states"] else None),
                                "qemu_port_state_last": (
                                    run["qemu_port_states"][-1]
                                    if run["qemu_port_states"] else None),
                                "trusted_reopen_seconds": run["trusted_reopen_seconds"]}
                               for run in report["runs"]]}, indent=2))


if __name__ == "__main__":
    main()
