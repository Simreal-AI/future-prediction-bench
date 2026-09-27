"""Operator entry point for a full-state QEMU/HVF real-world coding episode."""

from __future__ import annotations

import copy
import json
import shutil
import time
from pathlib import Path

from .http import strict_json_loads
from .microvm_coding import MicroVMCodingAdapter
from .microvm_runtime import MicroVMRuntime, _sha256_file
from .realworld import RealWorldEnv, RealWorldTaskRegistry, validate_task


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                          encoding="utf-8")


def run_microvm_task(*, task, verifier_dir, assets_dir, actions, output,
                     visible_check=("python3", "-B", "-c", "import boltons.strutils"),
                     policy_id="external-scripted-policy", registry_path=None,
                     stateless_verifier_contract=None, stateless_task_path=None):
    """Replay bounded policy actions inside a dedicated offline Linux VM.

    The caller provides a prepared, pinned guest image. Hidden expected values
    remain in the host verifier directory, never on the VM block devices.
    """
    output, assets = Path(output).resolve(), Path(assets_dir).resolve()
    verifier_dir = Path(verifier_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be a new or empty directory")
    if output.is_relative_to(assets) or assets.is_relative_to(output):
        raise ValueError("Output and immutable asset roots must be disjoint")
    if output.is_relative_to(verifier_dir) or verifier_dir.is_relative_to(output):
        raise ValueError("Output and trusted verifier roots must be disjoint")
    frozen_task = validate_task(task)
    if stateless_verifier_contract is not None:
        if stateless_task_path is None:
            raise ValueError("Stateless mode requires the exact operator task.json path")
        if strict_json_loads(Path(stateless_task_path).read_text(encoding="utf-8")) != task:
            raise ValueError("Stateless task path differs from requested episode task")
    if not frozen_task["is_fixture"] and registry_path is None:
        raise ValueError("A persistent registry is required for non-fixture tasks")
    manifest = strict_json_loads((assets / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != "boltons-microvm-assets-v2"
            or manifest.get("task_id") != frozen_task["task_id"]
            or manifest.get("source_sdist_sha256")
            != frozen_task.get("metadata", {}).get("source_sdist_sha256")):
        raise ValueError("Expected a prepared pinned microVM asset manifest")
    disk_seed = assets / "rootfs.qcow2"
    modloop = assets / "modloop-virt-padded.raw"
    if (_sha256_file(disk_seed) != manifest["rootfs_qcow2_sha256"]
            or _sha256_file(modloop) != manifest["modloop_disk_sha256"]):
        raise ValueError("Prepared microVM assets changed")
    output.mkdir(parents=True, exist_ok=True)
    disk = output / "episode.qcow2"
    shutil.copy2(disk_seed, disk)
    runtime = MicroVMRuntime(
        assets / "vmlinuz-virt", assets / "initramfs-virt", disk,
        kernel_sha256=manifest["alpine_sha256"]["vmlinuz-virt"],
        initramfs_sha256=manifest["alpine_sha256"]["initramfs-virt"],
        readonly_disk_paths=(modloop,), command_timeout=30)
    adapter = MicroVMCodingAdapter(
        runtime, verifier_dir=verifier_dir, visible_check=visible_check,
        stateless_verifier_contract=stateless_verifier_contract,
        stateless_task_path=stateless_task_path)
    bound_task = copy.deepcopy(task)
    if bound_task["is_fixture"]:
        bound_task["adapter_id"] = "qemu_hvf_coding"
        bound_task["adapter_version"] = "0.1"
    elif bound_task.get("adapter_id") != "qemu_hvf_coding":
        raise ValueError("Non-fixture task adapter_id must be qemu_hvf_coding")
    binding = adapter.artifact_binding()
    metadata = bound_task.setdefault("metadata", {})
    if "artifact_binding" in metadata and metadata["artifact_binding"] != binding:
        raise ValueError("MicroVM artifacts differ from frozen task binding")
    metadata["artifact_binding"] = binding
    validate_task(bound_task)
    if registry_path is not None:
        registry_file = Path(registry_path).resolve()
        if (registry_file.is_relative_to(verifier_dir)
                or registry_file.is_relative_to(assets)
                or registry_file.is_relative_to(output)):
            raise ValueError("Task registry must stay outside verifier, assets, and episode output")
        registry = RealWorldTaskRegistry(registry_path)
        try:
            registry.register(bound_task)
        finally:
            registry.close()
    env = RealWorldEnv(bound_task, adapter)
    started = time.monotonic()
    transitions = []
    try:
        opening = env.reset(policy_id)
        for action in actions:
            if env.status != "active":
                break
            transitions.append(env.step(action))
        verification = env.verify() if env.status == "pending" else None
        state = env.get_state()
        report = {"task_id": task["task_id"], "episode_id": state["episode_id"],
                  "status": state["status"], "reward": state["reward"],
                  "is_fixture": task["is_fixture"], "trainer_ready": False,
                  "runtime_kind": "qemu_hvf_full_vm_qcow2_v1",
                  "verifier_mode": state["adapter_state"].get("verifier_mode"),
                  "network_interface": "none", "host_workspace_mount": False,
                  "opening": opening, "transitions": transitions,
                  "verification": verification, "environment_metrics": state["metrics"],
                  "adapter_metrics": state["adapter_state"].get("metrics", {}),
                  "vm_metrics": state["adapter_state"].get("runtime", {}).get("metrics", {}),
                  "elapsed_seconds": time.monotonic() - started}
        _write_json(output / "report.json", report)
        _write_json(output / "trusted_audit.json", state)
        trajectory = env.export_trajectory()
        if trajectory is not None:
            _write_json(output / "training_trajectory.json", trajectory)
        return {"status": report["status"], "reward": report["reward"],
                "actions_used": state["metrics"]["actions_used"],
                "elapsed_seconds": report["elapsed_seconds"],
                "output": str(output), "trainer_ready": False}
    finally:
        adapter.close()
