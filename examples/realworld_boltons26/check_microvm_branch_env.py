"""Exercise RealWorldEnv's full-VM branch API on the pinned Boltons fixture.

This is a scripted environment integration check, not learned-policy training.
The verifier stays on the host. Each branch receives an independent QEMU/HVF
VM with a cloned qcow2, CPU, RAM, and device state from one parent snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from future_prediction_bench.branch_advantages import prepare_sibling_advantages
from future_prediction_bench.coding_env import _workspace_digest
from future_prediction_bench.microvm_coding import MicroVMCodingAdapter
from future_prediction_bench.microvm_runtime import _clone_or_copy_qcow2, _sha256_file
from future_prediction_bench.realworld import RealWorldEnv, validate_task

from .microvm_benchmark import _runtime


def _required(runtime, command):
    result = runtime.run_shell(command)
    if result["return_code"]:
        raise RuntimeError("Guest isolation check failed")
    return result["stdout"]


def _fixture_task(task):
    if task.get("is_fixture") is not True:
        raise ValueError("This integration check requires the public fixture")
    now = datetime.now(timezone.utc)
    task = dict(task)
    task.update({"issued_at": (now - timedelta(minutes=1)).isoformat(),
                 "action_deadline": (now + timedelta(minutes=30)).isoformat(),
                 "outcome_not_before": (now - timedelta(minutes=1)).isoformat(),
                 "verify_after": (now - timedelta(minutes=1)).isoformat(),
                 "adapter_id": "qemu_hvf_coding", "adapter_version": "0.1"})
    return task


def check(task_dir, assets_dir, output_dir, *, stateless_contract=None):
    task_dir = Path(task_dir).resolve()
    assets = Path(assets_dir).resolve()
    output = Path(output_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output must be new or empty")
    verifier = task_dir / "verifier"
    if (output.is_relative_to(task_dir) or task_dir.is_relative_to(output)
            or output.is_relative_to(assets) or assets.is_relative_to(output)):
        raise ValueError("Output, task, and asset roots must be disjoint")
    task = _fixture_task(json.loads((task_dir / "task.json").read_text(encoding="utf-8")))
    contract_path = Path(stateless_contract).resolve() if stateless_contract else None
    manifest = json.loads((assets / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != "boltons-microvm-assets-v2"
            or manifest.get("task_id") != task["task_id"]
            or manifest.get("source_sdist_sha256") != task["metadata"]["source_sdist_sha256"]
            or manifest.get("seed_workspace_sha256") != _workspace_digest(task_dir / "seed")
            or _sha256_file(assets / "rootfs.qcow2") != manifest.get("rootfs_qcow2_sha256")
            or _sha256_file(assets / "modloop-virt-padded.raw") != manifest.get("modloop_disk_sha256")):
        raise ValueError("Prepared VM assets differ from the pinned task")
    actions = [json.loads(line) for line in (task_dir / "actions.solution.jsonl").read_text(
        encoding="utf-8").splitlines() if line]
    if (len(actions) != 4 or actions[1].get("action") != "write_file"
            or actions[1].get("path") != "boltons/strutils.py"):
        raise ValueError("Pinned public repair action is missing")
    output.mkdir(parents=True, exist_ok=True)
    source_task_path = None
    if contract_path is not None:
        # A refreshed local action window is a distinct source task for the
        # strict stateless verifier binding. Keep its exact verifier beside
        # it, under the private run directory rather than in the VM.
        source = output / "bound-source"
        (source / "verifier").mkdir(parents=True)
        source_task_path = source / "task.json"
        source_task_path.write_text(json.dumps(task, indent=2) + "\n", encoding="utf-8")
        shutil.copy2(verifier / "verify.json", source / "verifier" / "verify.json")
        verifier = source / "verifier"
    parent_disk = output / "parent.qcow2"
    clone_mode = _clone_or_copy_qcow2(assets / "rootfs.qcow2", parent_disk)
    runtime = _runtime(assets, parent_disk)
    parent_adapter = MicroVMCodingAdapter(
        runtime, verifier_dir=verifier,
        visible_check=("python3", "-B", "-c", "import boltons.strutils"),
        stateless_verifier_contract=contract_path,
        stateless_task_path=source_task_path)
    task.setdefault("metadata", {})["artifact_binding"] = parent_adapter.artifact_binding()
    validate_task(task)
    parent = RealWorldEnv(task, parent_adapter)
    children = []
    started = time.monotonic()
    try:
        parent.reset("scripted-integration-check")
        parent.step(actions[0])
        # The two marker locations test VM memory and writable ext4 state.
        _required(runtime, "printf 'inherited' > /tmp/fpb-branch-inherited")
        _required(runtime, "printf 'inherited' > /mnt/root/.fpb-branch-inherited")
        ref = parent.create_branch_checkpoint()
        if ref["checkpoint_kind"] != "full_vm_state_qcow2_v1":
            raise RuntimeError("Expected a full VM branch checkpoint")
        # Continue the parent after checkpoint. Children must still load the
        # frozen state, not a later live-parent workspace write.
        parent.step({"action": "write_file", "path": "boltons/fpb_parent_after.py",
                     "content": "parent-only\n"})
        for name in ("repaired", "baseline"):
            adapter = parent_adapter.branch_adapter(output / f"{name}.qcow2")
            children.append(adapter)
        repaired = parent.fork_from_checkpoint(ref, children[0], branch_id="repaired")
        baseline = parent.fork_from_checkpoint(ref, children[1], branch_id="baseline")
        for adapter in children:
            _required(adapter.runtime, "test \"$(cat /tmp/fpb-branch-inherited)\" = inherited")
            _required(adapter.runtime, "test \"$(cat /mnt/root/.fpb-branch-inherited)\" = inherited")
            _required(adapter.runtime, "test ! -e /mnt/root/workspace/boltons/fpb_parent_after.py")
        _required(children[0].runtime, "printf a > /tmp/fpb-branch-a")
        _required(children[0].runtime, "printf a > /mnt/root/.fpb-branch-a")
        _required(children[1].runtime, "printf b > /tmp/fpb-branch-b")
        _required(children[1].runtime, "printf b > /mnt/root/.fpb-branch-b")
        for runtime_to_check, other in ((children[0].runtime, "b"),
                                        (children[1].runtime, "a"),
                                        (runtime, "a"), (runtime, "b")):
            _required(runtime_to_check, f"test ! -e /tmp/fpb-branch-{other}")
            _required(runtime_to_check, f"test ! -e /mnt/root/.fpb-branch-{other}")
        repaired.step(actions[1])
        seed_digest = _sha256_file(task_dir / "seed" / "boltons" / "strutils.py")
        fixed_digest = hashlib.sha256(actions[1]["content"].encode("utf-8")).hexdigest()
        repaired_digest = repaired.step({"action": "read_file", "path": "boltons/strutils.py"})[
            "observation"]["sha256"]
        baseline_digest = baseline.step({"action": "read_file", "path": "boltons/strutils.py"})[
            "observation"]["sha256"]
        if repaired_digest != fixed_digest or baseline_digest != seed_digest:
            raise RuntimeError("Repaired code leaked to sibling or did not land in repaired child")
        repaired.step({"action": "submit"})
        baseline.step({"action": "submit"})
        repaired_reward = repaired.verify()["reward"]
        baseline_reward = baseline.verify()["reward"]
        rewards = [repaired_reward, baseline_reward]
        if rewards != [1.0, 0.0]:
            raise RuntimeError(f"Unexpected branch rewards: {rewards}")
        if (repaired.branch_lineage["prefix_event_sha256"] != ref["prefix_event_sha256"]
                or baseline.branch_lineage["prefix_event_sha256"] != ref["prefix_event_sha256"]
                or repaired.task["task_sha256"] != baseline.task["task_sha256"]):
            raise RuntimeError("Branch lineage or task binding changed")
        sibling_group = prepare_sibling_advantages(
            [repaired.export_trajectory(), baseline.export_trajectory()],
            expected_siblings=2, current_policy_id="scripted-integration-check",
            as_of=datetime.now(timezone.utc))
        branch_advantages = {member["branch_id"]: member["advantage"]
                             for member in sibling_group["members"]}
        if (branch_advantages != {"repaired": 1.0, "baseline": -1.0}
                or sibling_group["trainer_ready"] is not False):
            raise RuntimeError("Real VM sibling-local advantages differ")
        parent_digest = parent.step({"action": "read_file", "path": "boltons/strutils.py"})[
            "observation"]["sha256"]
        if parent_digest != seed_digest:
            raise RuntimeError("Repair branch changed the parent workspace")
        report = {"scope": "public pinned Boltons 26.0.0 scripted integration fixture",
                  "uses_realworld_env": True, "measures_model_training": False,
                  "runtime_kind": "qemu_hvf_full_vm_qcow2_v1",
                  "verifier_mode": ("stateless_namespaced_batch_v1"
                                    if contract_path is not None
                                    else "full_vm_per_case_v1"),
                  "checkpoint_kind": ref["checkpoint_kind"],
                  "task_id": task["task_id"], "task_sha256": ref["task_sha256"],
                  "snapshot_disk_sha256": ref["snapshot_disk_sha256"],
                  "host_private_verifier_sha256": task["metadata"]["artifact_binding"]["verifier_sha256"],
                  "parent_disk_clone_mode": clone_mode,
                  "branches": [{"id": "repaired", "reward": repaired_reward,
                                "hidden_cases": children[0].metrics["hidden_cases"]},
                               {"id": "baseline", "reward": baseline_reward,
                                "hidden_cases": children[1].metrics["hidden_cases"]}],
                  "parent_and_sibling_ram_and_ext4_independent": True,
                  "inherited_parent_ram_and_ext4": True,
                  "post_checkpoint_parent_write_absent_from_children": True,
                  "repaired_code_absent_from_baseline_and_parent": True,
                  "parent_workspace_unchanged": True,
                  "branch_local_rloo": {"advantages": branch_advantages,
                                        "trainer_ready": sibling_group["trainer_ready"],
                                        "loss_scope": sibling_group["loss_scope"],
                                        "prefix_visible_event_count": len(
                                            sibling_group["shared_prefix_visible_event_sha256s"])},
                  "elapsed_seconds": time.monotonic() - started}
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report
    finally:
        for child in children:
            child.close()
        parent_adapter.close()
        for path in (output / "repaired.qcow2", output / "baseline.qcow2", parent_disk):
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stateless-contract",
                        help="Opt into the exact host-authored stateless verifier contract")
    arguments = parser.parse_args()
    result = check(arguments.task_dir, arguments.assets_dir, arguments.output,
                   stateless_contract=arguments.stateless_contract)
    print(json.dumps(result, indent=2))
