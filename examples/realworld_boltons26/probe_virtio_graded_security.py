"""Real guest adversarial probe for the graded virtio-read opt-in.

A candidate-visible check runs as nobody, tries to open the bound control
device, and starts a background process. The port access must be denied and
the later submit must fail closed before any VM snapshot or reward.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from future_prediction_bench.microvm_coding import MicroVMCodingAdapter
from future_prediction_bench.microvm_runtime import _clone_or_copy_qcow2
from future_prediction_bench.realworld import RealWorldEnv, validate_task

from .benchmark_semantic_recovery import _assets
from .benchmark_virtio_graded import _runtime
from .check_microvm_branch_env import _fixture_task


VISIBLE_PROBE = (
    "import os,subprocess\n"
    "try:\n"
    " fd=os.open('/dev/fpb.control',os.O_RDWR|os.O_NONBLOCK)\n"
    "except PermissionError:\n"
    " print('FPB_PORT_DENIED',flush=True)\n"
    "else:\n"
    " os.close(fd); print('FPB_PORT_OPEN',flush=True)\n"
    "subprocess.Popen(['/usr/local/bin/python3.12','-B','-c',"
    "'import time;time.sleep(60)'],stdout=subprocess.DEVNULL,"
    "stderr=subprocess.DEVNULL,start_new_session=True)\n"
)


def probe(task_dir, assets_dir, output_dir):
    task_dir, assets, output = (Path(path).resolve() for path in
                                (task_dir, assets_dir, output_dir))
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output directory must be new or empty")
    task_source, manifest, _ = _assets(task_dir, assets)
    output.mkdir(parents=True, exist_ok=True)
    disk = output / "adversarial.qcow2"
    _clone_or_copy_qcow2(assets / "rootfs.qcow2", disk)
    adapter = MicroVMCodingAdapter(
        _runtime(assets, manifest, disk, virtio=True, memory_mib=128),
        verifier_dir=task_dir / "verifier",
        visible_check=("python3", "-B", "-c", VISIBLE_PROBE),
        read_transport="virtio_serial_readonly_v1")
    task = _fixture_task(task_source)
    task.setdefault("metadata", {})["artifact_binding"] = adapter.artifact_binding()
    validate_task(task)
    env = RealWorldEnv(task, adapter)
    try:
        env.reset("adversarial-visible-check")
        check = env.step({"action": "run_visible_checks"})
        if (check["info"]["status"] != "active"
                or not check["observation"].get("passed")
                or check["observation"].get("stdout") != "FPB_PORT_DENIED\n"):
            raise RuntimeError("Unprivileged visible code opened the port or failed probe")
        submit = env.step({"action": "submit"})
        state = adapter.get_state()
        processes_after = adapter._scan_guest_processes()
        if (submit["info"]["status"] != "interrupted" or submit["reward"] is not None
                or state["metrics"]["vm_submit_snapshot_seconds"] != 0
                or state["runtime"]["metrics"]["snapshot_saves"] != 0
                or not adapter._virtio_client.stopped
                or processes_after == adapter._virtio_process_baseline):
            raise RuntimeError("Background candidate process did not block snapshot")
        report = {
            "kind": "graded_virtio_read_adversarial_guest_probe_v1",
            "task_id": task["task_id"],
            "candidate_visible_check_runs_unprivileged": True,
            "candidate_port_open_denied": True,
            "candidate_background_process_spawned": True,
            "submit_interrupted_before_snapshot": True,
            "reward_released": False,
            "qemu_snapshot_saves": 0,
            "agent_stop_acknowledged": True,
            "retirement_blocked_by_process_census": True,
        }
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n",
                                            encoding="utf-8")
        return report
    finally:
        adapter.close()
        disk.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(probe(args.task_dir, args.assets_dir, args.output), indent=2))


if __name__ == "__main__":
    main()
