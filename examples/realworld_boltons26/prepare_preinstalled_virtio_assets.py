"""Build a new pinned ARM64 Boltons v2 guest with the read RPC preinstalled.

The source task, Alpine boot files, module disk, and locally cached arm64
Python image are checked before an offline Docker mke2fs stage. Docker builds
the ext4 image only; graded episodes run in QEMU/HVF without a guest network.
This creates a *new* rootfs hash and never modifies existing benchmark assets.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
import uuid
from pathlib import Path

from future_prediction_bench import guest_action_rpc, prepared_microvm as pins
from future_prediction_bench.coding_env import _workspace_digest
from future_prediction_bench.microvm_runtime import _sha256_file

from .make_task_v2 import TASK_ID
from .prepare_microvm import _run


BOOT_FILES = ("vmlinuz-virt", "initramfs-virt", "modloop-virt-padded.raw")
AGENT_TARGET = "fpb_guest_action_rpc.py"


def build(task_dir, source_assets_dir, output_dir, *, disk_mib=256):
    task_dir, source, output = (Path(path).resolve() for path in
                                (task_dir, source_assets_dir, output_dir))
    if (output.exists() and (not output.is_dir() or any(output.iterdir()))) or any(
            output.is_relative_to(root) or root.is_relative_to(output)
            for root in (task_dir, source)):
        raise ValueError("New image output must be empty and disjoint")
    if type(disk_mib) is not int or not 128 <= disk_mib <= 4096:
        raise ValueError("disk_mib must be 128..4096")
    manifest_path = source / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("Source asset manifest missing or symlink")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    task_path = task_dir / "task.json"
    if task_path.is_symlink() or not task_path.is_file():
        raise ValueError("Task missing or symlink")
    task = json.loads(task_path.read_text(encoding="utf-8"))
    pins._check_v2_replace_helper(task)
    seed, verifier = task_dir / "seed", task_dir / "verifier" / "verify.json"
    if (task["task_id"] != TASK_ID or manifest.get("task_id") != TASK_ID
            or manifest.get("schema_version") != "boltons-microvm-assets-v2"
            or task.get("metadata", {}).get("source_sdist_sha256")
               != pins._SOURCE_SDIST_SHA256
            or manifest.get("source_sdist_sha256") != pins._SOURCE_SDIST_SHA256
            or verifier.is_symlink() or _sha256_file(verifier) != pins._VERIFIER_SHA256
            or _workspace_digest(seed) != pins._SEED_TREE_SHA256
            or manifest.get("seed_workspace_sha256") != pins._SEED_TREE_SHA256
            or _sha256_file(source / "rootfs.qcow2") != pins._ROOTFS_SHA256
            or manifest.get("rootfs_qcow2_sha256") != pins._ROOTFS_SHA256
            or _sha256_file(source / "modloop-virt-padded.raw") != pins._MODLOOP_SHA256
            or manifest.get("modloop_disk_sha256") != pins._MODLOOP_SHA256):
        raise ValueError("Source task and assets differ from pinned Boltons v2 inputs")
    for name, expected in (("vmlinuz-virt", pins._KERNEL_SHA256),
                           ("initramfs-virt", pins._INITRAMFS_SHA256),
                           ("modloop-virt-padded.raw", pins._MODLOOP_SHA256)):
        path = source / name
        if path.is_symlink() or not path.is_file() or _sha256_file(path) != expected:
            raise ValueError("Pinned boot or module asset differs: " + name)
    image_id = manifest.get("python_image_sha256")
    if not isinstance(image_id, str) or not image_id.startswith("sha256:"):
        raise ValueError("Pinned Python image digest missing")
    if (_run(["docker", "image", "inspect", image_id, "--format", "{{.Id}}"])
            != image_id or _run(["docker", "image", "inspect", image_id,
                                 "--format", "{{.Os}}/{{.Architecture}}"])
            != "linux/arm64"):
        raise ValueError("Locally cached pinned arm64 Python image differs")
    agent_path = Path(guest_action_rpc.__file__)
    if agent_path.is_symlink() or not agent_path.is_file():
        raise ValueError("Guest read agent source missing or symlink")
    agent_sha = _sha256_file(agent_path)
    build_started = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    stage = output / "build-stage"
    stage.mkdir()
    try:
        for name in BOOT_FILES:
            shutil.copy2(source / name, output / name)
        container = "fpb-virtio-image-" + uuid.uuid4().hex[:18]
        _run(["docker", "create", "--pull", "never", "--name", container,
              "--entrypoint", "/bin/true", image_id])
        try:
            copies = (
                ("/usr/local/bin/python3.12", "usr/local/bin/python3.12"),
                ("/usr/local/lib/libpython3.12.so.1.0", "usr/local/lib/libpython3.12.so.1.0"),
                ("/usr/local/lib/python3.12", "usr/local/lib/python3.12"),
                ("/usr/lib/aarch64-linux-gnu", "usr/lib/aarch64-linux-gnu"),
            )
            for path, relative in copies:
                destination = stage / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                _run(["docker", "cp", container + ":" + path, str(destination)])
        finally:
            _run(["docker", "rm", "-f", container])
        shutil.rmtree(stage / "usr/local/lib/python3.12/site-packages", ignore_errors=True)
        (stage / "lib").symlink_to("usr/lib", target_is_directory=True)
        (stage / "usr/lib/ld-linux-aarch64.so.1").symlink_to(
            "aarch64-linux-gnu/ld-linux-aarch64.so.1")
        shutil.copytree(seed, stage / "workspace")
        shutil.copy2(agent_path, stage / AGENT_TARGET)
        (stage / AGENT_TARGET).chmod(0o644)
        if _sha256_file(stage / AGENT_TARGET) != agent_sha:
            raise ValueError("Staged guest action agent differs")
        stage_seconds = time.monotonic() - build_started
        raw = output / "rootfs.raw"
        with raw.open("wb") as stream:
            stream.truncate(disk_mib * 1024 * 1024)
        mkfs_started = time.monotonic()
        _run(["docker", "run", "--rm", "--pull", "never", "--network", "none",
              "--user", "0:0", "--mount", f"type=bind,src={stage},dst=/input,readonly",
              "--mount", f"type=bind,src={output},dst=/output",
              "--entrypoint", "mke2fs", image_id, "-F", "-t", "ext4", "-d", "/input",
              "/output/rootfs.raw"], timeout=300)
        mkfs_seconds = time.monotonic() - mkfs_started
        convert_started = time.monotonic()
        _run(["qemu-img", "convert", "-f", "raw", "-O", "qcow2", str(raw),
              str(output / "rootfs.qcow2")], timeout=300)
        convert_seconds = time.monotonic() - convert_started
        rootfs_sha = _sha256_file(output / "rootfs.qcow2")
        updated = dict(manifest)
        updated.update({"task_dir": str(task_dir),
                        "rootfs_qcow2_sha256": rootfs_sha,
                        "rootfs_bytes": disk_mib * 1024 * 1024,
                        "image_variant": "preinstalled_virtio_read_agent_v1",
                        "preinstalled_guest_action_agent_sha256": agent_sha,
                        "preinstalled_guest_action_agent_path": "/" + AGENT_TARGET,
                        "source_assets_manifest_sha256": _sha256_file(manifest_path)})
        (output / "manifest.json").write_text(json.dumps(
            updated, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report = {"kind": "preinstalled_virtio_read_agent_image_build_v1",
                  "task_id": TASK_ID,
                  "source_rootfs_sha256": manifest["rootfs_qcow2_sha256"],
                  "new_rootfs_sha256": rootfs_sha,
                  "guest_action_agent_sha256": agent_sha,
                  "source_assets_manifest_sha256": _sha256_file(manifest_path),
                  "new_assets_manifest_sha256": _sha256_file(output / "manifest.json"),
                  "stage_seconds": stage_seconds,
                  "mkfs_seconds": mkfs_seconds,
                  "convert_seconds": convert_seconds,
                  "total_build_seconds": time.monotonic() - build_started}
        (output / "build_report.json").write_text(json.dumps(
            report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return report
    finally:
        shutil.rmtree(stage, ignore_errors=True)
        (output / "rootfs.raw").unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--source-assets-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--disk-mib", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(build(args.task_dir, args.source_assets_dir, args.output,
                           disk_mib=args.disk_mib), indent=2))


if __name__ == "__main__":
    main()
