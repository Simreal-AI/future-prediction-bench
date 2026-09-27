"""Create a v2 asset view from the exact pinned v1 Boltons guest image.

The v2 task changes only its tool contract. Its seed bytes and 14-case host
verifier are identical, so this copies immutable, digest-checked boot assets
and writes a new task-bound manifest without rebuilding ext4.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from future_prediction_bench import prepared_microvm as pins
from future_prediction_bench.coding_env import _workspace_digest
from future_prediction_bench.microvm_runtime import _clone_or_copy_qcow2, _sha256_file

try:
    from .make_task_v2 import TASK_ID
except ImportError:  # Direct `python3 examples/.../rebind_assets_v2.py` invocation.
    from make_task_v2 import TASK_ID


_ASSET_HASHES = {
    "rootfs.qcow2": pins._ROOTFS_SHA256,
    "vmlinuz-virt": pins._KERNEL_SHA256,
    "initramfs-virt": pins._INITRAMFS_SHA256,
    "modloop-virt-padded.raw": pins._MODLOOP_SHA256,
}


def rebind_assets_v2(task_dir, v1_assets_dir, output_dir):
    task_dir, source, output = (Path(item).resolve() for item in
                                (task_dir, v1_assets_dir, output_dir))
    if (output.exists() and (not output.is_dir() or any(output.iterdir()))) or any(
            output.is_relative_to(root) or root.is_relative_to(output)
            for root in (task_dir, source)):
        raise ValueError("V2 asset output must be new, empty, and disjoint")
    task = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    pins._check_v2_replace_helper(task)
    manifest_path = source / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("V1 asset manifest cannot be a symlink")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verifier = task_dir / "verifier" / "verify.json"
    seed_file = task_dir / "seed" / "boltons" / "strutils.py"
    if (task.get("task_id") != TASK_ID
            or task.get("metadata", {}).get("source_sdist_sha256")
               != pins._SOURCE_SDIST_SHA256
            or {item["name"] for item in task.get("tool_manifest", [])}
               != pins._TASK_TOOLS[TASK_ID]
            or verifier.is_symlink() or seed_file.is_symlink()
            or _sha256_file(verifier) != pins._VERIFIER_SHA256
            or _sha256_file(seed_file) != pins._SEED_FILE_SHA256
            or _workspace_digest(task_dir / "seed") != pins._SEED_TREE_SHA256
            or manifest.get("schema_version") != "boltons-microvm-assets-v2"
            or manifest.get("task_id") != pins._TASK_ID
            or manifest.get("source_sdist_sha256") != pins._SOURCE_SDIST_SHA256
            or manifest.get("seed_workspace_sha256") != pins._SEED_TREE_SHA256
            or manifest.get("rootfs_qcow2_sha256") != pins._ROOTFS_SHA256
            or manifest.get("modloop_disk_sha256") != pins._MODLOOP_SHA256
            or manifest.get("alpine_sha256") != {
                "vmlinuz-virt": pins._KERNEL_SHA256,
                "initramfs-virt": pins._INITRAMFS_SHA256,
                "modloop-virt": pins._MODLOOP_SHA256,
            }):
        raise ValueError("V1 assets and v2 task differ from exact pinned source")
    for name, expected in _ASSET_HASHES.items():
        path = source / name
        if path.is_symlink() or not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"Pinned V1 asset changed: {name}")
    output.mkdir(parents=True, exist_ok=True)
    try:
        clone_mode = _clone_or_copy_qcow2(source / "rootfs.qcow2",
                                          output / "rootfs.qcow2")
        for name in _ASSET_HASHES:
            if name != "rootfs.qcow2":
                shutil.copy2(source / name, output / name)
        for name, expected in _ASSET_HASHES.items():
            if _sha256_file(output / name) != expected:
                raise ValueError(f"V2 asset copy changed: {name}")
        rebound = dict(manifest, task_dir=str(task_dir), task_id=TASK_ID)
        (output / "manifest.json").write_text(
            json.dumps(rebound, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except BaseException:
        shutil.rmtree(output)
        raise
    return {"task_id": TASK_ID, "assets_dir": str(output),
            "source_manifest_sha256": _sha256_file(manifest_path),
            "v2_manifest_sha256": _sha256_file(output / "manifest.json"),
            "rootfs_clone_mode": clone_mode,
            "asset_sha256": dict(_ASSET_HASHES)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--v1-assets-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(rebind_assets_v2(args.task_dir, args.v1_assets_dir,
                                      args.output), indent=2))
