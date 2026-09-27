"""Build a pinned, offline ARM64 Humanize 4.15.0 guest image.

Docker is used only to extract the already-cached Python 3.12 runtime and
create ext4. Graded episodes run in QEMU/HVF without a NIC or host mount.
The Boltons assets are an immutable source of the SHA-checked Alpine boot
files and cached Python image identity, never of task workspace content.
"""

from __future__ import annotations

import argparse
import json
import shutil
import uuid
from pathlib import Path

from examples.realworld_boltons26.prepare_microvm import ALPINE_SHA256, _run, _sha
from future_prediction_bench.coding_env import _workspace_digest, _workspace_files

from .make_task_v2 import TASK_ID


SOURCE_FILE = "src/humanize/filesize.py"
SOURCE_SDIST_SHA256 = "1dd098483eb1c7ee8e32eb2e99ad1910baefa4b75c3aff3a82f4d78688993b10"
SOURCE_TREE_SHA256 = "b42ad06044a4bab6906fc95b16c4fa1348a8c82f36cbe490f18b9a929293cfd2"
SOURCE_FILE_SHA256 = "2e8b51584654471f91ab5234ca08fdc352b0b1d41b8c751de5d68d32cf932bda"
VERIFIER_SHA256 = "4e36079f09ed50c6b9d1c23aed746e9d6ffca22cf62878411717b8329dcb7305"
SCHEMA = "humanize4150-microvm-assets-v1"


def prepare(task_dir, source_assets_dir, output, *, image=None, disk_mib=256):
    task_dir = Path(task_dir).resolve()
    source_assets = Path(source_assets_dir).resolve()
    output = Path(output).resolve()
    seed = task_dir / "seed"
    verifier = task_dir / "verifier" / "verify.json"
    task = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    _workspace_files(seed)
    manifest = json.loads((source_assets / "manifest.json").read_text(encoding="utf-8"))
    if (task.get("task_id") != TASK_ID or task.get("is_fixture") is not True
            or task.get("metadata", {}).get("source_sdist_sha256")
               != SOURCE_SDIST_SHA256
            or _workspace_digest(seed) != SOURCE_TREE_SHA256
            or _sha(seed / SOURCE_FILE) != SOURCE_FILE_SHA256
            or _sha(verifier) != VERIFIER_SHA256
            or manifest.get("architecture") != "linux/arm64"
            or manifest.get("alpine_sha256") != ALPINE_SHA256
            or not isinstance(manifest.get("python_image_sha256"), str)
            or not manifest["python_image_sha256"].startswith("sha256:")):
        raise ValueError("Pinned Humanize source, verifier, or boot provenance differs")
    if (output.exists() and any(output.iterdir())
            or not 128 <= disk_mib <= 4096
            or output.is_relative_to(task_dir)
            or output.is_relative_to(source_assets)):
        raise ValueError("Output must be new, disjoint, and 128..4096 MiB")
    image_id = image or manifest["python_image_sha256"]
    if image_id != manifest["python_image_sha256"]:
        raise ValueError("Cached Python image differs from pinned source assets")
    actual_id = _run(["docker", "image", "inspect", image_id, "--format", "{{.Id}}"])
    architecture = _run(["docker", "image", "inspect", image_id,
                         "--format", "{{.Os}}/{{.Architecture}}"])
    if actual_id != image_id or architecture != "linux/arm64":
        raise ValueError("A cached immutable linux/arm64 Python image is required")
    output.mkdir(parents=True, exist_ok=True)
    stage = output / "build-stage"
    stage.mkdir()
    for name, path in (("vmlinuz-virt", "vmlinuz-virt"),
                       ("initramfs-virt", "initramfs-virt"),
                       ("modloop-virt", "modloop-virt-padded.raw")):
        source = source_assets / path
        if source.is_symlink() or not source.is_file() or _sha(source) != ALPINE_SHA256[name]:
            raise ValueError("Pinned Alpine input differs: " + name)
        shutil.copy2(source, output / path)
    container = "fpb-humanize-asset-" + uuid.uuid4().hex[:16]
    _run(["docker", "create", "--pull", "never", "--name", container,
          "--entrypoint", "/bin/true", image_id])
    try:
        copies = (
            ("/usr/local/bin/python3.12", "usr/local/bin/python3.12"),
            ("/usr/local/lib/libpython3.12.so.1.0", "usr/local/lib/libpython3.12.so.1.0"),
            ("/usr/local/lib/python3.12", "usr/local/lib/python3.12"),
            ("/usr/lib/aarch64-linux-gnu", "usr/lib/aarch64-linux-gnu"),
        )
        for source, relative in copies:
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            _run(["docker", "cp", container + ":" + source, str(destination)])
    finally:
        _run(["docker", "rm", "-f", container])
    shutil.rmtree(stage / "usr/local/lib/python3.12/site-packages", ignore_errors=True)
    (stage / "lib").symlink_to("usr/lib", target_is_directory=True)
    (stage / "usr/lib/ld-linux-aarch64.so.1").symlink_to(
        "aarch64-linux-gnu/ld-linux-aarch64.so.1")
    shutil.copytree(seed, stage / "workspace")
    raw = output / "rootfs.raw"
    with raw.open("wb") as stream:
        stream.truncate(disk_mib * 1024 * 1024)
    _run(["docker", "run", "--rm", "--pull", "never", "--network", "none",
          "--user", "0:0", "--mount", f"type=bind,src={stage},dst=/input,readonly",
          "--mount", f"type=bind,src={output},dst=/output",
          "--entrypoint", "mke2fs", image_id, "-F", "-t", "ext4", "-d", "/input",
          "/output/rootfs.raw"], timeout=300)
    disk = output / "rootfs.qcow2"
    _run(["qemu-img", "convert", "-f", "raw", "-O", "qcow2", str(raw), str(disk)],
         timeout=300)
    built = {
        "schema_version": SCHEMA,
        "task_id": TASK_ID,
        "seed_workspace_sha256": SOURCE_TREE_SHA256,
        "seed_file_sha256": SOURCE_FILE_SHA256,
        "source_sdist_sha256": SOURCE_SDIST_SHA256,
        "verifier_sha256": VERIFIER_SHA256,
        "python_image_sha256": image_id,
        "architecture": architecture,
        "alpine_sha256": ALPINE_SHA256,
        "rootfs_qcow2_sha256": _sha(disk),
        "modloop_disk_sha256": _sha(output / "modloop-virt-padded.raw"),
        "rootfs_bytes": disk_mib * 1024 * 1024,
    }
    (output / "manifest.json").write_text(
        json.dumps(built, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return built


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--source-assets-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image", help="Optional explicit match to the cached image SHA")
    parser.add_argument("--disk-mib", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(prepare(args.task_dir, args.source_assets_dir, args.output,
                             image=args.image, disk_mib=args.disk_mib), indent=2))
