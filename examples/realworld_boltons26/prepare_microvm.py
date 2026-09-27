"""Build an offline ARM64 Boltons microVM image from pinned public inputs.

Requires QEMU, Docker, and a locally cached linux/arm64 image containing
Python 3.12 and mke2fs. Docker is only an image-building tool here; the
resulting coding episode executes in QEMU/HVF with no NIC or host mount.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import uuid
from pathlib import Path
from urllib.request import urlopen

from future_prediction_bench.coding_env import _workspace_digest, _workspace_files


ALPINE_BASE = "https://dl-cdn.alpinelinux.org/alpine/v3.24/releases/aarch64/netboot/"
ALPINE_SHA256 = {
    "vmlinuz-virt": "e45e1f6083d1ed45db6647b422e32b6ae6dc54de7b8190b7b97744fb293412e3",
    "initramfs-virt": "ffe65ec5a0c0bf470042ad28f7ce7aa5f842ce8090e4230fb2703a7a34e1bebe",
    "modloop-virt": "32a189d5e4ae4417cf65bfcc91d2cc8ffdf5957fbbc1f055b037861c595b7328",
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(argv, *, timeout=180):
    completed = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, timeout=timeout, check=False)
    if completed.returncode:
        raise RuntimeError(f"{argv[0]} failed ({completed.returncode}): "
                           + completed.stderr[-2000:].decode("utf-8", "replace"))
    return completed.stdout.decode("utf-8", "replace").strip()


def _download_or_copy(name: str, target: Path, source_dir: Path | None):
    if source_dir is None:
        with urlopen(ALPINE_BASE + name, timeout=30) as response, target.open("wb") as output:
            shutil.copyfileobj(response, output)
    else:
        shutil.copy2(source_dir / name, target)
    if _sha(target) != ALPINE_SHA256[name]:
        raise ValueError(f"Pinned Alpine {name} digest mismatch")


def prepare(task_dir, output, *, image, alpine_source_dir=None, disk_mib=256):
    task_dir, output = Path(task_dir).resolve(), Path(output).resolve()
    source_dir = Path(alpine_source_dir).resolve() if alpine_source_dir else None
    seed = task_dir / "seed"
    if not seed.is_dir() or not (task_dir / "verifier" / "verify.json").is_file():
        raise ValueError("A generated Boltons task with seed and verifier is required")
    _workspace_files(seed)
    task = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    if (task.get("task_id") != "boltons-26-singularize-ss-v1" or not task.get("is_fixture")
            or task.get("metadata", {}).get("source_sdist_sha256")
            != "5566d6cfd5a1e873d8e8476496287a9f92979964611ad9a9cecb6b0ef29b1edd"):
        raise ValueError("This image builder only accepts the pinned public Boltons fixture")
    seed_sha = _workspace_digest(seed)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be a new or empty directory")
    if not 128 <= disk_mib <= 4096:
        raise ValueError("disk_mib must be 128..4096")
    image_id = _run(["docker", "image", "inspect", image, "--format", "{{.Id}}"])
    architecture = _run(["docker", "image", "inspect", image,
                         "--format", "{{.Os}}/{{.Architecture}}"])
    if architecture != "linux/arm64" or not image_id.startswith("sha256:"):
        raise ValueError("A locally cached immutable linux/arm64 image is required")
    output.mkdir(parents=True, exist_ok=True)
    stage = output / "build-stage"
    stage.mkdir()
    for name in ALPINE_SHA256:
        _download_or_copy(name, output / name, source_dir)
    modloop = output / "modloop-virt"
    padded = output / "modloop-virt-padded.raw"
    with modloop.open("rb") as source, padded.open("wb") as destination:
        shutil.copyfileobj(source, destination)
        destination.write(b"\0" * (-modloop.stat().st_size % 512))

    container = "fpb-asset-" + uuid.uuid4().hex[:18]
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
    _run(["qemu-img", "convert", "-f", "raw", "-O", "qcow2", str(raw), str(disk)], timeout=300)
    manifest = {"schema_version": "boltons-microvm-assets-v2", "task_dir": str(task_dir),
                "task_id": task["task_id"], "seed_workspace_sha256": seed_sha,
                "source_sdist_sha256": task["metadata"]["source_sdist_sha256"],
                "python_image_sha256": image_id, "architecture": architecture,
                "alpine_source": ALPINE_BASE, "alpine_sha256": ALPINE_SHA256,
                "rootfs_qcow2_sha256": _sha(disk), "modloop_disk_sha256": _sha(padded),
                "rootfs_bytes": disk_mib * 1024 * 1024}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image", required=True, help="Locally cached linux/arm64 image with Python 3.12 and mke2fs")
    parser.add_argument("--alpine-source-dir", help="Use previously downloaded pinned Alpine files")
    parser.add_argument("--disk-mib", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(prepare(args.task_dir, args.output, image=args.image,
                             alpine_source_dir=args.alpine_source_dir,
                             disk_mib=args.disk_mib), indent=2))
