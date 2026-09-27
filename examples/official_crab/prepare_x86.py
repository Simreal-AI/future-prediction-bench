"""Build pinned x86 probe assets offline from operator-downloaded inputs.

Docker runs only cached ARM mke2fs as an image-building utility. Guest Python
is the pinned standalone musl build; guest execution uses real x86 QEMU/TCG.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

from future_prediction_bench.coding_env import _workspace_digest
from examples.realworld_boltons26.prepare_microvm import _run, _sha

PINS = {
    "vmlinuz-virt": "40f620bc8c93d952e57dd8dfc0f94fca1759d192a4fc4a260705d50ca378559c",
    "initramfs-virt": "c990c63e4602aa84b92d7df54fd180cb0e56590d61b71221ba6d60e913d26357",
    "modloop-virt": "1e7a3eea707d2ecb3d0502f97fb4dc4dcbac1fe06075e24ae19bfdf4343bb1cf",
    "python-musl.tar.gz": "bb882e825aad8c76c4f16b84276e3f0205ac27a1588c42a3c38e2cf667592a20",
    "musl.apk": "573712e2f49c15bfc20a2699f204acdfc74c772722b15e7353d768057fae0e71",
}
PYTHON_URL = "https://github.com/astral-sh/python-build-standalone/releases/download/20260924/cpython-3.12.14%2B20260924-x86_64-unknown-linux-musl-install_only_stripped.tar.gz"
ALPINE_URL = "https://dl-cdn.alpinelinux.org/alpine/v3.24/releases/x86_64/netboot/"


def prepare(source, task_dir, output, *, build_image):
    source, task_dir, output = [Path(p).resolve() for p in (source, task_dir, output)]
    if output.exists() or any(output.is_relative_to(p) or p.is_relative_to(output)
                              for p in (source, task_dir)):
        raise ValueError("new_disjoint_output_required")
    for name, expected in PINS.items():
        if (source / name).is_symlink() or _sha(source / name) != expected:
            raise ValueError("input_pin_mismatch: " + name)
    task = json.loads((task_dir / "task.json").read_text())
    if task["task_id"] != "boltons-26-singularize-ss-v2":
        raise ValueError("pinned_boltons_task_required")
    architecture = _run(["docker", "image", "inspect", build_image, "--format", "{{.Os}}/{{.Architecture}}"])
    image_id = _run(["docker", "image", "inspect", build_image, "--format", "{{.Id}}"])
    if architecture != "linux/arm64" or not image_id.startswith("sha256:"):
        raise ValueError("cached_arm_build_image_required")
    output.mkdir(parents=True)
    stage = output / "stage"
    stage.mkdir()
    with tarfile.open(source / "python-musl.tar.gz") as archive:
        # Linux terminfo aliases collide on the host's case-insensitive APFS.
        # This no-terminal probe needs only bin/lib; no terminfo is installed.
        members = [m for m in archive.getmembers() if m.name == "python" or
                   m.name.startswith(("python/bin/", "python/lib/", "python/include/"))]
        archive.extractall(stage, members=members, filter="data")
    (stage / "usr").mkdir()
    (stage / "python").rename(stage / "usr/local")
    with tarfile.open(source / "musl.apk") as archive:
        archive.extractall(stage, members=[m for m in archive.getmembers()
                           if m.name.startswith("lib/")], filter="data")
    shutil.copytree(task_dir / "seed", stage / "workspace")
    for name in ("vmlinuz-virt", "initramfs-virt", "modloop-virt"):
        shutil.copyfile(source / name, output / name)
    module = output / "modloop-virt-padded.raw"
    shutil.copyfile(output / "modloop-virt", module)
    with module.open("ab") as stream:
        stream.write(b"\0" * (-module.stat().st_size % 512))
    raw = output / "rootfs.raw"
    with raw.open("wb") as stream:
        stream.truncate(256 * 1024 * 1024)
    _run(["docker", "run", "--rm", "--pull", "never", "--network", "none", "--user", "0:0",
          "--mount", f"type=bind,src={stage},dst=/input,readonly",
          "--mount", f"type=bind,src={output},dst=/output", "--entrypoint", "mke2fs",
          image_id, "-F", "-t", "ext4", "-d", "/input", "/output/rootfs.raw"], timeout=300)
    disk = output / "rootfs.qcow2"
    _run(["qemu-img", "convert", "-f", "raw", "-O", "qcow2", str(raw), str(disk)], timeout=300)
    manifest = {"schema_version": "boltons-microvm-assets-v2", "architecture": "linux/amd64",
        "task_id": task["task_id"], "seed_workspace_sha256": _workspace_digest(task_dir / "seed"),
        "source_sdist_sha256": task["metadata"]["source_sdist_sha256"],
        "alpine_sha256": {name: PINS[name] for name in ("vmlinuz-virt", "initramfs-virt", "modloop-virt")},
        "alpine_source": ALPINE_URL, "alpine_pin_basis": "HTTPS-fetched bytes pinned on 2026-09-27",
        "python_source": PYTHON_URL, "python_archive_sha256": PINS["python-musl.tar.gz"],
        "musl_source": "https://dl-cdn.alpinelinux.org/alpine/v3.24/main/x86_64/musl-1.2.6-r2.apk",
        "musl_apk_sha256": PINS["musl.apk"],
        "python_pin_basis": "GitHub release asset SHA256", "build_image_sha256": image_id,
        "rootfs_qcow2_sha256": _sha(disk), "modloop_disk_sha256": _sha(module),
        "rootfs_bytes": 256 * 1024 * 1024}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    raw.unlink()
    shutil.rmtree(stage)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--build-image", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.task_dir, args.output, build_image=args.build_image), indent=2))
