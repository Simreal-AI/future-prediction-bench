"""Guest-only owner of the bounded ZFS workspace recovery experiment.

Run through run_workspace_microvm. This compiles an owned fixture and creates
one file-backed pool inside the disposable guest, preserving real commands.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

if __package__:
    from .preflight import require_guest
else:
    from preflight import require_guest

MODES = ("original-rollback", "unmount-rollback-mount")
GUEST_SOURCES = ("workspace_driver.py", "workspace_probe.py", "workspace_worker.c",
                 "check_chain.py", "preflight.py")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def source_pins(source):
    manifest = source / "manifest.json"
    if manifest.is_symlink() or not manifest.is_file():
        raise ValueError("frozen_regular_source_manifest_required")
    pins = json.loads(manifest.read_bytes())
    if pins.get("schema_version") != "official-crab-workspace-frozen-source-v1":
        raise ValueError("workspace_frozen_source_schema_required")
    for name in GUEST_SOURCES:
        path = source / name
        if path.is_symlink() or not path.is_file() or sha(path) != pins["sha256"].get(name):
            raise ValueError("frozen_guest_source_pin_mismatch:" + name)
    return {name: sha(source / name) for name in GUEST_SOURCES}


def run(source, *, mode, memory_mib):
    if mode not in MODES or type(memory_mib) is not int or memory_mib not in (8, 64):
        raise ValueError("reviewed_mode_and_memory_scope_required")
    # Check the operator marker before compiling or touching any pool.
    require_guest()
    source = Path(source).absolute()
    if source != Path(__file__).absolute().parent or any(p.is_symlink() for p in (source, *source.parents)):
        raise ValueError("actual_frozen_driver_directory_required")
    before = source_pins(source)
    frozen_manifest_sha = sha(source / "manifest.json")
    root = Path("/tmp/owned-workspace-driver")
    root.mkdir(mode=0o700)
    report = {"schema_version": "official-crab-workspace-guest-driver-v1",
        "passed": False, "commands": [], "pool_created": False, "pool_destroyed": False,
        "filesystem_recovery_mode": mode, "memory_mib": memory_mib,
        "source_sha256": before, "network_used": False,
        "frozen_manifest_sha256": frozen_manifest_sha,
        "scope": "bounded_actual_RAM_workspace_FD_recovery_not_a_graded_repository_episode"}
    pool = "fpbworkspace"

    def command(argv, *, check=True, timeout=180):
        start = time.perf_counter()
        row = {"argv": argv}
        try:
            value = subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True,
                                   text=True, timeout=timeout, check=False)
            row.update(returncode=value.returncode, stdout=value.stdout, stderr=value.stderr)
        except (OSError, subprocess.TimeoutExpired) as exc:
            row.update(returncode=None, error=str(exc))
            raise
        finally:
            row["wall_seconds"] = time.perf_counter() - start
            report["commands"].append(row)
        if check and value.returncode:
            raise RuntimeError("actual_guest_command_failed:" + json.dumps(row))
        return value

    try:
        # Refuse even an existing guest pool with the same name.
        listed = command(["zpool", "list", "-H", "-o", "name"])
        if pool in listed.stdout.splitlines():
            raise ValueError("owned_pool_name_already_exists")
        worker = root / "workspace-worker"
        command(["gcc", "-O2", "-static", "-Wall", "-Wextra", "-o", str(worker),
                 str(source / "workspace_worker.c")])
        report["actual_compiled_worker_sha256"] = sha(worker)
        vdev = root / "owned-vdev.bin"
        descriptor = os.open(vdev, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        try:
            os.ftruncate(descriptor, 512 * 1024 * 1024)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        report["owned_vdev"] = {"path": str(vdev), "bytes": vdev.stat().st_size,
                                "scope": "new_guest_regular_file_never_host_block_device"}
        command(["zpool", "create", "-f", "-o", "cachefile=none", "-o", "ashift=12",
            "-O", "mountpoint=none", "-O", "atime=off", "-O", "compression=off", pool, str(vdev)])
        report["pool_created"] = True
        command(["zfs", "create", "-o", "mountpoint=none", pool + "/sandboxes"])
        argv = [sys.executable, "-B", str(source / "workspace_probe.py"),
            "--crab-source", "/opt/fpb/crab", "--existing-probe-root", str(source),
            "--worker", str(worker), "--zfs-prefix", pool + "/sandboxes",
            "--output", "/tmp/owned-workspace-probe", "--memory-mib", str(memory_mib),
            "--filesystem-recovery-mode", mode]
        value = command(argv, check=False, timeout=300)
        report["probe_execution"] = {"argv": argv, "returncode": value.returncode,
                                      "stderr": value.stderr}
        (root / "probe-stdout.json").write_text(value.stdout)
        report["probe_stdout_sha256"] = hashlib.sha256(value.stdout.encode()).hexdigest()
        report["probe_result"] = json.loads(value.stdout)
        report["passed"] = value.returncode == 0 and report["probe_result"].get("passed") is True
    except Exception as exc:
        report["error"] = str(exc)
        report["passed"] = False
    finally:
        if report["pool_created"]:
            try:
                command(["zpool", "destroy", pool])
                report["pool_destroyed"] = True
            except Exception as exc:
                report["cleanup_error"] = str(exc)
                report["passed"] = False
        try:
            report["source_sha256_after"] = source_pins(source)
            report["frozen_manifest_sha256_after"] = sha(source / "manifest.json")
            if (report["source_sha256_after"] != before or
                    report["frozen_manifest_sha256_after"] != frozen_manifest_sha):
                raise ValueError("guest_sources_changed_during_probe")
        except Exception as exc:
            report["source_guard_error"] = str(exc)
            report["passed"] = False
        (root / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--memory-mib", type=int, choices=(8, 64), required=True)
    args = parser.parse_args()
    result = run(args.source, mode=args.mode, memory_mib=args.memory_mib)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 2)
