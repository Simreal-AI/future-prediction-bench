"""Operator-facing construction API for the ignored resident-guest candidate.

The caller supplies a compatible, already booted Linux/arm64 guest runtime.
This module installs only reviewed helpers. It never transfers verifier
expectations or rewards into the guest. The runtime must expose ``run_shell``.
The operator must boot the clone of assets checked by ``verify_assets``:
this serial-shell connector does not attest a hostile guest, boot chain, or
privileged operator. The guest-root actor is trusted in this prototype.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import shlex
import time
from datetime import datetime, timedelta
from pathlib import Path

from future_prediction_bench.realworld import AdapterInfrastructureError, RealWorldEnv
from host_client import GuestSerialTransport, ResidentHostAdapter
from host_realworld_adapter import HostPrivateCases, ResidentRealWorldAdapter


GUEST_HELPERS = {
    "wire.py": "/mnt/root/fpb_resident_wire.py",
    "guest_supervisor.py": "/mnt/root/fpb_resident_guest.py",
    "case_runner.py": "/mnt/root/fpb_resident_case_runner.py",
    "hardened_edit.py": "/mnt/root/fpb_resident_hardened_edit.py",
}


def load_fixture(*, task_path, verifier_path, contract_path,
                 assets_manifest_path):
    """Freeze operator-supplied trusted paths and all helper digests."""
    return HostPrivateCases(task_path, verifier_path, contract_path,
                            assets_manifest_path)


def verify_assets(private: HostPrivateCases, assets_dir):
    """Check the VM input bytes before cloning or booting a guest."""
    root = Path(assets_dir)
    manifest_path = root / "manifest.json"
    if manifest_path.resolve() != private.paths["assets_manifest"].resolve():
        raise ValueError("resident_assets_manifest_path_mismatch")
    private.check_unchanged()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pins = {
        "rootfs.qcow2": manifest["rootfs_qcow2_sha256"],
        "vmlinuz-virt": manifest["alpine_sha256"]["vmlinuz-virt"],
        "initramfs-virt": manifest["alpine_sha256"]["initramfs-virt"],
        "modloop-virt-padded.raw": manifest["modloop_disk_sha256"],
    }
    for name, expected in pins.items():
        path = root / name
        if (path.is_symlink() or not path.is_file()
                or not isinstance(expected, str)
                or re.fullmatch(r"[0-9a-f]{64}", expected) is None):
            raise ValueError("resident_asset_missing_or_pin_invalid: " + name)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError("resident_asset_digest_mismatch: " + name)
    return {"task_id": private.task_id, "source_path": private.source_path,
            "case_count": len(private.cases), "assets_manifest_sha256":
            private.binding["assets_manifest_sha256"]}


def _required(runtime, command, *, timeout=30):
    result = runtime.run_shell(command, timeout=timeout)
    if result.get("return_code") != 0:
        raise RuntimeError("resident_guest_setup_failed")
    return result.get("stdout", "")


def _upload(runtime, source, target):
    data = Path(source).read_bytes()
    if not 0 < len(data) <= 65536:
        raise ValueError("resident_helper_size_invalid")
    _required(runtime, ": > " + shlex.quote(target))
    for offset in range(0, len(data), 2700):
        encoded = base64.b64encode(data[offset:offset + 2700]).decode("ascii")
        _required(runtime, "printf '%s' '" + encoded + "' | base64 -d >> "
                  + shlex.quote(target))
    output = _required(runtime, "sha256sum " + shlex.quote(target))
    match = re.fullmatch(r"([0-9a-f]{64})\s+\S+\n?", output)
    digest = hashlib.sha256(data).hexdigest()
    if match is None or match.group(1) != digest:
        raise RuntimeError("resident_uploaded_helper_digest_mismatch")
    return digest


def connect_booted_vm(runtime, private: HostPrivateCases, *, timeout=20.0):
    """Install helpers into the public microVM rootfs layout and connect.

    This checks the guest's outer source bytes before installation. The guest
    checks them again on startup and on each branch creation. The caller must
    have checked and booted the pinned assets; this is no boot attestation or
    defense against a hostile operator or root user inside the guest.
    """
    if not isinstance(private, HostPrivateCases):
        raise TypeError("host_private_cases_required")
    private.check_unchanged()
    guest_source = "/mnt/root/workspace/" + private.source_path
    output = _required(runtime, "sha256sum " + shlex.quote(guest_source))
    match = re.fullmatch(r"([0-9a-f]{64})\s+\S+\n?", output)
    if match is None or match.group(1) != private.seed_file_sha256:
        raise RuntimeError("resident_outer_workspace_seed_mismatch")
    _required(runtime, "mkdir -p /mnt/root/dev && "
              "(test -c /mnt/root/dev/null || mknod -m 666 /mnt/root/dev/null c 1 3)")
    for name, target in GUEST_HELPERS.items():
        if _upload(runtime, private.helper_paths[name], target) != private.helper_digests[name]:
            raise RuntimeError("resident_helper_install_binding_mismatch")
    command = ("LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/aarch64-linux-gnu "
               "chroot /mnt/root /usr/local/bin/python3.12 -I -B "
               "/fpb_resident_guest.py --serve --source-path "
               + shlex.quote(private.source_path) + " --seed-sha256 "
               + shlex.quote(private.seed_file_sha256)
               + " --case-count " + str(len(private.cases))
               + " </dev/null >/mnt/root/fpb_resident.log 2>&1 & true")
    _required(runtime, command)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if runtime.run_shell("test -S /mnt/root/fpb-resident.sock", timeout=5).get("return_code") == 0:
            break
        time.sleep(0.05)
    else:
        raise RuntimeError("resident_guest_daemon_not_ready")
    client = ResidentHostAdapter(GuestSerialTransport(runtime, timeout=timeout))
    client.connect()
    if (client.source_path != private.source_path
            or client.seed_file_sha256 != private.seed_file_sha256
            or client.case_count != len(private.cases)
            or client.guest_helper_sha256 != {name: private.helper_digests[name]
                                              for name in GUEST_HELPERS}
            or client.binding["prototype_sha256"] != private.binding["prototype_sha256"]):
        raise RuntimeError("resident_running_guest_binding_mismatch")
    return client


def make_env(client, private: HostPrivateCases, *, episode_mode="repair",
             case_transport="sequential", clock=None):
    """Construct one RealWorldEnv episode from a connected, idle client."""
    adapter = ResidentRealWorldAdapter(
        client, private, infrastructure_error_class=AdapterInfrastructureError,
        episode_mode=episode_mode, case_transport=case_transport)
    task = private.experimental_task()
    if clock is None:
        issued = datetime.fromisoformat(task["issued_at"])
        clock = lambda: issued + timedelta(seconds=1)
    return RealWorldEnv(task, adapter, clock=clock)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-path", required=True)
    parser.add_argument("--verifier-path", required=True)
    parser.add_argument("--contract-path", required=True)
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--check-assets", action="store_true")
    args = parser.parse_args(argv)
    private = load_fixture(
        task_path=args.task_path, verifier_path=args.verifier_path,
        contract_path=args.contract_path,
        assets_manifest_path=Path(args.assets_dir) / "manifest.json")
    result = {"status": "contract_valid", "task_id": private.task_id,
              "source_path": private.source_path, "case_count": len(private.cases),
              "binding": private.binding}
    if args.check_assets:
        verify_assets(private, args.assets_dir)
        result["status"] = "assets_valid"
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
