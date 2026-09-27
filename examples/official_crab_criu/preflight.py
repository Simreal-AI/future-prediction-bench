"""Probe actual CRIU prerequisites inside an operator-created disposable VM.

This never installs packages, changes sysctls, loads kernel modules, or starts
a container. A successful probe is a prerequisite, not a checkpoint result.
"""
import argparse
import ctypes
import gzip
import hashlib
import json
import mmap
import os
from pathlib import Path
import platform
import re
import shutil
import struct
import subprocess
import tempfile

GUEST_MARKER = Path("/etc/fpb-disposable-criu-guest")
MARKER_TEXT = "Future Prediction Bench disposable CRIU guest v1\n"
ALPINE_BINARY_PINS = {
    "runc": "09653b0f4c473d6c855a0cd522dcc27a69b34b2403214ff67ebbbf22b360ec9c",
    "criu": "7603b91bf98249ccb5b956c11c1b14e746309c21b90669893e29b84eb6d8795b",
    "tar": "2d3e170780a649c3a4cd8dd3e86960644b8a66ee7a76d39d9ecabe157bf98617",
}
GNU_TAR_PATH = "/bin/tar"
GNU_TAR_VERSION_LINE = "tar (GNU tar) 1.35"


def parse_mountinfo(text):
    """Parse the kernel format, keeping mount IDs and escaped paths distinct."""
    entries = []
    def unescape(value):
        return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), value)
    for line in text.splitlines():
        before, separator, after = line.partition(" - ")
        left, right = before.split(), after.split()
        if not separator or len(left) < 6 or len(right) < 3:
            raise RuntimeError("mountinfo_record_invalid")
        entries.append({"mount_id": int(left[0]), "parent_mount_id": int(left[1]),
                        "device": left[2], "root": unescape(left[3]),
                        "mountpoint": unescape(left[4]), "mount_options": left[5],
                        "filesystem": right[0], "source": unescape(right[1]),
                        "super_options": right[2]})
    return entries


def root_mount_probe():
    """Check this probe's actual root; CRIU still checks the container root."""
    result = {"root_link": os.readlink("/proc/self/root"), "passed": False,
              "scope": "probe_process_root_only_not_full_isolation_attestation"}
    root_fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        fdinfo = Path(f"/proc/self/fdinfo/{root_fd}").read_text()
        mount_ids = [int(line.split(":", 1)[1].strip()) for line in fdinfo.splitlines()
                     if line.startswith("mnt_id:")]
        if len(mount_ids) != 1:
            raise RuntimeError("root_fd_mount_id_missing_or_ambiguous")
        result["root_fd_mount_id"] = mount_ids[0]
        entries = parse_mountinfo(Path("/proc/self/mountinfo").read_text())
        roots = [entry for entry in entries if entry["mount_id"] == mount_ids[0]]
        if len(roots) != 1:
            raise RuntimeError("root_mountinfo_id_missing_or_ambiguous")
        result["mount"] = roots[0]
        device = os.fstat(root_fd).st_dev
        result["root_stat_device"] = f"{os.major(device)}:{os.minor(device)}"
        result["passed"] = (result["root_link"] == "/" and
            roots[0]["mountpoint"] == "/" and roots[0]["root"] == "/" and
            roots[0]["filesystem"] == "ext4" and
            roots[0]["device"] == result["root_stat_device"])
        if not result["passed"]:
            result["error"] = "probe_root_must_be_actual_ext4_mount_root"
    finally:
        os.close(root_fd)
    return result


def elf_identity(path):
    with Path(path).open("rb") as stream:
        header = stream.read(20)
    valid = (len(header) == 20 and header[:4] == b"\x7fELF" and
             header[4:7] == bytes((2, 1, 1)))
    elf_type, machine = struct.unpack("<HH", header[16:20]) if valid else (None, None)
    return {"elf64_little_endian": valid, "elf_type": elf_type,
            "machine": machine, "x86_64_executable": valid and machine == 62 and
            elf_type in (2, 3)}


def gnu_tar_roundtrip(tar_path):
    """Exercise v4.2 tmpfs tar flags on bounded files owned by this probe.

    The original flags come from CRIU v4.2 criu/filesystems.c:396-400,428-431.
    No process/container state or existing guest file is modified.
    """
    with tempfile.TemporaryDirectory(prefix="fpb-gnu-tar-") as temporary:
        root = Path(temporary)
        source, restored = root / "source", root / "restored"
        source.mkdir()
        restored.mkdir()
        payload = b"Future Prediction Bench CRIU GNU tar capability probe\n"
        filename = "owned space [literal]\\name.txt"
        (source / filename).write_bytes(payload)
        archive = root / "owned.tar.gz"
        create = command([tar_path, "--create", "--gzip", "--no-unquote",
            "--no-wildcards", "--one-file-system", "--check-links",
            "--preserve-permissions", "--sparse", "--numeric-owner",
            "--directory", str(source), "--file", str(archive), "."])
        extract = command([tar_path, "--extract", "--gzip", "--no-unquote",
            "--no-wildcards", "--directory", str(restored), "--file", str(archive)])
        recovered = restored / filename
        exact = recovered.is_file() and recovered.read_bytes() == payload
        return {"create": create, "extract": extract,
                "archive_bytes": archive.stat().st_size if archive.exists() else None,
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
                "payload_restored_exactly": exact,
                "passed": create.get("returncode") == 0 and
                          extract.get("returncode") == 0 and exact,
                "scope": "owned_tiny_tar_roundtrip_not_process_checkpoint"}


def require_guest():
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("disposable_x86_64_linux_guest_required")
    if os.geteuid() != 0:
        raise RuntimeError("guest_root_required")
    if GUEST_MARKER.is_symlink() or not GUEST_MARKER.is_file():
        raise RuntimeError("operator_created_disposable_guest_marker_required")
    if GUEST_MARKER.read_text() != MARKER_TEXT:
        raise RuntimeError("disposable_guest_marker_mismatch")


def command(argv, timeout=30):
    try:
        result = subprocess.run(argv, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            timeout=timeout, check=False)
        return {"argv": argv, "returncode": result.returncode,
                "stdout": result.stdout[-12000:], "stderr": result.stderr[-12000:]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"argv": argv, "returncode": None, "error": str(exc)}


def soft_dirty_roundtrip():
    """Check the real kernel bit on one owned anonymous writable page."""
    size = os.sysconf("SC_PAGE_SIZE")
    page = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    try:
        page[0] = 19
        view = ctypes.c_char.from_buffer(page)
        address = ctypes.addressof(view)
        del view
        def entry():
            with open("/proc/self/pagemap", "rb", buffering=0) as stream:
                stream.seek((address // size) * 8)
                raw = stream.read(8)
            if len(raw) != 8:
                raise RuntimeError("pagemap_short_read")
            return struct.unpack("Q", raw)[0]
        Path("/proc/self/clear_refs").write_text("4\n")
        cleared = entry()
        page[0] = 20
        changed = entry()
        return {"supported": bool(cleared & (1 << 63)) and
                not bool(cleared & (1 << 55)) and bool(changed & (1 << 55)),
                "present_after_clear": bool(cleared & (1 << 63)),
                "dirty_after_clear": bool(cleared & (1 << 55)),
                "dirty_after_write": bool(changed & (1 << 55))}
    finally:
        page.close()


def collect():
    require_guest()
    report = {"schema_version": "official-crab-criu-preflight-v1",
              "system": platform.system(), "architecture": platform.machine(),
              "kernel": platform.release(), "gpu_required": False,
              "scope": "process_CR_and_dirty_memory_tracking_only"}
    for name in ("criu", "runc", "nft", "iptables-restore", "ip6tables-restore",
                 "tar", "gcc", "cc", "zfs", "zpool", "clang"):
        report[name + "_path"] = shutil.which(name)
    report["versions"] = {name: command([name, "--version"])
                          for name in ALPINE_BINARY_PINS if shutil.which(name)}
    report["binary_sha256"] = {name: hashlib.sha256(Path(shutil.which(name)).read_bytes()).hexdigest()
                               for name in ALPINE_BINARY_PINS if shutil.which(name)}
    report["matches_alpine_binary_pins"] = report["binary_sha256"] == ALPINE_BINARY_PINS
    try:
        report["root_mount"] = root_mount_probe()
    except (OSError, RuntimeError, ValueError) as exc:
        report["root_mount"] = {"passed": False, "error": str(exc)}
    tar = {"path": report["tar_path"], "expected_resolved_path": GNU_TAR_PATH,
           "expected_version_line": GNU_TAR_VERSION_LINE,
           "expected_sha256": ALPINE_BINARY_PINS["tar"], "passed": False}
    report["gnu_tar"] = tar
    if report["tar_path"]:
        try:
            tar["resolved_path"] = str(Path(report["tar_path"]).resolve(strict=True))
            tar["elf"] = elf_identity(report["tar_path"])
            tar["sha256"] = report["binary_sha256"]["tar"]
            tar["version"] = report["versions"]["tar"]
            lines = tar["version"].get("stdout", "").splitlines()
            tar["identity_passed"] = bool(
                tar["resolved_path"] == GNU_TAR_PATH and
                tar["elf"]["x86_64_executable"] and
                tar["sha256"] == ALPINE_BINARY_PINS["tar"] and
                tar["version"].get("returncode") == 0 and
                lines and lines[0] == GNU_TAR_VERSION_LINE)
            if tar["identity_passed"]:
                tar["capability"] = gnu_tar_roundtrip(tar["resolved_path"])
                tar["passed"] = tar["capability"]["passed"]
            else:
                tar["error"] = "reviewed_gnu_tar_identity_required"
        except (OSError, RuntimeError, ValueError) as exc:
            tar["error"] = str(exc)
    else:
        tar["error"] = "gnu_tar_missing"
    config_path = Path("/proc/config.gz")
    if config_path.exists():
        config = gzip.decompress(config_path.read_bytes()).decode()
    else:
        config_path = Path("/boot/config-" + platform.release())
        config = config_path.read_text() if config_path.exists() else ""
    names = ("CHECKPOINT_RESTORE", "MEM_SOFT_DIRTY", "NAMESPACES", "PID_NS",
             "IPC_NS", "UTS_NS", "NET_NS", "UNIX_DIAG", "INET_DIAG",
             "PACKET_DIAG", "NETLINK_DIAG", "USERFAULTFD", "BPF", "BPF_SYSCALL")
    report["kernel_config"] = {name: next((line.split("=", 1)[1]
        for line in config.splitlines() if line.startswith("CONFIG_" + name + "=")),
        "unknown") for name in names}
    try:
        report["soft_dirty"] = soft_dirty_roundtrip()
    except (OSError, RuntimeError) as exc:
        report["soft_dirty"] = {"supported": False, "error": str(exc)}
    report["criu_check"] = command(["criu", "check"])
    report["mem_dirty_track"] = command(["criu", "check", "--feature", "mem_dirty_track"])
    report["network_lock_backend"] = "iptables_default"
    report["network_lock_bypass"] = False
    report["iptables_restore_version"] = command(["iptables-restore", "--version"])
    report["ip6tables_restore_version"] = command(["ip6tables-restore", "--version"])
    runc_config = Path("/etc/criu/runc.conf")
    report["default_runc_criu_config_present"] = runc_config.exists()
    report["default_runc_criu_config_sha256"] = hashlib.sha256(runc_config.read_bytes()).hexdigest() if runc_config.exists() else None
    report["default_runc_criu_config_option_free"] = not runc_config.exists() or all(
        not line.strip() or line.lstrip().startswith("#")
        for line in runc_config.read_text().splitlines())
    report["nftables_readonly_probe"] = command(["nft", "list", "ruleset"])
    report["process_chain_prerequisites_passed"] = bool(report["runc_path"] and
        report["criu_path"] and report["matches_alpine_binary_pins"] and
        report["root_mount"]["passed"] and report["gnu_tar"]["passed"] and
        report["soft_dirty"]["supported"] and
        report["iptables-restore_path"] and report["ip6tables-restore_path"] and
        report["iptables_restore_version"].get("returncode") == 0 and
        report["ip6tables_restore_version"].get("returncode") == 0 and
        report["default_runc_criu_config_option_free"] and
        report["criu_check"].get("returncode") == 0 and
        report["mem_dirty_track"].get("returncode") == 0)
    report["full_crab_backend_verified"] = False
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    try:
        result = collect()
    except (OSError, RuntimeError) as exc:
        result = {"schema_version": "official-crab-criu-preflight-v1",
                  "process_chain_prerequisites_passed": False, "error": str(exc)}
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(encoded)
    print(encoded, end="")
    raise SystemExit(0 if result.get("process_chain_prerequisites_passed") else 2)
