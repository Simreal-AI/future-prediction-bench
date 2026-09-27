"""Host-only QEMU runtime for real Linux VM sandbox experiments.

This is an isolated VM transport, not a policy tool. Only trusted task adapters
may issue fixed guest shell commands. QEMU ``savevm``/``loadvm`` stores VM
CPU/memory/device state together with its writable qcow2 disk; this differs
from the filesystem-only Docker branch adapter. The caller owns the guest
image and must supply a dedicated writable disk per concurrent VM.
"""

from __future__ import annotations

import hashlib
import ctypes
from concurrent.futures import ThreadPoolExecutor
import json
import os
import re
import secrets
import select
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path


class MicroVMRuntimeError(RuntimeError):
    """VM infrastructure failure; never interpret this as a policy reward."""


_BACKENDS = {
    "aarch64_hvf": ("qemu-system-aarch64", "virt,accel=hvf", "host",
                    "console=ttyAMA0", "virtio-blk-device", "virtio-serial-device"),
    "x86_64_tcg": ("qemu-system-x86_64", "q35,accel=tcg", "max",
                   "console=ttyS0", "virtio-blk-pci", "virtio-serial-pci"),
    "x86_64_kvm": ("qemu-system-x86_64", "q35,accel=kvm", "host",
                   "console=ttyS0", "virtio-blk-pci", "virtio-serial-pci"),
}


def _require_backend(value):
    if not isinstance(value, str) or value not in _BACKENDS:
        raise ValueError("backend must be aarch64_hvf, x86_64_tcg, or x86_64_kvm")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha(value: str, field: str):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")


def _file_identity(path: Path):
    """Metadata guard for an already-hashed, trusted host artifact.

    A guest cannot write a ``readonly=on`` block device. Changes from another
    host process alter the inode identity or its modification/change times;
    this guard fails closed instead of silently using changed bytes. It does
    not replace a fresh digest when the host itself is adversarial.
    """
    try:
        info = path.lstat()
    except OSError as exc:
        raise MicroVMRuntimeError("readonly_disk_changed_since_vm_start") from exc
    if not stat.S_ISREG(info.st_mode):
        raise MicroVMRuntimeError("readonly_disk_changed_since_vm_start")
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
            info.st_ctime_ns, info.st_mode)


def _quarantine_marker(path: Path) -> Path:
    return path.with_name(path.name + ".fpb-unsafe")


def _reject_quarantined_disk(path: Path):
    try:
        _quarantine_marker(path).lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise MicroVMRuntimeError("qcow2_quarantine_check_failed") from exc
    raise MicroVMRuntimeError("qcow2_disk_quarantined")


def _quarantine_disk(path: Path) -> bool:
    """Durably bar this qcow2 from our VM/clone helpers after uncertain save."""
    marker = _quarantine_marker(path)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(marker, flags, 0o600)
    except FileExistsError:
        return True
    except OSError:
        return False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(b"Unverified QEMU snapshot cleanup; discard this disk.\n")
            stream.flush()
            os.fsync(stream.fileno())
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError:
        return False
    return True


def _clone_or_copy_qcow2(source: Path, target: Path) -> str:
    """Publish a separate file atomically; never hard-link it to the source.

    The private staging directory is on the target filesystem. macOS
    ``clonefile`` produces an independent copy-on-write inode on APFS; a
    regular byte copy is used if that operation is unavailable. ``os.link``
    only publishes the completed staging file at a target that does not yet
    exist, then the staging name is removed.
    """
    _reject_quarantined_disk(source)
    staging_dir = Path(tempfile.mkdtemp(prefix=".fpb-vm-fork-", dir=target.parent))
    staging = staging_dir / "disk.qcow2"
    mode = "copy"
    try:
        if sys.platform == "darwin":
            try:
                clonefile = ctypes.CDLL(None, use_errno=True).clonefile
                clonefile.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
                clonefile.restype = ctypes.c_int
                if clonefile(os.fsencode(source), os.fsencode(staging), 0) == 0:
                    mode = "clonefile"
            except (AttributeError, OSError):
                pass
        if mode == "copy":
            # A failed clonefile may leave a partial destination, but the
            # staging directory is private to this call.
            staging.unlink(missing_ok=True)
            with source.open("rb") as reader, staging.open("xb") as writer:
                shutil.copyfileobj(reader, writer, 1024 * 1024)
        os.link(staging, target)
        return mode
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


class MicroVMRuntime:
    """One VM instance with no NIC, host directory mounts, or policy shell API."""

    def __init__(self, kernel_path, initramfs_path, disk_path, *, kernel_sha256,
                 initramfs_sha256, readonly_disk_paths=(), memory_mib=512, vcpus=1,
                 backend="aarch64_hvf", kernel_append=None, qemu_binary=None,
                 qemu_img_binary="qemu-img", command_timeout=30.0,
                 popen_factory=None, enable_action_port=False):
        _require_backend(backend)
        backend_configuration = _BACKENDS[backend]
        if kernel_append is None:
            kernel_append = backend_configuration[3]
        if qemu_binary is None:
            qemu_binary = backend_configuration[0]
        if (not isinstance(readonly_disk_paths, (list, tuple))
                or len(readonly_disk_paths) > 4):
            raise ValueError("readonly_disk_paths must contain at most four paths")
        for path in (kernel_path, initramfs_path, disk_path, *readonly_disk_paths):
            if Path(path).is_symlink():
                raise ValueError("VM artifacts cannot be symlinks")
        self.kernel_path = Path(kernel_path).resolve()
        self.initramfs_path = Path(initramfs_path).resolve()
        self.disk_path = Path(disk_path).resolve()
        self.readonly_disk_paths = tuple(Path(path).resolve() for path in readonly_disk_paths)
        all_paths = (self.kernel_path, self.initramfs_path, self.disk_path,
                     *self.readonly_disk_paths)
        if len(set(all_paths)) != len(all_paths):
            raise ValueError("VM artifact paths must be disjoint")
        _require_sha(kernel_sha256, "kernel_sha256")
        _require_sha(initramfs_sha256, "initramfs_sha256")
        self.kernel_sha256 = kernel_sha256
        self.initramfs_sha256 = initramfs_sha256
        if type(memory_mib) is not int or not 128 <= memory_mib <= 8192:
            raise ValueError("memory_mib must be in [128, 8192]")
        if type(vcpus) is not int or not 1 <= vcpus <= 8:
            raise ValueError("vcpus must be in [1, 8]")
        if (not isinstance(kernel_append, str) or not kernel_append
                or len(kernel_append) > 512 or any(c in kernel_append for c in "\x00\r\n")):
            raise ValueError("Invalid guest kernel command line")
        if (isinstance(command_timeout, bool) or not isinstance(command_timeout, (int, float))
                or not 0 < command_timeout <= 300):
            raise ValueError("command_timeout must be in (0, 300]")
        for value in (qemu_binary, qemu_img_binary):
            if not isinstance(value, str) or not value or any(c in value for c in "\x00\r\n"):
                raise ValueError("Invalid QEMU executable")
        if type(enable_action_port) is not bool:
            raise ValueError("enable_action_port must be boolean")
        self.memory_mib = memory_mib
        self.vcpus = vcpus
        self.backend = backend
        self.kernel_append = kernel_append
        self.qemu_binary = qemu_binary
        self.qemu_img_binary = qemu_img_binary
        self.command_timeout = float(command_timeout)
        self.enable_action_port = enable_action_port
        self._popen_factory = popen_factory or subprocess.Popen
        self._process = None
        self._monitor = None
        self._serial = None
        self._action_port = None
        # An action-port VM cannot be snapshotted while its guest RPC process
        # is live. A trusted adapter may retire the port once, after STOP and
        # guest-process quiescence; it can never be opened again in this VM.
        self._action_port_retired = False
        self._action_port_disconnect_attested = False
        self._action_port_listener_unlinked = False
        self._monitor_buffer = b""
        self._serial_buffer = b""
        self._socket_dir = None
        self._stderr_thread = None
        self._stderr_tail = bytearray()
        self._stderr_total = 0
        self._stderr_exceeded = False
        self._snapshots = set()
        self._readonly_disk_sha256s = ()
        self._readonly_disk_identities = ()
        self.metrics = {"vm_starts": 0, "guest_commands": 0,
                        "snapshot_saves": 0, "snapshot_loads": 0,
                        "forks_spawned": 0, "fork_reflink_disks": 0,
                        "fork_copied_disks": 0, "fork_seconds": 0.0,
                        "fork_disk_clone_seconds": 0.0,
                        "fork_child_start_restore_seconds": 0.0,
                        "vm_start_seconds": 0.0, "guest_command_seconds": 0.0,
                        "snapshot_save_seconds": 0.0, "snapshot_load_seconds": 0.0}

    def _check_artifacts(self):
        _reject_quarantined_disk(self.disk_path)
        for path, expected, label in (
            (self.kernel_path, self.kernel_sha256, "kernel"),
            (self.initramfs_path, self.initramfs_sha256, "initramfs"),
        ):
            if not path.is_file() or path.is_symlink():
                raise MicroVMRuntimeError(f"{label}_missing_or_symlink")
            if _sha256_file(path) != expected:
                raise MicroVMRuntimeError(f"{label}_digest_mismatch")
        if not self.disk_path.is_file() or self.disk_path.is_symlink():
            raise MicroVMRuntimeError("dedicated_qcow2_disk_required")
        readonly_hashes = []
        readonly_identities = []
        for path in self.readonly_disk_paths:
            if not path.is_file() or path.is_symlink():
                raise MicroVMRuntimeError("readonly_disk_missing_or_symlink")
            before = _file_identity(path)
            readonly_hashes.append(_sha256_file(path))
            after = _file_identity(path)
            if before != after:
                raise MicroVMRuntimeError("readonly_disk_changed_during_hash")
            readonly_identities.append(after)
        self._readonly_disk_sha256s = tuple(readonly_hashes)
        self._readonly_disk_identities = tuple(readonly_identities)
        for path in (self.kernel_path, self.initramfs_path, self.disk_path,
                     *self.readonly_disk_paths):
            if "," in str(path) or "\n" in str(path):
                raise MicroVMRuntimeError("qemu_path_contains_unsupported_character")
        try:
            result = subprocess.run(
                [self.qemu_img_binary, "info", "--output=json", str(self.disk_path)],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, timeout=10, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise MicroVMRuntimeError("qemu_img_unavailable_or_timed_out") from exc
        if result.returncode != 0 or len(result.stdout) > 65536 or len(result.stderr) > 65536:
            raise MicroVMRuntimeError("qemu_img_info_failed")
        try:
            info = json.loads(result.stdout)
        except (ValueError, UnicodeError) as exc:
            raise MicroVMRuntimeError("qemu_img_info_invalid") from exc
        if info.get("format") != "qcow2":
            raise MicroVMRuntimeError("writable_qcow2_disk_required")
        if "backing-filename" in info:
            raise MicroVMRuntimeError("qcow2_backing_file_unsupported")

    def _stderr_reader(self, stream):
        while True:
            try:
                chunk = stream.read1(4096) if hasattr(stream, "read1") else stream.read(4096)
            except (OSError, ValueError):
                break
            if not chunk:
                break
            self._stderr_total += len(chunk)
            self._stderr_tail.extend(chunk)
            if len(self._stderr_tail) > 8192:
                del self._stderr_tail[:-8192]
            if self._stderr_total > 1_048_576:
                self._stderr_exceeded = True
                if self._process is not None and self._process.poll() is None:
                    self._process.kill()
                break

    def _check_running(self):
        if self._process is None:
            raise MicroVMRuntimeError("vm_not_started")
        if self._stderr_exceeded:
            raise MicroVMRuntimeError("vm_stderr_output_limit_exceeded")
        if self._process.poll() is not None:
            raise MicroVMRuntimeError("vm_exited")

    def _connect(self, path: Path, deadline: float):
        while time.monotonic() < deadline:
            self._check_running()
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                sock.connect(str(path))
                return sock
            except (FileNotFoundError, ConnectionRefusedError, OSError):
                sock.close()
                time.sleep(0.02)
        raise MicroVMRuntimeError("vm_control_socket_timed_out")

    def _read_until(self, which: str, marker: bytes, *, timeout: float, limit: int):
        sock = self._monitor if which == "monitor" else self._serial
        if sock is None:
            raise MicroVMRuntimeError("vm_control_socket_unavailable")
        attr = "_monitor_buffer" if which == "monitor" else "_serial_buffer"
        deadline = time.monotonic() + timeout
        buffer = getattr(self, attr)
        while True:
            position = buffer.find(marker)
            if position >= 0:
                consumed = buffer[:position + len(marker)]
                setattr(self, attr, buffer[position + len(marker):])
                return consumed
            if len(buffer) >= limit:
                raise MicroVMRuntimeError(f"{which}_output_limit_exceeded")
            self._check_running()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise MicroVMRuntimeError(f"{which}_timed_out")
            readable, _, _ = select.select([sock], [], [], min(remaining, 0.25))
            if readable:
                data = sock.recv(min(4096, limit - len(buffer)))
                if not data:
                    raise MicroVMRuntimeError(f"{which}_closed")
                buffer += data

    def _hmp(self, command: str, *, timeout=None):
        self._check_running()
        if not re.fullmatch(r"[A-Za-z0-9 _-]{1,80}", command):
            raise ValueError("Invalid host monitor command")
        try:
            self._monitor.sendall((command + "\n").encode("ascii"))
            output = self._read_until("monitor", b"(qemu) ",
                                      timeout=timeout or self.command_timeout, limit=65536)
        except MicroVMRuntimeError:
            self.close()
            raise
        except OSError as exc:
            self.close()
            raise MicroVMRuntimeError("qemu_monitor_io_failed") from exc
        text = output.decode("utf-8", "replace")
        if re.search(r"(?im)^\s*(error|failed|unknown command)\b", text):
            raise MicroVMRuntimeError("qemu_monitor_command_failed")
        return text

    def start(self, *, paused=False):
        """Boot a dedicated VM, or start paused before loading a saved VM state."""
        if self._process is not None:
            raise ValueError("One runtime instance can only start once")
        if type(paused) is not bool:
            raise ValueError("paused must be boolean")
        _reject_quarantined_disk(self.disk_path)
        self._check_artifacts()
        # macOS temp directories can exceed AF_UNIX's path limit. Linux has
        # no /private/tmp, so keep its equally private socket directory in
        # /tmp rather than an operator's potentially long TMPDIR.
        socket_root = "/private/tmp" if sys.platform == "darwin" else "/tmp"
        self._socket_dir = Path(tempfile.mkdtemp(prefix="fpb-vm-", dir=socket_root))
        monitor_path = self._socket_dir / "hmp.sock"
        serial_path = self._socket_dir / "serial.sock"
        action_path = self._socket_dir / "action.sock"
        socket_paths = ((monitor_path, serial_path, action_path)
                        if self.enable_action_port else (monitor_path, serial_path))
        if max(len(os.fsencode(path)) for path in socket_paths) >= 100:
            self.close()
            raise MicroVMRuntimeError("vm_unix_socket_path_too_long")
        _, machine, cpu, _, block_device, serial_device = _BACKENDS[self.backend]
        args = [self.qemu_binary, "-machine", machine, "-cpu", cpu,
                "-m", str(self.memory_mib), "-smp", str(self.vcpus),
                "-display", "none", "-nic", "none", "-no-reboot",
                "-kernel", str(self.kernel_path), "-initrd", str(self.initramfs_path),
                "-append", self.kernel_append,
                "-drive", f"if=none,id=work,file={self.disk_path},format=qcow2",
                "-device", f"{block_device},drive=work",
                "-monitor", f"unix:{monitor_path},server=on,wait=off",
                "-serial", f"unix:{serial_path},server=on,wait=off"]
        for index, path in enumerate(self.readonly_disk_paths):
            name = f"read{index}"
            args.extend(["-drive", f"if=none,id={name},file={path},format=raw,readonly=on",
                         "-device", f"{block_device},drive={name}"])
        if self.enable_action_port:
            # A dedicated virtio stream keeps RPC frames separate from the
            # kernel console and its asynchronous shell/kernel messages.
            args.extend(["-chardev",
                         f"socket,id=fpbctl,path={action_path},server=on,wait=off",
                         "-device", f"{serial_device},id=fpbvs",
                         "-device", "virtserialport,chardev=fpbctl,name=fpb.control"])
        if paused:
            args.append("-S")
        started = time.monotonic()
        try:
            self._process = self._popen_factory(
                args, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE, close_fds=True)
            self._stderr_thread = threading.Thread(
                target=self._stderr_reader, args=(self._process.stderr,), daemon=True)
            self._stderr_thread.start()
            deadline = time.monotonic() + min(15.0, self.command_timeout)
            self._monitor = self._connect(monitor_path, deadline)
            self._read_until("monitor", b"(qemu) ",
                             timeout=max(0.1, deadline - time.monotonic()), limit=65536)
            self._serial = self._connect(serial_path, deadline)
            if self.enable_action_port:
                self._action_port = self._connect(action_path, deadline)
        except Exception as exc:
            self.close()
            if isinstance(exc, MicroVMRuntimeError):
                raise
            raise MicroVMRuntimeError("vm_start_failed") from exc
        self.metrics["vm_starts"] += 1
        self.metrics["vm_start_seconds"] += time.monotonic() - started

    def action_port_socket(self):
        """Trusted host handle for the optional read-only guest RPC client."""
        self._check_running()
        if (not self.enable_action_port or self._action_port is None
                or self._action_port_retired):
            raise MicroVMRuntimeError("vm_action_port_not_enabled")
        return self._action_port

    def begin_action_port_retirement(self, client):
        """Irreversibly close the host FD after a trusted agent has stopped.

        The caller must separately prove guest-process exit and unmount the
        guest port before invoking this. The live host client identity and its
        acknowledged STOP are checked here. Checkpointing stays disabled
        until QEMU independently reports its port and backend disconnected.
        """
        self._check_running()
        if (not self.enable_action_port or self._action_port_retired
                or self._action_port is None
                or getattr(client, "socket", None) is not self._action_port
                or getattr(client, "failed", True)
                or getattr(client, "stopped", False) is not True):
            raise MicroVMRuntimeError("vm_action_port_retirement_invalid")
        self._action_port.close()
        self._action_port = None
        self._action_port_retired = True
        if self._socket_dir is None:
            raise MicroVMRuntimeError("vm_action_port_listener_path_missing")
        listener = self._socket_dir / "action.sock"
        try:
            try:
                mode = listener.lstat().st_mode
            except FileNotFoundError:
                mode = None  # An already unlinked pathname cannot accept a new peer.
            if mode is not None:
                if not stat.S_ISSOCK(mode):
                    raise MicroVMRuntimeError("vm_action_port_listener_not_socket")
                listener.unlink()
        except OSError as exc:
            raise MicroVMRuntimeError("vm_action_port_listener_unlink_failed") from exc
        self._require_action_port_listener_absent()
        self._action_port_listener_unlinked = True

    def _require_action_port_listener_absent(self):
        if self._socket_dir is None:
            raise MicroVMRuntimeError("vm_action_port_listener_path_missing")
        listener = self._socket_dir / "action.sock"
        try:
            listener.lstat()
        except FileNotFoundError:
            return
        except OSError as exc:
            raise MicroVMRuntimeError("vm_action_port_listener_check_failed") from exc
        raise MicroVMRuntimeError("vm_action_port_listener_reappeared")

    def _require_retired_action_port(self):
        if (not self.enable_action_port or not self._action_port_retired
                or not self._action_port_disconnect_attested
                or not self._action_port_listener_unlinked
                or self._action_port is not None or self._socket_dir is None):
            raise MicroVMRuntimeError("vm_action_port_snapshot_not_supported")
        self._require_action_port_listener_absent()

    def _guard_retired_action_port(self):
        """Revoke this runtime's attestation after any unexpected port state."""
        try:
            self._require_retired_action_port()
            self._require_disconnected_action_port()
        except MicroVMRuntimeError:
            self._action_port_disconnect_attested = False
            raise

    def action_port_qemu_state(self):
        """Read the exact port/backend state from QEMU's trusted monitor."""
        if not self.enable_action_port:
            raise MicroVMRuntimeError("vm_action_port_not_enabled")
        qtree = self._hmp("info qtree")
        lines = qtree.splitlines()
        matching = []
        for index, line in enumerate(lines):
            if "dev: virtserialport" not in line:
                continue
            block = "\n".join(lines[index:index + 6])
            if 'chardev = "fpbctl"' in block:
                matching.append(block)
        if len(matching) != 1 or not all(marker in matching[0] for marker in (
                'nr = 1 (0x1)', 'name = "fpb.control"')):
            raise MicroVMRuntimeError("vm_action_port_qemu_state_invalid")
        match = re.search(r"port 1, guest (on|off), host (on|off), throttle (on|off)",
                          matching[0])
        if match is None or match.group(3) != "off":
            raise MicroVMRuntimeError("vm_action_port_qemu_state_invalid")
        chardev = self._hmp("info chardev")
        entries = [line for line in chardev.splitlines()
                   if line.startswith("fpbctl: filename=")]
        if len(entries) != 1:
            raise MicroVMRuntimeError("vm_action_port_qemu_state_invalid")
        filename = entries[0].removeprefix("fpbctl: filename=")
        if filename.startswith("disconnected:unix:"):
            disconnected = True
        elif filename.startswith("unix:"):
            disconnected = False
        else:
            raise MicroVMRuntimeError("vm_action_port_qemu_state_invalid")
        return {"guest": match.group(1), "host": match.group(2),
                "chardev_disconnected": disconnected}

    def _require_disconnected_action_port(self):
        """Fail closed unless QEMU reports the exact unused action device."""
        state = self.action_port_qemu_state()
        if state != {"guest": "off", "host": "off", "chardev_disconnected": True}:
            raise MicroVMRuntimeError("vm_action_port_qemu_not_disconnected")

    def attest_action_port_retirement(self):
        """Second stage: enable snapshots only after exact QEMU disconnection."""
        if (not self.enable_action_port or not self._action_port_retired
                or self._action_port is not None
                or not self._action_port_listener_unlinked):
            raise MicroVMRuntimeError("vm_action_port_retirement_invalid")
        self._require_action_port_listener_absent()
        self._require_disconnected_action_port()
        self._action_port_disconnect_attested = True

    def retire_action_port(self, client):
        """One-step retirement for callers that need no guest reconciliation."""
        self.begin_action_port_retirement(client)
        self.attest_action_port_retirement()

    def wait_for_serial(self, marker, *, timeout=30.0):
        """Wait for a trusted guest readiness string on the serial console."""
        if isinstance(marker, str):
            marker = marker.encode("utf-8")
        if not isinstance(marker, bytes) or not marker or len(marker) > 256:
            raise ValueError("Invalid serial marker")
        try:
            return self._read_until("serial", marker, timeout=timeout, limit=262144)
        except MicroVMRuntimeError:
            self.close()
            raise

    def run_shell(self, command: str, *, timeout=None):
        """Run a fixed trusted-host command in the guest's serial shell.

        A random marker is split across two shell string arguments, so echoed
        command input cannot be mistaken for the output boundary. The guest
        shell must be ready on the serial console before this method is called.
        """
        self._check_running()
        if (not isinstance(command, str) or not command or len(command) > 8192
                or any(char in command for char in "\x00\r\n")):
            raise ValueError("Guest command must be one bounded host-owned shell line")
        duration = self.command_timeout if timeout is None else float(timeout)
        if not 0 < duration <= 300:
            raise ValueError("Guest command timeout must be in (0, 300]")
        nonce = secrets.token_hex(16)
        prefix = f"__FPB_{nonce}_"
        begin = (prefix + "BEGIN__").encode("ascii")
        end_prefix = (prefix + "END__:").encode("ascii")
        # The entire transaction must be one interactive-shell input line.
        # Sending a line for every operation makes BusyBox print a prompt
        # between BEGIN, the command, and END; that corrupts exact stdout
        # comparisons. Splitting the marker between printf arguments also
        # prevents the shell's initial echo from impersonating a boundary.
        command = command.rstrip(" ;")
        script = ("stty -echo; "
                  f"printf '\\n%s%s\\n' '{prefix}' 'BEGIN__'; "
                  f"{command}; "
                  "_fpb_guest_rc=$?; "
                  f"printf '\\n%s%s:%s\\n' '{prefix}' 'END__' \"$_fpb_guest_rc\"\n")
        # The guest recovery shell is an interactive canonical TTY. Linux's
        # input line cap is about 4096 bytes including our marker wrapper;
        # a larger command can stall until timeout instead of executing.
        if len(script.encode("utf-8")) > 4000:
            raise ValueError("Guest serial input line exceeds canonical TTY bound")
        started = time.monotonic()
        deadline = started + duration
        def remaining():
            return max(0.001, deadline - time.monotonic())
        try:
            self._serial.sendall(script.encode("utf-8"))
            self._read_until("serial", begin, timeout=remaining(), limit=262144)
            self._read_until("serial", b"\n", timeout=remaining(), limit=1024)
            output = self._read_until("serial", end_prefix, timeout=remaining(), limit=262144)
            # The end marker is followed by a decimal exit code and CRLF.
            tail = self._read_until("serial", b"\n", timeout=remaining(), limit=1024)
            code_text = tail[:-1].rstrip(b"\r")
            if not re.fullmatch(rb"[0-9]{1,3}", code_text):
                raise MicroVMRuntimeError("guest_exit_marker_invalid")
            return_code = int(code_text)
            if return_code > 255:
                raise MicroVMRuntimeError("guest_exit_code_invalid")
            stdout = output[:-len(end_prefix)].replace(b"\r\n", b"\n")
            if stdout.startswith(b"\n"):
                stdout = stdout[1:]
            if stdout.endswith(b"\n"):
                stdout = stdout[:-1]
            return {"stdout": stdout.decode("utf-8", "replace"),
                    "return_code": return_code}
        except MicroVMRuntimeError:
            self.close()
            raise
        except (OSError, socket.timeout) as exc:
            self.close()
            raise MicroVMRuntimeError("guest_serial_io_failed") from exc
        finally:
            self.metrics["guest_commands"] += 1
            self.metrics["guest_command_seconds"] += time.monotonic() - started

    @staticmethod
    def _snapshot_listed(tag, listing):
        return re.search(rf"(?m)^\s*\S+\s+{re.escape(tag)}(?:\s|$)", listing) is not None

    @staticmethod
    def _vm_status(listing):
        match = re.search(r"(?im)^\s*VM status:\s*(running|paused)\b", listing)
        if match is None:
            raise MicroVMRuntimeError("vm_snapshot_status_unrecognized")
        return match.group(1).lower()

    def _save_retired_action_port_snapshot(self, tag):
        """Pause before attesting and save only an exact disconnected topology."""
        self._guard_retired_action_port()
        started = time.monotonic()
        paused_here = False
        save_attempted = False
        try:
            status = self._vm_status(self._hmp("info status"))
            if status == "running":
                self._hmp("stop")
                paused_here = True
            if self._vm_status(self._hmp("info status")) != "paused":
                raise MicroVMRuntimeError("vm_snapshot_not_paused")
            self._guard_retired_action_port()
            if self._snapshot_listed(tag, self._hmp("info snapshots")):
                raise MicroVMRuntimeError("vm_snapshot_tag_already_exists")
            save_attempted = True
            self._hmp("savevm " + tag, timeout=max(30.0, self.command_timeout))
            if self._vm_status(self._hmp("info status")) != "paused":
                raise MicroVMRuntimeError("vm_snapshot_resumed_during_save")
            self._guard_retired_action_port()
            if not self._snapshot_listed(tag, self._hmp("info snapshots")):
                raise MicroVMRuntimeError("qemu_snapshot_not_published")
            if paused_here:
                self._hmp("cont")
                if self._vm_status(self._hmp("info status")) != "running":
                    raise MicroVMRuntimeError("vm_snapshot_resume_failed")
                paused_here = False
            self._snapshots.add(tag)
            self.metrics["snapshot_saves"] += 1
            self.metrics["snapshot_save_seconds"] += time.monotonic() - started
            return {"kind": "full_vm_state_qcow2_v1", "tag": tag,
                    "backend": self.backend,
                    "kernel_sha256": self.kernel_sha256,
                    "initramfs_sha256": self.initramfs_sha256,
                    "readonly_disk_sha256s": list(self._readonly_disk_sha256s),
                    "disk_path": str(self.disk_path), "memory_mib": self.memory_mib}
        except BaseException:
            # savevm can have written a tag even if its HMP response failed.
            # A tag that cannot be proved absent poisons this runtime.
            cleanup_verified = True
            if save_attempted:
                try:
                    if self._snapshot_listed(tag, self._hmp("info snapshots")):
                        self._hmp("delvm " + tag)
                    if self._snapshot_listed(tag, self._hmp("info snapshots")):
                        raise MicroVMRuntimeError("vm_snapshot_delete_unverified")
                except BaseException:
                    cleanup_verified = False
            self._snapshots.discard(tag)
            self._action_port_disconnect_attested = False
            if not cleanup_verified:
                quarantine_written = _quarantine_disk(self.disk_path)
                self.close()
                if not quarantine_written:
                    raise MicroVMRuntimeError("vm_snapshot_disk_quarantine_failed")
                raise MicroVMRuntimeError("vm_action_port_snapshot_cleanup_unverified")
            if paused_here and self._process is not None:
                try:
                    self._hmp("cont")
                except BaseException:
                    self.close()
                    raise MicroVMRuntimeError("vm_action_port_snapshot_resume_failed")
            raise

    def save_snapshot(self, tag="warm"):
        """Create a full VM + writable qcow2 snapshot with a host-only tag."""
        if self.enable_action_port and not self._action_port_disconnect_attested:
            raise MicroVMRuntimeError("vm_action_port_snapshot_not_supported")
        if not isinstance(tag, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", tag):
            raise ValueError("Invalid snapshot tag")
        if self.enable_action_port:
            return self._save_retired_action_port_snapshot(tag)
        started = time.monotonic()
        # HMP savevm *replaces* a pre-existing tag. Check QEMU itself, not
        # merely this runtime's local set: a cloned qcow2 may inherit tags.
        if self._snapshot_listed(tag, self._hmp("info snapshots")):
            raise MicroVMRuntimeError("vm_snapshot_tag_already_exists")
        try:
            self._hmp("savevm " + tag, timeout=max(30.0, self.command_timeout))
            if not self._snapshot_listed(tag, self._hmp("info snapshots")):
                raise MicroVMRuntimeError("qemu_snapshot_not_published")
        except BaseException:
            # The monitor response can fail after QEMU has written a tag.
            # Remove a visible uncommitted tag, then discard this runtime:
            # HMP does not prove that a failed write left qcow2 unchanged.
            cleanup_verified = self._process is not None
            if cleanup_verified:
                try:
                    if self._snapshot_listed(tag, self._hmp("info snapshots")):
                        self._hmp("delvm " + tag)
                    if self._snapshot_listed(tag, self._hmp("info snapshots")):
                        raise MicroVMRuntimeError("vm_snapshot_delete_unverified")
                except BaseException:
                    cleanup_verified = False
            self._snapshots.discard(tag)
            quarantine_written = _quarantine_disk(self.disk_path) if not cleanup_verified else True
            self.close()
            if not cleanup_verified:
                if not quarantine_written:
                    raise MicroVMRuntimeError("vm_snapshot_disk_quarantine_failed")
                raise MicroVMRuntimeError("vm_snapshot_cleanup_unverified")
            raise
        self._snapshots.add(tag)
        self.metrics["snapshot_saves"] += 1
        self.metrics["snapshot_save_seconds"] += time.monotonic() - started
        return {"kind": "full_vm_state_qcow2_v1", "tag": tag,
                "backend": self.backend,
                "kernel_sha256": self.kernel_sha256,
                "initramfs_sha256": self.initramfs_sha256,
                "readonly_disk_sha256s": list(self._readonly_disk_sha256s),
                "disk_path": str(self.disk_path), "memory_mib": self.memory_mib}

    def load_snapshot(self, tag="warm", *, resume=False, full_validation=False):
        """Restore CPU/RAM/device and qcow2 disk state from a saved VM state.

        The normal in-process path checks an already-hashed read-only disk's
        identity and trusts a tag saved by this runtime, avoiding a repeated
        large-file digest and monitor listing. ``loadvm`` remains authoritative
        and fails if the tag is absent. Tags inherited by a cloned child still
        need QEMU listing. ``full_validation`` restores the previous per-load
        digest and listing behavior for audits and A/B measurements.
        """
        if self.enable_action_port and not self._action_port_disconnect_attested:
            raise MicroVMRuntimeError("vm_action_port_snapshot_not_supported")
        if not isinstance(tag, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", tag):
            raise ValueError("Invalid snapshot tag")
        if type(full_validation) is not bool:
            raise ValueError("full_validation must be boolean")
        if self.enable_action_port:
            self._guard_retired_action_port()
        if len(self._readonly_disk_sha256s) != len(self.readonly_disk_paths):
            raise MicroVMRuntimeError("readonly_disk_binding_missing")
        for index, (path, expected) in enumerate(zip(
                self.readonly_disk_paths, self._readonly_disk_sha256s)):
            if len(self._readonly_disk_identities) == len(self.readonly_disk_paths):
                if _file_identity(path) != self._readonly_disk_identities[index]:
                    raise MicroVMRuntimeError("readonly_disk_changed_since_vm_start")
                if not full_validation:
                    continue
            if not path.is_file() or path.is_symlink() or _sha256_file(path) != expected:
                raise MicroVMRuntimeError("readonly_disk_changed_since_vm_start")
        if full_validation or tag not in self._snapshots:
            listing = self._hmp("info snapshots")
            if not re.search(rf"(?m)^\s*\S+\s+{re.escape(tag)}(?:\s|$)", listing):
                raise MicroVMRuntimeError("qemu_snapshot_not_found")
        started = time.monotonic()
        self._hmp("loadvm " + tag, timeout=max(30.0, self.command_timeout))
        if self.enable_action_port:
            try:
                self._guard_retired_action_port()
            except MicroVMRuntimeError:
                self.close()
                raise
        if resume:
            response = self._hmp("cont")
            if "already running" in response.lower():
                pass
        self._serial_buffer = b""
        self.metrics["snapshot_loads"] += 1
        self.metrics["snapshot_load_seconds"] += time.monotonic() - started

    def _verify_fork_artifacts(self):
        for path, expected, label in (
            (self.kernel_path, self.kernel_sha256, "kernel"),
            (self.initramfs_path, self.initramfs_sha256, "initramfs"),
        ):
            if not path.is_file() or path.is_symlink() or _sha256_file(path) != expected:
                raise MicroVMRuntimeError(f"fork_{label}_digest_mismatch")
        if len(self._readonly_disk_sha256s) != len(self.readonly_disk_paths):
            raise MicroVMRuntimeError("fork_readonly_disk_binding_missing")
        for path, expected in zip(self.readonly_disk_paths, self._readonly_disk_sha256s):
            if not path.is_file() or path.is_symlink() or _sha256_file(path) != expected:
                raise MicroVMRuntimeError("fork_readonly_disk_digest_mismatch")
        if not self.disk_path.is_file() or self.disk_path.is_symlink():
            raise MicroVMRuntimeError("fork_source_disk_missing_or_symlink")

    def fork_snapshot(self, tag: str, child_disk_paths, *, resume_parent=True,
                      parallel_children=False):
        """Copy a quiescent full-state qcow2 and restore independent child VMs.

        This is a host-only copy-and-restore fan-out, not a live hypervisor
        fork. No child disk may already exist. The caller owns and must close
        every returned child. Do not concurrently issue commands to ``self``
        while this operation runs. ``parallel_children`` overlaps the paused
        start and snapshot restore of up to four independent child VMs. The
        parent stays quiescent until every child has finished or failed.
        """
        if self.enable_action_port:
            raise MicroVMRuntimeError("vm_action_port_fork_not_supported")
        self._check_running()
        if not isinstance(tag, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", tag):
            raise ValueError("Invalid snapshot tag")
        if type(resume_parent) is not bool:
            raise ValueError("resume_parent must be boolean")
        if type(parallel_children) is not bool:
            raise ValueError("parallel_children must be boolean")
        if (not isinstance(child_disk_paths, (list, tuple))
                or not 1 <= len(child_disk_paths) <= 8):
            raise ValueError("Provide one to eight child disk paths")
        if parallel_children and len(child_disk_paths) > 4:
            raise ValueError("Parallel fork supports at most four child disk paths")
        paths = []
        for value in child_disk_paths:
            raw = Path(value)
            if ".." in raw.parts:
                raise ValueError("Fork disk path cannot contain parent traversal")
            absolute = Path(os.path.abspath(raw))
            # Resolve only after inspecting the original path: a broken
            # symlink would otherwise disappear behind its nonexistent
            # target, and later existence checks could miss it.
            if absolute.is_symlink() or absolute.parent.is_symlink():
                raise ValueError("Fork disk path cannot contain symlinks")
            paths.append(absolute.resolve())
        paths = tuple(paths)
        if len(set(paths)) != len(paths) or self.disk_path in paths:
            raise ValueError("Fork disk paths must be unique and separate from parent")
        if any(path in (self.kernel_path, self.initramfs_path, *self.readonly_disk_paths)
               for path in paths):
            raise ValueError("Fork disk path overlaps a pinned VM artifact")
        for path in paths:
            if path.exists() or path.is_symlink():
                raise FileExistsError(path)
            if not path.parent.is_dir() or path.parent.is_symlink():
                raise ValueError("Fork target parent directory is required")

        started = time.monotonic()
        children, created = [], []
        paused_here = False
        try:
            status = self._hmp("info status")
            match = re.search(r"(?im)^\s*VM status:\s*(running|paused)\b", status)
            if match is None:
                raise MicroVMRuntimeError("qemu_status_unrecognized")
            if match.group(1).lower() == "running":
                self._hmp("stop")
                paused_here = True
            status = self._hmp("info status")
            if not re.search(r"(?im)^\s*VM status:\s*paused\b", status):
                raise MicroVMRuntimeError("qemu_guest_not_quiescent")
            listing = self._hmp("info snapshots")
            if not re.search(rf"(?m)^\s*\S+\s+{re.escape(tag)}(?:\s|$)", listing):
                raise MicroVMRuntimeError("qemu_snapshot_not_found")
            self._verify_fork_artifacts()
            source_sha = _sha256_file(self.disk_path)
            clone_started = time.monotonic()
            clone_modes = []
            for path in paths:
                mode = _clone_or_copy_qcow2(self.disk_path, path)
                created.append(path)
                clone_modes.append(mode)
                if _sha256_file(path) != source_sha:
                    raise MicroVMRuntimeError("fork_disk_digest_mismatch")
            self.metrics["fork_disk_clone_seconds"] += time.monotonic() - clone_started
            if _sha256_file(self.disk_path) != source_sha:
                raise MicroVMRuntimeError("fork_source_disk_changed_while_paused")
            self._verify_fork_artifacts()
            child_stage_started = time.monotonic()
            for path in paths:
                child = MicroVMRuntime(
                    self.kernel_path, self.initramfs_path, path,
                    kernel_sha256=self.kernel_sha256,
                    initramfs_sha256=self.initramfs_sha256,
                    readonly_disk_paths=self.readonly_disk_paths,
                    memory_mib=self.memory_mib, vcpus=self.vcpus,
                    backend=self.backend,
                    kernel_append=self.kernel_append, qemu_binary=self.qemu_binary,
                    qemu_img_binary=self.qemu_img_binary,
                    command_timeout=self.command_timeout,
                    popen_factory=self._popen_factory)
                children.append(child)

            def start_and_restore(child):
                child.start(paused=True)
                child.load_snapshot(tag, resume=True)

            if parallel_children and len(children) > 1:
                # The context manager joins all workers before either the
                # success path or failure cleanup touches child disks.
                with ThreadPoolExecutor(max_workers=len(children)) as executor:
                    pending = [executor.submit(start_and_restore, child)
                               for child in children]
                    for future in pending:
                        future.result()
            else:
                for child in children:
                    start_and_restore(child)
            self.metrics["fork_child_start_restore_seconds"] += (
                time.monotonic() - child_stage_started)
            self._verify_fork_artifacts()
            if paused_here and resume_parent:
                self._hmp("cont")
                paused_here = False
            self.metrics["forks_spawned"] += len(children)
            self.metrics["fork_reflink_disks"] += clone_modes.count("clonefile")
            self.metrics["fork_copied_disks"] += clone_modes.count("copy")
            self.metrics["fork_seconds"] += time.monotonic() - started
            return children
        except BaseException:
            for child in children:
                child.close()
            for path in created:
                path.unlink(missing_ok=True)
            if paused_here and resume_parent:
                try:
                    self._hmp("cont")
                except Exception:
                    pass
            self.metrics["fork_seconds"] += time.monotonic() - started
            raise

    def get_state(self):
        """Trusted diagnostics, never a policy-visible observation."""
        state = {"running": self._process is not None and self._process.poll() is None,
                "backend": self.backend,
                "checkpoint_kind": (None if self.enable_action_port and not self._action_port_disconnect_attested
                                    else "full_vm_state_qcow2_v1"),
                "snapshots_saved_here": sorted(self._snapshots),
                "kernel_sha256": self.kernel_sha256,
                "initramfs_sha256": self.initramfs_sha256,
                "readonly_disk_sha256s": list(self._readonly_disk_sha256s),
                "disk_path": str(self.disk_path),
                "metrics": dict(self.metrics),
                "stderr_tail": self._stderr_tail.decode("utf-8", "replace")}
        if self.enable_action_port:
            state["checkpoint_operations_supported"] = self._action_port_disconnect_attested
            state["action_port_qemu_disconnected_attested"] = self._action_port_disconnect_attested
            state["action_transport"] = ("virtio_serial_readonly_retired_v1"
                                          if self._action_port_retired
                                          else "virtio_serial_readonly_v1")
        return state

    def close(self):
        """Stop the VM and remove only this runtime's private control sockets."""
        for sock in (self._monitor, self._serial, self._action_port):
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
        self._monitor = self._serial = self._action_port = None
        process = self._process
        if process is not None:
            try:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                else:
                    process.wait(timeout=1)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    process.kill()
                except OSError:
                    pass
            if process.stderr is not None:
                try:
                    process.stderr.close()
                except OSError:
                    pass
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=1)
        self._process = None
        self._stderr_thread = None
        self._action_port_disconnect_attested = False
        self._action_port_listener_unlinked = False
        self._action_port_retired = False
        if self._socket_dir is not None:
            shutil.rmtree(self._socket_dir, ignore_errors=True)
            self._socket_dir = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
