"""A real QEMU/HVF coding-task adapter with host-private outcome checking.

The policy receives only bounded file tools and one operator-fixed visible
Python check. Host code drives an Alpine recovery shell over the VM's serial
socket. Hidden expected outputs remain on the host. By default, after submit,
each case is run from the same full VM (CPU, memory, device and qcow2 disk)
snapshot. An exact task-specific opt-in contract can instead use a single
guest batch with per-case mount/PID namespaces after one full VM restore.

This is an experimental VM adapter, not a hardened multi-tenant sandbox. In
particular, guest programs run inside the isolated VM, and the operator must
review the guest image and task files before accepting arbitrary policies.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import re
import secrets
import shlex
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path

from .coding_env import _relative_file
from . import guest_action_rpc
from .http import strict_json_loads
from .microvm_runtime import MicroVMRuntimeError, _sha256_file
from .realworld import AdapterInfrastructureError, validate_task
from . import replace_text as _replace_text
from .stateless_verifier import (GUEST_PROGRAM, StatelessCaseVerifier,
                                 validate_stateless_contract)
from .virtio_action import (VirtioActionClient, VirtioActionError,
                            VirtioSerialReadFallback)


class MicroVMCodingError(AdapterInfrastructureError):
    """VM or guest-control failure; it must not become a zero reward."""


def _validated_task_sha256(task):
    """Accept source or registry-stamped task, rejecting stale digest fields."""
    if not isinstance(task, dict):
        raise ValueError("VM task must be an object")
    source = copy.deepcopy(task)
    stamped = ("task_sha256" in source or "reward_contract_sha256" in source)
    if stamped:
        if not {"task_sha256", "reward_contract_sha256"} <= source.keys():
            raise ValueError("VM task digest fields are incomplete")
        task_digest = source.pop("task_sha256")
        reward_digest = source.pop("reward_contract_sha256")
    frozen = validate_task(source)
    if stamped and (task_digest != frozen["task_sha256"]
                    or reward_digest != frozen["reward_contract_sha256"]):
        raise ValueError("VM task digest fields differ from task content")
    return frozen["task_sha256"]


def _stateless_task_source_digest(task):
    """Hash task content without the self-referential frozen binding field."""
    source = copy.deepcopy(task)
    metadata = source.get("metadata")
    if isinstance(metadata, dict):
        metadata.pop("artifact_binding", None)
        if not metadata:
            source.pop("metadata", None)
    serialized = json.dumps(source, sort_keys=True, separators=(",", ":"),
                            ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


_REPLACE_GUEST_PROGRAM = _replace_text.guest_program()
_REPLACE_BINDING = _replace_text.helper_binding()
_REPLACE_SOURCE_SHA256 = _REPLACE_BINDING["source_sha256"]


def replace_text_helper_binding():
    """Task-visible code identity for the optional exact-edit action."""
    return dict(_REPLACE_BINDING)


def _check_replace_text_helper(task):
    tools = {entry["name"] for entry in task["tool_manifest"]}
    if "replace_text" not in tools:
        return False
    if (task.get("metadata", {}).get("replace_text_helper_binding")
            != replace_text_helper_binding()
            or hashlib.sha256(Path(_replace_text.__file__).read_bytes()).hexdigest()
               != _REPLACE_SOURCE_SHA256):
        raise ValueError("Replace-text helper differs from frozen task binding")
    return True


class MicroVMCodingAdapter:
    """One trusted real-world task episode, backed by one ARM64 Linux VM."""

    BOOT_MARKER = "Launching initramfs emergency recovery shell"
    WORKSPACE_ROOT = "/mnt/root/workspace"

    def __init__(self, runtime, *, verifier_dir, visible_check,
                 workspace_root=WORKSPACE_ROOT, command_timeout=30.0,
                 stateless_verifier_contract=None, stateless_task_path=None,
                 read_transport="serial_shell", preinstalled_read_agent=False):
        if getattr(runtime, "backend", "aarch64_hvf") != "aarch64_hvf":
            raise ValueError("MicroVMCodingAdapter requires the aarch64_hvf backend")
        self.runtime = runtime
        if read_transport not in {"serial_shell", "virtio_serial_readonly_v1"}:
            raise ValueError("Unsupported VM read transport")
        self.read_transport = read_transport
        self.virtio_read_enabled = read_transport == "virtio_serial_readonly_v1"
        if type(preinstalled_read_agent) is not bool or (
                preinstalled_read_agent and not self.virtio_read_enabled):
            raise ValueError("Preinstalled read agent requires virtio read transport")
        self.preinstalled_read_agent = preinstalled_read_agent
        if runtime is not None and bool(getattr(runtime, "enable_action_port", False)) != self.virtio_read_enabled:
            raise ValueError("VM action port and adapter read transport must match")
        self._virtio_client = None
        self._virtio_agent_pid = None
        self._virtio_process_baseline = None
        self._virtio_agent_sha256 = (_sha256_file(Path(guest_action_rpc.__file__))
                                     if self.virtio_read_enabled else None)
        verifier_root = Path(verifier_dir)
        if verifier_root.is_symlink():
            raise ValueError("Trusted verifier directory cannot be a symlink")
        self.verifier_dir = verifier_root.resolve()
        if not self.verifier_dir.is_dir():
            raise ValueError("Trusted verifier directory is required")
        if not isinstance(workspace_root, str) or not re.fullmatch(r"/[A-Za-z0-9/_-]{1,200}", workspace_root):
            raise ValueError("Invalid fixed guest workspace root")
        self.workspace_root = workspace_root.rstrip("/")
        self.visible_check = self._validate_python_argv(visible_check)
        self.stateless_contract_path = (Path(stateless_verifier_contract)
                                        if stateless_verifier_contract is not None else None)
        if self.stateless_contract_path is None and stateless_task_path is not None:
            raise ValueError("Stateless task path requires a stateless verifier contract")
        self.stateless_task_path = (Path(stateless_task_path)
                                    if stateless_task_path is not None else None)
        self.stateless_verifier = None
        if self.stateless_contract_path is not None:
            if self.virtio_read_enabled:
                raise ValueError("Virtio read transport does not support stateless verifier mode")
            if (self.stateless_task_path is None or self.stateless_task_path.is_symlink()
                    or not self.stateless_task_path.is_file()
                    or self.stateless_task_path.name != "task.json"
                    or self.verifier_dir != (self.stateless_task_path.parent / "verifier").resolve()):
                raise ValueError("Stateless verification needs the exact task.json path")
            self.stateless_source_task = strict_json_loads(
                self.stateless_task_path.read_text(encoding="utf-8"))
            self.stateless_verifier = StatelessCaseVerifier(
                runtime, self.stateless_task_path.parent, self.stateless_contract_path)
            self.stateless_task_source_sha256 = _stateless_task_source_digest(
                self.stateless_source_task)
            self.stateless_contract_sha256 = _sha256_file(self.stateless_contract_path)
            self.stateless_helper_sha256 = _sha256_file(GUEST_PROGRAM)
        if isinstance(command_timeout, bool) or not 0 < command_timeout <= 300:
            raise ValueError("command_timeout must be in (0, 300]")
        self.command_timeout = float(command_timeout)
        # Validate the actual serial command size before any episode starts.
        # A nominally valid hidden case that cannot fit the guest TTY would
        # otherwise keep its reward pending only after the policy submits.
        self._python_command(self.visible_check, unprivileged=True)
        if self.stateless_verifier is None:
            self._preflight_full_vm_verifier()
        self.started = False
        self.replace_text_enabled = False
        self.submitted = False
        self.verified = None
        self.expected_binding = None
        self._task_sha256 = None
        self.snapshot = None
        self._branch_tags = set()
        self.block_devices = None
        self._baseline_processes = None
        self.metrics = {"guest_tool_seconds": 0.0, "file_reads": 0,
                        "virtio_reads": 0, "virtio_read_seconds": 0.0,
                        "virtio_read_fallbacks": 0,
                        "virtio_agent_upload_seconds": 0.0,
                        "virtio_agent_prepare_seconds": 0.0,
                        "virtio_agent_start_seconds": 0.0,
                        "virtio_agent_retire_seconds": 0.0,
                        "virtio_agent_disconnect_probe_seconds": 0.0,
                        "file_writes": 0, "visible_checks": 0,
                        "hidden_cases": 0, "full_vm_restores": 0,
                        "stateless_batches": 0, "stateless_batch_fallbacks": 0,
                        "stateless_submit_precheck_seconds": 0.0,
                        "stateless_submit_helper_install_seconds": 0.0,
                        "stateless_submit_batch_upload_seconds": 0.0,
                        "stateless_submit_postupload_check_seconds": 0.0,
                        "vm_submit_snapshot_seconds": 0.0,
                        "stateless_verify_restore_seconds": 0.0,
                        "stateless_verify_precheck_seconds": 0.0,
                        "stateless_verify_batch_seconds": 0.0,
                        "stateless_verify_postcheck_seconds": 0.0}

    @staticmethod
    def _validate_python_argv(argv):
        if (not isinstance(argv, (list, tuple)) or len(argv) != 4
                or tuple(argv[:3]) != ("python3", "-B", "-c")
                or not isinstance(argv[3], str) or not argv[3]
                or len(argv[3]) > 2048 or "\x00" in argv[3]):
            raise ValueError("Only a fixed python3 -B -c check is supported")
        return tuple(argv)

    def _preflight_full_vm_verifier(self):
        path = self.verifier_dir / "verify.json"
        if path.is_symlink() or not path.is_file():
            raise ValueError("Host verifier must be a regular file")
        specification = strict_json_loads(path.read_text(encoding="utf-8"))
        if (not isinstance(specification, dict)
                or set(specification) != {"kind", "cases"}
                or specification["kind"] != "command_cases_v1"
                or not isinstance(specification["cases"], list)
                or not 1 <= len(specification["cases"]) <= 32):
            raise ValueError("Unsupported full-VM verifier specification")
        for case in specification["cases"]:
            if (not isinstance(case, dict)
                    or set(case) != {"argv", "expected_stdout", "expected_returncode"}
                    or not isinstance(case["expected_stdout"], str)
                    or len(case["expected_stdout"].encode("utf-8")) > 12000
                    or type(case["expected_returncode"]) is not int
                    or not 0 <= case["expected_returncode"] <= 123):
                raise ValueError("Unsupported full-VM hidden case")
            self._python_command(case["argv"], unprivileged=True)

    def _guest(self, command, *, timeout=None):
        started = time.monotonic()
        try:
            return self.runtime.run_shell(command, timeout=timeout or self.command_timeout)
        except MicroVMRuntimeError as exc:
            raise MicroVMCodingError(str(exc)) from exc
        finally:
            self.metrics["guest_tool_seconds"] += time.monotonic() - started

    def _timed(self, metric, operation, *args):
        started = time.monotonic()
        try:
            return operation(*args)
        finally:
            self.metrics[metric] += time.monotonic() - started

    def _guest_ok(self, command, *, timeout=None):
        result = self._guest(command, timeout=timeout)
        if result["return_code"] != 0:
            raise MicroVMCodingError("guest_setup_or_tool_command_failed")
        return result["stdout"]

    def _guest_file(self, relative):
        path = _relative_file(relative)
        return self.workspace_root + "/" + str(path)

    @staticmethod
    def _replace_text_command(action):
        _replace_text.validate_action(action)
        payload = base64.b64encode(json.dumps(
            action, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")).decode("ascii")
        code = ("exec(__import__('zlib').decompress(__import__('base64').b64decode('"
                + _REPLACE_GUEST_PROGRAM + "')))")
        command = ("LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/aarch64-linux-gnu "
                   "chroot /mnt/root /usr/local/bin/python3.12 -I -S -B -c "
                   + shlex.quote(code) + " " + payload)
        if len(command.encode("utf-8")) > 3800:
            raise ValueError("Replace text exceeds bounded guest command length")
        return command

    @staticmethod
    def _guest_path_assignment(path):
        """Load a bounded path without shell metacharacters or TTY newlines."""
        encoded = base64.b64encode(path.encode("utf-8")).decode("ascii")
        # The terminal newline is stripped by command substitution; append a
        # sentinel first so filenames ending in newlines remain exact.
        return ("fpb_target=$(printf '%s' '" + encoded
                + "' | base64 -d; printf x); fpb_target=${fpb_target%x}; ")

    def _detect_block_devices(self):
        # SquashFS starts with ASCII "hsqs" (68 73 71 73); the ext4 magic
        # is little-endian EF53 at offset 1080 (53 ef). Do not infer /dev/vdX
        # numbers from QEMU argument order: HVF enumeration may differ.
        probe = (
            "for d in /dev/vd?; do "
            "a=$(dd if=\"$d\" bs=4 count=1 2>/dev/null | od -An -tx1 | tr -d '[:space:]'); "
            "b=$(dd if=\"$d\" bs=1 skip=1080 count=2 2>/dev/null | od -An -tx1 | tr -d '[:space:]'); "
            "printf '%s:%s:%s\\n' \"$d\" \"$a\" \"$b\"; done")
        output = self._guest_ok(probe)
        squashfs, ext4 = [], []
        for line in output.splitlines():
            match = re.fullmatch(r"(/dev/vd[a-z]):([0-9a-f]*):([0-9a-f]*)", line.strip())
            if not match:
                raise MicroVMCodingError("guest_block_probe_invalid")
            device, start_magic, ext4_magic = match.groups()
            if start_magic == "68737173":
                squashfs.append(device)
            if ext4_magic == "53ef":
                ext4.append(device)
        if len(squashfs) != 1 or len(ext4) != 1 or squashfs[0] == ext4[0]:
            raise MicroVMCodingError("guest_block_device_types_ambiguous")
        return {"squashfs": squashfs[0], "ext4": ext4[0]}

    def _setup_guest(self):
        disks = self._detect_block_devices()
        self._guest_ok("mkdir -p /media/modloop /mnt/root /lib/modules")
        self._guest_ok(f"mount -t squashfs -o ro {disks['squashfs']} /media/modloop")
        self._guest_ok("mount --bind /media/modloop/modules /lib/modules")
        self._guest_ok("modprobe ext4")
        self._guest_ok(f"mount -t ext4 {disks['ext4']} /mnt/root")
        # The image intentionally contains no device tree. Python's
        # subprocess module opens /dev/null for discarded stderr, so create
        # only that guest-local character device inside the chroot.
        self._guest_ok("mkdir -p /mnt/root/dev && mknod -m 666 /mnt/root/dev/null c 1 3")
        self._guest_ok(f"test -d {shlex.quote(self.workspace_root)}")
        self.block_devices = disks

    def _start_virtio_read_agent(self):
        """Start the bounded read-only agent in the trusted coding guest."""
        if not self.virtio_read_enabled:
            return
        prepare_started = time.monotonic()
        source_path = Path(guest_action_rpc.__file__)
        if (source_path.is_symlink() or not source_path.is_file()
                or _sha256_file(source_path) != self._virtio_agent_sha256):
            raise MicroVMCodingError("guest_action_agent_changed")
        source = source_path.read_bytes()
        if len(source) > 100_000:
            raise MicroVMCodingError("guest_action_agent_oversize")
        target = "/mnt/root/fpb_guest_action_rpc.py"
        self._guest_ok("test -c /dev/virtio-ports/fpb.control")
        self._guest_ok("chmod 600 /dev/virtio-ports/fpb.control")
        self._guest_ok("touch /mnt/root/dev/fpb.control && "
                       "mount --bind /dev/virtio-ports/fpb.control /mnt/root/dev/fpb.control")
        if not self.preinstalled_read_agent:
            upload_started = time.monotonic()
            self._guest_ok(f": > {target}")
            for offset in range(0, len(source), 1500):
                payload = base64.b64encode(source[offset:offset + 1500]).decode("ascii")
                self._guest_ok(f"printf '%s' '{payload}' | base64 -d >> {target}")
            self.metrics["virtio_agent_upload_seconds"] += time.monotonic() - upload_started
        if self._guest_ok(f"sha256sum {target}").split()[0] != self._virtio_agent_sha256:
            raise MicroVMCodingError("guest_action_agent_image_or_upload_mismatch")
        self.metrics["virtio_agent_prepare_seconds"] += time.monotonic() - prepare_started
        start_started = time.monotonic()
        command = ("LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/aarch64-linux-gnu "
                   "chroot /mnt/root /usr/local/bin/python3.12 -I -B "
                   "/fpb_guest_action_rpc.py >/dev/null 2>&1 & "
                   "printf 'FPB_AGENT_PID=%s\\n' \"$!\"")
        match = re.fullmatch(r"FPB_AGENT_PID=([0-9]+)\n?", self._guest_ok(command))
        if match is None:
            raise MicroVMCodingError("guest_action_agent_pid_invalid")
        self._virtio_agent_pid = int(match.group(1))
        try:
            self._virtio_client = VirtioActionClient(self.runtime,
                                                    timeout=min(10.0, self.command_timeout))
            self._virtio_client.ping()
        except (MicroVMRuntimeError, VirtioActionError) as exc:
            raise MicroVMCodingError("guest_action_agent_connect_failed") from exc
        self.metrics["virtio_agent_start_seconds"] += time.monotonic() - start_started

    def _retire_virtio_read_agent(self):
        """Stop the agent and permanently remove this transport before savevm."""
        if not self.virtio_read_enabled:
            return
        client = self._virtio_client
        if client is None or self._virtio_agent_pid is None:
            raise MicroVMCodingError("guest_action_agent_missing")
        started = time.monotonic()
        try:
            client.stop()
            deadline = time.monotonic() + min(5.0, self.command_timeout)
            while True:
                status = self._guest_ok(
                    f"if kill -0 {self._virtio_agent_pid} 2>/dev/null; "
                    "then printf ALIVE; else printf EXITED; fi")
                if status == "EXITED":
                    break
                if status != "ALIVE" or time.monotonic() >= deadline:
                    raise MicroVMCodingError("guest_action_agent_not_quiescent")
                time.sleep(0.02)
            self._guest_ok("umount /mnt/root/dev/fpb.control && "
                           "rm /mnt/root/dev/fpb.control")
            if (self._virtio_process_baseline is None
                    or self._scan_guest_processes() != self._virtio_process_baseline):
                raise MicroVMCodingError("guest_action_process_state_changed")
            if self._scan_guest_action_port_holders():
                raise MicroVMCodingError("guest_action_port_fd_still_open")
            self.runtime.begin_action_port_retirement(client)
            state = self.runtime.action_port_qemu_state()
            if state == {"guest": "off", "host": "on",
                         "chardev_disconnected": False}:
                self._trusted_guest_reopen_for_qemu_disconnect()
                if (self._scan_guest_processes() != self._virtio_process_baseline
                        or self._scan_guest_action_port_holders()):
                    raise MicroVMCodingError("guest_action_reopen_not_quiescent")
            elif state != {"guest": "off", "host": "off",
                           "chardev_disconnected": True}:
                raise MicroVMCodingError("guest_action_qemu_retirement_state_invalid")
            self.runtime.attest_action_port_retirement()
        except (VirtioActionError, MicroVMRuntimeError) as exc:
            raise MicroVMCodingError("guest_action_agent_retirement_failed") from exc
        finally:
            self.metrics["virtio_agent_retire_seconds"] += time.monotonic() - started

    def _trusted_guest_reopen_for_qemu_disconnect(self):
        """Give QEMU a bounded read-ready interval to observe host EOF.

        The already-stopped agent and old host socket stay retired. This
        root-owned helper neither reads nor writes action-port bytes; it only
        opens O_NONBLOCK|O_NOFOLLOW briefly, closes, and removes its private
        bind. Checkpointing remains disabled until subsequent guest and QEMU
        quiescence checks pass.
        """
        started = time.monotonic()
        target = "/mnt/root/dev/fpb.retire-probe"
        source = (
            "import os,time\n"
            "fd=os.open('/dev/fpb.retire-probe',"
            "os.O_RDWR|os.O_NONBLOCK|os.O_CLOEXEC|os.O_NOFOLLOW)\n"
            "try:\n"
            " print('FPB_REOPEN_OK')\n"
            # Yield once; QEMU state attestation below, not a fixed sleep,
            # decides whether the port has actually disconnected.
            " time.sleep(0)\n"
            "finally: os.close(fd)\n")
        encoded = base64.b64encode(source.encode("utf-8")).decode("ascii")
        expression = f"exec(__import__('base64').b64decode('{encoded}'))"
        command = ("LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/aarch64-linux-gnu "
                   "chroot /mnt/root /usr/local/bin/python3.12 -I -S -B -c "
                   + shlex.quote(expression))
        try:
            self._guest_ok(f"test ! -e {target} && touch {target} && "
                           f"mount --bind /dev/virtio-ports/fpb.control {target}")
            try:
                if self._guest_ok(command, timeout=min(5.0, self.command_timeout)).strip() != "FPB_REOPEN_OK":
                    raise MicroVMCodingError("guest_action_reopen_output_invalid")
            finally:
                self._guest_ok(f"umount {target} && rm {target}")
        finally:
            self.metrics["virtio_agent_disconnect_probe_seconds"] += time.monotonic() - started

    def _check_no_symlinks(self):
        root = shlex.quote(self.workspace_root)
        output = self._guest_ok(f"find {root} -type l -print")
        if output.strip():
            raise MicroVMCodingError("guest_workspace_symlinks_unsupported")

    def _scan_guest_processes(self):
        """Snapshot live user-space PID/start pairs without exposing proc to policy.

        The proc mount exists only around this trusted host probe. In
        particular, visible policy checks run with /proc unmounted in their
        chroot. Excluding the probe's own PID avoids treating it as a
        persistent background process.
        """
        source = (
            "import base64,json,os\n"
            "items=[]\n"
            "for name in os.listdir('/proc'):\n"
            " if not name.isdecimal(): continue\n"
            " pid=int(name)\n"
            " if pid==os.getpid(): continue\n"
            " try:\n"
            "  stat=open('/proc/'+name+'/stat').read().rsplit(') ',1)[1].split()\n"
            "  start=int(stat[19])\n"
            "  exe=os.readlink('/proc/'+name+'/exe')\n"
            " except (OSError,ValueError,IndexError): continue\n"
            " if stat[0]!='Z' and exe: items.append([pid,start,exe])\n"
            "data=json.dumps(sorted(items),separators=(',',':')).encode()\n"
            "print('FPB_PROCESS_SET='+base64.b64encode(data).decode())")
        encoded = base64.b64encode(source.encode("utf-8")).decode("ascii")
        expression = f"exec(__import__('base64').b64decode('{encoded}'))"
        command = ("LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/aarch64-linux-gnu "
                   "chroot /mnt/root /usr/local/bin/python3.12 -I -S -B -c "
                   + shlex.quote(expression))
        self._guest_ok("mkdir -p /mnt/root/proc && mount -t proc proc /mnt/root/proc")
        try:
            output = self._guest_ok(command)
        finally:
            self._guest_ok("umount /mnt/root/proc")
        match = re.fullmatch(r"FPB_PROCESS_SET=([A-Za-z0-9+/=]+)\n?", output)
        if match is None:
            raise MicroVMCodingError("guest_process_probe_invalid")
        try:
            processes = strict_json_loads(base64.b64decode(match.group(1), validate=True).decode())
        except (ValueError, UnicodeError):
            raise MicroVMCodingError("guest_process_probe_invalid") from None
        if (not isinstance(processes, list) or len(processes) > 1024
                or any(not isinstance(item, list) or len(item) != 3
                       or type(item[0]) is not int or item[0] <= 0
                       or type(item[1]) is not int or item[1] < 0
                       or not isinstance(item[2], str) or not item[2]
                       for item in processes)):
            raise MicroVMCodingError("guest_process_probe_invalid")
        identities = {(item[0], item[1]): item[2] for item in processes}
        if len(identities) != len(processes):
            raise MicroVMCodingError("guest_process_probe_invalid")
        return identities

    def _scan_guest_action_port_holders(self):
        """Fail closed if any guest process still owns the retired port fd."""
        source = (
            "import base64,json,os\n"
            "holders=[]\n"
            "for name in os.listdir('/proc'):\n"
            " if not name.isdecimal(): continue\n"
            " try: fds=os.listdir('/proc/'+name+'/fd')\n"
            " except OSError: continue\n"
            " for fd in fds:\n"
            "  try: target=os.readlink('/proc/'+name+'/fd/'+fd)\n"
            "  except OSError: continue\n"
            "  if 'fpb.control' in target: holders.append([int(name),int(fd)])\n"
            "data=json.dumps(sorted(holders),separators=(',',':')).encode()\n"
            "print('FPB_PORT_HOLDERS='+base64.b64encode(data).decode())")
        encoded = base64.b64encode(source.encode("utf-8")).decode("ascii")
        expression = f"exec(__import__('base64').b64decode('{encoded}'))"
        command = ("LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/aarch64-linux-gnu "
                   "chroot /mnt/root /usr/local/bin/python3.12 -I -S -B -c "
                   + shlex.quote(expression))
        self._guest_ok("mkdir -p /mnt/root/proc && mount -t proc proc /mnt/root/proc")
        try:
            output = self._guest_ok(command)
        finally:
            self._guest_ok("umount /mnt/root/proc")
        match = re.fullmatch(r"FPB_PORT_HOLDERS=([A-Za-z0-9+/=]+)\n?", output)
        if match is None:
            raise MicroVMCodingError("guest_action_fd_probe_invalid")
        try:
            holders = strict_json_loads(base64.b64decode(
                match.group(1), validate=True).decode("utf-8"))
        except (ValueError, UnicodeError) as exc:
            raise MicroVMCodingError("guest_action_fd_probe_invalid") from exc
        if (not isinstance(holders, list) or len(holders) > 1024
                or any(not isinstance(item, list) or len(item) != 2
                       or any(type(number) is not int or number < 0 for number in item)
                       for item in holders)):
            raise MicroVMCodingError("guest_action_fd_probe_invalid")
        return holders

    def _ensure_stateless_quiescent(self):
        if self._baseline_processes is None:
            raise MicroVMCodingError("guest_process_baseline_missing")
        if self._scan_guest_processes() != self._baseline_processes:
            raise MicroVMCodingError("guest_not_quiescent_for_stateless_verifier")

    def artifact_binding(self):
        """Bind public VM assets and host-only verifier before task registration."""
        if getattr(self.runtime, "backend", "aarch64_hvf") != "aarch64_hvf":
            raise ValueError("MicroVMCodingAdapter requires the aarch64_hvf backend")
        verifier_file = self.verifier_dir / "verify.json"
        if not verifier_file.is_file() or verifier_file.is_symlink():
            raise ValueError("Verifier specification is missing")
        if self.stateless_verifier is None:
            # The host may have replaced the file after adapter construction.
            # Never freeze a verifier that cannot fit the serial transport.
            self._preflight_full_vm_verifier()
        binding = {"runtime_kind": "qemu_hvf_full_vm_qcow2_v1",
                "candidate_python_identity": "guest_uid_gid_65534_v2",
                "kernel_sha256": self.runtime.kernel_sha256,
                "initramfs_sha256": self.runtime.initramfs_sha256,
                "readonly_disk_sha256s": [_sha256_file(p) for p in self.runtime.readonly_disk_paths],
                "disk_seed_sha256": _sha256_file(self.runtime.disk_path),
                "verifier_sha256": _sha256_file(verifier_file),
                "visible_check_sha256": hashlib.sha256(json.dumps(
                    self.visible_check, separators=(",", ":")).encode("utf-8")).hexdigest()}
        if self.stateless_verifier is not None:
            self._check_stateless_source_binding()
            binding["stateless_verifier_kind"] = "stateless_python_cases_overlay_v1"
            binding["stateless_task_source_sha256"] = self.stateless_task_source_sha256
            binding["stateless_contract_sha256"] = self.stateless_contract_sha256
            binding["stateless_helper_sha256"] = self.stateless_helper_sha256
        if self.virtio_read_enabled:
            if _sha256_file(Path(guest_action_rpc.__file__)) != self._virtio_agent_sha256:
                raise ValueError("Guest action agent differs from frozen source")
            binding["policy_read_transport"] = self.read_transport
            binding["guest_action_agent_sha256"] = self._virtio_agent_sha256
            binding["guest_action_agent_install"] = (
                "preinstalled_image_v1" if self.preinstalled_read_agent
                else "serial_upload_v1")
        return binding

    def _check_stateless_source_binding(self):
        if self.stateless_verifier is None:
            return
        if (self.stateless_contract_path.is_symlink()
                or not self.stateless_contract_path.is_file()
                or self.stateless_task_path.is_symlink()
                or _sha256_file(self.stateless_contract_path) != self.stateless_contract_sha256
                or _sha256_file(GUEST_PROGRAM) != self.stateless_helper_sha256):
            raise ValueError("Stateless verifier contract or helper differs from frozen source")
        if _stateless_task_source_digest(strict_json_loads(
                self.stateless_task_path.read_text(encoding="utf-8"))) != self.stateless_task_source_sha256:
            raise ValueError("Stateless task source differs from frozen source")
        contract, task, _ = validate_stateless_contract(
            self.stateless_task_path.parent, self.stateless_contract_path)
        if (contract != self.stateless_verifier.contract
                or task["task_id"] != self.stateless_verifier.task["task_id"]):
            raise ValueError("Stateless verifier task contract differs")

    def _check_stateless_episode_task(self, task):
        if self.stateless_verifier is None:
            return
        candidate = copy.deepcopy(task)
        source = copy.deepcopy(self.stateless_source_task)
        candidate.pop("task_sha256", None)
        candidate.pop("reward_contract_sha256", None)
        for record in (candidate, source):
            metadata = record.get("metadata")
            if isinstance(metadata, dict):
                metadata.pop("artifact_binding", None)
                if not metadata:
                    record.pop("metadata", None)
        # The fixture CLI rewrites only this adapter tag for the VM runtime.
        if candidate.get("is_fixture"):
            for key in ("adapter_id", "adapter_version"):
                if key in self.stateless_source_task:
                    candidate[key] = self.stateless_source_task[key]
        if candidate != source:
            raise ValueError("Stateless verifier task source differs from episode task")

    def create_branch_checkpoint(self):
        """Freeze a full VM at a trusted-host action boundary for fan-out.

        The snapshot tag identifies the logical frozen state. QEMU can change
        qcow2 container bytes during later stop/cont even with no guest write,
        so the recorded disk digest is audit metadata, not a live-parent
        equality predicate. The runtime validates and restores that tag in
        every child.
        """
        if not self.started or self.submitted or self.expected_binding is None:
            raise ValueError("VM checkpoint requires an active unsubmitted episode")
        if self.virtio_read_enabled:
            raise ValueError("Virtio read transport does not support branch checkpoints")
        self._check_no_symlinks()
        if self.stateless_verifier is not None:
            # A namespaced verifier cannot inherit a writer from a branch
            # point. Require the exact clean-boot process identities before
            # publishing a snapshot that may fan out to many children.
            if self._scan_guest_processes() != self._baseline_processes:
                raise MicroVMCodingError("stateless_branch_not_quiescent")
        tag = "br" + secrets.token_hex(8)
        try:
            self.runtime.save_snapshot(tag)
            disk_sha256 = _sha256_file(self.runtime.disk_path)
        except (MicroVMRuntimeError, OSError) as exc:
            raise MicroVMCodingError("vm_branch_snapshot_failed") from exc
        self._branch_tags.add(tag)
        return {"checkpoint_kind": "full_vm_state_qcow2_v1",
                "snapshot_tag": tag,
                "snapshot_disk_sha256": disk_sha256,
                "artifact_binding": copy.deepcopy(self.expected_binding)}

    def branch_adapter(self, child_disk_path):
        """Build a host-only child adapter for RealWorldEnv.fork_from_checkpoint."""
        return MicroVMBranchCodingAdapter(self, child_disk_path)

    def reset(self, task, *, now):
        if self.started:
            raise ValueError("One VM adapter instance serves one episode")
        task_sha256 = _validated_task_sha256(task)
        self.replace_text_enabled = _check_replace_text_helper(task)
        if self.replace_text_enabled and self.workspace_root != self.WORKSPACE_ROOT:
            raise ValueError("Replace text requires the fixed guest workspace root")
        if (self.stateless_verifier is not None
                and task["task_id"] != self.stateless_verifier.task["task_id"]):
            raise ValueError("Stateless verifier is bound to a different task")
        self._check_stateless_episode_task(task)
        actual = self.artifact_binding()
        expected = task.get("metadata", {}).get("artifact_binding")
        if expected is None and not task["is_fixture"]:
            raise ValueError("Non-fixture VM tasks require frozen artifact binding")
        if expected is not None and expected != actual:
            raise ValueError("VM artifacts differ from the frozen task binding")
        self.expected_binding = actual
        self._task_sha256 = task_sha256
        try:
            self.runtime.start()
            self.runtime.wait_for_serial(self.BOOT_MARKER, timeout=45)
            self._setup_guest()
            self._check_no_symlinks()
            if self.virtio_read_enabled:
                self._virtio_process_baseline = self._scan_guest_processes()
            self._start_virtio_read_agent()
            if self.stateless_verifier is not None:
                self._baseline_processes = self._scan_guest_processes()
                if (len(self._baseline_processes) != 2
                        or not any(pid == 1 for pid, _ in self._baseline_processes)
                        or set(self._baseline_processes.values()) != {"/usr/bin/busybox"}):
                    raise MicroVMCodingError("guest_unexpected_baseline_processes")
        except (MicroVMRuntimeError, MicroVMCodingError) as exc:
            self.runtime.close()
            raise MicroVMCodingError("vm_episode_setup_failed") from exc
        self.started = True
        return {"task_id": task["task_id"],
                "tools": [tool["name"] for tool in task["tool_manifest"]],
                "runtime_kind": "qemu_hvf_full_vm_qcow2_v1",
                "workspace_root": "/workspace",
                "visible_check": list(self.visible_check)}

    def _python_command(self, argv, *, unprivileged=False):
        """Capture case stdout within Linux, away from shell prompt output.

        The candidate receives only its fixed check code, never the host's
        expected result. The wrapper caps captured output and kills the whole
        candidate process group on timeout or output overflow.
        """
        argv = self._validate_python_argv(argv)
        encoded_code = base64.b64encode(argv[3].encode("utf-8")).decode("ascii")
        case_timeout = min(25.0, max(1.0, self.command_timeout - 3.0))
        identity = ",user=65534,group=65534,extra_groups=[]" if unprivileged else ""
        wrapper = textwrap.dedent(f"""\
            import base64,json,os,select,signal,subprocess,time
            code=base64.b64decode({encoded_code!r}).decode('utf-8')
            env=os.environ.copy()
            env['PYTHONPATH']='/workspace'
            p=subprocess.Popen(['/usr/local/bin/python3.12','-B','-c',code],cwd='/workspace',env=env,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,start_new_session=True{identity})
            out=bytearray()
            deadline=time.monotonic()+{case_timeout!r}
            rc=None
            while True:
                remaining=deadline-time.monotonic()
                if remaining<=0:
                    rc=124
                    break
                if select.select([p.stdout],[],[],min(remaining,0.25))[0]:
                    chunk=os.read(p.stdout.fileno(),4096)
                    if not chunk:
                        # A candidate can close stdout while continuing to run.
                        # EOF is not process exit; keep the same overall case
                        # deadline and score a lingering child as a timeout.
                        try:
                            rc=p.wait(timeout=max(0.0,deadline-time.monotonic()))
                        except subprocess.TimeoutExpired:
                            rc=124
                        break
                    out.extend(chunk)
                    if len(out)>12000:
                        rc=125
                        break
            if rc in (124,125):
                try:
                    os.killpg(p.pid,signal.SIGKILL)
                except ProcessLookupError:
                    pass
                p.wait(timeout=2)
            payload={{'return_code':rc,'stdout_b64':base64.b64encode(out[:12000]).decode('ascii'),'truncated':len(out)>12000}}
            print('FPB_CASE_RESULT='+base64.b64encode(json.dumps(payload,separators=(',',':')).encode()).decode())
        """)
        # repr keeps the trusted multiline wrapper on one serial shell line.
        argument = shlex.quote("exec(" + repr(wrapper) + ")")
        command = (f"cd {shlex.quote(self.workspace_root)} && "
                   "LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/aarch64-linux-gnu "
                   f"chroot /mnt/root /usr/local/bin/python3.12 -I -c {argument}")
        if len(command.encode("utf-8")) > 3800:
            raise ValueError("Python case exceeds bounded guest command length")
        return command

    def _run_python_case(self, argv, *, timeout=None, unprivileged=False):
        result = self._guest(self._python_command(argv, unprivileged=unprivileged),
                             timeout=timeout)
        if result["return_code"] != 0:
            raise MicroVMCodingError("guest_case_capture_failed")
        match = re.fullmatch(r"FPB_CASE_RESULT=([A-Za-z0-9+/=]+)\n?", result["stdout"])
        if match is None:
            raise MicroVMCodingError("guest_case_result_invalid")
        try:
            decoded = base64.b64decode(match.group(1), validate=True)
            payload = json.loads(decoded)
        except (ValueError, UnicodeError) as exc:
            raise MicroVMCodingError("guest_case_result_invalid") from exc
        if (not isinstance(payload, dict)
                or set(payload) != {"return_code", "stdout_b64", "truncated"}
                or type(payload["return_code"]) is not int
                or not -255 <= payload["return_code"] <= 255
                or not isinstance(payload["stdout_b64"], str)
                or len(payload["stdout_b64"]) > 16000
                or type(payload["truncated"]) is not bool):
            raise MicroVMCodingError("guest_case_result_invalid")
        try:
            stdout_bytes = base64.b64decode(payload["stdout_b64"], validate=True)
        except ValueError as exc:
            raise MicroVMCodingError("guest_case_result_invalid") from exc
        if len(stdout_bytes) > 12000:
            raise MicroVMCodingError("guest_case_result_invalid")
        return {"return_code": payload["return_code"],
                "stdout_bytes": stdout_bytes,
                "stdout": stdout_bytes.decode("utf-8", "replace"),
                "truncated": payload["truncated"]}

    def step(self, action, *, now):
        if not self.started or self.submitted:
            raise ValueError("VM task is not active")
        if not isinstance(action, dict) or action.get("action") not in {
            "list_files", "read_file", "write_file", "replace_text",
            "run_visible_checks", "submit"
        }:
            raise ValueError("Unsupported VM coding action")
        kind = action["action"]
        expected = {"list_files": {"action"}, "read_file": {"action", "path"},
                    "write_file": {"action", "path", "content"},
                    "replace_text": {"action", "path", "expected_file_sha256",
                                     "old_text", "new_text"},
                    "run_visible_checks": {"action"}, "submit": {"action"}}[kind]
        if set(action) != expected:
            raise ValueError("Invalid VM coding action arguments")
        if kind == "list_files":
            self._check_no_symlinks()
            root = shlex.quote(self.workspace_root)
            output = self._guest_ok(f"find {root} -type f -print0 | base64")
            try:
                raw = base64.b64decode("".join(output.split()), validate=True)
                names = sorted(path.decode("utf-8") for path in raw.split(b"\x00") if path)
            except (ValueError, UnicodeError) as exc:
                raise MicroVMCodingError("guest_file_listing_invalid") from exc
            prefix = self.workspace_root + "/"
            if any(not name.startswith(prefix) for name in names):
                raise MicroVMCodingError("guest_file_listing_escape")
            files = [name[len(prefix):] for name in names]
            observation = {"files": files[:200], "truncated": len(files) > 200}
        elif kind == "read_file":
            self._check_no_symlinks()
            guest_path = self._guest_file(action["path"])
            if self.virtio_read_enabled:
                if self._virtio_client is None:
                    raise MicroVMCodingError("guest_action_agent_missing")
                started = time.monotonic()
                try:
                    observation = self._virtio_client.read_file(action["path"])
                except VirtioSerialReadFallback:
                    self.metrics["virtio_read_fallbacks"] += 1
                except VirtioActionError as exc:
                    raise MicroVMCodingError("guest_action_read_failed") from exc
                else:
                    self.metrics["virtio_reads"] += 1
                    self.metrics["file_reads"] += 1
                    return {"observation": observation, "terminated": False}
                finally:
                    self.metrics["virtio_read_seconds"] += time.monotonic() - started
            quoted = shlex.quote(guest_path)
            command = (f"if [ -f {quoted} ]; then sha256sum {quoted}; "
                       f"wc -c < {quoted}; head -c 16000 {quoted} | base64; "
                       "else printf 'MISSING\\n'; fi")
            if (len(command.encode("utf-8")) > 3800
                    or not guest_path.isprintable()
                    or any(char in command for char in "\r\n")):
                command = (self._guest_path_assignment(guest_path)
                           + 'if [ -f "$fpb_target" ]; then '
                           + 'sha256sum < "$fpb_target"; '
                           + 'wc -c < "$fpb_target"; '
                           + 'head -c 16000 "$fpb_target" | base64; '
                           + '''else printf 'MISSING\\n'; fi''')
            if len(command.encode("utf-8")) > 3800:
                raise ValueError("Workspace path exceeds guest serial line bound")
            output = self._guest_ok(command)
            if output == "MISSING" or output.startswith("MISSING\n"):
                raise ValueError("Workspace file not found")
            lines = output.splitlines()
            if len(lines) < 2:
                raise MicroVMCodingError("guest_file_read_invalid")
            match = re.fullmatch(r"([0-9a-f]{64})\s+.+", lines[0])
            if not match or not lines[1].strip().isdigit():
                raise MicroVMCodingError("guest_file_read_invalid")
            try:
                data = base64.b64decode("".join(lines[2:]), validate=True)
            except ValueError as exc:
                raise MicroVMCodingError("guest_file_read_invalid") from exc
            size = int(lines[1].strip())
            if len(data) != min(size, 16000):
                raise MicroVMCodingError("guest_file_read_invalid")
            self.metrics["file_reads"] += 1
            observation = {"path": action["path"], "text": data.decode("utf-8", "replace"),
                           "sha256": match.group(1), "truncated": size > 16000}
        elif kind == "write_file":
            content = action["content"]
            if not isinstance(content, str) or len(content.encode("utf-8")) > 65536:
                raise ValueError("Write content exceeds 64 KiB")
            self._check_no_symlinks()
            guest_path = self._guest_file(action["path"])
            target = shlex.quote(guest_path)
            parent = shlex.quote(str(Path(guest_path).parent))
            temp = shlex.quote(guest_path + ".fpb-write-temp")
            setup = f"mkdir -p {parent} && : > {temp}"
            move = f"mv -f {temp} {target}"
            prefix = "printf '%s' '"
            suffix = f"' | base64 -d >> {temp}"
            if (max(len(setup.encode("utf-8")), len(move.encode("utf-8"))) > 3800
                    or not guest_path.isprintable()
                    or any(char in setup + move for char in "\r\n")):
                loader = self._guest_path_assignment(guest_path)
                setup = (loader + 'mkdir -p "${fpb_target%/*}"'
                         + ' && : > "${fpb_target}.fpb-write-temp"')
                move = (loader + 'mv -f "${fpb_target}.fpb-write-temp"'
                        + ' "$fpb_target"')
                prefix = loader + prefix
                suffix = "' | base64 -d >> \"${fpb_target}.fpb-write-temp\""
            if max(len(setup.encode("utf-8")), len(move.encode("utf-8"))) > 3800:
                raise ValueError("Workspace path exceeds guest serial line bound")
            # The complete serial-shell line includes another 191 bytes of
            # trusted framing. Bound the base64 body by the quoted path size
            # so even unusual valid paths stay below the guest TTY line cap.
            capacity = 3800 - len(prefix.encode("utf-8")) - len(suffix.encode("utf-8"))
            chunk_bytes = min(2700, (capacity // 4) * 3)
            if chunk_bytes < 1:
                raise ValueError("Workspace path exceeds guest serial line bound")
            self._guest_ok(setup)
            raw = content.encode("utf-8")
            for offset in range(0, len(raw), chunk_bytes):
                encoded = base64.b64encode(raw[offset:offset + chunk_bytes]).decode("ascii")
                self._guest_ok(prefix + encoded + suffix)
            self._guest_ok(move)
            self._check_no_symlinks()
            self.metrics["file_writes"] += 1
            observation = {"path": action["path"],
                           "sha256": hashlib.sha256(raw).hexdigest()}
        elif kind == "replace_text":
            if not self.replace_text_enabled:
                raise ValueError("Replace text is not enabled by the frozen task")
            if hashlib.sha256(Path(_replace_text.__file__).read_bytes()).hexdigest() != _REPLACE_SOURCE_SHA256:
                raise MicroVMCodingError("replace_text_helper_changed")
            command = self._replace_text_command(action)
            self._check_no_symlinks()
            output = self._guest_ok(command)
            match = re.fullmatch(r"FPB_REPLACE_RESULT=([A-Za-z0-9+/=]+)\n?", output)
            if match is None:
                raise MicroVMCodingError("guest_replace_result_invalid")
            try:
                observation = strict_json_loads(
                    base64.b64decode(match.group(1), validate=True).decode("utf-8"))
            except (ValueError, UnicodeError) as exc:
                raise MicroVMCodingError("guest_replace_result_invalid") from exc
            if not isinstance(observation, dict):
                raise MicroVMCodingError("guest_replace_result_invalid")
            if observation == {"status": "error", "reason": "adapter_error",
                               "error_type": "ValueError"}:
                pass
            elif observation.get("path") != action["path"]:
                raise MicroVMCodingError("guest_replace_result_invalid")
            elif set(observation) == {"path", "sha256"}:
                if (not isinstance(observation["sha256"], str)
                        or re.fullmatch(r"[0-9a-f]{64}", observation["sha256"]) is None):
                    raise MicroVMCodingError("guest_replace_result_invalid")
                self._check_no_symlinks()
                self.metrics["file_writes"] += 1
            elif (set(observation) != {"status", "reason", "path"}
                  or observation["status"] != "conflict"
                  or observation["reason"] not in {
                      "file_missing", "file_too_large", "file_changed",
                      "sha256_mismatch", "old_text_not_unique"}):
                raise MicroVMCodingError("guest_replace_result_invalid")
        elif kind == "run_visible_checks":
            self._check_no_symlinks()
            result = self._run_python_case(self.visible_check, unprivileged=True)
            self._check_no_symlinks()
            self.metrics["visible_checks"] += 1
            observation = {"passed": result["return_code"] == 0 and not result["truncated"],
                           "return_code": result["return_code"],
                           "stdout": result["stdout"][:12000]}
        else:
            self._check_no_symlinks()
            self._retire_virtio_read_agent()
            if self.stateless_verifier is not None:
                try:
                    self._check_stateless_source_binding()
                    self._timed("stateless_submit_precheck_seconds",
                                self._ensure_stateless_quiescent)
                    self._timed("stateless_submit_helper_install_seconds",
                                self.stateless_verifier.install)
                    self._timed("stateless_submit_batch_upload_seconds",
                                self.stateless_verifier.prepare_batch)
                    self._timed("stateless_submit_postupload_check_seconds",
                                self._ensure_stateless_quiescent)
                except (RuntimeError, ValueError, OSError, MicroVMRuntimeError) as exc:
                    raise MicroVMCodingError("stateless_verifier_prepare_failed") from exc
            try:
                self.snapshot = self._timed("vm_submit_snapshot_seconds",
                                            self.runtime.save_snapshot, "submitted")
            except MicroVMRuntimeError as exc:
                raise MicroVMCodingError("vm_submit_snapshot_failed") from exc
            self.submitted = True
            observation = {"status": "submitted", "snapshot_kind": self.snapshot["kind"]}
        return {"observation": observation, "terminated": kind == "submit"}

    def verify(self, *, now):
        if not self.submitted:
            raise ValueError("Submit before verification")
        if self.verified is not None:
            return copy.deepcopy(self.verified)
        verifier_file = self.verifier_dir / "verify.json"
        if verifier_file.is_symlink():
            return {"status": "pending", "reason": "verifier_differs_from_frozen_task"}
        if _sha256_file(verifier_file) != self.expected_binding["verifier_sha256"]:
            return {"status": "pending", "reason": "verifier_differs_from_frozen_task"}
        if self.stateless_verifier is not None:
            return self._verify_stateless(verifier_file)
        try:
            specification = strict_json_loads(verifier_file.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeError):
            return {"status": "pending", "reason": "invalid_verifier_specification"}
        if (not isinstance(specification, dict) or set(specification) != {"kind", "cases"}
                or specification["kind"] != "command_cases_v1"
                or not isinstance(specification["cases"], list)
                or not 1 <= len(specification["cases"]) <= 32):
            return {"status": "pending", "reason": "invalid_verifier_specification"}
        results = []
        try:
            for case in specification["cases"]:
                if (not isinstance(case, dict)
                        or set(case) != {"argv", "expected_stdout", "expected_returncode"}
                        or not isinstance(case["expected_stdout"], str)
                        or len(case["expected_stdout"].encode("utf-8")) > 12000
                        or type(case["expected_returncode"]) is not int
                        or not 0 <= case["expected_returncode"] <= 123):
                    return {"status": "pending", "reason": "invalid_verifier_specification"}
                argv = self._validate_python_argv(case["argv"])
                self.runtime.load_snapshot("submitted")
                self.metrics["full_vm_restores"] += 1
                output = self._run_python_case(
                    argv, timeout=max(30.0, self.command_timeout), unprivileged=True)
                results.append({"return_code": output["return_code"],
                                "stdout_sha256": hashlib.sha256(output["stdout_bytes"]).hexdigest(),
                                "passed": output["return_code"] == case["expected_returncode"]
                                          and not output["truncated"]
                                          and output["stdout_bytes"] == case["expected_stdout"].encode("utf-8")})
                self.metrics["hidden_cases"] += 1
        except (MicroVMRuntimeError, MicroVMCodingError, ValueError):
            return {"status": "pending", "reason": "vm_verifier_infrastructure_error"}
        if _sha256_file(verifier_file) != self.expected_binding["verifier_sha256"]:
            return {"status": "pending", "reason": "verifier_changed_during_execution"}
        reward = 1.0 if all(result["passed"] for result in results) else 0.0
        self.verified = {"status": "resolved", "reward": reward,
                         "available_at": datetime.now(timezone.utc).isoformat(),
                         "evidence": {"kind": "host_checked_qemu_full_vm_cases_v1",
                                      "snapshot_kind": self.snapshot["kind"],
                                      "verifier_sha256": self.expected_binding["verifier_sha256"],
                                      "case_results": results}}
        return copy.deepcopy(self.verified)

    def _verify_stateless(self, verifier_file):
        """Run only an explicitly bound, task-specific stateless case batch."""
        try:
            self._check_stateless_source_binding()
        except (ValueError, OSError):
            return {"status": "pending", "reason": "stateless_contract_differs_from_frozen_task"}
        if (self.stateless_verifier.program_sha256
                != self.expected_binding["stateless_helper_sha256"]
                or self.stateless_verifier.batch_code_sha256 is None):
            return {"status": "pending", "reason": "stateless_verifier_not_prepared"}
        try:
            # The one full restore re-establishes the submitted state before
            # a batch. Each case then gets its own mount/PID namespace inside
            # that VM; no full-VM restore occurs between cases.
            self._timed("stateless_verify_restore_seconds",
                        self.runtime.load_snapshot, "submitted")
            self.metrics["full_vm_restores"] += 1
            # This is the first inspection of the exact saved state. Check it
            # before running any hidden case, then check again after the batch.
            self._timed("stateless_verify_precheck_seconds",
                        self._ensure_stateless_quiescent)
            graded = self._timed("stateless_verify_batch_seconds",
                                 self.stateless_verifier.grade_batch)
            self._timed("stateless_verify_postcheck_seconds",
                        self._ensure_stateless_quiescent)
            if (not isinstance(graded, dict)
                    or graded.get("case_count") != len(self.stateless_verifier.cases)
                    or graded.get("reward") not in (0.0, 1.0)
                    or not isinstance(graded.get("case_results"), list)):
                raise MicroVMCodingError("stateless_verifier_result_invalid")
        except Exception:
            return {"status": "pending", "reason": "stateless_verifier_infrastructure_error"}
        try:
            self._check_stateless_source_binding()
            verifier_unchanged = (_sha256_file(verifier_file)
                                  == self.expected_binding["verifier_sha256"])
        except (ValueError, OSError):
            verifier_unchanged = False
        if not verifier_unchanged:
            return {"status": "pending", "reason": "verifier_changed_during_execution"}
        self.metrics["hidden_cases"] += graded["case_count"]
        self.metrics["stateless_batches"] += 1
        if self.stateless_verifier.batch_fallback_used:
            self.metrics["stateless_batch_fallbacks"] += 1
        self.verified = {"status": "resolved", "reward": graded["reward"],
                         "available_at": datetime.now(timezone.utc).isoformat(),
                         "evidence": {"kind": "host_checked_guest_stateless_namespaced_cases_v1",
                                      "snapshot_kind": self.snapshot["kind"],
                                      "case_isolation_kind": "per_case_mount_pid_namespace_overlay_v1",
                                      "verifier_sha256": self.expected_binding["verifier_sha256"],
                                      "contract_sha256": self.expected_binding["stateless_contract_sha256"],
                                      "guest_helper_sha256": self.expected_binding["stateless_helper_sha256"],
                                      "batch_code_sha256": self.stateless_verifier.batch_code_sha256,
                                      "batch_fallback_used": self.stateless_verifier.batch_fallback_used,
                                      "case_results": graded["case_results"]}}
        return copy.deepcopy(self.verified)

    def get_state(self):
        return {"started": self.started, "submitted": self.submitted,
                "verified": self.verified is not None,
                "verifier_mode": ("stateless_namespaced_batch_v1"
                                  if self.stateless_verifier is not None
                                  else "full_vm_per_case_v1"),
                "stateless_process_baseline": (
                    [{"pid": pid, "start_ticks": start, "exe": executable}
                     for (pid, start), executable in sorted(self._baseline_processes.items())]
                    if self._baseline_processes is not None else None),
                "snapshot_kind": self.snapshot["kind"] if self.snapshot else None,
                "policy_read_transport": self.read_transport,
                "guest_action_agent_install": (
                    "preinstalled_image_v1" if self.preinstalled_read_agent
                    else "serial_upload_v1") if self.virtio_read_enabled else None,
                "block_devices": copy.deepcopy(self.block_devices),
                "metrics": dict(self.metrics), "runtime": self.runtime.get_state()}

    def close(self):
        self.runtime.close()


class MicroVMBranchCodingAdapter(MicroVMCodingAdapter):
    """An independent VM restored from a parent's frozen full-state snapshot."""

    def __init__(self, parent_adapter, child_disk_path):
        if not isinstance(parent_adapter, MicroVMCodingAdapter):
            raise TypeError("parent_adapter must be a MicroVMCodingAdapter")
        super().__init__(None, verifier_dir=parent_adapter.verifier_dir,
                         visible_check=parent_adapter.visible_check,
                         workspace_root=parent_adapter.workspace_root,
                         command_timeout=parent_adapter.command_timeout,
                         stateless_verifier_contract=parent_adapter.stateless_contract_path,
                         stateless_task_path=parent_adapter.stateless_task_path)
        self.parent_adapter = parent_adapter
        self.child_disk_path = Path(child_disk_path)

    def reset_from_checkpoint(self, task, checkpoint_ref, *, now):
        if self.started or self.runtime is not None:
            raise ValueError("One VM branch adapter serves one episode")
        parent = self.parent_adapter
        if not parent.started or parent.submitted or parent.expected_binding is None:
            raise ValueError("Parent VM is no longer available for branching")
        task_sha256 = _validated_task_sha256(task)
        if task_sha256 != parent._task_sha256:
            raise ValueError("VM branch task differs from parent task")
        self.replace_text_enabled = _check_replace_text_helper(task)
        if self.replace_text_enabled != parent.replace_text_enabled:
            raise ValueError("VM branch edit helper differs from parent task")
        expected = task.get("metadata", {}).get("artifact_binding")
        if (checkpoint_ref.get("checkpoint_kind") != "full_vm_state_qcow2_v1"
                or checkpoint_ref.get("snapshot_tag") not in parent._branch_tags
                or checkpoint_ref.get("artifact_binding") != parent.expected_binding
                or expected != parent.expected_binding):
            raise ValueError("VM checkpoint differs from the frozen parent task")
        try:
            children = parent.runtime.fork_snapshot(
                checkpoint_ref["snapshot_tag"], [self.child_disk_path])
            if len(children) != 1:
                raise MicroVMCodingError("vm_branch_child_count_invalid")
            self.runtime = children[0]
            if self.stateless_verifier is not None:
                self.stateless_verifier.runtime = self.runtime
            self.expected_binding = copy.deepcopy(parent.expected_binding)
            self._task_sha256 = task_sha256
            self.block_devices = copy.deepcopy(parent.block_devices)
            self._guest_ok(f"test -d {shlex.quote(self.workspace_root)}")
            self._check_no_symlinks()
            if self.stateless_verifier is not None:
                self._check_stateless_source_binding()
                self._baseline_processes = copy.deepcopy(parent._baseline_processes)
                if (self._baseline_processes is None
                        or self._scan_guest_processes() != self._baseline_processes):
                    raise MicroVMCodingError("stateless_branch_process_state_differs")
            self.started = True
        except (MicroVMRuntimeError, MicroVMCodingError, ValueError, OSError) as exc:
            self.close()
            raise MicroVMCodingError("vm_branch_restore_failed") from exc
        return {"task_id": task["task_id"],
                "tools": [tool["name"] for tool in task["tool_manifest"]],
                "runtime_kind": "qemu_hvf_full_vm_qcow2_v1",
                "workspace_root": "/workspace",
                "visible_check": list(self.visible_check)}

    def close(self):
        if self.runtime is not None:
            self.runtime.close()
