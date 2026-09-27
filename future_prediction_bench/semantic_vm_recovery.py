"""Conservative, turn-aligned recovery for the trusted QEMU coding adapter.

This is a transfer of Crab's *decision* to avoid unnecessary checkpoints, not
its eBPF inspector or its ZFS/CRIU backend.  A checkpoint here is always a
full QEMU CPU/RAM/device/qcow2 snapshot.  Only the adapter's bounded read and
list tools can skip one, and only while the guest's workspace and live process
census agree before and after the turn.  Unknown work permanently taints the
episode.  A skipped turn is recovered by deterministic observation replay.

The journal protects against a coordinator exception after ``savevm`` and
before manifest publication: the old manifest still names the old snapshot.
It does not promise crash consistency across host power loss or across a
mutable external service (the experimental VM has no NIC).
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import re
import shlex
import stat
import tempfile
import time
from pathlib import Path

from .http import strict_json_loads

class RecoveryError(RuntimeError):
    """A checkpoint or replay failed; callers must not treat it as reward 0."""


class InjectedAfterSave(RecoveryError):
    """Test-only failure after a real snapshot, before manifest publication."""


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _frozen_host_task(task):
    """Accept either original task input or its validated digest-stamped copy."""
    from .realworld import validate_task

    if not isinstance(task, dict):
        raise ValueError("host_restart_task_invalid")
    source = copy.deepcopy(task)
    stamped = "task_sha256" in source or "reward_contract_sha256" in source
    if stamped:
        if not {"task_sha256", "reward_contract_sha256"} <= source.keys():
            raise ValueError("host_restart_task_stamp_incomplete")
        expected_task = source.pop("task_sha256")
        expected_reward = source.pop("reward_contract_sha256")
    frozen = validate_task(source)
    if stamped and (expected_task != frozen["task_sha256"]
                    or expected_reward != frozen["reward_contract_sha256"]):
        raise ValueError("host_restart_task_stamp_mismatch")
    return frozen


# The code runs as trusted root inside the already-isolated guest.  It rejects
# links and non-regular entries, hashes file contents using O_NOATIME, and
# includes metadata (including atime) that later programs might observe.
# Exceeding the bounds means "unknown", which forces a full checkpoint.
_FINGERPRINT_PROGRAM = r'''
import base64,hashlib,json,os,stat
root='/workspace'
h=hashlib.sha256()
entries=0
total=0
def fail(error): raise error
if not os.path.isdir(root): raise RuntimeError('workspace_missing')
for here,dirs,files in os.walk(root,topdown=True,followlinks=False,onerror=fail):
 dirs.sort(); files.sort()
 for name in ['.']+dirs+files:
  path=here if name=='.' else os.path.join(here,name)
  st=os.lstat(path)
  if stat.S_ISLNK(st.st_mode) or (stat.S_ISREG(st.st_mode) and st.st_nlink!=1): raise RuntimeError('unsafe_link')
  if not (stat.S_ISDIR(st.st_mode) or stat.S_ISREG(st.st_mode)): raise RuntimeError('unsafe_type')
  rel=os.path.relpath(path,root)
  entries+=1
  if entries>10000 or len(os.fsencode(rel))>4096 or rel.count(os.sep)>64: raise RuntimeError('bounded_tree_exceeded')
  meta=[rel,st.st_mode,st.st_uid,st.st_gid,st.st_size,
        st.st_mtime_ns,st.st_ctime_ns,st.st_atime_ns]
  h.update(json.dumps(meta,separators=(',',':'),ensure_ascii=False).encode())
  if stat.S_ISREG(st.st_mode):
   total+=st.st_size
   if total>67108864: raise RuntimeError('bounded_tree_exceeded')
   fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NOATIME)
   try:
    opened=os.fstat(fd)
    if (opened.st_dev,opened.st_ino,opened.st_size,opened.st_mtime_ns)!=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns): raise RuntimeError('raced_file')
    consumed=0
    while True:
     chunk=os.read(fd,1048576)
     if not chunk: break
     consumed+=len(chunk)
     if consumed>67108864: raise RuntimeError('bounded_tree_exceeded')
     h.update(chunk)
    after=os.fstat(fd)
    if (after.st_size,after.st_mtime_ns)!=(opened.st_size,opened.st_mtime_ns): raise RuntimeError('raced_file')
   finally: os.close(fd)
print('FPB_TREE='+base64.b64encode(json.dumps([h.hexdigest(),entries,total],separators=(',',':')).encode()).decode())
'''


def _guest_tree(adapter):
    encoded = base64.b64encode(_FINGERPRINT_PROGRAM.encode()).decode("ascii")
    code = f"exec(__import__('base64').b64decode('{encoded}'))"
    command = ("LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/aarch64-linux-gnu "
               "chroot /mnt/root /usr/local/bin/python3.12 -I -S -B -c "
               + shlex.quote(code))
    # The serial transport is a canonical TTY with a hard line bound.  A
    # compressed program would fit, but the fingerprint is only an optional
    # optimization; retain a clear bounded fallback if it cannot be run.
    if len(command.encode()) > 3700:
        raise RecoveryError("guest_fingerprint_program_too_large")
    output = adapter._guest_ok(command)
    match = re.fullmatch(r"FPB_TREE=([A-Za-z0-9+/=]+)\n?", output)
    if match is None:
        raise RecoveryError("guest_fingerprint_invalid")
    try:
        values = json.loads(base64.b64decode(match.group(1), validate=True))
    except (ValueError, UnicodeError) as exc:
        raise RecoveryError("guest_fingerprint_invalid") from exc
    if (not isinstance(values, list) or len(values) != 3
            or not isinstance(values[0], str)
            or re.fullmatch(r"[0-9a-f]{64}", values[0]) is None
            or any(type(value) is not int or value < 0 for value in values[1:])):
        raise RecoveryError("guest_fingerprint_invalid")
    return tuple(values)


class CodingVMRecoveryJournal:
    """Host-only snapshot manifest for one active, unsubmitted VM episode.

    The coordinator owns the adapter exclusively between `begin`, `apply`,
    and `recover`; an out-of-band trusted guest command must be recorded with
    `apply_opaque`.  A new instance may open the same manifest after an
    injected coordinator failure, while the QEMU VM remains available.
    """

    _READ_ONLY = frozenset({"list_files", "read_file"})
    _ACTIONS = _READ_ONLY | {"write_file", "replace_text", "run_visible_checks"}

    def __init__(self, adapter, directory, *, mode="selective", inspector=None):
        if mode not in {"selective", "every_turn"}:
            raise ValueError("mode must be selective or every_turn")
        self.adapter = adapter
        self.runtime = adapter.runtime
        if getattr(self.runtime, "backend", "aarch64_hvf") != "aarch64_hvf":
            raise RecoveryError("recovery_requires_aarch64_hvf_backend")
        root = Path(directory)
        if root.is_symlink():
            raise ValueError("Recovery directory cannot be a symlink")
        self.directory = root.resolve()
        self.path = self.directory / "manifest.json"
        self.terminal_attempt_path = self.directory / "terminal-attempt.json"
        self.terminal_submit_path = self.directory / "terminal-submit.json"
        self.terminal_result_path = self.directory / "terminal-result.json"
        self.host_restart_path = self.directory / "host-restart-context.json"
        self.mode = mode
        self.inspector = inspector or self._inspect
        self.manifest = None
        self.needs_recovery = False
        self._terminal_submission = None
        self._host_restart_task = None
        # A failed VM restart remains retryable only while reward is gated.
        # The old runtime may already have been closed by the first attempt.
        self._restart_pending = False
        self.metrics = {"snapshot_saves": 0, "skipped_turns": 0,
                        "inspection_seconds": 0.0, "snapshot_seconds": 0.0,
                        "manifest_seconds": 0.0, "replay_seconds": 0.0,
                        "replayed_turns": 0}

    def _binding(self, *, allow_submitted=False):
        if (not self.adapter.started or self.adapter.expected_binding is None
                or (self.adapter.submitted and not allow_submitted)):
            raise RecoveryError("active_unsubmitted_vm_episode_required")
        return _digest(self.adapter.expected_binding)

    def _source_binding(self):
        try:
            adapter_source = Path(type(self.adapter).step.__code__.co_filename)
            runtime_source = Path(type(self.runtime).load_snapshot.__code__.co_filename)
            if (adapter_source.is_symlink() or not adapter_source.is_file()
                    or runtime_source.is_symlink() or not runtime_source.is_file()):
                raise OSError("adapter or runtime source missing")
            return {"journal_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "adapter_sha256": hashlib.sha256(adapter_source.read_bytes()).hexdigest(),
                    "runtime_sha256": hashlib.sha256(runtime_source.read_bytes()).hexdigest()}
        except (AttributeError, OSError) as exc:
            raise RecoveryError("recovery_code_identity_unavailable") from exc

    def _inspect(self):
        return {"processes": sorted([pid, start, exe]
                                    for (pid, start), exe in self.adapter._scan_guest_processes().items()),
                "tree": _guest_tree(self.adapter)}

    def _publish(self, value):
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.directory.is_symlink() or not self.directory.is_dir():
            raise RecoveryError("unsafe_recovery_directory")
        payload = _canonical(value) + b"\n"
        started = time.monotonic()
        descriptor, name = tempfile.mkstemp(prefix=".manifest-", dir=self.directory)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, self.path)
            directory_fd = os.open(self.directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            Path(name).unlink(missing_ok=True)
            self.metrics["manifest_seconds"] += time.monotonic() - started

    def _save(self, tag):
        started = time.monotonic()
        result = self.runtime.save_snapshot(tag)
        if result.get("kind") != "full_vm_state_qcow2_v1" or result.get("tag") != tag:
            raise RecoveryError("snapshot_backend_returned_wrong_artifact")
        # QEMU must have returned successfully before publication.  Syncing
        # this host fd narrows the publication window but is not a guarantee
        # against loss of host power or controller write-cache faults.
        fd = os.open(self.runtime.disk_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        self.metrics["snapshot_saves"] += 1
        self.metrics["snapshot_seconds"] += time.monotonic() - started

    def begin(self):
        if self.path.exists() or self.manifest is not None:
            raise RecoveryError("recovery_journal_already_exists")
        binding = self._binding()
        # An inspector failure at startup is a hard error: no safe process
        # baseline exists from which later read-only skips could be approved.
        baseline = self.inspector()
        tag = "sr" + os.urandom(8).hex()
        self._save(tag)
        value = {"schema": "fpb-semantic-vm-recovery-v1", "mode": self.mode,
                 "binding_sha256": binding, "disk_path": str(self.runtime.disk_path),
                 "code_binding": self._source_binding(),
                 "baseline_processes": baseline["processes"], "turn": 0,
                 "checkpoint_turn": 0, "checkpoint_tag": tag,
                 "replay": [], "opaque_taint": False}
        self._publish(value)
        self.manifest = value
        return copy.deepcopy(value)

    def _validate_manifest(self, value, *, allow_submitted=False):
        if (not isinstance(value, dict)
                or set(value) != {"schema", "mode", "binding_sha256", "disk_path",
                                  "code_binding",
                                  "baseline_processes", "turn", "checkpoint_turn",
                                  "checkpoint_tag", "replay", "opaque_taint"}
                or value["schema"] != "fpb-semantic-vm-recovery-v1"
                or value["mode"] != self.mode
                or value["binding_sha256"] != self._binding(
                    allow_submitted=allow_submitted)
                or value["disk_path"] != str(self.runtime.disk_path)
                or value["code_binding"] != self._source_binding()
                or type(value["turn"]) is not int or value["turn"] < 0
                or type(value["checkpoint_turn"]) is not int
                or not 0 <= value["checkpoint_turn"] <= value["turn"]
                or not isinstance(value["checkpoint_tag"], str)
                or re.fullmatch(r"sr[0-9a-f]{16}", value["checkpoint_tag"]) is None
                or type(value["opaque_taint"]) is not bool
                or not isinstance(value["baseline_processes"], list)
                or len(value["baseline_processes"]) > 1024
                or not isinstance(value["replay"], list)
                or len(value["replay"]) != value["turn"] - value["checkpoint_turn"]
                or len(value["replay"]) > 64):
            raise RecoveryError("invalid_recovery_manifest")
        if any(not isinstance(item, list) or len(item) != 3
               or type(item[0]) is not int or item[0] <= 0
               or type(item[1]) is not int or item[1] < 0
               or not isinstance(item[2], str) or not item[2]
               for item in value["baseline_processes"]):
            raise RecoveryError("invalid_recovery_process_baseline")
        for entry in value["replay"]:
            if (not isinstance(entry, dict) or set(entry) != {"action", "result_sha256"}
                    or not isinstance(entry["action"], dict)
                    or entry["action"].get("action") not in self._READ_ONLY
                    or (set(entry["action"]) != {"action"}
                        if entry["action"].get("action") == "list_files"
                        else set(entry["action"]) != {"action", "path"}
                        or not isinstance(entry["action"].get("path"), str))
                    or not isinstance(entry["result_sha256"], str)
                    or re.fullmatch(r"[0-9a-f]{64}", entry["result_sha256"]) is None):
                raise RecoveryError("invalid_recovery_replay_entry")
        return value

    def open(self):
        """Inspect a committed manifest without releasing the recovery gate.

        The live guest may contain an uncommitted partial action or orphan
        snapshot. Only ``recover`` can make the loaded manifest authoritative
        by restoring its checkpoint and replaying committed reads.
        """
        self.needs_recovery = True
        if self.path.is_symlink() or not self.path.is_file():
            raise RecoveryError("recovery_manifest_missing_or_symlink")
        try:
            raw = self.path.read_bytes()
            if len(raw) > 1_000_000:
                raise RecoveryError("recovery_manifest_too_large")
            value = strict_json_loads(raw.decode("utf-8"))
        except (OSError, ValueError, UnicodeError) as exc:
            raise RecoveryError("recovery_manifest_invalid") from exc
        self.manifest = self._validate_manifest(value)
        return copy.deepcopy(self.manifest)

    def _state(self):
        if (self._terminal_submission is not None
                or self.terminal_attempt_path.exists()
                or self.terminal_attempt_path.is_symlink()
                or self.terminal_submit_path.exists()
                or self.terminal_submit_path.is_symlink()
                or self.terminal_result_path.exists()
                or self.terminal_result_path.is_symlink()
                or self.adapter.submitted):
            raise RecoveryError("terminal_episode_already_submitted")
        if self.needs_recovery:
            raise RecoveryError("recover_before_next_turn")
        if self.manifest is None:
            raise RecoveryError("begin_or_open_journal_first")
        return self.manifest

    def _commit(self, action, result, *, safe, opaque, inject_after_save):
        old = self._state()
        new = copy.deepcopy(old)
        new["turn"] += 1
        new["opaque_taint"] |= opaque
        skip = (self.mode == "selective" and safe and not new["opaque_taint"]
                and len(new["replay"]) < 64)
        try:
            if skip:
                new["replay"].append({"action": copy.deepcopy(action),
                                      "result_sha256": _digest(result)})
            else:
                tag = "sr" + os.urandom(8).hex()
                self._save(tag)
                if inject_after_save:
                    raise InjectedAfterSave("injected_after_snapshot_before_manifest")
                new["checkpoint_tag"] = tag
                new["checkpoint_turn"] = new["turn"]
                new["replay"] = []
            self._publish(new)
        except BaseException:
            self.needs_recovery = True
            raise
        self.manifest = new
        if skip:
            self.metrics["skipped_turns"] += 1
        return {"result": result, "checkpointed": not skip, "turn": new["turn"]}

    def apply(self, action, *, now, inject_after_save=False):
        state = self._state()
        if not isinstance(action, dict) or action.get("action") not in self._ACTIONS:
            raise ValueError("unsupported_turn_action")
        # Keep the executed command identical to the durable replay record
        # even if a caller retains and mutates its input object mid-turn.
        try:
            frozen_action = strict_json_loads(_canonical(action).decode("utf-8"))
        except (TypeError, ValueError, UnicodeError) as exc:
            raise ValueError("turn_action_not_canonical_json") from exc
        readonly = frozen_action["action"] in self._READ_ONLY
        pre = None
        if readonly and self.mode == "selective" and not state["opaque_taint"]:
            started = time.monotonic()
            try:
                pre = self.inspector()
            except Exception:
                pre = None
            except BaseException:
                self.needs_recovery = True
                raise
            finally:
                self.metrics["inspection_seconds"] += time.monotonic() - started
        try:
            result = self.adapter.step(frozen_action, now=now)
            safe = False
            if pre is not None:
                started = time.monotonic()
                try:
                    post = self.inspector()
                    safe = (pre == post and pre.get("processes") == state["baseline_processes"])
                except Exception:
                    safe = False
                finally:
                    self.metrics["inspection_seconds"] += time.monotonic() - started
            return self._commit(frozen_action, result, safe=safe,
                                opaque=frozen_action["action"] == "run_visible_checks",
                                inject_after_save=inject_after_save)
        except BaseException:
            # The guard covers the entire action-to-manifest interval, not
            # only a failure inside the guest command or snapshot backend.
            self.needs_recovery = True
            raise

    def apply_opaque(self, callback, *, inject_after_save=False):
        """Record an operator-owned arbitrary guest action as full-state only."""
        self._state()
        try:
            result = callback()
            return self._commit({"action": "opaque_host_guest_work"}, result,
                                safe=False, opaque=True,
                                inject_after_save=inject_after_save)
        except BaseException:
            self.needs_recovery = True
            raise

    def _terminal_manifest(self):
        """Read the last active-turn commit even after the adapter submitted."""
        if self.path.is_symlink() or not self.path.is_file():
            raise RecoveryError("terminal_manifest_missing_or_changed")
        try:
            raw = self.path.read_bytes()
            if len(raw) > 1_000_000:
                raise ValueError("oversized manifest")
            value = strict_json_loads(raw.decode("utf-8"))
            value = self._validate_manifest(value, allow_submitted=True)
        except (OSError, ValueError, UnicodeError) as exc:
            raise RecoveryError("terminal_manifest_missing_or_changed") from exc
        if raw != _canonical(value) + b"\n":
            raise RecoveryError("terminal_manifest_missing_or_changed")
        if self.manifest is not None and self.manifest != value:
            raise RecoveryError("terminal_manifest_missing_or_changed")
        self.manifest = value
        return raw

    def _read_terminal_record(self, path, label):
        if path.is_symlink():
            raise RecoveryError(label + "_corrupt")
        if not path.exists():
            return None
        if not path.is_file():
            raise RecoveryError(label + "_corrupt")
        try:
            raw = path.read_bytes()
            if len(raw) > 1_000_000:
                raise ValueError("oversized terminal record")
            value = strict_json_loads(raw.decode("utf-8"))
            if raw != _canonical(value) + b"\n":
                raise ValueError("noncanonical terminal record")
        except (OSError, ValueError, UnicodeError, TypeError) as exc:
            raise RecoveryError(label + "_corrupt") from exc
        return value

    def _publish_terminal_record(self, path, value, label):
        """Create once and fsync; a retry may accept only identical bytes."""
        payload = _canonical(value) + b"\n"
        existing = self._read_terminal_record(path, label)
        if existing is not None:
            if existing != value:
                raise RecoveryError(label + "_corrupt")
        else:
            descriptor, staging = tempfile.mkstemp(prefix=".terminal-", dir=self.directory)
            try:
                os.fchmod(descriptor, 0o600)
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.link(staging, path)
                os.unlink(staging)
            finally:
                Path(staging).unlink(missing_ok=True)
        directory_fd = os.open(self.directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def _validate_submitted_snapshot(self):
        snapshot = getattr(self.adapter, "snapshot", None)
        if (self.adapter.submitted is not True
                or not isinstance(snapshot, dict)
                or snapshot.get("kind") != "full_vm_state_qcow2_v1"
                or snapshot.get("tag") != "submitted"
                or snapshot.get("disk_path") != str(self.runtime.disk_path)
                or snapshot.get("kernel_sha256") != self.runtime.kernel_sha256
                or snapshot.get("initramfs_sha256") != self.runtime.initramfs_sha256
                or snapshot.get("readonly_disk_sha256s") !=
                    self.adapter.expected_binding.get("readonly_disk_sha256s")):
            raise RecoveryError("terminal_submitted_snapshot_unavailable")

    def _terminal_marker(self, submission, manifest_raw):
        if (not isinstance(submission, dict)
                or set(submission) != {"observation", "terminated"}
                or submission["terminated"] is not True
                or submission["observation"] != {
                    "status": "submitted", "snapshot_kind": "full_vm_state_qcow2_v1"}):
            raise RecoveryError("vm_submission_failed")
        self._validate_submitted_snapshot()
        core = {"schema": "fpb-terminal-submit-v1",
                "journal_sha256": hashlib.sha256(manifest_raw).hexdigest(),
                "binding_sha256": self.manifest["binding_sha256"],
                "code_binding": self.manifest["code_binding"],
                "turn": self.manifest["turn"],
                "disk_path": self.manifest["disk_path"],
                "snapshot_tag": "submitted",
                "snapshot_kind": "full_vm_state_qcow2_v1",
                "submission": copy.deepcopy(submission)}
        return {**core, "terminal_id": _digest(core)}

    @staticmethod
    def _terminal_attempt(manifest_raw):
        return {"schema": "fpb-terminal-attempt-v1",
                "journal_sha256": hashlib.sha256(manifest_raw).hexdigest()}

    @staticmethod
    def _valid_grading(grading):
        return (isinstance(grading, dict)
                and grading.get("status") == "resolved"
                and type(grading.get("reward")) in (int, float)
                and grading["reward"] in (0.0, 1.0))

    def _host_runtime_config(self):
        runtime = self.runtime
        return {"kernel_path": str(runtime.kernel_path),
                "initramfs_path": str(runtime.initramfs_path),
                "disk_path": str(runtime.disk_path),
                "readonly_disk_paths": [str(path) for path in runtime.readonly_disk_paths],
                "kernel_sha256": runtime.kernel_sha256,
                "initramfs_sha256": runtime.initramfs_sha256,
                "memory_mib": runtime.memory_mib, "vcpus": runtime.vcpus,
                "kernel_append": runtime.kernel_append,
                "enable_action_port": runtime.enable_action_port,
                "qemu_binary": runtime.qemu_binary,
                "qemu_img_binary": runtime.qemu_img_binary,
                "command_timeout": runtime.command_timeout}

    def _host_adapter_config(self):
        adapter = self.adapter
        return {"verifier_dir": str(adapter.verifier_dir),
                "visible_check": list(adapter.visible_check),
                "workspace_root": adapter.workspace_root,
                "read_transport": adapter.read_transport,
                "preinstalled_read_agent": adapter.preinstalled_read_agent,
                "command_timeout": adapter.command_timeout}

    def _host_disk_identity(self):
        path = self.runtime.disk_path
        if path.is_symlink():
            raise RecoveryError("host_restart_disk_identity_unavailable")
        try:
            info = path.stat()
        except OSError as exc:
            raise RecoveryError("host_restart_disk_identity_unavailable") from exc
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RecoveryError("host_restart_disk_identity_unavailable")
        return [info.st_dev, info.st_ino]

    @staticmethod
    def _old_host_vm_stopped(pid):
        # A recycled PID is conservatively refused. A live orphan may still
        # have the qcow2 open; no second writer may start against that disk.
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        return False

    def _host_restart_context(self, task):
        """Construct current context, including the *current* QEMU PID."""
        frozen = _frozen_host_task(task)
        binding = frozen["metadata"]["artifact_binding"]
        process = self.runtime._process
        pid = process.pid
        if (type(pid) is not int or pid <= 0 or process.poll() is not None
                or binding != self.adapter.expected_binding
                or frozen["task_sha256"] != getattr(self.adapter, "_task_sha256", None)
                or _digest(binding) != self.manifest["binding_sha256"]):
            raise RecoveryError("host_restart_task_differs_from_reset")
        return {"schema": "fpb-host-restart-context-v1",
                "task_sha256": frozen["task_sha256"],
                "artifact_binding": copy.deepcopy(binding),
                "journal_sha256": hashlib.sha256(
                    _canonical(self.manifest) + b"\n").hexdigest(),
                "code_binding": self._source_binding(),
                "runtime_config": self._host_runtime_config(),
                "adapter_config": self._host_adapter_config(),
                "disk_identity": self._host_disk_identity(),
                "old_qemu_pid": pid}

    def enable_host_restart(self, task):
        """Arm exact private configuration for publication before submit.

        Call after ``begin`` and before ``submit_and_verify``. The verifier
        stays on the host; neither its expected outputs nor its path are made
        policy-visible. An ambiguous submit without a durable submit marker
        remains unrecoverable rather than risking a second submit.
        """
        self._state()
        if (self.adapter.stateless_verifier is not None
                or self.adapter.read_transport != "serial_shell"
                or self.runtime.enable_action_port):
            raise RecoveryError("host_restart_full_vm_serial_only")
        try:
            core = self._host_restart_context(task)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise RecoveryError("host_restart_context_unavailable") from exc
        if (self._host_restart_task is not None
                and self._host_restart_context(self._host_restart_task)["task_sha256"]
                    != core["task_sha256"]):
            raise RecoveryError("host_restart_task_changed")
        self._host_restart_task = copy.deepcopy(task)
        return {"task_sha256": core["task_sha256"], "armed": True}

    def resume_verification_after_host_restart(self, task, *, now):
        """Rebuild a fresh adapter after its coordinator process was lost.

        The caller constructs an unstarted runtime/adapter from trusted local
        configuration and passes the original frozen task. The prior QEMU
        process must be gone. Only a durable submit marker permits reopening;
        no ``reset`` or ``submit`` is reissued. A missing or inconsistent
        context, marker, asset, code identity, or disk inode fails closed.
        """
        self.needs_recovery = True
        if (self.adapter.started or self.adapter.submitted
                or self.adapter.expected_binding is not None
                or self.adapter.snapshot is not None
                or self.runtime._process is not None):
            raise RecoveryError("fresh_host_adapter_required")
        context = self._read_terminal_record(self.host_restart_path,
                                             "host_restart_context_record")
        if not isinstance(context, dict) or set(context) != {
                "schema", "task_sha256", "artifact_binding", "journal_sha256",
                "code_binding",
                "runtime_config", "adapter_config", "disk_identity",
                "old_qemu_pid", "context_sha256"}:
            raise RecoveryError("host_restart_context_record_corrupt")
        core = {key: value for key, value in context.items()
                if key != "context_sha256"}
        if (context["schema"] != "fpb-host-restart-context-v1"
                or context["context_sha256"] != _digest(core)
                or type(context["old_qemu_pid"]) is not int
                or context["old_qemu_pid"] <= 0):
            raise RecoveryError("host_restart_context_record_corrupt")
        if not self._old_host_vm_stopped(context["old_qemu_pid"]):
            raise RecoveryError("old_host_qemu_still_running")
        try:
            frozen = _frozen_host_task(task)
            binding = frozen["metadata"]["artifact_binding"]
            current = self.adapter.artifact_binding()
        except (AttributeError, KeyError, TypeError, ValueError, OSError) as exc:
            raise RecoveryError("host_restart_task_or_assets_invalid") from exc
        # The writable qcow2 has advanced since reset; retain the frozen seed
        # digest while checking every other current kernel/readonly/verifier
        # and visible-check binding against the original task.
        current["disk_seed_sha256"] = binding.get("disk_seed_sha256")
        if (frozen["task_sha256"] != context["task_sha256"]
                or binding != context["artifact_binding"]
                or current != binding
                or self._source_binding() != context["code_binding"]
                or self._host_runtime_config() != context["runtime_config"]
                or self._host_adapter_config() != context["adapter_config"]
                or self._host_disk_identity() != context["disk_identity"]
                or self.adapter.stateless_verifier is not None
                or self.adapter.read_transport != "serial_shell"
                or self.runtime.enable_action_port):
            raise RecoveryError("host_restart_frozen_identity_mismatch")
        self.adapter.expected_binding = copy.deepcopy(binding)
        self.adapter.started = True
        self.adapter.submitted = True
        self.adapter.snapshot = {
            "kind": "full_vm_state_qcow2_v1", "tag": "submitted",
            "disk_path": str(self.runtime.disk_path),
            "kernel_sha256": self.runtime.kernel_sha256,
            "initramfs_sha256": self.runtime.initramfs_sha256,
            "readonly_disk_sha256s": list(binding["readonly_disk_sha256s"])}
        manifest_raw = self._terminal_manifest()
        if context["journal_sha256"] != hashlib.sha256(manifest_raw).hexdigest():
            raise RecoveryError("host_restart_frozen_identity_mismatch")
        attempt = self._read_terminal_record(self.terminal_attempt_path,
                                             "terminal_attempt_record")
        marker = self._read_terminal_record(self.terminal_submit_path,
                                            "terminal_submit_record")
        if attempt != self._terminal_attempt(manifest_raw):
            raise RecoveryError("terminal_attempt_record_corrupt")
        if (not isinstance(marker, dict)
                or set(marker) != {"schema", "journal_sha256", "binding_sha256",
                                   "code_binding", "turn", "disk_path", "snapshot_tag",
                                   "snapshot_kind", "submission", "terminal_id"}
                or marker != self._terminal_marker(marker["submission"], manifest_raw)):
            raise RecoveryError("terminal_submit_record_corrupt")
        result = self._read_terminal_record(self.terminal_result_path,
                                            "terminal_result_record")
        if result is not None:
            if (not isinstance(result, dict)
                    or set(result) != {"schema", "terminal_id", "submit_record_sha256",
                                       "grading_sha256", "grading"}
                    or result["schema"] != "fpb-terminal-result-v1"
                    or result["terminal_id"] != marker["terminal_id"]
                    or result["submit_record_sha256"] != _digest(marker)
                    or not self._valid_grading(result["grading"])
                    or result["grading_sha256"] != _digest(result["grading"])):
                raise RecoveryError("terminal_result_record_corrupt")
            self.adapter.verified = copy.deepcopy(result["grading"])
            return self.resume_verification(now=now)
        try:
            self.runtime.start(paused=True)
            if (tuple(binding["readonly_disk_sha256s"]) !=
                    tuple(self.runtime._readonly_disk_sha256s)):
                raise RecoveryError("replacement_readonly_disk_mismatch")
            self.runtime.load_snapshot("submitted", resume=True,
                                       full_validation=True)
        except BaseException:
            self.runtime.close()
            raise
        return self.resume_verification(now=now)

    def resume_verification(self, *, now):
        """Retry grading once submitted, using one caller and the same live adapter.

        Full-VM verifier mode reloads `submitted` before each hidden case;
        stateless mode reloads once before a namespaced case batch. Reward is
        returned only after an immutable, host-private result record is durable.
        """
        self.needs_recovery = True  # Never permit more policy turns.
        manifest_raw = self._terminal_manifest()
        attempt = self._read_terminal_record(self.terminal_attempt_path,
                                             "terminal_attempt_record")
        if attempt != self._terminal_attempt(manifest_raw):
            raise RecoveryError("terminal_attempt_record_corrupt")
        self._publish_terminal_record(self.terminal_attempt_path, attempt,
                                      "terminal_attempt_record")
        self._validate_submitted_snapshot()
        marker = self._read_terminal_record(self.terminal_submit_path,
                                            "terminal_submit_record")
        if marker is None:
            if self._terminal_submission is None:
                raise RecoveryError("terminal_submit_record_missing")
            marker = self._terminal_marker(self._terminal_submission, manifest_raw)
            self._sync_submitted_disk()
        else:
            if (not isinstance(marker, dict)
                    or set(marker) != {"schema", "journal_sha256", "binding_sha256",
                                       "code_binding", "turn", "disk_path", "snapshot_tag",
                                       "snapshot_kind", "submission", "terminal_id"}):
                raise RecoveryError("terminal_submit_record_corrupt")
            expected = self._terminal_marker(marker["submission"], manifest_raw)
            if marker != expected:
                raise RecoveryError("terminal_submit_record_corrupt")
            if (self._terminal_submission is not None
                    and marker["submission"] != self._terminal_submission):
                raise RecoveryError("terminal_submit_record_corrupt")
        # An earlier call may have linked the record and then failed its
        # directory fsync. Re-fsync even when the identical file exists.
        self._publish_terminal_record(self.terminal_submit_path, marker,
                                      "terminal_submit_record")
        self._terminal_submission = copy.deepcopy(marker["submission"])
        result = self._read_terminal_record(self.terminal_result_path,
                                            "terminal_result_record")
        if result is not None:
            if (not isinstance(result, dict)
                    or set(result) != {"schema", "terminal_id", "submit_record_sha256",
                                       "grading_sha256", "grading"}
                    or result["schema"] != "fpb-terminal-result-v1"
                    or result["terminal_id"] != marker["terminal_id"]
                    or result["submit_record_sha256"] != _digest(marker)
                    or not self._valid_grading(result["grading"])
                    or result["grading_sha256"] != _digest(result["grading"])):
                raise RecoveryError("terminal_result_record_corrupt")
            cached = getattr(self.adapter, "verified", None)
            if cached is None or _digest(cached) != result["grading_sha256"]:
                raise RecoveryError("terminal_result_record_corrupt")
            self._publish_terminal_record(self.terminal_result_path, result,
                                          "terminal_result_record")
            return {"submission": copy.deepcopy(marker["submission"]),
                    "grading": copy.deepcopy(result["grading"]),
                    "terminal_id": marker["terminal_id"]}
        try:
            grading = self.adapter.verify(now=now)
            if (isinstance(grading, dict) and grading.get("status") == "pending"
                    and "reward" not in grading):
                return {"submission": copy.deepcopy(marker["submission"]),
                        "grading": copy.deepcopy(grading),
                        "terminal_id": marker["terminal_id"]}
            if not self._valid_grading(grading):
                raise RecoveryError("terminal_grading_invalid")
            value = {"schema": "fpb-terminal-result-v1",
                     "terminal_id": marker["terminal_id"],
                     "submit_record_sha256": _digest(marker),
                     "grading_sha256": _digest(grading),
                     "grading": copy.deepcopy(grading)}
            self._publish_terminal_record(self.terminal_result_path, value,
                                          "terminal_result_record")
        except BaseException:
            self.needs_recovery = True
            raise
        return {"submission": copy.deepcopy(marker["submission"]),
                "grading": copy.deepcopy(grading),
                "terminal_id": marker["terminal_id"]}

    def _sync_submitted_disk(self):
        descriptor = os.open(self.runtime.disk_path,
                             os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def submit_and_verify(self, *, now):
        """Gate terminal reward on a committed recovery boundary.

        The adapter's own `submit` saves a full VM for host-private grading.
        This method deliberately refuses to submit while any prior turn is
        uncommitted or requires rollback. The caller must route terminal
        grading through this coordinator rather than bypassing the adapter.
        Calls must be serialized by the owner of this journal and adapter.
        """
        if (self._terminal_submission is not None
                or self.terminal_attempt_path.exists()
                or self.terminal_attempt_path.is_symlink()
                or self.terminal_submit_path.exists()
                or self.terminal_submit_path.is_symlink()
                or self.terminal_result_path.exists()
                or self.terminal_result_path.is_symlink()
                or self.adapter.submitted):
            if not self.adapter.submitted:
                raise RecoveryError("terminal_submit_attempt_uncertain")
            return self.resume_verification(now=now)
        self._state()
        manifest_raw = self._terminal_manifest()
        self.needs_recovery = True
        if self._host_restart_task is not None:
            try:
                core = self._host_restart_context(self._host_restart_task)
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                raise RecoveryError("host_restart_context_unavailable") from exc
            self._publish_terminal_record(
                self.host_restart_path, {**core, "context_sha256": _digest(core)},
                "host_restart_context_record")
        # Publishing before savevm closes the ambiguous fixed-name-tag window:
        # if submit fails, neither this nor a fresh journal may submit again.
        self._publish_terminal_record(
            self.terminal_attempt_path, self._terminal_attempt(manifest_raw),
            "terminal_attempt_record")
        try:
            submission = self.adapter.step({"action": "submit"}, now=now)
        except BaseException:
            raise
        # Preserve the validated in-memory observation before publishing the
        # marker. A failed fsync/link can be retried without a second submit.
        self._terminal_marker(submission, manifest_raw)
        self._terminal_submission = copy.deepcopy(submission)
        return self.resume_verification(now=now)

    def recover(self, *, now):
        """Restore the last committed full VM and replay committed reads.

        The caller retries an action whose snapshot/manifest commit failed;
        that uncommitted action is intentionally not replayed here.
        """
        if (self.terminal_attempt_path.exists()
                or self.terminal_attempt_path.is_symlink()
                or self.terminal_submit_path.exists()
                or self.terminal_submit_path.is_symlink()):
            raise RecoveryError("terminal_submit_attempt_uncertain")
        state = self.open()
        started = time.monotonic()
        try:
            self.runtime.load_snapshot(state["checkpoint_tag"], resume=True)
            for entry in state["replay"]:
                # A replayed read must remain semantically read-only now, not
                # only when it was first recorded. Equal visible output alone
                # cannot rule out an invisible file or process mutation.
                before = self.inspector()
                if (not isinstance(before, dict)
                        or before.get("processes") != state["baseline_processes"]):
                    raise RecoveryError("replayed_read_state_mismatch")
                result = self.adapter.step(entry["action"], now=now)
                after = self.inspector()
                if before != after:
                    raise RecoveryError("replayed_read_state_mismatch")
                if _digest(result) != entry["result_sha256"]:
                    raise RecoveryError("replayed_read_observation_mismatch")
                self.metrics["replayed_turns"] += 1
        except Exception as exc:
            self.needs_recovery = True
            raise RecoveryError("vm_recovery_failed") from exc
        finally:
            self.metrics["replay_seconds"] += time.monotonic() - started
        self.needs_recovery = False
        return {"turn": state["turn"], "checkpoint_turn": state["checkpoint_turn"],
                "replayed_turns": len(state["replay"])}

    def recover_after_vm_crash(self, replacement_runtime, *, now):
        """Restart a crashed QEMU process from its persisted qcow2 snapshot.

        Host adapter and journal state remain alive.  This is VM-process crash
        recovery, not recovery of a killed host coordinator or agent process.
        The restored VM starts paused, then `loadvm` makes the committed CPU,
        RAM, device, and disk state authoritative before any guest command.
        """
        old = self.runtime
        process = getattr(old, "_process", None)
        if not ((process is not None and process.poll() is not None)
                or (process is None and self._restart_pending and self.needs_recovery)):
            raise RecoveryError("old_vm_has_not_crashed")
        if (getattr(self.adapter, "stateless_verifier", None) is not None
                or getattr(replacement_runtime, "backend", "aarch64_hvf") != "aarch64_hvf"
                or getattr(old, "backend", "aarch64_hvf") != "aarch64_hvf"
                or replacement_runtime.disk_path != old.disk_path
                or replacement_runtime.kernel_sha256 != old.kernel_sha256
                or replacement_runtime.initramfs_sha256 != old.initramfs_sha256
                or replacement_runtime.readonly_disk_paths != old.readonly_disk_paths
                or replacement_runtime.memory_mib != old.memory_mib
                or replacement_runtime.vcpus != old.vcpus):
            raise RecoveryError("replacement_vm_differs_from_frozen_runtime")
        self.needs_recovery = True
        self._restart_pending = True
        old.close()
        try:
            replacement_runtime.start(paused=True)
            frozen_readonly = self.adapter.expected_binding.get("readonly_disk_sha256s")
            actual_readonly = getattr(replacement_runtime, "_readonly_disk_sha256s", None)
            if (not isinstance(frozen_readonly, list)
                    or actual_readonly is None
                    or tuple(frozen_readonly) != tuple(actual_readonly)):
                raise RecoveryError("replacement_readonly_disk_mismatch")
            self.runtime = replacement_runtime
            self.adapter.runtime = replacement_runtime
            recovered = self.recover(now=now)
            self._restart_pending = False
            return recovered
        except BaseException:
            self.needs_recovery = True
            replacement_runtime.close()
            raise

    def resume_verification_after_vm_crash(self, replacement_runtime, *, now):
        """Resume an interrupted terminal verifier on a replacement QEMU VM.

        The submitted full-state snapshot and terminal marker must already be
        committed. The host adapter remains alive and owns its verifier and
        private expected outputs; this does not reconstruct a crashed host or
        reissue ``submit``. A failed replacement start/load leaves the reward
        gate closed and permits another replacement attempt.
        """
        self.needs_recovery = True
        manifest_raw = self._terminal_manifest()
        attempt = self._read_terminal_record(self.terminal_attempt_path,
                                             "terminal_attempt_record")
        if attempt != self._terminal_attempt(manifest_raw):
            raise RecoveryError("terminal_attempt_record_corrupt")
        marker = self._read_terminal_record(self.terminal_submit_path,
                                            "terminal_submit_record")
        if (not isinstance(marker, dict)
                or set(marker) != {"schema", "journal_sha256", "binding_sha256",
                                   "code_binding", "turn", "disk_path", "snapshot_tag",
                                   "snapshot_kind", "submission", "terminal_id"}
                or marker != self._terminal_marker(marker["submission"], manifest_raw)):
            raise RecoveryError("terminal_submit_record_corrupt")
        # Re-fsync an existing marker before exposing any recovered reward:
        # its earlier publication may have failed after link but before fsync.
        self._publish_terminal_record(self.terminal_submit_path, marker,
                                      "terminal_submit_record")
        if self._read_terminal_record(self.terminal_result_path,
                                      "terminal_result_record") is not None:
            return self.resume_verification(now=now)

        old = self.runtime
        process = getattr(old, "_process", None)
        # The serial transport may close a dead QEMU process while reporting
        # an infrastructure error, leaving `_process` as None. The committed
        # terminal marker and absent result still fence this restart.
        if process is not None and process.poll() is None:
            raise RecoveryError("old_vm_has_not_crashed")
        if (getattr(self.adapter, "stateless_verifier", None) is not None
                or getattr(replacement_runtime, "backend", "aarch64_hvf") != "aarch64_hvf"
                or getattr(old, "backend", "aarch64_hvf") != "aarch64_hvf"
                or replacement_runtime.disk_path != old.disk_path
                or replacement_runtime.kernel_sha256 != old.kernel_sha256
                or replacement_runtime.initramfs_sha256 != old.initramfs_sha256
                or replacement_runtime.readonly_disk_paths != old.readonly_disk_paths
                or replacement_runtime.memory_mib != old.memory_mib
                or replacement_runtime.vcpus != old.vcpus
                or getattr(replacement_runtime, "kernel_append", None) !=
                    getattr(old, "kernel_append", None)
                or getattr(replacement_runtime, "enable_action_port", False) !=
                    getattr(old, "enable_action_port", False)):
            raise RecoveryError("replacement_vm_differs_from_frozen_runtime")
        old.close()
        try:
            replacement_runtime.start(paused=True)
            frozen_readonly = self.adapter.expected_binding.get("readonly_disk_sha256s")
            actual_readonly = getattr(replacement_runtime, "_readonly_disk_sha256s", None)
            if (not isinstance(frozen_readonly, list)
                    or actual_readonly is None
                    or tuple(frozen_readonly) != tuple(actual_readonly)):
                raise RecoveryError("replacement_readonly_disk_mismatch")
            replacement_runtime.load_snapshot("submitted", resume=True)
        except BaseException:
            replacement_runtime.close()
            raise
        self.runtime = replacement_runtime
        self.adapter.runtime = replacement_runtime
        return self.resume_verification(now=now)
