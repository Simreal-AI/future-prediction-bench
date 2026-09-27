"""Reusable, host-only full-VM snapshot templates with pinned QEMU backends.

A template is a *copy* of a quiescent qcow2 containing QEMU's savevm state.
The parent may resume or exit after export. Each child receives an independent
copy-on-write (where APFS permits) qcow2 inode before loading the saved CPU,
RAM, device, and disk state. This is disk copy plus VM restore, not a live VM
fork or a security boundary beyond the underlying QEMU guest configuration.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .microvm_runtime import (
    MicroVMRuntime, MicroVMRuntimeError, _clone_or_copy_qcow2,
    _require_backend, _require_sha, _sha256_file,
)


_SCHEMA = "qemu_full_vm_template_v2"
_LEGACY_SCHEMA = "qemu_hvf_full_vm_template_v1"
_TAG = re.compile(r"[A-Za-z0-9_-]{1,32}\Z")


def _canonical_bytes(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("ascii")


def _check_new_file(path: Path, *, reserved=()) -> Path:
    raw = Path(path)
    if ".." in raw.parts:
        raise ValueError("Template or child path cannot contain parent traversal")
    absolute = Path(os.path.abspath(raw))
    if absolute.is_symlink() or absolute.parent.is_symlink():
        raise ValueError("Template or child path cannot be a symlink")
    resolved = absolute.resolve()
    if resolved in reserved:
        raise ValueError("Template or child path overlaps a VM artifact")
    if absolute.exists() or absolute.is_symlink():
        raise FileExistsError(absolute)
    if not absolute.parent.is_dir():
        raise ValueError("Template or child parent directory is required")
    return resolved


def _unlink_if_owned(path: Path, identity: tuple[int, int]):
    """Best-effort cleanup without deleting a replaced, unrelated pathname."""
    try:
        current = path.lstat()
    except FileNotFoundError:
        return
    if (stat.S_ISREG(current.st_mode)
            and (current.st_dev, current.st_ino) == identity):
        path.unlink()


def _read_manifest(path: Path, expected_id: str) -> dict:
    _require_sha(expected_id, "expected_template_id")
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16384:
        raise MicroVMRuntimeError("template_manifest_missing_invalid_or_symlink")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError, OSError) as exc:
        raise MicroVMRuntimeError("template_manifest_invalid") from exc
    if not isinstance(manifest, dict):
        raise MicroVMRuntimeError("template_manifest_invalid")
    claimed_id = manifest.pop("template_id", None)
    actual_id = hashlib.sha256(_canonical_bytes(manifest)).hexdigest()
    if claimed_id != expected_id or actual_id != expected_id:
        raise MicroVMRuntimeError("template_id_mismatch")
    if not isinstance(manifest.get("schema"), str) \
            or manifest["schema"] not in {_SCHEMA, _LEGACY_SCHEMA} \
            or not isinstance(manifest.get("tag"), str) \
            or not _TAG.fullmatch(manifest["tag"]):
        raise MicroVMRuntimeError("template_manifest_schema_or_tag_invalid")
    if manifest["schema"] == _LEGACY_SCHEMA:
        # Old manifests were exported only by the ARM/HVF runtime. Validate
        # their original digest before adding the inferred internal field;
        # a backend-bearing record must use the new schema and hash binding.
        if "backend" in manifest:
            raise MicroVMRuntimeError("template_legacy_backend_field_invalid")
        manifest["backend"] = "aarch64_hvf"
    else:
        try:
            _require_backend(manifest.get("backend"))
        except ValueError as exc:
            raise MicroVMRuntimeError("template_backend_invalid") from exc
    for field in ("disk_sha256", "kernel_sha256", "initramfs_sha256"):
        _require_sha(manifest.get(field), field)
    ro_paths = manifest.get("readonly_disk_paths")
    ro_hashes = manifest.get("readonly_disk_sha256s")
    if not isinstance(ro_paths, list) or not isinstance(ro_hashes, list) \
            or len(ro_paths) != len(ro_hashes) or len(ro_paths) > 4:
        raise MicroVMRuntimeError("template_readonly_disk_binding_invalid")
    for value in ro_hashes:
        _require_sha(value, "readonly_disk_sha256")
    for field in ("kernel_path", "initramfs_path", "*readonly_disk_paths"):
        values = ro_paths if field.startswith("*") else [manifest.get(field)]
        if any(not isinstance(value, str) or not Path(value).is_absolute()
               for value in values):
            raise MicroVMRuntimeError("template_artifact_path_invalid")
    if (type(manifest.get("memory_mib")) is not int
            or type(manifest.get("vcpus")) is not int
            or not isinstance(manifest.get("kernel_append"), str)
            or not isinstance(manifest.get("qemu_binary"), str)
            or not isinstance(manifest.get("qemu_img_binary"), str)
            or not isinstance(manifest.get("command_timeout"), (int, float))):
        raise MicroVMRuntimeError("template_vm_configuration_invalid")
    return manifest


@dataclass(frozen=True)
class MicroVMTemplate:
    """Handle to a sealed template; reopen with the exact returned digest."""

    manifest_path: Path
    template_id: str
    export_seconds: float = 0.0
    export_clone_mode: str | None = None

    @property
    def disk_path(self) -> Path:
        return self.manifest_path.with_suffix("")

    @classmethod
    def export(cls, parent: MicroVMRuntime, disk_path, *, tag="warm"):
        """Pause, save, copy, checksum, seal, and resume a running parent VM.

        The template remains usable after ``parent.close()``. Only the parent
        VM's host owner may call this method; its metadata is never returned
        to the guest or policy. Do not issue concurrent parent VM commands.
        """
        if not isinstance(parent, MicroVMRuntime):
            raise TypeError("parent must be a MicroVMRuntime")
        # The template manifest and child constructor omit the action-port
        # device. Even a retired port remains part of QEMU's saved topology.
        if parent.enable_action_port:
            raise MicroVMRuntimeError("template_action_port_not_supported")
        if not isinstance(tag, str) or not _TAG.fullmatch(tag):
            raise ValueError("Invalid template snapshot tag")
        parent._check_running()
        reserved = {parent.disk_path, parent.kernel_path, parent.initramfs_path,
                    *parent.readonly_disk_paths}
        disk = _check_new_file(Path(disk_path), reserved=reserved)
        manifest_path = _check_new_file(Path(str(disk) + ".json"),
                                        reserved=reserved | {disk})
        started = time.monotonic()
        paused_here = False
        created_disk = False
        created_manifest = False
        clone_mode = None
        try:
            status = parent._hmp("info status")
            match = re.search(r"(?im)^\s*VM status:\s*(running|paused)\b", status)
            if match is None:
                raise MicroVMRuntimeError("template_parent_status_unrecognized")
            if match.group(1).lower() == "running":
                parent._hmp("stop")
                paused_here = True
            if not re.search(r"(?im)^\s*VM status:\s*paused\b",
                             parent._hmp("info status")):
                raise MicroVMRuntimeError("template_parent_not_quiescent")
            parent._verify_fork_artifacts()
            parent.save_snapshot(tag)
            source_sha = _sha256_file(parent.disk_path)
            clone_mode = _clone_or_copy_qcow2(parent.disk_path, disk)
            created_disk = True
            if _sha256_file(disk) != source_sha or _sha256_file(parent.disk_path) != source_sha:
                raise MicroVMRuntimeError("template_disk_digest_mismatch")
            parent._verify_fork_artifacts()
            # Read-only mode is an accidental-write guard. Each child clone
            # is checked against the template digest before boot; chmod is not
            # an authenticity claim.
            disk.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            payload = {
                "schema": _SCHEMA, "tag": tag, "disk_sha256": source_sha,
                "backend": parent.backend,
                "kernel_path": str(parent.kernel_path),
                "kernel_sha256": parent.kernel_sha256,
                "initramfs_path": str(parent.initramfs_path),
                "initramfs_sha256": parent.initramfs_sha256,
                "readonly_disk_paths": [str(path) for path in parent.readonly_disk_paths],
                "readonly_disk_sha256s": list(parent._readonly_disk_sha256s),
                "memory_mib": parent.memory_mib, "vcpus": parent.vcpus,
                "kernel_append": parent.kernel_append,
                "qemu_binary": parent.qemu_binary,
                "qemu_img_binary": parent.qemu_img_binary,
                "command_timeout": parent.command_timeout,
            }
            template_id = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
            manifest = dict(payload, template_id=template_id)
            # Exclusive create: a pre-existing manifest is never overwritten.
            with manifest_path.open("x", encoding="utf-8") as stream:
                created_manifest = True
                json.dump(manifest, stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            manifest_path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            return cls(manifest_path, template_id, time.monotonic() - started,
                       clone_mode)
        except BaseException:
            if created_manifest:
                manifest_path.unlink(missing_ok=True)
            if created_disk:
                disk.unlink(missing_ok=True)
            raise
        finally:
            if paused_here:
                try:
                    parent._hmp("cont")
                except Exception:
                    # A failed resume means export cannot be reported as a
                    # successful, reusable operation.
                    if created_manifest:
                        manifest_path.unlink(missing_ok=True)
                    if created_disk:
                        disk.unlink(missing_ok=True)
                    raise

    @classmethod
    def open(cls, manifest_path, *, expected_template_id: str):
        """Reopen after the parent has exited, with the caller's saved ID."""
        raw = Path(manifest_path)
        if ".." in raw.parts or raw.is_symlink() or raw.parent.is_symlink():
            raise ValueError("Template manifest path cannot contain symlinks or traversal")
        path = raw.resolve()
        _read_manifest(path, expected_template_id)
        return cls(path, expected_template_id)

    def _verified_manifest(self) -> dict:
        manifest = _read_manifest(self.manifest_path, self.template_id)
        disk = self.disk_path
        if not disk.is_file() or disk.is_symlink():
            raise MicroVMRuntimeError("template_disk_missing_or_symlink")
        for path_value, expected in ((manifest["kernel_path"], manifest["kernel_sha256"]),
                                     (manifest["initramfs_path"], manifest["initramfs_sha256"]),
                                     *zip(manifest["readonly_disk_paths"],
                                          manifest["readonly_disk_sha256s"])):
            path = Path(path_value)
            if not path.is_file() or path.is_symlink() or _sha256_file(path) != expected:
                raise MicroVMRuntimeError("template_pinned_artifact_digest_mismatch")
        return manifest

    def spawn(self, child_disk_paths, *, max_workers=4, popen_factory=None,
              parallel_clone_verification=False):
        """Clone and restore one to four independent VMs from this template.

        Opt-in parallel provisioning overlaps each independent disk clone and
        full SHA-256 check. Every check completes before any child boots.
        """
        if type(max_workers) is not int or not 1 <= max_workers <= 4:
            raise ValueError("max_workers must be in [1, 4]")
        if type(parallel_clone_verification) is not bool:
            raise ValueError("parallel_clone_verification must be boolean")
        if not isinstance(child_disk_paths, (list, tuple)) \
                or not 1 <= len(child_disk_paths) <= 4:
            raise ValueError("Provide one to four child disk paths")
        manifest = self._verified_manifest()
        reserved = {self.disk_path, self.manifest_path,
                    Path(manifest["kernel_path"]), Path(manifest["initramfs_path"]),
                    *(Path(path) for path in manifest["readonly_disk_paths"])}
        paths = tuple(_check_new_file(Path(path), reserved=reserved)
                      for path in child_disk_paths)
        if len(set(paths)) != len(paths):
            raise ValueError("Child disk paths must be unique")
        children = []
        created = []
        created_lock = threading.Lock()
        try:
            def provision(path):
                # The clone helper itself publishes atomically, but keeping
                # its target inside a private directory also handles a
                # partial-target exception from an alternate clone backend.
                # os.link is no-replace publication: an unrelated file raced
                # into `path` is never overwritten or enrolled for cleanup.
                staging_dir = Path(tempfile.mkdtemp(
                    prefix=".fpb-template-child-", dir=path.parent))
                staging = staging_dir / "disk.qcow2"
                try:
                    _clone_or_copy_qcow2(self.disk_path, staging)
                    info = staging.lstat()
                    if not stat.S_ISREG(info.st_mode):
                        raise MicroVMRuntimeError("template_child_clone_not_regular")
                    identity = (info.st_dev, info.st_ino)
                    os.link(staging, path)
                    try:
                        with created_lock:
                            created.append((path, identity))
                    except BaseException:
                        _unlink_if_owned(path, identity)
                        raise
                    # No child boots until every independent published clone
                    # matches the pinned manifest digest.
                    if _sha256_file(path) != manifest["disk_sha256"]:
                        raise MicroVMRuntimeError("template_child_clone_digest_mismatch")
                    # The source is sealed read-only; QEMU needs a writable,
                    # independent child inode for active qcow2 changes.
                    path.chmod(stat.S_IRUSR | stat.S_IWUSR)
                finally:
                    shutil.rmtree(staging_dir, ignore_errors=True)

            if parallel_clone_verification and len(paths) > 1 and max_workers > 1:
                with ThreadPoolExecutor(max_workers=min(max_workers, len(paths))) as pool:
                    futures = [pool.submit(provision, path) for path in paths]
                    # Drain the entire batch even if one clone fails, so all
                    # workers stop touching targets before cleanup begins.
                    errors = []
                    for future in futures:
                        try:
                            future.result()
                        except BaseException as exc:
                            errors.append(exc)
                    if errors:
                        raise errors[0]
            else:
                for path in paths:
                    provision(path)

            for path in paths:
                children.append(MicroVMRuntime(
                    manifest["kernel_path"], manifest["initramfs_path"], path,
                    kernel_sha256=manifest["kernel_sha256"],
                    initramfs_sha256=manifest["initramfs_sha256"],
                    readonly_disk_paths=tuple(manifest["readonly_disk_paths"]),
                    memory_mib=manifest["memory_mib"], vcpus=manifest["vcpus"],
                    backend=manifest["backend"],
                    kernel_append=manifest["kernel_append"],
                    qemu_binary=manifest["qemu_binary"],
                    qemu_img_binary=manifest["qemu_img_binary"],
                    command_timeout=manifest["command_timeout"],
                    popen_factory=popen_factory))
            def start_and_restore(child):
                child.start(paused=True)
                child.load_snapshot(manifest["tag"], resume=True)

            with ThreadPoolExecutor(max_workers=min(max_workers, len(children))) as pool:
                futures = [pool.submit(start_and_restore, child) for child in children]
                for future in futures:
                    future.result()
            return children
        except BaseException:
            for child in children:
                child.close()
            for path, identity in created:
                _unlink_if_owned(path, identity)
            raise
