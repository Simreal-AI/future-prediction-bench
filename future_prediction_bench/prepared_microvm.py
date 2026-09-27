"""Opt-in, task-bound clean VM templates for repeated RealWorldEnv episodes.

Only the pinned public Boltons or Humanize fixtures are supported. A trusted host prepares a
VM before any policy action, seals its full CPU/RAM/device/ext4 state, and later
spawns independent children. Hidden expected outputs never enter a guest.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path

from . import replace_text as _replace_text
from .coding_env import _workspace_digest
from .http import strict_json_loads
from .microvm_coding import (MicroVMCodingAdapter, MicroVMCodingError,
                             _check_replace_text_helper,
                             _stateless_task_source_digest,
                             replace_text_helper_binding)
from .microvm_runtime import MicroVMRuntime, MicroVMRuntimeError, _sha256_file
from .microvm_template import MicroVMTemplate
from .realworld import validate_task
from .stateless_verifier import BATCH_TARGET, GUEST_PROGRAM, GUEST_TARGET


_SCHEMA = "prepared_boltons_realworld_full_vm_v1"
_HUMANIZE_SCHEMA = "prepared_humanize4150_realworld_full_vm_v1"
_TASK_ID = "boltons-26-singularize-ss-v1"
_TASK_ID_V2 = "boltons-26-singularize-ss-v2"
_HUMANIZE_TASK_ID = "humanize-4150-naturalsize-rounding-v2"
_TASK_TOOLS = {
    _TASK_ID: {"list_files", "read_file", "write_file", "run_visible_checks", "submit"},
    _TASK_ID_V2: {"list_files", "read_file", "write_file", "replace_text",
                  "run_visible_checks", "submit"},
    _HUMANIZE_TASK_ID: {"read_file", "write_file", "replace_text",
                         "run_visible_checks", "submit"},
}
_SHA = re.compile(r"[0-9a-f]{64}\Z")
# This adapter intentionally serves one published, immutable research fixture.
# Recompute these from the pinned upstream sdist and generated fixture when
# updating that fixture; a task ID alone is not an artifact provenance proof.
_SOURCE_SDIST_SHA256 = "5566d6cfd5a1e873d8e8476496287a9f92979964611ad9a9cecb6b0ef29b1edd"
_SEED_TREE_SHA256 = "4463d7c6acfced27ada226f4d26229fc3d78c8eec8fcde7734b8c73fd66a31b6"
_SEED_FILE_SHA256 = "f7f4873406d3913372c9d2b1296cc5e3efb87e88808457e93fc212a5df2de18e"
_VERIFIER_SHA256 = "2fbdc59f999b489b13052102d9b9b31b0e136ec9271242870084fb3d63b5eb58"
_ROOTFS_SHA256 = "98877b015eb530e022e15480d5c2f788a6f4a94587b92e318c3d66b4a2640545"
_KERNEL_SHA256 = "e45e1f6083d1ed45db6647b422e32b6ae6dc54de7b8190b7b97744fb293412e3"
_INITRAMFS_SHA256 = "ffe65ec5a0c0bf470042ad28f7ce7aa5f842ce8090e4230fb2703a7a34e1bebe"
_MODLOOP_SHA256 = "32a189d5e4ae4417cf65bfcc91d2cc8ffdf5957fbbc1f055b037861c595b7328"
_V2_REPLACE_HELPER_BINDING = {
    "source_sha256": "681c6a05f2eb00a5b6b415fc295fdf3a35dc85b7e5e539cd1a968ca39e2ec1b7",
    "guest_program_sha256": "db5d96a53be748bf375005489e0888bb90a7eb9a53b56d87545d023c9b06b30e",
}
_HUMANIZE_PROFILE = {
    "source_path": "src/humanize/filesize.py",
    "asset_schema": "humanize4150-microvm-assets-v1",
    "source_sdist_sha256": "1dd098483eb1c7ee8e32eb2e99ad1910baefa4b75c3aff3a82f4d78688993b10",
    "seed_tree_sha256": "b42ad06044a4bab6906fc95b16c4fa1348a8c82f36cbe490f18b9a929293cfd2",
    "seed_file_sha256": "2e8b51584654471f91ab5234ca08fdc352b0b1d41b8c751de5d68d32cf932bda",
    "verifier_sha256": "4e36079f09ed50c6b9d1c23aed746e9d6ffca22cf62878411717b8329dcb7305",
    "rootfs_sha256": "7d5fe2be77059694b546988a5cbebda0d1fe9dc00ffa00b5fb038f72843bbeba",
    "prepared_schema": _HUMANIZE_SCHEMA,
}
def _profile(task_id):
    if task_id == _HUMANIZE_TASK_ID:
        return _HUMANIZE_PROFILE
    if task_id in (_TASK_ID, _TASK_ID_V2):
        # Preserve the original module-level Boltons pins, which are also
        # deliberately injectable by the fake-runtime unit tests.
        return {
            "source_path": "boltons/strutils.py",
            "asset_schema": "boltons-microvm-assets-v2",
            "source_sdist_sha256": _SOURCE_SDIST_SHA256,
            "seed_tree_sha256": _SEED_TREE_SHA256,
            "seed_file_sha256": _SEED_FILE_SHA256,
            "verifier_sha256": _VERIFIER_SHA256,
            "rootfs_sha256": _ROOTFS_SHA256,
            "prepared_schema": _SCHEMA,
        }
    raise ValueError("Prepared VM supports only the pinned public Boltons or Humanize train fixture")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()


def _regular(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Prepared-template input must be a regular non-symlink file")
    return path.resolve()


def _check_v2_replace_helper(task):
    if task.get("task_id") in (_TASK_ID_V2, _HUMANIZE_TASK_ID):
        bound = task.get("metadata", {}).get("replace_text_helper_binding")
        source = Path(_replace_text.__file__)
        if (bound != _V2_REPLACE_HELPER_BINDING
                or replace_text_helper_binding() != _V2_REPLACE_HELPER_BINDING
                or source.is_symlink()
                or _sha256_file(source) != _V2_REPLACE_HELPER_BINDING["source_sha256"]):
            raise ValueError("Prepared v2 text-edit helper differs from pinned source")


def _clean_task(task, *, allow_stateless=False):
    frozen = validate_task(task)
    task_tools = _TASK_TOOLS.get(frozen["task_id"])
    profile = _profile(frozen["task_id"])
    if (task_tools is None or frozen["is_fixture"] is not True
            or {item["name"] for item in frozen["tool_manifest"]} != task_tools
            or len(frozen["tool_manifest"]) != len(task_tools)
            or frozen["split"] != "train"
            or frozen.get("metadata", {}).get("source_sdist_sha256")
               != profile["source_sdist_sha256"]):
        raise ValueError("Prepared VM supports only the pinned public Boltons train fixture"
                         if frozen["task_id"] != _HUMANIZE_TASK_ID else
                         "Prepared VM supports only the pinned public Humanize train fixture")
    _check_v2_replace_helper(frozen)
    binding = frozen.get("metadata", {}).get("artifact_binding")
    if (not isinstance(binding, dict)
            or binding.get("runtime_kind") != "qemu_hvf_full_vm_qcow2_v1"
            or binding.get("verifier_sha256") != profile["verifier_sha256"]):
        raise ValueError("Prepared VM requires the pinned full-VM artifact binding")
    stateless_fields = {key for key in binding if key.startswith("stateless_")}
    required_stateless = {"stateless_verifier_kind", "stateless_task_source_sha256",
                          "stateless_contract_sha256", "stateless_helper_sha256"}
    if stateless_fields:
        if (not allow_stateless or stateless_fields != required_stateless
                or binding["stateless_verifier_kind"]
                   != "stateless_python_cases_overlay_v1"
                or any(not isinstance(binding[key], str)
                       or not _SHA.fullmatch(binding[key])
                       for key in required_stateless - {"stateless_verifier_kind"})):
            raise ValueError("Prepared VM stateless binding is unsupported or incomplete")
    elif allow_stateless:
        raise ValueError("Prepared VM stateless binding is missing")
    return frozen, binding


def _validate_assets(task, runtime, seed_dir, assets_manifest):
    seed = Path(seed_dir)
    profile = _profile(task["task_id"])
    manifest_path = _regular(assets_manifest)
    manifest = strict_json_loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Invalid pinned asset manifest")
    source = _regular(seed / profile["source_path"])
    if (len(runtime.readonly_disk_paths) != 1
            or manifest.get("schema_version") != profile["asset_schema"]
            or manifest.get("task_id") != task["task_id"]
            or manifest.get("source_sdist_sha256")
               != profile["source_sdist_sha256"]
            or manifest.get("seed_workspace_sha256") != profile["seed_tree_sha256"]
            or _workspace_digest(seed) != profile["seed_tree_sha256"]
            or _sha256_file(source) != profile["seed_file_sha256"]
            or manifest.get("rootfs_qcow2_sha256") != profile["rootfs_sha256"]
            or _sha256_file(runtime.disk_path) != profile["rootfs_sha256"]
            or manifest.get("modloop_disk_sha256") != _MODLOOP_SHA256
            or _sha256_file(runtime.readonly_disk_paths[0]) != _MODLOOP_SHA256
            or runtime.kernel_sha256 != _KERNEL_SHA256
            or runtime.initramfs_sha256 != _INITRAMFS_SHA256
            or manifest.get("alpine_sha256", {}).get("vmlinuz-virt") != _KERNEL_SHA256
            or manifest.get("alpine_sha256", {}).get("initramfs-virt") != _INITRAMFS_SHA256
            or manifest.get("alpine_sha256", {}).get("modloop-virt") != _MODLOOP_SHA256):
        raise ValueError("Prepared VM assets differ from the pinned task")
    return manifest_path, profile["seed_file_sha256"], profile["seed_tree_sha256"]


def _processes(adapter):
    processes = adapter._scan_guest_processes()
    if (len(processes) != 2
            or not any(pid == 1 for pid, _ in processes)
            or set(processes.values()) != {"/usr/bin/busybox"}):
        raise MicroVMCodingError("prepared_vm_unexpected_background_process")
    return processes


def _guest_file_sha(adapter, path):
    output = adapter._guest_ok("sha256sum " + shlex.quote(path))
    match = re.fullmatch(r"([0-9a-f]{64})\s+\S+\n?", output)
    if match is None:
        raise MicroVMCodingError("prepared_vm_guest_digest_invalid")
    return match.group(1)


def _guest_source_digest(adapter, task_id):
    return _guest_file_sha(adapter, adapter.workspace_root + "/" +
                           _profile(task_id)["source_path"])


@dataclass(frozen=True)
class PreparedMicroVMTemplate:
    """Host-only sealed provenance record for a clean full-VM template."""

    manifest_path: Path
    prepared_id: str

    @classmethod
    def prepare(cls, parent_adapter, task, *, seed_dir, assets_manifest,
                template_disk_path, preinstall_stateless_helper=False):
        if type(preinstall_stateless_helper) is not bool:
            raise ValueError("preinstall_stateless_helper must be boolean")
        if (not isinstance(parent_adapter, MicroVMCodingAdapter)
                or type(parent_adapter) is not MicroVMCodingAdapter
                or not isinstance(parent_adapter.runtime, MicroVMRuntime)
                or parent_adapter.started or parent_adapter.submitted):
            raise ValueError("A new standard MicroVMCodingAdapter is required")
        use_stateless = parent_adapter.stateless_verifier is not None
        if preinstall_stateless_helper and not use_stateless:
            raise ValueError("Helper preinstallation requires an explicit stateless contract")
        frozen, binding = _clean_task(task, allow_stateless=use_stateless)
        if binding != parent_adapter.artifact_binding():
            raise ValueError("Prepared VM differs from frozen task artifacts")
        assets_path, seed_file_sha, seed_tree_sha = _validate_assets(
            frozen, parent_adapter.runtime, seed_dir, assets_manifest)
        disk_path = Path(template_disk_path).resolve()
        prepared_path = Path(str(disk_path) + ".prepared.json")
        if prepared_path.exists() or prepared_path.is_symlink():
            raise FileExistsError(prepared_path)
        template = None
        created_prepared = False
        try:
            parent_adapter.reset(frozen, now=None)
            if _guest_source_digest(parent_adapter, frozen["task_id"]) != seed_file_sha:
                raise MicroVMCodingError("prepared_vm_guest_source_differs_from_seed")
            preinstalled_helper_sha = None
            if preinstall_stateless_helper:
                parent_adapter._guest_ok("test ! -e " + shlex.quote(BATCH_TARGET))
                preinstalled_helper_sha = parent_adapter.stateless_verifier.install()
                if (preinstalled_helper_sha != binding["stateless_helper_sha256"]
                        or parent_adapter.stateless_verifier.batch_code_sha256 is not None
                        or _guest_file_sha(parent_adapter, GUEST_TARGET)
                           != preinstalled_helper_sha):
                    raise MicroVMCodingError("prepared_stateless_helper_preinstall_invalid")
                parent_adapter._guest_ok("test ! -e " + shlex.quote(BATCH_TARGET))
            baseline = _processes(parent_adapter)
            if (parent_adapter.metrics["file_reads"] or parent_adapter.metrics["file_writes"]
                    or parent_adapter.metrics["visible_checks"] or parent_adapter.submitted):
                raise MicroVMCodingError("prepared_vm_parent_has_policy_state")
            parent_adapter._guest_ok(
                "test ! -e /tmp/fpb-prepared-parent-after-export && "
                "test ! -e /mnt/root/.fpb-prepared-parent-after-export")
            template = MicroVMTemplate.export(parent_adapter.runtime, disk_path,
                                              tag="prepared")
            # Parent-only writes after sealing prove that later children load
            # the frozen point, rather than a live or subsequently edited VM.
            parent_adapter._guest_ok(
                "printf parent-only > /tmp/fpb-prepared-parent-after-export && "
                "printf parent-only > /mnt/root/.fpb-prepared-parent-after-export")
            payload = {
                "schema": _profile(frozen["task_id"])["prepared_schema"],
                "template_manifest_path": str(template.manifest_path),
                "template_id": template.template_id,
                "task_sha256": frozen["task_sha256"],
                "artifact_binding": copy.deepcopy(binding),
                "verifier_dir": str(parent_adapter.verifier_dir),
                "visible_check": list(parent_adapter.visible_check),
                "workspace_root": parent_adapter.workspace_root,
                "command_timeout": parent_adapter.command_timeout,
                "verifier_mode": ("stateless_namespaced_batch_v1" if use_stateless
                                  else "full_vm_per_case_v1"),
                "stateless_contract_path": (str(parent_adapter.stateless_contract_path.resolve())
                                            if use_stateless else None),
                "stateless_task_path": (str(parent_adapter.stateless_task_path.resolve())
                                        if use_stateless else None),
                "preinstalled_stateless_helper_sha256": preinstalled_helper_sha,
                "block_devices": copy.deepcopy(parent_adapter.block_devices),
                "baseline_processes": [[pid, start, exe]
                                       for (pid, start), exe in sorted(baseline.items())],
                "seed_dir": str(Path(seed_dir).resolve()),
                "seed_tree_sha256": seed_tree_sha,
                "seed_file_sha256": seed_file_sha,
                "assets_manifest_path": str(assets_path),
                "assets_manifest_sha256": _sha256_file(assets_path),
            }
            if frozen["task_id"] == _HUMANIZE_TASK_ID:
                payload["source_path"] = _HUMANIZE_PROFILE["source_path"]
            if frozen["task_id"] in (_TASK_ID_V2, _HUMANIZE_TASK_ID):
                payload["replace_text_helper_binding"] = copy.deepcopy(
                    _V2_REPLACE_HELPER_BINDING)
            prepared_id = _digest(payload)
            with prepared_path.open("x", encoding="utf-8") as stream:
                created_prepared = True
                json.dump(dict(payload, prepared_id=prepared_id), stream, indent=2,
                          sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            prepared_path.chmod(0o444)
            return cls(prepared_path, prepared_id)
        except BaseException:
            if template is not None:
                template.manifest_path.unlink(missing_ok=True)
                template.disk_path.unlink(missing_ok=True)
            if created_prepared:
                prepared_path.unlink(missing_ok=True)
            raise
        finally:
            parent_adapter.close()

    @classmethod
    def open(cls, manifest_path, *, expected_prepared_id):
        path = Path(manifest_path)
        if path.is_symlink() or ".." in path.parts:
            raise ValueError("Prepared manifest must not be a symlink or traversal")
        record = cls(path.resolve(), expected_prepared_id)
        record._verified()
        return record

    def _verified(self):
        path = _regular(self.manifest_path)
        if path.stat().st_size > 32768 or not _SHA.fullmatch(self.prepared_id):
            raise ValueError("Prepared manifest is invalid")
        record = strict_json_loads(path.read_text(encoding="utf-8"))
        if (not isinstance(record, dict) or record.get("schema") not in
                {_SCHEMA, _HUMANIZE_SCHEMA}
                or record.get("prepared_id") != self.prepared_id):
            raise ValueError("Prepared manifest identity differs")
        payload = dict(record)
        payload.pop("prepared_id")
        if _digest(payload) != self.prepared_id:
            raise ValueError("Prepared manifest digest differs")
        for field in ("task_sha256", "seed_tree_sha256", "seed_file_sha256",
                      "assets_manifest_sha256", "template_id"):
            if not isinstance(record.get(field), str) or not _SHA.fullmatch(record[field]):
                raise ValueError("Prepared manifest digest is invalid")
        if (record.get("replace_text_helper_binding") is not None
                and record["replace_text_helper_binding"] != _V2_REPLACE_HELPER_BINDING):
            raise ValueError("Prepared v2 helper manifest differs")
        if ((record["schema"] == _HUMANIZE_SCHEMA
             and record.get("source_path") != _HUMANIZE_PROFILE["source_path"])
                or (record["schema"] == _SCHEMA and "source_path" in record)):
            raise ValueError("Prepared source path differs")
        return record

    def spawn_adapters(self, task, child_disk_paths, *, max_workers=4):
        """Return fresh adapters, each for one independent RealWorldEnv.reset."""
        record = self._verified()
        frozen, binding = _clean_task(
            task, allow_stateless=record.get("verifier_mode") == "stateless_namespaced_batch_v1")
        if record["schema"] != _profile(frozen["task_id"])["prepared_schema"]:
            raise ValueError("Prepared template task profile differs")
        if frozen["task_sha256"] != record["task_sha256"] or binding != record["artifact_binding"]:
            raise ValueError("Prepared template differs from requested task")
        verifier_dir = Path(record["verifier_dir"])
        verifier_file = _regular(verifier_dir / "verify.json")
        assets_path = _regular(record["assets_manifest_path"])
        if (_sha256_file(verifier_file) != binding["verifier_sha256"]
                or _sha256_file(assets_path) != record["assets_manifest_sha256"]
                or _workspace_digest(Path(record["seed_dir"])) != record["seed_tree_sha256"]
                or _sha256_file(Path(record["seed_dir"]) /
                                _profile(frozen["task_id"])["source_path"])
                   != record["seed_file_sha256"]):
            raise ValueError("Prepared template source or verifier changed")
        if record.get("verifier_mode") == "stateless_namespaced_batch_v1":
            contract = _regular(record["stateless_contract_path"])
            source_task = _regular(record["stateless_task_path"])
            if (source_task.parent / "verifier").resolve() != verifier_dir:
                raise ValueError("Prepared stateless task/verifier path differs")
            if (_sha256_file(contract) != binding["stateless_contract_sha256"]
                    or _sha256_file(GUEST_PROGRAM) != binding["stateless_helper_sha256"]
                    or _stateless_task_source_digest(strict_json_loads(
                        source_task.read_text(encoding="utf-8")))
                       != binding["stateless_task_source_sha256"]):
                raise ValueError("Prepared stateless source, helper, or contract changed")
            preinstalled = record.get("preinstalled_stateless_helper_sha256")
            if preinstalled is not None and preinstalled != binding["stateless_helper_sha256"]:
                raise ValueError("Prepared preinstalled helper digest differs")
        elif record.get("verifier_mode") != "full_vm_per_case_v1":
            raise ValueError("Prepared verifier mode is unknown")
        elif record.get("preinstalled_stateless_helper_sha256") is not None:
            raise ValueError("Prepared default verifier cannot preinstall a stateless helper")
        template = MicroVMTemplate.open(record["template_manifest_path"],
                                        expected_template_id=record["template_id"])
        # The template verifies its disk and all immutable boot artifacts; this
        # additionally binds the original pristine seed recorded in the task.
        children = template.spawn(child_disk_paths, max_workers=max_workers)
        adapters = []
        try:
            for runtime in children:
                adapter = _PreparedCodingAdapter(
                    runtime, record=record, verifier_dir=verifier_dir)
                adapters.append(adapter)
            return adapters
        except BaseException:
            for child in children:
                child.close()
            for child in children:
                child.disk_path.unlink(missing_ok=True)
            raise


class _PreparedCodingAdapter(MicroVMCodingAdapter):
    def __init__(self, runtime, *, record, verifier_dir):
        super().__init__(runtime, verifier_dir=verifier_dir,
                         visible_check=record["visible_check"],
                         workspace_root=record["workspace_root"],
                         command_timeout=record["command_timeout"],
                         stateless_verifier_contract=record["stateless_contract_path"],
                         stateless_task_path=record["stateless_task_path"])
        self._prepared_record = copy.deepcopy(record)

    def artifact_binding(self):
        # Child qcow2 files contain the sealed snapshot, so their on-disk hash
        # differs from the pristine seed. All other fields are checked afresh.
        binding = super().artifact_binding()
        expected = self._prepared_record["artifact_binding"]
        binding["disk_seed_sha256"] = expected["disk_seed_sha256"]
        if binding != expected:
            raise ValueError("Prepared child artifact binding differs")
        return binding

    def reset(self, task, *, now):
        if self.started:
            raise ValueError("One prepared VM child serves one episode")
        record = self._prepared_record
        _check_v2_replace_helper(task)
        if (task.get("task_sha256") != record["task_sha256"]
                or task.get("metadata", {}).get("artifact_binding")
                   != record["artifact_binding"]):
            self.runtime.close()
            raise ValueError("Prepared child task differs from template")
        try:
            self.replace_text_enabled = _check_replace_text_helper(task)
            if self.replace_text_enabled and self.workspace_root != self.WORKSPACE_ROOT:
                raise ValueError("Replace text requires the fixed guest workspace root")
            self._check_stateless_episode_task(task)
            self.expected_binding = self.artifact_binding()
            self._guest_ok("test -d " + shlex.quote(self.workspace_root))
            self._guest_ok(
                "test ! -e /tmp/fpb-prepared-parent-after-export && "
                "test ! -e /mnt/root/.fpb-prepared-parent-after-export")
            self._check_no_symlinks()
            if _guest_source_digest(self, task["task_id"]) != record["seed_file_sha256"]:
                raise MicroVMCodingError("prepared_child_source_differs")
            actual_processes = _processes(self)
            if [[pid, start, exe] for (pid, start), exe in sorted(actual_processes.items())] \
                    != record["baseline_processes"]:
                raise MicroVMCodingError("prepared_child_inherited_background_state")
            if self.stateless_verifier is not None:
                self._baseline_processes = actual_processes
                if record["preinstalled_stateless_helper_sha256"] is not None:
                    self._attest_preinstalled_helper()
            self.block_devices = copy.deepcopy(record["block_devices"])
        except (ValueError, OSError, MicroVMRuntimeError, MicroVMCodingError) as exc:
            self.runtime.close()
            raise MicroVMCodingError("prepared_vm_episode_setup_failed") from exc
        self.started = True
        return {"task_id": task["task_id"],
                "tools": [tool["name"] for tool in task["tool_manifest"]],
                "runtime_kind": "qemu_hvf_full_vm_qcow2_v1",
                "workspace_root": "/workspace",
                "visible_check": list(self.visible_check)}

    def _attest_preinstalled_helper(self):
        expected = self._prepared_record["preinstalled_stateless_helper_sha256"]
        if expected is None or self.stateless_verifier is None:
            raise MicroVMCodingError("prepared_stateless_helper_not_declared")
        self._guest_ok("test ! -e " + shlex.quote(BATCH_TARGET))
        if _guest_file_sha(self, GUEST_TARGET) != expected:
            raise MicroVMCodingError("prepared_stateless_helper_changed")
        self.stateless_verifier.installed = True
        self.stateless_verifier.program_sha256 = expected

    def step(self, action, *, now):
        if (isinstance(action, dict) and action.get("action") == "submit"
                and self._prepared_record["preinstalled_stateless_helper_sha256"] is not None):
            self._attest_preinstalled_helper()
        return super().step(action, now=now)
