"""Diagnose real CRIU parent epochs with one explicit external fixture write.

Run only inside the marked disposable x86 guest. This diagnostic deliberately
adds a memory writer between real pre-dump and final-dump calls; it is not the
unmodified workload, a paper performance reproduction, or a training benchmark.
No upstream file, command runner, CRIU binary, or network-lock flag is replaced.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import struct
import sys

# The guest invokes the saved script directly; check_chain and preflight are
# the adjacent, asset-hashed project helpers, not installed upstream modules.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_chain as chain

ORIGINAL_BYTE = 23
CHALLENGE_BYTE = 107
SCHEMA = "official-crab-criu-paired-epoch-diagnostic-v1"


def _pid_epoch(pid):
    stat = Path(f"/proc/{pid}/stat").read_text()
    return stat.rsplit(")", 1)[1].split()[19]


def _check_owned_range(pid, identity, memory_mib):
    address, length, page_size = (identity.get(key) for key in
                                  ("address", "bytes", "page_size"))
    if (any(type(value) is not int for value in (address, length, page_size)) or
            length != memory_mib * 1024 * 1024 or
            not 0 < length <= chain.MAX_PRIVATE_RAM_BYTES or
            page_size != os.sysconf("SC_PAGE_SIZE") or
            length < 3 * page_size or address <= 0 or address % page_size or
            length % page_size or address + length > 2 ** 64):
        raise RuntimeError("bounded_owned_anonymous_range_required")
    for line in Path(f"/proc/{pid}/maps").read_text().splitlines():
        fields = line.split(None, 5)
        start, end = (int(value, 16) for value in fields[0].split("-"))
        name = fields[5] if len(fields) == 6 else ""
        if (start <= address and address + length <= end and fields[1] == "rw-p"
                and fields[4] == "0" and (not name or name.startswith("[anon:"))):
            return address + page_size
    raise RuntimeError("owned_anonymous_range_not_found")


def _read_byte(pid, address):
    descriptor = os.open(f"/proc/{pid}/mem", os.O_RDONLY | os.O_CLOEXEC)
    try:
        value = os.pread(descriptor, 1, address)
    finally:
        os.close(descriptor)
    if len(value) != 1:
        raise RuntimeError("owned_target_short_read")
    return value[0]


def _page_witness(pid, address, page_size):
    with open(f"/proc/{pid}/pagemap", "rb", buffering=0) as stream:
        stream.seek((address // page_size) * 8)
        raw = stream.read(8)
    if len(raw) != 8:
        raise RuntimeError("owned_target_pagemap_short_read")
    entry = struct.unpack("Q", raw)[0]
    return {"pagemap_entry_hex": f"{entry:016x}",
            "present": bool((entry >> 63) & 1),
            "swapped": bool((entry >> 62) & 1),
            "soft_dirty": bool((entry >> 55) & 1)}


def make_runtime_factory(*, witness, memory_mib, inject_write, complete_parent):
    """Create genuine RuncRuntime subclasses after the caller checks source pins."""
    from crab.runtime.runc import RuncRuntime

    class ExternalWriteRuncRuntime(RuncRuntime):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._fixture_observed = False
            self._inside_pre_dump = False
            self._current_pre_dump_id = None
            self._fixture_identity = None

        def pre_dump_process(self, sandbox_id, checkpoint_id, *, parent_checkpoint_id=None):
            if self._inside_pre_dump:
                raise RuntimeError("serialized_pre_dump_fixture_required")
            self._inside_pre_dump = True
            self._current_pre_dump_id = checkpoint_id
            try:
                status = super().pre_dump_process(sandbox_id, checkpoint_id,
                    parent_checkpoint_id=parent_checkpoint_id)
            finally:
                self._inside_pre_dump = False
                self._current_pre_dump_id = None
            witness["pre_dump_operations"].append({
                "checkpoint_id": str(checkpoint_id),
                "requested_parent_id": None if parent_checkpoint_id is None else str(parent_checkpoint_id),
                "executed": status.executed, "reason": status.reason,
                "actual_command": list(status.command)})
            if not status.executed:
                raise RuntimeError("genuine_pre_dump_execution_required")
            if not self._fixture_observed:
                self._fixture_observed = True
                pid = int(chain.runc_state(self.paths.state_root, sandbox_id)["pid"])
                epoch = _pid_epoch(pid)
                identity_path = self.paths.bundle_root / str(sandbox_id) / "rootfs/probe/identity.json"
                identity = json.loads(identity_path.read_text())
                target = _check_owned_range(pid, identity, memory_mib)
                row = {"hook": "after_real_pre_dump_before_matching_final_dump",
                    "checkpoint_id": str(checkpoint_id), "guest_pid": pid,
                    "pid_start_time_ticks": epoch, "owned_identity": dict(identity),
                    "target_page_index": 1, "target_address": target,
                    "normal_counter_pages_untouched": True,
                    "write_requested": inject_write, "write_count": 0,
                    "before_byte": _read_byte(pid, target),
                    "before_page": _page_witness(pid, target, identity["page_size"]),
                    "witness_passed": False}
                witness["injection"] = row
                self._fixture_identity = identity
                if row["before_byte"] != ORIGINAL_BYTE or not row["before_page"]["present"]:
                    raise RuntimeError("original_owned_second_page_required")
                if inject_write:
                    descriptor = os.open(f"/proc/{pid}/mem", os.O_RDWR | os.O_CLOEXEC)
                    try:
                        written = os.pwrite(descriptor, bytes([CHALLENGE_BYTE]), target)
                    finally:
                        os.close(descriptor)
                    row["write_count"] = written
                    if written != 1:
                        raise RuntimeError("owned_fixture_one_byte_write_required")
                row["after_byte"] = _read_byte(pid, target)
                row["after_page"] = _page_witness(pid, target, identity["page_size"])
                if _pid_epoch(pid) != epoch:
                    raise RuntimeError("fixture_pid_identity_changed")
                expected = CHALLENGE_BYTE if inject_write else ORIGINAL_BYTE
                if (row["after_byte"] != expected or not row["after_page"]["present"] or
                        (inject_write and not row["after_page"]["soft_dirty"])):
                    raise RuntimeError("actual_memory_write_and_soft_dirty_witness_required")
                row["witness_passed"] = True
            return status

        def restore_process(self, sandbox_id, checkpoint_id):
            status = super().restore_process(sandbox_id, checkpoint_id)
            observation = {"executed": status.executed, "reason": status.reason,
                "actual_command": list(status.command), "read_only": True}
            witness["restore_observation"] = observation
            if status.executed and self._fixture_identity is not None:
                pid = int(chain.runc_state(self.paths.state_root, sandbox_id)["pid"])
                epoch = _pid_epoch(pid)
                target = _check_owned_range(pid, self._fixture_identity, memory_mib)
                observation.update(guest_pid=pid, pid_start_time_ticks=epoch,
                    target_address=target, restored_byte=_read_byte(pid, target),
                    page=_page_witness(pid, target, self._fixture_identity["page_size"]))
                if _pid_epoch(pid) != epoch:
                    raise RuntimeError("restored_fixture_pid_identity_changed")
            return status

    class CompleteProcessParentRuntime(ExternalWriteRuncRuntime):
        def _resolve_parent_pre_dump_path(self, *, sandbox_id, parent_checkpoint_id):
            if self._inside_pre_dump and parent_checkpoint_id is not None:
                parent = Path(self.process_checkpoint_location(sandbox_id, parent_checkpoint_id) or "")
                if (not parent.is_dir() or not parent.resolve().is_relative_to(
                        self.paths.checkpoint_root.resolve())):
                    raise RuntimeError("previous_complete_process_parent_required")
                witness["parent_overrides"].append({
                    "new_checkpoint_id": str(self._current_pre_dump_id),
                    "requested_parent_id": str(parent_checkpoint_id),
                    "original_parent_kind": "previous_pre_dump",
                    "diagnostic_parent_kind": "previous_complete_process",
                    "actual_parent_directory": str(parent)})
                return parent
            # The current final dump still points to its own current pre-dump.
            return super()._resolve_parent_pre_dump_path(sandbox_id=sandbox_id,
                parent_checkpoint_id=parent_checkpoint_id)

    return CompleteProcessParentRuntime if complete_parent else ExternalWriteRuncRuntime


def classify(name, mode, witness):
    """Keep expected diagnostic outcome distinct from successful recovery."""
    injection = witness.get("injection", {})
    restored = witness.get("restore_observation", {})
    witness_ok = injection.get("witness_passed") is True
    actual_write = injection.get("write_count") == 1 and injection.get("after_byte") == CHALLENGE_BYTE
    if name == "original_zero_write_control":
        expected = (mode.get("passed") is True and witness_ok and
            injection.get("write_count") == 0 and restored.get("restored_byte") == ORIGINAL_BYTE)
        return {"expected_outcome_observed": expected,
                "status": "control_passed" if expected else "control_failed_or_inconclusive"}
    if name == "original_interleaved_write":
        mismatch = (mode.get("passed") is False and
            "owned_private_ram_hash_mismatch" in mode.get("error", "") and
            mode.get("private_ram_before_sha256") != mode.get("private_ram_after_sha256"))
        expected = (witness_ok and actual_write and mismatch and
            restored.get("executed") is True and restored.get("restored_byte") == ORIGINAL_BYTE)
        status = ("parent_chain_loss_reproduced" if expected else
            "hypothesis_not_reproduced" if mode.get("passed") is True else
            "inconclusive_backend_or_witness_failure")
        return {"expected_outcome_observed": expected, "status": status}
    expected = (witness_ok and actual_write and mode.get("passed") is True and
        mode.get("private_ram_restored_exactly") is True and
        mode.get("post_restore_progress_passed") is True and
        restored.get("restored_byte") == CHALLENGE_BYTE and bool(witness["parent_overrides"]))
    return {"expected_outcome_observed": expected,
            "status": "candidate_parent_fix_passed" if expected else "candidate_parent_fix_failed_or_inconclusive"}


def main(args):
    chain.require_guest()
    if type(args.memory_mib) is not int or not 1 <= args.memory_mib <= 64:
        raise ValueError("bounded_diagnostic_memory_mib_required")
    source, root = Path(args.crab_source).resolve(), Path(args.output).resolve()
    if root.exists() or root.is_relative_to(source) or source.is_relative_to(root):
        raise ValueError("new_disjoint_diagnostic_output_required")
    crab_digest = chain.source_digest(source)
    integrations_digest = chain.source_digest(source, tree="integrations")
    if "runtime_factory" not in inspect.signature(chain.execute_mode).parameters:
        raise RuntimeError("explicit_check_chain_runtime_factory_hook_required")
    preflight = chain.collect()
    if (not preflight["process_chain_prerequisites_passed"] or
            not preflight["matches_alpine_binary_pins"]):
        raise RuntimeError("actual_kernel_and_binary_preflight_required")
    # Three paired modes each may retain two full-sized images per boundary.
    # Preserve every ancestor; do not induce an ENOSPC-based negative result.
    required_free = 6 * (len(chain.ACTIONS) + 1) * args.memory_mib * 1024 * 1024 + 64 * 1024 * 1024
    free_bytes = shutil.disk_usage(root.parent).free
    capacity = {"free_bytes": free_bytes, "required_free_bytes": required_free,
        "formula": "3_trials * 2_commands * boundaries * private_RAM_bytes + 64_MiB_reserve",
        "passed": free_bytes >= required_free,
        "scope": "bounded fixture worst-case images; all ancestors and finals retained"}
    preflight["diagnostic_retained_image_capacity"] = capacity
    if not capacity["passed"]:
        return {"schema_version": SCHEMA, "expected_diagnostic_passed": False,
                "preflight": preflight, "error": "insufficient_space_for_all_diagnostic_images"}
    root.mkdir(parents=True)
    sys.path.insert(0, str(source))
    worker = root / "memory-worker"
    chain.run([shutil.which("gcc") or "cc", "-static", "-O2", "-Wall", "-Wextra", "-Werror",
               str(Path(__file__).with_name("memory_worker.c")), "-o", str(worker)])
    report = {"schema_version": SCHEMA, "crab_commit": chain.CRAB_COMMIT,
        "crab_python_sha256": crab_digest, "integrations_python_sha256": integrations_digest,
        "preflight": preflight, "memory_mib": args.memory_mib,
        "worker_sha256": hashlib.sha256(worker.read_bytes()).hexdigest(),
        "diagnostic_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "check_chain_sha256": hashlib.sha256(Path(chain.__file__).read_bytes()).hexdigest(),
        "scope": "external_interleaved_memory_writer_and_runtime_parent_hook_diagnostic",
        "modified_upstream_source_files": [], "paper_performance_reproduction": False,
        "training_speed_measurement": False, "trials": []}
    trials = (("original_zero_write_control", False, False),
              ("original_interleaved_write", True, False),
              ("complete_process_parent_candidate", True, True))
    for name, inject_write, complete_parent in trials:
        witness = {"external_fixture_memory_writer": True,
            "writes_enabled": inject_write, "proposed_parent_change": complete_parent,
            "pre_dump_operations": [], "parent_overrides": []}
        factory = make_runtime_factory(witness=witness, memory_mib=args.memory_mib,
            inject_write=inject_write, complete_parent=complete_parent)
        mode = chain.execute_mode(root / name, "selective_incremental", worker,
            args.memory_mib, runtime_factory=factory)
        row = {"trial": name, "actual_recovery_passed": mode.get("passed") is True,
               "mode_result": mode, "witness": witness, "diagnostic": classify(name, mode, witness)}
        report["trials"].append(row)
        (root / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        if name == "original_zero_write_control" and not row["diagnostic"]["expected_outcome_observed"]:
            report["stopped_after_failed_control"] = True
            break
    report["expected_diagnostic_passed"] = (len(report["trials"]) == 3 and all(
        row["diagnostic"]["expected_outcome_observed"] for row in report["trials"]))
    report["all_actual_recoveries_passed"] = (len(report["trials"]) == 3 and all(
        row["actual_recovery_passed"] for row in report["trials"]))
    (root / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--crab-source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--memory-mib", type=int, choices=range(1, 65), default=8)
    args = parser.parse_args()
    try:
        result = main(args)
    except Exception as exc:
        result = {"schema_version": SCHEMA, "expected_diagnostic_passed": False,
                  "error": str(exc), "execution_success_inferred": False}
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result.get("expected_diagnostic_passed") is True else 2)
