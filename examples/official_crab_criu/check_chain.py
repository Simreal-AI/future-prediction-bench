"""Execute pinned original Crab selector and CRIU process-chain backend.

This is a real runc/CRIU experiment, not a command-capturing runner. It needs
an explicitly marked disposable x86 Linux VM. Filesystem mutation, ZFS,
eBPF collection, model inference, and complete graded episodes are outside
this probe. No package is installed and no host/kernel setting is changed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import struct
import subprocess
import sys
import time

if __package__:
    from .preflight import collect, require_guest
else:
    from preflight import collect, require_guest

CRAB_COMMIT = "9607d61a41dc44358cf078c4b438bfd971c8ee9d"
CRAB_PYTHON_SHA256 = "c6d0439e627c75ecc9aea47943212ece93a900605cbac924e99d823ee44b657b"
INTEGRATIONS_PYTHON_SHA256 = "fe0edc09ea062f3495807d3da0f0d5a78e5855ec3f56c426f90d092c39bf897d"
MODES = ("every_turn_full", "selective_full", "selective_incremental")
MODE_ORDERS = {"forward": MODES, "rotate": MODES[1:] + MODES[:1],
               "rotate2": MODES[2:] + MODES[:2]}
ACTIONS = ("idle", "increment", "idle", "increment_tail", "idle")
NETWORK_LOCK_BACKEND = "iptables_default"
MAX_PRIVATE_RAM_BYTES = 64 * 1024 * 1024
PRIVATE_RAM_HASH_CHUNK_BYTES = 1024 * 1024
X86_64_RT_SIGTIMEDWAIT = "128"


def source_digest(root, *, tree="crab"):
    root = Path(root).resolve()
    digest = hashlib.sha256()
    expected = {"crab": CRAB_PYTHON_SHA256,
                "integrations": INTEGRATIONS_PYTHON_SHA256}[tree]
    files = sorted((root / tree).rglob("*.py"))
    for path in files:
        if path.is_symlink():
            raise ValueError("upstream_source_symlink_rejected")
        digest.update(path.relative_to(root).as_posix().encode() + b"\0" +
                      path.read_bytes() + b"\0")
    if len(files) != 52 or digest.hexdigest() != expected:
        raise ValueError("original_crab_source_pin_mismatch: " + tree)
    return digest.hexdigest()


def run(argv, *, cwd=None, detached=False, failure_log=None):
    kwargs = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL} if detached else {
        "stdout": subprocess.PIPE, "stderr": subprocess.PIPE}
    result = subprocess.run(argv, cwd=cwd, stdin=subprocess.DEVNULL, text=True,
                            timeout=120, check=False, **kwargs)
    if result.returncode:
        log = Path(failure_log).read_text()[-12000:] if failure_log and Path(failure_log).exists() else None
        raise RuntimeError("real_command_failed: " + json.dumps({"argv": argv,
            "returncode": result.returncode, "stdout": result.stdout,
            "stderr": result.stderr, "runtime_log": log}))
    return result.stdout or ""


def runc_state(state_root, sandbox_id):
    return json.loads(run(["runc", "--root", str(state_root), "state", str(sandbox_id)]))


def read_counters(pid, identity):
    with open(f"/proc/{pid}/mem", "rb", buffering=0) as stream:
        stream.seek(identity["address"])
        first = stream.read(8)
        stream.seek(identity["address"] + identity["bytes"] - identity["page_size"])
        last = stream.read(8)
    if len(first) != 8 or len(last) != 8:
        raise RuntimeError("counter_short_read")
    return [struct.unpack("Q", first)[0], struct.unpack("Q", last)[0]]


def wait_counters(pid, identity, expected):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        value = read_counters(pid, identity)
        syscall = Path(f"/proc/{pid}/syscall").read_text().split()
        if value == expected and syscall and syscall[0] == X86_64_RT_SIGTIMEDWAIT:
            return value
        time.sleep(0.005)
    raise RuntimeError("worker_counters_and_quiescent_signal_wait_not_observed")


def wait_identity_file(path):
    """Creation is not readiness: require the bounded complete JSON payload."""
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if path.is_symlink():
            raise RuntimeError("worker_identity_symlink_rejected")
        if path.exists():
            if not path.is_file():
                raise RuntimeError("worker_identity_regular_file_required")
            raw = path.read_bytes()
            if len(raw) > 1024:
                raise RuntimeError("worker_identity_payload_bound_exceeded")
            try:
                identity = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                identity = None
            if (isinstance(identity, dict) and
                    set(identity) == {"address", "bytes", "page_size"} and
                    all(type(value) is int and value > 0 for value in identity.values())):
                return identity
        time.sleep(0.005)
    raise RuntimeError("complete_worker_identity_not_observed")


def hash_owned_private_ram(pid, identity, memory_mib):
    """Read every byte of the worker's bounded anonymous allocation."""
    address, length, page_size = (identity[name] for name in
                                  ("address", "bytes", "page_size"))
    if (any(type(value) is not int for value in (address, length, page_size)) or
            not 0 < length <= MAX_PRIVATE_RAM_BYTES or
            length != memory_mib * 1024 * 1024 or
            page_size != os.sysconf("SC_PAGE_SIZE") or
            address <= 0 or address % page_size or length % page_size or
            address + length > 2 ** 64):
        raise RuntimeError("bounded_private_ram_identity_required")
    # The source is the pinned worker's mmap identity, but additionally
    # require that Linux exposes the entire range as private, writable,
    # anonymous RAM. This deliberately makes no claim about other VMAs.
    anonymous_range_found = False
    for line in Path(f"/proc/{pid}/maps").read_text().splitlines():
        fields = line.split(None, 5)
        start, end = (int(value, 16) for value in fields[0].split("-"))
        name = fields[5] if len(fields) == 6 else ""
        if (start <= address and address + length <= end and
                fields[1] == "rw-p" and fields[4] == "0" and
                (not name or name.startswith("[anon:"))):
            anonymous_range_found = True
            break
    if not anonymous_range_found:
        raise RuntimeError("owned_anonymous_ram_range_not_found")
    digest = hashlib.sha256()
    with open(f"/proc/{pid}/mem", "rb", buffering=0) as stream:
        stream.seek(address)
        remaining = length
        while remaining:
            requested = min(remaining, PRIVATE_RAM_HASH_CHUNK_BYTES)
            chunk = stream.read(requested)
            if len(chunk) != requested:
                raise RuntimeError("private_ram_hash_short_read")
            digest.update(chunk)
            remaining -= requested
    return digest.hexdigest()


def checked_dirty(pid, monitor):
    """Fail closed around upstream's missing-PID/short-pagemap fast paths."""
    os.kill(pid, 0)
    before = Path(f"/proc/{pid}/stat").read_text().split(")", 1)[1].split()[19]
    ranges = monitor.parse_writable_ranges(pid)
    if not ranges or len(ranges) > 1000:
        raise RuntimeError("bounded_writable_ranges_required")
    with open(f"/proc/{pid}/pagemap", "rb", buffering=0) as stream:
        for start, end in ranges:
            pages = (end + monitor.PAGE_SIZE - 1) // monitor.PAGE_SIZE - start // monitor.PAGE_SIZE
            if pages > 65536:
                raise RuntimeError("writable_range_exceeds_probe_bound")
            stream.seek((start // monitor.PAGE_SIZE) * 8)
            if len(stream.read(pages * 8)) != pages * 8:
                raise RuntimeError("pagemap_short_read")
    changed = bool(monitor.dirty_pids({pid}))
    after = Path(f"/proc/{pid}/stat").read_text().split(")", 1)[1].split()[19]
    if before != after:
        raise RuntimeError("pid_identity_changed")
    return changed


def image_bytes(directory):
    return sum(path.stat().st_size for path in directory.rglob("*.img")
               if path.is_file() and not path.is_symlink())


def dirty_mapping_details(pid, monitor):
    """Optional read-only diagnostics; never replace upstream selection."""
    os.kill(pid, 0)
    identity = Path(f"/proc/{pid}/stat").read_text().split(")", 1)[1].split()[19]
    rows = []
    with open(f"/proc/{pid}/pagemap", "rb", buffering=0) as stream:
        for line in Path(f"/proc/{pid}/maps").read_text().splitlines():
            fields = line.split(None, 5)
            if "w" not in fields[1]:
                continue
            start, end = (int(value, 16) for value in fields[0].split("-"))
            pages = (end + monitor.PAGE_SIZE - 1) // monitor.PAGE_SIZE - start // monitor.PAGE_SIZE
            if pages <= 0 or pages > 65536 or len(rows) >= 1000:
                raise RuntimeError("diagnostic_mapping_bound_exceeded")
            stream.seek((start // monitor.PAGE_SIZE) * 8)
            raw = stream.read(pages * 8)
            if len(raw) != pages * 8:
                raise RuntimeError("diagnostic_pagemap_short_read")
            entries = [value[0] for value in struct.iter_unpack("Q", raw)]
            rows.append({"mapping": line, "pages": pages,
                "soft_dirty_pages": sum(bool(value & (1 << 55)) for value in entries),
                "present_soft_dirty_pages": sum(bool(value & (1 << 55)) and
                    bool(value & (1 << 63)) for value in entries),
                "nonpresent_soft_dirty_pages": sum(bool(value & (1 << 55)) and
                    not bool(value & (1 << 63)) for value in entries)})
    if Path(f"/proc/{pid}/stat").read_text().split(")", 1)[1].split()[19] != identity:
        raise RuntimeError("diagnostic_pid_identity_changed")
    return rows


def checkpoint_image_evidence(directory):
    """Retain exact logical-byte inventory and actual relative parent links."""
    files, parents = [], []
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory).as_posix()
        if path.is_symlink():
            if path.name == "parent":
                resolved = path.resolve(strict=True)
                if not resolved.is_dir() or not resolved.is_relative_to(directory.parent):
                    raise RuntimeError("checkpoint_parent_outside_retained_chain")
                parents.append({"path": relative, "target": os.readlink(path)})
        elif path.is_file() and path.suffix == ".img":
            files.append({"path": relative, "bytes": path.stat().st_size})
    return {"regular_image_files": files, "parent_symlinks": parents,
            "logical_regular_image_bytes": sum(row["bytes"] for row in files)}


def prepare_bundle(root, sandbox_id, worker, memory_mib):
    bundle = root / "bundles" / str(sandbox_id)
    rootfs = bundle / "rootfs"
    rootfs.mkdir(parents=True)
    for name in ("bin", "probe", "proc", "dev", "sys", "tmp"):
        (rootfs / name).mkdir()
    shutil.copyfile(worker, rootfs / "bin/memory-worker")
    (rootfs / "bin/memory-worker").chmod(0o755)
    run(["runc", "spec"], cwd=bundle)
    config_path = bundle / "config.json"
    config = json.loads(config_path.read_text())
    config["process"]["terminal"] = False
    config["process"]["args"] = ["/bin/memory-worker", str(memory_mib)]
    config["process"]["cwd"] = "/"
    config["root"] = {"path": "rootfs", "readonly": False}
    config["hostname"] = "fpb-criu-probe"
    # Keep the original CRIU default network lock using real iptables
    # executables. CRIU 4.2's RPC parser crashes if a network-lock option
    # appears in a config file; official 4.2.1 fixes that NULL-argv bug.
    config.setdefault("annotations", {}).pop("org.criu.config", None)
    config["linux"].pop("resources", None)
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    return bundle


def execute_mode(root, mode, worker, memory_mib, *, diagnose_dirty=False,
                 runtime_factory=None):
    # Import the complete unmodified package after whole-package pin checking.
    from crab.config import SchedulerConfig
    from crab.host_inspector import process_monitor as monitor
    from crab.ids import SandboxId, CheckpointId, JobId
    from crab.models import CheckpointJob, SandboxSnapshot, utc_now
    from crab.runtime.runc import (RuncRuntime, RuncRuntimePaths,
                                   RuncCheckpointOptions, RuncRestoreOptions)
    from crab.scheduler import CRScheduler, FaultToleranceCheckpointingPolicy, InMemorySchedulerStateStore
    from crab.workers.process import AdapterProcessCWorker

    sandbox_id = SandboxId("fpb-" + mode.replace("_", "-"))
    runtime = (RuncRuntime if runtime_factory is None else runtime_factory)(
        paths=RuncRuntimePaths(state_root=root / "runc-state",
        bundle_root=root / "bundles", checkpoint_root=root / "checkpoints",
        metadata_root=root / "metadata", zfs_dataset_prefix="unused_process_only_probe"),
        checkpoint_options=RuncCheckpointOptions(extra_args=("--manage-cgroups-mode", "ignore")),
        restore_options=RuncRestoreOptions(extra_args=("--manage-cgroups-mode", "ignore")))
    config = SchedulerConfig(min_checkpoint_interval_seconds=0,
        force_checkpoint_after_seconds=0, prefer_checkpoint_during_llm_request=False,
        incremental_process_enabled=mode == "selective_incremental",
        full_process_checkpoint_interval=8, max_process_chain_length=16)
    scheduler_state = InMemorySchedulerStateStore()
    # evaluate(snapshot) uses these actual kernel observations directly; it
    # does not invoke the optional eBPF inspector/query_checkpoint path.
    scheduler = CRScheduler(config, inspector=None, runtime=runtime,
        state_store=scheduler_state, policy=FaultToleranceCheckpointingPolicy(config))
    checkpoint_worker = AdapterProcessCWorker(runtime)
    bundle = prepare_bundle(root, sandbox_id, worker, memory_mib)
    report = {"mode": mode, "memory_mib": memory_mib, "turns": [],
              "complete_graded_episode": False, "filesystem_checkpoint_executed": False,
              "separate_crab_zfs_filesystem_worker_executed": False,
              "criu_requests_mount_metadata_and_required_tmpfs_contents": True,
              "network_lock_backend": NETWORK_LOCK_BACKEND,
              "network_lock_bypass": False,
              "custom_criu_configuration": False,
              "criu_config_content": None,
              "criu_config_sha256": None,
              "worker_ready_contract": "complete_bounded_identity_and_x86_64_rt_sigtimedwait_128",
              "read_only_dirty_diagnostics_in_wall_timer": diagnose_dirty}
    started = time.perf_counter()
    try:
        launch_log = root / "runc-launch.jsonl"
        run(["runc", "--root", str(runtime.paths.state_root), "--log", str(launch_log),
             "--log-format", "json", "run", "-d", "--bundle", str(bundle), str(sandbox_id)],
            detached=True, failure_log=launch_log)
        pid = int(runc_state(runtime.paths.state_root, sandbox_id)["pid"])
        report["original_guest_pid"] = pid
        identity_path = bundle / "rootfs/probe/identity.json"
        identity = wait_identity_file(identity_path)
        expected = [41, 11]
        wait_counters(pid, identity, expected)
        last_checkpoint = None
        for turn, action in enumerate(("initial",) + ACTIONS):
            turn_start = time.perf_counter()
            if action == "increment":
                expected[0] += 1
                os.kill(pid, signal.SIGUSR1)
            elif action == "increment_tail":
                expected[1] += 1
                os.kill(pid, signal.SIGUSR2)
            wait_counters(pid, identity, expected)
            inspect_start = time.perf_counter()
            actual_changed = checked_dirty(pid, monitor)
            inspection_ms = (time.perf_counter() - inspect_start) * 1000
            snapshot = SandboxSnapshot(sandbox_id=sandbox_id, runtime_name="runc",
                is_running=True, process_changed=actual_changed or mode == "every_turn_full",
                filesystem_changed=False, observed_at=utc_now(), metadata={})
            decision = scheduler.evaluate(snapshot)
            row = {"turn": turn, "action": action, "counters": list(expected),
                "kernel_process_changed": actual_changed,
                "baseline_forces_checkpoint": mode == "every_turn_full",
                "inspection_ms": inspection_ms,
                "selected": decision.should_checkpoint, "reason": decision.reason,
                "incremental": decision.is_incremental_process,
                "produce_pre_dump": decision.produce_pre_dump,
                "requested_filesystem_scope": decision.checkpoint_filesystem,
                "filesystem_scope_executed": False}
            if diagnose_dirty:
                row["dirty_mapping_diagnostics"] = dirty_mapping_details(pid, monitor)
            if decision.should_checkpoint:
                if not decision.checkpoint_process:
                    raise RuntimeError("unexpected_non_process_decision")
                checkpoint_id = CheckpointId("cp-" + str(turn))
                job = CheckpointJob(job_id=JobId.new(), sandbox_id=sandbox_id,
                    requested_at=utc_now(), reason=decision.reason,
                    checkpoint_process=True, checkpoint_filesystem=False,
                    leave_running=True, is_incremental_process=decision.is_incremental_process,
                    parent_process_checkpoint_id=decision.parent_process_checkpoint_id,
                    produce_pre_dump=decision.produce_pre_dump)
                checkpoint_start = time.perf_counter()
                result = checkpoint_worker.checkpoint(job, checkpoint_id)
                row["checkpoint_ms"] = (time.perf_counter() - checkpoint_start) * 1000
                if not result.success or result.operation_status is None or not result.operation_status.executed:
                    raise RuntimeError("original_worker_did_not_execute_checkpoint")
                payload = json.loads(result.artifacts[0].data)
                if decision.produce_pre_dump and not payload["pre_dump_status"]["executed"]:
                    raise RuntimeError("original_worker_did_not_execute_pre_dump")
                row["process_kind"] = payload["process_kind"]
                row["checkpoint_commands"] = [payload["status"]["command"]]
                if decision.produce_pre_dump:
                    row["checkpoint_commands"].insert(0, payload["pre_dump_status"]["command"])
                checkpoint_dir = runtime.paths.checkpoint_root / str(sandbox_id) / str(checkpoint_id)
                row["new_image_bytes"] = image_bytes(checkpoint_dir)
                row["parent_checkpoint_id"] = None if decision.parent_process_checkpoint_id is None else str(decision.parent_process_checkpoint_id)
                scheduler_state.set_last_checkpoint(sandbox_id, utc_now())
                scheduler_state.record_process_checkpoint(sandbox_id, checkpoint_id,
                    is_incremental=decision.is_incremental_process)
                # CRIU owns the tracking epoch for parent-image chains.
                # Clearing its bits externally could discard modifications
                # relative to the parent pre-dump. Full dumps have no such
                # epoch; only those get our post-commit selector reset.
                if not config.incremental_process_enabled:
                    monitor.clear_soft_dirty(pid)
                row["tracking_epoch_owner"] = "CRIU_track_mem" if config.incremental_process_enabled else "selector_after_full_commit"
                last_checkpoint = checkpoint_id
            row["turn_ms"] = (time.perf_counter() - turn_start) * 1000
            report["turns"].append(row)
        if last_checkpoint is None:
            raise RuntimeError("no_restore_point")
        before_damage = list(expected)
        verification_start = time.perf_counter()
        report["expected_counters"] = wait_counters(pid, identity, before_damage)
        report["private_ram_identity"] = dict(identity)
        report["private_ram_bytes_verified"] = identity["bytes"]
        report["private_ram_before_sha256"] = hash_owned_private_ram(pid, identity, memory_mib)
        # The worker is quiescent throughout hashing; observations do not
        # signal it or write to the allocation. Check its counters again.
        wait_counters(pid, identity, before_damage)
        report["private_ram_before_hash_ms"] = (time.perf_counter() - verification_start) * 1000
        os.kill(pid, signal.SIGUSR1)
        verification_start = time.perf_counter()
        report["damaged_counters"] = wait_counters(pid, identity, [expected[0] + 1, expected[1]])
        report["private_ram_damaged_sha256"] = hash_owned_private_ram(pid, identity, memory_mib)
        report["private_ram_damaged_hash_ms"] = (time.perf_counter() - verification_start) * 1000
        if report["private_ram_damaged_sha256"] == report["private_ram_before_sha256"]:
            raise RuntimeError("post_checkpoint_ram_damage_not_verified")
        run(["runc", "--root", str(runtime.paths.state_root), "delete", "--force", str(sandbox_id)])
        restore_start = time.perf_counter()
        status = runtime.restore_process(sandbox_id, last_checkpoint)
        report["restore_ms"] = (time.perf_counter() - restore_start) * 1000
        report["restore_command"] = list(status.command)
        if not status.executed:
            raise RuntimeError("original_runtime_did_not_execute_restore")
        restored_pid = int(runc_state(runtime.paths.state_root, sandbox_id)["pid"])
        report["restored_guest_pid"] = restored_pid
        verification_start = time.perf_counter()
        report["restored_counters"] = wait_counters(restored_pid, identity, before_damage)
        report["private_ram_after_sha256"] = hash_owned_private_ram(restored_pid, identity, memory_mib)
        wait_counters(restored_pid, identity, before_damage)
        report["private_ram_after_hash_ms"] = (time.perf_counter() - verification_start) * 1000
        report["private_ram_verification_scope"] = "entire_owned_anonymous_allocation_including_unmodified_pages"
        report["private_ram_hash_in_checkpoint_or_restore_timer"] = False
        if report["private_ram_before_sha256"] != report["private_ram_after_sha256"]:
            report["private_ram_restored_exactly"] = False
            raise RuntimeError("owned_private_ram_hash_mismatch")
        report["private_ram_restored_exactly"] = True
        # Verify normal computation continues after byte-exact recovery.
        # Keep these challenge mutations separate from the saved-state
        # counters/hash and outside the primitive restore timer.
        progress_start = time.perf_counter()
        os.kill(restored_pid, signal.SIGUSR1)
        wait_counters(restored_pid, identity, [before_damage[0] + 1, before_damage[1]])
        os.kill(restored_pid, signal.SIGUSR2)
        report["post_restore_progress_counters"] = wait_counters(
            restored_pid, identity, [before_damage[0] + 1, before_damage[1] + 1])
        report["post_restore_progress_ms"] = (time.perf_counter() - progress_start) * 1000
        report["post_restore_progress_passed"] = True
        report["checkpoint_count"] = sum(row["selected"] for row in report["turns"])
        report["new_image_bytes"] = sum(row.get("new_image_bytes", 0) for row in report["turns"])
        report["wall_ms"] = (time.perf_counter() - started) * 1000
        report["image_evidence_in_wall_timer"] = False
        for row in report["turns"]:
            if not row["selected"]:
                continue
            directory = runtime.paths.checkpoint_root / str(sandbox_id) / ("cp-" + str(row["turn"]))
            row["retained_image_evidence"] = checkpoint_image_evidence(directory)
            if row["retained_image_evidence"]["logical_regular_image_bytes"] != row["new_image_bytes"]:
                raise RuntimeError("retained_image_inventory_size_changed")
        report["passed"] = True
        return report
    except Exception as exc:
        # Preserve byte-verification evidence and completed turn records
        # even if an actual checkpoint or restore fails.
        report["passed"] = False
        report["error"] = str(exc)
        report["checkpoint_count"] = sum(row["selected"] for row in report["turns"])
        report["new_image_bytes"] = sum(row.get("new_image_bytes", 0) for row in report["turns"])
        report["wall_ms"] = (time.perf_counter() - started) * 1000
        report["image_evidence_in_wall_timer"] = False
        try:
            for row in report["turns"]:
                if row["selected"]:
                    directory = runtime.paths.checkpoint_root / str(sandbox_id) / ("cp-" + str(row["turn"]))
                    row["retained_image_evidence"] = checkpoint_image_evidence(directory)
                    if row["retained_image_evidence"]["logical_regular_image_bytes"] != row["new_image_bytes"]:
                        raise RuntimeError("retained_image_inventory_size_changed")
        except Exception as evidence_error:
            # Keep the original recovery failure authoritative. Failure to
            # retain auxiliary evidence must never turn it into a success.
            report["image_evidence_error"] = str(evidence_error)
        return report
    finally:
        subprocess.run(["runc", "--root", str(runtime.paths.state_root), "delete", "--force", str(sandbox_id)],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=30, check=False)


def main(args):
    require_guest()
    source = Path(args.crab_source).resolve()
    digest = source_digest(source)
    integrations_digest = source_digest(source, tree="integrations")
    preflight = collect()
    if not preflight["process_chain_prerequisites_passed"]:
        return {"schema_version": "official-crab-criu-process-chain-v1", "passed": False,
                "preflight": preflight, "error": "real_kernel_or_binary_prerequisite_failed"}
    if not preflight["matches_alpine_binary_pins"]:
        return {"schema_version": "official-crab-criu-process-chain-v1", "passed": False,
                "preflight": preflight, "error": "reviewed_alpine_binary_pin_mismatch"}
    root = Path(args.output).resolve()
    if root.exists():
        raise ValueError("new_output_directory_required")
    # Keep every ancestor and prior final image. Budget both full modes
    # for every boundary and both incremental commands for full-sized pages.
    # This fixed process has only small mount/metadata images; reserve 64 MiB.
    memory_bytes = args.memory_mib * 1024 * 1024
    required_free = 4 * (len(ACTIONS) + 1) * memory_bytes + 64 * 1024 * 1024
    free_bytes = shutil.disk_usage(root.parent).free
    capacity = {"free_bytes": free_bytes, "required_free_bytes": required_free,
        "formula": "4 * boundaries * private_RAM_bytes + 64_MiB_metadata_reserve",
        "scope": "fixed bounded process fixture; all parent and prior images retained",
        "passed": free_bytes >= required_free}
    preflight["retained_image_capacity"] = capacity
    if not capacity["passed"]:
        return {"schema_version": "official-crab-criu-process-chain-v1", "passed": False,
                "preflight": preflight, "error": "insufficient_space_for_all_retained_images"}
    root.mkdir(parents=True)
    sys.path.insert(0, str(source))
    worker = root / "memory-worker"
    if args.worker_binary:
        shutil.copyfile(args.worker_binary, worker)
        worker.chmod(0o755)
    else:
        run([shutil.which("gcc") or "cc", "-static", "-O2", "-Wall", "-Wextra", "-Werror",
             str(Path(__file__).with_name("memory_worker.c")), "-o", str(worker)])
    report = {"schema_version": "official-crab-criu-process-chain-v1",
        "crab_commit": CRAB_COMMIT, "crab_python_sha256": digest,
        "integrations_python_sha256": integrations_digest,
        "source_python_file_counts": {"crab": 52, "integrations": 52},
        "worker_sha256": hashlib.sha256(worker.read_bytes()).hexdigest(),
        "preflight": preflight, "scope": "private_RAM_process_only_real_runc_CRIU",
        "original_components": ["CRScheduler", "FaultToleranceCheckpointingPolicy",
            "host_inspector.process_monitor", "AdapterProcessCWorker", "RuncRuntime"],
        "gpu_required": False, "full_crab_backend_verified": False,
        "separate_crab_zfs_filesystem_worker_executed": False,
        "criu_requests_mount_metadata_and_required_tmpfs_contents": True,
        "complete_graded_episode": False, "mode_order": args.mode_order,
        "read_only_dirty_diagnostics_enabled": args.diagnose_dirty, "modes": []}
    try:
        for mode in MODE_ORDERS[args.mode_order]:
            report["modes"].append(execute_mode(root / mode, mode, worker, args.memory_mib,
                                               diagnose_dirty=args.diagnose_dirty))
            if not report["modes"][-1]["passed"]:
                report["error"] = report["modes"][-1]["error"]
                break
        report["passed"] = all(row["passed"] for row in report["modes"])
    except Exception as exc:
        report["passed"] = False
        report["error"] = str(exc)
    (root / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--crab-source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--memory-mib", type=int, default=16, choices=range(1, 65))
    parser.add_argument("--worker-binary")
    parser.add_argument("--mode-order", choices=tuple(MODE_ORDERS), default="forward")
    parser.add_argument("--diagnose-dirty", action="store_true")
    args = parser.parse_args()
    try:
        result = main(args)
    except Exception as exc:
        result = {"schema_version": "official-crab-criu-process-chain-v1",
                  "passed": False, "error": str(exc)}
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 2)
