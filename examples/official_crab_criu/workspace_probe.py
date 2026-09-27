"""Bounded genuine ZFS and original Crab process recovery experiment.

Run only in the marked disposable x86 Linux guest with an owned ZFS parent.
The candidate adds nonforced unmount/mount around unchanged original rollback;
full private RAM, workspace bytes and held-FD identity remain strict.
Measured predecessor: c9b554aa92ad230466963b79439e2b3e6ab98b90f08decd29cc836e2ec44d0b3.
Promotion changes documentation, normal package imports and schema metadata;
these source bytes need a separate actual execution record.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys
import time
import traceback
import uuid


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(argv, timeout=120):
    result = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError('real_guest_command_failed:' + json.dumps({
            'argv': argv, 'returncode': result.returncode,
            'stdout': result.stdout[-2000:], 'stderr': result.stderr[-6000:]}))
    return result.stdout


def tracked_command(report, stage, argv, *, check=True, timeout=120):
    row = {'stage': stage, 'argv': list(argv), 'timeout_seconds': timeout}
    report['candidate_commands'].append(row)
    begin = time.perf_counter()
    try:
        result = subprocess.run(argv, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            timeout=timeout)
        row.update(returncode=result.returncode, stdout=result.stdout,
            stderr=result.stderr)
    except Exception as exc:
        row['execution_exception'] = str(exc)
        row['execution_traceback'] = traceback.format_exc()
        if isinstance(exc, subprocess.TimeoutExpired):
            for name in ('stdout', 'stderr'):
                value = getattr(exc, name)
                row[name] = value.decode('utf-8', errors='replace') if isinstance(value, bytes) else value
        raise
    finally:
        row['wall_ms'] = (time.perf_counter() - begin) * 1000
    if check and result.returncode:
        raise RuntimeError('actual_candidate_command_failed:' + json.dumps(row))
    return result


def parse_runc_container_list(raw):
    states = json.loads(raw)
    # Official runc v1.4.3 list.go declares an initially nil container slice
    # (line 118) and directly JSON-encodes it (line 97). Go encoding/json
    # represents that nil slice as literal null when no containers exist.
    if states is None and raw.strip() == 'null':
        return [], True
    if not isinstance(states, list):
        raise RuntimeError('actual_runc_list_json_array_or_exact_null_required')
    for row in states:
        if (not isinstance(row, dict) or not isinstance(row.get('id'), str)
                or not row['id'] or type(row.get('pid')) is not int or row['pid'] < 0
                or row.get('status') not in ('created', 'running', 'paused', 'stopped')):
            raise RuntimeError('actual_runc_container_state_object_required')
    return states, False


def deleted_runtime_evidence(runtime, sandbox_id, known_pids, worker_object, report):
    """Require genuine runc absence and no live process executing our inode."""
    evidence = {'sandbox_id': str(sandbox_id), 'known_pids': [],
        'live_processes_matching_owned_worker_inode': [],
        'worker_object': dict(worker_object)}
    report['runtime_deleted_before_positive_restore'] = evidence
    listed = tracked_command(report, 'verify_owned_runc_container_absent',
        ['runc', '--root', str(runtime.paths.state_root), 'list', '--format', 'json'])
    states, encoded_nil_slice = parse_runc_container_list(listed.stdout)
    evidence['actual_runc_list_stdout'] = listed.stdout
    evidence['runc_empty_nil_slice_encoded_as_exact_json_null'] = encoded_nil_slice
    evidence['null_handling_primary_reference'] = 'https://github.com/opencontainers/runc/blob/v1.4.3/list.go#L97'
    evidence['actual_runc_list'] = states
    evidence['container_absent_from_actual_runc_list'] = not any(
        row.get('id') == str(sandbox_id) for row in states)
    state = tracked_command(report, 'verify_owned_runc_state_absent',
        ['runc', '--root', str(runtime.paths.state_root), 'state', str(sandbox_id)], check=False)
    evidence['actual_runc_state_returncode'] = state.returncode
    evidence['actual_runc_state_stderr'] = state.stderr
    evidence['container_absence_error_confirmed'] = state.returncode != 0 and any(
        phrase in state.stderr for phrase in ('does not exist', 'container not found'))
    for pid in sorted(set(known_pids)):
        process = Path(f'/proc/{pid}')
        row = {'pid': pid, 'proc_entry_exists': process.exists()}
        if row['proc_entry_exists']:
            try:
                row['status'] = (process / 'status').read_text()
                row['state'] = next(line.split()[1] for line in row['status'].splitlines()
                    if line.startswith('State:'))
                if row['state'] in ('Z', 'X', 'x'):
                    row['known_owned_process_not_live_verified'] = True
                else:
                    info = os.stat(process / 'exe')
                    row['current_exe_object'] = {'device': info.st_dev, 'inode': info.st_ino}
                    row['pid_now_has_different_executable_object'] = (
                        (info.st_dev, info.st_ino) != (worker_object['device'], worker_object['inode']))
                    row['known_owned_process_not_live_verified'] = row['pid_now_has_different_executable_object']
            except FileNotFoundError:
                row['proc_entry_exists_after_read'] = process.exists()
                row['known_owned_process_not_live_verified'] = not row['proc_entry_exists_after_read']
        else:
            row['known_owned_process_not_live_verified'] = True
        evidence['known_pids'].append(row)
    errors = []
    for process in Path('/proc').iterdir():
        if not process.name.isdecimal():
            continue
        try:
            info = os.stat(process / 'exe')
        except FileNotFoundError:
            continue
        except PermissionError as exc:
            errors.append({'pid': int(process.name), 'error': str(exc)})
            continue
        if (info.st_dev, info.st_ino) == (worker_object['device'], worker_object['inode']):
            evidence['live_processes_matching_owned_worker_inode'].append({
                'pid': int(process.name), 'exe': os.readlink(process / 'exe'),
                'stat': (process / 'stat').read_text()})
    evidence['process_scan_errors'] = errors
    evidence['no_owned_live_process_verified'] = (
        evidence['container_absent_from_actual_runc_list'] and
        evidence['container_absence_error_confirmed'] and not errors and
        not evidence['live_processes_matching_owned_worker_inode'] and
        all(row.get('known_owned_process_not_live_verified') for row in evidence['known_pids']))
    if not evidence['no_owned_live_process_verified']:
        raise RuntimeError('owned_live_process_absence_not_verified_before_filesystem_lifecycle')
    return evidence


def identity_ready(path):
    keys = {'address', 'bytes', 'page_size', 'held_fd', 'file_inode', 'file_device',
            'namespace_pid', 'initial_file_offset'}
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if path.is_symlink():
            raise RuntimeError('identity_symlink_rejected')
        if path.is_file():
            raw = path.read_bytes()
            if len(raw) > 1024:
                raise RuntimeError('identity_payload_too_large')
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                data = None
            if (isinstance(data, dict) and set(data) == keys and
                    all(type(v) is int and v > 0 for v in data.values())):
                return data
        time.sleep(.005)
    raise RuntimeError('complete_fixture_identity_not_observed')


def workspace_evidence(root):
    entries = []
    for path in sorted(root.rglob('*')):
        name = path.relative_to(root).as_posix()
        info = path.lstat()
        row = {'path': name, 'mode': stat.S_IMODE(info.st_mode)}
        if path.is_symlink():
            row.update(kind='symlink', target=os.readlink(path))
        elif path.is_file():
            row.update(kind='file', bytes=info.st_size, sha256=sha(path))
        elif path.is_dir():
            row.update(kind='directory')
        else:
            raise RuntimeError('owned_workspace_unexpected_file_type')
        entries.append(row)
    encoded = json.dumps(entries, sort_keys=True, separators=(',', ':')).encode()
    return {'entries': entries, 'sha256': hashlib.sha256(encoded).hexdigest(),
            'scope': 'all_owned_workspace_paths_types_permissions_and_bytes; excludes_atime_mtime_device_numbers'}


def ledger_evidence(path):
    raw = path.read_bytes()
    info = path.stat()
    return {'path': str(path), 'bytes': len(raw), 'hex': raw.hex(),
        'sha256': hashlib.sha256(raw).hexdigest(), 'inode': info.st_ino,
        'device': info.st_dev, 'mode': stat.S_IMODE(info.st_mode),
        'mtime_ns': info.st_mtime_ns,
        'scope': 'exact_owned_ledger_bytes_and_actual_file_stat'}


def fd_evidence(pid, identity):
    descriptor = identity['held_fd']
    link = Path(f'/proc/{pid}/fd/{descriptor}')
    info = os.stat(link)
    fields = {}
    for line in Path(f'/proc/{pid}/fdinfo/{descriptor}').read_text().splitlines():
        if ':' in line:
            key, value = line.split(':', 1)
            fields[key] = value.strip()
    ns_line = next(line for line in Path(f'/proc/{pid}/status').read_text().splitlines()
                   if line.startswith('NSpid:'))
    return {'fd': descriptor, 'target': os.readlink(link), 'offset': int(fields['pos']),
            'inode': info.st_ino, 'device': info.st_dev, 'size': info.st_size,
            'namespace_pid': int(ns_line.split()[-1]), 'fdinfo': fields}


def stable_fd_identity(value):
    # CRIU creates a new mount namespace, whose kernel-local mnt_id may
    # legitimately differ. Preserve raw fdinfo as evidence, but compare the
    # actual held file object, descriptor, offset and namespace PID contract.
    return {key: value[key] for key in ('fd', 'target', 'offset', 'inode',
            'device', 'size', 'namespace_pid')}


def original_inputs(crab_source, existing_probe_root):
    sys.path.insert(0, str(existing_probe_root))
    if __package__:
        from . import check_chain
    else:
        import check_chain
    check_chain.require_guest()
    source_hash = check_chain.source_digest(crab_source)
    integration_hash = check_chain.source_digest(crab_source, tree='integrations')
    sys.path.insert(0, str(crab_source))
    return check_chain, source_hash, integration_hash


def serialize_status(value):
    return {'executed': value.executed, 'reason': value.reason,
            'command': list(value.command), 'metadata': value.metadata}


def read_restore_log(path):
    """Read genuine CRIU output without following a substituted log path."""
    evidence = {'source_path': str(path), 'available': False}
    try:
        if any(parent.is_symlink() for parent in path.parents):
            raise RuntimeError('restore_log_parent_symlink_rejected')
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, 'rb') as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise RuntimeError('restore_log_regular_file_required')
            raw = stream.read()
            after = os.fstat(stream.fileno())
        evidence.update(available=True, bytes=len(raw),
            sha256=hashlib.sha256(raw).hexdigest(), inode=before.st_ino,
            device=before.st_dev, mtime_ns=before.st_mtime_ns,
            stable_during_read=(before.st_size, before.st_mtime_ns) ==
                (after.st_size, after.st_mtime_ns) and len(raw) == after.st_size)
        return evidence, raw
    except FileNotFoundError:
        evidence['unavailable_reason'] = 'source_log_absent'
    except Exception as exc:
        evidence['unavailable_reason'] = 'source_log_read_failed'
        evidence['error'] = str(exc)
    return evidence, None


def preserve_restore_attempt(root, name, source_log, attempt):
    """Retain each attempt before CRIU can overwrite its shared work log."""
    destination = root / 'restore-attempts' / name
    destination.mkdir(mode=0o700, parents=True)
    log, raw = read_restore_log(source_log)
    if raw is not None:
        saved = destination / 'restore.log'
        descriptor = os.open(saved, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        log.update(retained_path=str(saved), retained_sha256=sha(saved),
            retained_bytes=saved.stat().st_size,
            tail_utf8=raw[-6000:].decode('utf-8', errors='replace'))
        if log['retained_sha256'] != log['sha256']:
            raise RuntimeError('retained_restore_log_digest_mismatch')
    previous = attempt.get('restore_log_before', {})
    log['same_bytes_as_pre_attempt_log'] = (
        log.get('sha256') == previous.get('sha256')
        if log['available'] and previous.get('available') else None)
    # Equal bytes alone cannot prove whether CRIU rewrote a previous log.
    log['unchanged_bytes_imply_new_log_was_written'] = False
    record = {**attempt, 'restore_log': log}
    record_path = destination / 'attempt.json'
    with record_path.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    return {**record, 'attempt_record_path': str(record_path),
        'attempt_record_sha256': sha(record_path)}


def main(args):
    root = Path(args.output).absolute()
    if (root.exists() or root.is_symlink() or not root.is_relative_to('/tmp')
            or any(p.is_symlink() for p in root.parents)):
        raise ValueError('new_owned_tmp_output_required')
    if not re.fullmatch(r'fpb[A-Za-z0-9_.-]*/[A-Za-z0-9_.-]+', args.zfs_prefix):
        raise ValueError('explicit_owned_fpb_pool_parent_dataset_required')
    worker = Path(args.worker).absolute()
    if worker.is_symlink() or not worker.is_file() or not os.access(worker, os.X_OK):
        raise ValueError('regular_executable_owned_fixture_required')
    binary = worker.read_bytes()
    if binary[:6] != b'\x7fELF\x02\x01' or binary[18:20] != b'\x3e\x00':
        raise ValueError('x86_64_fixture_binary_required')
    chain, crab_sha, integrations_sha = original_inputs(Path(args.crab_source), Path(args.existing_probe_root))
    from crab.config import StorageConfig
    from crab.ids import SandboxId, CheckpointId, JobId
    from crab.models import CheckpointJob, RestoreJob, utc_now
    from crab.runtime.runc import RuncRuntime, RuncRuntimePaths, RuncCheckpointOptions, RuncRestoreOptions
    from crab.storage.local import LocalCheckpointManager
    from crab.workers.composite import DefaultCWorker, DefaultRWorker
    from crab.workers.filesystem import AdapterFileSystemCWorker, AdapterFileSystemRWorker
    from crab.workers.process import AdapterProcessCWorker, AdapterProcessRWorker

    command(['zfs', 'list', '-H', '-o', 'name', args.zfs_prefix])
    root.mkdir(mode=0o700)
    report = {'schema_version': 'crab-zfs-owned-workspace-recovery-v1',
        'memory_mib': args.memory_mib, 'crab_commit': chain.CRAB_COMMIT,
        'original_crab_python_sha256': crab_sha, 'original_integrations_python_sha256': integrations_sha,
        'worker_binary_sha256': sha(worker), 'probe_source_sha256': sha(__file__),
        'measured_predecessor_source_sha256': 'c9b554aa92ad230466963b79439e2b3e6ab98b90f08decd29cc836e2ec44d0b3',
        'filesystem_backend': 'genuine_original_ZFS_snapshot_rollback',
        'upstream_runtime_replaced_or_stubbed': False, 'lazy_pages': False,
        'checkpoint_mode': 'one_full_composite_checkpoint', 'incremental_process': False,
        'bounded_single_process_single_file_writer': True,
        'quiescence_contract': 'observed_rt_sigtimedwait128_and_stable_full_RAM_file_FD_before_after_capture; no external writers during capture',
        'runc_pause_executed': False, 'filesystem_checkpoint_executed': False,
        'graded_repository_episode': False, 'general_concurrent_writer_atomicity_verified': False,
        'model_rollout_or_optimizer_executed': False, 'negative_control': {}, 'passed': False,
        'passed_meaning': 'positive_full_composite_recovery_and_continued_fd_write; negative_control_has_separate_explicit_witness_flag',
        'network_lock_backend': 'iptables_default', 'network_lock_bypass': False,
        'observer_instrumentation': 'filesystem_adapter_subclass_calls_original_super_restore_then_reads_owned_workspace_before_original_process_restore',
        'observer_modifies_restore_operation_or_status': False,
        'composite_restore_ms_includes_observer_reads': True,
        'composite_checkpoint_ms_includes_observer_reads': False,
        'filesystem_recovery_mode': args.filesystem_recovery_mode,
        'original_rollback_control': args.filesystem_recovery_mode == 'original-rollback',
        'genuine_remount_candidate_selected': args.filesystem_recovery_mode == 'unmount-rollback-mount',
        'original_upstream_source_patched': False,
        'global_drop_caches_executed': False, 'corrective_file_rewrites_executed': False,
        'device_inode_fd_contract_relaxed': False,
        'candidate_commands': [], 'snapshot_view': {'observations': {},
            'scope': 'genuine_readonly_clone_of_exact_original_composite_snapshot'}}
    sandbox_id = SandboxId('fpb-workspace-' + uuid.uuid4().hex[:12])
    report['sandbox_id'] = str(sandbox_id)
    dataset = args.zfs_prefix + '/' + str(sandbox_id)
    report['dataset'] = dataset
    clone_dataset = dataset + '-snapshot-view'
    clone_mountpoint = root / 'snapshot-view'
    clone_created = False
    known_pids = []
    worker_object = {}
    runtime = RuncRuntime(paths=RuncRuntimePaths(state_root=root / 'runc-state',
        bundle_root=root / 'bundles', checkpoint_root=root / 'checkpoints',
        metadata_root=root / 'metadata', zfs_dataset_prefix=args.zfs_prefix),
        checkpoint_options=RuncCheckpointOptions(extra_args=('--manage-cgroups-mode', 'ignore')),
        restore_options=RuncRestoreOptions(extra_args=('--manage-cgroups-mode', 'ignore')))
    storage = LocalCheckpointManager(StorageConfig(root_dir=root / 'storage'),
        runtime_image_path_in_use=runtime.runtime_image_path_in_use)
    process_c, process_r = AdapterProcessCWorker(runtime), AdapterProcessRWorker(runtime)

    class ObservedFileSystemRWorker(AdapterFileSystemRWorker):
        def restore(self, job, manifest):
            lifecycle = {'mode': args.filesystem_recovery_mode,
                'original_rollback_adapter_call_unchanged': True,
                'original_worker_step_returned_unchanged': True}
            report['filesystem_lifecycle'] = lifecycle
            rootfs_path = runtime.rootfs_path_for(job.sandbox_id)
            lifecycle['rootfs_before'] = {'path': str(rootfs_path),
                'device': rootfs_path.stat().st_dev, 'inode': rootfs_path.stat().st_ino}
            lifecycle['ledger_before'] = ledger_evidence(rootfs_path / 'workspace/ledger.bin')
            candidate = args.filesystem_recovery_mode == 'unmount-rollback-mount'
            if candidate:
                # No force, lazy detach, global cache clearing or file rewrite.
                # Genuine original process deletion is verified before entry.
                if not report['runtime_deleted_before_positive_restore']['no_owned_live_process_verified']:
                    raise RuntimeError('owned_process_absence_required_before_unmount')
                tracked_command(report, 'candidate_unmount_before_original_rollback',
                    ['zfs', 'unmount', dataset])
                lifecycle['actual_unmount_succeeded'] = True
                lifecycle['mounted_property_after_unmount'] = tracked_command(report,
                    'candidate_verify_dataset_unmounted',
                    ['zfs', 'get', '-H', '-o', 'value', 'mounted', dataset]).stdout.strip()
                if lifecycle['mounted_property_after_unmount'] != 'no':
                    raise RuntimeError('genuine_dataset_unmount_not_verified')
            # Invoke the pinned original adapter unchanged. Mount is a separate
            # candidate lifecycle operation, not a replacement rollback.
            try:
                step = super().restore(job, manifest)
                lifecycle['original_operation'] = serialize_status(step.operation_status)
            except Exception as exc:
                lifecycle['original_adapter_error'] = str(exc)
                lifecycle['original_adapter_traceback'] = traceback.format_exc()
                raise
            finally:
                if candidate and lifecycle.get('actual_unmount_succeeded'):
                    tracked_command(report, 'candidate_mount_after_original_rollback',
                        ['zfs', 'mount', dataset])
                    lifecycle['actual_mount_succeeded'] = True
                    lifecycle['mounted_property_after_mount'] = tracked_command(report,
                        'candidate_verify_dataset_mounted',
                        ['zfs', 'get', '-H', '-o', 'value', 'mounted', dataset]).stdout.strip()
                    if lifecycle['mounted_property_after_mount'] != 'yes':
                        raise RuntimeError('genuine_dataset_mount_not_verified')
            lifecycle['rootfs_after'] = {'path': str(rootfs_path),
                'device': rootfs_path.stat().st_dev, 'inode': rootfs_path.stat().st_ino}
            lifecycle['actual_rootfs_device_changed'] = (
                lifecycle['rootfs_before']['device'] != lifecycle['rootfs_after']['device'])
            observed = {'phase': 'after_original_filesystem_adapter_before_original_process_adapter',
                'original_step_success': step.success,
                'original_operation': serialize_status(step.operation_status),
                'observer_writes_to_owned_workspace': False}
            report['filesystem_restore_observer'] = observed
            try:
                observed_workspace = runtime.rootfs_path_for(job.sandbox_id) / 'workspace'
                observed['workspace'] = workspace_evidence(observed_workspace)
                observed['ledger'] = ledger_evidence(observed_workspace / 'ledger.bin')
                observed['whole_workspace_matches_saved'] = observed['workspace'] == report['saved_workspace']
                observed['ledger_bytes_match_saved'] = observed['ledger']['hex'] == report['saved_ledger']['hex']
            except Exception as exc:
                observed['observer_error'] = str(exc)
                observed['observer_traceback'] = traceback.format_exc()
                raise
            return step

    filesystem_c, filesystem_r = AdapterFileSystemCWorker(runtime), ObservedFileSystemRWorker(runtime)
    composite_c = DefaultCWorker(process_c, filesystem_c, storage, runtime)
    composite_r = DefaultRWorker(process_r, filesystem_r, storage, runtime=runtime)
    wall_start = time.perf_counter()
    launched = False

    def observe_snapshot(label):
        view = {'workspace': workspace_evidence(clone_mountpoint / 'workspace'),
            'ledger': ledger_evidence(clone_mountpoint / 'workspace/ledger.bin')}
        view['whole_workspace_matches_saved'] = view['workspace'] == report['saved_workspace']
        view['ledger_bytes_match_saved'] = view['ledger']['hex'] == report['saved_ledger']['hex']
        report['snapshot_view']['observations'][label] = view

    def destroy_snapshot_clone(stage):
        nonlocal clone_created
        if clone_created:
            tracked_command(report, stage + '_unmount_readonly_clone',
                ['zfs', 'unmount', clone_dataset])
            tracked_command(report, stage + '_destroy_readonly_clone',
                ['zfs', 'destroy', clone_dataset])
            clone_created = False
            report['snapshot_view']['clone_destroyed'] = True

    try:
        bundle = runtime.paths.bundle_root / str(sandbox_id)
        bundle.mkdir(parents=True)
        command(['runc', 'spec', '--bundle', str(bundle)])
        config_path = bundle / 'config.json'
        config = json.loads(config_path.read_text())
        config['process']['terminal'] = False
        config['process']['args'] = ['/bin/worker-workspace', str(args.memory_mib)]
        config['process']['cwd'] = '/'
        config['root'] = {'path': 'rootfs', 'readonly': False}
        config['hostname'] = 'fpb-zfs-probe'
        config['linux'].pop('resources', None)
        config.setdefault('annotations', {}).pop('org.criu.config', None)
        config_path.write_text(json.dumps(config, indent=2) + '\n')
        # Original launch preparation creates and populates the actual root
        # dataset, then normal original runc create/start registers runtime.
        metadata = {'sandbox_id': str(sandbox_id), 'bundle_path': str(bundle),
            'zfs_dataset': dataset, 'rootfs_init_dirs': ['bin', 'probe', 'workspace', 'proc', 'dev', 'sys', 'tmp'],
            'rootfs_copy_paths': [{'source': str(worker), 'destination': 'bin/worker-workspace'}]}
        runtime.launch('runc', metadata)
        launched = True
        pid = int(chain.runc_state(runtime.paths.state_root, sandbox_id)['pid'])
        known_pids.append(pid)
        report['original_host_pid'] = pid
        rootfs = runtime.rootfs_path_for(sandbox_id)
        owned_executable = (rootfs / 'bin/worker-workspace').stat()
        worker_object.update(device=owned_executable.st_dev, inode=owned_executable.st_ino)
        report['owned_worker_file_object'] = dict(worker_object)
        workspace = rootfs / 'workspace'
        identity = identity_ready(rootfs / 'probe/identity.json')
        report['identity'] = identity
        if identity['namespace_pid'] != 1 or identity['bytes'] != args.memory_mib * 1024 * 1024:
            raise RuntimeError('fixture_identity_scope_mismatch')
        chain.wait_counters(pid, identity, [41, 11])
        saved_ram = chain.hash_owned_private_ram(pid, identity, args.memory_mib)
        saved_files = workspace_evidence(workspace)
        saved_fd = fd_evidence(pid, identity)
        saved_ledger = (workspace / 'ledger.bin').read_bytes()
        expected_initial = bytes((13 * i + 7) % 251 for i in range(512))
        if (saved_ledger != expected_initial or saved_fd['offset'] != 64 or
                saved_fd['inode'] != identity['file_inode'] or
                saved_fd['device'] != identity['file_device']):
            raise RuntimeError('genuine_initial_file_and_held_fd_not_verified')
        report['saved_private_ram_sha256'] = saved_ram
        report['saved_workspace'] = saved_files
        report['saved_fd'] = saved_fd
        report['saved_ledger'] = ledger_evidence(workspace / 'ledger.bin')
        checkpoint_id = CheckpointId('full-owned-state')
        job = CheckpointJob(JobId.new(), sandbox_id, utc_now(), leave_running=True,
            checkpoint_process=True, checkpoint_filesystem=True,
            metadata={'checkpoint_id': str(checkpoint_id), 'benchmark_trace_cursor': 0})
        begin = time.perf_counter()
        checkpoint = composite_c.checkpoint(job)
        report['composite_checkpoint_ms'] = (time.perf_counter() - begin) * 1000
        report['checkpoint_result'] = {'status': checkpoint.status.value,
            'failure_code': checkpoint.failure_code.value, 'message': checkpoint.message,
            'operations': [serialize_status(v) for v in checkpoint.operation_statuses]}
        if checkpoint.status.value != 'succeeded' or checkpoint.manifest is None:
            raise RuntimeError('original_full_composite_checkpoint_failed')
        if len(checkpoint.operation_statuses) != 2 or not all(v.executed for v in checkpoint.operation_statuses):
            raise RuntimeError('both_real_process_and_filesystem_operations_required')
        manifest = checkpoint.manifest
        report['checkpoint_manifest'] = manifest.to_dict()
        report['filesystem_checkpoint_executed'] = True
        chain.wait_counters(pid, identity, [41, 11])
        if (chain.hash_owned_private_ram(pid, identity, args.memory_mib) != saved_ram or
                workspace_evidence(workspace) != saved_files or fd_evidence(pid, identity) != saved_fd):
            raise RuntimeError('single_writer_quiescence_changed_across_capture')
        report['capture_quiescence_verified'] = True
        report['snapshot_properties'] = command(['zfs', 'get', '-Hp', '-o', 'property,value',
            'guid,creation,used,written', dataset + '@' + str(checkpoint_id)])
        clone_mountpoint.mkdir(mode=0o700)
        report['snapshot_view'].update(dataset=clone_dataset,
            mountpoint=str(clone_mountpoint), origin_snapshot=dataset + '@' + str(checkpoint_id))
        tracked_command(report, 'create_owned_readonly_snapshot_clone',
            ['zfs', 'clone', '-o', 'readonly=on', '-o', 'atime=off', '-o', 'canmount=noauto',
             '-o', 'mountpoint=' + str(clone_mountpoint),
             dataset + '@' + str(checkpoint_id), clone_dataset])
        clone_created = True
        tracked_command(report, 'mount_owned_readonly_snapshot_clone', ['zfs', 'mount', clone_dataset])
        properties = tracked_command(report, 'verify_owned_clone_properties',
            ['zfs', 'get', '-Hp', '-o', 'property,value', 'origin,readonly,mounted', clone_dataset])
        report['snapshot_view']['actual_properties'] = properties.stdout
        values = dict(line.split('\t', 1) for line in properties.stdout.splitlines())
        if values != {'origin': dataset + '@' + str(checkpoint_id), 'readonly': 'on', 'mounted': 'yes'}:
            raise RuntimeError('genuine_readonly_snapshot_clone_properties_not_verified')
        observe_snapshot('before_live_damage')

        # Actual RAM+file progress first, then multiple persistent workspace
        # changes, then actual process destruction. Same-size ledger avoids
        # trivially relying on a CRIU file-length rejection for the negative.
        os.kill(pid, signal.SIGUSR1)
        chain.wait_counters(pid, identity, [42, 11])
        report['damaged_private_ram_sha256'] = chain.hash_owned_private_ram(pid, identity, args.memory_mib)
        with (workspace / 'ledger.bin').open('r+b') as stream:
            stream.seek(128)
            stream.write(b'\x6b')
            stream.flush()
            os.fsync(stream.fileno())
        (workspace / 'saved.txt').rename(workspace / 'renamed-after-save.txt')
        (workspace / 'renamed-after-save.txt').chmod(0o640)
        (workspace / 'post-checkpoint.txt').write_bytes(b'must disappear after full recovery\n')
        report['damaged_workspace'] = workspace_evidence(workspace)
        report['damaged_fd'] = fd_evidence(pid, identity)
        report['damaged_ledger'] = ledger_evidence(workspace / 'ledger.bin')
        observe_snapshot('after_live_damage')
        if report['damaged_private_ram_sha256'] == saved_ram or report['damaged_workspace'] == saved_files:
            raise RuntimeError('actual_ram_and_workspace_damage_not_verified')
        runtime.prepare_for_restore(sandbox_id)

        negative = report['negative_control']
        negative['scope'] = 'original_process_worker_restore_only_without_ZFS_rollback'
        restore_log = runtime.process_work_path(sandbox_id, checkpoint_id) / 'restore.log'
        try:
            negative_job = RestoreJob(JobId.new(), sandbox_id, checkpoint_id, utc_now())
            negative['restore_log_before'] = read_restore_log(restore_log)[0]
            before = time.perf_counter()
            try:
                step = process_r.restore(negative_job, manifest)
                negative['operation'] = serialize_status(step.operation_status)
            except Exception as exc:
                negative['restore_exception'] = {'message': str(exc),
                    'traceback': traceback.format_exc()}
                raise
            finally:
                negative['restore_ms'] = (time.perf_counter() - before) * 1000
                negative['restore_attempt_evidence'] = preserve_restore_attempt(
                    root, 'negative-process-only', restore_log, dict(negative))
            neg_pid = int(chain.runc_state(runtime.paths.state_root, sandbox_id)['pid'])
            known_pids.append(neg_pid)
            chain.wait_counters(neg_pid, identity, [41, 11])
            negative['restored_private_ram_sha256'] = chain.hash_owned_private_ram(neg_pid, identity, args.memory_mib)
            negative['workspace_after_process_only'] = workspace_evidence(workspace)
            negative['ledger_after_process_only'] = ledger_evidence(workspace / 'ledger.bin')
            negative['ram_recovered'] = negative['restored_private_ram_sha256'] == saved_ram
            negative['workspace_recovered'] = negative['workspace_after_process_only'] == saved_files
            negative['actual_process_only_complete_recovery'] = negative['ram_recovered'] and negative['workspace_recovered']
            negative['expected_failure_witness_verified'] = negative['ram_recovered'] and not negative['workspace_recovered']
            if not negative['expected_failure_witness_verified']:
                raise RuntimeError('process_only_negative_control_not_demonstrated')
        except Exception as exc:
            negative['error'] = str(exc)
            negative.setdefault('expected_failure_witness_verified', False)
            negative.setdefault('actual_process_only_complete_recovery', False)
        finally:
            runtime.prepare_for_restore(sandbox_id)

        observe_snapshot('after_negative_process_restore_and_delete')
        report['snapshot_view']['saved_snapshot_bytes_verified_correct'] = all(
            row['whole_workspace_matches_saved'] and row['ledger_bytes_match_saved']
            for row in report['snapshot_view']['observations'].values())
        destroy_snapshot_clone('before_positive_restore')
        report['snapshot_view']['destroyed_before_positive_restore'] = not clone_created
        deleted_runtime_evidence(runtime, sandbox_id, known_pids, worker_object, report)
        restore_job = RestoreJob(JobId.new(), sandbox_id, checkpoint_id, utc_now())
        positive_attempt = {'scope': 'original_full_composite_ZFS_then_process_restore_with_explicit_filesystem_lifecycle_mode',
            'filesystem_recovery_mode': args.filesystem_recovery_mode,
            'restore_log_before': read_restore_log(restore_log)[0]}
        begin = time.perf_counter()
        try:
            restored = composite_r.restore(restore_job)
            report['restore_result'] = {'status': restored.status.value,
                'failure_code': restored.failure_code.value, 'message': restored.message,
                'operations': [serialize_status(v) for v in restored.operation_statuses]}
            positive_attempt['restore_result'] = report['restore_result']
        except Exception as exc:
            positive_attempt['restore_exception'] = {'message': str(exc),
                'traceback': traceback.format_exc()}
            raise
        finally:
            report['composite_restore_ms'] = (time.perf_counter() - begin) * 1000
            positive_attempt['restore_ms'] = report['composite_restore_ms']
            report['positive_restore_attempt_evidence'] = preserve_restore_attempt(
                root, 'positive-composite', restore_log, positive_attempt)
        report['ledger_after_composite_restore_call'] = ledger_evidence(workspace / 'ledger.bin')
        if restored.status.value != 'succeeded' or len(restored.operation_statuses) != 2:
            raise RuntimeError('original_composite_restore_failed')
        if not all(v.executed for v in restored.operation_statuses):
            raise RuntimeError('real_ZFS_rollback_and_process_restore_required')
        runtime.mark_restored(sandbox_id)
        restored_pid = int(chain.runc_state(runtime.paths.state_root, sandbox_id)['pid'])
        report['restored_host_pid'] = restored_pid
        report['restored_identity_file'] = identity_ready(rootfs / 'probe/identity.json')
        chain.wait_counters(restored_pid, identity, [41, 11])
        report['restored_private_ram_sha256'] = chain.hash_owned_private_ram(restored_pid, identity, args.memory_mib)
        report['restored_workspace'] = workspace_evidence(workspace)
        report['restored_fd'] = fd_evidence(restored_pid, identity)
        report['private_ram_restored_exactly'] = report['restored_private_ram_sha256'] == saved_ram
        report['entire_owned_workspace_restored_exactly'] = report['restored_workspace'] == saved_files
        report['held_fd_object_and_offset_restored_exactly'] = (
            stable_fd_identity(report['restored_fd']) == stable_fd_identity(saved_fd))
        report['identity_file_restored_exactly'] = report['restored_identity_file'] == identity
        report['held_fd_contract_comparison'] = {key: {
            'saved': stable_fd_identity(saved_fd)[key],
            'restored': stable_fd_identity(report['restored_fd'])[key],
            'equal': stable_fd_identity(saved_fd)[key] == stable_fd_identity(report['restored_fd'])[key]}
            for key in stable_fd_identity(saved_fd)}
        report['mount_identity_evidence'] = {
            'saved_file_device': saved_fd['device'],
            'restored_file_device': report['restored_fd']['device'],
            'saved_fdinfo_mnt_id': saved_fd['fdinfo'].get('mnt_id'),
            'restored_fdinfo_mnt_id': report['restored_fd']['fdinfo'].get('mnt_id'),
            'raw_mnt_id_is_recorded_but_kernel_namespace_local': True,
            'file_device_remains_required_by_original_strict_contract': True}
        if saved_fd['device'] != report['restored_fd']['device']:
            report['original_device_contract_failed_after_candidate_lifecycle'] = True
            report['device_contract_failure_rationale'] = 'actual_genuine_remount_changed_st_dev; strict_original_object_identity_contract_was_not_relaxed'
        if not all(report[k] for k in ('private_ram_restored_exactly',
                'entire_owned_workspace_restored_exactly', 'held_fd_object_and_offset_restored_exactly',
                'identity_file_restored_exactly')):
            raise RuntimeError('composed_recovery_witness_mismatch')
        os.kill(restored_pid, signal.SIGUSR1)
        chain.wait_counters(restored_pid, identity, [42, 11])
        continued = saved_ledger[:64] + b'PH000042' + saved_ledger[72:]
        report['continued_fd'] = fd_evidence(restored_pid, identity)
        report['continued_file_bytes_sha256'] = sha(workspace / 'ledger.bin')
        report['continued_write_exactly_once_at_saved_offset'] = (
            (workspace / 'ledger.bin').read_bytes() == continued and
            report['continued_fd']['offset'] == 72 and
            report['continued_fd']['inode'] == saved_fd['inode'])
        os.kill(restored_pid, signal.SIGUSR2)
        report['continued_counters'] = chain.wait_counters(restored_pid, identity, [42, 12])
        report['post_restore_progress_verified'] = report['continued_write_exactly_once_at_saved_offset']
        if not report['post_restore_progress_verified']:
            raise RuntimeError('restored_fd_did_not_continue_correct_write')
        report['positive_composite_recovery_passed'] = True
        report['passed'] = True
    except Exception as exc:
        report['error'] = str(exc)
        report['traceback_tail'] = traceback.format_exc()[-6000:]
        report['passed'] = False
    finally:
        report['probe_wall_ms'] = (time.perf_counter() - wall_start) * 1000
        report['wall_includes_negative_control_and_verification'] = True
        report['verification_in_primitive_checkpoint_restore_timers'] = True
        report['verification_timer_scope'] = 'checkpoint_excludes_evidence_reads; composite_restore_includes_after_filesystem_observer_reads; whole_RAM_FD_and_post_restore_checks_outside_restore_timer'
        report['composite_restore_ms_includes_candidate_unmount_mount'] = args.filesystem_recovery_mode == 'unmount-rollback-mount'
        if clone_created:
            try:
                destroy_snapshot_clone('failure_cleanup')
            except Exception as exc:
                report['snapshot_clone_cleanup_error'] = str(exc)
        try:
            report['retained_process_images'] = chain.checkpoint_image_evidence(runtime.paths.checkpoint_root / str(sandbox_id))
        except Exception as exc:
            report['image_inventory_error'] = str(exc)
        try:
            report['zfs_state_after_probe'] = command(['zfs', 'list', '-Hp', '-o', 'name,used,available,refer,mountpoint', '-t', 'all', '-r', dataset])
        except Exception as exc:
            report['zfs_inventory_error'] = str(exc)
        composite_c.close()
        if launched:
            try:
                runtime.delete_runtime(sandbox_id, force=True, ignore_missing=True)
                report['live_process_cleanup'] = 'actual_original_delete_runtime'
            except Exception as exc:
                report['cleanup_error'] = str(exc)
        # Dataset/snapshot/images remain in the disposable guest for actual
        # evidence collection; the outer driver owns final VM disposal.
        (root / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crab-source', default='/opt/fpb/crab')
    parser.add_argument('--existing-probe-root', default='/opt/fpb/probe')
    parser.add_argument('--worker', required=True)
    parser.add_argument('--zfs-prefix', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--memory-mib', type=int, choices=(8, 64), default=8)
    parser.add_argument('--filesystem-recovery-mode',
        choices=('original-rollback', 'unmount-rollback-mount'), default='original-rollback')
    raise SystemExit(main(parser.parse_args()))
