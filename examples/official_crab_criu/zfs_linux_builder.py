"""Offline ZFS guest assembler; run only inside a bounded Linux utility container.

Guest x86 binaries are never executed. This builds a new experiment cohort,
preserving existing GNU tar, case-sensitive plugins, and original Crab source.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import stat
import struct
import subprocess
import tarfile


RELEASE = '6.18.53-0-virt'
ORIGINAL_CRAB_PYTHON_SHA256 = 'c6d0439e627c75ecc9aea47943212ece93a900605cbac924e99d823ee44b657b'
ORIGINAL_INTEGRATIONS_PYTHON_SHA256 = 'fe0edc09ea062f3495807d3da0f0d5a78e5855ec3f56c426f90d092c39bf897d'
ORIGINAL_GNU_TAR_SHA256 = '2d3e170780a649c3a4cd8dd3e86960644b8a66ee7a76d39d9ecabe157bf98617'
ORIGINAL_PLUGIN_MANIFEST_SHA256 = '21c778898bd3e0fe13296a9b4793e093e6151b531f953716e9eabe90d5269455'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for data in iter(lambda: source.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def run(argv, timeout=300):
    result = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, timeout=timeout)
    if result.returncode:
        raise RuntimeError('offline_utility_failed:' + repr(argv) + ':' +
                           result.stderr[-2000:].decode(errors='replace'))
    return result.stdout.decode(errors='replace')


def safe_name(value):
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or '\\' in value or '\0' in value:
        raise ValueError('unsafe_archive_path')
    return str(path)


def safe_parent(root, name):
    current = root
    for part in PurePosixPath(name).parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise ValueError('archive_parent_symlink:' + name)
        current.mkdir(exist_ok=True)
        if not current.is_dir():
            raise ValueError('archive_parent_not_directory')


def extract_apk(path, root):
    with tarfile.open(path, ignore_zeros=True) as archive:
        payload = [m for m in archive.getmembers() if not m.name.startswith('.')]
        names = [safe_name(m.name) for m in payload]
        if len(names) != len(set(names)):
            raise ValueError('duplicate_payload_path')
        for member, name in zip(payload, names):
            if not (member.isfile() or member.isdir() or member.issym()):
                raise ValueError('unsupported_apk_payload_type')
            if member.issym():
                target = member.linkname
                # Absolute guest links are preserved, but created only after
                # all files. No archive writes ever follow those links.
                if '\0' in target or '\\' in target:
                    raise ValueError('invalid_guest_link')
        for member, name in zip(payload, names):
            if member.issym():
                continue
            safe_parent(root, name)
            destination = root / name
            if destination.is_symlink():
                raise ValueError('payload_overwrites_base_symlink:' + name)
            if member.isdir():
                destination.mkdir(exist_ok=True)
            else:
                if destination.exists() and not destination.is_file():
                    raise ValueError('payload_overwrites_nonregular_base_path')
                with destination.open('wb') as sink:
                    shutil.copyfileobj(archive.extractfile(member), sink)
            destination.chmod(member.mode & 0o777)
            os.utime(destination, (member.mtime, member.mtime))
        for member, name in zip(payload, names):
            if not member.issym():
                continue
            safe_parent(root, name)
            destination = root / name
            if destination.is_symlink() or destination.is_file():
                destination.unlink()
            elif destination.exists():
                raise ValueError('guest_link_overwrites_directory')
            destination.symlink_to(member.linkname)
    return len(payload)


def module_info(path):
    data = gzip.decompress(path.read_bytes()) if path.name.endswith('.gz') else path.read_bytes()
    if data[:6] != b'\x7fELF\x02\x01' or struct.unpack_from('<H', data, 18)[0] != 62:
        raise ValueError('x86_64_little_endian_module_required')
    offset = struct.unpack_from('<Q', data, 40)[0]
    size, count, index = struct.unpack_from('<HHH', data, 58)
    if size != 64 or count > 65535 or offset + size * count > len(data):
        raise ValueError('bounded_elf_module_sections_required')
    headers = [struct.unpack_from('<IIQQQQIIQQ', data, offset + i * size) for i in range(count)]
    table = headers[index]
    strings = data[table[4]:table[4] + table[5]]
    sections = [h for h in headers if strings[h[0]:].split(b'\0', 1)[0] == b'.modinfo']
    if len(sections) != 1:
        raise ValueError('one_module_modinfo_required')
    section = sections[0]
    records = [item.decode() for item in data[section[4]:section[4] + section[5]].split(b'\0') if item]
    result = {k: next((s.split('=', 1)[1] for s in records if s.startswith(k + '=')), None)
              for k in ('name', 'depends', 'vermagic', 'version')}
    if not result['vermagic'].startswith(RELEASE + ' '):
        raise ValueError('actual_module_vermagic_mismatch')
    return {**result, 'compressed_sha256': sha(path),
            'uncompressed_sha256': hashlib.sha256(data).hexdigest(),
            'uncompressed_bytes': len(data)}


def read_cpio(path):
    data, offset, entries = gzip.decompress(path.read_bytes()), 0, []
    while offset + 110 <= len(data):
        header = data[offset:offset + 110]
        if header[:6] != b'070701':
            raise ValueError('original_newc_cpio_required')
        fields = [int(header[i:i + 8], 16) for i in range(6, 110, 8)]
        size, count = fields[6], fields[11]
        name = data[offset + 110:offset + 110 + count - 1].decode()
        safe_name(name)
        start = (offset + 110 + count + 3) // 4 * 4
        if start + size > len(data):
            raise ValueError('bounded_cpio_entry_required')
        offset = (start + size + 3) // 4 * 4
        if name == 'TRAILER!!!':
            return entries
        entries.append((name, fields, data[start:start + size]))
    raise ValueError('complete_cpio_trailer_required')


def write_cpio(entries, path):
    output = bytearray()
    for inode, (name, metadata, payload) in enumerate(entries + [('TRAILER!!!', [0] * 13, b'')], 1):
        fields = list(metadata)
        fields[0], fields[6], fields[11], fields[12] = inode, len(payload), len(name.encode()) + 1, 0
        output += b'070701' + b''.join(f'{value:08x}'.encode() for value in fields)
        output += name.encode() + b'\0'
        output += b'\0' * (-len(output) % 4)
        output += payload
        output += b'\0' * (-len(output) % 4)
    output += b'\0' * (-len(output) % 512)
    path.write_bytes(gzip.compress(output, compresslevel=6, mtime=0))


def tree_evidence(root):
    entries = []
    for path in sorted(root.rglob('*')):
        name = path.relative_to(root).as_posix()
        if path.is_symlink():
            entries.append({'path': name, 'kind': 'symlink', 'target': os.readlink(path)})
        elif path.is_file():
            entries.append({'path': name, 'kind': 'file', 'bytes': path.stat().st_size, 'sha256': sha(path)})
    digest = hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return {'entries': entries, 'entry_count': len(entries), 'tree_sha256': digest}


def fixed_original_python_trees(root):
    evidence = {}
    for tree, expected in [('crab', ORIGINAL_CRAB_PYTHON_SHA256),
                           ('integrations', ORIGINAL_INTEGRATIONS_PYTHON_SHA256)]:
        files = sorted((root / tree).rglob('*.py'))
        digest = hashlib.sha256()
        for path in files:
            if path.is_symlink() or not path.is_file():
                raise ValueError('original_Python_source_must_be_regular')
            digest.update(path.relative_to(root).as_posix().encode() + b'\0' + path.read_bytes() + b'\0')
        if len(files) != 52 or digest.hexdigest() != expected:
            raise ValueError('fixed_original_Python_tree_mismatch:' + tree)
        evidence[tree] = {'python_files': len(files), 'sha256': digest.hexdigest()}
    return evidence


def build(base, assets, inputs, output, plugin_builder):
    if platform.system() != 'Linux' or platform.machine() not in ('aarch64', 'arm64'):
        raise ValueError('native_arm_linux_utility_only')
    base, assets, inputs, output = map(Path, (base, assets, inputs, output))
    manifest = json.loads((assets / 'manifest.json').read_text())
    input_checker_path = Path(__file__).with_name('fetch_zfs_inputs.py')
    input_spec = importlib.util.spec_from_file_location('fixed_ZFS_input_checker', input_checker_path)
    input_checker = importlib.util.module_from_spec(input_spec)
    input_spec.loader.exec_module(input_checker)
    records = input_checker.validate_inputs(inputs)
    for record in records['packages']:
        path = inputs / record['filename']
        if path.is_symlink() or path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
            raise ValueError('downloaded_package_pin_mismatch')
    stage = Path('/linux-stage/root')
    stage.mkdir(parents=True)
    debugfs = run(['debugfs', '-R', 'rdump / ' + str(stage), str(base)])
    (output / 'debugfs-rdump.txt').write_text(debugfs)
    spec = importlib.util.spec_from_file_location('reviewed_plugin_builder', plugin_builder)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    if manifest['xtables_plugins']['manifest_sha256'] != ORIGINAL_PLUGIN_MANIFEST_SHA256:
        raise ValueError('fixed_original_plugin_manifest_required')
    plugin_manifest = helper.load_manifest(assets / 'xtables-manifest.json',
        expected_sha256=ORIGINAL_PLUGIN_MANIFEST_SHA256)[0]
    plugins_before = helper.verify_plugins(stage, plugin_manifest)
    tar_before = sha(stage / 'bin/tar')
    if tar_before != ORIGINAL_GNU_TAR_SHA256 or tar_before != manifest['gnu_tar']['binary_sha256']:
        raise ValueError('base_gnu_tar_pin_mismatch')
    crab_before = tree_evidence(stage / 'opt/fpb/crab')
    fixed_sources_before = fixed_original_python_trees(stage / 'opt/fpb/crab')
    counts = {}
    for record in records['packages']:
        counts[record['filename']] = extract_apk(inputs / record['filename'], stage)
    plugins_after = helper.verify_plugins(stage, plugin_manifest)
    if sha(stage / 'bin/tar') != tar_before or tree_evidence(stage / 'opt/fpb/crab') != crab_before:
        raise ValueError('pinned_base_program_or_original_crab_changed')
    fixed_sources_after = fixed_original_python_trees(stage / 'opt/fpb/crab')
    module_root = stage / 'lib/modules' / RELEASE
    releases = sorted(p.name for p in (stage / 'lib/modules').iterdir() if p.is_dir())
    if releases != [RELEASE]:
        raise ValueError('one_matching_module_release_required:' + repr(releases))
    config = (stage / 'boot' / ('config-' + RELEASE)).read_text()
    config_features = [line for line in config.splitlines() if any(name in line for name in
        ('USERFAULTFD', 'CGROUP_FREEZER', 'CHECKPOINT_RESTORE', 'FHANDLE', 'PROC_PAGE_MONITOR'))]
    spl, zfs = module_info(module_root / 'extra/spl.ko.gz'), module_info(module_root / 'extra/zfs.ko.gz')
    if spl['depends'] != '' or zfs['depends'] != 'spl':
        raise ValueError('reviewed_zfs_actual_module_dependency_changed')
    original_dep_sha = sha(module_root / 'modules.dep')
    dependency = module_root / 'modules.dep'
    dependency.write_bytes(dependency.read_bytes().rstrip() +
        b'\nextra/spl.ko.gz:\nextra/zfs.ko.gz: extra/spl.ko.gz\n')
    # Genuine kernel package metadata lacks external ZFS modules. Use a
    # minimal explicit addition derived from their actual .modinfo, and
    # retain derived text as evidence; the guest uses genuine kmod depmod
    # to regenerate its binary index before requesting any modules.
    (module_root / 'modules.dep.bin').unlink()
    original = read_cpio(assets / 'initramfs-virt')
    nonmodules = [entry for entry in original if entry[0] != 'usr/lib/modules'
                  and not entry[0].startswith('usr/lib/modules/')]
    nonmodules_sha = hashlib.sha256(b''.join(name.encode() + b'\0' + bytes(str(meta), 'ascii') +
        b'\0' + data + b'\0' for name, meta, data in nonmodules)).hexdigest()
    entries = list(nonmodules)
    for path in [stage / 'lib/modules', module_root] + sorted(module_root.rglob('*')):
        name = 'usr/' + path.relative_to(stage).as_posix()
        info = path.lstat()
        payload = os.readlink(path).encode() if path.is_symlink() else path.read_bytes() if path.is_file() else b''
        metadata = [0, info.st_mode, 0, 0, 1, 0, len(payload), 0, 0, 0, 0, 0, 0]
        entries.append((name, metadata, payload))
    target_initrd = output / 'initramfs-virt'
    write_cpio(entries, target_initrd)
    decoded = read_cpio(target_initrd)
    preserved = [entry for entry in decoded if entry[0] != 'usr/lib/modules'
                 and not entry[0].startswith('usr/lib/modules/')]
    if [(name, meta[1:6], data) for name, meta, data in preserved] != \
            [(name, meta[1:6], data) for name, meta, data in nonmodules]:
        raise ValueError('original_nonmodule_initramfs_contents_changed')
    if any('6.18.52' in name for name, _, _ in decoded):
        raise ValueError('old_kernel_modules_in_new_initramfs')
    for filename in ('vmlinuz-virt', 'config-' + RELEASE):
        shutil.copyfile(stage / 'boot' / filename, output / filename)
    disk = output / 'rootfs.raw'
    with disk.open('xb') as stream:
        stream.truncate(base.stat().st_size)
    mkfs = run(['mke2fs', '-F', '-t', 'ext4', '-d', str(stage), str(disk)])
    (output / 'mke2fs.txt').write_text(mkfs)
    report = {'schema_version': 'official-crab-zfs-guest-assets-v1',
        'architecture': 'linux/amd64', 'kernel_release': RELEASE, 'kernel_config_features': config_features,
        'base_manifest_sha256': sha(assets / 'manifest.json'), 'base_rootfs_qcow2_sha256': manifest['rootfs_qcow2_sha256'],
        'base_converted_raw_sha256': sha(base), 'fixed_inputs_sha256': sha(inputs / 'inputs.json'),
        'builder_sha256': sha(Path(__file__)), 'plugin_helper_sha256': sha(plugin_builder),
        'fixed_input_checker_sha256': sha(input_checker_path),
        'fixed_input_plan_sha256': input_checker.FIXED_INPUT_SHA256,
        'exact_cohort_control_payload_revalidated_inside_Linux_before_extraction': True,
        'whole_package_identities': records['packages'], 'package_payload_member_counts': counts,
        'gnu_tar_sha256_unchanged': tar_before, 'all_121_case_sensitive_plugins_before': plugins_before,
        'all_121_case_sensitive_plugins_after': plugins_after, 'whole_original_crab_tree_before_and_after': crab_before,
        'fixed_original_52_plus_52_trees_before': fixed_sources_before,
        'fixed_original_52_plus_52_trees_after': fixed_sources_after,
        'matching_modules': tree_evidence(module_root), 'zfs_actual_elf_modinfo': {'spl': spl, 'zfs': zfs},
        'module_dependency_metadata': {'original_kernel_modules_dep_sha256': original_dep_sha,
             'derived_modules_dep_sha256': sha(dependency), 'binary_dep_index_removed': True,
             'derivation': 'Only append spl/zfs edges verified from unchanged actual ELF .modinfo. No depmod execution claimed.'},
        'initramfs': {'original_sha256': sha(assets / 'initramfs-virt'), 'sha256': sha(target_initrd),
             'entries': len(decoded), 'nonmodule_entries': len(nonmodules), 'nonmodule_evidence_sha256': nonmodules_sha,
             'nonmodule_bytes_and_modes_preserved': True, 'all_original_6_18_52_modules_removed': True,
             'exact_full_matching_kernel_and_zfs_modules_included': True,
             'initrd_module_names': [name for name, _, _ in decoded if '/modules/' in name]},
        'kernel_sha256': sha(output / 'vmlinuz-virt'), 'kernel_config_sha256': sha(output / ('config-' + RELEASE)),
        'rootfs_raw_sha256': sha(disk), 'rootfs_bytes': disk.stat().st_size,
        'guest_executed': False, 'qemu_started': False, 'zfs_pool_created': False,
        'checkpoint_executed': False, 'performance_measured': False,
        'scope': 'Offline asset preparation only. Actual boot, module load and recovery require separate microVM execution evidence.'}
    (output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    parser.add_argument('--assets', required=True)
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--plugin-builder', required=True)
    args = parser.parse_args()
    result = build(args.base, args.assets, args.inputs, args.output, args.plugin_builder)
    print(json.dumps({k: result[k] for k in ('kernel_release', 'kernel_config_features',
                                            'rootfs_bytes', 'guest_executed')}, indent=2))
