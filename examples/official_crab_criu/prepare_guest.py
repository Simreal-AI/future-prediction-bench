"""Build a disposable x86 CRIU/runc guest from pinned local archives.

No network, APK installer, guest binary execution, or host-root modification
is performed. A cached immutable ARM image supplies native Python and mke2fs
for case-sensitive Linux staging and offline filesystem assembly.
CRIU comes from Alpine edge/testing while dependencies come from v3.24;
the manifest records that mixed-source feasibility scope, not compatibility.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import posixpath
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
from urllib.parse import urlparse

from examples.official_crab.prepare_x86 import ALPINE_URL, PINS, PYTHON_URL
from examples.realworld_boltons26.prepare_microvm import _run, _sha
from future_prediction_bench.coding_env import _workspace_digest, _workspace_files
from examples.official_crab_criu.linux_rootfs_builder import (
    IPTABLES_APK_SHA256, PLUGIN_PREFIX, load_manifest as load_plugin_manifest,
)

MARKER_TEXT = "Future Prediction Bench disposable CRIU guest v1\n"
CRAB_COMMIT = "9607d61a41dc44358cf078c4b438bfd971c8ee9d"
CRAB_PYTHON_SHA256 = "c6d0439e627c75ecc9aea47943212ece93a900605cbac924e99d823ee44b657b"
INTEGRATIONS_PYTHON_SHA256 = "fe0edc09ea062f3495807d3da0f0d5a78e5855ec3f56c426f90d092c39bf897d"
PYYAML_SHA256 = "d76623373421df22fb4cf8817020cbb7ef15c725b9d5e45f17e189bfc384190f"
PYYAML_URL = "https://files.pythonhosted.org/packages/05/8e/961c0007c59b8dd7729d542c61a4d537767a59645b82a0b521206e1e25c2/pyyaml-6.0.3.tar.gz"
REQUIRED_PACKAGES = {"criu", "runc", "gcc", "musl-dev", "musl", "busybox", "busybox-binsh",
                     "iptables", "libxtables", "tar", "acl-libs"}
GNU_TAR_APK_SHA256 = "5dad2fe0c7d18394dd7ffb64682760ce63718a7a1552b661f6e63036ce8b4958"
GNU_TAR_BINARY_SHA256 = "2d3e170780a649c3a4cd8dd3e86960644b8a66ee7a76d39d9ecabe157bf98617"
ACL_LIBS_APK_SHA256 = "3f02851d586c25d97ba8b5b845615aacc209ae7eaeb6f7843eca76d85da2249d"
ACL_LIBRARY_SHA256 = "b7a6945d1fc95aec631e0e262725ea18306e0b4df1194568c2d8cd4517d8a142"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _archive_name(name):
    """Accept guest-relative POSIX names, never archive path traversal."""
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\0" in name:
        raise ValueError("unsafe_archive_member_name")
    clean = str(path)
    if clean in ("", "."):
        return ""
    return clean


def _payload_member(member, *, python_only=False):
    name = _archive_name(member.name)
    if not name:
        return None
    # APK signatures/control records are not guest payload. Skip terminal
    # databases whose Linux aliases can collide on case-insensitive APFS.
    if name.split("/", 1)[0].startswith("."):
        return None
    if "terminfo" in PurePosixPath(name).parts:
        return None
    if python_only and name != "python" and not name.startswith(
            ("python/bin/", "python/lib/", "python/include/")):
        return None
    if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
        raise ValueError("archive_special_file_not_allowed: " + name)
    result = copy.copy(member)
    result.name = name
    if member.issym():
        target = member.linkname
        if not target or "\0" in target:
            raise ValueError("invalid_archive_symlink")
        if target.startswith("/"):
            # /bin/sh -> /bin/busybox must point inside the guest staging
            # root, never /bin/busybox on the builder host.
            rooted = posixpath.normpath(target.lstrip("/"))
            if rooted in ("", ".") or rooted == ".." or rooted.startswith("../"):
                raise ValueError("unsafe_guest_absolute_symlink")
            result.linkname = posixpath.relpath(rooted, posixpath.dirname(name) or ".")
        destination = posixpath.normpath(posixpath.join(posixpath.dirname(name), result.linkname))
        if destination == ".." or destination.startswith("../") or destination.startswith("/"):
            raise ValueError("escaping_archive_symlink")
    elif member.islnk():
        # A tar hardlink names a member relative to the archive root, unlike
        # a symlink's target relative to its parent directory.
        result.linkname = _archive_name(member.linkname)
        if not result.linkname:
            raise ValueError("invalid_archive_hardlink")
    return result


def _managed_plugin(name):
    return name.startswith(PLUGIN_PREFIX) and len(PurePosixPath(name).parts) == 4


def _register_casefold(name, registry):
    parts = PurePosixPath(name).parts
    # Archives can omit directory entries. Audit implicit parent prefixes too,
    # so usr/Lib/a and usr/lib/b cannot alias silently on APFS.
    for length in range(1, len(parts) + 1):
        prefix = "/".join(parts[:length])
        previous = registry.setdefault(prefix.casefold(), prefix)
        managed_leaf = length == len(parts) and _managed_plugin(previous) and _managed_plugin(prefix)
        if previous != prefix and not managed_leaf:
            raise ValueError("unhandled_casefold_collision: " + previous + " / " + prefix)


def _extract_payload(path, stage, *, python_only=False, casefold_registry=None,
                     exclude_xtables=False):
    registry = {} if casefold_registry is None else casefold_registry
    # APK files may contain concatenated signature/control/payload tar
    # streams; ignore zero blocks so every stream is inspected.
    with tarfile.open(path, mode="r:*", ignore_zeros=True) as archive:
        selected = []
        # Audit the complete selected archive before extracting any member.
        # Every xtables payload is excluded, not just known colliding pairs.
        for member in archive.getmembers():
            filtered = _payload_member(member, python_only=python_only)
            if filtered is None:
                continue
            guest_name = "usr/local" + filtered.name[len("python"):] if python_only else filtered.name
            _register_casefold(guest_name, registry)
            if filtered.name.startswith(PLUGIN_PREFIX) and not filtered.isdir():
                if not exclude_xtables or not _managed_plugin(filtered.name):
                    raise ValueError("unexpected_plugin_payload_outside_managed_archive")
                continue
            selected.append(filtered)
        archive.extractall(stage, members=selected, filter="data")
    return len(selected)


def _stash_xtables(path, output):
    if path.is_symlink() or _sha(path) != IPTABLES_APK_SHA256:
        raise ValueError("reviewed_iptables_archive_required_for_case_preserving_stage")
    objects = output / "xtables-objects"
    objects.mkdir()
    entries = []
    with tarfile.open(path, mode="r:*", ignore_zeros=True) as archive:
        for member in archive.getmembers():
            name = _archive_name(member.name)
            if not name.startswith(PLUGIN_PREFIX) or member.isdir():
                continue
            if not _managed_plugin(name):
                raise ValueError("unexpected_nested_plugin_payload")
            if member.isfile():
                if not 0 < member.size <= 2 * 1024 * 1024:
                    raise ValueError("bounded_plugin_payload_required")
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError("plugin_archive_bytes_unreadable")
                data = stream.read()
                if len(data) != member.size:
                    raise ValueError("plugin_archive_short_read")
                digest = hashlib.sha256(data).hexdigest()
                object_path = objects / (digest + ".bin")
                if object_path.exists():
                    if object_path.is_symlink() or _sha(object_path) != digest:
                        raise ValueError("encoded_plugin_object_conflict")
                else:
                    with object_path.open("xb") as destination:
                        destination.write(data)
                    object_path.chmod(0o644)
                entries.append({"path": name, "kind": "file", "mode": member.mode & 0o777,
                                "bytes": len(data), "sha256": digest, "object": object_path.name})
            elif member.issym():
                entries.append({"path": name, "kind": "symlink", "mode": member.mode & 0o777,
                                "target": member.linkname})
            else:
                raise ValueError("unsupported_original_plugin_type")
    manifest = {"schema_version": "fpb-xtables-case-preserving-v1",
                "source_apk_sha256": IPTABLES_APK_SHA256,
                "entries": sorted(entries, key=lambda entry: entry["path"])}
    manifest_path = output / "xtables-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    digest = _sha(manifest_path)
    _, pairs = load_plugin_manifest(manifest_path, expected_sha256=digest)
    return objects, manifest_path, digest, pairs


def _package_metadata(path):
    with tarfile.open(path, mode="r:*", ignore_zeros=True) as archive:
        records = [member for member in archive.getmembers()
                   if str(PurePosixPath(member.name)) == ".PKGINFO"]
        if len(records) != 1 or not records[0].isfile() or records[0].size > 65536:
            raise ValueError("one_bounded_apk_pkginfo_required")
        stream = archive.extractfile(records[0])
        if stream is None:
            raise ValueError("apk_metadata_unreadable")
        data = stream.read().decode("utf-8", "strict")
    metadata = {}
    for line in data.splitlines():
        if " = " in line and not line.startswith("#"):
            key, value = line.split(" = ", 1)
            if key in {"pkgname", "pkgver", "arch"}:
                if key in metadata:
                    raise ValueError("ambiguous_apk_identity")
                metadata[key] = value
    return metadata


def _extract_pyyaml(path, stage, *, casefold_registry=None):
    """Install the real pure-Python package and upstream license, offline."""
    if path.is_symlink() or _sha(path) != PYYAML_SHA256 or path.stat().st_size != 130960:
        raise ValueError("official_pyyaml_sdist_pin_mismatch")
    prefix = "pyyaml-6.0.3/lib/yaml/"
    package_path = "usr/local/lib/python3.12/site-packages/yaml"
    license_path = "usr/local/share/licenses/PyYAML-6.0.3/LICENSE"
    selected = []
    with tarfile.open(path, mode="r:*", ignore_zeros=True) as archive:
        for member in archive.getmembers():
            name = _archive_name(member.name)
            if name.startswith(prefix) and name.endswith(".py"):
                if not member.isfile():
                    raise ValueError("regular_pyyaml_module_required")
                target = package_path + "/" + name[len(prefix):]
            elif name == "pyyaml-6.0.3/LICENSE":
                if not member.isfile():
                    raise ValueError("regular_pyyaml_license_required")
                target = license_path
            else:
                continue
            rewritten = copy.copy(member)
            rewritten.name = target
            selected.append(_payload_member(rewritten))
        if len(selected) != 18 or not any(member.name.endswith("/yaml/__init__.py")
                                          for member in selected):
            raise ValueError("expected_pyyaml_python_package_and_license_required")
        if casefold_registry is not None:
            for member in selected:
                _register_casefold(member.name, casefold_registry)
        archive.extractall(stage, members=selected, filter="data")
    files = sorted((stage / package_path).rglob("*.py"))
    digest = hashlib.sha256()
    for module in files:
        if module.is_symlink():
            raise ValueError("pyyaml_module_symlink_not_allowed")
        digest.update(module.relative_to(stage / "usr/local/lib/python3.12/site-packages").as_posix().encode()
                      + b"\0" + module.read_bytes() + b"\0")
    if len(files) != 17:
        raise ValueError("pyyaml_copied_module_count_mismatch")
    return {"name": "PyYAML", "version": "6.0.3", "source": PYYAML_URL,
            "sdist_sha256": PYYAML_SHA256, "sdist_bytes": 130960,
            "python_package_sha256": digest.hexdigest(), "python_files": len(files),
            "location": "/" + package_path, "implementation": "unmodified upstream pure Python",
            "compiled_libyaml_extension": False, "license_sha256": _sha(stage / license_path)}


def _packages(package_dir):
    pin_file = package_dir / "pinned-package-inputs.json"
    if pin_file.is_symlink() or not pin_file.is_file():
        raise ValueError("actual_sha256_package_manifest_required")
    manifest = json.loads(pin_file.read_text(encoding="utf-8"))
    if manifest.get("architecture") != "x86_64":
        raise ValueError("x86_64_package_manifest_required")
    packages = manifest.get("packages")
    if not isinstance(packages, list) or not 1 <= len(packages) <= 128:
        raise ValueError("bounded_package_list_required")
    validated, names = [], set()
    for package in packages:
        if not isinstance(package, dict):
            raise ValueError("package_record_required")
        name, version = package.get("name"), package.get("version")
        digest, url = package.get("sha256"), package.get("url")
        if (not isinstance(name, str) or not name or name in names or
                not isinstance(version, str) or not version or
                not isinstance(digest, str) or SHA256.fullmatch(digest) is None or
                not isinstance(url, str) or not url.startswith("https://")):
            raise ValueError("unique_named_actual_sha256_package_required")
        filename = package.get("filename", PurePosixPath(urlparse(url).path).name)
        if (not isinstance(filename, str) or PurePosixPath(filename).name != filename or
                not filename.endswith(".apk") or filename != f"{name}-{version}.apk"):
            raise ValueError("apk_filename_identity_mismatch")
        path = package_dir / filename
        if path.is_symlink() or not path.is_file() or _sha(path) != digest:
            raise ValueError("apk_sha256_mismatch: " + filename)
        if "bytes" in package and (type(package["bytes"]) is not int or
                                   path.stat().st_size != package["bytes"]):
            raise ValueError("apk_size_mismatch: " + filename)
        info = _package_metadata(path)
        if (info.get("pkgname") != name or info.get("pkgver") != version or
                info.get("arch") not in {"x86_64", "noarch"}):
            raise ValueError("apk_metadata_identity_or_architecture_mismatch: " + filename)
        required_tools = {
            "tar": ("1.35-r5", GNU_TAR_APK_SHA256),
            "acl-libs": ("2.3.2-r1", ACL_LIBS_APK_SHA256),
        }
        if name in required_tools and (version, digest) != required_tools[name]:
            raise ValueError("reviewed_gnu_tar_dependency_pin_required: " + name)
        names.add(name)
        validated.append({"name": name, "version": version, "filename": filename,
                          "url": url, "sha256": digest, "bytes": path.stat().st_size,
                          "architecture": info["arch"]})
    if not REQUIRED_PACKAGES <= names:
        raise ValueError("required_criu_runtime_compiler_packages_missing")
    return manifest, sorted(validated, key=lambda item: item["name"]), _sha(pin_file)


def _require_x86_elf(path):
    resolved = path.resolve(strict=True)
    with resolved.open("rb") as stream:
        header = stream.read(20)
    if len(header) != 20 or header[:6] != b"\x7fELF\x02\x01" or header[18:20] != b"\x3e\x00":
        raise ValueError("x86_64_elf_required: " + path.name)


def _require_gnu_tar(stage):
    """Prove the original GNU tool is selected without executing x86 code."""
    stage = Path(stage)
    command = stage / "bin/tar"
    if (command.is_symlink() or not command.is_file() or
            _sha(command) != GNU_TAR_BINARY_SHA256):
        raise ValueError("original_gnu_tar_regular_binary_required")
    for prefix in ("usr/local/bin", "usr/sbin", "usr/bin", "sbin"):
        candidate = stage / prefix / "tar"
        if candidate.exists() or candidate.is_symlink():
            raise ValueError("gnu_tar_guest_path_shadow_rejected")
    _require_x86_elf(command)
    library = stage / "usr/lib/libacl.so.1"
    resolved = library.resolve(strict=True)
    if (not resolved.is_relative_to(stage.resolve()) or
            _sha(resolved) != ACL_LIBRARY_SHA256):
        raise ValueError("original_gnu_tar_acl_dependency_required")
    _require_x86_elf(resolved)
    return {"implementation": "unmodified Alpine GNU tar", "version": "1.35-r5",
            "path": "/bin/tar", "binary_sha256": GNU_TAR_BINARY_SHA256,
            "apk_sha256": GNU_TAR_APK_SHA256, "x86_64_elf_verified": True,
            "acl_library_sha256": ACL_LIBRARY_SHA256,
            "guest_path_shadow_present": False, "binary_executed_by_builder": False}


def _pinned_python_tree(root, subtree, expected_sha256):
    files = sorted((root / subtree).rglob("*.py"))
    digest = hashlib.sha256()
    for path in files:
        if path.is_symlink() or not path.is_file():
            raise ValueError("crab_source_symlink_or_special_file")
        digest.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    if len(files) != 52 or digest.hexdigest() != expected_sha256:
        raise ValueError("whole_original_python_tree_pin_mismatch: " + subtree)
    return files


def _crab_sources(root):
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True)
    if head != CRAB_COMMIT or dirty:
        raise ValueError("clean_pinned_crab_checkout_required")
    files = _pinned_python_tree(root, "crab", CRAB_PYTHON_SHA256)
    # Crab's eager package imports require these upstream modules even for
    # the process-only runtime. Include their exact Python closure only;
    # fixture data, deployment files, and optional assets remain excluded.
    files += _pinned_python_tree(root, "integrations", INTEGRATIONS_PYTHON_SHA256)
    license_path = root / "LICENSE"
    if license_path.is_symlink() or not license_path.is_file():
        raise ValueError("upstream_license_required")
    return files


def prepare(source, packages, task_dir, output, *, build_image, crab_source, disk_mib=512):
    source, package_dir, task_dir, output, crab_source = [Path(path).resolve() for path in
                                           (source, packages, task_dir, output, crab_source)]
    if type(disk_mib) is not int or disk_mib not in {512, 768, 1024, 2048, 3072}:
        raise ValueError("reviewed_disk_mib_required")
    rootfs_bytes = disk_mib * 1024 * 1024
    if output.exists() or any(output.is_relative_to(path) or path.is_relative_to(output)
                             for path in (source, package_dir, task_dir, crab_source)):
        raise ValueError("new_disjoint_output_required")
    for name, expected in {**PINS, "pyyaml.tar.gz": PYYAML_SHA256}.items():
        path = source / name
        if path.is_symlink() or not path.is_file() or _sha(path) != expected:
            raise ValueError("standalone_input_pin_mismatch: " + name)
    package_manifest, package_pins, pin_manifest_sha = _packages(package_dir)
    crab_files = _crab_sources(crab_source)
    task_path = task_dir / "task.json"
    if task_path.is_symlink():
        raise ValueError("task_manifest_symlink_not_allowed")
    task = json.loads(task_path.read_text(encoding="utf-8"))
    if (task.get("task_id") != "boltons-26-singularize-ss-v2" or not task.get("is_fixture") or
            task.get("metadata", {}).get("source_sdist_sha256") !=
            "5566d6cfd5a1e873d8e8476496287a9f92979964611ad9a9cecb6b0ef29b1edd"):
        raise ValueError("pinned_boltons_v2_fixture_required")
    _workspace_files(task_dir / "seed")
    seed_sha = _workspace_digest(task_dir / "seed")
    if not isinstance(build_image, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", build_image) is None:
        raise ValueError("immutable_cached_build_image_id_required")
    architecture = _run(["docker", "image", "inspect", build_image,
                         "--format", "{{.Os}}/{{.Architecture}}"])
    image_id = _run(["docker", "image", "inspect", build_image, "--format", "{{.Id}}"])
    if architecture != "linux/arm64" or image_id != build_image:
        raise ValueError("cached_immutable_arm_build_image_required")
    output.mkdir(parents=True)
    stage = output / "stage"
    stage.mkdir()
    casefold_registry = {}
    extraction_counts = {"standalone_python": _extract_payload(
        source / "python-musl.tar.gz", stage, python_only=True, casefold_registry=casefold_registry)}
    (stage / "usr").mkdir(exist_ok=True)
    (stage / "python").rename(stage / "usr/local")
    pyyaml_provenance = _extract_pyyaml(source / "pyyaml.tar.gz", stage,
                                      casefold_registry=casefold_registry)
    plugin_inputs = None
    for package in package_pins:
        package_path = package_dir / package["filename"]
        managed = package["name"] == "iptables"
        if managed:
            plugin_inputs = _stash_xtables(package_path, output)
        extraction_counts[package["name"]] = _extract_payload(package_path, stage,
            casefold_registry=casefold_registry, exclude_xtables=managed)
    if plugin_inputs is None:
        raise ValueError("complete_original_xtables_plugin_payload_required")
    # No host-root symlink may survive staging. This also guards ELF reads
    # and the offline filesystem builder's dereference behavior.
    for path in stage.rglob("*"):
        if path.is_symlink() and not path.resolve(strict=False).is_relative_to(stage):
            raise ValueError("staged_symlink_escapes_guest_root")
    _require_x86_elf(stage / "usr/local/bin/python3.12")
    for command in ("criu", "runc", "gcc", "iptables", "ip6tables",
                    "iptables-save", "iptables-restore", "ip6tables-save", "ip6tables-restore"):
        found = [stage / prefix / command for prefix in ("usr/sbin", "usr/bin", "sbin", "bin")
                 if (stage / prefix / command).exists()]
        if not found:
            raise ValueError("required_guest_command_missing: " + command)
        _require_x86_elf(found[0])
    gnu_tar_provenance = _require_gnu_tar(stage)
    shutil.copytree(task_dir / "seed", stage / "workspace")
    if _workspace_digest(stage / "workspace") != seed_sha:
        raise ValueError("copied_seed_digest_mismatch")
    for directory in ("etc", "proc", "sys", "dev", "tmp", "run", "root", "var/log"):
        (stage / directory).mkdir(parents=True, exist_ok=True)
    (stage / "tmp").chmod(0o1777)
    (stage / "etc/fpb-disposable-criu-guest").write_text(MARKER_TEXT, encoding="utf-8")
    # CRIU user/group lookups and shell tools need a minimal guest identity.
    for filename, text in (("passwd", "root:x:0:0:root:/root:/bin/sh\n"),
                           ("group", "root:x:0:\n")):
        path = stage / "etc" / filename
        if not path.exists():
            path.write_text(text, encoding="utf-8")
    guest_crab = stage / "opt/fpb/crab"
    for source_path in crab_files:
        target = guest_crab / source_path.relative_to(crab_source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target)
    # Preserve upstream licensing in the disposable guest. These files are
    # runtime inputs, never vendored into this project's public source tree.
    shutil.copyfile(crab_source / "LICENSE", guest_crab / "LICENSE")
    notices = crab_source / "THIRD_PARTY_NOTICES.md"
    if notices.is_file() and not notices.is_symlink():
        shutil.copyfile(notices, guest_crab / "THIRD_PARTY_NOTICES.md")
    guest_probe = stage / "opt/fpb/probe"
    guest_probe.mkdir(parents=True)
    probe_sha256 = {}
    for name in ("check_chain.py", "preflight.py", "memory_worker.c", "paired_epoch_probe.py"):
        path = Path(__file__).with_name(name)
        if path.is_symlink() or not path.is_file():
            raise ValueError("regular_local_probe_source_required: " + name)
        target = guest_probe / name
        shutil.copyfile(path, target)
        probe_sha256[name] = _sha(target)
    _pinned_python_tree(guest_crab, "crab", CRAB_PYTHON_SHA256)
    _pinned_python_tree(guest_crab, "integrations", INTEGRATIONS_PYTHON_SHA256)
    for name in ("vmlinuz-virt", "initramfs-virt", "modloop-virt"):
        shutil.copyfile(source / name, output / name)
    module = output / "modloop-virt-padded.raw"
    shutil.copyfile(output / "modloop-virt", module)
    with module.open("ab") as stream:
        stream.write(b"\0" * (-module.stat().st_size % 512))
    raw = output / "rootfs.raw"
    with raw.open("xb") as stream:
        stream.truncate(rootfs_bytes)
    plugin_objects, plugin_manifest, plugin_manifest_sha, plugin_pairs = plugin_inputs
    linux_builder_raw = Path(__file__).with_name("linux_rootfs_builder.py")
    if linux_builder_raw.is_symlink() or not linux_builder_raw.is_file():
        raise ValueError("regular_linux_rootfs_builder_required")
    linux_builder = linux_builder_raw.resolve()
    _run(["docker", "run", "--rm", "--pull", "never", "--network", "none", "--user", "0:0",
          "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--read-only",
          "--tmpfs", "/tmp:rw,nosuid,nodev,size=32m",
          "--tmpfs", "/linux-stage:rw,nosuid,nodev,size=768m",
          "--mount", f"type=bind,src={stage},dst=/input,readonly",
          "--mount", f"type=bind,src={plugin_objects},dst=/objects,readonly",
          "--mount", f"type=bind,src={linux_builder},dst=/builder/linux_rootfs_builder.py,readonly",
          "--mount", f"type=bind,src={output},dst=/output", "--entrypoint", "python3.12", image_id,
          "-I", "-B", "/builder/linux_rootfs_builder.py", "--input-stage", "/input",
          "--objects", "/objects", "--plugin-manifest", "/output/xtables-manifest.json",
          "--plugin-manifest-sha256", plugin_manifest_sha, "--linux-stage", "/linux-stage/root",
          "--verification-output", "/output/linux-stage-verification.json",
          "--output", "/output/rootfs.raw"], timeout=300)
    linux_verification = json.loads((output / "linux-stage-verification.json").read_text())
    if (linux_verification.get("plugin_manifest_sha256") != plugin_manifest_sha or
            linux_verification.get("plugin_paths_verified") != 121 or
            linux_verification.get("case_sensitive_probe_passed") is not True or
            linux_verification.get("mke2fs_executed") is not True):
        raise ValueError("actual_linux_stage_plugin_verification_required")
    disk = output / "rootfs.qcow2"
    _run(["qemu-img", "convert", "-f", "raw", "-O", "qcow2", str(raw), str(disk)], timeout=300)
    manifest = {"schema_version": "official-crab-criu-guest-assets-v1", "architecture": "linux/amd64",
        "task_id": task["task_id"], "seed_workspace_sha256": seed_sha,
        "source_sdist_sha256": task["metadata"]["source_sdist_sha256"],
        "alpine_sha256": {name: PINS[name] for name in ("vmlinuz-virt", "initramfs-virt", "modloop-virt")},
        "alpine_source": ALPINE_URL, "python_source": PYTHON_URL,
        "python_archive_sha256": PINS["python-musl.tar.gz"],
        "standalone_input_sha256": dict(PINS), "package_inputs_manifest_sha256": pin_manifest_sha,
        "package_distribution": package_manifest.get("distribution"),
        "package_index_sha256": package_manifest.get("index_sha256", {}), "package_pins": package_pins,
        "package_extraction_member_counts": extraction_counts,
        "gnu_tar": gnu_tar_provenance,
        "unhandled_archive_casefold_collisions": 0,
        "xtables_plugins": {"manifest_sha256": plugin_manifest_sha,
                            "paths_verified_on_linux": 121, "regular_files": 115, "symlinks": 6,
                            "casefold_collision_pairs": plugin_pairs,
                            "linux_verification_sha256": _sha(output / "linux-stage-verification.json"),
                            "linux_builder_sha256": _sha(linux_builder)},
        "scope": "Disposable CRIU/runc feasibility guest; edge/testing CRIU with v3.24 dependencies. "
                 "Archive extraction only, not apk installation; compatibility and runtime capabilities unproven.",
        "guest_marker_text": MARKER_TEXT, "guest_marker_sha256": hashlib.sha256(MARKER_TEXT.encode()).hexdigest(),
        "build_image_sha256": image_id, "build_image_architecture": architecture,
        "rootfs_qcow2_sha256": _sha(disk), "modloop_disk_sha256": _sha(module),
        "rootfs_bytes": rootfs_bytes, "network_used": False, "guest_binaries_executed_by_builder": False,
        "guest_crab": {"repository": "https://github.com/open-agent-infra/crab", "commit": CRAB_COMMIT,
                       "python_package_sha256": CRAB_PYTHON_SHA256, "python_files": 52,
                       "integrations_python_sha256": INTEGRATIONS_PYTHON_SHA256,
                       "integrations_python_files": 52,
                       "location": "/opt/fpb/crab", "modified_upstream_files": [],
                       "license_sha256": _sha(crab_source / "LICENSE")},
        "guest_probe_sha256": probe_sha256, "official_repository_code_included_in_guest": True,
        "guest_python_dependencies": [pyyaml_provenance],
        "upstream_source_vendored_in_public_package": False, "performance_measured": False}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    raw.unlink()
    shutil.rmtree(stage)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--packages", required=True)
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--build-image", required=True)
    parser.add_argument("--crab-source", required=True)
    parser.add_argument("--disk-mib", type=int, choices=(512, 768, 1024, 2048, 3072), default=512)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.packages, args.task_dir, args.output,
                             build_image=args.build_image, crab_source=args.crab_source,
                             disk_mib=args.disk_mib), indent=2))
