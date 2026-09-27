"""Reconstruct exact xtables paths on Linux before offline ext4 assembly.

This is a native image-building utility, not guest code. It executes only
the container's mke2fs; no x86 guest binary or plugin is loaded or executed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import stat
import subprocess
import tempfile

PLUGIN_PREFIX = "usr/lib/xtables/"
IPTABLES_APK_SHA256 = "af3aaf9303c829ccee795725965ee6894816a98f9afa62b5dd52b25e6e62e8f0"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path, *, expected_sha256):
    path = Path(path)
    if path.is_symlink() or sha(path) != expected_sha256:
        raise ValueError("plugin_manifest_sha256_mismatch")
    manifest = json.loads(path.read_text())
    if (manifest.get("schema_version") != "fpb-xtables-case-preserving-v1" or
            manifest.get("source_apk_sha256") != IPTABLES_APK_SHA256):
        raise ValueError("pinned_iptables_plugin_manifest_required")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != 121:
        raise ValueError("all_121_original_plugin_entries_required")
    names, files, links, folded = set(), 0, 0, {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("typed_plugin_entry_required")
        name = entry.get("path")
        if (not isinstance(name, str) or not name.startswith(PLUGIN_PREFIX) or
                len(PurePosixPath(name).parts) != 4 or ".." in PurePosixPath(name).parts or
                "\\" in name or "\0" in name or name in names):
            raise ValueError("unique_direct_original_plugin_path_required")
        names.add(name)
        folded.setdefault(name.casefold(), []).append(name)
        mode = entry.get("mode")
        if type(mode) is not int or not 0 <= mode <= 0o777:
            raise ValueError("bounded_plugin_permission_bits_required")
        if entry.get("kind") == "file":
            files += 1
            digest, size = entry.get("sha256"), entry.get("bytes")
            if (not isinstance(digest, str) or SHA256.fullmatch(digest) is None or
                    type(size) is not int or not 0 < size <= 2 * 1024 * 1024 or
                    entry.get("object") != digest + ".bin"):
                raise ValueError("hash_named_bounded_plugin_object_required")
        elif entry.get("kind") == "symlink":
            links += 1
            target = entry.get("target")
            if (not isinstance(target, str) or not target or
                    PurePosixPath(target).name != target or target in {".", ".."} or
                    "\\" in target or "\0" in target):
                raise ValueError("original_relative_plugin_alias_required")
        else:
            raise ValueError("only_regular_files_and_original_symlinks_supported")
    collisions = [sorted(group) for group in folded.values() if len(group) > 1]
    if files != 115 or links != 6 or len(collisions) != 9:
        raise ValueError("exact_reviewed_plugin_types_and_case_pairs_required")
    file_names = {entry["path"] for entry in entries if entry["kind"] == "file"}
    if any(PLUGIN_PREFIX + entry["target"] not in file_names
           for entry in entries if entry["kind"] == "symlink"):
        raise ValueError("plugin_alias_must_target_an_original_regular_file")
    return manifest, sorted(collisions)


def require_case_sensitive(directory):
    directory = Path(directory)
    # Probe only a directory exclusively created by this invocation, so an
    # existing similarly named host file can never be removed on failure.
    probe = Path(tempfile.mkdtemp(prefix=".fpb-case-probe-", dir=directory))
    upper = probe / "A"
    lower = probe / "a"
    try:
        with upper.open("xb") as stream:
            stream.write(b"upper")
        with lower.open("xb") as stream:
            stream.write(b"lower")
        if (upper.stat().st_ino == lower.stat().st_ino or upper.read_bytes() != b"upper" or
                lower.read_bytes() != b"lower"):
            raise ValueError("actual_case_sensitive_linux_stage_required")
    except FileExistsError as exc:
        raise ValueError("actual_case_sensitive_linux_stage_required") from exc
    finally:
        shutil.rmtree(probe)


def reconstruct(stage, objects, manifest):
    stage, objects = Path(stage), Path(objects)
    for relative in ("usr", "usr/lib", "usr/lib/xtables"):
        directory = stage / relative
        if directory.is_symlink():
            raise ValueError("managed_plugin_directory_symlink_forbidden")
        directory.mkdir(parents=True, exist_ok=True)
    plugin_root = stage / "usr/lib/xtables"
    if any(plugin_root.iterdir()):
        raise ValueError("plugins_must_be_absent_from_case_insensitive_input_stage")
    for entry in manifest["entries"]:
        target = stage / entry["path"]
        if entry["kind"] == "file":
            source = objects / entry["object"]
            if (source.is_symlink() or not source.is_file() or
                    source.stat().st_size != entry["bytes"] or sha(source) != entry["sha256"]):
                raise ValueError("encoded_plugin_object_corrupt: " + entry["path"])
            shutil.copyfile(source, target)
            target.chmod(entry["mode"])
    for entry in manifest["entries"]:
        if entry["kind"] == "symlink":
            (stage / entry["path"]).symlink_to(entry["target"])


def verify_plugins(stage, manifest):
    stage = Path(stage)
    root = stage / "usr/lib/xtables"
    actual_names = {path.relative_to(stage).as_posix() for path in root.iterdir()}
    expected_names = {entry["path"] for entry in manifest["entries"]}
    if actual_names != expected_names:
        raise ValueError("reconstructed_plugin_names_differ_from_original_archive")
    verified = []
    for entry in manifest["entries"]:
        path = stage / entry["path"]
        mode = path.lstat().st_mode
        if entry["kind"] == "file":
            if (not stat.S_ISREG(mode) or path.stat().st_size != entry["bytes"] or
                    stat.S_IMODE(mode) != entry["mode"] or sha(path) != entry["sha256"]):
                raise ValueError("reconstructed_plugin_bytes_or_type_mismatch: " + entry["path"])
        elif (not stat.S_ISLNK(mode) or str(path.readlink()) != entry["target"] or
              not path.resolve(strict=True).is_relative_to(root)):
            raise ValueError("reconstructed_plugin_alias_mismatch: " + entry["path"])
        verified.append(dict(entry))
    return {"plugin_paths_verified": len(verified), "regular_files_verified": 115,
            "symlinks_verified": 6, "exact_archive_paths_types_and_bytes": True,
            "entries": verified}


def assemble(input_stage, objects, manifest_path, linux_stage, verification_output, *,
             manifest_sha256, output=None):
    if platform.system() != "Linux":
        raise RuntimeError("native_linux_image_builder_required")
    input_stage, objects, manifest_path, linux_stage, verification_output = [Path(path) for path in
        (input_stage, objects, manifest_path, linux_stage, verification_output)]
    manifest, collisions = load_manifest(manifest_path, expected_sha256=manifest_sha256)
    if (linux_stage.exists() or linux_stage.is_symlink() or
            verification_output.exists() or verification_output.is_symlink()):
        raise ValueError("new_linux_stage_and_verification_target_required")
    linux_stage.parent.mkdir(parents=True, exist_ok=True)
    require_case_sensitive(linux_stage.parent)
    shutil.copytree(input_stage, linux_stage, symlinks=True)
    reconstruct(linux_stage, objects, manifest)
    verification = verify_plugins(linux_stage, manifest)
    report = {"schema_version": "fpb-linux-rootfs-assembly-v1", "builder_system": platform.system(),
              "builder_architecture": platform.machine(), "case_sensitive_probe_passed": True,
              "plugin_manifest_sha256": manifest_sha256,
              "source_apk_sha256": manifest["source_apk_sha256"],
              "casefold_collision_pairs_preserved": collisions,
              "guest_binaries_executed": False, "mke2fs_executed": False, **verification}
    if output is not None:
        output = Path(output)
        if output.is_symlink() or not output.is_file():
            raise ValueError("prepared_regular_rootfs_image_required")
        result = subprocess.run(["mke2fs", "-F", "-t", "ext4", "-d", str(linux_stage), str(output)],
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=300, check=False)
        if result.returncode:
            raise RuntimeError("offline_mke2fs_failed: " + result.stderr[-3000:].decode("utf-8", "replace"))
        report["mke2fs_executed"] = True
        report["rootfs_bytes"] = output.stat().st_size
    verification_output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-stage", required=True)
    parser.add_argument("--objects", required=True)
    parser.add_argument("--plugin-manifest", required=True)
    parser.add_argument("--plugin-manifest-sha256", required=True)
    parser.add_argument("--linux-stage", required=True)
    parser.add_argument("--verification-output", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = assemble(args.input_stage, args.objects, args.plugin_manifest, args.linux_stage,
                      args.verification_output, manifest_sha256=args.plugin_manifest_sha256,
                      output=args.output)
    print(json.dumps({key: value for key, value in result.items() if key != "entries"}, indent=2))
