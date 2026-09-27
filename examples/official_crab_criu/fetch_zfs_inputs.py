"""Fetch or copy the exact measured ZFS APK cohort without installation."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import uuid
import zlib

FIXED_INPUT_SHA256 = "af70532ef7fd32972d13caf5d936c36de8cfcabbd5f70a5df59fdbc3373e12fb"
PLAN = Path(__file__).with_name("zfs_package_inputs.json")


def new_output_path(value):
    raw = Path(value).absolute()
    if raw.exists() or raw.is_symlink() or any(p.is_symlink() for p in raw.parents):
        raise ValueError("new_nonsymlink_output_required")
    normalized = raw.resolve(strict=False)
    if normalized.exists() or normalized.is_symlink():
        raise ValueError("new_nonsymlink_output_required")
    return normalized


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def fixed_plan():
    if PLAN.is_symlink() or sha(PLAN) != FIXED_INPUT_SHA256:
        raise ValueError("reviewed_fixed_ZFS_input_plan_required")
    return json.loads(PLAN.read_bytes())


def apk_identity(path, record):
    """Verify whole bytes, APK control identity and its bound data stream."""
    path = Path(path)
    if (path.is_symlink() or not path.is_file() or
            path.stat().st_size != record["bytes"] or sha(path) != record["sha256"]):
        raise ValueError("whole_APK_pin_mismatch:" + record["filename"])
    data, parts = path.read_bytes(), []
    while data:
        decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
        raw = decoder.decompress(data)
        if not decoder.eof:
            raise ValueError("complete_APK_gzip_member_required")
        size = len(data) - len(decoder.unused_data)
        parts.append((data[:size], raw))
        data = decoder.unused_data
    controls = []
    for index, (compressed, raw) in enumerate(parts):
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:", ignore_zeros=True) as archive:
            entries = [m for m in archive.getmembers() if m.name == ".PKGINFO"]
            if not entries:
                continue
            if len(entries) != 1 or not entries[0].isfile() or entries[0].size > 65536:
                raise ValueError("one_bounded_APK_control_required")
            controls.append((index, compressed, archive.extractfile(entries[0]).read().decode()))
    if len(controls) != 1:
        raise ValueError("one_APK_control_stream_required")
    index, compressed, text = controls[0]
    import base64
    control = "Q1" + base64.b64encode(hashlib.sha1(compressed).digest()).decode()
    if control != record["apkindex_checksum"]:
        raise ValueError("recorded_APKINDEX_control_checksum_mismatch")
    fields = {}
    for line in text.splitlines():
        if " = " in line and not line.startswith("#"):
            key, value = line.split(" = ", 1)
            if key in {"pkgname", "pkgver", "arch", "datahash"}:
                if key in fields:
                    raise ValueError("duplicate_APK_identity_field")
                fields[key] = value
    if (fields.get("pkgname") != record["pkgname"] or
            fields.get("pkgver") != record["pkgver"] or
            fields.get("arch") not in {record["architecture"], "noarch"}):
        raise ValueError("APK_package_identity_mismatch")
    if (len(parts) != index + 2 or
            not re.fullmatch("[0-9a-f]{64}", fields.get("datahash", "")) or
            hashlib.sha256(parts[index + 1][0]).hexdigest() != fields["datahash"]):
        raise ValueError("control_bound_APK_payload_mismatch")
    return {"pkginfo": fields, "apkindex_control_checksum_verified": True,
            "control_datahash_verified": True}


def validate_inputs(directory):
    directory = Path(directory).resolve()
    plan = fixed_plan()
    manifest = directory / "inputs.json"
    if manifest.is_symlink() or not manifest.is_file() or manifest.stat().st_size > 131072:
        raise ValueError("marked_ZFS_inputs_required")
    recorded = json.loads(manifest.read_bytes())
    if (not isinstance(recorded, dict) or recorded.get("schema_version") != "official-crab-zfs-inputs-v1"
            or any(recorded.get(key) != plan[key] for key in ("architecture", "kernel_release", "zfs_version"))):
        raise ValueError("marked_fixed_ZFS_cohort_required")
    packages = recorded.get("packages")
    if not isinstance(packages, list) or len(packages) != len(plan["packages"]):
        raise ValueError("exact_fixed_package_cohort_required")
    for expected, actual in zip(plan["packages"], packages):
        if not isinstance(actual, dict) or any(actual.get(key) != value for key, value in expected.items()):
            raise ValueError("fixed_ZFS_package_record_mismatch")
        apk_identity(directory / expected["filename"], expected)
    return recorded


def fetch(output, *, cache=None):
    output = new_output_path(output)
    plan = fixed_plan()
    cache = Path(cache).resolve() if cache is not None else None
    if cache is not None and (output.is_relative_to(cache) or cache.is_relative_to(output)):
        raise ValueError("disjoint_cache_and_output_required")
    output.mkdir(parents=True, mode=0o700)

    def one(record):
        name, url = record["filename"], record["url"]
        if (not re.fullmatch(r"[A-Za-z0-9_.+-]+\.apk", name) or
                url != "https://dl-cdn.alpinelinux.org/alpine/v3.24/main/x86_64/" + name):
            raise ValueError("fixed_primary_APK_URL_required")
        partial = output / ("." + uuid.uuid4().hex + ".partial")
        descriptor = os.open(partial, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as sink:
                if cache is not None:
                    source = cache / name
                    apk_identity(source, record)
                    with source.open("rb") as stream:
                        shutil.copyfileobj(stream, sink)
                else:
                    result = subprocess.run(["curl", "--disable", "--fail", "--silent", "--show-error",
                        "--location", "--proto", "=https", "--proto-redir", "=https", "--connect-timeout", "15",
                        "--max-time", "180", "--retry", "2", "--url", url],
                        stdin=subprocess.DEVNULL, stdout=sink, stderr=subprocess.PIPE, timeout=400)
                    if result.returncode:
                        raise RuntimeError("primary_APK_download_failed:" + name + ":" +
                                           result.stderr[-1000:].decode(errors="replace"))
                sink.flush()
                os.fsync(sink.fileno())
            identity = apk_identity(partial, record)
            os.link(partial, output / name, follow_symlinks=False)
            return {**record, **identity}
        finally:
            partial.unlink(missing_ok=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(one, plan["packages"]))
    report = {**plan, "schema_version": "official-crab-zfs-inputs-v1", "packages": records,
              "input_plan_sha256": FIXED_INPUT_SHA256, "network_used": cache is None,
              "package_installation": False, "guest_execution": False,
              "APK_signatures_verified": False}
    (output / "inputs.json").write_text(json.dumps(report, indent=2) + "\n")
    validate_inputs(output)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache", help="Read-only local copy of these exact measured APK versions")
    args = parser.parse_args()
    report = fetch(args.output, cache=args.cache)
    print(json.dumps({"packages": len(report["packages"]), "network_used": report["network_used"],
                      "bytes": sum(p["bytes"] for p in report["packages"])}, indent=2))
