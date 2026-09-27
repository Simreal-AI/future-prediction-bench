"""Fetch only the fixed, SHA256-pinned public inputs for the CRIU guest.

Requires curl. No credential arguments, arbitrary URLs, package installation,
archive extraction, or guest binary execution are supported. Each download
lands in a private partial file and is published only after size/hash checks.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
from urllib.parse import urlparse
import uuid

from examples.official_crab.prepare_x86 import ALPINE_URL, PINS, PYTHON_URL
from examples.official_crab_criu.prepare_guest import PYYAML_SHA256, PYYAML_URL

PACKAGE_INPUTS_SHA256 = "e378e938f370f7894d0f95a72f03e812950a5e9490a6577ca93ded43982b6019"
PRIMARY_DOMAINS = {"dl-cdn.alpinelinux.org", "github.com", "files.pythonhosted.org"}
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
BASE_INPUTS = (
    {"filename": "vmlinuz-virt", "url": ALPINE_URL + "vmlinuz-virt",
     "sha256": PINS["vmlinuz-virt"], "bytes": 12608512},
    {"filename": "initramfs-virt", "url": ALPINE_URL + "initramfs-virt",
     "sha256": PINS["initramfs-virt"], "bytes": 9654237},
    {"filename": "modloop-virt", "url": ALPINE_URL + "modloop-virt",
     "sha256": PINS["modloop-virt"], "bytes": 22945792},
    {"filename": "python-musl.tar.gz", "url": PYTHON_URL,
     "sha256": PINS["python-musl.tar.gz"], "bytes": 28091991},
    {"filename": "musl.apk",
     "url": "https://dl-cdn.alpinelinux.org/alpine/v3.24/main/x86_64/musl-1.2.6-r2.apk",
     "sha256": PINS["musl.apk"], "bytes": 415475},
    {"filename": "pyyaml.tar.gz", "url": PYYAML_URL,
     "sha256": PYYAML_SHA256, "bytes": 130960},
)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_record(record, *, apk=False):
    filename, url, digest, size = (record.get(name) for name in
                                  ("filename", "url", "sha256", "bytes"))
    if (not isinstance(filename, str) or Path(filename).name != filename or
            filename in {"", ".", ".."} or "\\" in filename or
            not isinstance(url, str) or not isinstance(digest, str) or
            SHA256.fullmatch(digest) is None or type(size) is not int or
            not 0 < size <= 100 * 1024 * 1024):
        raise ValueError("invalid_fixed_download_record")
    parsed = urlparse(url)
    if (parsed.scheme != "https" or parsed.hostname not in PRIMARY_DOMAINS or
            parsed.username is not None or parsed.password is not None or
            parsed.port is not None or parsed.query or parsed.fragment):
        raise ValueError("fixed_primary_https_source_required")
    if apk and (parsed.hostname != "dl-cdn.alpinelinux.org" or
                not parsed.path.startswith(("/alpine/v3.24/main/x86_64/",
                                            "/alpine/v3.24/community/x86_64/",
                                            "/alpine/edge/testing/x86_64/")) or
                Path(parsed.path).name != filename or not filename.endswith(".apk")):
        raise ValueError("fixed_x86_alpine_package_required")


def load_inputs():
    path = Path(__file__).with_name("package_inputs.json")
    if path.is_symlink() or _sha(path) != PACKAGE_INPUTS_SHA256:
        raise ValueError("bundled_package_manifest_pin_mismatch")
    raw = path.read_bytes()
    manifest = json.loads(raw)
    packages = manifest.get("packages")
    if (manifest.get("architecture") != "x86_64" or
            not isinstance(packages, list) or not 1 <= len(packages) <= 128):
        raise ValueError("bounded_pinned_x86_apk_graph_required")
    names = set()
    for record in packages:
        _validate_record(record, apk=True)
        if record["filename"] in names:
            raise ValueError("duplicate_package_filename")
        names.add(record["filename"])
    for record in BASE_INPUTS:
        _validate_record(record)
    return raw, manifest


def _verify(path, record):
    if (path.is_symlink() or not path.is_file() or
            path.stat().st_size != record["bytes"] or _sha(path) != record["sha256"]):
        raise ValueError("download_size_or_sha256_mismatch: " + record["filename"])


def _publish_verified(partial, target, record):
    if partial.parent != target.parent or target.exists() or target.is_symlink():
        raise ValueError("new_target_in_owned_download_directory_required")
    _verify(partial, record)
    with partial.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(partial, target)
    descriptor = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_bytes(path, raw):
    if path.exists() or path.is_symlink():
        raise ValueError("new_manifest_target_required")
    partial = path.with_name(".fpb-manifest-" + uuid.uuid4().hex + ".part")
    descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        partial.unlink(missing_ok=True)


def validate_local(source, packages):
    """Read-only verification of already obtained inputs; makes no requests."""
    _, manifest = load_inputs()
    for record in BASE_INPUTS:
        _verify(Path(source) / record["filename"], record)
    for record in manifest["packages"]:
        _verify(Path(packages) / record["filename"], record)
    return {"base_and_python_inputs": len(BASE_INPUTS), "apk_inputs": len(manifest["packages"]),
            "all_size_and_sha256_checks_passed": True, "network_used": False}


def _download(curl, record, directory, stop):
    if stop.is_set():
        raise RuntimeError("download_batch_cancelled")
    target = directory / record["filename"]
    partial = directory / (".fpb-download-" + uuid.uuid4().hex + ".part")
    descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    try:
        # --disable is first to prevent ~/.curlrc from adding credentials or
        # changing request behavior. Redirects must also remain HTTPS.
        argv = [curl, "--disable", "--fail", "--silent", "--show-error", "--location",
                "--max-redirs", "5", "--proto", "=https", "--proto-redir", "=https",
                "--tlsv1.2", "--connect-timeout", "20", "--max-time", "240",
                "--retry", "2", "--retry-delay", "1", "--max-filesize", str(record["bytes"]),
                "--output", str(partial), "--url", record["url"]]
        process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.PIPE)
        deadline = time.monotonic() + 750
        try:
            while True:
                if stop.is_set() or time.monotonic() >= deadline:
                    raise RuntimeError("download_batch_cancelled_or_timed_out")
                try:
                    _, stderr = process.communicate(timeout=0.25)
                    break
                except subprocess.TimeoutExpired:
                    pass
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate()
        if process.returncode:
            raise RuntimeError("fixed_input_download_failed: " + record["filename"] + ": " +
                               stderr[-1500:].decode("utf-8", "replace"))
        if stop.is_set():
            raise RuntimeError("download_batch_cancelled")
        _publish_verified(partial, target, record)
        return record["filename"]
    finally:
        partial.unlink(missing_ok=True)


def fetch(source_output, packages_output, *, workers=4):
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("download_concurrency_must_be_one_to_four")
    raw, manifest = load_inputs()
    source_raw, packages_raw = Path(source_output), Path(packages_output)
    source, packages = source_raw.resolve(), packages_raw.resolve()
    inputs = Path(__file__).parent.resolve()
    if (source_raw.is_symlink() or packages_raw.is_symlink() or
            source.exists() or packages.exists() or source.is_relative_to(packages) or
            packages.is_relative_to(source) or any(
                path.is_relative_to(inputs) or inputs.is_relative_to(path)
                for path in (source, packages))):
        raise ValueError("new_disjoint_source_and_package_outputs_required")
    curl = shutil.which("curl")
    if curl is None:
        raise RuntimeError("curl_required")
    source.mkdir(parents=True, mode=0o700)
    packages.mkdir(parents=True, mode=0o700)
    stop = threading.Event()
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_download, curl, record, directory, stop)
                       for directory, records in ((source, BASE_INPUTS), (packages, manifest["packages"]))
                       for record in records]
            try:
                for future in as_completed(futures):
                    future.result()
            except BaseException:
                stop.set()
                for pending in futures:
                    pending.cancel()
                raise
        validate_local(source, packages)
        summary = {"schema_version": "criu-fixed-input-download-v1",
                   "downloaded_files": len(BASE_INPUTS) + len(manifest["packages"]),
                   "max_concurrent_downloads": workers, "actual_sha256_verified": True,
                   "base_inputs": list(BASE_INPUTS), "package_manifest_sha256": PACKAGE_INPUTS_SHA256,
                   "scope": "Pinned public download inputs only; no extraction, installation, or execution."}
        _atomic_bytes(source / "download-summary.json", (json.dumps(summary, indent=2) + "\n").encode())
        # Publish the builder's completion input only after all downloads and
        # source summary have successfully committed.
        _atomic_bytes(packages / "pinned-package-inputs.json", raw)
        return summary
    except BaseException:
        stop.set()
        # Verified files are retained for diagnosis; a failed batch publishes
        # no completed package manifest. Only its own partial files are removed.
        for directory in (source, packages):
            for partial in directory.glob(".fpb-*.part"):
                partial.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-output", required=True)
    parser.add_argument("--packages-output", required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    result = (validate_local(args.source_output, args.packages_output) if args.verify_only
              else fetch(args.source_output, args.packages_output, workers=args.workers))
    print(json.dumps(result, indent=2))
