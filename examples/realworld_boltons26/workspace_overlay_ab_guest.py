"""Trusted guest-local workspace-preparation A/B in one booted ARM64 QEMU VM.

Compare extracting the pinned repository from an uncompressed tar to a fresh
tmpfs overlay over the same ext4 repository bound read-only. The timed region
ends before hashing, repair, execution, and cleanup. No candidate model or
hidden verifier is involved.
"""

from __future__ import annotations

import base64
import ctypes
import hashlib
import importlib.util
import json
import os
import shutil
import stat
import tarfile
import tempfile
import time
from pathlib import Path


SOURCE = Path("boltons/strutils.py")
OLD = "    else:\n        singular = word[:-1]\n    return _match_case(orig_word, singular)"
NEW = "    elif word.endswith('ss'):\n        singular = word\n" + OLD
MS_RDONLY = 1
MS_REMOUNT = 32
MS_BIND = 4096


def _libc():
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.mount.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
                           ctypes.c_ulong, ctypes.c_char_p]
    libc.mount.restype = ctypes.c_int
    libc.umount2.argtypes = [ctypes.c_char_p, ctypes.c_int]
    libc.umount2.restype = ctypes.c_int
    return libc


def _mount(libc, source, target, kind=None, flags=0, options=None):
    result = libc.mount(os.fsencode(source), os.fsencode(target),
                        kind, flags, options)
    if result != 0:
        raise OSError(ctypes.get_errno(), "mount failed")


def _unmount(libc, target):
    if libc.umount2(os.fsencode(target), 0) != 0:
        raise OSError(ctypes.get_errno(), "unmount failed")


def _tree_digest(root):
    digest = hashlib.sha256()
    entries = sorted(root.rglob("*"))
    for path in entries:
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise RuntimeError("unsupported_seed_entry")
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(b"F" if path.is_file() else b"D")
        digest.update(stat.S_IMODE(path.stat().st_mode).to_bytes(2, "big"))
        if path.is_file():
            digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii"))
    return digest.hexdigest()


def _source_result(workspace):
    path = workspace / SOURCE
    contents = path.read_text(encoding="utf-8")
    if contents.count(OLD) != 1:
        raise RuntimeError("repair_anchor_not_unique")
    source_before = hashlib.sha256(contents.encode("utf-8")).hexdigest()
    spec = importlib.util.spec_from_file_location("fpb_dsec_strutils", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if (module.singularize("glass"), module.singularize("glasses")) != ("glas", "glass"):
        raise RuntimeError("baseline_behavior_changed")
    path.write_text(contents.replace(OLD, NEW), encoding="utf-8")
    # Both arms should observe the same repaired source and function output.
    spec = importlib.util.spec_from_file_location("fpb_dsec_fixed", str(path))
    fixed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixed)
    if (fixed.singularize("glass"), fixed.singularize("glasses")) != ("glass", "glass"):
        raise RuntimeError("repair_behavior_changed")
    return source_before, hashlib.sha256(path.read_bytes()).hexdigest()


def _summary_ns(values):
    values = sorted(values)
    count = len(values)
    return {"count": count,
            "median_ms": round((values[(count - 1) // 2] + values[count // 2]) / 2e6, 6),
            "p95_ms": round(values[min(count - 1, (95 * count + 99) // 100 - 1)] / 1e6, 6),
            "min_ms": round(values[0] / 1e6, 6),
            "max_ms": round(values[-1] / 1e6, 6)}


def run(pairs):
    if not 5 <= pairs <= 50:
        raise ValueError("pairs_out_of_range")
    libc = _libc()
    root = Path(tempfile.mkdtemp(prefix="fpb-dsec-overlay-", dir="/"))
    lower = root / "lower"
    lower.mkdir()
    scratch = root / "scratch"
    scratch.mkdir()
    source = Path("/workspace")
    scratch_mounted = lower_mounted = False
    try:
        setup_ns = {}
        setup_started = time.perf_counter_ns()
        _mount(libc, source, lower, flags=MS_BIND)
        lower_mounted = True
        _mount(libc, source, lower, flags=MS_BIND | MS_REMOUNT | MS_RDONLY)
        # A read-only bind is the actual lower; this is ext4, not EROFS.
        if not os.statvfs(lower).f_flag & os.ST_RDONLY:
            raise RuntimeError("lower_bind_not_readonly")
        setup_ns["readonly_bind"] = time.perf_counter_ns() - setup_started
        setup_started = time.perf_counter_ns()
        _mount(libc, b"tmpfs", scratch, kind=b"tmpfs", options=b"size=32m,mode=0700")
        scratch_mounted = True
        setup_ns["scratch_tmpfs_mount"] = time.perf_counter_ns() - setup_started
        setup_started = time.perf_counter_ns()
        seed_digest = _tree_digest(lower)
        setup_ns["seed_tree_digest"] = time.perf_counter_ns() - setup_started
        archive = root / "seed.tar"
        setup_started = time.perf_counter_ns()
        with tarfile.open(archive, "w") as stream:
            stream.add(lower, arcname=".")
        setup_ns["tar_creation"] = time.perf_counter_ns() - setup_started
        seed_file_count = sum(path.is_file() for path in lower.rglob("*"))
        seed_payload_bytes = sum(path.stat().st_size for path in lower.rglob("*") if path.is_file())
        attempts = []
        seen_repair_sha = None

        def attempt(arm, pair, warmup=False):
            nonlocal seen_repair_sha
            index = len(attempts)
            parent = scratch / ("attempt-" + str(index))
            parent.mkdir()
            target = parent / "workspace"
            mounted = False
            started = time.perf_counter_ns()
            if arm == "tar_extract":
                target.mkdir()
                with tarfile.open(archive, "r") as stream:
                    stream.extractall(target, filter="data")
            elif arm == "overlay":
                upper, work = parent / "upper", parent / "work"
                for path in (upper, work, target):
                    path.mkdir()
                options = f"lowerdir={lower},upperdir={upper},workdir={work}".encode()
                _mount(libc, b"overlay", target, kind=b"overlay", options=options)
                mounted = True
            else:
                raise ValueError("invalid_arm")
            prepare_ns = time.perf_counter_ns() - started
            try:
                actual_digest = _tree_digest(target)
                if actual_digest != seed_digest:
                    raise RuntimeError("source_tree_digest_mismatch")
                source_before, repair_sha = _source_result(target)
                if seen_repair_sha is not None and repair_sha != seen_repair_sha:
                    raise RuntimeError("repair_digest_mismatch_across_arms")
                seen_repair_sha = repair_sha
                if source_before != hashlib.sha256((lower / SOURCE).read_bytes()).hexdigest():
                    raise RuntimeError("source_digest_mismatch")
                if _tree_digest(lower) != seed_digest:
                    raise RuntimeError("lower_mutated_during_attempt")
            finally:
                if mounted:
                    _unmount(libc, target)
                shutil.rmtree(parent)
            if parent.exists():
                raise RuntimeError("attempt_cleanup_failed")
            row = {"arm": arm, "pair": pair, "warmup": warmup,
                   "prepare_ns": prepare_ns, "tree_digest_ok": True,
                   "baseline_glass": "glas", "repaired_glass": "glass",
                   "repaired_source_sha256": repair_sha, "lower_unchanged": True,
                   "cleanup_complete": True}
            attempts.append(row)

        for arm in ("tar_extract", "overlay"):
            attempt(arm, -1, True)
        for pair in range(pairs):
            order = ("tar_extract", "overlay") if pair % 2 == 0 else ("overlay", "tar_extract")
            for arm in order:
                attempt(arm, pair)
        measured = [row for row in attempts if not row["warmup"]]
        tar_times = [row["prepare_ns"] for row in measured if row["arm"] == "tar_extract"]
        overlay_times = [row["prepare_ns"] for row in measured if row["arm"] == "overlay"]
        return {"kind": "guest_workspace_preparation_ab_v1", "status": "passed",
                "boundary": "one_booted_arm64_qemu_guest_ext4_readonly_bind_lower_tmpfs_overlay_upper",
                "pairs": pairs, "seed_tree_sha256": seed_digest,
                "seed_file_count": seed_file_count,
                "seed_payload_bytes": seed_payload_bytes,
                "archive_bytes": archive.stat().st_size,
                "one_time_guest_setup_ms": {key: round(value / 1e6, 6)
                                            for key, value in setup_ns.items()},
                "attempts": attempts,
                "summary": {"tar_extract": _summary_ns(tar_times),
                            "overlay": _summary_ns(overlay_times),
                            "median_ratio_tar_to_overlay": round(
                                _summary_ns(tar_times)["median_ms"] /
                                _summary_ns(overlay_times)["median_ms"], 4)},
                "all_tree_digests_equal": True,
                "all_repairs_equal": True,
                "lower_readonly_and_unchanged": True,
                "all_attempts_cleaned": True,
                "physical_write_bytes": None,
                "physical_write_bytes_reason": "tmpfs and overlay metadata writes are not separately attributable"}
    finally:
        if scratch_mounted:
            _unmount(libc, scratch)
        if lower_mounted:
            _unmount(libc, lower)
        shutil.rmtree(root)


if __name__ == "__main__":
    import sys
    outcome = run(int(sys.argv[1]))
    encoded = base64.b64encode(json.dumps(outcome, separators=(",", ":")).encode()).decode()
    print("FPB_DSEC_RESULT:" + encoded)
