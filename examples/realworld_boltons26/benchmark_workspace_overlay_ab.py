"""Reproduce the pinned Boltons guest-local tar versus overlay preparation A/B.

The timing covers only creation of a ready workspace in one booted ARM64 QEMU
guest. It excludes boot, seed/archive preparation, checking, repair, and
cleanup. This is a solved public fixture, not a held-out agent task.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples.realworld_boltons26.microvm_benchmark import _boot, _runtime, _sha
from future_prediction_bench.coding_env import _workspace_digest
from future_prediction_bench.microvm_runtime import _clone_or_copy_qcow2


HERE = Path(__file__).resolve().parent
GUEST = Path("/mnt/root/fpb_workspace_overlay_ab_guest.py")
TASK_ID = "boltons-26-singularize-ss-v2"
SEED_SOURCE_SHA256 = "f7f4873406d3913372c9d2b1296cc5e3efb87e88808457e93fc212a5df2de18e"
SHA256 = re.compile(r"[a-f0-9]{64}\Z")


def _required(runtime, command, *, timeout=30):
    result = runtime.run_shell(command, timeout=timeout)
    if result["return_code"]:
        raise RuntimeError("guest_command_failed: " + result["stdout"][-1600:])
    return result["stdout"]


def _upload(runtime, local):
    data = local.read_bytes()
    if not 0 < len(data) < 65536:
        raise ValueError("guest_program_size_out_of_bounds")
    _required(runtime, ": > " + str(GUEST))
    for offset in range(0, len(data), 2700):
        encoded = base64.b64encode(data[offset:offset + 2700]).decode("ascii")
        _required(runtime, f"printf '%s' '{encoded}' | base64 -d >> {GUEST}")
    actual = _required(runtime, f"sha256sum {GUEST}").split()[0]
    expected = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise RuntimeError("guest_program_digest_mismatch")
    return expected


def _summary_ns(values):
    ordered = sorted(values)
    count = len(ordered)
    return {"count": count,
            "median_ms": round(statistics.median(ordered) / 1e6, 6),
            "p95_ms": round(ordered[min(count - 1, (95 * count + 99) // 100 - 1)] / 1e6, 6),
            "min_ms": round(ordered[0] / 1e6, 6),
            "max_ms": round(ordered[-1] / 1e6, 6)}


def _check_guest_report(guest, *, pairs, seed_digest):
    """Reject incomplete or internally inconsistent serial reports."""
    if (not isinstance(guest, dict) or guest.get("kind") != "guest_workspace_preparation_ab_v1"
            or guest.get("status") != "passed" or guest.get("pairs") != pairs
            or guest.get("seed_tree_sha256") != seed_digest
            or guest.get("boundary")
               != "one_booted_arm64_qemu_guest_ext4_readonly_bind_lower_tmpfs_overlay_upper"
            or not all(guest.get(key) is True for key in (
                "all_tree_digests_equal", "all_repairs_equal",
                "lower_readonly_and_unchanged", "all_attempts_cleaned"))
            or guest.get("physical_write_bytes") is not None
            or type(guest.get("seed_file_count")) is not int
            or not 1 <= guest["seed_file_count"] <= 5000
            or type(guest.get("seed_payload_bytes")) is not int
            or not 0 < guest["seed_payload_bytes"] <= 50_000_000
            or type(guest.get("archive_bytes")) is not int
            or guest["archive_bytes"] < guest["seed_payload_bytes"]):
        raise RuntimeError("guest_report_contract_failed")
    setup = guest.get("one_time_guest_setup_ms")
    if (not isinstance(setup, dict)
            or set(setup) != {"readonly_bind", "scratch_tmpfs_mount",
                                  "seed_tree_digest", "tar_creation"}
            or any(type(value) not in (int, float) or not math.isfinite(value)
                   or value < 0 for value in setup.values())):
        raise RuntimeError("guest_setup_report_invalid")
    attempts = guest.get("attempts")
    if not isinstance(attempts, list) or len(attempts) != 2 * pairs + 2:
        raise RuntimeError("guest_attempt_count_invalid")
    expected = [("tar_extract", -1, True), ("overlay", -1, True)]
    for pair in range(pairs):
        order = ("tar_extract", "overlay") if pair % 2 == 0 else ("overlay", "tar_extract")
        expected.extend((arm, pair, False) for arm in order)
    repaired_digests = set()
    for row, (arm, pair, warmup) in zip(attempts, expected):
        if (not isinstance(row, dict) or row.get("arm") != arm
                or row.get("pair") != pair or row.get("warmup") is not warmup
                or type(row.get("prepare_ns")) is not int or row["prepare_ns"] < 0
                or row.get("tree_digest_ok") is not True
                or row.get("lower_unchanged") is not True
                or row.get("cleanup_complete") is not True
                or row.get("baseline_glass") != "glas"
                or row.get("repaired_glass") != "glass"
                or not isinstance(row.get("repaired_source_sha256"), str)
                or not SHA256.fullmatch(row["repaired_source_sha256"])):
            raise RuntimeError("guest_attempt_semantics_invalid")
        repaired_digests.add(row["repaired_source_sha256"])
    if len(repaired_digests) != 1:
        raise RuntimeError("guest_repair_digest_mismatch")
    values = {arm: [row["prepare_ns"] for row in attempts[2:] if row["arm"] == arm]
              for arm in ("tar_extract", "overlay")}
    summary = {arm: _summary_ns(values[arm]) for arm in values}
    summary["median_ratio_tar_to_overlay"] = round(
        summary["tar_extract"]["median_ms"] / summary["overlay"]["median_ms"], 4)
    if guest.get("summary") != summary:
        raise RuntimeError("guest_summary_mismatch")
    return guest


def run(task_dir, assets_dir, output, *, pairs=20):
    task, assets, output = map(lambda value: Path(value).resolve(),
                               (task_dir, assets_dir, output))
    if (output.exists() and (not output.is_dir() or any(output.iterdir()))
            or output.is_relative_to(task) or task.is_relative_to(output)
            or output.is_relative_to(assets) or assets.is_relative_to(output)):
        raise ValueError("output_must_be_new_or_empty")
    if type(pairs) is not int or not 5 <= pairs <= 50:
        raise ValueError("pairs_out_of_range")
    manifest = json.loads((assets / "manifest.json").read_text(encoding="utf-8"))
    task_spec = json.loads((task / "task.json").read_text(encoding="utf-8"))
    seed_digest = _workspace_digest(task / "seed")
    if (manifest.get("schema_version") != "boltons-microvm-assets-v2"
            or manifest.get("task_id") != TASK_ID
            or task_spec.get("task_id") != TASK_ID
            or manifest.get("source_sdist_sha256")
               != task_spec.get("metadata", {}).get("source_sdist_sha256")
            or manifest.get("seed_workspace_sha256") != seed_digest
            or _sha(assets / "rootfs.qcow2") != manifest.get("rootfs_qcow2_sha256")):
        raise RuntimeError("pinned_vm_asset_changed")
    source_sha = hashlib.sha256((task / "seed/boltons/strutils.py").read_bytes()).hexdigest()
    if source_sha != SEED_SOURCE_SHA256:
        raise RuntimeError("pinned_seed_source_changed")
    output.mkdir(parents=True, exist_ok=True)
    disk = output / "child.qcow2"
    report = {"kind": "host_observed_guest_workspace_preparation_ab_v1",
              "status": "incomplete", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "pairs": pairs,
              "timing_scope": "guest_local_workspace_preparation_only",
              "setup": {}, "setup_seconds": {},
              "asset_binding": {
                  "manifest_schema_version": manifest["schema_version"],
                  "task_id": TASK_ID,
                  "source_sdist_sha256": manifest["source_sdist_sha256"],
                  "seed_workspace_sha256": seed_digest,
                  "rootfs_seed_sha256": manifest["rootfs_qcow2_sha256"],
                  "source_sha256": source_sha},
              "guest_program_sha256": None}
    runtime = None
    try:
        started = time.perf_counter()
        report["setup"]["disk_clone_mode"] = _clone_or_copy_qcow2(assets / "rootfs.qcow2", disk)
        report["setup_seconds"]["clone"] = time.perf_counter() - started
        runtime = _runtime(assets, disk)
        started = time.perf_counter()
        _boot(runtime)
        report["setup_seconds"]["boot"] = time.perf_counter() - started
        _required(runtime, "modprobe overlay")
        started = time.perf_counter()
        report["guest_program_sha256"] = _upload(runtime, HERE / "workspace_overlay_ab_guest.py")
        report["setup_seconds"]["upload"] = time.perf_counter() - started
        started = time.perf_counter()
        stdout = _required(runtime,
                           "LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/aarch64-linux-gnu "
                           f"chroot /mnt/root /usr/local/bin/python3.12 -I -B /{GUEST.name} {pairs}",
                           timeout=120)
        report["guest_call_seconds"] = time.perf_counter() - started
        matches = re.findall(r"FPB_DSEC_RESULT:([A-Za-z0-9+/=]+)", stdout)
        if len(matches) != 1:
            raise RuntimeError("guest_report_framing_failed")
        guest = json.loads(base64.b64decode(matches[0], validate=True))
        report["guest"] = _check_guest_report(guest, pairs=pairs, seed_digest=seed_digest)
        report["vm_metrics"] = runtime.get_state()["metrics"]
        report["status"] = "passed"
    except BaseException as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        raise
    finally:
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if runtime is not None:
            runtime.close()
        disk.unlink(missing_ok=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=20)
    args = parser.parse_args()
    result = run(args.task_dir, args.assets_dir, args.output, pairs=args.pairs)
    print(json.dumps({"status": result["status"], "summary": result["guest"]["summary"]},
                     indent=2))
