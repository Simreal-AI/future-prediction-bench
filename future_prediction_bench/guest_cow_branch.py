"""Guest-local fork plus overlayfs experiment for one trusted Boltons fixture.

This module is copied into an already-running ARM64 Linux VM and executed by
the trusted host. It is deliberately self-contained: no benchmark policy can
invoke its mount or fork operations. Each fork inherits a warmed Python heap;
each overlay gets a distinct upper/work directory over the same lower tree.
Those are useful COW primitives, not a secure replacement for a VM sandbox.
"""

from __future__ import annotations

import base64
import ctypes
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path


LOWER = Path("/workspace")
SOURCE_RELATIVE = Path("boltons/strutils.py")
OLD = "    else:\n        singular = word[:-1]\n    return _match_case(orig_word, singular)"
NEW = "    elif word.endswith('ss'):\n        singular = word\n" + OLD
WARM_STATE = {"counter": 7, "marker": "prefork"}


def _mount_api():
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.mount.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
                           ctypes.c_ulong, ctypes.c_char_p]
    libc.mount.restype = ctypes.c_int
    libc.umount2.argtypes = [ctypes.c_char_p, ctypes.c_int]
    libc.umount2.restype = ctypes.c_int
    return libc


def _mount_overlay(libc, lower: Path, upper: Path, work: Path, merged: Path):
    for path in (upper, work, merged):
        path.mkdir(parents=True, exist_ok=False)
    options = f"lowerdir={lower},upperdir={upper},workdir={work}".encode()
    if libc.mount(b"overlay", os.fsencode(merged), b"overlay", 0, options) != 0:
        error = ctypes.get_errno()
        raise OSError(error, "overlayfs mount failed")


def _unmount_overlay(libc, merged: Path):
    if libc.umount2(os.fsencode(merged), 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, "overlayfs unmount failed")


def _module_value(path: Path):
    spec = importlib.util.spec_from_file_location("fpb_branch_strutils", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load branch source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.singularize("glass"), module.singularize("glasses")


def _child_result(mode: str, merged: Path):
    inherited = WARM_STATE["counter"]
    WARM_STATE["counter"] = 11 if mode == "fix" else 12
    source = merged / SOURCE_RELATIVE
    original = source.read_text(encoding="utf-8")
    if mode == "fix":
        if original.count(OLD) != 1:
            raise RuntimeError("Pinned Boltons repair anchor differs")
        source.write_text(original.replace(OLD, NEW), encoding="utf-8")
    elif mode != "baseline":
        raise ValueError("Unknown branch mode")
    glass, glasses = _module_value(source)
    content = source.read_bytes()
    return {"mode": mode, "inherited_counter": inherited,
            "branch_counter": WARM_STATE["counter"], "glass": glass,
            "glasses": glasses, "source_sha256": hashlib.sha256(content).hexdigest(),
            "contains_fix": NEW in content.decode("utf-8")}


def _fork_json(function):
    reader, writer = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(reader)
        try:
            payload = {"ok": True, "value": function()}
            status = 0
        except BaseException as exc:
            payload = {"ok": False, "error_type": type(exc).__name__,
                       "error": str(exc)[:500]}
            status = 1
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        try:
            os.write(writer, data[:8192])
        finally:
            os.close(writer)
        os._exit(status)
    os.close(writer)
    return pid, reader


def _receive_json(pid: int, reader: int):
    chunks = []
    total = 0
    try:
        while True:
            chunk = os.read(reader, 4096)
            if not chunk:
                break
            total += len(chunk)
            if total > 8192:
                raise RuntimeError("Branch result exceeded bound")
            chunks.append(chunk)
    finally:
        os.close(reader)
    _, status = os.waitpid(pid, 0)
    payload = json.loads(b"".join(chunks))
    if not os.WIFEXITED(status) or os.WEXITSTATUS(status) != 0 or not payload.get("ok"):
        raise RuntimeError("Branch failed: " + repr(payload))
    return payload["value"]


def _fork_ready_and_write(marker: Path):
    reader, writer = os.pipe()
    started = time.perf_counter_ns()
    pid = os.fork()
    if pid == 0:
        os.close(reader)
        try:
            if WARM_STATE["counter"] != 7:
                os._exit(2)
            os.write(writer, b"R")
            os.close(writer)
            WARM_STATE["counter"] = 99
            marker.write_text("child-only", encoding="ascii")
            os._exit(0)
        except BaseException:
            os._exit(3)
    os.close(writer)
    try:
        ready = os.read(reader, 1)
        fork_ready_ns = time.perf_counter_ns() - started
    finally:
        os.close(reader)
    reap_started = time.perf_counter_ns()
    _, status = os.waitpid(pid, 0)
    reap_ns = time.perf_counter_ns() - reap_started
    if ready != b"R" or not os.WIFEXITED(status) or os.WEXITSTATUS(status) != 0:
        raise RuntimeError("Forked branch failed")
    if marker.read_text(encoding="ascii") != "child-only":
        raise RuntimeError("Forked branch did not write through overlay")
    return fork_ready_ns, reap_ns


def _summarize_ns(values):
    ordered = sorted(values)
    return {"count": len(ordered), "median_ms": round(
                (ordered[(len(ordered) - 1) // 2] + ordered[len(ordered) // 2]) / 2_000_000, 6),
            "p95_ms": round(ordered[min(len(ordered) - 1,
                                    (95 * len(ordered) + 99) // 100 - 1)] / 1_000_000, 6),
            "min_ms": round(ordered[0] / 1_000_000, 6),
            "max_ms": round(ordered[-1] / 1_000_000, 6)}


def run_experiment(repetitions: int):
    if type(repetitions) is not int or not 10 <= repetitions <= 500:
        raise ValueError("repetitions must be 10..500")
    if not (LOWER / SOURCE_RELATIVE).is_file():
        raise RuntimeError("Pinned Boltons source is missing")
    source_before = hashlib.sha256((LOWER / SOURCE_RELATIVE).read_bytes()).hexdigest()
    from boltons.strutils import singularize  # warmed before any fork
    baseline = (singularize("glass"), singularize("glasses"))
    if baseline != ("glas", "glass"):
        raise RuntimeError("Expected known failing Boltons baseline")
    libc = _mount_api()
    pool = Path(tempfile.mkdtemp(prefix="fpb-cow-", dir="/"))
    mounted = []
    try:
        views = {}
        for name in ("fix", "baseline"):
            upper, work, merged = (pool / part / name for part in ("upper", "work", "merged"))
            _mount_overlay(libc, LOWER, upper, work, merged)
            mounted.append(merged)
            views[name] = merged
        # Both children inherit the same prefork Python heap. Their writes go
        # to different overlay upperdirs, so each reads a separate file view.
        fix_pid, fix_reader = _fork_json(lambda: _child_result("fix", views["fix"]))
        base_pid, base_reader = _fork_json(lambda: _child_result("baseline", views["baseline"]))
        results = {}
        errors = []
        for name, pid, reader in (("fix", fix_pid, fix_reader),
                                  ("baseline", base_pid, base_reader)):
            try:
                results[name] = _receive_json(pid, reader)
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise errors[0]
        fix, untouched = results["fix"], results["baseline"]
        if (fix["glass"], fix["glasses"], untouched["glass"], untouched["glasses"]) != (
                "glass", "glass", "glas", "glass"):
            raise RuntimeError("Branch outputs do not match the pinned baseline and fix")
        if (fix["inherited_counter"], untouched["inherited_counter"],
                WARM_STATE["counter"]) != (7, 7, 7):
            raise RuntimeError("Prefork Python heap state leaked across branches")
        if fix["source_sha256"] == untouched["source_sha256"]:
            raise RuntimeError("Overlay file views were not independent")
        if NEW not in (views["fix"] / SOURCE_RELATIVE).read_text(encoding="utf-8"):
            raise RuntimeError("Repaired overlay lost its patch")
        if NEW in (views["baseline"] / SOURCE_RELATIVE).read_text(encoding="utf-8"):
            raise RuntimeError("Baseline overlay observed sibling patch")
        while mounted:
            _unmount_overlay(libc, mounted[-1])
            mounted.pop()

        create_ns, fork_ready_ns, reap_ns, rollback_ns = [], [], [], []
        for index in range(repetitions):
            name = "perf"
            upper, work, merged = (pool / part / name for part in ("upper", "work", "merged"))
            create_started = time.perf_counter_ns()
            _mount_overlay(libc, LOWER, upper, work, merged)
            mounted.append(merged)
            create_ns.append(time.perf_counter_ns() - create_started)
            marker = merged / (".fpb-cow-marker-" + str(index))
            if marker.exists():
                raise RuntimeError("Fresh branch retained a prior overlay write")
            ready, reap = _fork_ready_and_write(marker)
            fork_ready_ns.append(ready)
            reap_ns.append(reap)
            if WARM_STATE["counter"] != 7:
                raise RuntimeError("Child heap mutation leaked to parent")
            rollback_started = time.perf_counter_ns()
            _unmount_overlay(libc, merged)
            mounted.pop()
            for path in (upper, work, merged):
                shutil.rmtree(path)
            rollback_ns.append(time.perf_counter_ns() - rollback_started)
            if (LOWER / marker.name).exists():
                raise RuntimeError("Child overlay write leaked to lowerdir")
        if hashlib.sha256((LOWER / SOURCE_RELATIVE).read_bytes()).hexdigest() != source_before:
            raise RuntimeError("Shared lower source changed")
        combined_ns = [a + b + c + d for a, b, c, d in
                       zip(create_ns, fork_ready_ns, reap_ns, rollback_ns)]
        rollback_total_ns = [a + b for a, b in zip(reap_ns, rollback_ns)]
        return {"kind": "guest_linux_fork_overlayfs_microbenchmark_v1",
                "scope": "trusted guest-local primitive inside one already-running ARM64 QEMU/HVF VM",
                "repetitions": repetitions,
                "correctness": {"parent_glass": baseline[0], "fix_branch": fix,
                                "baseline_branch": untouched, "parent_counter": WARM_STATE["counter"],
                                "shared_lower_sha256_unchanged": True,
                                "isolated_overlay_writes": True},
                "timings": {"overlay_create": _summarize_ns(create_ns),
                            "fork_ready": _summarize_ns(fork_ready_ns),
                            "process_reap": _summarize_ns(reap_ns),
                            "overlay_rollback": _summarize_ns(rollback_ns),
                            "rollback_total": _summarize_ns(rollback_total_ns),
                            "combined_branch_cycle": _summarize_ns(combined_ns)}}
    finally:
        failed_unmount = False
        while mounted:
            try:
                _unmount_overlay(libc, mounted[-1])
                mounted.pop()
            except OSError:
                failed_unmount = True
                break
        # Never recurse through a still-mounted overlay: that could delete
        # files in the lower workspace. The VM disk is disposable if a guest
        # unmount fails, so preserve the directory for VM teardown instead.
        if not failed_unmount:
            shutil.rmtree(pool)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        raise SystemExit("usage: guest_cow_branch.py REPETITIONS")
    report = run_experiment(int(argv[0]))
    encoded = base64.b64encode(json.dumps(report, separators=(",", ":")).encode()).decode()
    print("FPB_COW_RESULT:" + encoded, flush=True)


if __name__ == "__main__":
    main()
