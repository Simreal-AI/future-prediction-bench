"""Trusted, cooperative process-plus-filesystem checkpoint inside one Linux VM.

This is a deliberately narrow DeltaBox-inspired experiment.  An idle Python
template owns one frozen overlay workspace and forks a child for each restore.
Each child creates a mount-namespace-local nested overlay before doing work.
It does not capture arbitrary processes, writable descriptors, or the kernel.
"""

from __future__ import annotations

import base64
import ctypes
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path


SOURCE = Path("boltons/strutils.py")
WORKSPACE = Path("/workspace")
OLD = "    else:\n        singular = word[:-1]\n    return _match_case(orig_word, singular)"
NEW = "    elif word.endswith('ss'):\n        singular = word\n" + OLD
CLONE_NEWNS = 0x00020000
MS_RDONLY = 1
MS_NOSUID = 2
MS_NODEV = 4
MS_NOEXEC = 8
MS_REMOUNT = 32
MS_BIND = 4096
MS_REC = 16384
MS_PRIVATE = 1 << 18
MNT_DETACH = 2
STATE = {"turn": 0, "prefix": "uninitialized"}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tree_digest(root):
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError("workspace symlink outside pinned contract")
        details = path.stat(follow_symlinks=False)
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(stat.S_IMODE(details.st_mode).to_bytes(2, "big"))
        if stat.S_ISDIR(details.st_mode):
            digest.update(b"D")
        elif stat.S_ISREG(details.st_mode):
            digest.update(b"F")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        else:
            raise RuntimeError("unsupported workspace node")
    return digest.hexdigest()


def checked_number(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(name)
    return value


def ns_stats(values):
    if not values or any(type(value) is not int or value < 0 for value in values):
        raise ValueError("invalid timing samples")
    ordered = sorted(values)
    middle = (ordered[(len(ordered) - 1) // 2] + ordered[len(ordered) // 2]) / 2
    nearest95 = ordered[(95 * len(ordered) + 99) // 100 - 1]
    return {"n": len(values), "p50_ms": round(middle / 1_000_000, 6),
            "p95_ms": round(nearest95 / 1_000_000, 6),
            "min_ms": round(ordered[0] / 1_000_000, 6),
            "max_ms": round(ordered[-1] / 1_000_000, 6)}


def _libc():
    lib = ctypes.CDLL("libc.so.6", use_errno=True)
    lib.unshare.argtypes = [ctypes.c_int]
    lib.unshare.restype = ctypes.c_int
    lib.mount.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
                          ctypes.c_ulong, ctypes.c_char_p]
    lib.mount.restype = ctypes.c_int
    lib.umount2.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.umount2.restype = ctypes.c_int
    return lib


def _call(result, name):
    if result != 0:
        raise OSError(ctypes.get_errno(), name)


def _mount(lib, source, target, kind=None, flags=0, options=None):
    _call(lib.mount(source, os.fsencode(target), kind, flags, options),
          "mount " + str(target))


def _umount(lib, target):
    _call(lib.umount2(os.fsencode(target), MNT_DETACH), "umount " + str(target))


def _new_mount_namespace(lib):
    _call(lib.unshare(CLONE_NEWNS), "unshare mount namespace")
    _mount(lib, None, "/", flags=MS_REC | MS_PRIVATE)


def _overlay(lib, lower, upper, work, merged):
    for path in (upper, work, merged):
        path.mkdir(parents=True, exist_ok=False)
    options = f"lowerdir={lower},upperdir={upper},workdir={work}".encode("ascii")
    _mount(lib, b"overlay", merged, b"overlay", 0, options)


def _write_all(fd, data):
    while data:
        written = os.write(fd, data)
        if written <= 0:
            raise RuntimeError("pipe write failed")
        data = data[written:]


def _read_all(fd, limit=131072):
    chunks = []
    total = 0
    while True:
        block = os.read(fd, 8192)
        if not block:
            break
        total += len(block)
        if total > limit:
            raise RuntimeError("child report exceeded bound")
        chunks.append(block)
    return b"".join(chunks)


def _run_pinned_cases(view, codes):
    results = []
    for code in codes:
        if type(code) is not str or not 0 < len(code) <= 2048:
            raise ValueError("invalid pinned case")
        env = dict(os.environ, PYTHONPATH=str(view), PYTHONDONTWRITEBYTECODE="1")
        completed = subprocess.run([sys.executable, "-I", "-B", "-c",
                                    "import sys; sys.path.insert(0, " + repr(str(view))
                                    + "); exec(" + repr(code) + ")"],
                                   cwd=view, env=env, capture_output=True,
                                   timeout=10, check=False)
        if len(completed.stdout) > 8192 or len(completed.stderr) > 8192:
            raise RuntimeError("pinned case output exceeded bound")
        results.append({"return_code": completed.returncode,
                        "stdout_b64": base64.b64encode(completed.stdout).decode("ascii")})
    return results


def _branch_body(lib, pool, frozen, cycle, index, codes, seed_sha, ready_fd):
    _new_mount_namespace(lib)
    branch = pool / f"branch-{cycle}-{index}"
    branch.mkdir()
    upper, work, view = branch / "upper", branch / "work", branch / "view"
    _overlay(lib, frozen, upper, work, view)
    branch_mount_namespace = os.readlink(pool / "proc/self/ns/mnt")
    _write_all(ready_fd, b"R")
    os.close(ready_fd)
    if STATE != {"turn": 17, "prefix": "checkpointed"}:
        raise RuntimeError("process state did not restore from checkpoint")
    prefix = view / ".fpb-checkpoint-prefix"
    if prefix.read_text(encoding="ascii") != "prefix-committed":
        raise RuntimeError("filesystem prefix missing from checkpoint")
    for prior in range(index):
        if (view / f".fpb-branch-{cycle}-{prior}").exists():
            raise RuntimeError("sibling filesystem state leaked")
    if sha((view / SOURCE).read_bytes()) != seed_sha:
        raise RuntimeError("branch did not inherit checkpoint source")
    (view / f".fpb-branch-{cycle}-{index}").write_text("branch-only", encoding="ascii")
    mode = "repair" if index % 2 == 0 else "baseline"
    STATE["turn"] = 111 if mode == "repair" else 222
    source = view / SOURCE
    if mode == "repair":
        text = source.read_text(encoding="utf-8")
        if text.count(OLD) != 1:
            raise RuntimeError("pinned repair anchor changed")
        source.write_text(text.replace(OLD, NEW), encoding="utf-8")
    final = source.read_bytes()
    if (NEW.encode() in final) != (mode == "repair"):
        raise RuntimeError("branch repair was not isolated")
    result = {"mode": mode, "inherited_turn": 17,
              "local_turn": STATE["turn"], "source_sha256": sha(final),
              "prefix_seen": True, "sibling_markers_absent": True,
              "mount_namespace": branch_mount_namespace}
    if cycle == 0 and index < 2:
        result["case_results"] = _run_pinned_cases(view, codes)
    return result


def _restore_one(lib, pool, frozen, cycle, index, codes, seed_sha):
    ready_read, ready_write = os.pipe()
    report_read, report_write = os.pipe()
    started = time.perf_counter_ns()
    pid = os.fork()
    if pid == 0:
        os.close(ready_read)
        os.close(report_read)
        try:
            value = _branch_body(lib, pool, frozen, cycle, index, codes,
                                 seed_sha, ready_write)
            payload = {"ok": True, "value": value}
            status = 0
        except BaseException as error:
            payload = {"ok": False, "error": type(error).__name__ + ": "
                       + str(error)[:300]}
            status = 1
        try:
            _write_all(report_write, json.dumps(payload, separators=(",", ":")).encode())
        finally:
            os.close(report_write)
        os._exit(status)
    os.close(ready_write)
    os.close(report_write)
    try:
        ready = os.read(ready_read, 1)
        restore_ns = time.perf_counter_ns() - started
        payload = json.loads(_read_all(report_read))
        _, status = os.waitpid(pid, 0)
    finally:
        os.close(ready_read)
        os.close(report_read)
    if ready != b"R" or not os.WIFEXITED(status) or os.WEXITSTATUS(status) != 0:
        raise RuntimeError("branch failed before or after restore: " + repr(payload))
    if not payload.get("ok"):
        raise RuntimeError("branch failed: " + repr(payload))
    if (STATE != {"turn": 17, "prefix": "checkpointed"}
            or sha((frozen / SOURCE).read_bytes()) != seed_sha):
        raise RuntimeError("template process or frozen source mutated")
    shutil.rmtree(pool / f"branch-{cycle}-{index}")
    return restore_ns, time.perf_counter_ns() - started, payload["value"]


def _template_loop(lib, pool, frozen, cycle, restores, codes, seed_sha, ready_fd):
    if STATE != {"turn": 17, "prefix": "checkpointed"}:
        raise RuntimeError("template heap is not checkpointed")
    _write_all(ready_fd, b"T")
    os.close(ready_fd)
    samples = []
    branch_cycles = []
    branches = []
    template_mount_namespace = os.readlink(pool / "proc/self/ns/mnt")
    for index in range(restores):
        duration, cycle_ns, branch = _restore_one(lib, pool, frozen, cycle, index,
                                                  codes, seed_sha)
        if branch["mount_namespace"] == template_mount_namespace:
            raise RuntimeError("branch inherited template mount namespace")
        samples.append(duration)
        branch_cycles.append(cycle_ns)
        branches.append(branch)
    if STATE != {"turn": 17, "prefix": "checkpointed"}:
        raise RuntimeError("template heap changed after restores")
    return {"restore_ns": samples, "branch_cycle_ns": branch_cycles,
            "template_mount_namespace": template_mount_namespace,
            "branches": branches,
            "template_turn_after_restores": STATE["turn"]}


def _cycle(lib, cycle, restores, codes, seed_sha):
    pool = Path(tempfile.mkdtemp(prefix="fpb-coupled-", dir="/"))
    mounted_pool = False
    mounted_frozen = False
    try:
        _mount(lib, b"tmpfs", pool, b"tmpfs", MS_NOSUID | MS_NODEV,
               b"size=64m,mode=0700")
        mounted_pool = True
        proc = pool / "proc"
        proc.mkdir()
        _mount(lib, b"proc", proc, b"proc", MS_NOSUID | MS_NODEV | MS_NOEXEC)
        lower = pool / "lower"
        lower.mkdir()
        _mount(lib, os.fsencode(WORKSPACE), lower, flags=MS_BIND | MS_REC)
        _mount(lib, None, lower, flags=MS_BIND | MS_REMOUNT | MS_RDONLY)
        frozen = pool / "frozen"
        _overlay(lib, lower, pool / "stage-upper", pool / "stage-work", frozen)
        mounted_frozen = True
        STATE.update(turn=17, prefix="checkpointed")
        (frozen / ".fpb-checkpoint-prefix").write_text("prefix-committed", encoding="ascii")
        if sha((frozen / SOURCE).read_bytes()) != seed_sha:
            raise RuntimeError("checkpoint prefix altered source")
        ready_read, ready_write = os.pipe()
        report_read, report_write = os.pipe()
        checkpoint_started = time.perf_counter_ns()
        _mount(lib, None, frozen, flags=MS_REMOUNT | MS_RDONLY)
        pid = os.fork()
        if pid == 0:
            os.close(ready_read)
            os.close(report_read)
            try:
                value = _template_loop(lib, pool, frozen, cycle, restores,
                                       codes, seed_sha, ready_write)
                payload = {"ok": True, "value": value}
                status = 0
            except BaseException as error:
                payload = {"ok": False, "error": type(error).__name__ + ": "
                           + str(error)[:300]}
                status = 1
            try:
                _write_all(report_write, json.dumps(payload, separators=(",", ":")).encode())
            finally:
                os.close(report_write)
            os._exit(status)
        os.close(ready_write)
        os.close(report_write)
        try:
            ready = os.read(ready_read, 1)
            checkpoint_ns = time.perf_counter_ns() - checkpoint_started
            STATE.update(turn=999, prefix="parent-after-checkpoint")
            payload = json.loads(_read_all(report_read, limit=262144))
            _, status = os.waitpid(pid, 0)
        finally:
            os.close(ready_read)
            os.close(report_read)
        if ready != b"T" or not os.WIFEXITED(status) or os.WEXITSTATUS(status) != 0:
            raise RuntimeError("template failed: " + repr(payload))
        if STATE != {"turn": 999, "prefix": "parent-after-checkpoint"}:
            raise RuntimeError("live parent heap did not diverge from template")
        if not payload.get("ok"):
            raise RuntimeError("template failed: " + repr(payload))
        if sha((frozen / SOURCE).read_bytes()) != seed_sha:
            raise RuntimeError("frozen source changed")
        if (frozen / ".fpb-checkpoint-prefix").read_text(encoding="ascii") != "prefix-committed":
            raise RuntimeError("frozen prefix changed")
        return {"checkpoint_ns": checkpoint_ns, **payload["value"],
                "parent_turn_after_checkpoint": STATE["turn"],
                "frozen_prefix_unchanged": True, "frozen_source_unchanged": True}
    finally:
        if mounted_frozen:
            _umount(lib, pool / "frozen")
        if mounted_pool:
            _umount(lib, pool)
        shutil.rmtree(pool)


def experiment(cycles, restores):
    checked_number(cycles, 2, 30, "cycles must be 2..30")
    checked_number(restores, 2, 100, "restores must be 2..100")
    case_file = Path("/fpb_coupled_cases.json")
    codes = json.loads(case_file.read_text(encoding="utf-8"))
    if not isinstance(codes, list) or len(codes) != 14:
        raise ValueError("expected exactly 14 pinned case programs")
    seed_sha = sha((WORKSPACE / SOURCE).read_bytes())
    if (WORKSPACE / SOURCE).read_text(encoding="utf-8").count(OLD) != 1:
        raise ValueError("pinned seed source changed")
    outer_tree_before = _tree_digest(WORKSPACE)
    lib = _libc()
    _new_mount_namespace(lib)
    results = [_cycle(lib, index, restores, codes, seed_sha)
               for index in range(cycles)]
    checkpoints = [row["checkpoint_ns"] for row in results]
    restore_ns = [value for row in results for value in row["restore_ns"]]
    no_case_branch_cycles = [value for cycle, row in enumerate(results)
                             for index, value in enumerate(row["branch_cycle_ns"])
                             if cycle != 0 or index >= 2]
    if sha((WORKSPACE / SOURCE).read_bytes()) != seed_sha:
        raise RuntimeError("outer ext4 source changed")
    outer_tree_after = _tree_digest(WORKSPACE)
    if outer_tree_before != outer_tree_after:
        raise RuntimeError("outer ext4 workspace tree changed")
    return {"kind": "cooperative_coupled_fork_overlay_checkpoint_v1",
            "scope": "trusted one-process template inside one booted Linux QEMU guest",
            "cycles": cycles, "restores_per_cycle": restores,
            "seed_source_sha256": seed_sha,
            "checkpoint_ns": checkpoints, "restore_ns": restore_ns,
            "no_case_branch_cycle_ns": no_case_branch_cycles,
            "checkpoint_stats": ns_stats(checkpoints),
            "restore_stats": ns_stats(restore_ns),
            "no_case_branch_cycle_stats": ns_stats(no_case_branch_cycles),
            "cycles_data": results,
            "outer_ext4_source_unchanged": True,
            "outer_ext4_tree_sha256": outer_tree_after,
            "outer_ext4_tree_unchanged": True}


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: guest_coupled.py CYCLES RESTORES")
    report = experiment(int(sys.argv[1]), int(sys.argv[2]))
    encoded = base64.b64encode(json.dumps(report, separators=(",", ":")).encode()).decode()
    print("FPB_COUPLED_RESULT:" + encoded)


if __name__ == "__main__":
    main()
