"""Trusted guest experiment: forked heaps with private Linux mount views.

Each child unshares CLONE_NEWNS, makes mount propagation private, mounts a
private tmpfs for overlayfs upper/work, and mounts the resulting view at the
same pathname. The parent and sibling do not inherit that view. This shares a
kernel and root authority, so it is not an adversarial agent sandbox.
"""

from __future__ import annotations

import base64
import ctypes
import hashlib
import importlib.util
import json
import os
import select
import shutil
import sys
import tempfile
import time
from pathlib import Path


LOWER = Path("/workspace")
SOURCE_RELATIVE = Path("boltons/strutils.py")
OLD = "    else:\n        singular = word[:-1]\n    return _match_case(orig_word, singular)"
NEW = "    elif word.endswith('ss'):\n        singular = word\n" + OLD
CLONE_NEWNS = 0x00020000
MS_REC = 0x4000
MS_PRIVATE = 0x40000
MS_NOSUID = 0x2
MS_NODEV = 0x4
MS_NOEXEC = 0x8
WARM_STATE = {"counter": 7}


def _mount_api():
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.unshare.argtypes = [ctypes.c_int]
    libc.unshare.restype = ctypes.c_int
    libc.mount.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
                           ctypes.c_ulong, ctypes.c_char_p]
    libc.mount.restype = ctypes.c_int
    libc.umount2.argtypes = [ctypes.c_char_p, ctypes.c_int]
    libc.umount2.restype = ctypes.c_int
    return libc


def _check_syscall(value: int, label: str):
    if value != 0:
        raise OSError(ctypes.get_errno(), label)


def _setup_private_view(libc, upper_root: Path, view: Path):
    """Return guest-clock durations; called only after fork in a child."""
    started = time.perf_counter_ns()
    _check_syscall(libc.unshare(CLONE_NEWNS), "unshare(CLONE_NEWNS)")
    unshare_ns = time.perf_counter_ns() - started
    started = time.perf_counter_ns()
    _check_syscall(libc.mount(None, b"/", None, MS_REC | MS_PRIVATE, None),
                   "mount propagation private")
    private_ns = time.perf_counter_ns() - started
    started = time.perf_counter_ns()
    _check_syscall(libc.mount(b"tmpfs", os.fsencode(upper_root), b"tmpfs",
                              MS_NOSUID | MS_NODEV | MS_NOEXEC,
                              b"size=64m,mode=0700"),
                   "mount private upper tmpfs")
    tmpfs_ns = time.perf_counter_ns() - started
    (upper_root / "upper").mkdir()
    (upper_root / "work").mkdir()
    started = time.perf_counter_ns()
    options = (f"lowerdir={LOWER},upperdir={upper_root / 'upper'},"
               f"workdir={upper_root / 'work'}").encode()
    _check_syscall(libc.mount(b"overlay", os.fsencode(view), b"overlay", 0, options),
                   "mount private overlay")
    overlay_ns = time.perf_counter_ns() - started
    return {"unshare_ns": unshare_ns, "private_propagation_ns": private_ns,
            "tmpfs_mount_ns": tmpfs_ns, "overlay_mount_ns": overlay_ns}


def _exercise(mode: str, view: Path, marker_name: str):
    if WARM_STATE["counter"] != 7:
        raise RuntimeError("Pre-fork heap state changed")
    WARM_STATE["counter"] = 11 if mode == "fix" else 12
    source = view / SOURCE_RELATIVE
    original = source.read_text(encoding="utf-8")
    if mode == "fix":
        if original.count(OLD) != 1:
            raise RuntimeError("Pinned repair anchor differs")
        source.write_text(original.replace(OLD, NEW), encoding="utf-8")
    elif mode not in ("baseline", "marker"):
        raise ValueError("Unknown branch mode")
    marker = view / marker_name
    if marker.exists():
        raise RuntimeError("New overlay retained a previous marker")
    marker.write_text(mode, encoding="ascii")
    if mode == "marker":
        # Keep the repeated timing focused on branch construction and teardown.
        # The two sibling correctness branches load the changed source below.
        from boltons.strutils import singularize
    else:
        spec = importlib.util.spec_from_file_location("fpb_namespace_strutils", str(source))
        if spec is None or spec.loader is None:
            raise RuntimeError("Cannot load branch source")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        singularize = module.singularize
    return {"glass": singularize("glass"),
            "glasses": singularize("glasses"),
            "inherited_counter": 7, "branch_counter": WARM_STATE["counter"],
            "marker_visible": marker.read_text(encoding="ascii") == mode,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "mount_namespace": os.readlink("/proc/self/ns/mnt"),
            "upper_device": os.stat(view.parent / "upper").st_dev,
            "overlay_device": os.stat(view).st_dev}


def _write_result(fd: int, payload: dict):
    data = json.dumps(payload, separators=(",", ":")).encode() + b"\n"
    if len(data) > 8192:
        raise RuntimeError("Child result exceeds bound")
    os.write(fd, data)


def _child(mode: str, upper_root: Path, view: Path, marker_name: str,
           ready_writer: int, control_reader: int, result_writer: int):
    libc = _mount_api()
    tmpfs_mounted = overlay_mounted = False
    payload: dict = {"ok": False}
    status = 1
    try:
        timings = _setup_private_view(libc, upper_root, view)
        tmpfs_mounted = overlay_mounted = True
        value = _exercise(mode, view, marker_name)
        os.write(ready_writer, b"R")
        if os.read(control_reader, 1) != b"X":
            raise RuntimeError("Parent did not release child")
        payload = {"ok": True, "value": value, "timings_ns": timings}
        status = 0
    except BaseException as exc:
        os.write(ready_writer, b"E")
        payload = {"ok": False, "error_type": type(exc).__name__,
                   "error": str(exc)[:500]}
    finally:
        started = time.perf_counter_ns()
        try:
            if overlay_mounted:
                _check_syscall(libc.umount2(os.fsencode(view), 0), "unmount private overlay")
            if tmpfs_mounted:
                _check_syscall(libc.umount2(os.fsencode(upper_root), 0),
                               "unmount private tmpfs")
        except BaseException as exc:
            payload = {"ok": False, "error_type": type(exc).__name__,
                       "error": "cleanup: " + str(exc)[:500]}
            status = 1
        payload["cleanup_mounts_ns"] = time.perf_counter_ns() - started
        try:
            _write_result(result_writer, payload)
        finally:
            for fd in (ready_writer, control_reader, result_writer):
                os.close(fd)
            os._exit(status)


class _Branch:
    def __init__(self, mode: str, upper_root: Path, view: Path, marker_name: str):
        ready_reader, ready_writer = os.pipe()
        control_reader, control_writer = os.pipe()
        result_reader, result_writer = os.pipe()
        self.started_ns = time.perf_counter_ns()
        pid = os.fork()
        if pid == 0:
            for fd in (ready_reader, control_writer, result_reader):
                os.close(fd)
            _child(mode, upper_root, view, marker_name,
                   ready_writer, control_reader, result_writer)
        self.pid = pid
        for fd in (ready_writer, control_reader, result_writer):
            os.close(fd)
        self.ready_reader = ready_reader
        self.control_writer = control_writer
        self.result_reader = result_reader
        self.reaped = False

    def ready(self):
        if not select.select([self.ready_reader], [], [], 10)[0]:
            raise TimeoutError("Namespace branch did not become ready")
        status = os.read(self.ready_reader, 1)
        self.ready_ns = time.perf_counter_ns() - self.started_ns
        if status != b"R":
            raise RuntimeError("Namespace branch setup failed")

    def finish(self):
        started = time.perf_counter_ns()
        os.write(self.control_writer, b"X")
        os.close(self.control_writer)
        self.control_writer = -1
        blocks = []
        size = 0
        deadline = time.monotonic() + 10
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.result_reader], [], [], remaining)[0]:
                raise TimeoutError("Namespace branch result timed out")
            block = os.read(self.result_reader, 4096)
            if not block:
                break
            size += len(block)
            if size > 8192:
                raise RuntimeError("Namespace branch result exceeds bound")
            blocks.append(block)
        _, code = os.waitpid(self.pid, 0)
        self.reaped = True
        release_to_reap_ns = time.perf_counter_ns() - started
        payload = json.loads(b"".join(blocks))
        if not os.WIFEXITED(code) or os.WEXITSTATUS(code) != 0 or not payload.get("ok"):
            raise RuntimeError("Namespace branch failed: " + repr(payload))
        payload["fork_to_ready_ns"] = self.ready_ns
        payload["release_to_reap_ns"] = release_to_reap_ns
        payload["complete_cycle_ns"] = time.perf_counter_ns() - self.started_ns
        return payload

    def close(self):
        for name in ("ready_reader", "control_writer", "result_reader"):
            fd = getattr(self, name)
            if fd >= 0:
                os.close(fd)
                setattr(self, name, -1)
        if not self.reaped:
            try:
                os.kill(self.pid, 9)
            except ProcessLookupError:
                pass
            os.waitpid(self.pid, 0)
            self.reaped = True


def _summary(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    median = (ordered[middle] if len(ordered) % 2 else
              (ordered[middle - 1] + ordered[middle]) / 2)
    p95 = ordered[min(len(ordered) - 1, (95 * len(ordered) + 99) // 100 - 1)]
    return {"count": len(values), "median_ms": round(median / 1e6, 6),
            "p95_ms": round(p95 / 1e6, 6),
            "min_ms": round(ordered[0] / 1e6, 6),
            "max_ms": round(ordered[-1] / 1e6, 6)}


def run_experiment(repetitions: int):
    if type(repetitions) is not int or not 10 <= repetitions <= 500:
        raise ValueError("repetitions must be 10..500")
    source = LOWER / SOURCE_RELATIVE
    if not source.is_file():
        raise RuntimeError("Pinned Boltons source is missing")
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    from boltons.strutils import singularize  # warm imports and heap before fork
    if (singularize("glass"), singularize("glasses")) != ("glas", "glass"):
        raise RuntimeError("Expected pinned failing Boltons baseline")
    parent_ns = os.readlink("/proc/self/ns/mnt")
    pool = Path(tempfile.mkdtemp(prefix="fpb-mount-ns-", dir="/"))
    upper_root, view = pool / "upper", pool / "view"
    upper_root.mkdir()
    view.mkdir()
    parent_upper_device = os.stat(upper_root).st_dev
    parent_overlay_device = os.stat(view).st_dev
    live = []
    try:
        # Both children see the same pathname, but a different overlay mount.
        sibling_results = {}
        for mode in ("fix", "baseline"):
            live.append(_Branch(mode, upper_root, view, ".fpb-sibling-marker"))
        for branch in live:
            branch.ready()
        if list(view.iterdir()) or list(upper_root.iterdir()):
            raise RuntimeError("Child mount or upper files leaked into parent mount view")
        for mode, branch in zip(("fix", "baseline"), live):
            sibling_results[mode] = branch.finish()["value"]
        for branch in live:
            branch.close()
        live.clear()
        if ((LOWER / ".fpb-sibling-marker").exists()
                or list(view.iterdir()) or list(upper_root.iterdir())):
            raise RuntimeError("Sibling writes or mounts survived in parent view")
        fix, baseline = sibling_results["fix"], sibling_results["baseline"]
        if (fix["glass"], baseline["glass"], fix["glasses"], baseline["glasses"]) != (
                "glass", "glas", "glass", "glass"):
            raise RuntimeError("Sibling outcomes differ from pinned repair and baseline")
        if (fix["upper_device"] == baseline["upper_device"]
                or fix["upper_device"] == parent_upper_device
                or baseline["upper_device"] == parent_upper_device
                or fix["overlay_device"] == parent_overlay_device
                or baseline["overlay_device"] == parent_overlay_device):
            raise RuntimeError("Child mount views were not private")
        if (fix["mount_namespace"] == baseline["mount_namespace"]
                or fix["mount_namespace"] == parent_ns
                or baseline["mount_namespace"] == parent_ns):
            raise RuntimeError("Mount namespace identifiers were not distinct")
        if fix["source_sha256"] == baseline["source_sha256"]:
            raise RuntimeError("Sibling overlay file contents were not independent")
        if not fix["marker_visible"] or not baseline["marker_visible"]:
            raise RuntimeError("Child marker was not visible within branch")
        fields = ("unshare_ns", "private_propagation_ns", "tmpfs_mount_ns",
                  "overlay_mount_ns", "cleanup_mounts_ns", "fork_to_ready_ns",
                  "release_to_reap_ns", "complete_cycle_ns")
        samples = {field: [] for field in fields}
        for index in range(repetitions):
            name = ".fpb-cycle-" + str(index)
            branch = _Branch("marker", upper_root, view, name)
            live.append(branch)
            branch.ready()
            if list(view.iterdir()) or list(upper_root.iterdir()) or (LOWER / name).exists():
                raise RuntimeError("Branch mount or write appeared in parent view")
            result = branch.finish()
            branch.close()
            live.pop()
            if (LOWER / name).exists() or list(view.iterdir()) or list(upper_root.iterdir()):
                raise RuntimeError("Completed branch leaked writes or mounts to parent")
            value = result["value"]
            if value["glass"] != "glas" or not value["marker_visible"]:
                raise RuntimeError("Fresh namespace branch lost baseline or marker")
            for field in fields:
                samples[field].append(result["timings_ns"].get(field, result.get(field)))
        if (hashlib.sha256(source.read_bytes()).hexdigest() != before
                or WARM_STATE["counter"] != 7):
            raise RuntimeError("Parent lower source or warmed heap changed")
        return {"kind": "guest_linux_mount_namespace_overlayfs_microbenchmark_v1",
                "scope": "trusted guest-local fork with per-child mount namespace and tmpfs upper",
                "repetitions": repetitions,
                "correctness": {"parent_namespace": parent_ns,
                                "parent_upper_device": parent_upper_device,
                                "parent_overlay_device": parent_overlay_device,
                                "namespace_verification":
                                    "distinct /proc mount-namespace identifiers and tmpfs devices; empty parent mountpoint",
                                "fix_branch": fix, "baseline_branch": baseline,
                                "parent_view_empty": True,
                                "distinct_child_mount_namespaces": True,
                                "shared_lower_sha256_unchanged": True,
                                "parent_heap_unchanged": True},
                "timings": {field: _summary(values) for field, values in samples.items()}}
    finally:
        for branch in live:
            branch.close()
        # Child mounts belong to their own namespaces and disappear at exit.
        # The parent sees only the empty underlying mountpoint directories.
        if not list(view.iterdir()) and not list(upper_root.iterdir()):
            shutil.rmtree(pool)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        raise SystemExit("usage: guest_mount_namespace.py REPETITIONS")
    report = run_experiment(int(argv[0]))
    encoded = base64.b64encode(json.dumps(report, separators=(",", ":")).encode()).decode()
    print("FPB_MOUNT_NS_RESULT:" + encoded, flush=True)


if __name__ == "__main__":
    main()
