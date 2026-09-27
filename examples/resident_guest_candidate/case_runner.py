"""Trusted nested verifier-case runner for the resident episode branch.

Each host-authored Python case runs as uid/gid 65534 in a fresh nested
mount/PID namespace. Its overlay lower is the *submitted resident branch*;
its tmpfs upper is discarded after the case. No expected output enters here.
This helper is experimental; the pinned Boltons smoke exercises it in QEMU.
"""

from __future__ import annotations

import base64
import ctypes
import os
import resource
import select
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from wire import MAX_RESPONSE, read_frame, send_frame


CLONE_NEWNS = 0x00020000
CLONE_NEWPID = 0x20000000
MS_RDONLY = 1
MS_REMOUNT = 32
MS_BIND = 4096
MS_REC = 16384
MS_PRIVATE = 1 << 18
PR_SET_PDEATHSIG = 1
MAX_OUTPUT = 12000
CASE_SECONDS = 10.0
CASE_SUPERVISOR_SECONDS = 15.0
# The verifier is an unprivileged Python process in its own PID namespace.
# These inherited hard limits keep one case from exhausting the 512 MiB guest.
# RLIMIT_AS and RLIMIT_NPROC apply per process / real UID, respectively; the
# 64 MiB episode tmpfs is the separate aggregate workspace-write bound.
CASE_RESOURCE_LIMITS = (
    (resource.RLIMIT_CPU, 8),
    (resource.RLIMIT_AS, 128 * 1024 * 1024),
    (resource.RLIMIT_NPROC, 2),
    (resource.RLIMIT_FSIZE, 8 * 1024 * 1024),
    (resource.RLIMIT_NOFILE, 64),
    (resource.RLIMIT_CORE, 0),
)


def _libc():
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.unshare.argtypes = [ctypes.c_int]
    libc.unshare.restype = ctypes.c_int
    libc.mount.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
                           ctypes.c_ulong, ctypes.c_char_p]
    libc.mount.restype = ctypes.c_int
    libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong,
                           ctypes.c_ulong, ctypes.c_ulong]
    libc.prctl.restype = ctypes.c_int
    return libc


def _checked(result, operation):
    if result != 0:
        raise OSError(ctypes.get_errno(), operation)


def _mount(libc, source, target, filesystem=None, flags=0, options=None):
    _checked(libc.mount(source, os.fsencode(target), filesystem, flags, options),
             "mount " + str(target))


def _set_case_limits(limits=CASE_RESOURCE_LIMITS):
    """Apply hard caps in the forked child before candidate code can run."""
    for kind, requested in limits:
        soft, hard = resource.getrlimit(kind)
        # Never raise an inherited operator limit. A candidate cannot raise
        # its resulting hard limit after the uid/gid drop.
        cap = requested
        for inherited in (soft, hard):
            if inherited != resource.RLIM_INFINITY:
                cap = min(cap, inherited)
        resource.setrlimit(kind, (cap, cap))


def _capture(code):
    tmp = "/workspace/.fpb-resident-case-tmp"
    os.mkdir(tmp, 0o1777)
    os.chmod(tmp, 0o1777)
    env = os.environ.copy()
    env.update({"PYTHONPATH": "/workspace", "PYTHONDONTWRITEBYTECODE": "1",
                "TMPDIR": tmp, "HOME": tmp})
    candidate = subprocess.Popen(
        [sys.executable, "-B", "-c", code], cwd="/workspace", env=env,
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        start_new_session=True, user=65534, group=65534, extra_groups=[],
        preexec_fn=_set_case_limits)
    return _collect(candidate)


def _collect(candidate):
    """Bound the whole candidate lifetime, including after stdout closes."""
    output = bytearray()
    deadline = time.monotonic() + CASE_SECONDS
    return_code = None
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return_code = 124
                break
            if not select.select([candidate.stdout], [], [], min(remaining, 0.25))[0]:
                continue
            block = os.read(candidate.stdout.fileno(), 4096)
            if not block:
                # Closing fd 1 is not a signal that the process exited. Keep
                # the same deadline, then score timeout as rc=124 rather than
                # turning an adversarial stdout-close into infra pending.
                try:
                    return_code = candidate.wait(timeout=max(0.0, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    return_code = 124
                break
            output.extend(block)
            if len(output) > MAX_OUTPUT:
                return_code = 125
                break
    finally:
        if candidate.poll() is None:
            try:
                os.killpg(candidate.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            candidate.wait(timeout=2)
        candidate.stdout.close()
    return {"return_code": return_code,
            "stdout_b64": base64.b64encode(output[:MAX_OUTPUT]).decode("ascii"),
            "truncated": len(output) > MAX_OUTPUT}


def _namespace_case(libc, code, case_pool):
    _mount(libc, None, "/", flags=MS_REC | MS_PRIVATE)
    lower = case_pool / "lower"
    _mount(libc, b"/workspace", lower, flags=MS_BIND | MS_REC)
    _mount(libc, None, lower, flags=MS_BIND | MS_REMOUNT | MS_RDONLY)
    options = (f"lowerdir={lower},upperdir={case_pool / 'upper'},"
               f"workdir={case_pool / 'work'}").encode("ascii")
    merged = case_pool / "merged"
    _mount(libc, b"overlay", merged, b"overlay", 0, options)
    _mount(libc, os.fsencode(merged), "/workspace", flags=MS_BIND | MS_REC)
    Path("/workspace").chmod(0o777)
    return _capture(code)


def _worker(sock, code, case_pool):
    libc = _libc()
    live_reader, live_writer = os.pipe()
    try:
        _checked(libc.unshare(CLONE_NEWNS | CLONE_NEWPID), "unshare nested mount/PID")
        child_pid = os.fork()
        if child_pid == 0:
            os.close(live_writer)
            try:
                _checked(libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0),
                         "set nested parent-death signal")
                if (select.select([live_reader], [], [], 0)[0]
                        and os.read(live_reader, 1) == b""):
                    raise RuntimeError("nested_worker_died_before_case")
                result = {"ok": True, "case": _namespace_case(libc, code, case_pool)}
                status = 0
            except BaseException as exc:
                result = {"ok": False, "error": type(exc).__name__ + ": " + str(exc)[:200]}
                status = 1
            try:
                send_frame(sock, result, limit=MAX_RESPONSE)
            finally:
                os.close(live_reader)
                sock.close()
            os._exit(status)
        os.close(live_reader)
        sock.close()
        _, status = os.waitpid(child_pid, 0)
        os.close(live_writer)
        os._exit(0 if os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0 else 1)
    except BaseException as exc:
        try:
            send_frame(sock, {"ok": False, "error": type(exc).__name__ + ": " + str(exc)[:200]},
                       limit=MAX_RESPONSE)
        except BaseException:
            pass
        os._exit(1)


def _wait_worker(pid, *, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        got, status = os.waitpid(pid, os.WNOHANG)
        if got:
            return status
        time.sleep(0.005)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    os.waitpid(pid, 0)
    raise TimeoutError("nested_case_reap_timeout")


def run_one(code, upper_root):
    if (not isinstance(code, str) or not 1 <= len(code.encode("utf-8")) <= 512
            or "\x00" in code):
        raise ValueError("bounded_case_code_required")
    case_pool = Path(tempfile.mkdtemp(prefix=".fpb-case-", dir=upper_root))
    case_pool.chmod(0o700)
    for name in ("lower", "upper", "work", "merged"):
        (case_pool / name).mkdir()
    parent, child = socket.socketpair()
    parent.settimeout(CASE_SUPERVISOR_SECONDS)
    pid = None
    reaped = False
    try:
        pid = os.fork()
        if pid == 0:
            parent.close()
            _worker(child, code, case_pool)
            os._exit(1)
        child.close()
        result = read_frame(parent, limit=MAX_RESPONSE)
        status = _wait_worker(pid, timeout=CASE_SUPERVISOR_SECONDS)
        reaped = True
        if (not os.WIFEXITED(status) or os.WEXITSTATUS(status) != 0
                or not isinstance(result, dict) or set(result) != {"ok", "case"}
                or result["ok"] is not True or not isinstance(result["case"], dict)):
            raise RuntimeError("nested_case_process_failed: " + str(result)[:250])
        return result["case"]
    finally:
        parent.close()
        child.close()
        if pid is not None and not reaped:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass
        # Mounts belonged to the nested namespace, which must be gone now.
        # A failed cleanup is infrastructure failure, never a scored case.
        shutil.rmtree(case_pool)
