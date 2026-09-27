"""Trusted guest helper for one stateless Python verifier case.

This file is installed by the host into the private ARM64 guest image. It is
not a policy tool. The host passes only candidate code, never expected output.
Each invocation gets a new PID and mount namespace, an overlayfs repository
view with a tmpfs upper layer, and a separate tmpfs for temporary files.
The candidate runs as uid/gid 65534. When namespace PID 1 exits, the kernel
terminates its remaining descendants. This is an experimental shared-kernel
fast path for an explicitly declared stateless verifier contract.
"""

from __future__ import annotations

import base64
import ctypes
import json
import os
import re
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path


CLONE_NEWNS = 0x00020000
CLONE_NEWPID = 0x20000000
MS_REC = 0x4000
MS_PRIVATE = 0x40000
MS_BIND = 0x1000
MS_REMOUNT = 0x20
MS_RDONLY = 0x1
MS_NOSUID = 0x2
MS_NODEV = 0x4
MS_NOEXEC = 0x8
MAX_OUTPUT = 12000
CASE_SECONDS = 10.0
BATCH_FILE = Path("/fpb_stateless_cases.json")
MAX_BATCH_FILE = 100_000
MAX_BATCH_RESULT = 64_000
PR_SET_PDEATHSIG = 1


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


def _checked(result: int, operation: str):
    if result != 0:
        raise OSError(ctypes.get_errno(), operation)


def _mount(libc, source: bytes | None, target: Path | str,
           filesystem: bytes | None, flags: int, options: bytes | None):
    _checked(libc.mount(source, os.fsencode(target), filesystem, flags, options),
             "mount " + str(target))


def _capture_candidate(code: str, workspace: Path, temp_dir: Path):
    env = os.environ.copy()
    env.update({"PYTHONPATH": str(workspace), "TMPDIR": str(temp_dir),
                "HOME": str(temp_dir), "PYTHONDONTWRITEBYTECODE": "1"})
    candidate = subprocess.Popen(
        [sys.executable, "-B", "-c", code], cwd=str(workspace), env=env,
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        start_new_session=True, user=65534, group=65534, extra_groups=[])
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
            chunk = os.read(candidate.stdout.fileno(), 4096)
            if not chunk:
                return_code = candidate.wait(timeout=1)
                break
            output.extend(chunk)
            if len(output) > MAX_OUTPUT:
                return_code = 125
                break
    finally:
        if candidate.poll() is None:
            try:
                os.killpg(candidate.pid, 9)
            except ProcessLookupError:
                pass
            candidate.wait(timeout=2)
        candidate.stdout.close()
    return {"return_code": return_code,
            "stdout_b64": base64.b64encode(output[:MAX_OUTPUT]).decode("ascii"),
            "truncated": len(output) > MAX_OUTPUT}


def _namespace_case(libc, code: str, pool: Path):
    _mount(libc, None, "/", None, MS_REC | MS_PRIVATE, None)
    upper_root, merged, temp_dir = pool / "upperfs", pool / "merged", Path("/tmp")
    # Pin the whole chroot base read-only in this private namespace. A
    # per-mount bind remount, unlike an ext4 superblock remount, must not
    # change the outer VM's writable filesystem. The writable exceptions
    # below are new tmpfs mounts and the per-case overlay view.
    _mount(libc, b"/", "/", None, MS_BIND | MS_REC, None)
    _mount(libc, None, "/", None, MS_BIND | MS_REMOUNT | MS_RDONLY, None)
    # Direct access to the submitted lower tree must not write back to the
    # shared VM. This bind-remount is per mount namespace, not an ext4
    # superblock remount. The candidate's writable copy is `merged` below.
    _mount(libc, b"/workspace", "/workspace", None, MS_BIND | MS_REC, None)
    _mount(libc, None, "/workspace", None,
           MS_BIND | MS_REMOUNT | MS_RDONLY, None)
    _mount(libc, b"tmpfs", temp_dir, b"tmpfs", MS_NOSUID | MS_NODEV | MS_NOEXEC,
           b"size=32m,mode=1777")
    temp_dir.chmod(0o1777)
    _mount(libc, b"tmpfs", upper_root, b"tmpfs", MS_NOSUID | MS_NODEV | MS_NOEXEC,
           b"size=64m,mode=0700")
    (upper_root / "upper").mkdir()
    (upper_root / "work").mkdir()
    options = (f"lowerdir=/workspace,upperdir={upper_root / 'upper'},"
               f"workdir={upper_root / 'work'}").encode("ascii")
    _mount(libc, b"overlay", merged, b"overlay", 0, options)
    # Changes to the repository view land only in this case's tmpfs upper.
    merged.chmod(0o777)
    return _capture_candidate(code, merged, temp_dir)


def run_one(encoded_code: str):
    if not isinstance(encoded_code, str) or len(encoded_code) > 4096:
        raise ValueError("Bounded base64 Python code required")
    code = base64.b64decode(encoded_code, validate=True).decode("utf-8")
    if not code or len(code) > 2048 or "\x00" in code:
        raise ValueError("Bounded Python code required")
    pinned_sources = (Path("/workspace/boltons/strutils.py"),
                      Path("/workspace/src/humanize/filesize.py"))
    present = [path for path in pinned_sources if path.is_file() and not path.is_symlink()]
    if len(present) != 1:
        raise RuntimeError("Pinned workspace source missing or ambiguous")
    pool = Path(tempfile.mkdtemp(prefix=".fpb-stateless-", dir="/"))
    pool.chmod(0o711)
    for name in ("upperfs", "merged"):
        (pool / name).mkdir()
    reader, writer = os.pipe()
    libc = _libc()
    worker_pid = None
    try:
        # The original supervisor stays in its original mount namespace. An
        # intermediate worker unshares both namespaces; its next fork enters
        # the PID namespace as PID 1. The worker owns the private mount view.
        worker_pid = os.fork()
        if worker_pid == 0:
            os.close(reader)
            status = 1
            try:
                _checked(libc.unshare(CLONE_NEWNS | CLONE_NEWPID),
                         "unshare mount and PID namespaces")
                liveness_reader, liveness_writer = os.pipe()
                namespace_pid = os.fork()
                if namespace_pid == 0:
                    os.close(liveness_writer)
                    try:
                        _checked(libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0),
                                 "protect PID namespace supervisor from orphaning")
                        # getppid() can be 0 here because the worker lives
                        # outside the new PID namespace. A worker-held pipe
                        # closes on an early worker death, closing the race
                        # between fork and PR_SET_PDEATHSIG.
                        if (select.select([liveness_reader], [], [], 0)[0]
                                and os.read(liveness_reader, 1) == b""):
                            raise RuntimeError("Namespace worker exited before supervision")
                        result = {"ok": True, "case": _namespace_case(libc, code, pool)}
                        case_status = 0
                    except BaseException as exc:
                        result = {"ok": False, "error": type(exc).__name__ + ": " + str(exc)[:300]}
                        case_status = 1
                    data = json.dumps(result, separators=(",", ":")).encode("ascii")
                    offset = 0
                    while offset < len(data):
                        offset += os.write(writer, data[offset:])
                    os.close(liveness_reader)
                    os.close(writer)
                    os._exit(case_status)
                os.close(liveness_reader)
                _, wait_status = os.waitpid(namespace_pid, 0)
                os.close(liveness_writer)
                status = 0 if os.WIFEXITED(wait_status) and os.WEXITSTATUS(wait_status) == 0 else 1
            except BaseException as exc:
                result = {"ok": False, "error": type(exc).__name__ + ": " + str(exc)[:300]}
                data = json.dumps(result, separators=(",", ":")).encode("ascii")
                os.write(writer, data[:20000])
            os.close(writer)
            os._exit(status)
        os.close(writer)
        writer = -1
        chunks = []
        size = 0
        deadline = time.monotonic() + CASE_SECONDS + 5.0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([reader], [], [], remaining)[0]:
                raise TimeoutError("Guest stateless case supervisor timed out")
            chunk = os.read(reader, 4096)
            if not chunk:
                break
            size += len(chunk)
            if size > 20000:
                raise RuntimeError("Guest stateless result exceeds bound")
            chunks.append(chunk)
        _, wait_status = os.waitpid(worker_pid, 0)
        worker_pid = None
        result = json.loads(b"".join(chunks))
        if (not os.WIFEXITED(wait_status) or os.WEXITSTATUS(wait_status) != 0
                or not isinstance(result, dict) or not result.get("ok")):
            raise RuntimeError("Guest stateless case failed: " + repr(result)[:400])
        return result["case"]
    finally:
        os.close(reader)
        if writer >= 0:
            os.close(writer)
        if worker_pid is not None:
            try:
                os.kill(worker_pid, 9)
            except ProcessLookupError:
                pass
            os.waitpid(worker_pid, 0)
        # Child mounts never propagated to this namespace. These paths are
        # empty from here; never delete through a mounted overlay.
        shutil.rmtree(pool)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) == 2 and argv[0] == "--batch":
        digest = argv[1]
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Batch code digest must be SHA-256")
        if BATCH_FILE.is_symlink() or not BATCH_FILE.is_file():
            raise ValueError("Trusted case code file is missing")
        if BATCH_FILE.stat().st_mode & 0o777 != 0o600:
            raise ValueError("Trusted case code file must be root-private")
        raw = BATCH_FILE.read_bytes()
        if not 0 < len(raw) <= MAX_BATCH_FILE:
            raise ValueError("Trusted case code file exceeds bound")
        import hashlib
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("Trusted case code file digest changed")
        codes = json.loads(raw)
        if (not isinstance(codes, list) or not 1 <= len(codes) <= 32
                or not all(isinstance(code, str) for code in codes)):
            raise ValueError("Trusted case code file has invalid shape")
        results = []
        for code in codes:
            results.append(run_one(code))
            # Bound aggregate serial output; individual case output is already
            # capped at 12 KiB and no expected result is ever in this file.
            if len(json.dumps(results, separators=(",", ":")).encode()) > MAX_BATCH_RESULT:
                # A legitimate but wrong candidate may print enough data to
                # exceed the one-call transport. Tell the trusted host to rerun
                # exact comparisons through bounded one-case calls; this is a
                # scored task failure, not verifier infrastructure failure.
                print("FPB_STATELESS_BATCH_RESULT_OVERSIZE=1", flush=True)
                return
        encoded = base64.b64encode(json.dumps(results, separators=(",", ":")).encode()).decode()
        print("FPB_STATELESS_BATCH_RESULT=" + encoded, flush=True)
    elif len(argv) == 1:
        result = run_one(argv[0])
        encoded = base64.b64encode(json.dumps(result, separators=(",", ":")).encode()).decode()
        print("FPB_STATELESS_RESULT=" + encoded, flush=True)
    else:
        raise SystemExit("usage: guest_stateless_verifier.py BASE64_CODE | --batch SHA256")


if __name__ == "__main__":
    main()
