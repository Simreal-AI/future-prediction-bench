"""EXPERIMENTAL trusted resident supervisor for one already booted Linux VM.

The host never sends expected outputs. This prototype deliberately exposes no
arbitrary shell/Python policy action. The host-only verifier sends bounded
case code after submission and grades the returned results on the host. Its
long-lived process owns one episode child at a time; the child is PID 1 in a
private PID namespace and sees a private overlayfs workspace.

Run inside the guest chroot as root: ``python3 -I -B guest_supervisor.py --serve``.
The serial bridge uses ``--call BASE64_FRAME`` on a separate process.
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import hashlib
import importlib.util
import os
import re
import select
import signal
import socket
import stat
import sys
import tempfile
import time
import traceback
from pathlib import Path

if __name__ == "__main__" and "wire" not in sys.modules:
    # Python -I omits the script directory from sys.path. The trusted host
    # installs both reviewed files at fixed absolute paths in the guest.
    spec = importlib.util.spec_from_file_location("wire", "/fpb_resident_wire.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("resident_wire_missing")
    module = importlib.util.module_from_spec(spec)
    sys.modules["wire"] = module
    spec.loader.exec_module(module)

from wire import (IDENTITY, MAX_REQUEST, MAX_BATCH_REQUEST, MAX_RESPONSE, MAX_CASES,
                  CASE_REQUEST_TIMEOUT, BATCH_REQUEST_TIMEOUT, ProtocolError,
                  read_frame, relative_path, response, send_frame, unframe,
                  validate_request, frame)


SOCKET_PATH = "/fpb-resident.sock"
WORKSPACE = Path("/workspace")
MAX_FILE = 65536
MAX_PREVIEW = 8192
CHILD_TIMEOUT = 10.0
CLONE_NEWNS = 0x00020000
CLONE_NEWPID = 0x20000000
MS_RDONLY = 1
MS_NOSUID = 2
MS_NODEV = 4
MS_NOEXEC = 8
MS_REMOUNT = 32
MS_BIND = 4096
MS_REC = 16384
MS_PRIVATE = 1 << 18
PR_SET_PDEATHSIG = 1


class BranchFault(RuntimeError):
    """Episode process, namespace, or cleanup failure: no score is possible."""


class ActionRejected(ValueError):
    """Policy action was invalid, but the episode can continue."""


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


def _checked(result, name):
    if result != 0:
        raise OSError(ctypes.get_errno(), name)


def _mount(libc, source, target, filesystem=None, flags=0, options=None):
    _checked(libc.mount(source, os.fsencode(target), filesystem, flags, options),
             "mount " + str(target))


def _setup_namespace(libc, pool):
    # All mounts below belong to this child namespace. A bind-remount of / is
    # per mount namespace; never remount the underlying ext4 superblock.
    _mount(libc, None, "/", flags=MS_REC | MS_PRIVATE)
    _mount(libc, b"/", "/", flags=MS_BIND | MS_REC)
    _mount(libc, None, "/", flags=MS_BIND | MS_REMOUNT | MS_RDONLY)
    lower = pool / "lower"
    _mount(libc, os.fsencode(WORKSPACE), lower, flags=MS_BIND | MS_REC)
    _mount(libc, None, lower, flags=MS_BIND | MS_REMOUNT | MS_RDONLY)
    upperfs = pool / "upperfs"
    _mount(libc, b"tmpfs", upperfs, b"tmpfs", MS_NOSUID | MS_NODEV | MS_NOEXEC,
           b"size=64m,mode=0700")
    (upperfs / "upper").mkdir()
    (upperfs / "work").mkdir()
    options = (f"lowerdir={lower},upperdir={upperfs / 'upper'},"
               f"workdir={upperfs / 'work'}").encode("ascii")
    merged = pool / "merged"
    _mount(libc, b"overlay", merged, b"overlay", 0, options)
    _mount(libc, os.fsencode(merged), WORKSPACE, flags=MS_BIND | MS_REC)
    # This pinned rootfs has no /tmp directory. No candidate process runs in
    # this milestone; exact-edit temporary files stay inside the overlay.
    # A future case runner must add a private temporary mount deliberately.
    _mount(libc, b"proc", pool / "proc", b"proc", MS_NOSUID | MS_NODEV | MS_NOEXEC)


def _open_workspace_file(path):
    parts = relative_path(path)
    root = os.open(WORKSPACE, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    opened = [root]
    try:
        parent = root
        for part in parts[:-1]:
            parent = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                             dir_fd=parent)
            opened.append(parent)
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            raise ActionRejected("not_regular_file")
        return fd, parent, parts[-1], opened
    except BaseException:
        for opened_fd in reversed(opened):
            os.close(opened_fd)
        raise


def _bounded_file(path):
    try:
        fd, parent, basename, dirs = _open_workspace_file(path)
    except (OSError, ProtocolError) as exc:
        raise ActionRejected("invalid_file") from exc
    try:
        mode = stat.S_IMODE(os.fstat(fd).st_mode)
        data = b""
        while len(data) <= MAX_FILE:
            block = os.read(fd, min(8192, MAX_FILE + 1 - len(data)))
            if not block:
                break
            data += block
        if len(data) > MAX_FILE:
            raise ActionRejected("file_too_large")
        try:
            data.decode("utf-8")
        except UnicodeError as exc:
            raise ActionRejected("non_utf8_file") from exc
        return data, mode, parent, basename, dirs
    except BaseException:
        for opened_fd in reversed(dirs):
            os.close(opened_fd)
        raise
    finally:
        os.close(fd)


def _read_file(path):
    data, _, _, _, dirs = _bounded_file(path)
    try:
        digest = hashlib.sha256(data).hexdigest()
        preview = data[:MAX_PREVIEW].decode("utf-8", "ignore")
        return {"accepted": True, "sha256": digest, "bytes": len(data),
                "text": preview, "truncated": len(data) > MAX_PREVIEW}
    finally:
        for fd in reversed(dirs):
            os.close(fd)


def _source_sha(source_path):
    data, _, _, _, dirs = _bounded_file(source_path)
    try:
        return hashlib.sha256(data).hexdigest()
    finally:
        for fd in reversed(dirs):
            os.close(fd)


def _assert_quiescent(pool):
    pids = {name for name in os.listdir(pool / "proc") if name.isdecimal()}
    if pids != {"1"}:
        raise BranchFault("episode_not_quiescent")


def _load_case_runner():
    target = "/fpb_resident_case_runner.py"
    spec = importlib.util.spec_from_file_location("fpb_resident_case_runner", target)
    if spec is None or spec.loader is None:
        raise BranchFault("case_runner_missing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_hardened_edit():
    target = "/fpb_resident_hardened_edit.py"
    spec = importlib.util.spec_from_file_location("fpb_resident_hardened_edit", target)
    if spec is None or spec.loader is None:
        raise BranchFault("hardened_edit_missing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _replace_text(inp):
    try:
        return _load_hardened_edit().apply("/workspace", inp)
    except ValueError:
        # The released helper exposes unsafe targets and malformed input as
        # policy errors, distinct from ordinary stale-state conflicts.
        return {"accepted": False, "reason": "adapter_error",
                "error_type": "ValueError"}


def _bound_batch_case_result(result):
    # Batch transport carries at most 64 stdout bytes per case. A larger
    # candidate output is a definite mismatch when the host-preflighted
    # expected output is <=64 bytes, not an infrastructure failure.
    if len(base64.b64decode(result["stdout_b64"], validate=True)) > 64:
        result["stdout_b64"] = ""
        result["output_over_batch_cap"] = True
    else:
        result["output_over_batch_cap"] = False
    return result


def _child_loop(sock, episode_id, mode, pool, source_path, seed_file_sha256,
                case_limit):
    try:
        setup_started = time.perf_counter_ns()
        _setup_namespace(_libc(), pool)
        setup_ns = time.perf_counter_ns() - setup_started
        if _source_sha(source_path) != seed_file_sha256:
            raise RuntimeError("workspace_source_seed_mismatch")
        namespaces = {
            "pid": os.readlink(pool / "proc/self/ns/pid"),
            "mount": os.readlink(pool / "proc/self/ns/mnt"),
            "pid_one": os.getpid() == 1,
            "process_nonce": os.urandom(16).hex(),
        }
        if not namespaces["pid_one"]:
            raise RuntimeError("branch_not_pid_one")
        send_frame(sock, {"ready": True, "episode_id": episode_id, "mode": mode,
                          "namespaces": namespaces, "namespace_setup_ns": setup_ns},
                   limit=MAX_RESPONSE)
        submitted = False
        submitted_sha = None
        case_count = 0
        while True:
            command = read_frame(sock, limit=MAX_BATCH_REQUEST)
            if set(command) != {"op", "args"} or not isinstance(command["args"], dict):
                raise ProtocolError("invalid_child_command")
            op, args = command["op"], command["args"]
            if op == "action" and not submitted:
                try:
                    if args["name"] == "read_file":
                        value = _read_file(args["input"]["path"])
                    elif args["name"] == "replace_text":
                        value = _replace_text(args["input"])
                    else:
                        raise ProtocolError("unknown_child_action")
                except ActionRejected as exc:
                    value = {"accepted": False, "reason": str(exc)[:80]}
                send_frame(sock, {"ok": True, "value": value}, limit=MAX_RESPONSE)
            elif op == "submit" and not submitted and not args:
                _assert_quiescent(pool)
                submitted_sha = _source_sha(source_path)
                submitted = True
                send_frame(sock, {"ok": True, "value": {"status": "pending", "reward": None}},
                           limit=MAX_RESPONSE)
            elif op == "case" and submitted and case_count < case_limit:
                _assert_quiescent(pool)
                if _source_sha(source_path) != submitted_sha:
                    raise BranchFault("submitted_source_changed")
                case_started = time.perf_counter_ns()
                result = _load_case_runner().run_one(args["code"], pool / "upperfs")
                result["guest_case_ns"] = time.perf_counter_ns() - case_started
                result["output_over_batch_cap"] = False
                _assert_quiescent(pool)
                if _source_sha(source_path) != submitted_sha:
                    raise BranchFault("case_changed_submitted_source")
                case_count += 1
                result["branch_sha256"] = submitted_sha
                send_frame(sock, {"ok": True, "value": result}, limit=MAX_RESPONSE)
            elif op == "case_batch" and submitted and case_count == 0:
                if len(args["codes"]) != case_limit:
                    raise ProtocolError("case_batch_wrong_count")
                _assert_quiescent(pool)
                if _source_sha(source_path) != submitted_sha:
                    raise BranchFault("submitted_source_changed")
                runner = _load_case_runner()
                results = []
                for code in args["codes"]:
                    case_started = time.perf_counter_ns()
                    result = runner.run_one(code, pool / "upperfs")
                    result["guest_case_ns"] = time.perf_counter_ns() - case_started
                    _assert_quiescent(pool)
                    if _source_sha(source_path) != submitted_sha:
                        raise BranchFault("case_changed_submitted_source")
                    results.append(_bound_batch_case_result(result))
                case_count = case_limit
                value = {"branch_sha256": submitted_sha, "cases": results}
                send_frame(sock, {"ok": True, "value": value}, limit=MAX_RESPONSE)
            elif op == "close" and set(args) == {"completed"}:
                if args["completed"] and (not submitted or case_count != case_limit):
                    raise ProtocolError("submitted_cases_incomplete")
                send_frame(sock, {"ok": True, "value": {"closed": True}},
                           limit=MAX_RESPONSE)
                return
            else:
                raise ProtocolError("invalid_child_state")
    except BaseException as exc:
        try:
            send_frame(sock, {"ready": False, "error": type(exc).__name__ + ": " + str(exc)[:120]},
                       limit=MAX_RESPONSE)
        except BaseException:
            pass
        raise


def _worker(sock, episode_id, mode, pool, source_path, seed_file_sha256,
            case_limit):
    libc = _libc()
    live_reader, live_writer = os.pipe()
    try:
        _checked(libc.unshare(CLONE_NEWNS | CLONE_NEWPID), "unshare mount/PID")
        child_pid = os.fork()
        if child_pid == 0:
            os.close(live_writer)
            try:
                _checked(libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0),
                         "set parent-death signal")
                if (select.select([live_reader], [], [], 0)[0]
                        and os.read(live_reader, 1) == b""):
                    raise BranchFault("worker_died_before_child_ready")
                _child_loop(sock, episode_id, mode, pool, source_path,
                            seed_file_sha256, case_limit)
                status = 0
            except BaseException:
                status = 1
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
            send_frame(sock, {"ready": False, "error": type(exc).__name__ + ": " + str(exc)[:120]},
                       limit=MAX_RESPONSE)
        except BaseException:
            pass
        os._exit(1)


class LinuxNamespaceSession:
    def __init__(self, episode_id, mode, source_path, seed_file_sha256,
                 case_limit):
        reset_started = time.perf_counter_ns()
        self.pool = Path(tempfile.mkdtemp(prefix=".fpb-resident-", dir="/"))
        self.pool.chmod(0o700)
        for name in ("lower", "upperfs", "merged", "proc"):
            (self.pool / name).mkdir()
        parent, child = socket.socketpair()
        parent.settimeout(CHILD_TIMEOUT)
        pid = os.fork()
        if pid == 0:
            parent.close()
            _worker(child, episode_id, mode, self.pool, source_path,
                    seed_file_sha256, case_limit)
            os._exit(1)
        child.close()
        self.sock = parent
        self.pid = pid
        self.reaped = False
        try:
            ready = read_frame(self.sock, limit=MAX_RESPONSE)
            if (ready.get("ready") is not True or ready.get("episode_id") != episode_id
                    or ready.get("mode") != mode or not isinstance(ready.get("namespaces"), dict)
                    or ready["namespaces"].get("pid_one") is not True):
                raise BranchFault("namespace_start_failed: " + str(ready)[:250])
            self.namespaces = ready["namespaces"]
            self.namespace_setup_ns = ready["namespace_setup_ns"]
            self.guest_reset_ns = time.perf_counter_ns() - reset_started
        except BaseException as exc:
            self._abort()
            raise BranchFault("namespace_start_failed: " + str(exc)[:250]) from exc

    def call(self, op, args):
        try:
            self.sock.settimeout(BATCH_REQUEST_TIMEOUT if op == "case_batch"
                                 else CASE_REQUEST_TIMEOUT if op == "case"
                                 else CHILD_TIMEOUT)
            send_frame(self.sock, {"op": op, "args": args},
                       limit=MAX_BATCH_REQUEST if op == "case_batch" else MAX_REQUEST)
            answer = read_frame(self.sock, limit=MAX_RESPONSE)
            if set(answer) != {"ok", "value"} or answer["ok"] is not True or not isinstance(answer["value"], dict):
                raise ProtocolError("invalid_child_response")
            return answer["value"]
        except BaseException as exc:
            self._abort()
            raise BranchFault("branch_process_or_ipc_failed") from exc

    def _reap(self, *, kill):
        if self.reaped:
            return 0
        if kill:
            try:
                os.kill(self.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + CHILD_TIMEOUT
        pidfd = None
        try:
            if hasattr(os, "pidfd_open"):
                try:
                    pidfd = os.pidfd_open(self.pid)
                except OSError:
                    pass
            while time.monotonic() < deadline:
                got, status = os.waitpid(self.pid, os.WNOHANG)
                if got:
                    self.reaped = True
                    return status
                remaining = max(0.0, deadline - time.monotonic())
                if pidfd is not None:
                    select.select([pidfd], [], [], remaining)
                else:
                    time.sleep(min(0.001, remaining))
        finally:
            if pidfd is not None:
                os.close(pidfd)
        try:
            os.kill(self.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        _, status = os.waitpid(self.pid, 0)
        self.reaped = True
        raise BranchFault("branch_reap_timeout")

    def _remove_pool(self):
        # The child mount namespace has exited. No overlay or tmpfs mounts may
        # still exist in the supervisor mount namespace.
        for name in ("lower", "upperfs", "merged", "proc"):
            (self.pool / name).rmdir()
        self.pool.rmdir()

    def _abort(self):
        try:
            self.sock.close()
        finally:
            try:
                self._reap(kill=True)
            finally:
                self._remove_pool()

    def abort(self):
        self._abort()

    def close(self, *, completed=False):
        cleanup_started = time.perf_counter_ns()
        value = self.call("close", {"completed": completed})
        if value != {"closed": True}:
            self._abort()
            raise BranchFault("invalid_branch_close")
        self.sock.close()
        try:
            status = self._reap(kill=False)
        finally:
            self._remove_pool()
        if not os.WIFEXITED(status) or os.WEXITSTATUS(status) != 0:
            raise BranchFault("branch_close_failed")
        return time.perf_counter_ns() - cleanup_started


class LinuxNamespaceBackend:
    def __init__(self, source_path, seed_file_sha256, case_limit):
        self.source_path = source_path
        self.seed_file_sha256 = seed_file_sha256
        self.case_limit = case_limit

    def start(self, episode_id, mode):
        return LinuxNamespaceSession(episode_id, mode, self.source_path,
                                     self.seed_file_sha256, self.case_limit)


class Supervisor:
    def __init__(self, backend=None, *, helper_paths=None, source_path=None,
                 seed_file_sha256=None, case_count=None):
        relative_path(source_path)
        if (not isinstance(seed_file_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", seed_file_sha256) is None):
            raise ValueError("seed_file_sha256_required")
        if type(case_count) is not int or not 1 <= case_count <= MAX_CASES:
            raise ValueError("case_count_required")
        self.source_path = source_path
        self.seed_file_sha256 = seed_file_sha256
        self.case_count = case_count
        if backend is None and _source_sha(source_path) != seed_file_sha256:
            raise RuntimeError("outer_workspace_source_seed_mismatch")
        self.backend = backend or LinuxNamespaceBackend(source_path, seed_file_sha256,
                                                        case_count)
        self.helper_paths = helper_paths or (
            {name: Path(__file__).with_name(name) for name in
             ("wire.py", "guest_supervisor.py", "case_runner.py",
              "hardened_edit.py")}
            if backend is not None else
            {"wire.py": Path("/fpb_resident_wire.py"),
             "guest_supervisor.py": Path("/fpb_resident_guest.py"),
             "case_runner.py": Path("/fpb_resident_case_runner.py"),
             "hardened_edit.py": Path("/fpb_resident_hardened_edit.py")})
        self.boot_id = os.urandom(16).hex()
        self.next_seq = 0
        self.state = "idle"
        self.session = None
        self.executed_cases = 0

    def _poison(self):
        session, self.session = self.session, None
        self.state = "interrupted"
        if session is not None:
            try:
                session.abort()
            except BaseException:
                pass

    def handle(self, request):
        seq = request.get("seq", -1) if isinstance(request, dict) else -1
        try:
            validate_request(request)
            if request["seq"] != self.next_seq:
                raise ProtocolError("nonmonotonic_sequence")
            self.next_seq += 1
            op, args = request["op"], request["args"]
            if op == "hello":
                value = {"identity": IDENTITY, "boot_id": self.boot_id,
                         "state": self.state,
                         "source_path": self.source_path,
                         "seed_file_sha256": self.seed_file_sha256,
                         "case_count": self.case_count,
                         "guest_helper_sha256": {
                             name: hashlib.sha256(path.read_bytes()).hexdigest()
                             for name, path in self.helper_paths.items()}}
            elif self.state == "interrupted":
                return response(seq, error="runtime_interrupted")
            elif op == "create":
                if self.state != "idle":
                    return response(seq, error="episode_already_active")
                self.session = self.backend.start(args["episode_id"], args["mode"])
                self.state = "active"
                self.executed_cases = 0
                value = {"status": "active", "episode_id": args["episode_id"],
                         "mode": args["mode"], "reward": None,
                         "namespaces": self.session.namespaces,
                         "guest_reset_ns": self.session.guest_reset_ns,
                         "namespace_setup_ns": self.session.namespace_setup_ns}
            elif op == "action":
                if self.state != "active":
                    return response(seq, error="episode_not_active")
                value = self.session.call("action", args)
            elif op == "case":
                if self.state != "pending":
                    return response(seq, error="episode_not_submitted")
                if self.executed_cases >= self.case_count:
                    raise ProtocolError("too_many_cases")
                value = self.session.call("case", args)
                self.executed_cases += 1
            elif op == "case_batch":
                if self.state != "pending":
                    return response(seq, error="episode_not_submitted")
                if self.executed_cases or len(args["codes"]) != self.case_count:
                    raise ProtocolError("case_batch_wrong_count_or_order")
                value = self.session.call("case_batch", args)
                self.executed_cases = self.case_count
            elif op == "submit":
                if self.state != "active":
                    return response(seq, error="episode_not_active")
                value = self.session.call("submit", {})
                if value != {"status": "pending", "reward": None}:
                    raise BranchFault("invalid_submit_result")
                self.state = "pending"
            elif op == "close":
                if self.state not in {"active", "pending"}:
                    return response(seq, error="episode_not_active")
                if args["completed"] and (self.state != "pending"
                                           or self.executed_cases != self.case_count):
                    raise ProtocolError("submitted_cases_incomplete")
                cleanup_ns = self.session.close(completed=args["completed"])
                self.session = None
                self.state = "idle"
                self.executed_cases = 0
                value = {"status": "closed", "reward": None,
                         "guest_cleanup_ns": cleanup_ns}
            else:
                raise ProtocolError("unknown_operation")
            return response(seq, value=value)
        except ProtocolError:
            self._poison()
            return response(seq, error="protocol_failed")
        except BaseException:
            if isinstance(self.backend, LinuxNamespaceBackend):
                traceback.print_exc(file=sys.stderr)
            self._poison()
            return response(seq, error="resident_infrastructure_failed")


def serve(path=SOCKET_PATH, *, supervisor, max_connections=None):
    if os.path.lexists(path):
        raise RuntimeError("resident_socket_already_exists")
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    bound = False
    try:
        server.bind(path)
        bound = True
        os.chmod(path, 0o600)
        server.listen(1)
        count = 0
        while max_connections is None or count < max_connections:
            with server.accept()[0] as client:
                client.settimeout(CHILD_TIMEOUT)
                serve_one(client, supervisor)
            count += 1
    finally:
        server.close()
        if bound:
            os.unlink(path)


def serve_one(client, supervisor):
    """One socket exchange; a partial frame/response poisons active branch."""
    try:
        request = read_frame(client, limit=MAX_BATCH_REQUEST)
        client.settimeout(BATCH_REQUEST_TIMEOUT if request.get("op") == "case_batch"
                          else CASE_REQUEST_TIMEOUT if request.get("op") == "case"
                          else CHILD_TIMEOUT)
        answer = supervisor.handle(request)
        send_frame(client, answer, limit=MAX_RESPONSE)
    except (OSError, ProtocolError):
        supervisor._poison()


def call(encoded, path=SOCKET_PATH):
    if not re.fullmatch(r"[A-Za-z0-9+/=]{1,2800}", encoded):
        raise ProtocolError("invalid_serial_request")
    request_bytes = base64.b64decode(encoded, validate=True)
    request = unframe(request_bytes, limit=MAX_BATCH_REQUEST)
    validate_request(request)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(BATCH_REQUEST_TIMEOUT if request["op"] == "case_batch"
                          else CASE_REQUEST_TIMEOUT if request["op"] == "case"
                          else CHILD_TIMEOUT)
        client.connect(path)
        client.sendall(request_bytes)
        result = read_frame(client, limit=MAX_RESPONSE)
    return "FPB_RESIDENT_V0=" + base64.b64encode(frame(result, limit=MAX_RESPONSE)).decode("ascii")


def main(argv=None):
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--serve", action="store_true")
    group.add_argument("--call")
    parser.add_argument("--source-path")
    parser.add_argument("--seed-sha256")
    parser.add_argument("--case-count", type=int)
    args = parser.parse_args(argv)
    if args.serve:
        serve(supervisor=Supervisor(source_path=args.source_path,
                                    seed_file_sha256=args.seed_sha256,
                                    case_count=args.case_count))
    else:
        print(call(args.call))


if __name__ == "__main__":
    main()
