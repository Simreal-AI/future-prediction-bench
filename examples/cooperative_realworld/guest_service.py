"""Trusted, pinned process-and-OverlayFS checkpoint service.

An idle template process inherits a frozen, read-only OverlayFS workspace and
Python heap state. It forks exactly one resident episode at a time. The
bounded wire protocol and per-case verifier sandbox are reused from the
resident guest example. This service does not checkpoint the Linux kernel or
survive guest failure; guest root remains trusted.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import re
import socket
import sys
import tempfile
import time
from pathlib import Path


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("required_guest_helper_missing")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


wire = _load("wire", "/fpb_resident_wire.py")
resident = _load("fpb_resident_guest", "/fpb_resident_guest.py")
wire.MAX_RESPONSE = 32768
resident.MAX_RESPONSE = 32768

SOCKET_PATH = "/fpb-cooperative.sock"
SOURCES = frozenset(("boltons/strutils.py", "src/humanize/filesize.py"))
STATE = {"turn": 0, "prefix": "uninitialized"}
FROZEN = None


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _setup_branch_namespace(libc, pool):
    """Nested writable view over the template's frozen workspace."""
    if STATE != {"turn": 17, "prefix": "checkpointed"} or FROZEN is None:
        raise RuntimeError("template_process_state_not_restored")
    resident._mount(libc, None, "/", flags=resident.MS_REC | resident.MS_PRIVATE)
    resident._mount(libc, b"/", "/", flags=resident.MS_BIND | resident.MS_REC)
    resident._mount(libc, None, "/", flags=resident.MS_BIND | resident.MS_REMOUNT |
                    resident.MS_RDONLY)
    lower = pool / "lower"
    resident._mount(libc, os.fsencode(FROZEN), lower,
                    flags=resident.MS_BIND | resident.MS_REC)
    resident._mount(libc, None, lower, flags=resident.MS_BIND |
                    resident.MS_REMOUNT | resident.MS_RDONLY)
    upperfs = pool / "upperfs"
    resident._mount(libc, b"tmpfs", upperfs, b"tmpfs",
                    resident.MS_NOSUID | resident.MS_NODEV | resident.MS_NOEXEC,
                    b"size=64m,mode=0700")
    (upperfs / "upper").mkdir()
    (upperfs / "work").mkdir()
    merged = pool / "merged"
    options = (f"lowerdir={lower},upperdir={upperfs / 'upper'},"
               f"workdir={upperfs / 'work'}").encode("ascii")
    resident._mount(libc, b"overlay", merged, b"overlay", 0, options)
    resident._mount(libc, os.fsencode(merged), "/workspace",
                    flags=resident.MS_BIND | resident.MS_REC)
    resident._mount(libc, b"proc", pool / "proc", b"proc",
                    resident.MS_NOSUID | resident.MS_NODEV | resident.MS_NOEXEC)
    if Path("/workspace/.fpb-cooperative-prefix").read_text(encoding="ascii") != "prefix-committed":
        raise RuntimeError("frozen_filesystem_prefix_missing")
    marker = Path("/workspace/.fpb-cooperative-branch-marker")
    if marker.exists():
        raise RuntimeError("prior_episode_filesystem_state_leaked")
    marker.write_text(os.urandom(16).hex(), encoding="ascii")


def _template_loop(sock, source_path, seed_sha, case_count, frozen):
    global FROZEN
    FROZEN = frozen
    resident._setup_namespace = _setup_branch_namespace
    resident.MAX_PREVIEW = 16000  # Match MicroVMCodingAdapter's policy read cap.
    def load_cooperative_case_runner():
        return _load("fpb_coop_case_runner", "/fpb_coop_case_runner.py")
    resident._load_case_runner = load_cooperative_case_runner
    template_mount = os.readlink(frozen.parent / "proc/self/ns/mnt")
    wire.send_frame(sock, {"ok": True, "value": {
        "template_turn": STATE["turn"], "template_mount_namespace": template_mount,
        "frozen_source_sha256": _sha(frozen / source_path)}},
        limit=wire.MAX_RESPONSE)
    session = None
    while True:
        request = wire.read_frame(sock, limit=wire.MAX_BATCH_REQUEST)
        try:
            op = request.get("op")
            if op == "start" and session is None:
                if (STATE != {"turn": 17, "prefix": "checkpointed"}
                        or _sha(frozen / source_path) != seed_sha):
                    raise RuntimeError("template_changed_before_restore")
                session = resident.LinuxNamespaceSession(
                    request["episode_id"], request["mode"], source_path,
                    seed_sha, case_count)
                if session.namespaces["mount"] == template_mount:
                    raise RuntimeError("episode_reused_template_mount_namespace")
                value = {"namespaces": session.namespaces,
                         "guest_reset_ns": session.guest_reset_ns,
                         "namespace_setup_ns": session.namespace_setup_ns}
            elif op == "call" and session is not None:
                value = session.call(request["name"], request["args"])
            elif op == "close" and session is not None:
                cleanup = session.close(completed=request["completed"])
                session = None
                if (STATE != {"turn": 17, "prefix": "checkpointed"}
                        or _sha(frozen / source_path) != seed_sha):
                    raise RuntimeError("template_changed_after_episode")
                value = {"guest_cleanup_ns": cleanup,
                         "template_turn_after": STATE["turn"],
                         "frozen_source_sha256": seed_sha}
            elif op == "abort" and session is not None:
                session.abort()
                session = None
                value = {"aborted": True}
            else:
                raise wire.ProtocolError("invalid_template_operation")
            wire.send_frame(sock, {"ok": True, "value": value},
                            limit=wire.MAX_RESPONSE)
        except BaseException as exc:
            if session is not None:
                try:
                    session.abort()
                except BaseException:
                    pass
                session = None
            wire.send_frame(sock, {"ok": False,
                                   "error": type(exc).__name__ + ": " + str(exc)[:180]},
                            limit=wire.MAX_RESPONSE)


class TemplateBackend:
    def __init__(self, source_path, seed_sha, case_count):
        if (source_path not in SOURCES
                or re.fullmatch(r"[0-9a-f]{64}", seed_sha) is None
                or case_count != 14
                or _sha(Path("/workspace") / source_path) != seed_sha):
            raise ValueError("pinned_template_contract_required")
        self.source_path = source_path
        self.seed_sha = seed_sha
        self.case_count = case_count
        self.pool = Path(tempfile.mkdtemp(prefix=".fpb-cooperative-", dir="/"))
        self.pool.chmod(0o700)
        libc = resident._libc()
        resident._checked(libc.unshare(resident.CLONE_NEWNS), "unshare template mount")
        resident._mount(libc, None, "/", flags=resident.MS_REC | resident.MS_PRIVATE)
        resident._mount(libc, b"tmpfs", self.pool, b"tmpfs",
                        resident.MS_NOSUID | resident.MS_NODEV,
                        b"size=64m,mode=0700")
        proc = self.pool / "proc"
        proc.mkdir()
        resident._mount(libc, b"proc", proc, b"proc",
                        resident.MS_NOSUID | resident.MS_NODEV | resident.MS_NOEXEC)
        lower = self.pool / "lower"
        lower.mkdir()
        resident._mount(libc, b"/workspace", lower,
                        flags=resident.MS_BIND | resident.MS_REC)
        resident._mount(libc, None, lower,
                        flags=resident.MS_BIND | resident.MS_REMOUNT | resident.MS_RDONLY)
        frozen = self.pool / "frozen"
        for name in ("stage-upper", "stage-work", "frozen"):
            (self.pool / name).mkdir()
        options = (f"lowerdir={lower},upperdir={self.pool / 'stage-upper'},"
                   f"workdir={self.pool / 'stage-work'}").encode("ascii")
        resident._mount(libc, b"overlay", frozen, b"overlay", 0, options)
        (frozen / ".fpb-cooperative-prefix").write_text(
            "prefix-committed", encoding="ascii")
        STATE.update(turn=17, prefix="checkpointed")
        if _sha(frozen / source_path) != seed_sha:
            raise RuntimeError("frozen_source_changed_during_setup")
        parent, child = socket.socketpair()
        parent.settimeout(wire.BATCH_REQUEST_TIMEOUT)
        checkpoint_started = time.perf_counter_ns()
        resident._mount(libc, None, frozen,
                        flags=resident.MS_REMOUNT | resident.MS_RDONLY)
        pid = os.fork()
        if pid == 0:
            parent.close()
            try:
                _template_loop(child, source_path, seed_sha, case_count, frozen)
            finally:
                child.close()
            os._exit(0)
        child.close()
        ready = wire.read_frame(parent, limit=wire.MAX_RESPONSE)
        self.checkpoint_ns = time.perf_counter_ns() - checkpoint_started
        STATE.update(turn=999, prefix="parent-after-checkpoint")
        if (ready.get("ok") is not True
                or ready["value"].get("template_turn") != 17
                or ready["value"].get("frozen_source_sha256") != seed_sha):
            raise RuntimeError("template_not_ready")
        self.template_pid = pid
        self.template_mount_namespace = ready["value"]["template_mount_namespace"]
        self.sock = parent
        self.active = False

    def _request(self, request):
        wire.send_frame(self.sock, request, limit=wire.MAX_BATCH_REQUEST)
        answer = wire.read_frame(self.sock, limit=wire.MAX_RESPONSE)
        if (not isinstance(answer, dict) or set(answer) not in
                ({"ok", "value"}, {"ok", "error"})
                or answer.get("ok") is not True):
            raise RuntimeError("template_process_or_communication_failed")
        return answer["value"]

    def start(self, episode_id, mode):
        if self.active or STATE != {"turn": 999, "prefix": "parent-after-checkpoint"}:
            raise RuntimeError("live_parent_state_invalid")
        started = time.perf_counter_ns()
        value = self._request({"op": "start", "episode_id": episode_id,
                               "mode": mode})
        self.active = True
        return TemplateSession(self, value, time.perf_counter_ns() - started)


class TemplateSession:
    def __init__(self, backend, created, host_reset_ns):
        self.backend = backend
        self.namespaces = created["namespaces"]
        self.guest_reset_ns = created["guest_reset_ns"]
        self.namespace_setup_ns = created["namespace_setup_ns"]
        self.host_reset_ns = host_reset_ns

    def call(self, op, args):
        return self.backend._request({"op": "call", "name": op, "args": args})

    def close(self, *, completed=False):
        value = self.backend._request({"op": "close", "completed": completed})
        self.backend.active = False
        if (value.get("template_turn_after") != 17
                or value.get("frozen_source_sha256") != self.backend.seed_sha):
            raise RuntimeError("template_post_episode_state_invalid")
        return value["guest_cleanup_ns"]

    def abort(self):
        try:
            self.backend._request({"op": "abort"})
        finally:
            self.backend.active = False


class CooperativeSupervisor(resident.Supervisor):
    def handle(self, request):
        answer = super().handle(request)
        if request.get("op") == "hello" and answer.get("ok"):
            answer["value"]["cooperative_checkpoint"] = {
                "kind": "quiescent_process_fork_frozen_overlay_v1",
                "template_ready": True,
                "checkpoint_ns": self.backend.checkpoint_ns,
                "template_mount_namespace": self.backend.template_mount_namespace,
                "frozen_source_sha256": self.backend.seed_sha,
                "guest_service_sha256": _sha("/fpb_coop_guest.py"),
                "guest_case_runner_sha256": _sha("/fpb_coop_case_runner.py"),
            }
        return answer


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
        backend = TemplateBackend(args.source_path, args.seed_sha256,
                                  args.case_count)
        supervisor = CooperativeSupervisor(
            backend=backend, source_path=args.source_path,
            seed_file_sha256=args.seed_sha256, case_count=args.case_count,
            helper_paths={
                "wire.py": Path("/fpb_resident_wire.py"),
                "guest_supervisor.py": Path("/fpb_resident_guest.py"),
                "case_runner.py": Path("/fpb_resident_case_runner.py"),
                "hardened_edit.py": Path("/fpb_resident_hardened_edit.py"),
            })
        resident.serve(path=SOCKET_PATH, supervisor=supervisor)
    else:
        print("FPB_COOP_V0=" + resident.call(args.call, path=SOCKET_PATH).split("=", 1)[1])


if __name__ == "__main__":
    main()
