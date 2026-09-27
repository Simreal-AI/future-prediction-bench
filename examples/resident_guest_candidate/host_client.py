"""Host-only transport/lifecycle client for a trusted resident guest process.

The guest receives actions and bounded verifier case code only. Hidden
expected outputs and reward computation have no transport path. The separate
host RealWorldEnv adapter owns grading; this client's ``verify`` placeholder
does not itself expose reward calculation.
"""

from __future__ import annotations

import base64
import hashlib
import re
import shlex
import time
from pathlib import Path

from wire import (IDENTITY, MAX_REQUEST, MAX_BATCH_REQUEST, MAX_RESPONSE, MAX_CASES,
                  CASE_REQUEST_TIMEOUT, BATCH_REQUEST_TIMEOUT, ProtocolError, VERSION,
                  frame, relative_path, unframe, validate_request, validate_response)


class ResidentInterrupted(RuntimeError):
    pass


class ResidentPending(RuntimeError):
    pass


def _safe_relative(path):
    try:
        relative_path(path)
    except ProtocolError:
        return False
    return True


def prototype_binding():
    files = ("wire.py", "guest_supervisor.py", "case_runner.py", "hardened_edit.py",
             "host_client.py", "host_realworld_adapter.py")
    base = Path(__file__).resolve().parent
    return {"runtime_kind": IDENTITY,
            "prototype_sha256": hashlib.sha256(b"".join(
                hashlib.sha256((base / name).read_bytes()).digest() for name in files
            )).hexdigest()}


class GuestSerialTransport:
    """One bounded request via an already running guest Unix-socket server.

    Installation and daemon startup are intentionally separate, trusted VM
    setup steps. This class never boots a VM or injects verifier expectations.
    """

    def __init__(self, runtime, *, timeout=20.0):
        self.runtime = runtime
        self.timeout = timeout

    def exchange(self, request):
        validate_request(request)
        limit = MAX_BATCH_REQUEST if request["op"] == "case_batch" else MAX_REQUEST
        encoded = base64.b64encode(frame(request, limit=limit)).decode("ascii")
        command = ("LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/aarch64-linux-gnu "
                   "chroot /mnt/root /usr/local/bin/python3.12 -I -B "
                   f"/fpb_resident_guest.py --call {shlex.quote(encoded)}")
        if len(command.encode("ascii")) > 3800:
            raise ProtocolError("serial_command_too_large")
        timeout = (max(self.timeout, BATCH_REQUEST_TIMEOUT) if request["op"] == "case_batch"
                   else max(self.timeout, CASE_REQUEST_TIMEOUT) if request["op"] == "case"
                   else self.timeout)
        result = self.runtime.run_shell(command, timeout=timeout)
        if result.get("return_code") != 0:
            raise ResidentInterrupted("resident_serial_call_failed")
        stdout = result.get("stdout")
        match = re.fullmatch(r"FPB_RESIDENT_V0=([A-Za-z0-9+/=]+)\n?", stdout or "")
        if match is None or len(match.group(1)) > 22000:
            raise ResidentInterrupted("resident_serial_response_invalid")
        try:
            data = base64.b64decode(match.group(1), validate=True)
            return unframe(data, limit=MAX_RESPONSE)
        except (ValueError, ProtocolError) as exc:
            raise ResidentInterrupted("resident_serial_response_invalid") from exc


class ResidentHostAdapter:
    """Lifecycle bridge used by the separate host-only grading adapter."""

    def __init__(self, transport):
        self.transport = transport
        self.seq = 0
        self.state = "new"
        self.boot_id = None
        self.guest_helper_sha256 = None
        self.source_path = None
        self.seed_file_sha256 = None
        self.case_count = None
        self.reward = None
        self.binding = prototype_binding()
        self.request_timings = []

    def _exchange(self, op, args):
        request = {"v": VERSION, "seq": self.seq, "op": op, "args": args}
        started = time.perf_counter()
        try:
            validate_request(request)
            answer = self.transport.exchange(request)
            validate_response(answer, seq=self.seq)
            self.seq += 1
        except BaseException as exc:
            self.state = "interrupted"
            self.reward = None
            raise ResidentInterrupted("resident_protocol_or_process_failed") from exc
        finally:
            self.request_timings.append({"op": op, "seq": request["seq"],
                                         "seconds": time.perf_counter() - started})
        if not answer["ok"]:
            self.state = "interrupted"
            self.reward = None
            raise ResidentInterrupted(answer["error"])
        return answer["value"]

    def connect(self):
        if self.state != "new":
            raise ValueError("already_connected")
        value = self._exchange("hello", {})
        if (value.get("identity") != IDENTITY
                or not isinstance(value.get("boot_id"), str)
                or not re.fullmatch(r"[0-9a-f]{32}", value["boot_id"])
                or value.get("state") != "idle"
                or not isinstance(value.get("guest_helper_sha256"), dict)
                or set(value["guest_helper_sha256"]) != {
                    "wire.py", "guest_supervisor.py", "case_runner.py",
                    "hardened_edit.py"}
                or not isinstance(value.get("source_path"), str)
                or not _safe_relative(value["source_path"])
                or not isinstance(value.get("seed_file_sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", value["seed_file_sha256"])
                or type(value.get("case_count")) is not int
                or not 1 <= value["case_count"] <= MAX_CASES
                or any(not isinstance(digest, str)
                       or not re.fullmatch(r"[0-9a-f]{64}", digest)
                       for digest in value["guest_helper_sha256"].values())):
            self.state = "interrupted"
            raise ResidentInterrupted("resident_identity_or_state_mismatch")
        self.boot_id = value["boot_id"]
        self.guest_helper_sha256 = dict(value["guest_helper_sha256"])
        self.source_path = value["source_path"]
        self.seed_file_sha256 = value["seed_file_sha256"]
        self.case_count = value["case_count"]
        self.state = "idle"
        return dict(self.binding, boot_id=self.boot_id,
                    guest_helper_sha256=dict(self.guest_helper_sha256),
                    source_path=self.source_path,
                    seed_file_sha256=self.seed_file_sha256,
                    case_count=self.case_count)

    def reset(self, episode_id, mode):
        if self.state != "idle":
            raise ValueError("not_idle")
        value = self._exchange("create", {"episode_id": episode_id, "mode": mode})
        namespaces = value.get("namespaces")
        if (set(value) != {"status", "episode_id", "mode", "reward", "namespaces",
                           "guest_reset_ns", "namespace_setup_ns"}
                or value["status"] != "active" or value["episode_id"] != episode_id
                or value["mode"] != mode or value["reward"] is not None
                or not isinstance(namespaces, dict)
                or set(namespaces) != {"pid", "mount", "pid_one", "process_nonce"}
                or namespaces["pid_one"] is not True
                or not isinstance(namespaces["process_nonce"], str)
                or not re.fullmatch(r"[0-9a-f]{32}", namespaces["process_nonce"])
                or any(type(value[key]) is not int or value[key] < 0
                       for key in ("guest_reset_ns", "namespace_setup_ns"))
                or any(not isinstance(namespaces[key], str)
                       or not re.fullmatch(r"(pid|mnt):\[[0-9]+\]", namespaces[key])
                       for key in ("pid", "mount"))):
            self.state = "interrupted"
            raise ResidentInterrupted("invalid_episode_start")
        self.state = "active"
        self.reward = None
        return value

    def action(self, name, arguments):
        if self.state != "active":
            raise ValueError("episode_not_active")
        return self._exchange("action", {"name": name, "input": arguments})

    def submit(self):
        if self.state != "active":
            raise ValueError("episode_not_active")
        value = self._exchange("submit", {})
        if value != {"status": "pending", "reward": None}:
            self.state = "interrupted"
            raise ResidentInterrupted("invalid_submit_status")
        self.state = "pending"
        return value

    def verify(self):
        if self.state != "pending":
            raise ValueError("episode_not_submitted")
        raise ResidentPending("hidden_case_verifier_not_integrated")

    def run_case(self, code, *, expected_branch_sha256):
        """Host-private operation; the policy-facing adapter never exposes it."""
        if self.state != "pending":
            raise ValueError("episode_not_submitted")
        value = self._exchange("case", {"code": code})
        if value.get("branch_sha256") != expected_branch_sha256:
            self.state = "interrupted"
            raise ResidentInterrupted("wrong_branch_case_result")
        return self._parse_case_payload(
            {key: value[key] for key in value if key != "branch_sha256"})

    def run_case_batch(self, codes, *, expected_branch_sha256):
        """One host-only transport call; every case still forks a new namespace."""
        if self.state != "pending":
            raise ValueError("episode_not_submitted")
        value = self._exchange("case_batch", {"codes": codes})
        if (set(value) != {"branch_sha256", "cases"}
                or value["branch_sha256"] != expected_branch_sha256
                or not isinstance(value["cases"], list)
                or len(value["cases"]) != len(codes)):
            self.state = "interrupted"
            raise ResidentInterrupted("invalid_or_wrong_branch_batch_result")
        return [self._parse_case_payload(case) for case in value["cases"]]

    def _parse_case_payload(self, value):
        if (set(value) != {"return_code", "stdout_b64", "truncated",
                           "output_over_batch_cap", "guest_case_ns"}
                or type(value["return_code"]) is not int
                or not -255 <= value["return_code"] <= 255
                or type(value["truncated"]) is not bool
                or type(value["output_over_batch_cap"]) is not bool
                or not isinstance(value["stdout_b64"], str)
                or len(value["stdout_b64"]) > 16000
                or type(value["guest_case_ns"]) is not int
                or value["guest_case_ns"] < 0):
            self.state = "interrupted"
            raise ResidentInterrupted("invalid_case_result")
        try:
            stdout = base64.b64decode(value["stdout_b64"], validate=True)
        except ValueError as exc:
            self.state = "interrupted"
            raise ResidentInterrupted("invalid_case_stdout") from exc
        if len(stdout) > 12000:
            self.state = "interrupted"
            raise ResidentInterrupted("case_stdout_exceeds_bound")
        return {"return_code": value["return_code"], "stdout_bytes": stdout,
                "truncated": value["truncated"],
                "output_over_batch_cap": value["output_over_batch_cap"],
                "guest_case_ns": value["guest_case_ns"]}

    def close(self, *, completed=False):
        if self.state not in {"active", "pending"}:
            raise ValueError("no_open_episode")
        value = self._exchange("close", {"completed": completed})
        if (set(value) != {"status", "reward", "guest_cleanup_ns"}
                or value["status"] != "closed" or value["reward"] is not None
                or type(value["guest_cleanup_ns"]) is not int
                or value["guest_cleanup_ns"] < 0):
            self.state = "interrupted"
            raise ResidentInterrupted("invalid_episode_close")
        self.state = "idle"
        self.reward = None
        return value
