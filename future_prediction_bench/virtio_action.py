"""Bounded host RPC client for the opt-in, read-only virtio-serial probe.

This is a trusted host transport, not a policy tool. Rewards and hidden
expected outputs never enter this channel. The initial protocol supports
only PING, READ_FILE, and a host-owned STOP for agent lifecycle.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import secrets
import select
import socket
import struct
import threading
import time

from .coding_env import _relative_file
from .http import strict_json_loads


VERSION = 1
MAX_REQUEST = 8192
MAX_RESPONSE = 65536


class VirtioActionError(RuntimeError):
    """Transport/protocol failure; never score it as a policy loss."""


class VirtioSerialReadFallback(RuntimeError):
    """A regular file exceeded the RPC cap; preserve serial read semantics."""


class VirtioActionClient:
    def __init__(self, runtime, *, timeout=10.0):
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= 300:
            raise ValueError("timeout must be in (0, 300]")
        self.socket = runtime.action_port_socket()
        self.socket.setblocking(False)
        self.timeout = float(timeout)
        self.session = secrets.token_hex(16)
        self.sequence = 0
        self.failed = False
        self.stopped = False
        self._request_lock = threading.Lock()
        self.metrics = {"requests": 0, "request_seconds": 0.0,
                        "bytes_sent": 0, "bytes_received": 0}

    def _fail_protocol(self, reason):
        self.failed = True
        try:
            self.socket.close()
        except OSError:
            pass
        raise VirtioActionError(reason)

    def _wait(self, *, readable=False, writable=False, deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise VirtioActionError("action_port_timed_out")
        ready_read, ready_write, _ = select.select(
            [self.socket] if readable else [], [self.socket] if writable else [], [], remaining)
        if readable and not ready_read or writable and not ready_write:
            raise VirtioActionError("action_port_timed_out")

    def _send_all(self, data: bytes, deadline):
        view = memoryview(data)
        while view:
            self._wait(writable=True, deadline=deadline)
            count = self.socket.send(view)
            if count <= 0:
                raise VirtioActionError("action_port_closed")
            view = view[count:]
        self.metrics["bytes_sent"] += len(data)

    def _read_exact(self, count: int, deadline):
        parts = []
        remaining = count
        while remaining:
            self._wait(readable=True, deadline=deadline)
            chunk = self.socket.recv(remaining)
            if not chunk:
                raise VirtioActionError("action_port_closed")
            parts.append(chunk)
            remaining -= len(chunk)
        self.metrics["bytes_received"] += count
        return b"".join(parts)

    def request(self, operation: str, arguments=None, *, timeout=None):
        """Send one request; concurrent host callers are rejected immediately."""
        if not self._request_lock.acquire(blocking=False):
            raise VirtioActionError("action_port_concurrent_request_unsupported")
        try:
            return self._request_unlocked(operation, arguments, timeout=timeout)
        finally:
            self._request_lock.release()

    def _request_unlocked(self, operation: str, arguments=None, *, timeout=None):
        if self.failed or self.stopped:
            raise VirtioActionError("action_port_session_unavailable")
        if operation not in {"PING", "READ_FILE", "STOP"}:
            raise ValueError("Unsupported read-only RPC operation")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise ValueError("RPC arguments must be an object")
        duration = self.timeout if timeout is None else timeout
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not 0 < duration <= 300:
            raise ValueError("timeout must be in (0, 300]")
        sequence = self.sequence + 1
        message = {"v": VERSION, "session": self.session, "seq": sequence,
                   "op": operation, "args": arguments}
        payload = json.dumps(message, separators=(",", ":"), ensure_ascii=False,
                             allow_nan=False).encode("utf-8")
        if not 1 <= len(payload) <= MAX_REQUEST:
            raise ValueError("RPC request exceeds bound")
        started = time.monotonic()
        deadline = started + duration
        try:
            self._send_all(struct.pack(">I", len(payload)) + payload, deadline)
            size = struct.unpack(">I", self._read_exact(4, deadline))[0]
            if not 1 <= size <= MAX_RESPONSE:
                raise VirtioActionError("action_port_response_length_invalid")
            response = strict_json_loads(self._read_exact(size, deadline).decode("utf-8"))
            if (not isinstance(response, dict)
                    or set(response) != {"v", "session", "seq", "op", "ok", "data", "error"}
                    or response["v"] != VERSION
                    or response["session"] != self.session
                    or type(response["seq"]) is not int or response["seq"] != sequence
                    or response["op"] != operation
                    or type(response["ok"]) is not bool
                    or not isinstance(response["data"], dict)
                    or response["error"] is not None and not isinstance(response["error"], str)):
                raise VirtioActionError("action_port_response_mismatch")
            self.sequence = sequence
            self.metrics["requests"] += 1
            self.metrics["request_seconds"] += time.monotonic() - started
            if operation == "STOP" and response["ok"]:
                self.stopped = True
            return response
        except (OSError, UnicodeError, ValueError, VirtioActionError) as exc:
            if isinstance(exc, ValueError) and str(exc) in {"RPC request exceeds bound"}:
                raise
            self.failed = True
            try:
                self.socket.close()
            except OSError:
                pass
            if isinstance(exc, VirtioActionError):
                raise
            raise VirtioActionError("action_port_protocol_or_io_failure") from exc

    def ping(self):
        response = self.request("PING")
        if not response["ok"] or response["error"] is not None or response["data"] != {"pong": True}:
            self._fail_protocol("action_port_ping_failed")
        return True

    def read_file(self, relative_path: str):
        _relative_file(relative_path)
        response = self.request("READ_FILE", {"path": relative_path})
        if not response["ok"]:
            if response["error"] == "file_missing":
                raise ValueError("Workspace file not found")
            if response["error"] == "file_not_regular":
                raise ValueError("Workspace file not found")
            if response["error"] == "file_too_large":
                raise VirtioSerialReadFallback("action_port_read_serial_fallback")
            raise VirtioActionError("action_port_read_failed")
        data = response["data"]
        if (response["error"] is not None or set(data) != {"path", "size", "sha256", "head_b64"}
                or data["path"] != relative_path
                or type(data["size"]) is not int or not 0 <= data["size"] <= 50_000_000
                or not isinstance(data["sha256"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", data["sha256"])
                or not isinstance(data["head_b64"], str) or len(data["head_b64"]) > 22000):
            self._fail_protocol("action_port_read_result_invalid")
        try:
            head = base64.b64decode(data["head_b64"], validate=True)
        except ValueError:
            self._fail_protocol("action_port_read_result_invalid")
        if len(head) != min(data["size"], 16000):
            self._fail_protocol("action_port_read_result_invalid")
        if data["size"] <= 16000 and hashlib.sha256(head).hexdigest() != data["sha256"]:
            self._fail_protocol("action_port_read_result_invalid")
        return {"path": relative_path, "text": head.decode("utf-8", "replace"),
                "sha256": data["sha256"], "truncated": data["size"] > 16000}

    def stop(self):
        response = self.request("STOP")
        if not response["ok"] or response["error"] is not None or response["data"] != {}:
            self._fail_protocol("action_port_stop_failed")
        return True
