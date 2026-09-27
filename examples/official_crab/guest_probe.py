"""Trusted, bounded Linux probe for unmodified Crab soft-dirty inspection.

Only the stopped child worker and one fixed file constitute the measured
state. The supervisor, shell, kernel and other guest processes are outside
this inspection contract. The host snapshots the whole VM conservatively.
"""
from __future__ import annotations

import argparse
import base64
import ctypes
import hashlib
import json
import mmap
import os
from pathlib import Path
import select
import signal
import socket
import struct
import sys
import time

sys.path.insert(0, "/")
import fpb_crab_process_monitor as monitor

SOCKET = "/fpb-crab-probe.sock"
FILE = Path("/workspace/.fpb-crab-state")
LIMIT = 4096


def _read(fd, size):
    result = b""
    while len(result) < size:
        if not select.select([fd], [], [], 5)[0]:
            raise RuntimeError("worker_timeout")
        block = os.read(fd, size - len(result))
        if not block:
            raise RuntimeError("worker_closed")
        result += block
    return result


class Probe:
    def __init__(self):
        commands, commands_out = os.pipe()
        replies, replies_out = os.pipe()
        self.pid = os.fork()
        if self.pid == 0:
            os.close(commands_out)
            os.close(replies)
            page = mmap.mmap(-1, monitor.PAGE_SIZE,
                             flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
            struct.pack_into("q", page, 0, 123)
            address = ctypes.addressof(ctypes.c_char.from_buffer(page))
            os.write(replies_out, struct.pack("Q", address))
            while True:
                command = _read(commands, 8)
                page[:8] = command
                os.write(replies_out, b"K")
        os.close(commands)
        os.close(replies_out)
        self.command_fd, self.reply_fd = commands_out, replies
        self.address = struct.unpack("Q", _read(replies, 8))[0]
        self._stop()
        self.identity = self._identity()
        FILE.write_text("seed", encoding="ascii")
        self.baseline_file = None
        # A present pagemap is insufficient: some ARM kernels expose it but
        # do not track soft-dirty writes. Calibrate against a known mutation.
        monitor.clear_soft_dirty(self.pid)
        self.call({"op": "memory", "value": 124})
        self.soft_dirty_supported = self.pid in monitor.dirty_pids({self.pid})
        self.call({"op": "memory", "value": 123})
        monitor.clear_soft_dirty(self.pid)

    def _identity(self):
        # comm may contain spaces or parentheses; starttime is field 22.
        tail = Path(f"/proc/{self.pid}/stat").read_text().rsplit(")", 1)[1].split()
        if tail[0] != "T":
            raise RuntimeError("worker_not_stopped")
        return int(tail[19])

    def _stop(self):
        os.kill(self.pid, signal.SIGSTOP)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status = Path(f"/proc/{self.pid}/status").read_text()
            if "State:\tT" in status:
                return
            time.sleep(0.001)
        raise RuntimeError("worker_stop_timeout")

    def value(self):
        with open(f"/proc/{self.pid}/mem", "rb", buffering=0) as stream:
            stream.seek(self.address)
            raw = stream.read(8)
        if len(raw) != 8:
            raise RuntimeError("short_memory_read")
        return struct.unpack("q", raw)[0]

    def file_digest(self):
        stat = FILE.lstat()
        if not FILE.is_file() or FILE.is_symlink() or stat.st_nlink != 1:
            raise RuntimeError("invalid_probe_file")
        return {"sha256": hashlib.sha256(FILE.read_bytes()).hexdigest(),
                "mode": stat.st_mode, "uid": stat.st_uid, "gid": stat.st_gid,
                "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                "ctime_ns": stat.st_ctime_ns}

    def inspect(self):
        started = time.monotonic_ns()
        try:
            if not self.soft_dirty_supported:
                raise RuntimeError("soft_dirty_calibration_failed")
            if self._identity() != self.identity:
                raise RuntimeError("worker_identity_changed")
            ranges = monitor.parse_writable_ranges(self.pid)
            if not ranges:
                raise RuntimeError("no_writable_ranges")
            # Upstream returns False on a short pagemap read. Establish that
            # every range is readable first, so unknown cannot become clean.
            with open(f"/proc/{self.pid}/pagemap", "rb", buffering=0) as stream:
                for start, end in ranges:
                    first = start // monitor.PAGE_SIZE
                    count = (end + monitor.PAGE_SIZE - 1) // monitor.PAGE_SIZE - first
                    stream.seek(first * monitor.PAGEMAP_ENTRY_SIZE)
                    if len(stream.read(count * monitor.PAGEMAP_ENTRY_SIZE)) != count * monitor.PAGEMAP_ENTRY_SIZE:
                        raise RuntimeError("short_pagemap_read")
            changed = self.pid in monitor.dirty_pids({self.pid})
            result = {"known": True, "process_changed": changed,
                      "filesystem_changed": self.file_digest() != self.baseline_file}
        except (OSError, RuntimeError, ValueError) as exc:
            result = {"known": False, "process_changed": True,
                      "filesystem_changed": True, "error": type(exc).__name__}
        result["inspection_ns"] = time.monotonic_ns() - started
        return result

    def call(self, request):
        op = request.get("op")
        if op == "step":
            action = request.get("action")
            if not isinstance(action, dict) or action.get("op") not in {"state", "transient", "file", "memory"}:
                raise ValueError("invalid_step_action")
            if type(request.get("inspect")) is not bool:
                raise ValueError("inspect_flag_required")
            self.call(action)
            inspection = self.inspect() if request["inspect"] else None
            state = self.call({"op": "state"})
            # This is a speculative baseline reset at a quiescent commit
            # boundary. The host may continue only after a successful save,
            # or a known-clean skip. Any uncertainty/save failure aborts VM.
            self.call({"op": "baseline"})
            return {"inspection": inspection, "state": state, "baseline_prepared": True}
        if op == "capabilities":
            return {"soft_dirty_known_write_detected": self.soft_dirty_supported,
                    "machine": os.uname().machine,
                    "inspection_scope": "one_stopped_worker_and_one_file"}
        if op == "baseline":
            if self._identity() != self.identity:
                raise RuntimeError("worker_identity_changed")
            monitor.clear_soft_dirty(self.pid)
            self.baseline_file = self.file_digest()
            return {"baseline": True}
        if op == "inspect":
            return self.inspect()
        if op == "state":
            return {"value": self.value(), "file": FILE.read_text(encoding="ascii"),
                    "worker_identity": self._identity()}
        if op == "memory":
            value = request.get("value")
            if type(value) is not int or not -10000 <= value <= 10000:
                raise ValueError("invalid_value")
            os.kill(self.pid, signal.SIGCONT)
            os.write(self.command_fd, struct.pack("q", value))
            if _read(self.reply_fd, 1) != b"K":
                raise RuntimeError("worker_ack_invalid")
            self._stop()
            return {"value": self.value()}
        if op == "file":
            value = request.get("value")
            if value not in {"seed", "changed", "corrupted"}:
                raise ValueError("invalid_file_value")
            FILE.write_text(value, encoding="ascii")
            return {"file": value}
        if op == "transient":
            transient = FILE.with_name(".fpb-crab-transient")
            transient.write_text("temporary", encoding="ascii")
            transient.unlink()
            return {"transient_removed": True}
        if op == "kill_worker":
            os.kill(self.pid, signal.SIGKILL)
            os.waitpid(self.pid, 0)
            return {"worker_killed": True}
        raise ValueError("unknown_operation")


def receive(sock):
    chunks = bytearray()
    while not chunks.endswith(b"\n"):
        block = sock.recv(1024)
        if not block or len(chunks) + len(block) > LIMIT:
            raise ValueError("invalid_frame")
        chunks.extend(block)
    return json.loads(chunks)


def serve():
    probe = Probe()
    Path(SOCKET).unlink(missing_ok=True)
    with socket.socket(socket.AF_UNIX) as listener:
        listener.bind(SOCKET)
        os.chmod(SOCKET, 0o600)
        listener.listen(1)
        while True:
            connection, _ = listener.accept()
            with connection:
                connection.settimeout(5)
                try:
                    result = probe.call(receive(connection))
                except Exception as exc:
                    result = {"error": str(exc)}
                connection.sendall(json.dumps(result, allow_nan=False).encode() + b"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--call")
    args = parser.parse_args()
    if args.serve:
        serve()
    else:
        with socket.socket(socket.AF_UNIX) as client:
            client.settimeout(10)
            client.connect(SOCKET)
            client.sendall(base64.b64decode(args.call, validate=True) + b"\n")
            response = receive(client)
        print("FPB_CRAB_RESULT:" + base64.b64encode(json.dumps(response).encode()).decode())
