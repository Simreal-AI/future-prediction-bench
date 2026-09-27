"""Self-contained, opt-in read-only virtio-serial agent for the pinned guest.

This file is uploaded by a trusted host and run inside the already-isolated
Linux VM. It has no shell, write, submit, verifier, or reward operation.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import stat
import struct
from pathlib import PurePosixPath


PORT = "/dev/fpb.control"
WORKSPACE = "/workspace"
MAX_REQUEST = 8192
MAX_RESPONSE = 65536


class FileTooLarge(ValueError):
    pass


class NotRegularFile(ValueError):
    pass


def _read_exact(fd: int, count: int):
    chunks = []
    while count:
        chunk = os.read(fd, count)
        if not chunk:
            raise EOFError("virtio port closed")
        chunks.append(chunk)
        count -= len(chunk)
    return b"".join(chunks)


def _write_all(fd: int, data: bytes):
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise EOFError("virtio port closed")
        view = view[written:]


def _strict_json(data: bytes):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    def constant(value):
        raise ValueError("non-finite JSON constant")
    return json.loads(data.decode("utf-8"), object_pairs_hook=pairs,
                      parse_constant=constant)


def _relative_parts(value: str):
    if (not isinstance(value, str) or not value or len(value) > 512
            or value.startswith("/") or "\\" in value or "\x00" in value):
        raise ValueError("invalid_path")
    path = PurePosixPath(value)
    if str(path) != value or any(part in (".", "..", ".git") for part in path.parts):
        raise ValueError("invalid_path")
    return path.parts


def _read_workspace_file(relative: str):
    parts = _relative_parts(relative)
    directory = os.open(WORKSPACE, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=directory)
            os.close(directory)
            directory = child
        try:
            entry = os.stat(parts[-1], dir_fd=directory, follow_symlinks=False)
            if stat.S_ISLNK(entry.st_mode):
                raise ValueError("invalid_file")
            if not stat.S_ISREG(entry.st_mode):
                raise NotRegularFile("not_regular_file")
            file_fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                              dir_fd=directory)
        except FileNotFoundError:
            return None
        try:
            before = os.fstat(file_fd)
            if not stat.S_ISREG(before.st_mode):
                raise NotRegularFile("not_regular_file")
            if before.st_size > 50_000_000:
                raise FileTooLarge("file_too_large")
            digest = hashlib.sha256()
            head = bytearray()
            total = 0
            while True:
                chunk = os.read(file_fd, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                total += len(chunk)
                if len(head) < 16000:
                    head.extend(chunk[:16000 - len(head)])
                if total > 50_000_000:
                    raise FileTooLarge("file_too_large")
            after = os.fstat(file_fd)
            if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                    != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                    or total != after.st_size):
                raise RuntimeError("file_changed_during_read")
            return {"path": relative, "size": total, "sha256": digest.hexdigest(),
                    "head_b64": base64.b64encode(head).decode("ascii")}
        finally:
            os.close(file_fd)
    finally:
        os.close(directory)


def _response(request, *, ok, data=None, error=None):
    payload = {"v": 1, "session": request["session"], "seq": request["seq"],
               "op": request["op"], "ok": ok, "data": data or {}, "error": error}
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False,
                         allow_nan=False).encode("utf-8")
    if not 1 <= len(encoded) <= MAX_RESPONSE:
        raise RuntimeError("response_too_large")
    return struct.pack(">I", len(encoded)) + encoded


def serve(fd: int):
    session = None
    expected_seq = 1
    while True:
        try:
            size = struct.unpack(">I", _read_exact(fd, 4))[0]
        except EOFError:
            return
        if not 1 <= size <= MAX_REQUEST:
            raise RuntimeError("request_length_invalid")
        request = _strict_json(_read_exact(fd, size))
        if (not isinstance(request, dict)
                or set(request) != {"v", "session", "seq", "op", "args"}
                or request["v"] != 1
                or not isinstance(request["session"], str)
                or re.fullmatch(r"[0-9a-f]{32}", request["session"]) is None
                or type(request["seq"]) is not int or request["seq"] != expected_seq
                or not isinstance(request["args"], dict)
                or request["op"] not in {"PING", "READ_FILE", "STOP"}
                or session is not None and request["session"] != session):
            raise RuntimeError("request_contract_invalid")
        if session is None:
            session = request["session"]
        expected_seq += 1
        operation, arguments = request["op"], request["args"]
        if operation == "PING":
            if arguments:
                raise RuntimeError("ping_arguments_invalid")
            reply = _response(request, ok=True, data={"pong": True})
        elif operation == "READ_FILE":
            if set(arguments) != {"path"}:
                raise RuntimeError("read_arguments_invalid")
            try:
                data = _read_workspace_file(arguments["path"])
            except FileTooLarge:
                reply = _response(request, ok=False, error="file_too_large")
                _write_all(fd, reply)
                continue
            except NotRegularFile:
                reply = _response(request, ok=False, error="file_not_regular")
                _write_all(fd, reply)
                continue
            except (OSError, ValueError) as exc:
                if isinstance(exc, FileNotFoundError):
                    data = None
                else:
                    reply = _response(request, ok=False, error="invalid_path_or_file")
                    _write_all(fd, reply)
                    continue
            if data is None:
                reply = _response(request, ok=False, error="file_missing")
            else:
                reply = _response(request, ok=True, data=data)
        else:
            if arguments:
                raise RuntimeError("stop_arguments_invalid")
            reply = _response(request, ok=True)
            _write_all(fd, reply)
            return
        _write_all(fd, reply)


def main():
    fd = os.open(PORT, os.O_RDWR | os.O_NOCTTY)
    try:
        serve(fd)
    finally:
        os.close(fd)


if __name__ == "__main__":
    main()
