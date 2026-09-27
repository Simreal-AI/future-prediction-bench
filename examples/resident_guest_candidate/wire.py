"""Bounded, versioned JSON frames for the experimental resident guest.

This module has no dependency on the release package. Each connection carries
one request and one response; the 4-byte big-endian length is not trusted.
"""

from __future__ import annotations

import json
import re
import struct


VERSION = 1
MAX_REQUEST = 1024
MAX_BATCH_REQUEST = 2048
MAX_RESPONSE = 16384
MAX_PATH = 200
MAX_CASES = 14
# Each verifier case has a separate 10-second candidate deadline and up to
# 15 seconds for nested supervisor cleanup. Transport must outlive the case,
# and a 14-case batch must outlive the sum; it must never turn an ordinary
# candidate timeout into an infrastructure-pending reward.
CASE_REQUEST_TIMEOUT = 25.0
BATCH_REQUEST_TIMEOUT = 290.0
REQUEST_OPS = {"hello", "create", "action", "submit", "case", "case_batch", "close"}
ACTION_NAMES = {"read_file", "replace_text"}
IDENTITY = "trusted_resident_guest_mount_pid_overlay_experimental_v1"


class ProtocolError(RuntimeError):
    pass


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError("duplicate_json_key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ProtocolError("nonfinite_json_number")


def json_bytes(value, *, limit):
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False,
                         separators=(",", ":"), sort_keys=True).encode("utf-8")
    if len(encoded) > limit:
        raise ProtocolError("frame_too_large")
    return encoded


def parse_json(data, *, limit):
    if len(data) > limit:
        raise ProtocolError("frame_too_large")
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_pairs,
                           parse_constant=_reject_constant)
    except (UnicodeError, ValueError) as exc:
        raise ProtocolError("invalid_json") from exc
    if not isinstance(value, dict):
        raise ProtocolError("object_required")
    return value


def frame(value, *, limit):
    payload = json_bytes(value, limit=limit)
    return struct.pack("!I", len(payload)) + payload


def unframe(data, *, limit):
    if not isinstance(data, bytes) or len(data) < 4:
        raise ProtocolError("short_frame")
    length = struct.unpack("!I", data[:4])[0]
    if length > limit or len(data) != length + 4:
        raise ProtocolError("invalid_frame_length")
    return parse_json(data[4:], limit=limit)


def read_frame(sock, *, limit):
    def exact(size):
        chunks = []
        while size:
            block = sock.recv(size)
            if not block:
                raise ProtocolError("truncated_frame")
            chunks.append(block)
            size -= len(block)
        return b"".join(chunks)

    length = struct.unpack("!I", exact(4))[0]
    if length > limit:
        raise ProtocolError("frame_too_large")
    return parse_json(exact(length), limit=limit)


def send_frame(sock, value, *, limit):
    sock.sendall(frame(value, limit=limit))


def validate_request(value):
    if (set(value) != {"v", "seq", "op", "args"}
            or type(value["v"]) is not int or value["v"] != VERSION
            or type(value["seq"]) is not int or not 0 <= value["seq"] < 2**31
            or not isinstance(value["op"], str) or value["op"] not in REQUEST_OPS
            or not isinstance(value["args"], dict)):
        raise ProtocolError("invalid_request")
    op, args = value["op"], value["args"]
    json_bytes(value, limit=MAX_BATCH_REQUEST if op == "case_batch" else MAX_REQUEST)
    if op in {"hello", "submit"} and args:
        raise ProtocolError("unexpected_args")
    if op == "close" and (set(args) != {"completed"}
                           or type(args["completed"]) is not bool):
        raise ProtocolError("invalid_close_args")
    if op == "create" and (set(args) != {"episode_id", "mode"}
                           or not isinstance(args["episode_id"], str)
                           or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", args["episode_id"])
                           or not isinstance(args["mode"], str)
                           or args["mode"] not in {"repair", "baseline"}):
            raise ProtocolError("invalid_create_args")
    if op == "case" and (set(args) != {"code"}
                         or not isinstance(args["code"], str)
                         or not 1 <= len(args["code"].encode("utf-8")) <= 512
                         or "\x00" in args["code"]):
        raise ProtocolError("invalid_case_args")
    if op == "case_batch":
        codes = args.get("codes")
        if (set(args) != {"codes"} or not isinstance(codes, list)
                or not 1 <= len(codes) <= MAX_CASES
                or any(not isinstance(code, str)
                       or not 1 <= len(code.encode("utf-8")) <= 512
                       or "\x00" in code for code in codes)):
            raise ProtocolError("invalid_case_batch_args")
    if op == "action":
        if (set(args) != {"name", "input"}
                or not isinstance(args["name"], str)
                or args["name"] not in ACTION_NAMES
                or not isinstance(args["input"], dict)):
            raise ProtocolError("invalid_action_args")
        inp = args["input"]
        if args["name"] == "read_file":
            if set(inp) != {"path"}:
                raise ProtocolError("invalid_read_args")
            relative_path(inp["path"])
        else:
            if (set(inp) != {"path", "expected_sha256", "old", "new"}
                    or not isinstance(inp["expected_sha256"], str)
                    or not re.fullmatch(r"[0-9a-f]{64}", inp["expected_sha256"])
                    or any(not isinstance(inp[key], str) for key in ("old", "new"))
                    or not inp["old"] or len(inp["old"].encode("utf-8")) > 256
                    or len(inp["new"].encode("utf-8")) > 256):
                raise ProtocolError("invalid_replace_args")
            relative_path(inp["path"])
    return value


def validate_response(value, *, seq):
    if (set(value) != {"v", "seq", "ok", "value", "error"}
            or value["v"] != VERSION or type(value["seq"]) is not int
            or value["seq"] != seq or type(value["ok"]) is not bool):
        raise ProtocolError("invalid_response")
    if value["ok"]:
        if value["error"] is not None or not isinstance(value["value"], dict):
            raise ProtocolError("invalid_success_response")
    elif value["value"] is not None or not isinstance(value["error"], str) or len(value["error"]) > 120:
        raise ProtocolError("invalid_error_response")
    return value


def response(seq, *, value=None, error=None):
    return {"v": VERSION, "seq": seq, "ok": error is None,
            "value": value if error is None else None, "error": error}


def relative_path(path):
    if (not isinstance(path, str) or not 1 <= len(path.encode("utf-8")) <= MAX_PATH
            or path.startswith("/") or "\\" in path or "\x00" in path):
        raise ProtocolError("invalid_relative_path")
    parts = path.split("/")
    if any(part in {"", ".", "..", ".git"} for part in parts):
        raise ProtocolError("invalid_relative_path")
    return parts
