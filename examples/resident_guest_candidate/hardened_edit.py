"""Private, bounded exact edit for the resident guest's overlay workspace.

This is self-contained because the experimental guest does not install the
release package. Policy input is data only. Conflicts return ``accepted=False``;
unsafe targets and unexpected I/O raise so the caller can distinguish a policy
error from an infrastructure interruption.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import stat
from pathlib import PurePosixPath


MAX_FILE_BYTES = 65536
MAX_EDIT_BYTES = 256
MAX_INPUT_BYTES = 600
MAX_PATH_BYTES = 200
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class UnsafeTarget(ValueError):
    """The requested target is outside the supported safe-file contract."""


def validate_input(value):
    """Validate the wire-level replace payload before transport or mutation."""
    if not isinstance(value, dict) or set(value) != {
        "path", "expected_sha256", "old", "new"
    }:
        raise ValueError("invalid_replace_input")
    path, digest, old, new = (value[key] for key in
                              ("path", "expected_sha256", "old", "new"))
    if not all(isinstance(item, str) for item in (path, digest, old, new)):
        raise ValueError("replace_strings_required")
    try:
        path_bytes, old_bytes, new_bytes = (item.encode("utf-8") for item in
                                            (path, old, new))
    except UnicodeError as exc:
        raise ValueError("replace_utf8_required") from exc
    if (not 1 <= len(path_bytes) <= MAX_PATH_BYTES or path in {".", ".."}
            or path.startswith("/")
            or "\\" in path or "\x00" in path):
        raise ValueError("invalid_workspace_path")
    parsed = PurePosixPath(path)
    if (str(parsed) != path or any(part in {".", "..", ".git"}
                                   for part in parsed.parts)):
        raise ValueError("invalid_workspace_path")
    if _SHA256.fullmatch(digest) is None:
        raise ValueError("invalid_expected_sha256")
    if not old_bytes or len(old_bytes) > MAX_EDIT_BYTES or len(new_bytes) > MAX_EDIT_BYTES:
        raise ValueError("replace_field_exceeds_bound")
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"),
                            ensure_ascii=False, allow_nan=False).encode("utf-8")
    if len(serialized) > MAX_INPUT_BYTES:
        raise ValueError("replace_input_exceeds_transport_bound")
    return old_bytes, new_bytes


def _conflict(reason):
    return {"accepted": False, "reason": reason}


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
            info.st_ctime_ns, info.st_mode, info.st_uid, info.st_gid,
            info.st_nlink)


def _parent(root, path):
    parts = path.split("/")
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            following = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                dir_fd=directory)
            os.close(directory)
            directory = following
        return directory, parts[-1]
    except BaseException:
        os.close(directory)
        raise


def _stat_target(parent, name):
    return os.stat(name, dir_fd=parent, follow_symlinks=False)


def _read_bounded(fd):
    chunks = []
    remaining = MAX_FILE_BYTES + 1
    while remaining:
        chunk = os.read(fd, min(remaining, 65536))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _unsafe_or_raise(exc):
    if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
        raise UnsafeTarget("unsafe_workspace_path") from exc
    raise exc


def apply(root, value):
    """Replace one nonoverlapping *and* overlapping-unique anchor in a file.

    This is best-effort stale-state detection followed by atomic same-directory
    rename, not strict compare-and-swap against an actively concurrent writer.
    """
    old, new = validate_input(value)
    try:
        parent, name = _parent(root, value["path"])
    except FileNotFoundError:
        return _conflict("file_missing")
    except OSError as exc:
        _unsafe_or_raise(exc)
    try:
        try:
            source = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                             dir_fd=parent)
        except FileNotFoundError:
            return _conflict("file_missing")
        except OSError as exc:
            _unsafe_or_raise(exc)
        try:
            before = os.fstat(source)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise UnsafeTarget("regular_single_link_file_required")
            data = _read_bounded(source)
            after = os.fstat(source)
        finally:
            os.close(source)
        if len(data) > MAX_FILE_BYTES:
            return _conflict("file_too_large")
        if _identity(before) != _identity(after) or len(data) != after.st_size:
            return _conflict("file_changed")
        try:
            data.decode("utf-8")
        except UnicodeError:
            return _conflict("non_utf8_file")
        if hashlib.sha256(data).hexdigest() != value["expected_sha256"]:
            return _conflict("sha256_mismatch")
        first = data.find(old)
        if first < 0 or data.find(old, first + 1) >= 0:
            return _conflict("old_text_not_unique")
        changed = data.replace(old, new, 1)
        if len(changed) > MAX_FILE_BYTES:
            return _conflict("file_too_large")
        try:
            current = _stat_target(parent, name)
        except FileNotFoundError:
            return _conflict("file_changed")
        if _identity(before) != _identity(current):
            return _conflict("file_changed")
        temp_name = ".fpb-resident-edit-" + os.urandom(12).hex()
        temp = os.open(temp_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                       stat.S_IMODE(before.st_mode), dir_fd=parent)
        try:
            temp_info = os.fstat(temp)
            if (temp_info.st_uid, temp_info.st_gid) != (before.st_uid, before.st_gid):
                os.fchown(temp, before.st_uid, before.st_gid)
            os.fchmod(temp, stat.S_IMODE(before.st_mode))
            with os.fdopen(temp, "wb", closefd=False) as stream:
                stream.write(changed)
                stream.flush()
            os.fsync(temp)
            try:
                latest = _stat_target(parent, name)
            except FileNotFoundError:
                return _conflict("file_changed")
            if _identity(before) != _identity(latest):
                return _conflict("file_changed")
            # A concurrent symlink swap is replaced as an entry, never followed.
            os.replace(temp_name, name, src_dir_fd=parent, dst_dir_fd=parent)
        finally:
            os.close(temp)
            try:
                os.unlink(temp_name, dir_fd=parent)
            except FileNotFoundError:
                pass
        return {"accepted": True, "sha256": hashlib.sha256(changed).hexdigest(),
                "bytes": len(changed)}
    finally:
        os.close(parent)
