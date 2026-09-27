"""Bounded, exact-match edits shared by the host and the VM guest.

The guest runs this trusted source through a bounded, base64-only serial
transport. Policy-controlled strings are data, never shell or Python source.
"""

from __future__ import annotations

import base64
import errno
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import PurePosixPath


MAX_FILE_BYTES = 65536
MAX_EDIT_BYTES = 1024
# With the fixed inlined guest program, 600 raw JSON bytes encode to at most
# 800 base64 bytes and leave room under the 3,800-byte serial line cap.
MAX_ACTION_BYTES = 600
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def validate_action(action):
    """Return encoded edit bytes after validating the complete action shape."""
    if (not isinstance(action, dict)
            or set(action) != {"action", "path", "expected_file_sha256",
                               "old_text", "new_text"}
            or action["action"] != "replace_text"):
        raise ValueError("Invalid replace_text action")
    path = action["path"]
    if (not isinstance(path, str) or not path or len(path) > 512
            or path.startswith("/") or "\\" in path or "\x00" in path):
        raise ValueError("invalid_workspace_path")
    parsed = PurePosixPath(path)
    if (str(parsed) != path or any(part in {".", "..", ".git"}
                                   for part in parsed.parts)):
        raise ValueError("invalid_workspace_path")
    digest = action["expected_file_sha256"]
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise ValueError("Invalid expected_file_sha256")
    old, new = action["old_text"], action["new_text"]
    if not isinstance(old, str) or not isinstance(new, str) or not old:
        raise ValueError("Replace text must be nonempty old text and string new text")
    try:
        old_bytes, new_bytes = old.encode("utf-8"), new.encode("utf-8")
        path.encode("utf-8")
    except UnicodeError as exc:
        raise ValueError("Replace text and path must be UTF-8") from exc
    if len(old_bytes) > MAX_EDIT_BYTES or len(new_bytes) > MAX_EDIT_BYTES:
        raise ValueError("Replace text exceeds 1 KiB field bound")
    serialized = json.dumps(action, sort_keys=True, separators=(",", ":"),
                            ensure_ascii=False).encode("utf-8")
    if len(serialized) > MAX_ACTION_BYTES:
        raise ValueError("Replace text exceeds bounded action transport")
    return old_bytes, new_bytes


def _open_parent(root, relative):
    """Walk every component with NOFOLLOW and return a held parent dirfd."""
    components = relative.split("/")
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in components[:-1]:
            following = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                dir_fd=directory)
            os.close(directory)
            directory = following
        return directory, components[-1]
    except BaseException:
        os.close(directory)
        raise


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


def _file_identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
            info.st_ctime_ns, info.st_mode, info.st_uid, info.st_gid,
            info.st_nlink)


def apply(root, action):
    """Atomically replace exactly one occurrence in an existing regular file.

    Hash/uniqueness conflicts are ordinary policy observations. Unsafe paths,
    symlinks, and I/O failures abort without replacing the target.
    """
    old, new = validate_action(action)
    try:
        parent, name = _open_parent(root, action["path"])
    except FileNotFoundError:
        return {"status": "conflict", "reason": "file_missing", "path": action["path"]}
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ValueError("Unsafe workspace path") from exc
        raise
    try:
        try:
            source = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                             dir_fd=parent)
        except FileNotFoundError:
            return {"status": "conflict", "reason": "file_missing", "path": action["path"]}
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise ValueError("Unsafe workspace path") from exc
            raise
        try:
            before = os.fstat(source)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise ValueError("Workspace target must be a regular file")
            data = _read_bounded(source)
            after = os.fstat(source)
        finally:
            os.close(source)
        if len(data) > MAX_FILE_BYTES:
            return {"status": "conflict", "reason": "file_too_large", "path": action["path"]}
        if (_file_identity(before) != _file_identity(after)
                or len(data) != after.st_size):
            return {"status": "conflict", "reason": "file_changed", "path": action["path"]}
        digest = hashlib.sha256(data).hexdigest()
        if digest != action["expected_file_sha256"]:
            return {"status": "conflict", "reason": "sha256_mismatch", "path": action["path"]}
        first = data.find(old)
        if first < 0 or data.find(old, first + 1) >= 0:
            return {"status": "conflict", "reason": "old_text_not_unique", "path": action["path"]}
        changed = data.replace(old, new, 1)
        if len(changed) > MAX_FILE_BYTES:
            return {"status": "conflict", "reason": "file_too_large", "path": action["path"]}
        # The original descriptor's inode must still be the current target.
        # A running actor has no direct shell action, but this also catches
        # common concurrent-write races before the atomic rename.
        try:
            current = os.stat(name, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            return {"status": "conflict", "reason": "file_changed", "path": action["path"]}
        if _file_identity(before) != _file_identity(current):
            return {"status": "conflict", "reason": "file_changed", "path": action["path"]}
        temp_name = ".fpb-replace-" + os.urandom(12).hex()
        temp = os.open(temp_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                       stat.S_IMODE(before.st_mode), dir_fd=parent)
        try:
            temp_stat = os.fstat(temp)
            if (temp_stat.st_uid, temp_stat.st_gid) != (before.st_uid, before.st_gid):
                # If ownership cannot be preserved, fail before the rename.
                os.fchown(temp, before.st_uid, before.st_gid)
            os.fchmod(temp, stat.S_IMODE(before.st_mode))
            with os.fdopen(temp, "wb", closefd=False) as stream:
                stream.write(changed)
                stream.flush()
            os.fsync(temp)
            try:
                latest = os.stat(name, dir_fd=parent, follow_symlinks=False)
            except FileNotFoundError:
                return {"status": "conflict", "reason": "file_changed", "path": action["path"]}
            if _file_identity(before) != _file_identity(latest):
                return {"status": "conflict", "reason": "file_changed", "path": action["path"]}
            # A replacement of this entry by a symlink is safe: renameat
            # replaces the link itself, never follows it.
            os.replace(temp_name, name, src_dir_fd=parent, dst_dir_fd=parent)
        finally:
            os.close(temp)
            try:
                os.unlink(temp_name, dir_fd=parent)
            except FileNotFoundError:
                pass
        return {"path": action["path"], "sha256": hashlib.sha256(changed).hexdigest()}
    finally:
        os.close(parent)


def _guest_main():
    try:
        if len(sys.argv) != 2:
            raise ValueError("Invalid payload")
        payload = base64.b64decode(sys.argv[1], validate=True)
        if len(payload) > 4096:
            raise ValueError("Payload exceeds bound")
        action = json.loads(payload)
        result = apply("/workspace", action)
    except ValueError:
        result = {"status": "error", "reason": "adapter_error",
                  "error_type": "ValueError"}
    encoded = base64.b64encode(json.dumps(result, sort_keys=True,
                                          separators=(",", ":")).encode("utf-8"))
    print("FPB_REPLACE_RESULT=" + encoded.decode("ascii"))


def guest_program():
    """Return the committed guest helper; its bytes are stable across Python versions.

    The source digest binds this payload, and the guest digest binds its decoded
    program. When shared logic changes, regenerate this literal and update the
    pinned v2 binding. Tests compare parsed syntax trees, not unparse output.
    """
    return (
        "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"
        "kRXTjgmeOCZDqOKxcbg4dv+Iajnh/wCtisEZ"
    )


def helper_binding():
    """Bind both source bytes and the exact compressed-program plaintext."""
    import zlib
    from pathlib import Path

    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    guest_sha = hashlib.sha256(zlib.decompress(base64.b64decode(guest_program()))).hexdigest()
    return {"source_sha256": source_sha, "guest_program_sha256": guest_sha}


if __name__ == "__main__":
    _guest_main()
