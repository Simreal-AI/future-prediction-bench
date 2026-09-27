"""A bounded, stateful coding adapter for the separate real-world RL track.

The policy can read and edit a task workspace and invoke one operator-defined
visible check. Hidden verification runs after submission in separate containers
and the host checks expected outputs. No direct policy-selected shell command,
network access, or host path is exposed; candidate code invoked by a fixed check
can still execute inside the isolated actor container.
"""

from __future__ import annotations

import hashlib
import json
import os
import ctypes
import sys
import selectors
import shutil
import stat
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from .http import strict_json_loads
from .realworld import AdapterInfrastructureError
from . import replace_text as _replace_text


class CodingRuntimeError(AdapterInfrastructureError):
    """An infrastructure failure, never a task failure or training reward."""


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _relative_file(value: str) -> PurePosixPath:
    if (not isinstance(value, str) or not value or len(value) > 512 or "\\" in value
            or "\x00" in value or value.startswith("/")):
        raise ValueError("invalid_workspace_path")
    path = PurePosixPath(value)
    if any(part in {".", "..", ".git"} for part in path.parts) or str(path) != value:
        raise ValueError("invalid_workspace_path")
    return path


def _workspace_files(root: Path, *, max_files=5000, max_bytes=50_000_000,
                     max_entries=10000):
    files, total, entries = [], 0, 0
    for path in sorted(root.rglob("*")):
        entries += 1
        if entries > max_entries:
            raise ValueError("Workspace exceeds entry limit")
        if path.is_symlink():
            raise ValueError("Workspace symlinks are unsupported")
        if ".git" in path.relative_to(root).parts:
            raise ValueError("Git metadata must not be exposed in a task workspace")
        if not path.is_file() and not path.is_dir():
            raise ValueError("Workspace supports only regular files and directories")
        if path.is_file():
            files.append(path)
            total += path.stat().st_size
            if len(files) > max_files or total > max_bytes:
                raise ValueError("Workspace exceeds file or byte limit")
    return files


def _workspace_digest(root: Path):
    digest = hashlib.sha256()
    files = set(_workspace_files(root))
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(b"F" if path in files else b"D")
        digest.update(stat.S_IMODE(path.stat().st_mode).to_bytes(2, "big"))
        if path in files:
            digest.update(_digest_bytes(path.read_bytes()).encode("ascii"))
    return digest.hexdigest()


def _clone_or_copy_tree(source: Path, target: Path):
    """Copy a bounded workspace, using APFS clonefile when the host supports it.

    Clonefile creates independent copy-on-write files, never hard links. The
    portable fallback is a real copy. Only the regular-file tree is copied;
    container process state and /tmp are outside this contract.
    """
    _workspace_files(source)
    if target.exists():
        raise FileExistsError(target)
    clonefile = None
    if sys.platform == "darwin":
        try:
            clonefile = ctypes.CDLL(None, use_errno=True).clonefile
            clonefile.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
            clonefile.restype = ctypes.c_int
        except AttributeError:
            clonefile = None
    target.mkdir(parents=True)
    cloned = copied = 0
    try:
        for path in sorted(source.rglob("*")):
            relative = path.relative_to(source)
            destination = target / relative
            if path.is_symlink():
                raise ValueError("Workspace symlinks are unsupported")
            if path.is_dir():
                destination.mkdir()
            elif path.is_file():
                if clonefile is not None and clonefile(os.fsencode(path), os.fsencode(destination), 0) == 0:
                    cloned += 1
                else:
                    shutil.copy2(path, destination)
                    copied += 1
            else:
                raise ValueError("Workspace supports only regular files and directories")
        for path in sorted(source.rglob("*"), reverse=True):
            if path.is_dir():
                os.chmod(target / path.relative_to(source), stat.S_IMODE(path.stat().st_mode))
        os.chmod(target, stat.S_IMODE(source.stat().st_mode))
        return {"reflink_files": cloned, "copied_files": copied}
    except BaseException:
        shutil.rmtree(target, ignore_errors=True)
        raise


def _run_docker(arguments, *, timeout):
    """Drain bounded CLI output so a task cannot exhaust the host's memory."""
    process = None
    try:
        process = subprocess.Popen(["docker", *arguments], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        outputs = {process.stdout: bytearray(), process.stderr: bytearray()}
        deadline = time.monotonic() + timeout
        with selectors.DefaultSelector() as selector:
            for stream in outputs:
                selector.register(stream, selectors.EVENT_READ)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise CodingRuntimeError("docker_cli_timed_out")
                for key, _ in selector.select(min(remaining, 0.5)):
                    data = os.read(key.fileobj.fileno(), 65536)
                    if not data:
                        selector.unregister(key.fileobj)
                        continue
                    output = outputs[key.fileobj]
                    if len(output) + len(data) > 262144:
                        raise CodingRuntimeError("docker_cli_output_limit_exceeded")
                    output.extend(data)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise CodingRuntimeError("docker_cli_timed_out")
        returncode = process.wait(timeout=remaining)
        return subprocess.CompletedProcess(["docker", *arguments], returncode,
                                           bytes(outputs[process.stdout]), bytes(outputs[process.stderr]))
    except (OSError, subprocess.TimeoutExpired):
        raise CodingRuntimeError("docker_unavailable_or_timed_out") from None
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        if process is not None:
            process.stdout.close()
            process.stderr.close()


class DockerCodingAdapter:
    """One trusted task instance. Use one adapter per real-world RL episode.

    The seed and verifier are operator-controlled paths. The verifier directory
    is never mounted in the actor container. The image is resolved to its local
    immutable SHA-256 ID; network pulls are forbidden.
    """

    def __init__(self, *, seed_dir, verifier_dir, image, output_root,
                 visible_check=("python3", "-m", "py_compile", "math_utils.py"),
                 command_timeout=20.0, verifier_workers=1):
        self.seed_dir = Path(seed_dir).resolve()
        self.verifier_dir = Path(verifier_dir).resolve()
        self.output_root = Path(output_root).resolve()
        if not self.seed_dir.is_dir() or not self.verifier_dir.is_dir():
            raise ValueError("Seed workspace and trusted verifier directories are required")
        if (self.seed_dir.is_relative_to(self.verifier_dir)
                or self.verifier_dir.is_relative_to(self.seed_dir)):
            raise ValueError("Seed and hidden verifier directories must be disjoint")
        if (self.output_root.is_relative_to(self.seed_dir)
                or self.output_root.is_relative_to(self.verifier_dir)
                or self.seed_dir.is_relative_to(self.output_root)
                or self.verifier_dir.is_relative_to(self.output_root)):
            raise ValueError("Output, seed, and verifier roots must be disjoint")
        if not isinstance(image, str) or not image.strip() or any(char.isspace() for char in image):
            raise ValueError("A locally available container image is required")
        if not isinstance(visible_check, (list, tuple)) or not visible_check or any(
            not isinstance(part, str) or not part or "\x00" in part for part in visible_check
        ):
            raise ValueError("Visible check must be a fixed nonempty argv list")
        if isinstance(command_timeout, bool) or not 0 < command_timeout <= 300:
            raise ValueError("command_timeout must be between 0 and 300 seconds")
        if type(verifier_workers) is not int or not 1 <= verifier_workers <= 8:
            raise ValueError("verifier_workers must be an integer between 1 and 8")
        self.image = image
        self.visible_check = tuple(visible_check)
        self.command_timeout = float(command_timeout)
        self.verifier_workers = verifier_workers
        self.instance_root = None
        self.workspace = None
        self.submitted_workspace = None
        self.container_name = None
        self.image_id = None
        self.submitted = False
        self.replace_text_enabled = False
        self.expected_replace_text_binding = None
        self.failed = False
        self.verified = None
        self.expected_binding = None
        self.task_sha256 = None
        self.checkpoints = []
        self.metrics = {"docker_start_seconds": 0.0, "read_seconds": 0.0,
                        "write_seconds": 0.0, "visible_check_seconds": 0.0,
                        "checkpoint_seconds": 0.0, "verify_seconds": 0.0,
                        "docker_starts": 0, "checkpoints_created": 0,
                        "read_actions_without_checkpoint": 0, "docker_exec_calls": 0,
                        "verifier_workers": verifier_workers, "verifier_cases": 0,
                        "reflink_files": 0, "copied_files": 0,
                        "branch_checkpoints_created": 0}

    def _copy_tree(self, source, target):
        counts = _clone_or_copy_tree(source, target)
        for key, value in counts.items():
            self.metrics[key] += value

    def _docker(self, arguments, *, timeout=None):
        return _run_docker(arguments, timeout=timeout or self.command_timeout)

    def _require_reset(self):
        if self.workspace is None:
            raise ValueError("Call reset first")
        if self.failed:
            raise CodingRuntimeError("coding_instance_failed")

    def _file(self, relative):
        self._require_reset()
        path = self.workspace.joinpath(*_relative_file(relative).parts)
        if not path.resolve().is_relative_to(self.workspace):
            raise ValueError("workspace_path_escape")
        return path

    def _checkpoint(self):
        """Create a restorable state only when workspace content has changed."""
        started = time.monotonic()
        content_sha = _workspace_digest(self.workspace)
        target = self.instance_root / "snapshots" / content_sha
        if self.checkpoints and self.checkpoints[-1]["workspace_sha256"] == content_sha:
            if not target.is_dir() or _workspace_digest(target) != content_sha:
                raise CodingRuntimeError("workspace_snapshot_changed")
            self.metrics["checkpoint_seconds"] += time.monotonic() - started
            return self.checkpoints[-1]
        if not target.exists():
            staging = self.instance_root / "snapshots" / (".staging-" + uuid.uuid4().hex)
            self._copy_tree(self.workspace, staging)
            if _workspace_digest(staging) != content_sha:
                shutil.rmtree(staging, ignore_errors=True)
                raise CodingRuntimeError("workspace_changed_during_checkpoint")
            os.replace(staging, target)
        elif _workspace_digest(target) != content_sha:
            raise CodingRuntimeError("workspace_snapshot_changed")
        record = {"workspace_sha256": content_sha, "snapshot_index": len(self.checkpoints),
                  "file_count": len(_workspace_files(self.workspace))}
        self.checkpoints.append(record)
        self.metrics["checkpoints_created"] += 1
        self.metrics["checkpoint_seconds"] += time.monotonic() - started
        return record

    def _stop_actor(self):
        if self.container_name:
            name = self.container_name
            result = self._docker(["rm", "-f", name], timeout=15)
            if result.returncode == 0 or b"No such container" in result.stderr:
                self.container_name = None
            else:
                raise CodingRuntimeError("actor_container_cleanup_failed")

    def _actor_docker(self, arguments):
        try:
            return self._docker(arguments)
        except CodingRuntimeError:
            self.failed = True
            try:
                self._stop_actor()
            except CodingRuntimeError:
                pass
            raise

    def _start_actor(self):
        self.container_name = "fpb-" + uuid.uuid4().hex[:24]
        started = time.monotonic()
        command = ["run", "--detach", "--rm", "--pull", "never", "--name", self.container_name,
                   "--network", "none", "--read-only", "--cap-drop", "ALL",
                   "--security-opt", "no-new-privileges", "--pids-limit", "64",
                   "--memory", "512m", "--cpus", "1", "--user", f"{os.getuid()}:{os.getgid()}",
                   "--tmpfs", "/tmp:rw,nosuid,nodev,size=64m",
                   "--mount", f"type=bind,src={self.workspace},dst=/workspace",
                   "--workdir", "/workspace", "--entrypoint", "sh", self.image_id,
                   "-c", "while :; do sleep 3600; done"]
        result = self._docker(command, timeout=45)
        if result.returncode != 0:
            self.container_name = None
            raise CodingRuntimeError("actor_container_failed_to_start")
        status = self._docker(["inspect", "--format", "{{.State.Running}}", self.container_name])
        if status.returncode != 0 or status.stdout.strip() != b"true":
            self._stop_actor()
            raise CodingRuntimeError("actor_container_not_running")
        self.metrics["docker_start_seconds"] += time.monotonic() - started
        self.metrics["docker_starts"] += 1

    def artifact_binding(self):
        """Resolve all trusted grading inputs before task registration."""
        seed_sha = _workspace_digest(self.seed_dir)
        verifier_sha = _workspace_digest(self.verifier_dir)
        if not (self.verifier_dir / "verify.json").is_file():
            raise ValueError("Verifier specification is missing")
        image = self._docker(["image", "inspect", self.image, "--format", "{{.Id}}"])
        if image.returncode != 0:
            raise CodingRuntimeError("local_image_not_available")
        image_id = image.stdout.decode("ascii", "replace").strip()
        if not image_id.startswith("sha256:") or len(image_id) != 71:
            raise CodingRuntimeError("invalid_image_identity")
        visible_check_sha = _digest_bytes(json.dumps(self.visible_check, separators=(",", ":")).encode("utf-8"))
        return {"seed_workspace_sha256": seed_sha, "verifier_sha256": verifier_sha,
                "image_sha256": image_id, "visible_check_sha256": visible_check_sha}

    def _reset_from_source(self, task, source, source_sha):
        if self.workspace is not None:
            raise ValueError("One adapter instance serves one episode")
        self.replace_text_enabled = any(
            entry["name"] == "replace_text" for entry in task["tool_manifest"])
        if self.replace_text_enabled:
            binding = task.get("metadata", {}).get("replace_text_helper_binding")
            if (binding != _replace_text.helper_binding()
                    or _digest_bytes(Path(_replace_text.__file__).read_bytes())
                       != binding["source_sha256"]):
                raise ValueError("Replace-text helper differs from frozen task binding")
            self.expected_replace_text_binding = binding["source_sha256"]
        _workspace_files(self.seed_dir)
        if any(path.is_symlink() for path in self.verifier_dir.rglob("*")):
            raise ValueError("Verifier symlinks are unsupported")
        actual_binding = self.artifact_binding()
        expected_binding = task.get("metadata", {}).get("artifact_binding")
        if expected_binding is None and not task["is_fixture"]:
            raise ValueError("Non-fixture coding tasks require frozen artifact binding")
        if expected_binding is not None and expected_binding != actual_binding:
            raise ValueError("Coding artifacts differ from the frozen task binding")
        self.expected_binding = actual_binding
        image_id = actual_binding["image_sha256"]
        self.image_id = image_id
        self.task_sha256 = task["task_sha256"]
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.instance_root = self.output_root / ("coding-" + uuid.uuid4().hex)
        self.workspace = self.instance_root / "workspace"
        (self.instance_root / "snapshots").mkdir(parents=True)
        self._copy_tree(source, self.workspace)
        if _workspace_digest(self.workspace) != source_sha:
            raise CodingRuntimeError("workspace_changed_during_reset")
        self._start_actor()
        first = self._checkpoint()
        return {"task_id": task["task_id"], "tools": [item["name"] for item in task["tool_manifest"]],
                "workspace_files": [path.relative_to(self.workspace).as_posix() for path in _workspace_files(self.workspace)][:100],
                "workspace_sha256": first["workspace_sha256"], "image_sha256": image_id,
                "visible_check": list(self.visible_check)}

    def reset(self, task, *, now):
        return self._reset_from_source(task, self.seed_dir, _workspace_digest(self.seed_dir))

    def create_branch_checkpoint(self):
        """Trusted host boundary: quiesce the actor and freeze filesystem state.

        The actor is restarted for the parent after the snapshot, so processes
        and /tmp do not survive. This is not a container/process checkpoint.
        """
        self._require_reset()
        if self.submitted:
            raise ValueError("Cannot branch after submission")
        self._stop_actor()
        try:
            record = self._checkpoint()
            snapshot = self.instance_root / "snapshots" / record["workspace_sha256"]
            reference = {"snapshot_path": str(snapshot),
                         "workspace_sha256": record["workspace_sha256"],
                         "artifact_binding": dict(self.expected_binding),
                         "task_sha256": self.task_sha256,
                         "image_sha256": self.image_id}
            self.metrics["branch_checkpoints_created"] += 1
        except Exception:
            self.failed = True
            raise
        self._start_actor()
        return reference

    def reset_from_checkpoint(self, task, checkpoint_ref, *, now):
        """Trusted host operation: clone an immutable parent snapshot."""
        if not isinstance(checkpoint_ref, dict) or not isinstance(checkpoint_ref.get("snapshot_path"), str):
            raise ValueError("Invalid branch checkpoint")
        if checkpoint_ref.get("task_sha256") != task.get("task_sha256"):
            raise ValueError("Branch task differs from checkpoint")
        source = Path(checkpoint_ref["snapshot_path"]).resolve()
        expected_sha = checkpoint_ref.get("workspace_sha256")
        if not source.is_dir() or not isinstance(expected_sha, str) or _workspace_digest(source) != expected_sha:
            raise CodingRuntimeError("branch_snapshot_changed")
        if checkpoint_ref.get("artifact_binding") != self.artifact_binding():
            raise ValueError("Branch artifacts differ from checkpoint")
        if checkpoint_ref.get("image_sha256") != checkpoint_ref["artifact_binding"]["image_sha256"]:
            raise ValueError("Branch image differs from checkpoint")
        return self._reset_from_source(task, source, expected_sha)

    def step(self, action, *, now):
        self._require_reset()
        if self.submitted:
            raise ValueError("Episode already submitted")
        if not isinstance(action, dict) or action.get("action") not in {
            "list_files", "read_file", "write_file", "replace_text",
            "run_visible_checks", "submit"
        }:
            raise ValueError("Unsupported coding action")
        kind = action["action"]
        expected = {"list_files": {"action"}, "read_file": {"action", "path"},
                    "write_file": {"action", "path", "content"},
                    "replace_text": {"action", "path", "expected_file_sha256",
                                     "old_text", "new_text"},
                    "run_visible_checks": {"action"}, "submit": {"action"}}[kind]
        if set(action) != expected:
            raise ValueError("Invalid coding action arguments")
        started = time.monotonic()
        if kind == "list_files":
            files = [item.relative_to(self.workspace).as_posix() for item in _workspace_files(self.workspace)]
            observation = {"files": files[:200], "truncated": len(files) > 200}
            self.metrics["read_actions_without_checkpoint"] += 1
            self.metrics["read_seconds"] += time.monotonic() - started
        elif kind == "read_file":
            path = self._file(action["path"])
            if not path.is_file():
                raise ValueError("Workspace file not found")
            data = path.read_bytes()
            observation = {"path": action["path"], "text": data[:16000].decode("utf-8", "replace"),
                           "sha256": _digest_bytes(data), "truncated": len(data) > 16000}
            self.metrics["read_actions_without_checkpoint"] += 1
            self.metrics["read_seconds"] += time.monotonic() - started
        elif kind == "write_file":
            content = action["content"]
            if not isinstance(content, str) or len(content.encode("utf-8")) > 65536:
                raise ValueError("Write content exceeds 64 KiB")
            path = self._file(action["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.parent.resolve().is_relative_to(self.workspace):
                raise ValueError("workspace_path_escape")
            path.write_text(content, encoding="utf-8")
            _workspace_files(self.workspace)
            checkpoint = self._checkpoint()
            observation = {"path": action["path"], "sha256": _digest_bytes(content.encode("utf-8")),
                           "workspace_sha256": checkpoint["workspace_sha256"]}
            self.metrics["write_seconds"] += time.monotonic() - started
        elif kind == "replace_text":
            if not self.replace_text_enabled:
                raise ValueError("Replace text is not enabled by the frozen task")
            binding = self.expected_replace_text_binding
            if _digest_bytes(Path(_replace_text.__file__).read_bytes()) != binding:
                raise CodingRuntimeError("replace_text_helper_changed")
            # The helper checks metadata again before atomic rename; this is
            # best-effort stale-state detection, not a concurrent-writer CAS.
            try:
                observation = _replace_text.apply(self.workspace, action)
            except OSError as exc:
                raise CodingRuntimeError("replace_text_io_failure") from exc
            if "sha256" in observation:
                _workspace_files(self.workspace)
                checkpoint = self._checkpoint()
                observation["workspace_sha256"] = checkpoint["workspace_sha256"]
            self.metrics["write_seconds"] += time.monotonic() - started
        elif kind == "run_visible_checks":
            before = _workspace_digest(self.workspace)
            result = self._actor_docker(["exec", "--workdir", "/workspace", self.container_name, *self.visible_check])
            self.metrics["docker_exec_calls"] += 1
            # A check can leave background descendants or /tmp state. Destroy
            # that runtime before publishing any filesystem checkpoint.
            self._stop_actor()
            after = _workspace_digest(self.workspace)
            if before != after:
                self._checkpoint()
            self._start_actor()
            if result.returncode < 0 or result.returncode in {125, 126, 127, 137}:
                raise CodingRuntimeError("visible_check_runtime_failure")
            observation = {"passed": result.returncode == 0,
                           "return_code": result.returncode,
                           "stdout": result.stdout[:12000].decode("utf-8", "replace"),
                           "stderr": result.stderr[:12000].decode("utf-8", "replace"),
                           "workspace_sha256": after}
            self.metrics["visible_check_seconds"] += time.monotonic() - started
        else:
            # Freeze the submitted files before running a trusted verifier.
            self._stop_actor()
            checkpoint = self._checkpoint()
            self.submitted_workspace = self.instance_root / "snapshots" / checkpoint["workspace_sha256"]
            self.submitted = True
            observation = {"status": "submitted", "workspace_sha256": checkpoint["workspace_sha256"]}
        return {"observation": observation, "terminated": kind == "submit"}

    def verify(self, *, now):
        if not self.submitted:
            raise ValueError("Submit before verification")
        if self.verified is not None:
            return dict(self.verified)
        started = time.monotonic()
        if _workspace_digest(self.submitted_workspace) != self.submitted_workspace.name:
            return {"status": "pending", "reason": "submitted_workspace_changed"}
        # Expected results stay on the trusted host. Candidate code is run in
        # an isolated container with only the submitted workspace mounted.
        verifier_sha = _workspace_digest(self.verifier_dir)
        if verifier_sha != self.expected_binding["verifier_sha256"]:
            return {"status": "pending", "reason": "verifier_differs_from_frozen_task"}
        try:
            specification = strict_json_loads((self.verifier_dir / "verify.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeError):
            return {"status": "pending", "reason": "invalid_verifier_specification"}
        if (not isinstance(specification, dict) or set(specification) != {"kind", "cases"}
                or specification["kind"] != "command_cases_v1"
                or not isinstance(specification["cases"], list)
                or not 1 <= len(specification["cases"]) <= 32):
            return {"status": "pending", "reason": "invalid_verifier_specification"}
        for case in specification["cases"]:
            if (not isinstance(case, dict) or set(case) != {"argv", "expected_stdout", "expected_returncode"}
                    or not isinstance(case["argv"], list) or not 1 <= len(case["argv"]) <= 32
                    or any(not isinstance(part, str) or not part or len(part) > 2048 or "\x00" in part
                           for part in case["argv"])
                    or not isinstance(case["expected_stdout"], str)
                    or len(case["expected_stdout"].encode("utf-8")) > 12000
                    or type(case["expected_returncode"]) is not int
                    or not 0 <= case["expected_returncode"] <= 124):
                return {"status": "pending", "reason": "invalid_verifier_specification"}
        def run_case(case):
            verifier_name = "fpb-verify-" + uuid.uuid4().hex[:20]
            command = ["run", "--rm", "--pull", "never", "--name", verifier_name,
                       "--network", "none", "--read-only", "--cap-drop", "ALL",
                       "--security-opt", "no-new-privileges", "--pids-limit", "64",
                       "--memory", "512m", "--cpus", "1", "--user", f"{os.getuid()}:{os.getgid()}",
                       "--tmpfs", "/tmp:rw,nosuid,nodev,size=64m",
                       "--mount", f"type=bind,src={self.submitted_workspace},dst=/workspace,readonly",
                       "--workdir", "/workspace", "--entrypoint", case["argv"][0],
                       self.image_id, *case["argv"][1:]]
            try:
                result = self._docker(command, timeout=max(30, self.command_timeout))
            except CodingRuntimeError as exc:
                cleaned = False
                try:
                    cleanup = self._docker(["rm", "-f", verifier_name], timeout=15)
                    cleaned = cleanup.returncode == 0 or b"No such container" in cleanup.stderr
                except CodingRuntimeError:
                    pass
                if cleaned and str(exc) in {"docker_cli_timed_out", "docker_cli_output_limit_exceeded"}:
                    return {"return_code": None, "stdout_sha256": None,
                            "passed": False, "limit_exceeded": str(exc)}
                return {"status": "pending", "reason": "verifier_infrastructure_error"}
            if result.returncode == 125 or result.returncode < 0:
                return {"status": "pending", "reason": "verifier_infrastructure_error"}
            return {"return_code": result.returncode,
                    "stdout_sha256": _digest_bytes(result.stdout),
                    "passed": (result.returncode == case["expected_returncode"]
                               and result.stdout == case["expected_stdout"].encode("utf-8"))}

        with ThreadPoolExecutor(max_workers=self.verifier_workers) as pool:
            results = list(pool.map(run_case, specification["cases"]))
        self.metrics["verify_seconds"] += time.monotonic() - started
        self.metrics["verifier_cases"] += len(results)
        if any(result.get("status") == "pending" for result in results):
            return {"status": "pending", "reason": "verifier_infrastructure_error"}
        if _workspace_digest(self.verifier_dir) != verifier_sha:
            return {"status": "pending", "reason": "verifier_changed_during_execution"}
        if _workspace_digest(self.submitted_workspace) != self.submitted_workspace.name:
            return {"status": "pending", "reason": "submitted_workspace_changed"}
        reward = 1.0 if all(result["passed"] for result in results) else 0.0
        completed_at = datetime.now(timezone.utc).isoformat()
        self.verified = {"status": "resolved", "reward": reward,
                         "available_at": completed_at,
                         "evidence": {"kind": "host_checked_command_cases_v1",
                                      "workspace_sha256": self.submitted_workspace.name,
                                      "verifier_sha256": verifier_sha,
                                      "image_sha256": self.image_id,
                                      "case_results": results}}
        return dict(self.verified)

    def get_state(self):
        return {"submitted": self.submitted, "verified": self.verified is not None,
                "checkpoints": list(self.checkpoints), "metrics": dict(self.metrics)}

    def close(self):
        if self.container_name:
            self._stop_actor()
