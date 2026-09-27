"""Proof-only two-child full-VM hidden-case verifier.

This is deliberately outside the production adapter. The policy-facing task,
actions, artifact binding, and evidence remain the serial full-VM contract;
only trusted host scheduling of its cases changes. Each child has its own
writable qcow2, QEMU process, RAM, serial channel, and snapshot restores.
"""

from __future__ import annotations

import copy
import hashlib
import secrets
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from future_prediction_bench.http import strict_json_loads
from future_prediction_bench.microvm_coding import MicroVMCodingAdapter
from future_prediction_bench.microvm_runtime import _sha256_file


def _case_result(runner, child, case):
    """Run exactly the serial adapter's unprivileged case capture in a child."""
    argv = runner._validate_python_argv(case["argv"])
    child.load_snapshot("submitted")
    output = runner._run_python_case(
        argv, timeout=max(30.0, runner.command_timeout), unprivileged=True)
    return {"return_code": output["return_code"],
            "stdout_sha256": hashlib.sha256(output["stdout_bytes"]).hexdigest(),
            "passed": output["return_code"] == case["expected_returncode"]
                      and not output["truncated"]
                      and output["stdout_bytes"] == case["expected_stdout"].encode("utf-8")}


def _required(runtime, command):
    result = runtime.run_shell(command, timeout=30)
    if result["return_code"] != 0:
        raise RuntimeError("full_vm_pool_isolation_probe_failed")
    return result["stdout"]


def _probe_independent_state(parent, children):
    """Check sibling/parent RAM and writable ext4 isolation before grading."""
    nonce = secrets.token_hex(8)
    for index, child in enumerate(children):
        for path in (f"/tmp/fpb-pool-{nonce}-{index}",
                     f"/mnt/root/.fpb-pool-{nonce}-{index}"):
            _required(child, f"printf '{index}' > {path}")
            if _required(child, f"cat {path}") != str(index):
                raise RuntimeError("full_vm_pool_isolation_probe_failed")
            for other in (parent, children[1 - index]):
                _required(other, f"test ! -e {path}")
    # Every case subsequently loads the submitted snapshot and loses markers.


class TwoChildFullVMVerifierPool(MicroVMCodingAdapter):
    """Opt-in host-side grading pool for a submitted serial full-VM episode."""

    def __init__(self, runtime, *, verifier_dir, visible_check, pool_dir,
                 isolation_probe=False, workspace_root=MicroVMCodingAdapter.WORKSPACE_ROOT,
                 command_timeout=30.0):
        if bool(getattr(runtime, "enable_action_port", False)):
            raise ValueError("Pool proof requires the serial, full-VM adapter")
        if type(isolation_probe) is not bool:
            raise ValueError("isolation_probe must be boolean")
        super().__init__(runtime, verifier_dir=verifier_dir, visible_check=visible_check,
                         workspace_root=workspace_root, command_timeout=command_timeout)
        root = Path(pool_dir)
        if root.is_symlink() or not root.is_dir():
            raise ValueError("Pool output must be an existing regular directory")
        self.pool_dir = root.resolve()
        if (self.pool_dir == self.verifier_dir
                or self.pool_dir.is_relative_to(self.verifier_dir)
                or self.verifier_dir.is_relative_to(self.pool_dir)
                or self.runtime.disk_path.is_relative_to(self.pool_dir)):
            raise ValueError("Pool disks must be separate from parent and verifier")
        self.isolation_probe = isolation_probe
        self.active_children = ()
        self.pool_timings = {}

    def _make_case_runner(self, child):
        # Reuse the production parser, process-group timeout, 12 kB output
        # bound, and UID/GID 65534 execution; never pass expected outputs in.
        return MicroVMCodingAdapter(
            child, verifier_dir=self.verifier_dir, visible_check=self.visible_check,
            workspace_root=self.workspace_root, command_timeout=self.command_timeout)

    def _worker(self, child, indexed_cases):
        runner = self._make_case_runner(child)
        output = []
        started = time.monotonic()
        for index, case in indexed_cases:
            output.append((index, _case_result(runner, child, case)))
        return output, time.monotonic() - started

    def verify(self, *, now):
        if not self.submitted:
            raise ValueError("Submit before verification")
        if self.verified is not None:
            return copy.deepcopy(self.verified)
        verifier_file = self.verifier_dir / "verify.json"
        if verifier_file.is_symlink() or _sha256_file(verifier_file) != self.expected_binding["verifier_sha256"]:
            return {"status": "pending", "reason": "verifier_differs_from_frozen_task"}
        try:
            self._preflight_full_vm_verifier()
            specification = strict_json_loads(verifier_file.read_text(encoding="utf-8"))
            if (not isinstance(specification, dict)
                    or set(specification) != {"kind", "cases"}
                    or specification["kind"] != "command_cases_v1"
                    or not isinstance(specification["cases"], list)
                    or not 1 <= len(specification["cases"]) <= 32):
                raise ValueError("invalid verifier")
            cases = specification["cases"]
            for case in cases:
                if (not isinstance(case, dict)
                        or set(case) != {"argv", "expected_stdout", "expected_returncode"}
                        or not isinstance(case["expected_stdout"], str)
                        or len(case["expected_stdout"].encode("utf-8")) > 12000
                        or type(case["expected_returncode"]) is not int
                        or not 0 <= case["expected_returncode"] <= 123):
                    raise ValueError("invalid case")
                self._validate_python_argv(case["argv"])
        except (OSError, ValueError, UnicodeError):
            return {"status": "pending", "reason": "invalid_verifier_specification"}

        started = time.monotonic()
        before = self.runtime.get_state()["metrics"]
        paths = [self.pool_dir / f"pool-{secrets.token_hex(12)}-{index}.qcow2"
                 for index in range(2)]
        children = []
        results = [None] * len(cases)
        failed = False
        fork_end = worker_end = cleanup_start = None
        worker_seconds = []
        isolation_seconds = 0.0
        try:
            children = self.runtime.fork_snapshot(
                "submitted", paths, resume_parent=True, parallel_children=True)
            fork_end = time.monotonic()
            if len(children) != 2 or children[0] is children[1]:
                raise RuntimeError("full_vm_pool_child_count_invalid")
            self.active_children = tuple(children)
            if self.isolation_probe:
                probe_start = time.monotonic()
                _probe_independent_state(self.runtime, children)
                isolation_seconds = time.monotonic() - probe_start
            indexed = list(enumerate(cases))
            with ThreadPoolExecutor(max_workers=2) as executor:
                pending = [executor.submit(self._worker, children[index], indexed[index::2])
                           for index in range(2)]
                for future in pending:
                    pairs, duration = future.result()
                    worker_seconds.append(duration)
                    for index, result in pairs:
                        if (type(index) is not int or not 0 <= index < len(results)
                                or results[index] is not None):
                            raise RuntimeError("full_vm_pool_index_invalid")
                        results[index] = result
            worker_end = time.monotonic()
            if any(result is None for result in results):
                raise RuntimeError("full_vm_pool_case_missing")
        except Exception:
            failed = True
        finally:
            cleanup_start = time.monotonic()
            for child in children:
                try:
                    child.close()
                except Exception:
                    failed = True
            self.active_children = ()
            for path in paths:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    failed = True
            cleanup_end = time.monotonic()
            after = self.runtime.get_state()["metrics"]
            self.pool_timings = {
                "fork_seconds": (fork_end - started) if fork_end is not None else None,
                "disk_clone_seconds": after.get("fork_disk_clone_seconds", 0.0)
                                      - before.get("fork_disk_clone_seconds", 0.0),
                "child_start_restore_seconds": after.get("fork_child_start_restore_seconds", 0.0)
                                               - before.get("fork_child_start_restore_seconds", 0.0),
                "isolation_probe_seconds": isolation_seconds,
                "parallel_case_wall_seconds": (worker_end - fork_end - isolation_seconds)
                                              if worker_end is not None else None,
                "worker_seconds": worker_seconds,
                "cleanup_seconds": cleanup_end - cleanup_start,
                "total_seconds": cleanup_end - started,
                "failed": failed,
            }
        if failed:
            return {"status": "pending", "reason": "vm_verifier_infrastructure_error"}
        self.metrics["full_vm_restores"] += len(cases)
        self.metrics["hidden_cases"] += len(cases)
        if _sha256_file(verifier_file) != self.expected_binding["verifier_sha256"]:
            return {"status": "pending", "reason": "verifier_changed_during_execution"}
        reward = 1.0 if all(result["passed"] for result in results) else 0.0
        self.verified = {"status": "resolved", "reward": reward,
                         "available_at": datetime.now(timezone.utc).isoformat(),
                         "evidence": {"kind": "host_checked_qemu_full_vm_cases_v1",
                                      "snapshot_kind": self.snapshot["kind"],
                                      "verifier_sha256": self.expected_binding["verifier_sha256"],
                                      "case_results": results}}
        return copy.deepcopy(self.verified)

    def close(self):
        for child in self.active_children:
            child.close()
        self.active_children = ()
        super().close()
