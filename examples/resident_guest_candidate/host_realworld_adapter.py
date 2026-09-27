"""Experimental RealWorldEnv-compatible host adapter for a resident VM.

Only the host opens the 14-case verifier and compares expected stdout. The
guest receives bounded case source, never expected output or reward. This
adapter intentionally declares a narrower three-tool, shared-kernel task
contract; it is not a drop-in full-VM task or a parity claim.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import secrets
import time
from datetime import datetime
from pathlib import Path

from future_prediction_bench.coding_env import _workspace_digest
from future_prediction_bench.realworld import validate_task
from host_client import ResidentHostAdapter, ResidentInterrupted, prototype_binding
from hardened_edit import validate_input
from wire import IDENTITY, MAX_CASES, ProtocolError, VERSION, relative_path, validate_request


TOOLS = ("read_file", "replace_text", "submit")
CONTRACT_KIND = "stateless_python_cases_resident_v1"
SHA_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(",", ":"))


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def _strict_file(path, *, max_bytes=100000):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("trusted_host_file_missing_or_symlink")
    data = path.read_bytes()
    if not 0 < len(data) <= max_bytes:
        raise ValueError("trusted_host_file_size_invalid")
    return data


def _strict_json(path, *, max_bytes=100000):
    data = _strict_file(path, max_bytes=max_bytes)
    value = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_pairs)
    if not isinstance(value, dict):
        raise ValueError("trusted_host_json_object_required")
    return value, _sha(data)


class HostPrivateCases:
    """Frozen host-only verifier and exact opt-in stateless contract."""

    def __init__(self, task_path, verifier_path, contract_path,
                 assets_manifest_path):
        self.paths = {"task": Path(task_path), "verifier": Path(verifier_path),
                      "contract": Path(contract_path),
                      "assets_manifest": Path(assets_manifest_path)}
        source_task, task_sha = _strict_json(self.paths["task"])
        specification, verifier_sha = _strict_json(self.paths["verifier"])
        contract, contract_sha = _strict_json(self.paths["contract"])
        assets, assets_sha = _strict_json(self.paths["assets_manifest"])
        validate_task(source_task)
        task_id = source_task.get("task_id")
        source_path = contract.get("source_path")
        try:
            relative_path(source_path)
        except ProtocolError as exc:
            raise ValueError("resident_source_path_invalid") from exc
        seed_path = self.paths["task"].parent / "seed" / source_path
        seed_sha = _sha(_strict_file(seed_path, max_bytes=65536))
        case_count = contract.get("case_count")
        if (not isinstance(task_id, str) or not task_id
                or source_task.get("is_fixture") is not True
                or contract.get("kind") != CONTRACT_KIND
                or set(contract) != {
                    "kind", "task_id", "verifier_sha256", "source_path",
                    "seed_file_sha256", "case_count",
                    "requires_live_background_process_state",
                    "requires_shared_case_filesystem_state",
                    "requires_quiescent_submitted_state",
                    "allow_unprivileged_case_execution"}
                or contract.get("task_id") != task_id
                or contract.get("verifier_sha256") != verifier_sha
                or contract.get("seed_file_sha256") != seed_sha
                or type(case_count) is not int or not 1 <= case_count <= MAX_CASES
                or contract.get("requires_live_background_process_state") is not False
                or contract.get("requires_shared_case_filesystem_state") is not False
                or contract.get("requires_quiescent_submitted_state") is not True
                or contract.get("allow_unprivileged_case_execution") is not True
                or assets.get("task_id") != task_id
                or assets.get("architecture") != "linux/arm64"
                or not isinstance(assets.get("schema_version"), str)
                or not isinstance(assets.get("seed_workspace_sha256"), str)
                or SHA_PATTERN.fullmatch(assets["seed_workspace_sha256"]) is None
                or not isinstance(assets.get("rootfs_qcow2_sha256"), str)
                or SHA_PATTERN.fullmatch(assets["rootfs_qcow2_sha256"]) is None
                or _workspace_digest(self.paths["task"].parent / "seed") !=
                   assets["seed_workspace_sha256"]
                or specification.get("kind") != "command_cases_v1"
                or set(specification) != {"kind", "cases"}
                or not isinstance(specification["cases"], list)
                or len(specification["cases"]) != case_count):
            raise ValueError("resident_verifier_contract_not_pinned")
        cases = []
        for case in specification["cases"]:
            if (not isinstance(case, dict)
                    or set(case) != {"argv", "expected_stdout", "expected_returncode"}
                    or not isinstance(case["argv"], list)
                    or len(case["argv"]) != 4
                    or case["argv"][:3] != ["python3", "-B", "-c"]
                    or not isinstance(case["argv"][3], str)
                    or not 1 <= len(case["argv"][3].encode("utf-8")) <= 512
                    or "\x00" in case["argv"][3]
                    or not isinstance(case["expected_stdout"], str)
                    or len(case["expected_stdout"].encode("utf-8")) > 12000
                    or type(case["expected_returncode"]) is not int
                    or not 0 <= case["expected_returncode"] <= 123):
                raise ValueError("resident_case_outside_bound")
            cases.append(copy.deepcopy(case))
        self.source_task = source_task
        self.task_id = task_id
        self.source_path = source_path
        self.seed_file_path = seed_path
        self.seed_file_sha256 = seed_sha
        self.cases = cases
        self.file_digests = {"task": task_sha, "verifier": verifier_sha,
                             "contract": contract_sha, "assets_manifest": assets_sha,
                             "seed_file": seed_sha}
        self.helper_paths = {name: Path(__file__).with_name(name) for name in
                             ("wire.py", "guest_supervisor.py", "case_runner.py",
                              "hardened_edit.py", "host_client.py",
                              "host_realworld_adapter.py")}
        self.helper_digests = {name: _sha(_strict_file(path, max_bytes=65536))
                               for name, path in self.helper_paths.items()}
        self.binding = {
            "runtime_kind": IDENTITY,
            "security_boundary": "one_vm_shared_guest_kernel_trusted_task_v1",
            "case_isolation_kind": "nested_mount_pid_namespace_tmpfs_overlay_v1",
            "prototype_sha256": prototype_binding()["prototype_sha256"],
            "source_task_sha256": task_sha,
            "verifier_sha256": verifier_sha,
            "stateless_contract_sha256": contract_sha,
            "assets_manifest_sha256": assets_sha,
            "source_path": source_path,
            "case_count": case_count,
            "seed_file_sha256": seed_sha,
            "guest_wire_sha256": self.helper_digests["wire.py"],
            "guest_supervisor_sha256": self.helper_digests["guest_supervisor.py"],
            "guest_case_runner_sha256": self.helper_digests["case_runner.py"],
            "guest_hardened_edit_sha256": self.helper_digests["hardened_edit.py"],
        }

    def check_unchanged(self):
        for name, path in self.paths.items():
            if _sha(_strict_file(path)) != self.file_digests[name]:
                raise RuntimeError("trusted_host_file_changed: " + name)
        for name, path in self.helper_paths.items():
            if _sha(_strict_file(path, max_bytes=65536)) != self.helper_digests[name]:
                raise RuntimeError("resident_helper_changed: " + name)
        if _sha(_strict_file(self.seed_file_path, max_bytes=65536)) != self.seed_file_sha256:
            raise RuntimeError("trusted_host_file_changed: seed_file")

    def experimental_task(self):
        task = copy.deepcopy(self.source_task)
        task["adapter_id"] = "resident_microvm_coding_experimental"
        task["adapter_version"] = "0.1"
        task["tool_manifest"] = [tool for tool in task["tool_manifest"]
                                 if tool["name"] in TOOLS]
        if {tool["name"] for tool in task["tool_manifest"]} != set(TOOLS):
            raise ValueError("pinned_source_task_lacks_required_tools")
        task.setdefault("metadata", {}).pop("replace_text_helper_binding", None)
        task["metadata"]["artifact_binding"] = copy.deepcopy(self.binding)
        task["metadata"]["resident_scope"] = (
            "trusted_shared_kernel_stateless_python_cases_three_tools_experimental")
        return task


class ResidentRealWorldAdapter:
    """Implements RealWorldEnv's four adapter methods without importing it."""

    def __init__(self, client: ResidentHostAdapter, private_cases: HostPrivateCases,
                 *, infrastructure_error_class, episode_mode="repair",
                 case_transport="sequential"):
        if not isinstance(client, ResidentHostAdapter) or client.state != "idle":
            raise ValueError("connected_idle_resident_client_required")
        if not isinstance(private_cases, HostPrivateCases):
            raise ValueError("host_private_cases_required")
        if (not isinstance(infrastructure_error_class, type)
                or not issubclass(infrastructure_error_class, Exception)):
            raise ValueError("realworld_infrastructure_error_class_required")
        if episode_mode not in {"repair", "baseline"}:
            raise ValueError("resident_episode_mode_invalid")
        if case_transport not in {"sequential", "batch"}:
            raise ValueError("resident_case_transport_invalid")
        guest_names = ("wire.py", "guest_supervisor.py", "case_runner.py",
                       "hardened_edit.py")
        if (client.guest_helper_sha256 != {
                name: private_cases.helper_digests[name] for name in guest_names}
                or client.binding["prototype_sha256"] !=
                   private_cases.binding["prototype_sha256"]
                or client.source_path != private_cases.source_path
                or client.seed_file_sha256 != private_cases.seed_file_sha256
                or client.case_count != len(private_cases.cases)):
            raise ValueError("resident_guest_helper_binding_mismatch")
        self.client = client
        self.private_cases = private_cases
        self.infrastructure_error_class = infrastructure_error_class
        self.episode_mode = episode_mode
        self.case_transport = case_transport
        self.guest_episode_id = None
        self.branch_process_nonce = None
        self.expected_task = private_cases.experimental_task()
        self.started = False
        self.submitted = False
        self.interrupted = False
        self.verified = None
        self._private_case_results = None
        self.action_count = 0
        self.submitted_source_sha = None
        self.request_start_index = len(client.request_timings)
        self.metrics = {"host_reset_seconds": None, "guest_reset_ns": None,
                        "namespace_setup_ns": None, "host_case_seconds": [],
                        "guest_case_ns": [], "host_close_seconds": None,
                        "guest_cleanup_ns": None, "effective_case_transport": None,
                        "batch_fallback_reason": None}

    def _policy_request(self, name, arguments):
        try:
            validate_request({"v": VERSION, "seq": self.client.seq, "op": "action",
                              "args": {"name": name, "input": arguments}})
        except (ProtocolError, UnicodeError) as exc:
            raise ValueError("resident_action_exceeds_wire_contract") from exc

    def _guest_action(self, name, arguments):
        self._policy_request(name, arguments)
        value = self.client.action(name, arguments)
        if not isinstance(value, dict):
            self.client.state = "interrupted"
            raise ResidentInterrupted("malformed_guest_action_result")
        if value.get("accepted") is False:
            valid = ((set(value) == {"accepted", "reason"}
                      and isinstance(value["reason"], str)
                      and 0 < len(value["reason"]) <= 80)
                     or (name == "replace_text"
                         and value == {"accepted": False, "reason": "adapter_error",
                                       "error_type": "ValueError"}))
        elif name == "read_file":
            valid = (set(value) == {"accepted", "sha256", "bytes", "text", "truncated"}
                     and value["accepted"] is True
                     and isinstance(value["sha256"], str)
                     and SHA_PATTERN.fullmatch(value["sha256"]) is not None
                     and type(value["bytes"]) is int and 0 <= value["bytes"] <= 65536
                     and isinstance(value["text"], str)
                     and len(value["text"].encode("utf-8")) <= 8192
                     and type(value["truncated"]) is bool)
        else:
            valid = (set(value) == {"accepted", "sha256", "bytes"}
                     and value["accepted"] is True
                     and isinstance(value["sha256"], str)
                     and SHA_PATTERN.fullmatch(value["sha256"]) is not None
                     and type(value["bytes"]) is int and 0 <= value["bytes"] <= 65536)
        if not valid:
            self.client.state = "interrupted"
            raise ResidentInterrupted("malformed_guest_action_result")
        return value

    def reset(self, task, *, now):
        frozen = {key: value for key, value in task.items()
                  if key not in {"task_sha256", "reward_contract_sha256"}}
        if self.started or _canonical(frozen) != _canonical(self.expected_task):
            raise ValueError("resident_frozen_task_binding_mismatch")
        self.private_cases.check_unchanged()
        if self.client.guest_helper_sha256 != {
                name: self.private_cases.helper_digests[name] for name in
                ("wire.py", "guest_supervisor.py", "case_runner.py",
                 "hardened_edit.py")}:
            raise ValueError("resident_guest_helper_binding_changed")
        self.guest_episode_id = ("resident-" + task["task_sha256"][:12]
                                 + "-" + secrets.token_hex(8))
        started = time.perf_counter()
        created = self.client.reset(self.guest_episode_id, self.episode_mode)
        self.metrics["host_reset_seconds"] = time.perf_counter() - started
        self.metrics["guest_reset_ns"] = created["guest_reset_ns"]
        self.metrics["namespace_setup_ns"] = created["namespace_setup_ns"]
        self.branch_process_nonce = created["namespaces"]["process_nonce"]
        self.started = True
        return {"task_id": self.private_cases.task_id, "tools": list(TOOLS), "runtime_kind": IDENTITY,
                "workspace_root": "/workspace",
                "case_isolation_kind": self.private_cases.binding["case_isolation_kind"]}

    def step(self, action, *, now):
        if not self.started or self.submitted or self.interrupted:
            raise ValueError("resident_episode_not_active")
        if not isinstance(action, dict) or action.get("action") not in TOOLS:
            raise ValueError("unsupported_resident_action")
        self.action_count += 1
        if self.action_count > self.expected_task["budgets"]["max_actions"]:
            raise ValueError("resident_action_budget_exceeded")
        kind = action["action"]
        try:
            if kind == "read_file":
                if set(action) != {"action", "path"}:
                    raise ValueError("invalid_read_action")
                try:
                    relative_path(action["path"])
                except ProtocolError as exc:
                    raise ValueError("invalid_read_path") from exc
                value = self._guest_action("read_file", {"path": action["path"]})
                if value.get("accepted") is False:
                    observation = {"status": "error", "reason": value["reason"]}
                else:
                    observation = {"path": action["path"], "text": value["text"],
                                   "sha256": value["sha256"], "truncated": value["truncated"]}
            elif kind == "replace_text":
                if set(action) != {"action", "path", "expected_file_sha256",
                                   "old_text", "new_text"}:
                    raise ValueError("invalid_replace_action")
                try:
                    relative_path(action["path"])
                except ProtocolError as exc:
                    raise ValueError("invalid_replace_path") from exc
                if (not isinstance(action["expected_file_sha256"], str)
                        or not SHA_PATTERN.fullmatch(action["expected_file_sha256"])
                        or any(not isinstance(action[key], str) for key in ("old_text", "new_text"))
                        or not action["old_text"]
                        or len(action["old_text"].encode("utf-8")) > 256
                        or len(action["new_text"].encode("utf-8")) > 256):
                    raise ValueError("invalid_replace_payload")
                payload = {
                    "path": action["path"], "expected_sha256": action["expected_file_sha256"],
                    "old": action["old_text"], "new": action["new_text"]}
                validate_input(payload)
                value = self._guest_action("replace_text", payload)
                if value.get("error_type") == "ValueError":
                    observation = {"status": "error", "reason": "adapter_error",
                                   "error_type": "ValueError"}
                elif value.get("accepted") is False:
                    observation = {"status": "conflict", "reason": value["reason"],
                                   "path": action["path"]}
                else:
                    observation = {"path": action["path"], "sha256": value["sha256"]}
            else:
                if set(action) != {"action"}:
                    raise ValueError("invalid_submit_action")
                try:
                    self.private_cases.check_unchanged()
                except (OSError, RuntimeError, ValueError) as exc:
                    raise ResidentInterrupted("trusted_submit_asset_changed") from exc
                source = self._guest_action("read_file", {"path": self.private_cases.source_path})
                if source.get("accepted") is not True or not SHA_PATTERN.fullmatch(source["sha256"]):
                    raise ResidentInterrupted("submitted_source_unreadable")
                self.submitted_source_sha = source["sha256"]
                self.client.submit()
                self.submitted = True
                observation = {"status": "submitted",
                               "snapshot_kind": "resident_mount_pid_overlay_branch_v1"}
            return {"observation": observation, "terminated": kind == "submit"}
        except ResidentInterrupted as exc:
            self.interrupted = True
            raise self.infrastructure_error_class("resident_guest_interrupted") from exc

    def verify(self, *, now):
        if not self.submitted:
            raise ValueError("resident_submit_required")
        if self.verified is not None:
            return copy.deepcopy(self.verified)
        if self.interrupted:
            return {"status": "pending", "reason": "resident_runtime_interrupted"}
        try:
            self.private_cases.check_unchanged()
            results = []
            codes = [case["argv"][3] for case in self.private_cases.cases]
            transport = self.case_transport
            if transport == "batch":
                if any(len(case["expected_stdout"].encode("utf-8")) > 64
                       for case in self.private_cases.cases):
                    transport = "sequential"
                    self.metrics["batch_fallback_reason"] = "expected_stdout_over_batch_cap"
                else:
                    try:
                        validate_request({"v": VERSION, "seq": self.client.seq,
                                          "op": "case_batch", "args": {"codes": codes}})
                    except ProtocolError:
                        transport = "sequential"
                        self.metrics["batch_fallback_reason"] = "batch_request_exceeds_bound"
            self.metrics["effective_case_transport"] = transport
            if transport == "batch":
                case_started = time.perf_counter()
                outputs = self.client.run_case_batch(
                    codes,
                    expected_branch_sha256=self.submitted_source_sha)
                self.metrics["host_case_seconds"].append(
                    time.perf_counter() - case_started)
            else:
                outputs = []
                for case in self.private_cases.cases:
                    case_started = time.perf_counter()
                    outputs.append(self.client.run_case(
                        case["argv"][3],
                        expected_branch_sha256=self.submitted_source_sha))
                    self.metrics["host_case_seconds"].append(
                        time.perf_counter() - case_started)
            for case, output in zip(self.private_cases.cases, outputs, strict=True):
                self.metrics["guest_case_ns"].append(output["guest_case_ns"])
                passed = (output["return_code"] == case["expected_returncode"]
                          and not output["truncated"]
                          and not output["output_over_batch_cap"]
                          and output["stdout_bytes"] == case["expected_stdout"].encode("utf-8"))
                results.append({"return_code": output["return_code"],
                                "stdout_sha256": _sha(output["stdout_bytes"]),
                                "passed": passed})
            self.private_cases.check_unchanged()
            close_started = time.perf_counter()
            closed = self.client.close(completed=True)  # Cleanup must succeed before publishing a score.
            self.metrics["host_close_seconds"] = time.perf_counter() - close_started
            self.metrics["guest_cleanup_ns"] = closed["guest_cleanup_ns"]
        except BaseException:
            self.interrupted = True
            try:
                if self.client.state in {"active", "pending"}:
                    self.client.close()
            except BaseException:
                pass
            return {"status": "pending", "reason": "resident_verifier_infrastructure_error"}
        reward = 1.0 if all(item["passed"] for item in results) else 0.0
        self._private_case_results = copy.deepcopy(results)
        available = now.isoformat() if isinstance(now, datetime) else str(now)
        self.verified = {"status": "resolved", "reward": reward,
                         "available_at": available,
                         "evidence": {"verifier_sha256": self.private_cases.binding["verifier_sha256"],
                                      "case_count": len(results),
                                      "passed_count": sum(item["passed"] for item in results)}}
        return copy.deepcopy(self.verified)

    def private_case_audit(self):
        """Trusted host only; never forward this per-case detail to policy."""
        if self.verified is None or self._private_case_results is None:
            raise ValueError("private_case_audit_unavailable")
        return copy.deepcopy(self._private_case_results)

    def get_state(self):
        return {"started": self.started, "submitted": self.submitted,
                "interrupted": self.interrupted, "verified": self.verified is not None,
                "action_count": self.action_count, "runtime_kind": IDENTITY,
                "guest_episode_id": self.guest_episode_id,
                "episode_mode": self.episode_mode,
                "branch_process_nonce": self.branch_process_nonce,
                "metrics": copy.deepcopy(self.metrics),
                "host_requests": copy.deepcopy(
                    self.client.request_timings[self.request_start_index:]),
                "artifact_binding": copy.deepcopy(self.private_cases.binding)}

    def close(self):
        if self.client.state in {"active", "pending"}:
            self.client.close()
