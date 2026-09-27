"""Trusted shared-prefix, filesystem-only coding rollouts.

One policy-generated prefix is executed once. Every suffix receives its own
workspace clone, fresh sandbox, remaining action/wall budget, and a copy of the
auditable prefix. Suffixes are correlated samples from one task, not new tasks.
"""

from __future__ import annotations

import copy
import json
import re
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from .coding_env import DockerCodingAdapter
from .realworld import RealWorldEnv, RealWorldTaskRegistry, validate_task


_BRANCH_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def run_coding_branches(*, task, seed_dir, verifier_dir, image, prefix_actions,
                        branches, output_root, policy_id, registry_path=None,
                        visible_check=("python3", "-B", "-c", "import math_utils"),
                        verifier_workers=1, max_workers=1,
                        adapter_factory=DockerCodingAdapter):
    """Run one common prefix and bounded independent suffixes.

    This host API accepts scripted actions; it neither samples a model nor
    computes policy gradients. The checkpoint reference and verifier remain
    private. A branch failure never becomes a zero reward.
    """
    if type(max_workers) is not int or not 1 <= max_workers <= 8:
        raise ValueError("max_workers must be an integer in [1, 8]")
    if not isinstance(policy_id, str) or not policy_id.strip():
        raise ValueError("policy_id must be a nonblank string")
    if not isinstance(prefix_actions, (list, tuple)) or any(not isinstance(a, dict) for a in prefix_actions):
        raise ValueError("prefix_actions must be a sequence of action objects")
    if not isinstance(branches, (list, tuple)) or not branches:
        raise ValueError("branches must be a nonempty sequence")
    ids = []
    for branch in branches:
        if (not isinstance(branch, dict) or set(branch) != {"branch_id", "actions"}
                or not isinstance(branch["branch_id"], str)
                or not _BRANCH_ID.fullmatch(branch["branch_id"])
                or not isinstance(branch["actions"], (list, tuple))
                or any(not isinstance(a, dict) for a in branch["actions"])):
            raise ValueError("Invalid branch specification")
        ids.append(branch["branch_id"])
    if len(ids) != len(set(ids)):
        raise ValueError("branch_id values must be unique")
    frozen = validate_task(task)
    if not frozen["is_fixture"] and registry_path is None:
        raise ValueError("A persistent task registry is required for non-fixture tasks")
    root = Path(output_root).resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError("Output directory must be new or empty")
    seed_dir, verifier_dir = Path(seed_dir).resolve(), Path(verifier_dir).resolve()
    if registry_path is not None:
        registry_file = Path(registry_path).resolve()
        if registry_file.is_relative_to(seed_dir) or registry_file.is_relative_to(verifier_dir):
            raise ValueError("Task registry must stay outside seed and verifier directories")
    root.mkdir(parents=True, exist_ok=True)
    (root / "branches").mkdir()
    started = time.monotonic()

    def adapter(output):
        return adapter_factory(seed_dir=seed_dir, verifier_dir=verifier_dir,
                               image=image, output_root=output,
                               visible_check=visible_check,
                               verifier_workers=verifier_workers)

    parent_adapter = adapter(root / "parent" / "instances")
    try:
        binding = parent_adapter.artifact_binding()
        bound_task = copy.deepcopy(task)
        metadata = bound_task.setdefault("metadata", {})
        if "artifact_binding" in metadata and metadata["artifact_binding"] != binding:
            raise ValueError("Coding artifacts differ from task's frozen binding")
        metadata["artifact_binding"] = binding
        validate_task(bound_task)
        if registry_path is not None:
            registry = RealWorldTaskRegistry(registry_path)
            try:
                registry.register(bound_task)
            finally:
                registry.close()
        parent = RealWorldEnv(bound_task, parent_adapter)
        parent.reset(policy_id)
        if parent.status != "active":
            raise ValueError("Parent episode did not become active")
        for action in prefix_actions:
            parent.step(copy.deepcopy(action))
            if parent.status != "active":
                raise ValueError("Prefix ended before branch checkpoint")
        checkpoint = parent.create_branch_checkpoint()
        _write_json(root / "trusted_parent_audit.json", parent.get_state())

        def run_one(index, specification):
            branch_root = root / "branches" / f"{index:06d}"
            branch_root.mkdir()
            branch_adapter = adapter(branch_root / "instances")
            branch_start = time.monotonic()
            try:
                episode = parent.fork_from_checkpoint(checkpoint, branch_adapter,
                                                      branch_id=specification["branch_id"])
                opening = episode.opening_observation()
                transitions = []
                for action in specification["actions"]:
                    if episode.status != "active":
                        break
                    transitions.append(episode.step(copy.deepcopy(action)))
                verification = episode.verify() if episode.status == "pending" else None
                state = episode.get_state()
                summary = {"branch_id": specification["branch_id"],
                           "episode_id": state["episode_id"], "status": state["status"],
                           "reward": state["reward"], "actions_used": state["metrics"]["actions_used"],
                           "opening": opening, "transitions": transitions,
                           "verification": verification,
                           "environment_metrics": state["metrics"],
                           "adapter_metrics": state["adapter_state"].get("metrics", {}),
                           "elapsed_seconds": time.monotonic() - branch_start,
                           "is_fixture": bound_task["is_fixture"], "trainer_ready": False}
                trajectory = episode.export_trajectory()
                result = {"index": index, "branch_id": specification["branch_id"],
                          "status": state["status"], "reward": state["reward"],
                          "actions_used": state["metrics"]["actions_used"],
                          "elapsed_seconds": summary["elapsed_seconds"],
                          "output": str(branch_root)}
            except Exception as exc:
                summary = state = trajectory = None
                result = {"index": index, "branch_id": specification["branch_id"],
                          "status": "infrastructure_error", "reward": None,
                          "error_type": type(exc).__name__, "error": str(exc),
                          "elapsed_seconds": time.monotonic() - branch_start,
                          "output": str(branch_root)}
            try:
                branch_adapter.close()
            except Exception as exc:
                result = {"index": index, "branch_id": specification["branch_id"],
                          "status": "infrastructure_error", "reward": None,
                          "error_type": type(exc).__name__, "error": "sandbox_cleanup_failed",
                          "elapsed_seconds": time.monotonic() - branch_start,
                          "output": str(branch_root)}
                summary = state = trajectory = None
            if summary is not None:
                try:
                    _write_json(branch_root / "report.json", summary)
                    _write_json(branch_root / "trusted_audit.json", state)
                    if trajectory is not None:
                        _write_json(branch_root / "training_trajectory.json", trajectory)
                except (OSError, ValueError, TypeError) as exc:
                    (branch_root / "training_trajectory.json").unlink(missing_ok=True)
                    result = {"index": index, "branch_id": specification["branch_id"],
                              "status": "infrastructure_error", "reward": None,
                              "error_type": type(exc).__name__, "error": "branch_output_write_failed",
                              "elapsed_seconds": time.monotonic() - branch_start,
                              "output": str(branch_root)}
            return result

        results = [None] * len(branches)
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="coding-branch") as pool:
            futures = {}
            next_index = 0
            in_flight_limit = min(len(branches), 2 * max_workers)
            while next_index < len(branches) or futures:
                while next_index < len(branches) and len(futures) < in_flight_limit:
                    future = pool.submit(run_one, next_index, branches[next_index])
                    futures[future] = next_index
                    next_index += 1
                completed, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in completed:
                    index = futures.pop(future)
                    results[index] = future.result()
        report = {"schema_version": "coding-branches-0.1", "task_id": bound_task["task_id"],
                  "policy_id": policy_id, "branch_count": len(results),
                  "max_workers": max_workers, "max_in_flight": in_flight_limit,
                  "prefix_actions_used": checkpoint["actions_used"],
                  "prefix_event_sha256": checkpoint["prefix_event_sha256"],
                  "workspace_sha256": checkpoint["workspace_sha256"],
                  "parent_adapter_metrics": parent_adapter.get_state()["metrics"],
                  "results": results, "wall_seconds": time.monotonic() - started,
                  "trainer_ready": False, "filesystem_only_checkpoint": True}
        _write_json(root / "branch_report.json", report)
        return report
    finally:
        parent_adapter.close()
