"""Operator-driven coding episodes and an offline fixture for the real-world track."""

from __future__ import annotations

import copy
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .coding_env import DockerCodingAdapter
from .http import strict_json_loads
from .realworld import RealWorldEnv, RealWorldTaskRegistry, validate_task


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def run_coding_task(*, task, seed_dir, verifier_dir, image, actions, output,
                    visible_check=("python3", "-B", "-c", "import math_utils"),
                    policy_id="scripted-smoke-policy", registry_path=None,
                    verifier_workers=1):
    """Execute a trusted task with supplied actions; no model or weight trainer is used.

    The operator supplies the seed, hidden verifier, image, and visible check.
    Policy actions never choose a shell command or a host path.
    """
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be new or empty")
    frozen_task = validate_task(task)
    if not frozen_task["is_fixture"] and registry_path is None:
        raise ValueError("A persistent task registry is required for non-fixture tasks")
    adapter = DockerCodingAdapter(seed_dir=seed_dir, verifier_dir=verifier_dir,
                                  image=image, output_root=output / "instances",
                                  visible_check=visible_check,
                                  verifier_workers=verifier_workers)
    binding = adapter.artifact_binding()
    bound_task = copy.deepcopy(task)
    metadata = bound_task.setdefault("metadata", {})
    if "artifact_binding" in metadata and metadata["artifact_binding"] != binding:
        raise ValueError("Coding artifacts differ from task's frozen binding")
    metadata["artifact_binding"] = binding
    validate_task(bound_task)
    if registry_path is not None:
        registry_file = Path(registry_path).resolve()
        if (registry_file.is_relative_to(Path(seed_dir).resolve())
                or registry_file.is_relative_to(Path(verifier_dir).resolve())):
            raise ValueError("Task registry must stay outside seed and verifier directories")
        registry = RealWorldTaskRegistry(registry_path)
        try:
            registry.register(bound_task)
        finally:
            registry.close()
    output.mkdir(parents=True, exist_ok=True)
    env = RealWorldEnv(bound_task, adapter)
    started = time.monotonic()
    transitions = []
    try:
        opening = env.reset(policy_id)
        for action in actions:
            if env.status != "active":
                break
            transition = env.step(action)
            transitions.append(transition)
        result = env.verify() if env.status == "pending" else None
        state = env.get_state()
        summary = {"task_id": task["task_id"], "episode_id": state["episode_id"],
                   "status": state["status"], "reward": state["reward"],
                   "is_fixture": task["is_fixture"], "trainer_ready": False,
                   "opening": opening, "transitions": transitions, "verification": result,
                   "environment_metrics": state["metrics"],
                   "adapter_metrics": state["adapter_state"].get("metrics", {}),
                   "elapsed_seconds": time.monotonic() - started,
                   "image_sha256": adapter.image_id}
        _write_json(output / "report.json", summary)
        _write_json(output / "trusted_audit.json", state)
        trajectory = env.export_trajectory()
        if trajectory is not None:
            _write_json(output / "training_trajectory.json", trajectory)
        return {"status": summary["status"], "reward": summary["reward"],
                "is_fixture": summary["is_fixture"], "trainer_ready": False,
                "actions_used": state["metrics"]["actions_used"],
                "checkpoints_created": summary["adapter_metrics"].get("checkpoints_created"),
                "elapsed_seconds": summary["elapsed_seconds"],
                "output": str(output), "image_sha256": adapter.image_id}
    finally:
        adapter.close()


def run_coding_smoke(*, image, output):
    """Run one deliberately broken local module through real, offline containers."""
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be new or empty")
    seed = output / "fixture" / "seed"
    verifier = output / "fixture" / "verifier"
    seed.mkdir(parents=True)
    verifier.mkdir(parents=True)
    (seed / "math_utils.py").write_text("def add(left, right):\n    return left - right\n", encoding="utf-8")
    _write_json(verifier / "verify.json", {
        "kind": "command_cases_v1",
        "cases": [
            {"argv": ["python3", "-B", "-c", "from math_utils import add; print(add(2, 3))"],
             "expected_stdout": "5\n", "expected_returncode": 0},
            {"argv": ["python3", "-B", "-c", "from math_utils import add; print(add(-2, 2))"],
             "expected_stdout": "0\n", "expected_returncode": 0},
        ],
    })
    now = datetime.now(timezone.utc)
    task = {
        "schema_version": "realworld-0.1", "task_id": "fixture-code-add-v1",
        "event_id": "fixture-code-add", "cluster_id": "fixture-code-add",
        "split": "train", "prompt": "Repair add(left, right) in math_utils.py.",
        "issued_at": (now - timedelta(minutes=1)).isoformat(),
        "action_deadline": (now + timedelta(minutes=30)).isoformat(),
        "outcome_not_before": (now - timedelta(minutes=1)).isoformat(),
        "verify_after": (now - timedelta(minutes=1)).isoformat(),
        "tool_manifest": [
            {"name": name, "description": description} for name, description in (
                ("list_files", "List workspace files."), ("read_file", "Read a workspace file."),
                ("write_file", "Replace a workspace file."),
                ("run_visible_checks", "Run the fixed syntax check."),
                ("submit", "Freeze and submit the workspace."))],
        "reward_contract": {"id": "fixture-hidden-tests-v1",
                            "description": "One point only when all hidden tests pass.",
                            "min_reward": 0, "max_reward": 1},
        "budgets": {"max_actions": 8, "max_wall_seconds": 900},
        "is_fixture": True, "adapter_id": "docker_coding", "adapter_version": "0.1",
    }
    actions = [{"action": "list_files"}, {"action": "read_file", "path": "math_utils.py"},
               {"action": "write_file", "path": "math_utils.py",
                "content": "def add(left, right):\n    return left + right\n"},
               {"action": "run_visible_checks"}, {"action": "submit"}]
    # The fixture inputs live beside operational output; run_coding_task uses
    # a fresh child directory so its no-overwrite rule still applies.
    return run_coding_task(task=task, seed_dir=seed, verifier_dir=verifier,
                           image=image, actions=actions, output=output / "episode",
                           visible_check=("python3", "-B", "-c", "import math_utils"))


def load_coding_actions(path):
    actions = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        action = strict_json_loads(line)
        if not isinstance(action, dict):
            raise ValueError(f"Action on line {number} is not an object")
        actions.append(action)
    if not actions:
        raise ValueError("Action file is empty")
    return actions
