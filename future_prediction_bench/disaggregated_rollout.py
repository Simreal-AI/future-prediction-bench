"""Same-host actor/verifier pipeline for real-world coding episodes.

The actor stage runs ``reset`` and policy-supplied actions through the real
Docker coding adapter. Once ``submit`` stops the actor container and freezes
the workspace, it hands the episode to a separate verifier queue. A bounded
queue applies backpressure rather than accumulating live episode state without
limit. This is a local environment pipeline, not RollArt's distributed
prefill/decode/trainer architecture or a model trainer.
"""

from __future__ import annotations

import copy
import json
import time
from datetime import datetime, timezone
from heapq import heappop, heappush
from pathlib import Path
from queue import PriorityQueue, Queue
from threading import Event, Lock, Thread

from .coding_env import DockerCodingAdapter
from .realworld import RealWorldEnv, RealWorldTaskRegistry, validate_task


_STOP = object()
_REQUIRED = {"task", "seed_dir", "verifier_dir", "image", "actions", "policy_id"}
_OPTIONAL = {"visible_check", "verifier_workers", "verifier_priority"}
_DEFAULT_VISIBLE_CHECK = ("python3", "-B", "-c", "import math_utils")
_DEFAULT_VERIFIER_PRIORITY = 5


class _StablePriorityQueue(PriorityQueue):
    """Break priority ties at actual enqueue time under Queue's mutex."""

    def __init__(self, maxsize=0):
        super().__init__(maxsize=maxsize)
        self._next_sequence = 0

    def _put(self, item):
        priority, context = item
        heappush(self.queue, (priority, self._next_sequence, context))
        self._next_sequence += 1

    def _get(self):
        _, _, context = heappop(self.queue)
        return context


def _timestamp():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path, value):
    encoded = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def _job(job):
    if not isinstance(job, dict) or not _REQUIRED <= job.keys() or job.keys() - _REQUIRED - _OPTIONAL:
        raise ValueError("Invalid disaggregated coding job fields")
    if not isinstance(job["policy_id"], str) or not job["policy_id"].strip():
        raise ValueError("Each coding job requires an explicit policy_id")
    if not isinstance(job["actions"], (list, tuple)) or any(not isinstance(action, dict) for action in job["actions"]):
        raise ValueError("actions must be a sequence of objects")
    priority = job.get("verifier_priority", _DEFAULT_VERIFIER_PRIORITY)
    if type(priority) is not int or not 0 <= priority <= 9:
        raise ValueError("verifier_priority must be an integer in [0, 9]")
    frozen = validate_task(job["task"])
    return copy.deepcopy(job), frozen


def _artifact_bound_task(task, binding):
    bound = copy.deepcopy(task)
    metadata = bound.setdefault("metadata", {})
    if "artifact_binding" in metadata and metadata["artifact_binding"] != binding:
        raise ValueError("Coding artifacts differ from task's frozen binding")
    metadata["artifact_binding"] = binding
    validate_task(bound)
    return bound


def run_disaggregated_coding_batch(*, jobs, output_root, actor_workers=2,
                                   verification_workers=2, actor_queue_capacity=2,
                                   verifier_queue_capacity=2, registry_path=None,
                                   adapter_factory=DockerCodingAdapter):
    """Collect coding episodes with independent, bounded actor/verifier pools.

    Each job has ``task``, ``seed_dir``, ``verifier_dir``, ``image``, ``actions``
    and explicit ``policy_id``. Optional ``visible_check`` and
    ``verifier_workers`` configure that job's adapter; the latter controls
    parallel hidden cases *inside* one verification, distinct from the
    pipeline-wide ``verification_workers``. Trusted operator-supplied
    ``verifier_priority`` (0 highest, 9 lowest; default 5) orders waiting
    verifications. Equal priorities retain handoff order. It cannot preempt
    a running verifier and is not passed to the policy or adapter. Results
    preserve input order.

    The caller supplies actions; no inference, token logprobs, gradient update,
    weight synchronization, remote RPC, or multi-host execution is provided.
    Phase and queue timings are measured with a monotonic clock. A pending
    outcome remains ungraded and is never converted into a failed reward.
    """
    for name, value, maximum in (("actor_workers", actor_workers, 32),
                                 ("verification_workers", verification_workers, 32),
                                 ("actor_queue_capacity", actor_queue_capacity, 1024),
                                 ("verifier_queue_capacity", verifier_queue_capacity, 1024)):
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError(f"{name} must be an integer in [1, {maximum}]")
    if not callable(adapter_factory):
        raise ValueError("adapter_factory must be callable")
    if isinstance(jobs, (str, bytes, dict)):
        raise ValueError("jobs must be a nonempty sequence")
    try:
        planned = list(jobs)
    except TypeError as exc:
        raise ValueError("jobs must be a nonempty sequence") from exc
    if not planned:
        raise ValueError("jobs must be a nonempty sequence")
    root = Path(output_root).resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError("Output directory must be new or empty")
    root.mkdir(parents=True, exist_ok=True)
    (root / "episodes").mkdir()
    started_at = _timestamp()
    batch_start = time.monotonic()
    results = [None] * len(planned)
    actor_queue = Queue(maxsize=actor_queue_capacity)
    verifier_queue = _StablePriorityQueue(maxsize=verifier_queue_capacity)
    metrics_lock = Lock()
    queue_metrics = {"actor_queue_high_water": 0, "verifier_queue_high_water": 0,
                     "producer_backpressure_seconds": 0.0,
                     "actor_handoff_backpressure_seconds": 0.0,
                     "actor_handoffs": 0}

    def update_queue_metric(name, value):
        with metrics_lock:
            queue_metrics[name] = max(queue_metrics[name], value)

    def add_queue_metric(name, value):
        with metrics_lock:
            queue_metrics[name] += value

    def new_result(index, job, enqueued_at):
        policy = job.get("policy_id") if isinstance(job, dict) else None
        return {"index": index, "task_id": None, "task_sha256": None,
                "policy_id": policy if isinstance(policy, str) else None,
                "output": str(root / "episodes" / f"{index:06d}"),
                "execution_status": "error", "episode_status": None,
                "reward": None, "error_type": None, "error": None,
                "verifier_priority": None,
                "timings": {"actor_queue_seconds": None, "actor_seconds": None,
                            "actor_handoff_backpressure_seconds": None,
                            "verifier_queue_seconds": None,
                            "verification_seconds": None,
                            "total_seconds": None,
                            "completion_offset_seconds": None},
                "_enqueued_at": enqueued_at}

    def finish_failure(item, error, adapter=None):
        if adapter is not None:
            try:
                adapter.close()
            except Exception as cleanup_error:
                item["cleanup_error"] = f"{type(cleanup_error).__name__}: {cleanup_error}"
        item["execution_status"] = "error"
        item["episode_status"] = None
        item["reward"] = None
        item["error_type"] = type(error).__name__
        item["error"] = str(error)
        item["timings"]["total_seconds"] = max(0.0, time.monotonic() - item.pop("_enqueued_at"))
        item["timings"]["completion_offset_seconds"] = max(0.0, time.monotonic() - batch_start)
        results[item["index"]] = item

    def finish_episode(context, verification):
        item = context["item"]
        adapter = context["adapter"]
        env = context["env"]
        output = context["output"]
        try:
            state = env.get_state()
            summary = {"task_id": item["task_id"], "episode_id": state["episode_id"],
                       "status": state["status"], "reward": state["reward"],
                       "is_fixture": state["task"]["is_fixture"],
                       "trainer_ready": False, "opening": context["opening"],
                       "transitions": context["transitions"], "verification": verification,
                       "environment_metrics": state["metrics"],
                       "adapter_metrics": state["adapter_state"].get("metrics", {}),
                       "elapsed_seconds": time.monotonic() - context["actor_started"],
                       "image_sha256": adapter.image_id}
            trajectory = env.export_trajectory()
            adapter.close()
            _write_json(output / "report.json", summary)
            _write_json(output / "trusted_audit.json", state)
            if trajectory is not None:
                _write_json(output / "training_trajectory.json", trajectory)
            item["episode_status"] = state["status"]
            item["reward"] = state["reward"]
            item["execution_status"] = "completed"
            item["episode_id"] = state["episode_id"]
            item["actions_used"] = state["metrics"]["actions_used"]
        except Exception as exc:
            finish_failure(item, exc, adapter)
            return
        item["timings"]["total_seconds"] = max(0.0, time.monotonic() - item.pop("_enqueued_at"))
        item["timings"]["completion_offset_seconds"] = max(0.0, time.monotonic() - batch_start)
        results[item["index"]] = item

    def actor_worker():
        while True:
            payload = actor_queue.get()
            try:
                if payload is _STOP:
                    return
                index, raw_job, enqueued_at = payload
                item = new_result(index, raw_job, enqueued_at)
                actor_started = time.monotonic()
                item["timings"]["actor_queue_seconds"] = max(0.0, actor_started - enqueued_at)
                adapter = None
                try:
                    job, frozen = _job(raw_job)
                    item["verifier_priority"] = job.get("verifier_priority", _DEFAULT_VERIFIER_PRIORITY)
                    item["task_id"] = frozen["task_id"]
                    item["task_sha256"] = frozen["task_sha256"]
                    if not frozen["is_fixture"] and registry_path is None:
                        raise ValueError("A persistent task registry is required for non-fixture tasks")
                    output = Path(item["output"])
                    adapter = adapter_factory(seed_dir=job["seed_dir"],
                                              verifier_dir=job["verifier_dir"],
                                              image=job["image"],
                                              output_root=output / "instances",
                                              visible_check=job.get("visible_check", _DEFAULT_VISIBLE_CHECK),
                                              verifier_workers=job.get("verifier_workers", 1))
                    bound = _artifact_bound_task(job["task"], adapter.artifact_binding())
                    item["task_sha256"] = validate_task(bound)["task_sha256"]
                    if registry_path is not None:
                        registry_file = Path(registry_path).resolve()
                        if (registry_file.is_relative_to(Path(job["seed_dir"]).resolve())
                                or registry_file.is_relative_to(Path(job["verifier_dir"]).resolve())):
                            raise ValueError("Task registry must stay outside seed and verifier directories")
                        registry = RealWorldTaskRegistry(registry_path)
                        try:
                            registry.register(bound)
                        finally:
                            registry.close()
                    output.mkdir(parents=True, exist_ok=True)
                    env = RealWorldEnv(bound, adapter)
                    opening = env.reset(job["policy_id"])
                    transitions = []
                    for action in job["actions"]:
                        if env.status != "active":
                            break
                        transitions.append(env.step(action))
                    item["timings"]["actor_seconds"] = max(0.0, time.monotonic() - actor_started)
                    context = {"item": item, "adapter": adapter, "env": env,
                               "opening": opening, "transitions": transitions,
                               "output": output, "actor_started": actor_started,
                               "handoff_ready": Event()}
                    adapter = None  # ownership moves to finalizer/verifier
                    if env.status == "pending":
                        before_put = time.monotonic()
                        verifier_queue.put((item["verifier_priority"], context))
                        enqueued_verify = time.monotonic()
                        item["timings"]["actor_handoff_backpressure_seconds"] = max(0.0, enqueued_verify - before_put)
                        context["verifier_enqueued_at"] = enqueued_verify
                        add_queue_metric("actor_handoff_backpressure_seconds", item["timings"]["actor_handoff_backpressure_seconds"])
                        add_queue_metric("actor_handoffs", 1)
                        update_queue_metric("verifier_queue_high_water", verifier_queue.qsize())
                        context["handoff_ready"].set()
                    else:
                        item["timings"]["actor_handoff_backpressure_seconds"] = 0.0
                        finish_episode(context, None)
                except Exception as exc:
                    finish_failure(item, exc, adapter)
            finally:
                actor_queue.task_done()

    def verifier_worker():
        while True:
            context = verifier_queue.get()
            try:
                if context is _STOP:
                    return
                context["handoff_ready"].wait()
                item = context["item"]
                verify_started = time.monotonic()
                item["timings"]["verifier_queue_seconds"] = max(0.0, verify_started - context["verifier_enqueued_at"])
                try:
                    verification = context["env"].verify()
                    item["timings"]["verification_seconds"] = max(0.0, time.monotonic() - verify_started)
                    finish_episode(context, verification)
                except Exception as exc:
                    item["timings"]["verification_seconds"] = max(0.0, time.monotonic() - verify_started)
                    finish_failure(item, exc, context["adapter"])
            finally:
                verifier_queue.task_done()

    actor_threads = [Thread(target=actor_worker, name=f"coding-actor-{index}", daemon=True)
                     for index in range(actor_workers)]
    verifier_threads = [Thread(target=verifier_worker, name=f"coding-verifier-{index}", daemon=True)
                        for index in range(verification_workers)]
    for worker in (*verifier_threads, *actor_threads):
        worker.start()
    for index, job in enumerate(planned):
        enqueued_at = time.monotonic()
        actor_queue.put((index, job, enqueued_at))
        add_queue_metric("producer_backpressure_seconds", max(0.0, time.monotonic() - enqueued_at))
        update_queue_metric("actor_queue_high_water", actor_queue.qsize())
    for _ in actor_threads:
        actor_queue.put(_STOP)
    actor_queue.join()
    for worker in actor_threads:
        worker.join()
    for _ in verifier_threads:
        verifier_queue.put((10, _STOP))
    verifier_queue.join()
    for worker in verifier_threads:
        worker.join()
    if any(item is None for item in results):
        raise RuntimeError("A pipeline worker exited without recording its episode")
    wall = max(0.0, time.monotonic() - batch_start)
    graded = sum(item["episode_status"] == "graded" for item in results)
    report = {"schema_version": "realworld-disaggregated-0.1",
              "architecture": "same_host_actor_verifier_pipeline",
              "trainer_ready": False,
              "created_at": started_at, "completed_at": _timestamp(),
              "actor_workers": actor_workers, "verification_workers": verification_workers,
              "verifier_scheduling": "trusted_priority_then_handoff_order",
              "actor_queue_capacity": actor_queue_capacity,
              "verifier_queue_capacity": verifier_queue_capacity,
              "registry_path": str(Path(registry_path).resolve()) if registry_path is not None else None,
              "job_count": len(results),
              "completed_count": sum(item["execution_status"] == "completed" for item in results),
              "error_count": sum(item["execution_status"] == "error" for item in results),
              "graded_count": graded, "wall_seconds": wall,
              "graded_episodes_per_second": graded / wall if wall else None,
              "queue_metrics": queue_metrics, "jobs": results}
    _write_json(root / "batch_report.json", report)
    return report
