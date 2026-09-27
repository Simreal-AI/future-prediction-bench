"""Real Docker end-to-end benchmark and verifier-process crash recovery.

Both arms use the pinned public Boltons v2 repair and untouched baseline,
the same 14 private host-side cases, one actor and one verifier. The existing
in-memory actor/verifier pipeline is the control; the durable arm uses a
separate verifier process and SQLite leases. Neither arm runs model inference
or an optimizer. One fault probe kills a verifier worker after all hidden
cases finish but before its reward transaction, then retries the same frozen
submission in a replacement process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import platform
import statistics
import time
from pathlib import Path

from future_prediction_bench.coding_env import DockerCodingAdapter, _workspace_digest
from future_prediction_bench.disaggregated_rollout import run_disaggregated_coding_batch
from future_prediction_bench.durable_rollout import DurableDockerRolloutQueue, LeaseLost, VerificationLease
from future_prediction_bench.realworld import RealWorldEnv
from future_prediction_bench import coding_env, durable_rollout, disaggregated_rollout, realworld
from .benchmark_revision_fence import _inputs, VISIBLE


REVISION = "pinned-scripted-weight-revision-v1"
POLICY = "scripted-public-boltons-v2"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _observations_sha(opening, transitions):
    visible = {"opening": opening["observation"],
               "transitions": [{"observation": item["observation"],
                                "reward": item["reward"],
                                "terminated": item["terminated"],
                                "status": item["info"]["status"]}
                               for item in transitions]}
    return hashlib.sha256(json.dumps(visible, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _submission(name, task, actions, task_dir, image, batch_root, queue):
    episode = batch_root / "episodes" / name
    adapter = DockerCodingAdapter(seed_dir=task_dir / "seed",
                                  verifier_dir=task_dir / "verifier", image=image,
                                  output_root=episode / "instances",
                                  visible_check=VISIBLE, verifier_workers=1)
    try:
        env = RealWorldEnv(task, adapter)
        opening = env.reset(POLICY)
        transitions = [env.step(action) for action in actions]
        if env.status != "pending":
            raise RuntimeError("Scripted actor failed to submit")
        receipt = queue.enqueue_submitted(name, env, policy_revision=REVISION)
        state = env.get_state()
        episode.mkdir(parents=True, exist_ok=True)
        _write(episode / "actor_audit.json", {"opening": opening,
                                               "transitions": transitions,
                                               "receipt": receipt, "state": state})
        return {"job_id": name, "receipt": receipt,
                "state": state, "source_sha256": _workspace_digest(adapter.submitted_workspace),
                "action_observations_sha256": _observations_sha(opening, transitions)}
    finally:
        adapter.close()


def _worker_until_done(queue_path, names, output):
    queue = DurableDockerRolloutQueue(queue_path)
    started = time.monotonic()
    processed = []
    try:
        while time.monotonic() - started < 180:
            if all((row := queue.get(name)) is not None and row["state"] == "graded"
                   for name in names):
                _write(output, {"status": "completed", "processed": processed})
                return
            result = queue.process_one("durable-verifier-process",
                                       revision_provider=lambda: REVISION, lease_seconds=10,
                                       retry_seconds=0.2)
            if result is None:
                time.sleep(0.02)
            else:
                processed.append(result)
        _write(output, {"status": "timed_out", "processed": processed})
        raise RuntimeError("Durable verifier did not grade the batch")
    except Exception as exc:
        _write(output, {"status": "error", "error": f"{type(exc).__name__}: {exc}",
                        "processed": processed})
        raise


def _fault_worker(queue_path, marker):
    queue = DurableDockerRolloutQueue(queue_path)

    def after_verify(lease, result):
        if result["status"] != "graded":
            raise RuntimeError("Fault probe did not reach a resolved verifier result")
        _write(marker, {"job_id": lease.job_id, "token": lease.token,
                        "attempt": lease.attempt, "result_sha256": hashlib.sha256(
                            json.dumps(result, sort_keys=True).encode()).hexdigest()})
        while True:
            time.sleep(1)

    queue.process_one("doomed-verifier", revision_provider=lambda: REVISION,
                      lease_seconds=1.5, after_verify=after_verify)


def _await(path, process, seconds=120):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        if not process.is_alive():
            raise RuntimeError(f"Worker exited before publishing {path.name}: {process.exitcode}")
        time.sleep(0.02)
    raise RuntimeError(f"Timed out waiting for {path.name}")


def _signature(reward, evidence, source, actions, action_observations_sha256):
    return {"reward": reward, "case_results": evidence["case_results"],
            "workspace_sha256": evidence["workspace_sha256"],
            "verifier_sha256": evidence["verifier_sha256"],
            "image_sha256": evidence["image_sha256"],
            "source_sha256": source,
            "action_observations_sha256": action_observations_sha256,
            "action_kinds": [action["action"] for action in actions]}


def _control(task, actions, task_dir, image, root):
    jobs = [{"task": task, "seed_dir": task_dir / "seed",
             "verifier_dir": task_dir / "verifier", "image": image,
             "actions": actions[name], "policy_id": POLICY,
             "visible_check": VISIBLE, "verifier_workers": 1}
            for name in ("repair", "baseline")]
    started = time.monotonic()
    report = run_disaggregated_coding_batch(
        jobs=jobs, output_root=root, actor_workers=1, verification_workers=1,
        actor_queue_capacity=1, verifier_queue_capacity=1)
    wall = time.monotonic() - started
    if report["graded_count"] != 2 or report["error_count"]:
        raise RuntimeError("Control failed to grade both complete episodes")
    signatures = {}
    for name, item in zip(("repair", "baseline"), report["jobs"]):
        detail = json.loads((Path(item["output"]) / "report.json").read_text(encoding="utf-8"))
        evidence = detail["verification"]["evidence"]
        signatures[name] = _signature(item["reward"], evidence,
                                      evidence["workspace_sha256"], actions[name],
                                      _observations_sha(detail["opening"], detail["transitions"]))
    return {"condition": "in_memory_pipeline", "wall_seconds": wall,
            "graded_episodes_per_second": 2 / wall, "signatures": signatures,
            "control_report_wall_seconds": report["wall_seconds"]}


def _durable(task, actions, task_dir, image, root):
    root.mkdir()
    queue_dir = root / "queue"
    queue = DurableDockerRolloutQueue(queue_dir / "work.sqlite")
    output = root / "worker.json"
    worker = mp.get_context("spawn").Process(target=_worker_until_done,
        args=(str(queue.path), ("repair", "baseline"), str(output)))
    started = time.monotonic()
    worker.start()
    try:
        submissions = {name: _submission(name, task, actions[name], task_dir,
                                         image, root, queue)
                       for name in ("repair", "baseline")}
        worker.join(timeout=190)
        if worker.is_alive():
            raise RuntimeError("Durable worker exceeded its deadline")
        if worker.exitcode != 0:
            raise RuntimeError(f"Durable worker exited {worker.exitcode}")
        worker_report = json.loads(output.read_text(encoding="utf-8"))
        if worker_report["status"] != "completed":
            raise RuntimeError("Durable worker did not complete")
        wall = time.monotonic() - started
        signatures = {}
        for name in ("repair", "baseline"):
            row = queue.get(name)
            result = row["result"]
            if (row["state"] != "graded" or row["attempt"] != 1
                    or result["policy_revision"] != REVISION
                    or result["task_sha256"] != submissions[name]["receipt"]["task_sha256"]):
                raise RuntimeError("Durable row did not retain exact submission binding")
            signatures[name] = _signature(result["reward"], result["evidence"],
                                          submissions[name]["source_sha256"], actions[name],
                                          submissions[name]["action_observations_sha256"])
        return {"condition": "durable_sqlite_process", "wall_seconds": wall,
                "graded_episodes_per_second": 2 / wall,
                "signatures": signatures, "worker": worker_report,
                "attempts": {name: queue.get(name)["attempt"]
                             for name in ("repair", "baseline")}}
    finally:
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=10)


def _crash_probe(task, actions, task_dir, image, root, expected):
    root.mkdir()
    queue = DurableDockerRolloutQueue(root / "queue" / "work.sqlite")
    submitted = _submission("repair", task, actions["repair"], task_dir, image, root, queue)
    marker = root / "verified_before_commit.json"
    worker = mp.get_context("spawn").Process(target=_fault_worker,
        args=(str(queue.path), str(marker)))
    worker.start()
    try:
        signal = _await(marker, worker)
    finally:
        worker.terminate()
        worker.join(timeout=10)
    if worker.is_alive() or worker.exitcode == 0:
        raise RuntimeError("Fault worker was not killed")
    before = queue.get("repair")
    if before["state"] != "leased" or before["reward"] is not None or before["attempt"] != 1:
        raise RuntimeError("Reward appeared before its commit transaction")
    while time.time() < before["lease_until"] + 0.05:
        time.sleep(0.02)
    replacement = queue.process_one("replacement-verifier",
                                    revision_provider=lambda: REVISION, lease_seconds=10)
    if replacement is None or replacement["status"] != "graded":
        raise RuntimeError("Replacement failed to grade the durable submission")
    after = queue.get("repair")
    if after["attempt"] != 2 or after["state"] != "graded":
        raise RuntimeError("Crash recovery did not use the second lease")
    signature = _signature(after["result"]["reward"], after["result"]["evidence"],
                           submitted["source_sha256"], actions["repair"],
                           submitted["action_observations_sha256"])
    if signature != expected:
        raise RuntimeError("Recovered grade differs from clean control")
    old = VerificationLease("repair", signal["token"], 1,
                            before["lease_until"], {})
    try:
        queue._finish(old, result={"status": "graded", "reward": 1.0})
    except LeaseLost:
        old_commit_rejected = True
    else:
        old_commit_rejected = False
    if not old_commit_rejected or queue.get("repair")["reward"] != 1.0:
        raise RuntimeError("Old verifier was not fenced after replacement")
    return {"worker_killed_after_all_cases_before_commit": True,
            "first_attempt_had_no_reward": True, "replacement_attempt": 2,
            "old_token_rejected": old_commit_rejected,
            "signature": signature, "worker_exitcode": worker.exitcode}


def benchmark(*, task_dir, image, output, pairs=2):
    if type(pairs) is not int or not 1 <= pairs <= 5:
        raise ValueError("pairs must be in [1, 5]")
    task_dir, output = Path(task_dir).resolve(), Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    task, frozen, binding, actions = _inputs(task_dir, image, output)
    rows = []
    for pair in range(1, pairs + 1):
        order = ("in_memory_pipeline", "durable_sqlite_process") if pair % 2 else (
            "durable_sqlite_process", "in_memory_pipeline")
        for condition in order:
            root = output / f"pair-{pair}-{condition}"
            row = (_control(task, actions, task_dir, image, root)
                   if condition == "in_memory_pipeline"
                   else _durable(task, actions, task_dir, image, root))
            row["pair"] = pair
            rows.append(row)
            print(f"pair {pair} {condition}: {row['wall_seconds']:.3f}s / 2 graded", flush=True)
        if rows[-1]["signatures"] != rows[-2]["signatures"]:
            raise RuntimeError("Complete episode signatures differ between arms")
    expected = rows[0]["signatures"]
    if ([expected[name]["reward"] for name in ("repair", "baseline")] != [1.0, 0.0]
            or any(len(expected[name]["case_results"]) != 14
                   for name in ("repair", "baseline"))
            or any(row["signatures"] != expected for row in rows)):
        raise RuntimeError("Pinned Boltons reward/case/source parity failed")
    crash = _crash_probe(task, actions, task_dir, image, output / "crash-probe",
                         expected["repair"])
    medians = {condition: statistics.median(row["wall_seconds"] for row in rows
                                            if row["condition"] == condition)
               for condition in ("in_memory_pipeline", "durable_sqlite_process")}
    report = {"schema": "durable-docker-rollout-ab-v1", "task_id": frozen["task_id"],
              "task_sha256": frozen["task_sha256"], "policy_revision": REVISION,
              "source_sdist_sha256": task["metadata"]["source_sdist_sha256"],
              "seed_sha256": binding["seed_workspace_sha256"],
              "verifier_sha256": binding["verifier_sha256"],
              "image_sha256": binding["image_sha256"], "host": platform.platform(),
              "implementation_sha256": {
                  "benchmark": _sha(__file__), "durable_queue": _sha(durable_rollout.__file__),
                  "docker_adapter": _sha(coding_env.__file__),
                  "in_memory_pipeline": _sha(disaggregated_rollout.__file__),
                  "realworld_contract": _sha(realworld.__file__)},
              "pairs": pairs, "episodes_per_arm": 2, "cases_per_episode": 14,
              "actor_workers": 1, "verifier_workers": 1,
              "durable_worker_processes": 1,
              "median_wall_seconds": medians,
              "median_graded_throughput_ratio_durable_to_control":
                  medians["in_memory_pipeline"] / medians["durable_sqlite_process"],
              "rows": rows, "crash_probe": crash,
              "limitations": ["Scripted public repair/baseline, not model inference or learning",
                              "SQLite same-host durability, not distributed RollArt service",
                              "The verifier may re-execute after crash; only reward publication is at most once",
                              "The actor-to-queue handoff is not recoverable before enqueue returns"]}
    _write(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pairs", type=int, default=2)
    args = parser.parse_args()
    result = benchmark(task_dir=args.task_dir, image=args.image,
                       output=args.output, pairs=args.pairs)
    print(json.dumps({"median_wall_seconds": result["median_wall_seconds"],
                      "throughput_ratio": result[
                          "median_graded_throughput_ratio_durable_to_control"],
                      "crash_recovered": result["crash_probe"]["old_token_rejected"]},
                     indent=2))
