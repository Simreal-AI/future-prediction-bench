"""Portable entry points for collection, private baselines, and forecasting."""

import argparse
import json
import os
import signal
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

from . import __version__
from .baselines import internal_model_from_env, load_baseline_config, seal_baselines
from .coding_env import CodingRuntimeError
from .demo import run_demo, write_jsonl
from .http import strict_json_loads
from .pipeline import Pipeline, load_config
from .providers import BraveResearchProvider, ChatCompletionModel, ProviderError
from .runner import run_questions
from .schema import parse_timestamp, public_question, validate_question
from .store import Store
from .training import ADVANTAGE_METHODS, collect_rollout_group, prepare_training_groups


@contextmanager
def database_lock(path):
    """An OS advisory lock prevents overlapping collectors/workers for one DB."""
    path = Path(str(path) + ".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            if path.stat().st_size == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                raise ValueError("Another worker owns this database") from None
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                raise ValueError("Another worker owns this database") from None
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


def eligible_question_ids(store, *, limit=20, track="benchmark", question_ids=None):
    now = store.now()
    output = []
    requested = set(question_ids) if question_ids else None
    rows = store.db.execute("SELECT question_id FROM questions ORDER BY question_id").fetchall()
    known = {row[0] for row in rows}
    if requested and not requested <= known:
        raise ValueError("Unknown requested question ID")
    questions = sorted((store.question(row[0]) for row in rows), key=lambda q: parse_timestamp(q["forecast_deadline"]))
    for question in questions:
        if requested is not None and question["question_id"] not in requested:
            continue
        if track == "rl" and question["split"] != "train":
            continue
        if not parse_timestamp(question["issued_at"]) <= now < parse_timestamp(question["forecast_deadline"]):
            continue
        if store.db.execute("SELECT 1 FROM resolutions WHERE question_id=?", (question["question_id"],)).fetchone():
            continue
        output.append(question["question_id"])
    return output[:limit]


def _baseline_model():
    keys = ("FPB_BASELINE_BASE_URL", "FPB_BASELINE_API_KEY", "FPB_BASELINE_MODEL_NAME")
    values = [os.environ.get(key) for key in keys]
    if any(values) and not all(values):
        raise ValueError("All three FPB_BASELINE_* settings must be configured together")
    return internal_model_from_env() if all(values) else None


def forecast(store, args):
    group_size = getattr(args, "rollouts_per_question", 1)
    revision = getattr(args, "policy_revision", None)
    collection_id = getattr(args, "collection_id", None)
    if not 1 <= group_size <= 64:
        raise ValueError("rollouts-per-question must be between 1 and 64")
    if group_size > 1 and (args.track != "rl" or not revision or not collection_id):
        raise ValueError("Grouped rollouts require --track rl, --policy-revision, and --collection-id")
    model = ChatCompletionModel.from_env()
    temperature = getattr(args, "temperature", None)
    if temperature is None:
        temperature = 0.7 if group_size > 1 else 0.0
    if not 0 <= temperature <= 2 or (group_size > 1 and temperature == 0):
        raise ValueError("temperature must be in [0, 2] and positive for grouped rollouts")
    model.temperature = temperature
    provider = BraveResearchProvider.from_env() if args.research_mode == "self_research" else None
    questions = eligible_question_ids(store, limit=args.limit, track=args.track, question_ids=args.question_id)
    baselines = []
    if args.reward_mode == "baseline_improvement":
        baselines = seal_baselines(store, questions, config=load_baseline_config(args.baseline_config), internal_model=_baseline_model())
    options = {"market_mode": args.market_mode, "research_mode": args.research_mode,
               "reward_mode": args.reward_mode, "max_steps": args.max_steps, "max_calls": args.max_calls,
               "max_output_tokens": args.max_output_tokens, "max_wall_seconds": args.max_wall_seconds}
    if group_size > 1:
        from .store import digest
        reports = []
        for question_id in questions:
            group_id = "group-" + digest({"collection_id": collection_id, "question_id": question_id})
            reports.extend(collect_rollout_group(store, question_id, group_id=group_id, group_size=group_size,
                                                policy_revision=revision, model=model, provider=provider, **options))
    else:
        reports = run_questions(store, questions, model=model, provider=provider, track=args.track, **options)
    return {"baselines": baselines, "forecasts": reports}


def public_dataset_question(question):
    result = {**public_question(question), "event_id": question["event_id"], "cluster_id": question["cluster_id"], "split": question["split"]}
    metadata = question.get("metadata", {})
    result["metadata"] = {key: metadata[key] for key in ("source_id", "template_version", "target_at") if key in metadata}
    return result


def parser():
    root = argparse.ArgumentParser(description=f"Future Prediction Bench {__version__}")
    sub = root.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Run forecasting tracks with synthetic fixtures")
    demo.add_argument("--output", default="runs/demo")
    smoke = sub.add_parser("rl-smoke", help="Exercise analyst trajectories and RL preparation with synthetic fixtures")
    smoke.add_argument("--output", default="runs/rl-smoke")
    coding_smoke = sub.add_parser("realworld-code-smoke", help="Run an isolated coding fixture with a locally cached Docker image")
    coding_smoke.add_argument("--image", required=True, help="Locally cached image; the command never pulls an image")
    coding_smoke.add_argument("--output", default="runs/realworld-code-smoke")
    coding_run = sub.add_parser("realworld-code-run", help="Replay scripted actions on a trusted real-world coding task")
    for required in ("task", "seed", "verifier", "image", "actions", "output", "visible-check"):
        coding_run.add_argument("--" + required, required=True)
    coding_run.add_argument("--policy-id", default="external-scripted-policy")
    coding_run.add_argument("--registry", required=True, help="Persistent train/dev/test task registry")
    coding_run.add_argument("--verifier-workers", type=int, default=1,
                            help="Independent hidden-case containers in parallel (1-8)")
    coding_microvm = sub.add_parser("realworld-code-microvm", help="Replay coding actions in an offline ARM64 QEMU/HVF VM with full-state checkpoints")
    for required in ("task", "verifier", "assets", "actions", "output", "visible-check"):
        coding_microvm.add_argument("--" + required, required=True)
    coding_microvm.add_argument("--policy-id", default="external-scripted-policy")
    coding_microvm.add_argument("--registry", help="Required for a non-fixture VM task")
    coding_microvm.add_argument(
        "--stateless-verifier-contract",
        help="Explicit host-authored task contract for one namespaced hidden-case batch")
    coding_batch = sub.add_parser("realworld-code-batch", help="Run bounded independent coding episodes from a JSON job list")
    coding_batch.add_argument("--jobs", required=True, help="JSON array of operator-authored jobs")
    coding_batch.add_argument("--output", required=True)
    coding_batch.add_argument("--registry", help="Required when any task is not a fixture")
    coding_batch.add_argument("--max-workers", type=int, default=4)
    coding_pipeline = sub.add_parser(
        "realworld-code-pipeline",
        help="Run coding episodes with separate, bounded interaction and verification workers")
    coding_pipeline.add_argument("--jobs", required=True, help="JSON array of operator-authored jobs")
    coding_pipeline.add_argument("--output", required=True)
    coding_pipeline.add_argument("--registry", help="Required when any task is not a fixture")
    coding_pipeline.add_argument("--actor-workers", type=int, default=2)
    coding_pipeline.add_argument("--verification-workers", type=int, default=2)
    coding_pipeline.add_argument("--actor-queue-capacity", type=int, default=2)
    coding_pipeline.add_argument("--verifier-queue-capacity", type=int, default=2)
    coding_branches = sub.add_parser("realworld-code-branches", help="Run trusted shared-prefix filesystem branches")
    for required in ("task", "seed", "verifier", "image", "plan", "output", "visible-check"):
        coding_branches.add_argument("--" + required, required=True)
    coding_branches.add_argument("--policy-id", required=True)
    coding_branches.add_argument("--registry", help="Required when the task is not a fixture")
    coding_branches.add_argument("--max-workers", type=int, default=1)
    coding_branches.add_argument("--verifier-workers", type=int, default=1)
    pack = sub.add_parser("seal-evidence-pack", help="Seal recorded pre-cutoff search/open events for closed historical replay")
    for required in ("events", "forecast-cutoff", "output"):
        pack.add_argument("--" + required, required=True)
    replay_bench = sub.add_parser("evidence-replay-bench", help="Time a bounded synthetic source against in-memory evidence replay")
    replay_bench.add_argument("--iterations", type=int, default=20)
    validate = sub.add_parser("validate", help="Validate question JSONL")
    validate.add_argument("questions")
    names = ("collect", "resolve-due", "cycle", "worker", "retry-resolution", "pipeline-status",
             "forecast", "seal-baselines", "export-questions", "status", "export-training", "prepare-rl", "resolve")
    for name in names:
        command = sub.add_parser(name)
        command.add_argument("--db", required=True)
        command.add_argument("--mode", choices=("live", "fixture"), default="live")
        if name in {"collect", "resolve-due", "cycle", "worker", "retry-resolution", "pipeline-status"}:
            command.add_argument("--config", default="configs/sources.json")
        if name in {"cycle", "worker"}:
            command.add_argument("--with-forecasts", action="store_true", help="Opt in to credential-backed baseline/model/search calls")
        if name == "worker":
            command.add_argument("--interval-seconds", type=int, default=3600)
            command.add_argument("--max-cycles", type=int)
        if name in {"forecast", "seal-baselines", "cycle", "worker"}:
            command.add_argument("--baseline-config")
            command.add_argument("--limit", type=int, default=20)
            command.add_argument("--question-id", action="append")
            command.add_argument("--track", choices=("benchmark", "rl"), default="benchmark")
        if name in {"forecast", "cycle", "worker"}:
            command.add_argument("--research-mode", choices=("self_research", "no_search"), default="self_research")
            command.add_argument("--market-mode", choices=("no_consensus", "market_aware"), default="no_consensus")
            command.add_argument("--reward-mode", choices=("baseline_improvement", "negative_brier"), default="baseline_improvement")
            command.add_argument("--max-steps", type=int, default=12)
            command.add_argument("--max-calls", type=int, default=8)
            command.add_argument("--max-output-tokens", type=int, default=8192)
            command.add_argument("--max-wall-seconds", type=float, default=300)
            command.add_argument("--rollouts-per-question", type=int, default=1)
            command.add_argument("--policy-revision")
            command.add_argument("--collection-id")
            command.add_argument("--temperature", type=float)
        if name in {"export-questions", "export-training", "prepare-rl"}:
            command.add_argument("--output", required=True)
        if name == "prepare-rl":
            command.add_argument("--current-policy-revision", required=True)
            command.add_argument("--advantage-method", choices=ADVANTAGE_METHODS, default="rloo")
        if name == "resolve":
            command.add_argument("--resolution", required=True)
        if name == "retry-resolution":
            command.add_argument("--question-id", required=True)
    return root


def _print(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


def _cycle(pipeline, store, args):
    report = pipeline.cycle()
    if args.with_forecasts:
        report["prediction"] = forecast(store, args)
    return report


def _worker(pipeline, store, args):
    if not 60 <= args.interval_seconds <= 604800 or (args.max_cycles is not None and args.max_cycles < 1):
        raise ValueError("Worker interval must be 60..604800 seconds; max-cycles must be positive")
    stop = threading.Event()
    original = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in original:
        signal.signal(sig, lambda *_: stop.set())
    cycles = 0
    try:
        while not stop.is_set() and (args.max_cycles is None or cycles < args.max_cycles):
            try:
                _print(_cycle(pipeline, store, args))
            except Exception as exc:
                _print({"status": "cycle_error", "error": type(exc).__name__, "message": str(exc)[:300]})
            cycles += 1
            if args.max_cycles is not None and cycles >= args.max_cycles:
                break
            stop.wait(args.interval_seconds)
    finally:
        for sig, handler in original.items():
            signal.signal(sig, handler)
    return {"status": "worker_stopped", "cycles": cycles}


def main(argv=None):
    args = parser().parse_args(argv)
    store = None
    try:
        if hasattr(args, "limit") and not 1 <= args.limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        if args.command == "demo":
            result = run_demo(args.output)
        elif args.command == "rl-smoke":
            from .training_demo import run_rl_smoke
            result = run_rl_smoke(args.output)
        elif args.command == "realworld-code-smoke":
            from .realworld_demo import run_coding_smoke
            result = run_coding_smoke(image=args.image, output=args.output)
        elif args.command == "realworld-code-run":
            from .realworld_demo import load_coding_actions, run_coding_task
            visible_check = strict_json_loads(args.visible_check)
            result = run_coding_task(task=strict_json_loads(Path(args.task).read_text(encoding="utf-8")),
                                     seed_dir=args.seed, verifier_dir=args.verifier, image=args.image,
                                     actions=load_coding_actions(args.actions), output=args.output,
                                     visible_check=visible_check, policy_id=args.policy_id,
                                     registry_path=args.registry,
                                     verifier_workers=args.verifier_workers)
        elif args.command == "realworld-code-microvm":
            from .microvm_demo import run_microvm_task
            from .realworld_demo import load_coding_actions
            result = run_microvm_task(
                task=strict_json_loads(Path(args.task).read_text(encoding="utf-8")),
                verifier_dir=args.verifier, assets_dir=args.assets,
                actions=load_coding_actions(args.actions), output=args.output,
                visible_check=strict_json_loads(args.visible_check),
                policy_id=args.policy_id, registry_path=args.registry,
                stateless_verifier_contract=args.stateless_verifier_contract,
                stateless_task_path=args.task if args.stateless_verifier_contract else None)
        elif args.command == "realworld-code-batch":
            from .rollout_batch import load_coding_batch_jobs, run_coding_batch
            result = run_coding_batch(jobs=load_coding_batch_jobs(args.jobs),
                                      output_root=args.output, max_workers=args.max_workers,
                                      registry_path=args.registry)
        elif args.command == "realworld-code-pipeline":
            from .rollout_batch import load_coding_batch_jobs
            from .disaggregated_rollout import run_disaggregated_coding_batch
            result = run_disaggregated_coding_batch(
                jobs=load_coding_batch_jobs(args.jobs), output_root=args.output,
                actor_workers=args.actor_workers,
                verification_workers=args.verification_workers,
                actor_queue_capacity=args.actor_queue_capacity,
                verifier_queue_capacity=args.verifier_queue_capacity,
                registry_path=args.registry)
        elif args.command == "realworld-code-branches":
            from .branch_rollouts import run_coding_branches
            from .realworld_demo import load_coding_actions
            plan_path = Path(args.plan).resolve()
            plan = strict_json_loads(plan_path.read_text(encoding="utf-8"))
            if not isinstance(plan, dict) or set(plan) != {"prefix_actions", "branches"}:
                raise ValueError("Branch plan must have prefix_actions and branches")
            def actions(value):
                return load_coding_actions(plan_path.parent / value) if isinstance(value, str) else value
            branches = [{**branch, "actions": actions(branch["actions"])} for branch in plan["branches"]]
            result = run_coding_branches(
                task=strict_json_loads(Path(args.task).read_text(encoding="utf-8")),
                seed_dir=args.seed, verifier_dir=args.verifier, image=args.image,
                prefix_actions=actions(plan["prefix_actions"]), branches=branches,
                output_root=args.output, policy_id=args.policy_id,
                registry_path=args.registry, visible_check=strict_json_loads(args.visible_check),
                verifier_workers=args.verifier_workers, max_workers=args.max_workers)
        elif args.command == "seal-evidence-pack":
            from .evidence_pack import EvidencePack
            events = strict_json_loads(Path(args.events).read_text(encoding="utf-8"))
            sealed = EvidencePack.from_events(events, args.forecast_cutoff)
            output = Path(args.output)
            if output.exists():
                raise ValueError("Evidence pack output already exists")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(sealed.to_dict(), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            result = {"pack_sha256": sealed.sha256, "output": str(output.resolve()),
                      "replay_scope": "recorded exact search and open calls only"}
        elif args.command == "evidence-replay-bench":
            from .evidence_pack import synthetic_replay_microbenchmark
            result = synthetic_replay_microbenchmark(iterations=args.iterations)
        elif args.command == "validate":
            seen = set()
            for number, line in enumerate(Path(args.questions).read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                question = strict_json_loads(line)
                validate_question(question)
                if question["question_id"] in seen:
                    raise ValueError(f"Duplicate question_id on line {number}")
                seen.add(question["question_id"])
            result = {"valid_questions": len(seen), "validation": "schema only"}
        else:
            if args.command not in {"collect", "cycle", "worker"} and not Path(args.db).is_file():
                raise ValueError("Database does not exist")
            with database_lock(args.db):
                store = Store(args.db, mode=args.mode)
                if hasattr(args, "config"):
                    pipeline = Pipeline(store, load_config(args.config))
                    if args.command == "collect":
                        result = pipeline.collect()
                    elif args.command == "resolve-due":
                        result = pipeline.resolve_due()
                    elif args.command == "cycle":
                        result = _cycle(pipeline, store, args)
                    elif args.command == "worker":
                        result = _worker(pipeline, store, args)
                    elif args.command == "retry-resolution":
                        result = pipeline.retry(args.question_id)
                    else:
                        result = pipeline.status()
                elif args.command == "forecast":
                    result = forecast(store, args)
                elif args.command == "seal-baselines":
                    ids = eligible_question_ids(store, limit=args.limit, track=args.track, question_ids=args.question_id)
                    result = seal_baselines(store, ids, config=load_baseline_config(args.baseline_config), internal_model=_baseline_model())
                elif args.command == "prepare-rl":
                    prepared = prepare_training_groups(store.export_training(), current_policy_revision=args.current_policy_revision,
                                                       available_at=store.now().isoformat(), method=args.advantage_method, run_mode=args.mode)
                    path = Path(args.output)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(prepared, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
                    result = {**prepared["summary"], "trainer_ready": False, "output": str(path.resolve())}
                elif args.command in {"export-questions", "export-training"}:
                    records = (store.export_training() if args.command == "export-training" else
                               [public_dataset_question(store.question(row[0])) for row in store.db.execute("SELECT question_id FROM questions ORDER BY question_id")])
                    write_jsonl(args.output, records)
                    result = {"records": len(records), "output": str(Path(args.output).resolve())}
                elif args.command == "resolve":
                    result = store.resolve(**strict_json_loads(Path(args.resolution).read_text(encoding="utf-8")))
                else:
                    result = store.summary()
        _print(result)
        return 0
    except (ValueError, TypeError, KeyError, OSError, ProviderError, CodingRuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        if store is not None:
            store.close()
