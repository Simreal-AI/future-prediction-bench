"""Offline, fail-closed input audit for a future Qwen3-8B coding RL job.

This does not load a model, run a GPU, execute an optimizer, or attest that
self-reported token likelihoods came from the pinned weights. A successful
input contract remains ``trainer_ready=False`` until an independent training
smoke observes an actual optimizer update and validates likelihood provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse


SCHEMA = "qwen8b-gpu-job-preflight-0.1"
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
VERSION = re.compile(r"([0-9]+)\.([0-9]+)\.([0-9]+)\Z")
MAX_JSON_BYTES = 20_000_000


class PreflightError(ValueError):
    """A structural input contract failed without exposing file content."""


def _fail(reason):
    raise PreflightError(reason)


def _exact(value, keys, reason):
    if not isinstance(value, dict) or set(value) != set(keys):
        _fail(reason)


def _hex(value, pattern, reason):
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        _fail(reason)


def _positive_int(value, reason, *, maximum=None):
    if type(value) is not int or value < 1 or (maximum is not None and value > maximum):
        _fail(reason)


def _finite(value, reason, *, minimum=None, maximum=None):
    if type(value) not in {int, float}:
        _fail(reason)
    try:
        finite = math.isfinite(value)
    except OverflowError as exc:
        raise PreflightError(reason) from exc
    if (not finite or (minimum is not None and value < minimum)
            or (maximum is not None and value > maximum)):
        _fail(reason)


def _strict_json(data, reason):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                _fail(reason + ":duplicate_key")
            result[key] = value
        return result

    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=pairs,
                          parse_constant=lambda _: _fail(reason + ":nonfinite"))
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise PreflightError(reason + ":invalid_json") from exc


def _read_ref(base, ref, reason, *, parse_json=True):
    _exact(ref, {"path", "sha256"}, reason + ":reference_fields")
    name = ref["path"]
    if (not isinstance(name, str) or not name or Path(name).is_absolute()
            or any(part in {"", ".", ".."} for part in Path(name).parts)):
        _fail(reason + ":unsafe_path")
    _hex(ref["sha256"], HEX64, reason + ":sha256")
    path = base / name
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError) as exc:
        raise PreflightError(reason + ":missing_or_external_file") from exc
    if (not resolved.is_relative_to(base) or path.is_symlink()
            or not resolved.is_file()):
        _fail(reason + ":missing_or_external_file")
    if resolved.stat().st_size > MAX_JSON_BYTES:
        _fail(reason + ":file_too_large")
    data = resolved.read_bytes()
    if hashlib.sha256(data).hexdigest() != ref["sha256"]:
        _fail(reason + ":file_hash_mismatch")
    return _strict_json(data, reason) if parse_json else data


def _check_model(base, model):
    _exact(model, {"repository", "revision", "tokenizer_revision", "config",
                   "tokenizer_config", "chat_template_sha256", "enable_thinking"},
           "model_fields")
    if model["repository"] != "Qwen/Qwen3-8B":
        _fail("model_repository")
    _hex(model["revision"], HEX40, "model_revision")
    _hex(model["tokenizer_revision"], HEX40, "tokenizer_revision")
    if model["tokenizer_revision"] != model["revision"]:
        _fail("tokenizer_model_revision_mismatch")
    _hex(model["chat_template_sha256"], HEX64, "chat_template_sha256")
    if type(model["enable_thinking"]) is not bool:
        _fail("thinking_mode_not_explicit")
    config = _read_ref(base, model["config"], "model_config")
    expected = {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"],
                "hidden_size": 4096, "num_hidden_layers": 36,
                "num_attention_heads": 32, "num_key_value_heads": 8,
                "head_dim": 128, "vocab_size": 151936}
    if not isinstance(config, dict) or any(config.get(key) != value for key, value in expected.items()):
        _fail("qwen3_8b_config_mismatch")
    if config.get("torch_dtype") != "bfloat16":
        _fail("model_config_precision")
    tokenizer = _read_ref(base, model["tokenizer_config"], "tokenizer_config")
    template = tokenizer.get("chat_template") if isinstance(tokenizer, dict) else None
    if not isinstance(template, str) or not template:
        _fail("missing_chat_template")
    if hashlib.sha256(template.encode("utf-8")).hexdigest() != model["chat_template_sha256"]:
        _fail("chat_template_mismatch")


def _check_runtime(runtime):
    _exact(runtime, {"engine", "version", "logprobs_mode", "image_sha256",
                     "max_model_len", "max_completion_len"}, "runtime_fields")
    if runtime["engine"] != "vllm" or runtime["logprobs_mode"] != "processed_logprobs":
        _fail("unsupported_sampling_logprob_contract")
    parsed_version = (VERSION.fullmatch(runtime["version"])
                      if isinstance(runtime["version"], str) else None)
    if parsed_version is None:
        _fail("unpinned_runtime_version")
    if tuple(map(int, parsed_version.groups())) < (0, 8, 5):
        _fail("qwen3_unsupported_vllm_version")
    if (not isinstance(runtime["image_sha256"], str)
            or not runtime["image_sha256"].startswith("sha256:")):
        _fail("runtime_image_digest")
    _hex(runtime["image_sha256"][7:], HEX64, "runtime_image_digest")
    _positive_int(runtime["max_model_len"], "max_model_len", maximum=32768)
    _positive_int(runtime["max_completion_len"], "max_completion_len",
                  maximum=runtime["max_model_len"])


def _canonical_repository(value, reason):
    if (not isinstance(value, str) or not value or "%" in value or "\\" in value
            or any(ord(char) < 32 or ord(char) == 127 for char in value)):
        _fail(reason)
    try:
        parsed = urlparse(value)
        port = parsed.port
    except ValueError as exc:
        raise PreflightError(reason) from exc
    host = parsed.hostname
    if (parsed.scheme != "https" or not host
            or re.fullmatch(r"[a-z0-9.-]+", host) is None
            or host.startswith(".") or host.endswith(".") or ".." in host
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.params or parsed.netloc.endswith(":")
            or not parsed.path.startswith("/")):
        _fail(reason)
    segments = parsed.path.split("/")[1:]
    if segments and segments[-1] == "":
        segments.pop()
    if len(segments) < 2 or any(segment in {"", ".", ".."} for segment in segments):
        _fail(reason)
    host = "github.com" if host == "www.github.com" else host
    if host == "github.com" and len(segments) != 2:
        _fail(reason)
    repository = segments[-1].lower().removesuffix(".git")
    if not repository:
        _fail(reason)
    segments[-1] = repository
    authority = host if port in {None, 443} else f"{host}:{port}"
    return authority + "/" + "/".join(segment.lower() for segment in segments)


def _check_manifest(base, ref, split):
    manifest = _read_ref(base, ref, split + "_manifest")
    _exact(manifest, {"schema_version", "split", "track", "records"}, split + "_manifest_fields")
    if (manifest["schema_version"] != "qwen8b-coding-manifest-0.1"
            or manifest["split"] != split or manifest["track"] != "coding"
            or not isinstance(manifest["records"], list)
            or len(manifest["records"]) < (2 if split == "train" else 1)):
        _fail(split + "_manifest_scope")
    seen = set()
    for row in manifest["records"]:
        _exact(row, {"task_id", "cluster_id", "repository", "source_commit",
                     "source_sha256", "verifier_sha256", "environment_image_sha256",
                     "license", "is_fixture"}, split + "_record_fields")
        for key in ("task_id", "cluster_id", "license"):
            if not isinstance(row[key], str) or not row[key].strip() or len(row[key]) > 256:
                _fail(split + "_record_" + key)
        if row["task_id"] in seen:
            _fail(split + "_duplicate_task")
        seen.add(row["task_id"])
        _canonical_repository(row["repository"], split + "_repository_url")
        _hex(row["source_commit"], HEX40, split + "_source_commit")
        _hex(row["source_sha256"], HEX64, split + "_source_sha256")
        _hex(row["verifier_sha256"], HEX64, split + "_verifier_sha256")
        image = row["environment_image_sha256"]
        if not isinstance(image, str) or not image.startswith("sha256:"):
            _fail(split + "_environment_image_sha256")
        _hex(image[7:], HEX64, split + "_environment_image_sha256")
        if row["is_fixture"] is not False:
            _fail(split + "_public_fixture_not_training_data")
    return manifest["records"]


def _check_dataset(base, dataset):
    _exact(dataset, {"train", "dev"}, "dataset_fields")
    train = _check_manifest(base, dataset["train"], "train")
    dev = _check_manifest(base, dataset["dev"], "dev")
    for key in ("task_id", "cluster_id", "repository", "source_commit", "source_sha256"):
        if key == "repository":
            canonical = lambda row: _canonical_repository(row[key], "repository_url")
        else:
            canonical = lambda row: row[key].lower()
        left = {canonical(row) for row in train}
        right = {canonical(row) for row in dev}
        if left & right:
            _fail("train_dev_" + key + "_overlap")
    return train, dev


def _check_trainer(base, trainer):
    _exact(trainer, {"framework", "version", "entrypoint", "algorithm", "optimizer",
                     "learning_rate", "max_steps", "policy_revision", "max_policy_lag",
                     "beta", "scale_rewards", "loss_type"}, "trainer_fields")
    if trainer["framework"] != "trl" or trainer["algorithm"] != "grpo":
        _fail("unsupported_trainer_algorithm")
    if not isinstance(trainer["version"], str) or not VERSION.fullmatch(trainer["version"]):
        _fail("unpinned_trainer_version")
    if trainer["optimizer"] != "adamw_torch_fused":
        _fail("optimizer_not_explicitly_supported")
    _finite(trainer["learning_rate"], "learning_rate", minimum=1e-9, maximum=1e-3)
    _positive_int(trainer["max_steps"], "max_steps", maximum=1_000_000)
    _hex(trainer["policy_revision"], HEX64, "policy_revision")
    if type(trainer["max_policy_lag"]) is not int or trainer["max_policy_lag"] != 0:
        _fail("stale_policy_requires_off_policy_validation")
    _finite(trainer["beta"], "beta", minimum=0, maximum=1)
    if trainer["scale_rewards"] != "none" or trainer["loss_type"] != "dr_grpo":
        _fail("unsupported_grpo_loss_configuration")
    if (not isinstance(trainer["entrypoint"], dict)
            or not isinstance(trainer["entrypoint"].get("path"), str)
            or not trainer["entrypoint"]["path"].endswith(".py")):
        _fail("trainer_entrypoint_python_file")
    _read_ref(base, trainer["entrypoint"], "trainer_entrypoint", parse_json=False)


def _check_rollouts(base, ref, *, model, runtime, trainer, train, train_manifest_sha256):
    export = _read_ref(base, ref, "tokenized_rollouts")
    _exact(export, {"schema_version", "record_type", "trainer_ready", "model_revision",
                    "tokenizer_revision", "chat_template_sha256", "enable_thinking",
                    "policy_revision", "logprobs_mode", "train_manifest_sha256",
                    "samples"}, "rollout_export_fields")
    if (export["schema_version"] != "qwen8b-tokenized-rollouts-0.1"
            or export["record_type"] != "sampled_token_trajectories"
            or export["trainer_ready"] is not False):
        _fail("not_a_tokenized_rollout_export")
    for key, expected in (("model_revision", model["revision"]),
                          ("tokenizer_revision", model["tokenizer_revision"]),
                          ("chat_template_sha256", model["chat_template_sha256"]),
                          ("enable_thinking", model["enable_thinking"]),
                          ("policy_revision", trainer["policy_revision"]),
                          ("logprobs_mode", runtime["logprobs_mode"]),
                          ("train_manifest_sha256", train_manifest_sha256)):
        if export[key] != expected:
            _fail("rollout_" + key + "_mismatch")
    samples = export["samples"]
    if not isinstance(samples, list) or not samples or len(samples) > 100_000:
        _fail("missing_or_excessive_tokenized_samples")
    train_tasks = {row["task_id"]: row for row in train}
    seen_tasks, groups = set(), defaultdict(list)
    max_delta = 0.0
    token_count = 0
    for item in samples:
        _exact(item, {"task_id", "group_id", "sample_index", "group_size", "reward",
                      "input_ids", "assistant_loss_mask", "behavior_token_logprobs",
                      "teacher_forced_token_logprobs", "source_commit",
                      "truncated"}, "token_sample_fields")
        if item["task_id"] not in train_tasks:
            _fail("nontrain_task_in_token_export")
        if item["source_commit"] != train_tasks[item["task_id"]]["source_commit"]:
            _fail("rollout_task_source_commit_mismatch")
        seen_tasks.add(item["task_id"])
        if not isinstance(item["group_id"], str) or not item["group_id"].strip():
            _fail("missing_group_id")
        _positive_int(item["group_size"], "group_size", maximum=64)
        if item["group_size"] < 2 or type(item["sample_index"]) is not int or not 0 <= item["sample_index"] < item["group_size"]:
            _fail("group_index_or_size")
        _finite(item["reward"], "reward", minimum=-1, maximum=1)
        if item["truncated"] is not False:
            _fail("truncated_token_trajectory")
        ids, mask = item["input_ids"], item["assistant_loss_mask"]
        behavior = item["behavior_token_logprobs"]
        teacher = item["teacher_forced_token_logprobs"]
        if (not isinstance(ids, list) or not 2 <= len(ids) <= runtime["max_model_len"]
                or not all(isinstance(value, list) and len(value) == len(ids)
                           for value in (mask, behavior, teacher))):
            _fail("token_array_alignment")
        if any(type(token) is not int or not 0 <= token < 151936 for token in ids):
            _fail("token_id_out_of_vocabulary")
        if any(type(flag) is not int or flag not in {0, 1} for flag in mask):
            _fail("invalid_assistant_mask")
        if not 0 in mask or not 1 in mask or sum(mask) > runtime["max_completion_len"]:
            _fail("missing_context_or_generated_tokens")
        for flag, old, replay in zip(mask, behavior, teacher):
            if flag == 0:
                if old is not None or replay is not None:
                    _fail("observation_or_context_has_policy_loss")
            else:
                _finite(old, "missing_or_nonfinite_behavior_logprob", maximum=0)
                _finite(replay, "missing_or_nonfinite_teacher_logprob", maximum=0)
                max_delta = max(max_delta, abs(old - replay))
        token_count += sum(mask)
        groups[item["group_id"]].append(item)
    if seen_tasks != set(train_tasks):
        _fail("train_task_without_tokenized_group")
    for members in groups.values():
        first = members[0]
        if (len(members) != first["group_size"]
                or sorted(item["sample_index"] for item in members) != list(range(first["group_size"]))
                or any(item["task_id"] != first["task_id"]
                       or item["group_size"] != first["group_size"] for item in members)):
            _fail("incomplete_or_mixed_rollout_group")
    if max_delta > 0.05:
        _fail("behavior_teacher_logprob_mismatch")
    return len(samples), len(groups), token_count, max_delta


def audit_gpu_job(plan_path):
    """Check local input structure and hashes; never assert a GPU update ran."""
    plan_path = Path(plan_path).resolve()
    report = {"schema_version": SCHEMA, "input_contract_passed": False,
              "trainer_ready": False, "gpu_executed": False,
              "optimizer_update_observed": False, "errors": []}
    try:
        if not plan_path.is_file() or plan_path.stat().st_size > 1_000_000:
            _fail("plan_missing_or_too_large")
        data = plan_path.read_bytes()
        report["plan_sha256"] = hashlib.sha256(data).hexdigest()
        plan = _strict_json(data, "plan")
        _exact(plan, {"schema_version", "model", "runtime", "dataset", "rollouts", "trainer"},
               "plan_fields")
        if plan["schema_version"] != SCHEMA:
            _fail("plan_schema_version")
        base = plan_path.parent
        _check_model(base, plan["model"])
        _check_runtime(plan["runtime"])
        train, dev = _check_dataset(base, plan["dataset"])
        _check_trainer(base, plan["trainer"])
        samples, groups, generated_tokens, max_delta = _check_rollouts(
            base, plan["rollouts"], model=plan["model"], runtime=plan["runtime"],
            trainer=plan["trainer"], train=train,
            train_manifest_sha256=plan["dataset"]["train"]["sha256"])
        report.update({"input_contract_passed": True,
                       "model_repository": plan["model"]["repository"],
                       "model_revision": plan["model"]["revision"],
                       "train_tasks": len(train), "dev_tasks": len(dev),
                       "tokenized_samples": samples, "complete_groups": groups,
                       "generated_policy_tokens": generated_tokens,
                       "max_behavior_teacher_logprob_delta": max_delta})
    except (PreflightError, OSError, TypeError, KeyError, ValueError) as exc:
        report["errors"] = [str(exc) if isinstance(exc, PreflightError) else "malformed_or_unreadable_input"]
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    result = audit_gpu_job(args.plan)
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        path = Path(args.output).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["input_contract_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
