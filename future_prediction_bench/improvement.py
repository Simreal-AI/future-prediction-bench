"""Pure, offline candidate manifests and sealed development promotion checks.

Hashes provide content integrity, not provenance or access control. A trusted
caller must retain commitments before outcomes exist and export honest scores.
This module never reads files, calls a model, changes weights, or deploys code.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from statistics import fmean

from .schema import parse_timestamp


_QUESTION_FIELDS = {
    "question_id", "event_id", "cluster_id", "forecast_deadline",
    "outcome_not_before", "resolve_after",
}
_ROW_FIELDS = {
    "question_id", "incumbent_submitted_at", "candidate_submitted_at",
    "resolved_at", "incumbent_brier", "candidate_brier",
}
_CONTRACT_FIELDS = {
    "schema_version", "split", "metric", "aggregation", "frozen_at", "dev_start",
    "dev_end", "questions", "min_clusters", "min_improvement", "confidence",
    "max_comparisons", "bootstrap_samples", "seed", "sha256",
}
_MANIFEST_FIELDS = {
    "schema_version", "version", "created_at", "training_data_cutoff",
    "parent_sha256", "contract_sha256", "artifact_sha256", "sha256",
}
_SEAL_FIELDS = {
    "schema_version", "split", "sealed_at", "contract_sha256",
    "incumbent_sha256", "candidate_sha256", "comparison_index", "rows", "sha256",
}


def artifact_sha256(content: bytes | str) -> str:
    """Hash supplied artifact contents, never a path or an external resource."""
    if isinstance(content, str):
        content = content.encode("utf-8")
    if not isinstance(content, bytes):
        raise ValueError("artifact content must be bytes or text")
    return hashlib.sha256(content).hexdigest()


def _digest(value: dict) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("objects must contain finite JSON values") from exc
    return artifact_sha256(encoded)


def _seal(value: dict) -> dict:
    result = copy.deepcopy(value)
    result["sha256"] = _digest(result)
    return result


def _exact(value: object, fields: set[str], name: str) -> None:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{name} must contain exactly {sorted(fields)}")


def _string(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonblank text")


def _hash(value: object, name: str) -> None:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _integer(value: object, low: int, high: int, name: str) -> None:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")


def _number(value: object, low: float, high: float, name: str) -> None:
    try:
        valid = (type(value) in (int, float) and math.isfinite(value)
                 and low <= value <= high)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be a finite number in [{low}, {high}]")


def _integrity(value: dict, fields: set[str], name: str) -> None:
    _exact(value, fields, name)
    _hash(value["sha256"], f"{name}.sha256")
    if value["sha256"] != _digest({key: item for key, item in value.items() if key != "sha256"}):
        raise ValueError(f"{name} digest does not match its contents")
    if value["schema_version"] != "0.1":
        raise ValueError(f"{name} schema_version must be '0.1'")


def create_evaluation_contract(
    *, frozen_at: str, dev_start: str, dev_end: str, questions: list[dict],
    min_clusters: int = 30, min_improvement: float = 0.0,
    confidence: float = 0.95, max_comparisons: int = 1,
    bootstrap_samples: int = 2000, seed: int = 0,
) -> dict:
    """Commit a complete future dev cohort and paired cluster-Brier procedure.

    ``questions`` contains only _QUESTION_FIELDS; see SELF_IMPROVEMENT.md.
    Every registered question must later have a valid pair. Pending, void, failed,
    and missing runs cannot be silently excluded; they block this contract.
    """
    result = _seal({
        "schema_version": "0.1", "split": "dev", "metric": "normalized_brier",
        "aggregation": "equal_event_cluster", "frozen_at": frozen_at,
        "dev_start": dev_start, "dev_end": dev_end, "questions": questions,
        "min_clusters": min_clusters, "min_improvement": min_improvement,
        "confidence": confidence, "max_comparisons": max_comparisons,
        "bootstrap_samples": bootstrap_samples, "seed": seed,
    })
    validate_evaluation_contract(result)
    return result


def validate_evaluation_contract(contract: dict) -> None:
    _integrity(contract, _CONTRACT_FIELDS, "contract")
    if (contract["split"], contract["metric"], contract["aggregation"]) != (
        "dev", "normalized_brier", "equal_event_cluster"
    ):
        raise ValueError("only dev normalized-Brier event-cluster evaluation is supported")
    frozen, start, end = (parse_timestamp(contract[key]) for key in
                          ("frozen_at", "dev_start", "dev_end"))
    if not frozen < start < end:
        raise ValueError("contract requires frozen_at < dev_start < dev_end")
    _integer(contract["min_clusters"], 20, 100000, "min_clusters")
    _number(contract["min_improvement"], 0.0, 1.0, "min_improvement")
    _number(contract["confidence"], 0.90, 0.999, "confidence")
    _integer(contract["max_comparisons"], 1, 20, "max_comparisons")
    _integer(contract["bootstrap_samples"], 2000, 100000, "bootstrap_samples")
    _integer(contract["seed"], 0, 2**63 - 1, "seed")
    tail_draws = (contract["bootstrap_samples"] * (1 - contract["confidence"])
                  / (2 * contract["max_comparisons"]))
    if tail_draws < 10 - 1e-9:
        raise ValueError("bootstrap_samples must provide at least 10 draws per adjusted tail")
    questions = contract["questions"]
    if not isinstance(questions, list) or not questions:
        raise ValueError("questions must be a nonempty list")
    seen, event_clusters = set(), {}
    for question in questions:
        _exact(question, _QUESTION_FIELDS, "contract question")
        for key in ("question_id", "event_id", "cluster_id"):
            _string(question[key], key)
        if question["question_id"] in seen:
            raise ValueError("duplicate contract question_id")
        seen.add(question["question_id"])
        event, cluster = question["event_id"], question["cluster_id"]
        if event in event_clusters and event_clusters[event] != cluster:
            raise ValueError("all questions for an event must share one cluster")
        event_clusters[event] = cluster
        deadline, outcome, resolve = (parse_timestamp(question[key]) for key in
                                      ("forecast_deadline", "outcome_not_before", "resolve_after"))
        if not start < deadline < outcome <= resolve <= end:
            raise ValueError("question chronology must fit entirely inside the dev window")


def create_candidate_manifest(
    *, artifacts: dict[str, bytes | str], created_at: str,
    training_data_cutoff: str, contract: dict, parent: dict | None = None,
) -> dict:
    """Register supplied prompt/skill/harness/config artifacts without executing them.

    Include model identity, sampling and resource budgets in a config artifact.
    The training cutoff covers every task-specific example or reflection used.
    The base model's historical pretraining cutoff is a separate config detail.
    """
    validate_evaluation_contract(contract)
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("artifacts must be a nonempty mapping of names to contents")
    hashes = {}
    for name, content in artifacts.items():
        _string(name, "artifact name")
        hashes[name] = artifact_sha256(content)
    if parent is not None:
        validate_candidate_manifest(parent)
        if parse_timestamp(parent["created_at"]) >= parse_timestamp(created_at):
            raise ValueError("child must be created after its parent")
        if parse_timestamp(parent["training_data_cutoff"]) > parse_timestamp(training_data_cutoff):
            raise ValueError("child training_data_cutoff cannot precede its parent's cutoff")
    result = _seal({
        "schema_version": "0.1", "version": 1 if parent is None else parent["version"] + 1,
        "created_at": created_at, "training_data_cutoff": training_data_cutoff,
        "parent_sha256": None if parent is None else parent["sha256"],
        "contract_sha256": contract["sha256"], "artifact_sha256": hashes,
    })
    validate_candidate_manifest(result)
    if parse_timestamp(created_at) >= parse_timestamp(contract["dev_start"]):
        raise ValueError("candidate must be committed before dev_start")
    return result


def validate_candidate_manifest(manifest: dict) -> None:
    _integrity(manifest, _MANIFEST_FIELDS, "manifest")
    _integer(manifest["version"], 1, 1000000, "version")
    _hash(manifest["contract_sha256"], "contract_sha256")
    if manifest["version"] == 1:
        if manifest["parent_sha256"] is not None:
            raise ValueError("version 1 has no parent")
    else:
        _hash(manifest["parent_sha256"], "parent_sha256")
    if parse_timestamp(manifest["training_data_cutoff"]) > parse_timestamp(manifest["created_at"]):
        raise ValueError("training_data_cutoff cannot follow candidate creation")
    hashes = manifest["artifact_sha256"]
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("artifact_sha256 must be a nonempty mapping")
    for name, digest in hashes.items():
        _string(name, "artifact name")
        _hash(digest, "artifact digest")


def seal_dev_outcomes(
    *, contract: dict, incumbent: dict, candidate: dict, rows: list[dict],
    sealed_at: str, comparison_index: int = 1,
) -> dict:
    """Package an entire trusted-exporter cohort after its development window ends.

    Scores are paired normalized Brier values supplied by a trusted scorer. This
    is an integrity envelope, not a signature or proof of honest timestamps.
    """
    validate_evaluation_contract(contract)
    validate_candidate_manifest(incumbent)
    validate_candidate_manifest(candidate)
    result = _seal({
        "schema_version": "0.1", "split": "dev", "sealed_at": sealed_at,
        "contract_sha256": contract.get("sha256"),
        "incumbent_sha256": incumbent.get("sha256"),
        "candidate_sha256": candidate.get("sha256"),
        "comparison_index": comparison_index, "rows": rows,
    })
    _validate_sealed_outcomes(contract, incumbent, candidate, result)
    return result


def _validate_sealed_outcomes(contract: dict, incumbent: dict, candidate: dict, sealed: dict) -> None:
    validate_evaluation_contract(contract)
    for manifest in (incumbent, candidate):
        validate_candidate_manifest(manifest)
        if manifest["contract_sha256"] != contract["sha256"]:
            raise ValueError("candidate and incumbent must use the frozen evaluation contract")
        if parse_timestamp(manifest["created_at"]) >= parse_timestamp(contract["dev_start"]):
            raise ValueError("candidate and incumbent must precede dev_start")
    if (candidate["parent_sha256"] != incumbent["sha256"]
            or candidate["version"] != incumbent["version"] + 1
            or parse_timestamp(candidate["created_at"]) <= parse_timestamp(incumbent["created_at"])
            or parse_timestamp(candidate["training_data_cutoff"])
            < parse_timestamp(incumbent["training_data_cutoff"])):
        raise ValueError("candidate must be a chronological direct child of incumbent")
    _integrity(sealed, _SEAL_FIELDS, "sealed outcomes")
    if (sealed["split"] != "dev" or sealed["contract_sha256"] != contract["sha256"]
            or sealed["incumbent_sha256"] != incumbent["sha256"]
            or sealed["candidate_sha256"] != candidate["sha256"]):
        raise ValueError("sealed development outcomes do not match this comparison")
    _integer(sealed["comparison_index"], 1, contract["max_comparisons"], "comparison_index")
    sealed_time = parse_timestamp(sealed["sealed_at"])
    if sealed_time < parse_timestamp(contract["dev_end"]):
        raise ValueError("outcomes can only be sealed after dev_end")
    rows = sealed["rows"]
    if not isinstance(rows, list):
        raise ValueError("rows must be a list")
    questions = {item["question_id"]: item for item in contract["questions"]}
    seen = set()
    for row in rows:
        _exact(row, _ROW_FIELDS, "outcome row")
        _string(row["question_id"], "question_id")
        question_id = row["question_id"]
        if question_id not in questions or question_id in seen:
            raise ValueError("outcome rows contain an unknown or duplicate question")
        seen.add(question_id)
        question = questions[question_id]
        deadline = parse_timestamp(question["forecast_deadline"])
        for name, manifest in (("incumbent", incumbent), ("candidate", candidate)):
            submitted = parse_timestamp(row[f"{name}_submitted_at"])
            if not (parse_timestamp(manifest["created_at"]) <= submitted
                    and parse_timestamp(contract["dev_start"]) <= submitted < deadline):
                raise ValueError("submissions must follow manifest creation, occur in dev, and precede the deadline")
            _number(row[f"{name}_brier"], 0.0, 1.0, f"{name}_brier")
        if not parse_timestamp(question["resolve_after"]) <= parse_timestamp(row["resolved_at"]) <= sealed_time:
            raise ValueError("resolution must follow resolve_after and precede sealing")
    if seen != set(questions):
        raise ValueError("every precommitted question requires a resolved paired outcome")


def evaluate_promotion(
    *, contract: dict, incumbent: dict, candidate: dict, sealed_outcomes: dict,
    expected_contract_sha256: str,
) -> dict:
    """Return eligibility only; never install, mutate, or automatically promote.

    The expected digest must come from the evaluator's earlier commitment, not
    from the submitted candidate. The caller must enforce unique comparison
    slots and prospective, disjoint development windows across adaptive rounds.
    """
    _hash(expected_contract_sha256, "expected_contract_sha256")
    _validate_sealed_outcomes(contract, incumbent, candidate, sealed_outcomes)
    if contract["sha256"] != expected_contract_sha256:
        raise ValueError("evaluation contract differs from the trusted commitment")
    questions = {row["question_id"]: row for row in contract["questions"]}
    events: dict[str, list[float]] = {}
    event_clusters = {}
    for row in sorted(sealed_outcomes["rows"], key=lambda item: item["question_id"]):
        question = questions[row["question_id"]]
        event = question["event_id"]
        events.setdefault(event, []).append(row["incumbent_brier"] - row["candidate_brier"])
        event_clusters[event] = question["cluster_id"]
    clusters: dict[str, list[float]] = {}
    for event in sorted(events):
        clusters.setdefault(event_clusters[event], []).append(fmean(events[event]))
    deltas = [fmean(clusters[cluster]) for cluster in sorted(clusters)]
    result = {
        "schema_version": "0.1", "contract_sha256": contract["sha256"],
        "incumbent_sha256": incumbent["sha256"], "candidate_sha256": candidate["sha256"],
        "sealed_outcomes_sha256": sealed_outcomes["sha256"],
        "comparison_index": sealed_outcomes["comparison_index"],
        "question_count": len(questions), "event_count": len(events),
        "cluster_count": len(deltas), "mean_brier_improvement": fmean(deltas),
        "confidence_interval": None, "eligible": False, "reason": "insufficient_clusters",
    }
    if len(deltas) < contract["min_clusters"]:
        return result
    rng = random.Random(contract["seed"])
    samples = sorted(fmean(deltas[rng.randrange(len(deltas))] for _ in deltas)
                     for _ in range(contract["bootstrap_samples"]))
    tail = (1 - contract["confidence"]) / (2 * contract["max_comparisons"])

    def quantile(fraction: float) -> float:
        position = fraction * (len(samples) - 1)
        index = math.floor(position)
        remainder = position - index
        return samples[index] * (1 - remainder) + samples[min(index + 1, len(samples) - 1)] * remainder

    lower, upper = quantile(tail), quantile(1 - tail)
    result["confidence_interval"] = [lower, upper]
    result["eligible"] = lower > contract["min_improvement"]
    result["reason"] = "passes_paired_cluster_guard" if result["eligible"] else "improvement_not_established"
    return result
