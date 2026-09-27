"""Validation for version 0.1 forecast questions and probability distributions.

The options describe mutually exclusive, collectively exhaustive outcomes. The
validator checks their structure; a human curator must check that semantic claim.
All validation failures raise ``ValueError``. Validation never normalizes a
submitted probability distribution or changes a question in place.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from numbers import Real
from typing import Iterable
from urllib.parse import urlsplit


PROBABILITY_SUM_TOLERANCE = 1e-6


def public_question(question: dict) -> dict:
    """Allowlisted model-facing fields; private metadata never enters prompts."""
    import copy
    fields = ("schema_version", "question_id", "kind", "prompt", "options", "domain", "issued_at",
              "forecast_deadline", "outcome_not_before", "resolve_after", "resolution", "is_fixture")
    return {key: copy.deepcopy(question[key]) for key in fields if key in question}


def _nonblank_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonblank string")
    return value


def parse_timestamp(value: str) -> datetime:
    """Parse an ISO 8601 timestamp with an explicit offset, returning UTC.

    A trailing ``Z`` is supported on Python 3.10 as well as newer Python versions.
    Naive timestamps are rejected rather than interpreted in a machine's timezone.
    """
    _nonblank_string(value, "timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("timestamp must include an explicit timezone offset")
        return parsed.astimezone(timezone.utc)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"invalid timezone-aware ISO 8601 timestamp: {value!r}") from exc


def option_ids(question: dict) -> list[str]:
    """Return the ordered option IDs after checking option structure and uniqueness."""
    if not isinstance(question, dict):
        raise ValueError("question must be an object")
    options = question.get("options")
    if not isinstance(options, list) or len(options) < 2:
        raise ValueError("options must be a list containing at least two options")
    ids: list[str] = []
    for index, option in enumerate(options):
        if not isinstance(option, dict):
            raise ValueError(f"options[{index}] must be an object")
        ids.append(_nonblank_string(option.get("id"), f"options[{index}].id"))
        _nonblank_string(option.get("text"), f"options[{index}].text")
    if len(ids) != len(set(ids)):
        raise ValueError("option IDs must be unique")
    return ids


def validate_question(question: dict) -> None:
    """Validate a binary or categorical future-event question against schema 0.1."""
    if not isinstance(question, dict):
        raise ValueError("question must be an object")
    if question.get("schema_version") != "0.1":
        raise ValueError("schema_version must be '0.1'")
    for field in ("question_id", "event_id", "cluster_id", "prompt", "domain"):
        _nonblank_string(question.get(field), field)
    if question.get("split") not in ("train", "dev", "test"):
        raise ValueError("split must be train, dev, or test")
    kind = question.get("kind")
    if kind not in ("binary", "categorical"):
        raise ValueError("kind must be binary or categorical")
    ids = option_ids(question)
    if kind == "binary" and len(ids) != 2:
        raise ValueError("binary questions must have exactly two options")
    if kind == "categorical" and len(ids) < 3:
        raise ValueError("categorical questions must have at least three options")

    timestamps = {}
    for field in ("issued_at", "forecast_deadline", "outcome_not_before", "resolve_after"):
        try:
            timestamps[field] = parse_timestamp(question.get(field))
        except ValueError as exc:
            raise ValueError(f"{field}: {exc}") from exc
    if not (
        timestamps["issued_at"] < timestamps["forecast_deadline"]
        < timestamps["outcome_not_before"] <= timestamps["resolve_after"]
    ):
        raise ValueError(
            "timestamps must satisfy issued_at < forecast_deadline "
            "< outcome_not_before <= resolve_after"
        )

    resolution = question.get("resolution")
    if not isinstance(resolution, dict):
        raise ValueError("resolution must be an object")
    _nonblank_string(resolution.get("criteria"), "resolution.criteria")
    if resolution.get("on_ambiguous") != "void":
        raise ValueError("resolution.on_ambiguous must be 'void'")
    urls = resolution.get("source_urls")
    if not isinstance(urls, list) or not urls:
        raise ValueError("resolution.source_urls must be a nonempty list")
    for index, url in enumerate(urls):
        _nonblank_string(url, f"resolution.source_urls[{index}]")
        try:
            parsed_url = urlsplit(url)
            valid = (
                parsed_url.scheme in ("http", "https")
                and parsed_url.hostname is not None
                and not any(character.isspace() for character in url)
            )
            # Accessing .port also rejects malformed nonnumeric or out-of-range ports.
            parsed_url.port
        except ValueError as exc:
            raise ValueError(f"invalid resolution source URL: {url!r}") from exc
        if not valid:
            raise ValueError(f"resolution source URL must be an absolute HTTP(S) URL: {url!r}")
    if "metadata" in question and not isinstance(question["metadata"], dict):
        raise ValueError("metadata must be an object when supplied")
    if "is_fixture" in question and not isinstance(question["is_fixture"], bool):
        raise ValueError("is_fixture must be a boolean when supplied")


def validate_probabilities(probabilities: dict, option_ids: Iterable[str]) -> dict[str, float]:
    """Check full probability coverage and return an independent float-valued copy.

    The absolute sum tolerance is 1e-6. Accepted values are not renormalized.
    Booleans, nonnumeric values, infinities, and NaNs are invalid probabilities.
    """
    if not isinstance(probabilities, dict):
        raise ValueError("probabilities must be an object keyed by option ID")
    if isinstance(option_ids, (str, bytes)):
        raise ValueError("option_ids must be an iterable of option ID strings")
    try:
        ids = list(option_ids)
    except TypeError as exc:
        raise ValueError("option_ids must be an iterable of option ID strings") from exc
    for index, identifier in enumerate(ids):
        _nonblank_string(identifier, f"option_ids[{index}]")
    if len(ids) < 2 or len(ids) != len(set(ids)):
        raise ValueError("option_ids must contain at least two unique IDs")
    if set(probabilities) != set(ids):
        raise ValueError("probability keys must match option IDs exactly")
    result: dict[str, float] = {}
    for identifier in ids:
        value = probabilities[identifier]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"probability for {identifier!r} must be a real number, not a boolean")
        try:
            number = float(value)
        except (ValueError, OverflowError) as exc:
            raise ValueError(f"probability for {identifier!r} is not a finite number") from exc
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError(f"probability for {identifier!r} must be finite and in [0, 1]")
        result[identifier] = number
    total = math.fsum(result.values())
    if abs(total - 1.0) > PROBABILITY_SUM_TOLERANCE:
        raise ValueError(f"probabilities must sum to 1 within 1e-6; got {total!r}")
    return result
