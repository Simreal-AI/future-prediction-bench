"""Private baseline generation. No baseline probabilities enter agent observations."""

from __future__ import annotations

import json
import os
from datetime import timedelta
from pathlib import Path

from .http import HttpClient, strict_json_loads
from .providers import ChatCompletionModel
from .schema import option_ids, parse_timestamp, public_question, validate_probabilities
from .store import canonical, digest


BASELINE_PROMPT = """You are the fixed internal forecasting baseline. Given only a future-event question and its resolution rules, estimate the probability of EVERY listed option using your existing knowledge. Do not search or access outcomes. The options are mutually exclusive and collectively exhaustive. Return exactly a JSON object with a probabilities object keyed by every option ID. Values must be finite numbers in [0,1] and sum to 1. Do not include an explanation or a selected answer. You do not see any evaluated agent's prediction."""


def internal_model_from_env():
    """Separate configuration avoids silently using the evaluated model as baseline."""
    return ChatCompletionModel(os.environ.get("FPB_BASELINE_BASE_URL", ""),
                               os.environ.get("FPB_BASELINE_API_KEY", ""),
                               os.environ.get("FPB_BASELINE_MODEL_NAME", ""), temperature=0.0)


def load_baseline_config(path=None):
    if path is None:
        return {"market_mappings": {}}
    value = strict_json_loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("market_mappings", {}), dict):
        raise ValueError("Baseline config must contain an object of market_mappings")
    return value


def _market_baseline(question, mapping, client, now):
    if mapping.get("provider") != "polymarket" or mapping.get("reviewed") is not True:
        raise ValueError("A reviewed Polymarket mapping is required")
    if mapping.get("question_sha256") != digest(public_question(question)):
        raise ValueError("Mapping does not match this frozen question")
    if not isinstance(mapping.get("resolution_equivalence_note"), str) or not mapping["resolution_equivalence_note"].strip():
        raise ValueError("Document equivalence of options, time window, and resolution rules")
    market_id = mapping.get("market_id")
    if not isinstance(market_id, str) or not market_id.isascii() or not market_id.isdigit():
        raise ValueError("market_id must be a numeric Gamma market ID")
    url = "https://gamma-api.polymarket.com/markets/" + market_id
    market = client.get_json(url)
    if not isinstance(market, dict) or str(market.get("id")) != market_id:
        raise ValueError("Unexpected market identity")
    if (market.get("question") != mapping.get("market_question") or not mapping.get("market_question")
            or not isinstance(market.get("description"), str)
            or digest(market["description"]) != mapping.get("market_description_sha256")
            or market.get("endDate") != mapping.get("market_end_date") or not mapping.get("market_end_date")):
        raise ValueError("Market wording, rules, or date changed since mapping review")
    if market.get("active") is not True or market.get("closed") is not False or market.get("acceptingOrders") is not True:
        raise ValueError("Market is not accepting active prices")
    maximum_age = mapping.get("max_age_seconds", 3600)
    if isinstance(maximum_age, bool) or not isinstance(maximum_age, int) or not 1 <= maximum_age <= 86400:
        raise ValueError("max_age_seconds must be an integer in [1,86400]")
    updated = parse_timestamp(market["updatedAt"])
    observed = parse_timestamp(client.last_snapshot["observed_at"])
    if not timedelta(0) <= observed - updated <= timedelta(seconds=maximum_age):
        raise ValueError("Market metadata timestamp is stale or in the future")
    if observed >= parse_timestamp(question["forecast_deadline"]):
        raise ValueError("Market snapshot arrived after the forecast deadline")
    labels = strict_json_loads(market["outcomes"]) if isinstance(market.get("outcomes"), str) else market.get("outcomes")
    prices = strict_json_loads(market["outcomePrices"]) if isinstance(market.get("outcomePrices"), str) else market.get("outcomePrices")
    if (not isinstance(labels, list) or not isinstance(prices, list) or len(labels) != len(prices)
            or not all(isinstance(label, str) for label in labels) or len(set(labels)) != len(labels)):
        raise ValueError("Invalid market outcomes/prices")
    mapping_options = mapping.get("outcome_map", {})
    if set(mapping_options) != set(labels) or set(mapping_options.values()) != set(option_ids(question)) or len(mapping_options) != len(option_ids(question)):
        raise ValueError("Market options must map one-to-one to every question option")
    if any(isinstance(price, bool) for price in prices):
        raise ValueError("Invalid price type")
    probabilities = validate_probabilities({mapping_options[label]: float(price) for label, price in zip(labels, prices)}, option_ids(question))
    return {"probabilities": probabilities, "identity": "polymarket:" + market_id,
            "observed_at": observed.isoformat(), "metadata": {"mapping": mapping, "snapshot": client.last_snapshot,
            "quote_convention": "Gamma outcomePrices, no renormalization; updatedAt is market metadata freshness, not an independent trade-time guarantee"}}


def generate_internal(question, model, *, max_tokens=1024):
    messages = [{"role": "system", "content": BASELINE_PROMPT},
                {"role": "user", "content": canonical(public_question(question))}]
    response = model.complete(messages, [], max_tokens=max_tokens)
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("Internal model must return one response")
    message = choices[0]["message"]
    if message.get("tool_calls") or not isinstance(message.get("content"), str):
        raise ValueError("Internal baseline must return a probability JSON object")
    payload = strict_json_loads(message["content"])
    if not isinstance(payload, dict) or set(payload) != {"probabilities"}:
        raise ValueError("Internal baseline output must contain only probabilities")
    probabilities = validate_probabilities(payload["probabilities"], option_ids(question))
    configuration = {"model": model.public_config(), "prompt_sha256": digest(BASELINE_PROMPT),
                     "max_tokens": max_tokens, "research": "none"}
    return {"probabilities": probabilities, "identity": "internal-" + digest(configuration),
            "metadata": {"config": configuration, "request": messages, "response": message,
                         "usage": response.get("usage", {})}}


def seal_baselines(store, question_ids, *, config=None, client=None, internal_model=None):
    """Try a reviewed market match first, then actual internal inference.

    Reports expose availability and provenance kind, never the probability vector.
    No fallback is fabricated when credentials, matching, or inference fail.
    """
    config = config or {"market_mappings": {}}
    client = client or HttpClient("runs/private-baseline-snapshots")
    reports = []
    for qid in question_ids:
        report = {"question_id": qid, "status": "missing", "market_match": "not_configured"}
        if store.private_baseline(qid) is not None:
            report["status"] = "already_sealed"
            reports.append(report)
            continue
        question = store.question(qid)
        if not parse_timestamp(question["issued_at"]) <= store.now() < parse_timestamp(question["forecast_deadline"]):
            report["reason"] = "outside_forecast_window"
            reports.append(report)
            continue
        if store.db.execute("SELECT 1 FROM episodes WHERE question_id=?", (qid,)).fetchone():
            report["reason"] = "forecast_already_started"
            reports.append(report)
            continue
        mapping = config.get("market_mappings", {}).get(qid)
        if mapping is not None:
            try:
                value = _market_baseline(question, mapping, client, store.now())
                store.seal_baseline(qid, kind="market", **value)
                report.update(status="sealed", kind="market", market_match="validated")
                reports.append(report)
                continue
            except Exception as exc:
                report["market_match"] = "unavailable_or_invalid"
                report["market_error"] = type(exc).__name__
        if internal_model is None:
            report["reason"] = "internal_model_not_configured"
        else:
            try:
                value = generate_internal(question, internal_model)
                store.seal_baseline(qid, kind="internal_model", **value)
                report.update(status="sealed", kind="internal_model")
            except Exception as exc:
                report["reason"] = "internal_baseline_failed"
                report["error"] = type(exc).__name__
        reports.append(report)
    return reports
