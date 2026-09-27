"""Audited search/open gateway with replaceable, deliberately offline providers.

The market filter is a heuristic, not a clean-data or historical leakage proof.
Only a trusted host should create sessions, provide the clock, or read events.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Callable, Protocol
from urllib.parse import urlsplit

from .analyst import (AnalystNotebook, MARKET_TOOLS, PolymarketPublicProvider,
                      RESEARCH_TOOLS, TOOLKIT_VERSION, calculate, validate_arguments)


class ResearchProvider(Protocol):
    """Providers return text snapshots, never instructions to execute."""

    def search(self, query: str) -> list[dict]:
        """Return dictionaries with url, title, text and optional published_at."""
        ...

    def open(self, url: str) -> dict:
        """Return the same fields; url MUST be the final URL after redirects."""
        ...


MARKET_DOMAINS = frozenset({
    "polymarket.com", "kalshi.com", "metaculus.com", "manifold.markets",
    "predictit.org", "oddsportal.com",
})
_CONSENSUS_LABEL = (
    r"(?:polymarket|kalshi|metaculus|manifold|predictit|oddsportal|"
    r"(?:market|crowd|consensus|collective|community)[\s-]+"
    r"(?:probabilit(?:y|ies)|odds|forecast|prediction|estimate|chance)|"
    r"prediction[\s-]+markets?|market[\s-]+consensus|"
    r"(?:crowd|community)[\s-]+(?:assigns?|gives?|expects?)|"
    r"implied[\s-]+probability|"
    r"(?:市场|共识|群体|集体|社区|大众).{0,12}(?:概率|赔率|预测))"
)
_PROBABILITY = r"(?:\d+(?:\.\d+)?\s*(?:[%％]|percent\b)|(?<!\d)0\.\d+)"
_CONSENSUS_NUMBER = re.compile(
    rf"(?:{_CONSENSUS_LABEL}.{{0,100}}{_PROBABILITY}|"
    rf"{_PROBABILITY}.{{0,70}}{_CONSENSUS_LABEL}|"
    r"(?:market|betting|bookmaker)[\s-]+odds.{0,30}\d+(?:\.\d+)?)",
    re.IGNORECASE | re.DOTALL,
)


def _parse_time(value: str | datetime, name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an ISO 8601 timestamp") from exc
    if not isinstance(parsed, datetime) or parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _hash(value: dict) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _host(url: str) -> str:
    if not isinstance(url, str) or not url or any(ord(char) < 32 for char in url):
        raise ValueError("invalid_url")
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        if parsed.scheme not in {"http", "https"} or not hostname or parsed.username or parsed.password:
            raise ValueError("invalid_url")
        # Percent-escaped hosts and backslashes are ambiguous across URL parsers.
        if "%" in hostname or "\\" in url:
            raise ValueError("invalid_url")
        return hostname.rstrip(".").encode("idna").decode("ascii").lower()
    except (UnicodeError, ValueError) as exc:
        raise ValueError("invalid_url") from exc


def _market_host(hostname: str) -> bool:
    return any(hostname == domain or hostname.endswith("." + domain) for domain in MARKET_DOMAINS)


class ResearchSession:
    """One forecast's tool budget and immutable-by-copy observation snapshots.

    The injected clock is a trusted host dependency for tests, not a model input.
    Do not expose ``provider``, ``clock``, or ``events`` to an untrusted model.
    """

    def __init__(
        self,
        question: dict,
        provider: ResearchProvider,
        market_mode: str = "no_consensus",
        max_calls: int = 8,
        clock: Callable[[], datetime] | None = None,
        market_provider=None,
    ):
        if market_mode not in {"no_consensus", "market_aware"}:
            raise ValueError("market_mode must be no_consensus or market_aware")
        if isinstance(max_calls, bool) or not isinstance(max_calls, int) or max_calls < 1:
            raise ValueError("max_calls must be a positive integer")
        self.issued_at = _parse_time(question["issued_at"], "issued_at")
        self.forecast_deadline = _parse_time(question["forecast_deadline"], "forecast_deadline")
        if self.forecast_deadline <= self.issued_at:
            raise ValueError("forecast_deadline must be after issued_at")
        self.question = copy.deepcopy(question)
        self.provider = provider
        self.market_mode = market_mode
        self.max_calls = max_calls
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.calls_used = 0
        self.successful_searches = 0
        self.events: list[dict] = []
        self.sources: dict[str, dict] = {}
        self.notebook = AnalystNotebook(question)
        self.market_provider = market_provider or PolymarketPublicProvider()

    def _now(self) -> datetime:
        return _parse_time(self.clock(), "clock")

    def _record(self, kind: str, tool: str, payload: dict, at: datetime) -> None:
        event = {
            "type": kind,
            "tool": tool,
            "at": _iso(at),
            "observed_at": _iso(at),
            "payload": copy.deepcopy(payload),
            "loss_mask": 0 if kind == "observation" else 1,
            "toolkit_version": TOOLKIT_VERSION,
        }
        event["sha256"] = _hash(event)
        self.events.append(event)

    def _window_error(self, now: datetime) -> str | None:
        if now < self.issued_at:
            return "question_not_issued"
        if now >= self.forecast_deadline:
            return "forecast_deadline_reached"
        return None

    def _finish(self, tool: str, payload: dict, at: datetime) -> dict:
        payload = copy.deepcopy(payload)
        payload["observed_at"] = _iso(at)
        payload["market_mode"] = self.market_mode
        payload["calls_used"] = self.calls_used
        payload["calls_remaining"] = max(0, self.max_calls - self.calls_used)
        if payload.get("success") is True:
            snapshots = payload.get("results", [payload.get("result")])
            for snapshot in snapshots:
                if isinstance(snapshot, dict) and "sha256" in snapshot and "url" in snapshot:
                    self.sources[snapshot["sha256"]] = copy.deepcopy(snapshot)
        if tool == "search":
            payload.setdefault("results", [])
            payload.setdefault("filtered_count", 0)
            payload.setdefault("filter_reasons", {})
        self._record("observation", tool, payload, at)
        return payload

    def _blocked(self, tool: str, reason: str, at: datetime, **extra: object) -> dict:
        return self._finish(tool, {"success": False, "status": "blocked", "reason": reason, **extra}, at)

    def _begin(self, tool: str, argument: str) -> tuple[datetime, dict | None]:
        argument_name = "query" if tool == "search" else "url"
        # API arguments are strings, allowing every accepted action to be hashed.
        if not isinstance(argument, str):
            raise TypeError(f"{argument_name} must be a string")
        now, rejected = self._begin_payload(tool, {argument_name: argument})
        if rejected is not None:
            return now, rejected
        if not argument.strip():
            return now, self._blocked(tool, "empty_argument", now)
        return now, None

    def _begin_payload(self, tool, payload):
        now = self._now()
        self._record("action", tool, payload, now)
        window_error = self._window_error(now)
        if window_error:
            return now, self._blocked(tool, window_error, now)
        if self.calls_used >= self.max_calls:
            return now, self._blocked(tool, "call_budget_exhausted", now)
        self.calls_used += 1
        return now, None

    def dispatch(self, tool, arguments):
        """Execute an allowlisted capability under one shared budget and deadline."""
        if tool not in RESEARCH_TOOLS:
            raise ValueError("tool_not_available")
        # Keep legacy search/open wrappers while auditing malformed attempts too.
        if tool in {"search", "open"}:
            try:
                validate_arguments(tool, arguments, self.question, market_mode=self.market_mode)
            except (ValueError, TypeError, OverflowError):
                now, rejected = self._begin_payload(tool, arguments)
                return rejected or self._blocked(tool, "invalid_tool_arguments", now)
            return self.search(arguments["query"]) if tool == "search" else self.open(arguments["url"])
        now, rejected = self._begin_payload(tool, arguments)
        if rejected is not None:
            return rejected
        if tool in MARKET_TOOLS and self.market_mode != "market_aware":
            return self._blocked(tool, "market_tools_require_market_aware", now)
        try:
            validate_arguments(tool, arguments, self.question, market_mode=self.market_mode)
        except (ValueError, TypeError, OverflowError):
            return self._blocked(tool, "invalid_tool_arguments", now)
        try:
            if tool == "calculator":
                payload = {"value": calculate(arguments["expression"])}
            elif tool == "notebook":
                payload = {"note": self.notebook.note(sources=self.sources, **arguments)}
            elif tool == "draft":
                payload = {"draft": self.notebook.draft(sources=self.sources, **arguments)}
            else:
                remaining = max(0.001, (self.forecast_deadline - now).total_seconds())
                raw = (self.market_provider.search(arguments["query"], timeout=remaining)
                       if tool == "market_search" else [self.market_provider.snapshot(arguments["market_id"], timeout=remaining)])
                now = self._now()
                error = self._window_error(now)
                if error:
                    return self._blocked(tool, error, now)
                if not isinstance(raw, list):
                    raise ValueError("invalid_provider_response")
                results, reasons = [], Counter()
                for item in raw[:5]:
                    snapshot, reason = self._snapshot(item, now)
                    if reason:
                        reasons[reason] += 1
                    else:
                        # Structured market data is encoded in the hashed text;
                        # no unfiltered provider fields pass into observations.
                        results.append(snapshot)
                payload = {"results": results, "filtered_count": sum(reasons.values()),
                           "filter_reasons": dict(reasons), "consensus_exposure": bool(results),
                           "match_verified": False}
                return self._finish(tool, {"success": bool(results), "status": "ok" if results else "empty", **payload}, now)
        except ValueError as exc:
            now = self._now()
            return self._blocked(tool, self._window_error(now) or "invalid_tool_input", now)
        except Exception as exc:
            now = self._now()
            error = self._window_error(now)
            if error:
                return self._blocked(tool, error, now)
            return self._finish(tool, {"success": False, "status": "error", "reason": "provider_error",
                                      "error_type": type(exc).__name__}, now)
        now = self._now()
        error = self._window_error(now)
        if error:
            return self._blocked(tool, error, now)
        return self._finish(tool, {"success": True, "status": "ok", **payload}, now)

    def _snapshot(self, item: dict, now: datetime) -> tuple[dict | None, str | None]:
        if not isinstance(item, dict):
            return None, "invalid_result"
        url, title, text = item.get("url"), item.get("title"), item.get("text")
        if not isinstance(title, str) or not isinstance(text, str):
            return None, "invalid_result"
        try:
            hostname = _host(url)
        except ValueError:
            return None, "invalid_url"
        is_market = _market_host(hostname)
        consensus_text = bool(_CONSENSUS_NUMBER.search(title + "\n" + text))
        if self.market_mode == "no_consensus":
            if is_market:
                return None, "market_source"
            if consensus_text:
                return None, "consensus_text"
        published_at = item.get("published_at")
        publication_status = "unknown"
        if published_at is not None:
            try:
                published_at = _parse_time(published_at, "published_at")
            except ValueError:
                return None, "invalid_published_at"
            if published_at > now:
                return None, "future_published_at"
            published_at = _iso(published_at)
            publication_status = "known"
        source_updated_at = item.get("source_updated_at")
        if source_updated_at is not None:
            try:
                source_updated_at = _parse_time(source_updated_at, "source_updated_at")
            except ValueError:
                return None, "invalid_source_updated_at"
            if source_updated_at > now:
                return None, "future_source_updated_at"
            source_updated_at = _iso(source_updated_at)
        retrieved_at = item.get("retrieved_at")
        if retrieved_at is not None:
            try:
                retrieved_at = _parse_time(retrieved_at, "retrieved_at")
            except ValueError:
                return None, "invalid_retrieved_at"
            if retrieved_at > now:
                return None, "future_retrieved_at"
            retrieved_at = _iso(retrieved_at)
        result = {
            "url": url,
            "title": title,
            "text": text,
            "published_at": published_at,
            "published_at_status": publication_status,
            "source_updated_at": source_updated_at,
            "retrieved_at": retrieved_at,
            "observed_at": _iso(now),
            "is_market_source": is_market,
            "contains_consensus_text": consensus_text,
            "consensus_exposure": is_market or consensus_text,
        }
        result["sha256"] = _hash(result)
        return result, None

    def search(self, query: str) -> dict:
        """Return permitted snippets, preserving their full tool-visible content."""
        _, rejected = self._begin("search", query)
        if rejected is not None:
            return rejected
        try:
            raw_results = self.provider.search(query)
        except Exception as exc:
            now = self._now()
            reason = self._window_error(now)
            if reason:
                return self._blocked("search", reason, now)
            return self._finish("search", {
                "success": False, "status": "error", "reason": "provider_error",
                "error_type": type(exc).__name__,
            }, now)
        now = self._now()
        reason = self._window_error(now)
        if reason:
            return self._blocked("search", reason, now,
                                 discarded_result_count=len(raw_results) if isinstance(raw_results, list) else None)
        if not isinstance(raw_results, list):
            return self._finish("search", {
                "success": False, "status": "error", "reason": "invalid_provider_response",
            }, now)
        allowed, reasons = [], Counter()
        for raw in raw_results:
            snapshot, reason = self._snapshot(raw, now)
            if reason:
                reasons[reason] += 1
            else:
                allowed.append(snapshot)
        success = bool(allowed)
        if success:
            self.successful_searches += 1
        return self._finish("search", {
            "success": success,
            "status": "ok" if success else "empty",
            "results": allowed,
            "filtered_count": sum(reasons.values()),
            "filter_reasons": dict(reasons),
            "consensus_exposure": any(result["consensus_exposure"] for result in allowed),
        }, now)

    def open(self, url: str) -> dict:
        """Open one URL and validate both requested and provider-returned URLs."""
        now, rejected = self._begin("open", url)
        if rejected is not None:
            return rejected
        try:
            hostname = _host(url)
        except ValueError:
            return self._blocked("open", "invalid_url", now)
        if self.market_mode == "no_consensus" and _market_host(hostname):
            return self._blocked("open", "market_source", now)
        try:
            raw = self.provider.open(url)
        except Exception as exc:
            now = self._now()
            reason = self._window_error(now)
            if reason:
                return self._blocked("open", reason, now)
            return self._finish("open", {
                "success": False, "status": "error", "reason": "provider_error",
                "error_type": type(exc).__name__,
            }, now)
        now = self._now()
        reason = self._window_error(now)
        if reason:
            return self._blocked("open", reason, now, discarded_result_count=1)
        snapshot, reason = self._snapshot(raw, now)
        if reason:
            return self._blocked("open", reason, now)
        return self._finish("open", {
            "success": True, "status": "ok", "result": snapshot,
            "consensus_exposure": snapshot["consensus_exposure"],
        }, now)
