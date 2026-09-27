"""Immutable, point-in-time research evidence and a closed replay provider.

An evidence pack contains only successful search/open observations emitted by
``ResearchSession``. Its SHA-256 digest detects accidental changes; it is not
an attestation that a source or the recorder was honest. Historical training
must supply a trusted virtual clock to ``ResearchSession`` and keep outcome
data outside the agent's tool surface.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .research import _host, _iso, _parse_time


SCHEMA_VERSION = "evidence_pack_v1"
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SNAPSHOT_FIELDS = frozenset({
    "url", "title", "text", "published_at", "published_at_status",
    "source_updated_at", "retrieved_at", "observed_at", "is_market_source",
    "contains_consensus_text", "consensus_exposure", "sha256",
})
_BODY_FIELDS = frozenset({
    "schema_version", "forecast_cutoff", "created_as_of", "searches", "pages",
})


class EvidencePackError(ValueError):
    """A recorded observation or serialized pack violates replay invariants."""


class EvidenceNotRecorded(LookupError):
    """The exact requested evidence was not present at the replay cutoff."""


def _digest(value: dict) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _timestamp(value, name: str) -> datetime:
    try:
        return _parse_time(value, name)
    except (TypeError, ValueError) as exc:
        raise EvidencePackError(f"invalid_{name}") from exc


def _valid_hash(value) -> bool:
    return isinstance(value, str) and _HEX_SHA256.fullmatch(value) is not None


def _check_event(event: dict, cutoff: datetime) -> tuple[str, datetime]:
    if not isinstance(event, dict) or not _valid_hash(event.get("sha256")):
        raise EvidencePackError("invalid_recorded_event")
    try:
        actual = _digest({key: val for key, val in event.items() if key != "sha256"})
    except (TypeError, ValueError, OverflowError) as exc:
        raise EvidencePackError("invalid_recorded_event") from exc
    if event["sha256"] != actual:
        raise EvidencePackError("recorded_event_hash_mismatch")
    if (event.get("type") not in {"action", "observation"}
            or event.get("tool") not in {"search", "open", "calculator", "notebook", "draft",
                                          "market_search", "market_snapshot"}
            or not isinstance(event.get("payload"), dict)):
        raise EvidencePackError("invalid_recorded_event")
    observed = _timestamp(event.get("observed_at"), "observed_at")
    at = _timestamp(event.get("at"), "event_at")
    if at != observed or observed >= cutoff:
        raise EvidencePackError("post_cutoff_or_inconsistent_event")
    expected_mask = 1 if event["type"] == "action" else 0
    if event.get("loss_mask") != expected_mask:
        raise EvidencePackError("invalid_recorded_event")
    return event["type"], observed


def _check_snapshot(snapshot: dict, observation_at: datetime, cutoff: datetime) -> None:
    required = {"url", "title", "text", "published_at", "published_at_status",
                "source_updated_at", "observed_at", "is_market_source",
                "contains_consensus_text", "consensus_exposure", "sha256"}
    if (not isinstance(snapshot, dict) or not _valid_hash(snapshot.get("sha256"))
            or not set(snapshot).issubset(_SNAPSHOT_FIELDS)
            or not required.issubset(snapshot)):
        raise EvidencePackError("invalid_recorded_snapshot")
    try:
        actual = _digest({key: val for key, val in snapshot.items() if key != "sha256"})
        _host(snapshot["url"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise EvidencePackError("invalid_recorded_snapshot") from exc
    if snapshot["sha256"] != actual:
        raise EvidencePackError("recorded_snapshot_hash_mismatch")
    if not isinstance(snapshot["title"], str) or not isinstance(snapshot["text"], str):
        raise EvidencePackError("invalid_recorded_snapshot")
    if any(type(snapshot[key]) is not bool for key in
           ("is_market_source", "contains_consensus_text", "consensus_exposure")):
        raise EvidencePackError("invalid_recorded_snapshot")
    if snapshot["consensus_exposure"] != (snapshot["is_market_source"] or snapshot["contains_consensus_text"]):
        raise EvidencePackError("invalid_recorded_snapshot")
    source_observed = _timestamp(snapshot["observed_at"], "snapshot_observed_at")
    if source_observed != observation_at or source_observed >= cutoff:
        raise EvidencePackError("post_cutoff_or_inconsistent_snapshot")
    for key in ("published_at", "source_updated_at", "retrieved_at"):
        value = snapshot.get(key)
        if value is not None:
            stamped = _timestamp(value, key)
            if stamped > source_observed or stamped >= cutoff:
                raise EvidencePackError(f"future_{key}")
    publication_status = "known" if snapshot["published_at"] is not None else "unknown"
    if snapshot.get("published_at_status") != publication_status:
        raise EvidencePackError("invalid_publication_status")


def _validate_body(body: dict) -> None:
    if not isinstance(body, dict) or set(body) != _BODY_FIELDS or body["schema_version"] != SCHEMA_VERSION:
        raise EvidencePackError("invalid_evidence_pack_schema")
    cutoff = _timestamp(body["forecast_cutoff"], "forecast_cutoff")
    if not isinstance(body["searches"], dict) or not isinstance(body["pages"], dict):
        raise EvidencePackError("invalid_evidence_pack_schema")
    newest = None
    count = 0
    for kind, records in (("search", body["searches"]), ("open", body["pages"])):
        for key, history in records.items():
            if not isinstance(key, str) or not key.strip() or not isinstance(history, list) or not history:
                raise EvidencePackError("invalid_evidence_pack_schema")
            if kind == "open":
                try:
                    _host(key)
                except ValueError as exc:
                    raise EvidencePackError("invalid_evidence_pack_schema") from exc
            previous = None
            for record in history:
                expected = {"observed_at", "event_sha256", "market_mode", "snapshots" if kind == "search" else "snapshot"}
                if not isinstance(record, dict) or set(record) != expected or not _valid_hash(record["event_sha256"]):
                    raise EvidencePackError("invalid_evidence_pack_schema")
                observed = _timestamp(record["observed_at"], "record_observed_at")
                if observed >= cutoff or (previous is not None and observed < previous):
                    raise EvidencePackError("post_cutoff_or_unordered_record")
                if (not isinstance(record["market_mode"], str)
                        or record["market_mode"] not in {"market_aware", "no_consensus"}):
                    raise EvidencePackError("invalid_recorded_market_mode")
                snapshots = record["snapshots"] if kind == "search" else [record["snapshot"]]
                if not isinstance(snapshots, list) or not snapshots:
                    raise EvidencePackError("invalid_recorded_snapshot")
                for snapshot in snapshots:
                    _check_snapshot(snapshot, observed, cutoff)
                previous = observed
                newest = max(newest, observed) if newest is not None else observed
                count += 1
    if count == 0 or _timestamp(body["created_as_of"], "created_as_of") != newest:
        raise EvidencePackError("invalid_evidence_pack_as_of")


@dataclass(frozen=True, slots=True)
class EvidencePack:
    """A frozen manifest; consumers only receive detached copies of its data."""

    _body_json: str
    sha256: str

    def __post_init__(self) -> None:
        try:
            body = json.loads(self._body_json)
            _validate_body(body)
            canonical = json.dumps(body, ensure_ascii=False, sort_keys=True,
                                   separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError, OverflowError) as exc:
            raise EvidencePackError("invalid_evidence_pack_schema") from exc
        if self._body_json != canonical or not _valid_hash(self.sha256) or _digest(body) != self.sha256:
            raise EvidencePackError("evidence_pack_hash_mismatch")

    @classmethod
    def from_events(cls, events: list[dict], forecast_cutoff: str | datetime) -> EvidencePack:
        """Seal complete ResearchSession event pairs captured before cutoff."""
        cutoff = _timestamp(forecast_cutoff, "forecast_cutoff")
        if not isinstance(events, list):
            raise EvidencePackError("events_must_be_a_list")
        searches: dict[str, list[dict]] = {}
        pages: dict[str, list[dict]] = {}
        pending = None
        newest = None
        for event in events:
            kind, observed = _check_event(event, cutoff)
            if kind == "action":
                if pending is not None:
                    raise EvidencePackError("unpaired_recorded_action")
                pending = (event, observed)
                continue
            if pending is None:
                raise EvidencePackError("unpaired_recorded_observation")
            action, action_at = pending
            pending = None
            if action["tool"] != event["tool"] or action_at > observed:
                raise EvidencePackError("mismatched_recorded_pair")
            if event["tool"] not in {"search", "open"} or event["payload"].get("success") is not True:
                continue
            payload = event["payload"]
            if (payload.get("status") != "ok" or _timestamp(payload.get("observed_at"), "payload_observed_at") != observed
                    or payload.get("market_mode") not in {"market_aware", "no_consensus"}):
                raise EvidencePackError("invalid_successful_observation")
            key_name = "query" if event["tool"] == "search" else "url"
            if set(action["payload"]) != {key_name} or not isinstance(action["payload"][key_name], str):
                raise EvidencePackError("invalid_recorded_action")
            key = action["payload"][key_name]
            if not key.strip():
                raise EvidencePackError("invalid_recorded_action")
            if event["tool"] == "search":
                snapshots = payload.get("results")
                if not isinstance(snapshots, list) or not snapshots:
                    raise EvidencePackError("invalid_successful_observation")
                record = {"observed_at": _iso(observed), "event_sha256": event["sha256"],
                          "market_mode": payload["market_mode"], "snapshots": copy.deepcopy(snapshots)}
                searches.setdefault(key, []).append(record)
            else:
                snapshot = payload.get("result")
                record = {"observed_at": _iso(observed), "event_sha256": event["sha256"],
                          "market_mode": payload["market_mode"], "snapshot": copy.deepcopy(snapshot)}
                pages.setdefault(key, []).append(record)
            newest = max(newest, observed) if newest is not None else observed
        if pending is not None:
            raise EvidencePackError("unpaired_recorded_action")
        if newest is None:
            raise EvidencePackError("no_successful_research_observations")
        for records in (searches, pages):
            for history in records.values():
                history.sort(key=lambda item: item["observed_at"])
        body = {"schema_version": SCHEMA_VERSION, "forecast_cutoff": _iso(cutoff),
                "created_as_of": _iso(newest), "searches": searches, "pages": pages}
        _validate_body(body)
        return cls(json.dumps(body, ensure_ascii=False, sort_keys=True,
                              separators=(",", ":"), allow_nan=False), _digest(body))

    @classmethod
    def from_dict(cls, document: dict) -> EvidencePack:
        if not isinstance(document, dict) or set(document) != _BODY_FIELDS | {"pack_sha256"}:
            raise EvidencePackError("invalid_evidence_pack_schema")
        body = {key: value for key, value in document.items() if key != "pack_sha256"}
        _validate_body(body)
        if not _valid_hash(document["pack_sha256"]) or _digest(body) != document["pack_sha256"]:
            raise EvidencePackError("evidence_pack_hash_mismatch")
        return cls(json.dumps(body, ensure_ascii=False, sort_keys=True,
                              separators=(",", ":"), allow_nan=False), document["pack_sha256"])

    def to_dict(self) -> dict:
        return {**json.loads(self._body_json), "pack_sha256": self.sha256}

    def provider(self, replay_as_of: str | datetime | None = None) -> EvidencePackProvider:
        return EvidencePackProvider(self, replay_as_of=replay_as_of)


class EvidencePackProvider:
    """Read-only ResearchProvider with exact-match, timestamped replay.

    It never contains or invokes a live provider. A miss raises, and a normal
    ``ResearchSession`` converts that to an auditable provider error.
    """

    def __init__(self, pack: EvidencePack | dict, *, replay_as_of: str | datetime | None = None):
        self.pack = pack if isinstance(pack, EvidencePack) else EvidencePack.from_dict(pack)
        body = self.pack.to_dict()
        cutoff = _timestamp(body["forecast_cutoff"], "forecast_cutoff")
        self.replay_as_of = _timestamp(replay_as_of or body["created_as_of"], "replay_as_of")
        if self.replay_as_of >= cutoff:
            raise EvidencePackError("replay_as_of_must_precede_forecast_cutoff")
        self._searches = body["searches"]
        self._pages = body["pages"]
        self._cutoff = cutoff
        self._lock = threading.Lock()
        self._search_calls = 0
        self._open_calls = 0
        self._misses = 0
        self._replay_seconds = 0.0

    def public_config(self) -> dict:
        modes = sorted({record["market_mode"] for records in (self._searches, self._pages)
                        for history in records.values() for record in history})
        return {"adapter": SCHEMA_VERSION, "pack_sha256": self.pack.sha256,
                "forecast_cutoff": _iso(self._cutoff), "replay_as_of": _iso(self.replay_as_of),
                "recorded_searches": len(self._searches), "recorded_pages": len(self._pages),
                "captured_market_modes": modes, "transport": "closed_local_replay"}

    def _record(self, kind: str, started: float, missed: bool = False) -> None:
        with self._lock:
            if kind == "search":
                self._search_calls += 1
            else:
                self._open_calls += 1
            self._misses += int(missed)
            self._replay_seconds += time.perf_counter() - started

    def metrics(self) -> dict:
        with self._lock:
            return {"search_calls": self._search_calls, "open_calls": self._open_calls,
                    "misses": self._misses, "replay_seconds": self._replay_seconds}

    def _select(self, index: dict, key: str) -> dict:
        history = index.get(key) if isinstance(key, str) else None
        selected = None
        if history:
            for record in history:
                if _timestamp(record["observed_at"], "record_observed_at") <= self.replay_as_of:
                    selected = record
                else:
                    break
        if selected is None:
            raise EvidenceNotRecorded("exact_evidence_not_recorded_at_replay_time")
        return selected

    @staticmethod
    def _raw(snapshot: dict) -> dict:
        # ResearchSession will stamp delivery time, re-filter market content,
        # and reject evidence retrieved after its trusted virtual clock.
        return {"url": snapshot["url"], "title": snapshot["title"], "text": snapshot["text"],
                "published_at": snapshot["published_at"],
                "source_updated_at": snapshot.get("source_updated_at"),
                "retrieved_at": snapshot.get("retrieved_at") or snapshot["observed_at"]}

    def search(self, query: str) -> list[dict]:
        started, missed = time.perf_counter(), False
        try:
            record = self._select(self._searches, query)
            return [self._raw(snapshot) for snapshot in record["snapshots"]]
        except EvidenceNotRecorded:
            missed = True
            raise
        finally:
            self._record("search", started, missed)

    def open(self, url: str) -> dict:
        started, missed = time.perf_counter(), False
        try:
            record = self._select(self._pages, url)
            return self._raw(record["snapshot"])
        except EvidenceNotRecorded:
            missed = True
            raise
        finally:
            self._record("open", started, missed)


def synthetic_replay_microbenchmark(iterations: int = 20, slow_call_seconds: float = 0.002) -> dict:
    """Measure an offline fixture, not a forecast of real network speedup."""
    if isinstance(iterations, bool) or not isinstance(iterations, int) or not 1 <= iterations <= 100:
        raise ValueError("iterations must be between 1 and 100")
    if not isinstance(slow_call_seconds, (int, float)) or not 0 <= slow_call_seconds <= 0.05:
        raise ValueError("slow_call_seconds must be between 0 and 0.05")

    from .research import ResearchSession

    captured = datetime(2030, 1, 1, 12, tzinfo=timezone.utc)
    deadline = captured + timedelta(minutes=10)
    question = {"issued_at": _iso(captured - timedelta(minutes=1)),
                "forecast_deadline": _iso(deadline)}

    class SlowFixture:
        def __init__(self):
            self.calls = 0

        def search(self, query):
            self.calls += 1
            time.sleep(slow_call_seconds)
            return [{"url": "https://example.org/report", "title": "Synthetic report",
                     "text": "Synthetic evidence available before the forecast deadline.",
                     "published_at": _iso(captured - timedelta(hours=1))}]

        def open(self, url):
            self.calls += 1
            time.sleep(slow_call_seconds)
            return {"url": "https://example.org/report", "title": "Synthetic report",
                    "text": "Synthetic evidence available before the forecast deadline.",
                    "published_at": _iso(captured - timedelta(hours=1))}

    slow = SlowFixture()
    session = ResearchSession(question, slow, max_calls=2, clock=lambda: captured)
    session.search("synthetic forecast")
    session.open("https://example.org/report")
    pack = EvidencePack.from_events(session.events, deadline)
    replay = pack.provider(replay_as_of=captured)
    slow.calls = 0
    started = time.perf_counter()
    for _ in range(iterations):
        slow.search("synthetic forecast")
        slow.open("https://example.org/report")
    live_seconds = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(iterations):
        replay.search("synthetic forecast")
        replay.open("https://example.org/report")
    replay_seconds = time.perf_counter() - started
    return {"fixture": "synthetic_sleep_provider_v1", "iterations": iterations,
            "slow_call_seconds": slow_call_seconds, "synthetic_provider_calls": slow.calls,
            "replay_metrics": replay.metrics(), "synthetic_provider_seconds": live_seconds,
            "pack_replay_seconds": replay_seconds, "pack_sha256": pack.sha256,
            "scope": "Offline synthetic timing only; no real-world speedup claim."}


if __name__ == "__main__":
    print(json.dumps(synthetic_replay_microbenchmark(), sort_keys=True))
