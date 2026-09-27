"""Preregistered question templates backed by public, first-party JSON sources.

Adapters only propose questions and resolutions. The pipeline owns publication,
retries, review, persistence, and immutable snapshot storage. No adapter treats a
network failure, malformed response, or absent result as a negative outcome.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from typing import Protocol
from urllib.parse import urlencode

from .schema import parse_timestamp, validate_question


MLB_API = "https://statsapi.mlb.com/api/v1/schedule"
USGS_API = "https://earthquake.usgs.gov/fdsnws/event/1/"


class JsonClient(Protocol):
    last_snapshot: dict

    def get_json(self, url: str) -> dict | list: ...


def _utc(now: datetime) -> datetime:
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    return now.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return _utc(value).isoformat().replace("+00:00", "Z")


def _number(config: dict, key: str, default: float, minimum: float,
            maximum: float) -> float:
    value = config.get(key, default)
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not minimum <= value <= maximum):
        raise ValueError(f"{key} must be between {minimum} and {maximum}")
    return value


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _snapshot(client: JsonClient) -> dict:
    return copy.deepcopy(client.last_snapshot)


def _proposal(status: str, reason: str, url: str | None = None,
              outcome: str | None = None, evidence: str = "",
              snapshots: list[dict] | None = None) -> dict:
    return {
        "status": status, "outcome": outcome, "reason": reason,
        "evidence_urls": [url] if url else [],
        "evidence_text": evidence or reason,
        "source_snapshots": snapshots or [],
    }


class SourceAdapter:
    source_id: str

    def discover(self, now: datetime, config: dict, client: JsonClient) -> list[dict]:
        raise NotImplementedError

    def resolve(self, question: dict, now: datetime, client: JsonClient) -> dict:
        raise NotImplementedError


class MLBSource(SourceAdapter):
    """Home-team winner, including extra innings, for a scheduled MLB game."""

    source_id = "mlb"
    template_version = "mlb-home-win-v1"

    def discover(self, now: datetime, config: dict, client: JsonClient) -> list[dict]:
        now = _utc(now)
        horizon = timedelta(hours=_number(config, "horizon_hours", 168, 1, 168))
        lead = timedelta(minutes=_number(config, "min_lead_minutes", 30, 1, 1440))
        cutoff = timedelta(minutes=_number(config, "deadline_before_start_minutes", 60, 1, 1440))
        url = MLB_API + "?" + urlencode({
            "sportId": 1, "startDate": now.date().isoformat(),
            "endDate": (now + horizon).date().isoformat(),
        })
        payload = client.get_json(url)
        games = self._games(payload)
        provenance = _snapshot(client)
        questions = []
        for game in games:
            status = game.get("status", {})
            if (status.get("abstractGameState") != "Preview"
                    or status.get("detailedState") not in ("Scheduled", "Pre-Game")
                    or status.get("startTimeTBD") is not False
                    or game.get("ifNecessary") not in (None, "N")
                    or game.get("resumeDate") or game.get("resumedFrom")):
                continue
            try:
                start = parse_timestamp(game["gameDate"])
                home = game["teams"]["home"]["team"]
                away = game["teams"]["away"]["team"]
                if not all(_positive_int(value) for value in (game["gamePk"], home["id"], away["id"])):
                    continue
                if not all(isinstance(team["name"], str) and team["name"].strip()
                           for team in (home, away)):
                    continue
            except (KeyError, TypeError, ValueError):
                continue
            deadline = start - cutoff
            target = start + timedelta(hours=6)
            if deadline < now + lead or target > now + horizon or target - deadline > horizon:
                continue
            event_id = f"mlb-game-{game['gamePk']}"
            resolution_url = MLB_API + "?" + urlencode({"sportId": 1, "gamePk": game["gamePk"]})
            question = {
                "schema_version": "0.1", "is_fixture": False,
                "question_id": f"{event_id}-home-win-v1",
                "event_id": event_id, "cluster_id": event_id,
                "split": config.get("split", "test"), "kind": "binary", "domain": "sports",
                "prompt": (f"Will the {home['name']} beat the {away['name']} in MLB game "
                           f"{game['gamePk']}, scheduled to start at {_iso(start)}?"),
                "options": [{"id": "yes", "text": f"{home['name']} win"},
                            {"id": "no", "text": f"{away['name']} win"}],
                "issued_at": _iso(now), "forecast_deadline": _iso(deadline),
                "outcome_not_before": _iso(start),
                "resolve_after": _iso(start + timedelta(hours=8)),
                "resolution": {
                    "criteria": (
                        "Use MLB's Final status and final home/away scores, including extra innings. "
                        "Resolve yes iff the home score is greater; no iff the away score is greater. "
                        "Void if cancelled, postponed, or rescheduled outside the original start-plus-8-hour "
                        "window. A start moved to or before the forecast deadline voids the question. "
                        "Unfinished games wait until original start plus 24 hours, then void. "
                        "Finals first observed after 24 hours, tied scores, changed participants, or "
                        "resume/suspension metadata require review; never infer a result."
                    ),
                    "source_urls": [resolution_url], "on_ambiguous": "void",
                },
                "metadata": {
                    "source_id": self.source_id, "template_version": self.template_version,
                    "source_event_id": str(game["gamePk"]), "target_at": _iso(target),
                    "discovery_provenance": provenance,
                    "resolution_parameters": {
                        "game_pk": game["gamePk"], "home_team_id": home["id"],
                        "away_team_id": away["id"], "original_start": _iso(start),
                        "latest_start": _iso(start + timedelta(hours=8)),
                        "void_after": _iso(start + timedelta(hours=24)),
                    },
                },
            }
            validate_question(question)
            questions.append(question)
        return sorted(questions, key=lambda q: (q["forecast_deadline"], q["question_id"]))

    @staticmethod
    def _games(payload: object) -> list[dict]:
        if not isinstance(payload, dict) or not isinstance(payload.get("dates"), list):
            raise ValueError("MLB schedule response must contain a dates array")
        games = []
        for date in payload["dates"]:
            if not isinstance(date, dict) or not isinstance(date.get("games"), list):
                raise ValueError("Malformed MLB schedule date")
            if not all(isinstance(game, dict) for game in date["games"]):
                raise ValueError("Malformed MLB schedule game")
            games.extend(date["games"])
        total = payload.get("totalGames")
        if not isinstance(total, int) or isinstance(total, bool) or total != len(games):
            raise ValueError("Incomplete or inconsistent MLB schedule response")
        return games

    def resolve(self, question: dict, now: datetime, client: JsonClient) -> dict:
        now = _utc(now)
        if now < parse_timestamp(question["resolve_after"]):
            return _proposal("pending", "The game settlement window has not opened")
        params = question["metadata"]["resolution_parameters"]
        url = MLB_API + "?" + urlencode({"sportId": 1, "gamePk": params["game_pk"]})
        games = self._games(client.get_json(url))
        snapshots = [_snapshot(client)]
        matching = [game for game in games if game.get("gamePk") == params["game_pk"]]
        if len(matching) != 1:
            return _proposal("needs_review", "Expected one matching MLB game", url, snapshots=snapshots)
        game = matching[0]
        status = game.get("status", {})
        detail = status.get("detailedState", "Unknown")
        if any(word in detail.lower() for word in ("postponed", "cancelled", "canceled")):
            return _proposal("void", f"MLB reports {detail}", url, snapshots=snapshots)
        try:
            current_start = parse_timestamp(game["gameDate"])
            home, away = game["teams"]["home"], game["teams"]["away"]
            if home["team"]["id"] != params["home_team_id"] or away["team"]["id"] != params["away_team_id"]:
                raise ValueError("Changed participants")
        except (ValueError, KeyError, TypeError):
            return _proposal("needs_review", "Malformed or changed game identity", url, snapshots=snapshots)
        if (current_start > parse_timestamp(params["latest_start"])
                or current_start <= parse_timestamp(question["forecast_deadline"])):
            return _proposal("void", "Game start moved outside the preregistered window", url, snapshots=snapshots)
        if game.get("resumeDate") or game.get("resumedFrom"):
            return _proposal("needs_review", "Resumed game needs a timing review", url, snapshots=snapshots)
        if status.get("abstractGameState") == "Final" and detail == "Final":
            if now > parse_timestamp(params["void_after"]):
                return _proposal("needs_review", "Final first observed after the 24-hour window", url, snapshots=snapshots)
            scores = (home.get("score"), away.get("score"))
            if not all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in scores) or scores[0] == scores[1]:
                return _proposal("needs_review", "Final game lacks unequal valid scores", url, snapshots=snapshots)
            return _proposal("resolved", "Official MLB Final score", url,
                             "yes" if scores[0] > scores[1] else "no",
                             f"Game {params['game_pk']}: home {scores[0]}, away {scores[1]}; MLB status Final.", snapshots)
        if now >= parse_timestamp(params["void_after"]):
            return _proposal("void", "No official Final result by the 24-hour window", url, snapshots=snapshots)
        return _proposal("pending", f"MLB status is {detail}; waiting for Final", url, snapshots=snapshots)


class USGSSource(SourceAdapter):
    """Daily global earthquake count in fixed bins, frozen after a settling delay."""

    source_id = "usgs"
    template_version = "usgs-daily-count-v1"

    def discover(self, now: datetime, config: dict, client: JsonClient) -> list[dict]:
        now = _utc(now)
        if config.get("region", "global") != "global":
            raise ValueError("USGS v1 supports the global region only")
        magnitude = _number(config, "min_magnitude", 5.0, 0, 10)
        settle = _number(config, "settle_hours", 24, 1, 168)
        lead = timedelta(minutes=_number(config, "min_lead_minutes", 30, 1, 1440))
        cutoff = timedelta(minutes=_number(config, "deadline_before_start_minutes", 60, 1, 1440))
        horizon = timedelta(hours=_number(config, "horizon_hours", 168, 1, 168))
        lookahead = _number(config, "lookahead_days", 5, 1, 7)
        if int(lookahead) != lookahead:
            raise ValueError("lookahead_days must be an integer")
        capability = client.get_json(USGS_API + "application.json")
        if not isinstance(capability, dict) or "earthquake" not in capability.get("eventtypes", []):
            raise ValueError("USGS capability response lacks the earthquake event type")
        provenance = _snapshot(client)
        questions = []
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        for offset in range(1, int(lookahead) + 1):
            start = midnight + timedelta(days=offset)
            end = start + timedelta(days=1)
            deadline = start - cutoff
            if deadline < now + lead or end > now + horizon or end - deadline > horizon:
                continue
            params = {
                "start": _iso(start), "end": _iso(end), "region": "global",
                "min_magnitude": magnitude, "settle_hours": settle,
                "bins": [{"id": "zero", "min": 0, "max": 0},
                         {"id": "one", "min": 1, "max": 1},
                         {"id": "two_or_more", "min": 2, "max": None}],
                "revision_policy": "freeze_first_successful_count_at_or_after_resolve_after",
            }
            fingerprint = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]
            event_id = f"usgs-global-{start.date().isoformat()}"
            url = self._count_url(params)
            question = {
                "schema_version": "0.1", "is_fixture": False,
                "question_id": f"{event_id}-count-v1-{fingerprint}",
                "event_id": event_id, "cluster_id": event_id,
                "split": config.get("split", "test"), "kind": "categorical", "domain": "geophysics",
                "prompt": (f"How many earthquakes worldwide with USGS preferred magnitude at least "
                           f"{magnitude:g} will have an origin time on {start.date().isoformat()} UTC, "
                           f"according to the USGS catalog snapshot first successfully fetched at or "
                           f"after {_iso(end + timedelta(hours=settle))}?"),
                "options": [{"id": "zero", "text": "0 earthquakes"},
                            {"id": "one", "text": "1 earthquake"},
                            {"id": "two_or_more", "text": "2 or more earthquakes"}],
                "issued_at": _iso(now), "forecast_deadline": _iso(deadline),
                "outcome_not_before": _iso(start), "resolve_after": _iso(end + timedelta(hours=settle)),
                "resolution": {
                    "criteria": (
                        f"Count global USGS eventtype=earthquake records with minmagnitude={magnitude:g}, "
                        f"origin times in [{_iso(start)}, {_iso(end)}), and reviewstatus=all. "
                        f"Use the first successful count response at or after {settle:g} hours after the day ends. "
                        "Count 0 maps to zero, 1 to one, and >=2 to two_or_more. "
                        "The API endtime is inclusive, so query through the final millisecond of the day. "
                        "Freeze the archived response: later additions, deletions, or magnitude revisions "
                        "do not change the label. This measures that catalog snapshot, not all physical earthquakes. "
                        "Missing, malformed, or failed responses stay unresolved and are never interpreted as zero."
                    ),
                    "source_urls": [url], "on_ambiguous": "void",
                },
                "metadata": {
                    "source_id": self.source_id, "template_version": self.template_version,
                    "source_event_id": event_id, "target_at": _iso(end),
                    "discovery_provenance": provenance, "resolution_parameters": params,
                },
            }
            validate_question(question)
            questions.append(question)
        return questions

    @staticmethod
    def _count_url(params: dict) -> str:
        end_inclusive = parse_timestamp(params["end"]) - timedelta(milliseconds=1)
        return USGS_API + "count?" + urlencode({
            "format": "geojson", "starttime": params["start"], "endtime": _iso(end_inclusive),
            # Omitting reviewstatus includes all records. Although the API docs
            # name "all" as the default, explicitly sending it returns HTTP 400.
            "minmagnitude": params["min_magnitude"], "eventtype": "earthquake",
        })

    def resolve(self, question: dict, now: datetime, client: JsonClient) -> dict:
        now = _utc(now)
        if now < parse_timestamp(question["resolve_after"]):
            return _proposal("pending", "UTC day or preregistered catalog settling delay has not ended")
        params = question["metadata"]["resolution_parameters"]
        url = self._count_url(params)
        payload = client.get_json(url)
        snapshots = [_snapshot(client)]
        if (not isinstance(payload, dict) or not isinstance(payload.get("count"), int)
                or isinstance(payload.get("count"), bool) or payload["count"] < 0
                or not _positive_int(payload.get("maxAllowed"))):
            return _proposal("needs_review", "Malformed USGS aggregate count response", url, snapshots=snapshots)
        count = payload["count"]
        matching = [entry["id"] for entry in params["bins"]
                    if count >= entry["min"] and (entry["max"] is None or count <= entry["max"])]
        if len(matching) != 1:
            return _proposal("needs_review", "Count bins are not exhaustive and disjoint", url, snapshots=snapshots)
        return _proposal("resolved", "Official USGS aggregate count after settling delay", url,
                         matching[0], f"USGS aggregate count={count}; UTC interval [{params['start']}, "
                         f"{params['end']}); minimum magnitude={params['min_magnitude']}; "
                         f"reviewstatus=all. Archived snapshot is final under this question's revision policy.", snapshots)


SOURCE_REGISTRY = {"mlb": MLBSource, "usgs": USGSSource}


def build_sources(config: dict) -> list[tuple[SourceAdapter, dict]]:
    """Build enabled adapters with their source-specific configuration."""
    entries = config.get("sources")
    if not isinstance(entries, list):
        raise ValueError("sources must be a list")
    result = []
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("id") not in SOURCE_REGISTRY:
            raise ValueError("Every source must have a registered id")
        identifier = entry["id"]
        if identifier in seen:
            raise ValueError(f"Duplicate source id: {identifier}")
        seen.add(identifier)
        if not isinstance(entry.get("enabled", True), bool):
            raise ValueError("source.enabled must be boolean")
        if entry.get("split", "test") not in ("train", "dev", "test"):
            raise ValueError("source.split must be train, dev, or test")
        if entry.get("enabled", True):
            result.append((SOURCE_REGISTRY[identifier](), copy.deepcopy(entry)))
    return result
