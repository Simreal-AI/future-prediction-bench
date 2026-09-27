"""Trusted-runner storage. SQLite receipts are local audit records, not attestations."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from .schema import option_ids, parse_timestamp, public_question, validate_probabilities, validate_question
from .scoring import score_prediction


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


class Store:
    """Single trusted runner; fixture databases and clocks cannot be used as live runs."""

    def __init__(self, path, *, mode="live", clock=None):
        if mode not in {"live", "fixture"}:
            raise ValueError("mode must be live or fixture")
        if clock is not None and mode != "fixture":
            raise ValueError("Custom clocks are only allowed for explicitly marked fixtures")
        self.mode = mode
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS questions (
                question_id TEXT PRIMARY KEY, event_id TEXT NOT NULL, cluster_id TEXT NOT NULL,
                split TEXT NOT NULL, payload TEXT NOT NULL, sha256 TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id TEXT PRIMARY KEY, question_id TEXT NOT NULL REFERENCES questions,
                policy_id TEXT NOT NULL, track TEXT NOT NULL, market_mode TEXT NOT NULL,
                research_mode TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL,
                submitted_at TEXT, prediction TEXT, raw_response TEXT, error TEXT,
                receipt_sha256 TEXT, reward REAL, scores TEXT);
            CREATE TABLE IF NOT EXISTS events (
                episode_id TEXT NOT NULL REFERENCES episodes, sequence INTEGER NOT NULL,
                payload TEXT NOT NULL, PRIMARY KEY(episode_id, sequence));
            CREATE TABLE IF NOT EXISTS resolutions (
                question_id TEXT PRIMARY KEY REFERENCES questions, payload TEXT NOT NULL,
                resolved_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS private_baselines (
                question_id TEXT PRIMARY KEY REFERENCES questions, probabilities TEXT NOT NULL,
                kind TEXT NOT NULL, identity TEXT NOT NULL, metadata TEXT NOT NULL,
                observed_at TEXT NOT NULL, sealed_at TEXT NOT NULL, sha256 TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS model_turns (
                episode_id TEXT NOT NULL REFERENCES episodes, sequence INTEGER NOT NULL,
                payload TEXT NOT NULL, PRIMARY KEY(episode_id,sequence));
            CREATE TABLE IF NOT EXISTS rollout_metadata (
                episode_id TEXT PRIMARY KEY REFERENCES episodes, group_id TEXT NOT NULL,
                sample_index INTEGER NOT NULL, payload TEXT NOT NULL,
                UNIQUE(group_id, sample_index));
        """)
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(episodes)")}
        if "reward_mode" not in columns:
            self.db.execute("ALTER TABLE episodes ADD COLUMN reward_mode TEXT NOT NULL DEFAULT 'negative_brier'")
            self.db.commit()
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('mode', ?)", (mode,))
        existing_mode = self.db.execute("SELECT value FROM metadata WHERE key='mode'").fetchone()[0]
        if existing_mode != mode:
            self.db.close()
            raise ValueError("Database run mode cannot be changed")

    def close(self):
        self.db.close()

    def now(self):
        value = self.clock()
        if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("clock must return a timezone-aware datetime")
        return value.astimezone(timezone.utc)

    def add_question(self, question):
        validate_question(question)
        if bool(question.get("is_fixture", False)) != (self.mode == "fixture"):
            raise ValueError("Question fixture flag does not match database mode")
        encoded = canonical(question)
        with self.db:
            old = self.db.execute("SELECT payload FROM questions WHERE question_id=?", (question["question_id"],)).fetchone()
            if old:
                if old[0] != encoded:
                    raise ValueError("Published question is immutable; use a new version/id")
                return question["question_id"]
            conflict = self.db.execute(
                "SELECT question_id FROM questions WHERE (event_id=? OR cluster_id=?) AND split!=?",
                (question["event_id"], question["cluster_id"], question["split"]),
            ).fetchone()
            if conflict:
                raise ValueError("An event or cluster cannot cross train/dev/test splits")
            self.db.execute("INSERT INTO questions VALUES (?, ?, ?, ?, ?, ?)",
                            (question["question_id"], question["event_id"], question["cluster_id"],
                             question["split"], encoded, digest(question)))
        return question["question_id"]

    def question(self, question_id):
        row = self.db.execute("SELECT payload FROM questions WHERE question_id=?", (question_id,)).fetchone()
        if not row:
            raise ValueError("Unknown question")
        return json.loads(row[0])

    def _episode(self, episode_id):
        row = self.db.execute("SELECT * FROM episodes WHERE episode_id=?", (episode_id,)).fetchone()
        if not row:
            raise ValueError("Unknown episode")
        return dict(row)

    def episode(self, episode_id):
        self.expire(episode_id)
        row = self._episode(episode_id)
        for field in ("prediction", "raw_response", "scores"):
            if row[field] is not None:
                row[field] = json.loads(row[field])
        row["run_mode"] = self.mode
        return row

    def expire(self, episode_id=None):
        """Expired assignments remain in coverage/penalty denominators, never RL data."""
        parameters = (episode_id,) if episode_id is not None else ()
        query = "SELECT episode_id,question_id FROM episodes WHERE status='active'"
        if episode_id is not None:
            query += " AND episode_id=?"
        now = self.now()
        expired = [row["episode_id"] for row in self.db.execute(query, parameters)
                   if now >= parse_timestamp(self.question(row["question_id"])["forecast_deadline"])]
        with self.db:
            self.db.executemany("UPDATE episodes SET status='missed',error='forecast_deadline_reached' WHERE episode_id=?", [(item,) for item in expired])
        return expired

    def _check_open(self, question, now):
        if not parse_timestamp(question["issued_at"]) <= now < parse_timestamp(question["forecast_deadline"]):
            raise ValueError("Outside the forecast window; deadline is exclusive")
        if self.db.execute("SELECT 1 FROM resolutions WHERE question_id=?", (question["question_id"],)).fetchone():
            raise ValueError("Question is already resolved or void")

    def create_episode(self, question_id, policy_id, *, track="benchmark", market_mode="no_consensus", research_mode="self_research", reward_mode="negative_brier"):
        if not isinstance(policy_id, str) or not policy_id.strip():
            raise ValueError("A model/checkpoint policy_id is required")
        if track not in {"benchmark", "rl"} or market_mode not in {"no_consensus", "market_aware"}:
            raise ValueError("Invalid track or market mode")
        if research_mode not in {"self_research", "no_search"}:
            raise ValueError("Invalid research mode")
        if reward_mode not in {"negative_brier", "baseline_improvement"}:
            raise ValueError("Invalid reward mode")
        question = self.question(question_id)
        now = self.now()
        self._check_open(question, now)
        if reward_mode == "baseline_improvement" and self.private_baseline(question_id) is None:
            raise ValueError("A sealed baseline is required before baseline-improvement episodes")
        if track == "rl" and question["split"] != "train":
            raise ValueError("RL episodes are restricted to the train split")
        if track == "benchmark" and self.db.execute("""SELECT 1 FROM episodes WHERE question_id=?
            AND policy_id=? AND track='benchmark' AND market_mode=? AND research_mode=? AND reward_mode=?""",
            (question_id, policy_id, market_mode, research_mode, reward_mode)).fetchone():
            raise ValueError("One benchmark episode per question and system configuration")
        episode_id = uuid.uuid4().hex
        with self.db:
            self.db.execute("""INSERT INTO episodes
                (episode_id, question_id, policy_id, track, market_mode, research_mode, status, created_at, reward_mode)
                VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?)""",
                (episode_id, question_id, policy_id, track, market_mode, research_mode, now.isoformat(), reward_mode))
        return episode_id

    def register_rollout(self, episode_id, context):
        """Freeze trusted collection metadata before sampling any policy actions.

        Group IDs refer to one question, policy revision, sampling configuration,
        and information regime. They are not agent-selectable tool arguments.
        """
        from .training import validate_rollout_context

        value = validate_rollout_context(context)
        episode = self.episode(episode_id)
        if episode["track"] != "rl" or episode["status"] != "active":
            raise ValueError("Rollout groups require active RL episodes")
        if self.events(episode_id) or self.model_turns(episode_id):
            raise ValueError("Register rollout metadata before collecting actions")
        if self.rollout(episode_id) is not None:
            raise ValueError("Rollout metadata is immutable")
        scope = {key: episode[key] for key in ("question_id", "market_mode", "research_mode", "reward_mode")}
        value.update(scope)
        value["registered_at"] = self.now().isoformat()
        value["question_sha256"] = digest(public_question(self.question(episode["question_id"])))
        prior = self.db.execute("SELECT payload FROM rollout_metadata WHERE group_id=? LIMIT 1", (value["group_id"],)).fetchone()
        if prior:
            previous = json.loads(prior[0])
            fields = (*scope, "policy_revision", "group_size", "collection_config_hash", "evidence_pack_id", "question_sha256")
            if any(previous.get(key) != value.get(key) for key in fields):
                raise ValueError("Rollout group cannot mix questions, policies, configurations, or evidence packs")
        value["sha256"] = digest(value)
        try:
            with self.db:
                self.db.execute("INSERT INTO rollout_metadata VALUES (?,?,?,?)",
                                (episode_id, value["group_id"], value["sample_index"], canonical(value)))
        except sqlite3.IntegrityError:
            raise ValueError("A rollout sample index cannot be collected twice") from None
        return value

    def rollout(self, episode_id):
        row = self.db.execute("SELECT payload FROM rollout_metadata WHERE episode_id=?", (episode_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def seal_baseline(self, question_id, probabilities, *, kind, identity, metadata=None, observed_at=None):
        """Private administrator action; seal before any forecast episode starts."""
        if kind not in {"market", "internal_model"} or not isinstance(identity, str) or not identity.strip():
            raise ValueError("A real baseline kind and source/model identity are required")
        question = self.question(question_id)
        distribution = validate_probabilities(probabilities, option_ids(question))
        now = self.now()
        self._check_open(question, now)
        if self.db.execute("SELECT 1 FROM episodes WHERE question_id=?", (question_id,)).fetchone():
            raise ValueError("Baselines must be sealed before any forecast episode begins")
        observed = parse_timestamp(observed_at) if observed_at is not None else now
        if not parse_timestamp(question["issued_at"]) <= observed <= now:
            raise ValueError("Baseline observation must fall within the current forecast window")
        value = {"probabilities": distribution, "kind": kind, "identity": identity,
                 "metadata": metadata or {}, "observed_at": observed.isoformat()}
        if self.private_baseline(question_id) is not None:
            raise ValueError("A sealed baseline is immutable")
        with self.db:
            self.db.execute("INSERT INTO private_baselines VALUES(?,?,?,?,?,?,?,?)",
                            (question_id, canonical(distribution), kind, identity, canonical(metadata or {}),
                             observed.isoformat(), now.isoformat(), digest(value)))
        return {"question_id": question_id, "status": "sealed", "kind": kind}

    def private_baseline(self, question_id):
        """Trusted administration only. Never expose this method through agent tools."""
        row = self.db.execute("SELECT * FROM private_baselines WHERE question_id=?", (question_id,)).fetchone()
        if row is None:
            return None
        value = dict(row)
        value["probabilities"] = json.loads(value["probabilities"])
        value["metadata"] = json.loads(value["metadata"])
        return value

    def append_model_turn(self, episode_id, record):
        episode = self._episode(episode_id)
        if episode["status"] != "active":
            raise ValueError("Model turns can only be appended during an active episode")
        at = parse_timestamp(record["at"])
        if not parse_timestamp(episode["created_at"]) <= at <= self.now():
            raise ValueError("Invalid model turn timestamp")
        # Late responses may be audited but must never be submitted as a prediction.
        with self.db:
            index = self.db.execute("SELECT COUNT(*) FROM model_turns WHERE episode_id=?", (episode_id,)).fetchone()[0]
            self.db.execute("INSERT INTO model_turns VALUES(?,?,?)", (episode_id, index, canonical(record)))

    def model_turns(self, episode_id):
        return [json.loads(row[0]) for row in self.db.execute("SELECT payload FROM model_turns WHERE episode_id=? ORDER BY sequence", (episode_id,))]

    def events(self, episode_id):
        self._episode(episode_id)
        return [json.loads(row[0]) for row in self.db.execute(
            "SELECT payload FROM events WHERE episode_id=? ORDER BY sequence", (episode_id,))]

    def append_events(self, episode_id, events):
        episode = self._episode(episode_id)
        if episode["status"] != "active":
            raise ValueError("Cannot append to a submitted episode")
        if episode["research_mode"] == "no_search":
            raise ValueError("No-search ablation cannot receive tool observations")
        question = self.question(episode["question_id"])
        now = self.now()
        if now < parse_timestamp(question["issued_at"]):
            raise ValueError("Question is not yet issued")
        if self.db.execute("SELECT 1 FROM resolutions WHERE question_id=?", (question["question_id"],)).fetchone():
            raise ValueError("Cannot append after resolution")
        prior = self.events(episode_id)
        previous_time = parse_timestamp(prior[-1]["at"]) if prior else parse_timestamp(episode["created_at"])
        encoded = []
        for event in events:
            if event.get("type") not in {"action", "observation"} or event.get("tool") not in {
                "search", "open", "calculator", "notebook", "draft", "market_search", "market_snapshot"
            }:
                raise ValueError("Invalid tool event")
            if event.get("loss_mask") != (1 if event["type"] == "action" else 0):
                raise ValueError("Only policy actions may have loss_mask=1")
            at = parse_timestamp(event["at"])
            payload = event.get("payload")
            late_failure = event["type"] == "observation" and isinstance(payload, dict) and payload.get("success") is False and payload.get("reason") == "forecast_deadline_reached"
            if not previous_time <= at <= now or (at >= parse_timestamp(question["forecast_deadline"]) and not late_failure):
                raise ValueError("Tool events must be chronological and received before deadline")
            if not isinstance(payload, dict):
                raise ValueError("Tool payload must be an object")
            if late_failure and (payload.get("results") or payload.get("result")):
                raise ValueError("A late failure cannot contain retrieved evidence")
            if event.get("sha256") != digest({key: value for key, value in event.items() if key != "sha256"}):
                raise ValueError("Tool event digest mismatch")
            encoded.append(canonical(event))
            previous_time = at
        with self.db:
            self.db.executemany("INSERT INTO events VALUES (?, ?, ?)",
                                [(episode_id, len(prior) + i, item) for i, item in enumerate(encoded)])
        self.expire(episode_id)

    def submit(self, episode_id, probabilities, *, agent_generated=True):
        if not isinstance(agent_generated, bool):
            raise ValueError("agent_generated must be boolean")
        self.expire(episode_id)
        episode = self._episode(episode_id)
        if episode["status"] != "active":
            raise ValueError("One immutable final submission per episode")
        question = self.question(episode["question_id"])
        now = self.now()
        self._check_open(question, now)
        events = self.events(episode_id)
        searches = sum(e["type"] == "observation" and e["tool"] == "search"
                       and e["payload"].get("success") is True and bool(e["payload"].get("results")) for e in events)
        error = None
        normalized = None
        try:
            normalized = validate_probabilities(probabilities, option_ids(question))
            if episode["research_mode"] == "self_research" and searches < 1:
                raise ValueError("Self-research requires at least one successful search")
        except (ValueError, TypeError) as exc:
            error = str(exc)
        try:
            raw = canonical(probabilities)
        except (TypeError, ValueError):
            raw = canonical({"unserializable_repr": repr(probabilities)})
        status = "invalid" if error else "pending_reward"
        reward = -1.0 if error else None
        receipt = digest({"episode_id": episode_id, "question_sha256": digest(question),
                          "policy_id": episode["policy_id"], "submitted_at": now.isoformat(),
                          "raw_response": json.loads(raw), "events": events, "agent_generated": agent_generated})
        final_action = {"type": "action", "tool": "submit", "at": now.isoformat(),
                        "payload": json.loads(raw), "loss_mask": int(agent_generated),
                        "origin": "policy" if agent_generated else "host"}
        final_action["sha256"] = digest(final_action)
        with self.db:
            self.db.execute("""UPDATE episodes SET status=?,submitted_at=?,prediction=?,raw_response=?,
                error=?,receipt_sha256=?,reward=? WHERE episode_id=?""",
                (status, now.isoformat(), canonical(normalized) if not error else None, raw,
                 error, receipt, reward, episode_id))
            self.db.execute("INSERT INTO events VALUES (?, ?, ?)", (episode_id, len(events), canonical(final_action)))
        return self.episode(episode_id)

    def resolve(self, question_id, *, outcome=None, status="resolved", evidence_urls, evidence_text):
        """Trusted administrator operation; never exposed in the agent action space."""
        question = self.question(question_id)
        now = self.now()
        if now < parse_timestamp(question["outcome_not_before"]):
            raise ValueError("Too early to resolve this question")
        if status not in {"resolved", "void"}:
            raise ValueError("Resolution status must be resolved or void")
        if status == "resolved" and outcome not in option_ids(question):
            raise ValueError("Unknown resolved option")
        if status == "void" and outcome is not None:
            raise ValueError("A void question cannot have an outcome")
        if not isinstance(evidence_text, str) or not evidence_text.strip():
            raise ValueError("Resolution evidence or void reason is required")
        if not isinstance(evidence_urls, list) or not evidence_urls:
            raise ValueError("Resolution evidence URLs are required")
        for url in evidence_urls:
            if not isinstance(url, str) or urlparse(url).scheme not in {"http", "https"} or not urlparse(url).hostname:
                raise ValueError("Invalid evidence URL")
        permitted_hosts = {urlparse(url).hostname for url in question["resolution"]["source_urls"]}
        if not any(urlparse(url).hostname in permitted_hosts for url in evidence_urls):
            raise ValueError("Evidence must include a preregistered resolution source host")
        resolution = {"status": status, "outcome": outcome, "evidence_urls": evidence_urls,
                      "evidence_text": evidence_text}
        encoded = canonical(resolution)
        with self.db:
            existing = self.db.execute("SELECT payload FROM resolutions WHERE question_id=?", (question_id,)).fetchone()
            if existing:
                if existing[0] != encoded:
                    raise ValueError("Conflicting resolution; adjudication/versioning is required")
                return json.loads(existing[0])
            self.db.execute("INSERT INTO resolutions VALUES (?, ?, ?)", (question_id, encoded, now.isoformat()))
            self.db.execute("UPDATE episodes SET status=?,error='forecast_deadline_reached' WHERE question_id=? AND status='active'",
                            ("missed" if status == "resolved" else "void", question_id))
            if status == "void":
                self.db.execute("UPDATE episodes SET status='void' WHERE question_id=? AND status='missed'", (question_id,))
            for episode in self.db.execute("SELECT * FROM episodes WHERE question_id=? AND status='pending_reward'", (question_id,)).fetchall():
                if status == "void":
                    self.db.execute("UPDATE episodes SET status='void' WHERE episode_id=?", (episode["episode_id"],))
                else:
                    scores = score_prediction(json.loads(episode["prediction"]), outcome)
                    if episode["reward_mode"] == "baseline_improvement":
                        baseline = self.private_baseline(question_id)
                        if baseline is None:
                            raise ValueError("Sealed baseline is missing; refusing reward substitution")
                        baseline_loss = score_prediction(baseline["probabilities"], outcome)["brier"]
                        scores["reward"] = baseline_loss - scores["brier"]
                    scores["reward_mode"] = episode["reward_mode"]
                    self.db.execute("UPDATE episodes SET status='graded', reward=?, scores=? WHERE episode_id=?",
                                    (scores["reward"], canonical(scores), episode["episode_id"]))
        return resolution

    def export_training(self):
        """Audit/replay JSONL records; deliberately NOT a tokenized GRPO training batch."""
        records = []
        for row in self.db.execute("""SELECT e.episode_id FROM episodes e JOIN questions q USING(question_id)
            WHERE q.split='train' AND e.track='rl' AND e.status IN ('graded','invalid')
            ORDER BY e.created_at, e.episode_id"""):
            episode = self.episode(row[0])
            resolution_row = self.db.execute("SELECT payload,resolved_at FROM resolutions WHERE question_id=?", (episode["question_id"],)).fetchone()
            resolution = json.loads(resolution_row[0]) if resolution_row else None
            if resolution and resolution["status"] == "void":
                continue
            question = self.question(episode["question_id"])
            records.append({"schema_version": "0.1", "record_type": "text_trajectory",
                            "trainer_ready": False, "run_mode": self.mode,
                            "question": {**public_question(question), "split": question["split"],
                                         "event_id": question["event_id"], "cluster_id": question["cluster_id"]},
                            "split": question["split"],
                            "episode": episode, "events": self.events(episode["episode_id"]),
                            "model_turns": self.model_turns(episode["episode_id"]),
                            "resolution": resolution,
                            "resolved_at": resolution_row[1] if resolution_row else None,
                            "rollout": self.rollout(episode["episode_id"]),
                            "group_key": [question["question_id"], episode["policy_id"], episode["market_mode"], episode["research_mode"], episode["reward_mode"]]})
        return records

    def summary(self):
        """Question-weighted summaries, separated by system configuration and split."""
        self.expire()
        groups = {}
        for row in self.db.execute("SELECT episode_id FROM episodes ORDER BY created_at, episode_id"):
            episode = self.episode(row[0])
            question = self.question(episode["question_id"])
            key = (episode["policy_id"], episode["track"], episode["market_mode"], episode["research_mode"], question["split"], episode["reward_mode"])
            group = groups.setdefault(key, {"policy_id": key[0], "track": key[1], "market_mode": key[2],
                                          "research_mode": key[3], "split": key[4], "reward_mode": key[5], "status_counts": {},
                                          "_scores": {}, "_clusters": set(), "_events": set()})
            status = episode["status"]
            group["status_counts"][status] = group["status_counts"].get(status, 0) + 1
            resolution = self.db.execute("SELECT payload FROM resolutions WHERE question_id=?", (question["question_id"],)).fetchone()
            resolved = resolution and json.loads(resolution[0])["status"] == "resolved"
            if status == "graded" or (status in {"invalid", "missed"} and resolved):
                loss = episode["scores"]["brier"] if status == "graded" else 1.0
                group["_scores"].setdefault(question["question_id"], []).append(loss)
                group["_clusters"].add(question["cluster_id"])
                group["_events"].add(question["event_id"])
        output = []
        for group in groups.values():
            scores = group.pop("_scores")
            group["n_scored_questions"] = len(scores)
            group["n_scored_rollouts"] = sum(map(len, scores.values()))
            group["n_scored_clusters"] = len(group.pop("_clusters"))
            group["n_scored_events"] = len(group.pop("_events"))
            group["brier_penalized_question_mean"] = sum(sum(v) / len(v) for v in scores.values()) / len(scores) if scores else None
            output.append(group)
        return {"run_mode": self.mode, "official_leaderboard": False,
                "aggregation": "equal weight per assigned question within each configuration and split; invalid/missed=1 only on resolved questions",
                "groups": output}
