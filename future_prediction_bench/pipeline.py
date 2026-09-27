"""Automated discovery, immutable publication, and retryable source resolution."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

from .http import HttpClient, strict_json_loads
from .schema import parse_timestamp, validate_question
from .sources import SOURCE_REGISTRY, build_sources
from .store import canonical, digest


def load_config(path):
    config = strict_json_loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != "0.2":
        raise ValueError("Source config schema_version must be '0.2'")
    build_sources(config)
    settings = config.get("pipeline", {})
    for key, default, minimum, maximum in (("max_questions_per_source", 20, 1, 1000),
                                           ("retry_base_seconds", 900, 1, 86400),
                                           ("retry_max_seconds", 21600, 1, 604800),
                                           ("max_resolution_attempts", 20, 1, 1000)):
        value = settings.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
            raise ValueError(f"Invalid pipeline.{key}")
    if not isinstance(settings.get("auto_publish", True), bool):
        raise ValueError("pipeline.auto_publish must be boolean")
    return config


def semantic_question(question):
    value = json.loads(canonical(question))
    value.pop("issued_at", None)
    value.get("metadata", {}).pop("discovery_provenance", None)
    return value


class Pipeline:
    def __init__(self, store, config, *, client=None):
        self.store = store
        self.config = config
        self.sources = build_sources(config)
        self.settings = config.get("pipeline", {})
        self.client = client or HttpClient(self.settings.get("snapshot_dir", "runs/snapshots"))
        store.db.executescript("""
            CREATE TABLE IF NOT EXISTS source_candidates (
                question_id TEXT PRIMARY KEY, source_id TEXT NOT NULL, status TEXT NOT NULL,
                payload TEXT NOT NULL, error TEXT, first_seen TEXT NOT NULL, last_seen TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS resolution_jobs (
                question_id TEXT PRIMARY KEY REFERENCES questions, source_id TEXT NOT NULL,
                status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, next_attempt TEXT NOT NULL,
                last_error TEXT, last_proposal TEXT);
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL,
                started_at TEXT NOT NULL, finished_at TEXT NOT NULL, payload TEXT NOT NULL);
        """)
        # Reconcile durable publications after a crash, even if their forecast
        # windows have closed and discovery no longer returns those events.
        for row in store.db.execute("SELECT payload FROM questions").fetchall():
            question = json.loads(row[0])
            source_id = question.get("metadata", {}).get("source_id")
            if source_id in SOURCE_REGISTRY:
                self._ensure_job(question, source_id)

    def _record_run(self, kind, started, result):
        with self.store.db:
            self.store.db.execute("INSERT INTO pipeline_runs(kind,started_at,finished_at,payload) VALUES(?,?,?,?)",
                                  (kind, started.isoformat(), self.store.now().isoformat(), canonical(result)))
        return result

    def collect(self):
        started = self.store.now()
        report = {"published": [], "duplicates": [], "review": [], "source_errors": []}
        limit = self.settings.get("max_questions_per_source", 20)
        for adapter, source_config in self.sources:
            try:
                questions = adapter.discover(started, source_config, self.client)
            except Exception as exc:
                report["source_errors"].append({"source_id": adapter.source_id, "error": type(exc).__name__, "message": str(exc)[:300]})
                continue
            published_count = 0
            for question in questions:
                qid = question.get("question_id", "invalid-" + digest(question)[:16])
                try:
                    validate_question(question)
                    target = parse_timestamp(question["metadata"]["target_at"])
                    deadline = parse_timestamp(question["forecast_deadline"])
                    if not timedelta(0) < target - deadline <= timedelta(days=7):
                        raise ValueError("Target is outside the seven-day forecast horizon")
                    if deadline <= self.store.now():
                        raise ValueError("Forecast deadline is no longer in the future")
                    if question["metadata"].get("source_id") != adapter.source_id:
                        raise ValueError("Source identity mismatch")
                    old = self.store.db.execute("SELECT payload FROM questions WHERE question_id=?", (qid,)).fetchone()
                    if old:
                        frozen = json.loads(old[0])
                        self._ensure_job(frozen, adapter.source_id)
                        if semantic_question(frozen) != semantic_question(question):
                            raise ValueError("Published source event changed; review required")
                        self._candidate(frozen, adapter.source_id, "published", None)
                        report["duplicates"].append(qid)
                        continue
                    if published_count >= limit:
                        continue
                    if not self.settings.get("auto_publish", True):
                        raise ValueError("Automatic publication disabled; candidate requires review")
                    self.store.add_question(question)
                    self._ensure_job(question, adapter.source_id)
                    self._candidate(question, adapter.source_id, "published", None)
                    report["published"].append(qid)
                    published_count += 1
                except (ValueError, KeyError, TypeError) as exc:
                    self._candidate(question, adapter.source_id, "needs_review", str(exc))
                    report["review"].append({"question_id": qid, "reason": str(exc)})
        return self._record_run("collect", started, report)

    def _ensure_job(self, question, source_id):
        with self.store.db:
            self.store.db.execute("INSERT OR IGNORE INTO resolution_jobs VALUES (?,?,'pending',0,?,NULL,NULL)",
                                  (question["question_id"], source_id, question["resolve_after"]))

    def _candidate(self, question, source_id, status, error):
        now = self.store.now().isoformat()
        qid = question.get("question_id", "invalid-" + digest(question)[:16])
        with self.store.db:
            self.store.db.execute("""INSERT INTO source_candidates VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(question_id) DO UPDATE SET status=excluded.status,payload=excluded.payload,
                error=excluded.error,last_seen=excluded.last_seen""",
                (qid, source_id, status, canonical(question), error, now, now))

    def resolve_due(self, *, limit=100):
        started = self.store.now()
        self.store.expire()
        report = {"resolved": [], "void": [], "pending": [], "review": [], "errors": []}
        jobs = self.store.db.execute("SELECT * FROM resolution_jobs WHERE status='pending' ORDER BY next_attempt,question_id").fetchall()
        due = [job for job in jobs if parse_timestamp(job["next_attempt"]) <= started][:limit]
        for job in due:
            qid = job["question_id"]
            proposal = json.loads(job["last_proposal"]) if job["last_proposal"] else None
            if self.store.db.execute("SELECT 1 FROM resolutions WHERE question_id=?", (qid,)).fetchone():
                with self.store.db:
                    self.store.db.execute("UPDATE resolution_jobs SET status='complete' WHERE question_id=?", (qid,))
                continue
            try:
                adapter_class = SOURCE_REGISTRY[job["source_id"]]
                if not isinstance(proposal, dict) or proposal.get("status") not in {"resolved", "void"}:
                    proposal = adapter_class().resolve(self.store.question(qid), started, self.client)
                status = proposal.get("status")
                if status in {"resolved", "void"}:
                    # Write evidence first. Recovery must reuse this snapshot,
                    # especially for catalogs whose historical entries can change.
                    with self.store.db:
                        self.store.db.execute("UPDATE resolution_jobs SET last_proposal=? WHERE question_id=?", (canonical(proposal), qid))
                    self.store.resolve(qid, status=status, outcome=proposal.get("outcome"),
                                       evidence_urls=proposal["evidence_urls"], evidence_text=proposal["evidence_text"])
                    with self.store.db:
                        self.store.db.execute("UPDATE resolution_jobs SET status='complete',attempts=attempts+1,last_proposal=?,last_error=NULL WHERE question_id=?", (canonical(proposal), qid))
                    report[status].append(qid)
                    continue
                if status not in {"pending", "needs_review"}:
                    raise ValueError("Resolver returned an invalid status")
                reason = proposal.get("reason", status)
            except Exception as exc:
                status, reason = "pending", f"{type(exc).__name__}: {str(exc)[:250]}"
                report["errors"].append({"question_id": qid, "reason": reason})
            attempts = job["attempts"] + 1
            if attempts >= self.settings.get("max_resolution_attempts", 20):
                status, reason = "needs_review", "Retry limit reached: " + reason
            delay = min(self.settings.get("retry_base_seconds", 900) * 2 ** min(attempts - 1, 20),
                        self.settings.get("retry_max_seconds", 21600))
            with self.store.db:
                self.store.db.execute("""UPDATE resolution_jobs SET status=?,attempts=?,next_attempt=?,last_error=?,last_proposal=? WHERE question_id=?""",
                                      (status, attempts, (started + timedelta(seconds=delay)).isoformat(), reason,
                                       canonical(proposal) if proposal is not None else None, qid))
            report["review" if status == "needs_review" else "pending"].append(qid)
        return self._record_run("resolve_due", started, report)

    def retry(self, question_id):
        if self.store.db.execute("SELECT 1 FROM resolutions WHERE question_id=?", (question_id,)).fetchone():
            raise ValueError("A terminal resolution cannot be retried")
        with self.store.db:
            cursor = self.store.db.execute("UPDATE resolution_jobs SET status='pending',attempts=0,next_attempt=?,last_error=NULL WHERE question_id=?", (self.store.now().isoformat(), question_id))
        if not cursor.rowcount:
            raise ValueError("Unknown resolution job")
        return {"question_id": question_id, "status": "pending"}

    def cycle(self):
        return {"resolution": self.resolve_due(), "collection": self.collect(), "status": self.status()}

    def status(self):
        return {"questions": self.store.db.execute("SELECT COUNT(*) FROM questions").fetchone()[0],
                "candidates": {row[0]: row[1] for row in self.store.db.execute("SELECT status,COUNT(*) FROM source_candidates GROUP BY status")},
                "resolution_jobs": {row[0]: row[1] for row in self.store.db.execute("SELECT status,COUNT(*) FROM resolution_jobs GROUP BY status")},
                "run_mode": self.store.mode}
