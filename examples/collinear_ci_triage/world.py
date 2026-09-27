"""Fresh, stateful CI task worlds with a Collinear-inspired tool interface.

This is a local simulation, not Collinear Simulation Lab, a Docker sandbox, or
an untrusted-code execution boundary. Agent-facing calls never receive the
host verifier's target state or reward.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path


TARGET_RUN = 1842
REPO = "northstar/api"
BRANCH = "release/2.4"
JOB = "linux-integration"
COMMIT = "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678"
TARGET_TIMEOUT_MS = 120_000
HISTORY_ROWS = 300
TASK_PROMPT = (
    "Investigate failed CI run 1842 for northstar/api release/2.4. "
    "If the job timeout caused the failure, set only that job's timeout to "
    "120000 ms and queue a rerun of the same commit. A note alone is not a fix."
)

TABLE_KEYS = {"ci_runs": ("run_id",),
              "job_config": ("repo", "branch", "job_name"),
              "reruns": ("rerun_id",), "triage_notes": ("note_id",),
              "meta": ("key",)}

SCHEMA = """
CREATE TABLE ci_runs (
  run_id INTEGER PRIMARY KEY, repo TEXT NOT NULL, branch TEXT NOT NULL,
  commit_sha TEXT NOT NULL, job_name TEXT NOT NULL, status TEXT NOT NULL,
  failure_reason TEXT NOT NULL, observed_duration_ms INTEGER NOT NULL
);
CREATE TABLE job_config (
  repo TEXT NOT NULL, branch TEXT NOT NULL, job_name TEXT NOT NULL,
  timeout_ms INTEGER NOT NULL, updated_seq INTEGER NOT NULL,
  PRIMARY KEY (repo, branch, job_name)
);
CREATE TABLE reruns (
  rerun_id INTEGER PRIMARY KEY AUTOINCREMENT, source_run_id INTEGER NOT NULL,
  repo TEXT NOT NULL, branch TEXT NOT NULL, commit_sha TEXT NOT NULL,
  job_name TEXT NOT NULL, timeout_ms INTEGER NOT NULL, status TEXT NOT NULL,
  created_seq INTEGER NOT NULL
);
CREATE TABLE triage_notes (
  note_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER NOT NULL,
  body TEXT NOT NULL, created_seq INTEGER NOT NULL
);
CREATE TABLE meta (key TEXT PRIMARY KEY, value INTEGER NOT NULL);
"""

TOOL_SPECS = (
    {"name": "inspect_failure", "description": "Read a CI run and its current job timeout.",
     "parameters": {"type": "object", "properties": {"run_id": {"type": "integer"}},
                    "required": ["run_id"], "additionalProperties": False}},
    {"name": "update_job_timeout", "description": "Set the timeout for a run's repository, branch, and job.",
     "parameters": {"type": "object", "properties": {
         "run_id": {"type": "integer"}, "timeout_ms": {"type": "integer"}},
                    "required": ["run_id", "timeout_ms"], "additionalProperties": False}},
    {"name": "queue_rerun", "description": "Queue a CI rerun using the current job configuration.",
     "parameters": {"type": "object", "properties": {"run_id": {"type": "integer"}},
                    "required": ["run_id"], "additionalProperties": False}},
    {"name": "post_triage_note", "description": "Record an analyst note; this does not change CI configuration.",
     "parameters": {"type": "object", "properties": {
         "run_id": {"type": "integer"}, "body": {"type": "string"}},
                    "required": ["run_id", "body"], "additionalProperties": False}},
)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _seed_rows():
    rows = []
    for offset in range(HISTORY_ROWS):
        rows.append((1000 + offset, REPO, BRANCH,
                     f"{offset:040x}", JOB,
                     "failed" if offset % 17 == 0 else "passed",
                     "assertion_failure" if offset % 17 == 0 else "",
                     20_000 + (offset * 97) % 35_000))
    rows.append((TARGET_RUN, REPO, BRANCH, COMMIT, JOB, "failed",
                 "tests/test_async_export.py exceeded 60000 ms; last observed 87000 ms",
                 87_000))
    rows.append((1843, REPO, "main", "f" * 40, "linux-unit", "passed", "", 32_000))
    return rows


def build_seed(path):
    """Create one deterministic on-disk application seed, not a VM image."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    connection = sqlite3.connect(path)
    try:
        connection.executescript(SCHEMA)
        connection.executemany("INSERT INTO ci_runs VALUES (?,?,?,?,?,?,?,?)", _seed_rows())
        connection.executemany("INSERT INTO job_config VALUES (?,?,?,?,?)", (
            (REPO, BRANCH, JOB, 60_000, 0),
            (REPO, "main", "linux-unit", 90_000, 0),
        ))
        connection.execute("INSERT INTO meta VALUES ('mutation_seq',0)")
        connection.commit()
    finally:
        connection.close()


def _snapshot(connection):
    state = {}
    for table, keys in TABLE_KEYS.items():
        order = ",".join(keys)
        state[table] = [dict(row) for row in connection.execute(
            f"SELECT * FROM {table} ORDER BY {order}")]
    return state


def state_diff(before, after):
    """Structured application rows, not a filesystem or process-state diff."""
    changes = []
    for table, keys in TABLE_KEYS.items():
        old = {tuple(row[key] for key in keys): row for row in before[table]}
        new = {tuple(row[key] for key in keys): row for row in after[table]}
        for key in sorted(set(old) | set(new)):
            if old.get(key) != new.get(key):
                changes.append({"table": table, "key": list(key),
                                "before": old.get(key), "after": new.get(key)})
    return changes


class SeedArtifact:
    """Operator-owned seed used to create independent SQLite files."""

    def __init__(self, directory):
        self.path = Path(directory) / "ci-seed.sqlite3"
        started = time.perf_counter()
        build_seed(self.path)
        self.build_seconds = time.perf_counter() - started
        with sqlite3.connect(self.path) as connection:
            connection.row_factory = sqlite3.Row
            self.initial_state = _snapshot(connection)
        self.initial_digest = digest(self.initial_state)

    def new_world(self, mode, *, directory=None):
        if mode not in {"sql_reseed", "seed_copy"}:
            raise ValueError("mode must be sql_reseed or seed_copy")
        started = time.perf_counter()
        temp = tempfile.TemporaryDirectory(prefix="fpb-ci-rollout-", dir=directory)
        database = Path(temp.name) / "state.sqlite3"
        try:
            if mode == "sql_reseed":
                build_seed(database)
            else:
                shutil.copyfile(self.path, database)
            world = CITriageWorld(temp, database, self.initial_digest,
                                  setup_seconds=0.0)
            world.setup_seconds = time.perf_counter() - started
            if world.initial_digest != self.initial_digest:
                raise RuntimeError("seed materialization changed application state")
            return world
        except BaseException:
            temp.cleanup()
            raise


class CITriageWorld:
    """One fresh rollout. The SQLite connection and its directory are private."""

    MAX_STEPS = 16

    def __init__(self, temp, database, expected_seed_digest, *, setup_seconds):
        self._temp = temp
        self.database_path = database
        self._connection = sqlite3.connect(database)
        self._connection.row_factory = sqlite3.Row
        self.closed = False
        self.initial_digest = digest(_snapshot(self._connection))
        if self.initial_digest != expected_seed_digest:
            self.close()
            raise RuntimeError("rollout seed digest mismatch")
        self.setup_seconds = setup_seconds
        self.trace = []
        self._step_attempts = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if not self.closed:
            self.closed = True
            self._connection.close()
            self._temp.cleanup()

    def state(self):
        if self.closed:
            raise RuntimeError("rollout_closed")
        return _snapshot(self._connection)

    def state_digest(self):
        return digest(self.state())

    def _run(self, run_id):
        if type(run_id) is not int or run_id < 1:
            raise ValueError("run_id must be a positive integer")
        row = self._connection.execute("SELECT * FROM ci_runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise ValueError("CI run not found")
        return dict(row)

    def _config(self, run):
        row = self._connection.execute(
            "SELECT * FROM job_config WHERE repo=? AND branch=? AND job_name=?",
            (run["repo"], run["branch"], run["job_name"])).fetchone()
        if row is None:
            raise ValueError("CI job configuration not found")
        return dict(row)

    def _next_seq(self):
        self._connection.execute("UPDATE meta SET value=value+1 WHERE key='mutation_seq'")
        return self._connection.execute(
            "SELECT value FROM meta WHERE key='mutation_seq'").fetchone()[0]

    def _execute(self, name, parameters):
        if name == "inspect_failure":
            if set(parameters) != {"run_id"}:
                raise ValueError("inspect_failure needs run_id")
            run = self._run(parameters["run_id"])
            return {"run": run, "job_config": self._config(run)}
        if name == "update_job_timeout":
            if set(parameters) != {"run_id", "timeout_ms"}:
                raise ValueError("update_job_timeout needs run_id and timeout_ms")
            run = self._run(parameters["run_id"])
            timeout = parameters["timeout_ms"]
            if type(timeout) is not int or not 30_000 <= timeout <= 300_000:
                raise ValueError("timeout_ms must be 30000..300000")
            self._config(run)
            sequence = self._next_seq()
            self._connection.execute(
                "UPDATE job_config SET timeout_ms=?,updated_seq=? "
                "WHERE repo=? AND branch=? AND job_name=?",
                (timeout, sequence, run["repo"], run["branch"], run["job_name"]))
            return {"updated": True, "repo": run["repo"], "branch": run["branch"],
                    "job_name": run["job_name"], "timeout_ms": timeout}
        if name == "queue_rerun":
            if set(parameters) != {"run_id"}:
                raise ValueError("queue_rerun needs run_id")
            run = self._run(parameters["run_id"])
            config = self._config(run)
            sequence = self._next_seq()
            status = ("passed" if config["timeout_ms"] >= run["observed_duration_ms"] + 10_000
                      else "failed")
            cursor = self._connection.execute(
                "INSERT INTO reruns (source_run_id,repo,branch,commit_sha,job_name,"
                "timeout_ms,status,created_seq) VALUES (?,?,?,?,?,?,?,?)",
                (run["run_id"], run["repo"], run["branch"], run["commit_sha"],
                 run["job_name"], config["timeout_ms"], status, sequence))
            return {"rerun_id": cursor.lastrowid, "status": status,
                    "source_run_id": run["run_id"]}
        if name == "post_triage_note":
            if set(parameters) != {"run_id", "body"}:
                raise ValueError("post_triage_note needs run_id and body")
            run = self._run(parameters["run_id"])
            body = parameters["body"]
            if not isinstance(body, str) or not 1 <= len(body) <= 1000:
                raise ValueError("body must contain 1..1000 characters")
            sequence = self._next_seq()
            cursor = self._connection.execute(
                "INSERT INTO triage_notes (run_id,body,created_seq) VALUES (?,?,?)",
                (run["run_id"], body, sequence))
            return {"note_id": cursor.lastrowid, "recorded": True}
        raise ValueError("unknown tool")

    def _reject_step(self, reason, started):
        """Charge and trace a malformed step without persisting any change."""
        current = self.state_digest()
        response = {"observation": "Error: " + reason, "is_error": True}
        self.trace.append({"action": {"tool_name": "_invalid_action",
                                      "parameters": {"reason": reason}},
                           "response": copy.deepcopy(response),
                           "before_digest": current, "after_digest": current,
                           "state_diff": [],
                           "step_seconds": time.perf_counter() - started})
        return response

    def handle(self, method, path, body=None):
        """In-process GET /tools and POST /step shapes, without an HTTP server."""
        if self.closed:
            raise RuntimeError("rollout_closed")
        if method == "GET" and path == "/tools" and body is None:
            return {"tools": copy.deepcopy(list(TOOL_SPECS))}
        if method != "POST" or path != "/step":
            return {"observation": "Error: unsupported endpoint", "is_error": True}
        if self._step_attempts >= self.MAX_STEPS:
            return {"observation": "Error: step budget exceeded", "is_error": True}
        self._step_attempts += 1
        started = time.perf_counter()
        if (not isinstance(body, dict) or set(body) != {"action"}
                or not isinstance(body["action"], dict)
                or set(body["action"]) != {"tool_name", "parameters"}
                or not isinstance(body["action"]["tool_name"], str)
                or not isinstance(body["action"]["parameters"], dict)):
            return self._reject_step("invalid action envelope", started)
        action = body["action"]
        try:
            action_size = len(_canonical(action))
        except (TypeError, ValueError, UnicodeError):
            return self._reject_step("invalid JSON action", started)
        if action_size > 4096:
            return self._reject_step("action exceeds bound", started)
        before = self.state()
        before_digest = digest(before)
        try:
            with self._connection:
                observation = self._execute(action["tool_name"], action["parameters"])
            response = {"observation": observation}
        except ValueError as exc:
            response = {"observation": "Error: " + str(exc), "is_error": True}
        after = self.state()
        changes = state_diff(before, after)
        elapsed = time.perf_counter() - started
        self.trace.append({"action": copy.deepcopy(action),
                           "response": copy.deepcopy(response),
                           "before_digest": before_digest,
                           "after_digest": digest(after),
                           "state_diff": changes,
                           "step_seconds": elapsed})
        return response


class HostVerifier:
    """Programmatic grade from application state, never agent-authored text."""

    def __init__(self, seed):
        self.seed_digest = seed.initial_digest
        self._seed = copy.deepcopy(seed.initial_state)

    def verify(self, world):
        final = world.state()
        target = copy.deepcopy(self._seed)
        config = next(row for row in target["job_config"] if
                      (row["repo"], row["branch"], row["job_name"]) == (REPO, BRANCH, JOB))
        config["timeout_ms"] = TARGET_TIMEOUT_MS
        config["updated_seq"] = 1
        target["meta"][0]["value"] = 2
        target["reruns"] = [{"rerun_id": 1, "source_run_id": TARGET_RUN,
                            "repo": REPO, "branch": BRANCH, "commit_sha": COMMIT,
                            "job_name": JOB, "timeout_ms": TARGET_TIMEOUT_MS,
                            "status": "passed", "created_seq": 2}]
        chain_valid = (world.initial_digest == self.seed_digest
                       and (not world.trace or world.trace[0]["before_digest"] == self.seed_digest)
                       and all(a["after_digest"] == b["before_digest"] for a, b in
                               zip(world.trace, world.trace[1:]))
                       and (not world.trace or world.trace[-1]["after_digest"] == digest(final)))
        checks = {"seed_bound": world.initial_digest == self.seed_digest,
                  "state_exact": final == target,
                  "trace_chain": chain_valid}
        return {"status": "resolved", "reward": 1.0 if all(checks.values()) else 0.0,
                "checks": checks, "seed_digest": self.seed_digest,
                "final_digest": digest(final),
                "application_diff_digest": digest(state_diff(self._seed, final))}


def step(world, tool_name, **parameters):
    return world.handle("POST", "/step", {"action": {
        "tool_name": tool_name, "parameters": parameters}})


def solved_trace(world, *, discovered=None):
    """Scripted operator reference, not a model rollout."""
    specs = discovered if discovered is not None else world.handle("GET", "/tools")["tools"]
    names = {item["name"] for item in specs}
    if not {"inspect_failure", "update_job_timeout", "queue_rerun"} <= names:
        raise RuntimeError("tool discovery failed")
    observations = [step(world, "inspect_failure", run_id=TARGET_RUN),
                    step(world, "update_job_timeout", run_id=TARGET_RUN,
                         timeout_ms=TARGET_TIMEOUT_MS),
                    step(world, "queue_rerun", run_id=TARGET_RUN)]
    if any(value.get("is_error") for value in observations):
        raise RuntimeError("reference trace failed")
    return observations
