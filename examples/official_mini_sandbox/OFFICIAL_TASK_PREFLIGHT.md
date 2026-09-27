# SWE-bench Verified task preflight

The pinned official checkout contains the public SWE-bench Verified instance
`pallets__flask-5014`, based on Flask commit
`7ee9ceb71e868944a46e1ff00b506772a53a4f1d`. Its problem asks for an
error when a Blueprint has an empty name. This is a small real task, not a
substitute fixture. The checker intentionally omits the private test patch and
all test contents from its output.

`task_asset_preflight.py` checks the exact upstream Git revision, the local
Python source-tree digest, parquet row, official terminal bundle, and three
local prerequisites for an
**offline prepared-task run**:

```sh
python3 -m examples.official_mini_sandbox.task_asset_preflight \
  --upstream-root runs/upstream-swe-mini-381ada53 \
  --cache-root runs/official-mini-task-cache
```

It requires `pyarrow` only for reading the official task parquet. Paths in the
result are relative to `--cache-root`. The expected cache layout follows the
pinned upstream `SandboxDeployment.cached_git_dir` and `build_env` code. If a
cache is absent, upstream preparation would need the Flask Git history,
Python 3.11, and dependency installation. At the time of this historical
preflight, the local Ubuntu image had Python 3.12 and none of those caches
existed. Presence of the three files alone does not prove that they are valid or
that the environment can grade. An absent cache must not be interpreted as a
failed agent patch or given a zero reward.

The exact public Flask base commit was shallow-fetched into the ignored local
cache at `runs/official-mini-task-cache/source/flask`, verified by `git
rev-parse HEAD`, and occupied 3.0 MB on this host. This source checkout
contains no SWE-bench test patch and is distinct from the upstream's prepared
`testbed.tar.gz` and Python environment caches. It can be reproduced with:

```sh
git init runs/official-mini-task-cache/source/flask
git -C runs/official-mini-task-cache/source/flask fetch --depth=1 \
  --filter=blob:none https://github.com/pallets/flask.git \
  7ee9ceb71e868944a46e1ff00b506772a53a4f1d
git -C runs/official-mini-task-cache/source/flask checkout --detach FETCH_HEAD
git -C runs/official-mini-task-cache/source/flask rev-parse HEAD
```

The [upstream cache-preparation guide](https://github.com/lblankl/SWE-MiniSandbox/blob/381ada53ab35dadb342add33ff006f3157c22fb7/docs/guide/data/swe-bench.md)
uses a Conda installation with Python 3.11 for Flask 2.3, prepares a repository
cache and venv, and then evaluates patch submissions through SWE-Agent. Its
documented `--instances.filter` option can restrict preparation to
`^pallets__flask-5014$`. We subsequently built a disposable Debian-based
Python 3.11 image and completed one real task grade with an operator-prepared
cache. See [the task runbook](OFFICIAL_TASK_RUN.md) for the exact scope and
measurement.

The live integration keeps a host-owned dataset and verifier, mounts only the
actor checkout into the actor's namespace, terminates actor access before
applying the test patch, and grades in a separate trusted environment. The
pinned upstream `_calculate_reward` catches exceptions and
converts them to `0.0`; our trusted integration must instead treat setup and
test infrastructure errors as **ungraded**. Its `_calculate_reward_swebench`
also has an explicit instance skip list, so any selected task must be checked
against it before a grade is accepted.

`strict_grade.py` makes that correctness transfer concrete for this one
official task. It calls `_calculate_reward_swebench` directly in a trusted
verifier, rejects skip-listed tasks, requires every expected test to appear in
the parsed output, and raises `AdapterInfrastructureError` when evaluation
cannot be trusted. It was first tested with interface doubles and then against
the real Flask task: all 60 expected cases were observed, with baseline
reward 0 and official patch reward 1.

The [preflight result](../../docs/measurements/official_mini_swebench_feasibility_2026-09-25.json)
records the earlier missing-cache state. It is not a timing or reward
measurement. The later [single-task result](../../docs/measurements/official_mini_flask_5014_grade_2026-09-25.json)
records the actual grade and its limits.
