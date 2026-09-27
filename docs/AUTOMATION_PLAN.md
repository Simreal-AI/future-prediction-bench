# Automation guide

The automation pipeline pairs event discovery with an exact result parser for each supported source template. It creates future questions, preserves evidence, resolves due events, and retains pending/review state across restarts. Forecasting-model calls and private-baseline generation are separate stages so the data-production service does not become an agent evidence channel.

## End-to-end lifecycle

```text
Source configuration and versioned templates
    -> discover candidate events and preserve source snapshots
    -> validate dates, options, source identity, and seven-day horizon
    -> deduplicate and freeze published questions
    -> generate and seal a private baseline before the deadline
    -> each forecasting system researches and submits probabilities
    -> fetch authoritative results after the resolution window opens
    -> resolved: score and fill delayed reward
    -> pending: retry with backoff
    -> ambiguous or exhausted retries: retain for review
```

The initial implementations and source-specific semantics are in [SOURCES.md](SOURCES.md). They do not make arbitrary news pages automatically resolvable. Review a new template's rules and representative cases before enabling automatic publication.

## Commands

Run from the repository root after the [quick start](../README.md#quick-start):

```bash
python -m future_prediction_bench collect --config configs/sources.json --db runs/live.sqlite
python -m future_prediction_bench resolve-due --config configs/sources.json --db runs/live.sqlite
python -m future_prediction_bench cycle --config configs/sources.json --db runs/live.sqlite
python -m future_prediction_bench export-questions --db runs/live.sqlite --output runs/questions.public.jsonl
```

`cycle` resolves currently due jobs and collects eligible new questions. Use a foreground worker for repeated cycles:

```bash
python -m future_prediction_bench worker \
  --config configs/sources.json --db runs/live.sqlite --interval-seconds 3600
```

Add `--max-cycles N` for a bounded run. The command does not install a system service. A deployment can supervise this foreground process and persist the state directory; keep one worker per state database.

Collection and resolution do not require model/search credentials. The optional forecasting stage does:

```bash
python -m future_prediction_bench seal-baselines \
  --db runs/live.sqlite --baseline-config configs/baselines.local.json
python -m future_prediction_bench forecast \
  --db runs/live.sqlite --limit 20 --baseline-config configs/baselines.local.json
python -m future_prediction_bench cycle \
  --config configs/sources.json --db runs/live.sqlite --with-forecasts
```

The baseline config is optional. Copy [baselines.example.json](../configs/baselines.example.json) to a local ignored file before adding reviewed mappings; its initial active mapping object is empty. Separate baseline model credentials enable the fixed-model fallback. `forecast` attempts baseline sealing itself, so `seal-baselines` is useful for a separate earlier stage rather than a required duplicate invocation.

`--with-forecasts` also works with `worker` and explicitly enables model calls. The runner defaults to baseline-improvement reward and skips questions without a valid sealed reference. An explicit `--reward-mode negative_brier` run uses the raw-outcome reward. See [`.env.example`](../.env.example) and [live providers](ADAPTERS.md#included-live-providers) for configuration and limits.

After reviewing a nonterminal resolution failure, requeue it with:

```bash
python -m future_prediction_bench retry-resolution \
  --config configs/sources.json --db runs/live.sqlite --question-id QUESTION_ID
```

Use `status --db runs/live.sqlite` to inspect stored run summaries. Command reports also show new publications, duplicate IDs, pending jobs, review cases, and source errors. A terminal resolution cannot be retried to overwrite its outcome. For a reviewed final outcome, use the administrative `resolve` command described in the README.

## Collection behavior

The source configuration identifies supported adapters and their parameters. `pipeline.auto_publish` controls automatic publication; when disabled, eligible candidates are retained for review. `max_questions_per_source` caps newly published questions per source per collection cycle.

The pipeline validates the question schema, source identity, a future forecasting deadline, and `0 < metadata.target_at - forecast_deadline <= 168 hours`. Existing questions are deduplicated by stable ID. A material source change to an already published question is retained for review instead of rewriting its original rules. Publication time and discovery provenance may vary across repeated fetches without creating a new event.

Question options, target, time window, units, thresholds, source rules, template version, and split must remain fixed after publication. Stable event/cluster identities prevent variants from crossing dataset splits. Default example sources target test questions; a distinct train configuration is needed for train-only RL episodes.

The generic question validator checks structural validity. It cannot prove that free-text options are exhaustive or that an arbitrary resolution rule is semantically sound. Automated admission relies on reviewed deterministic templates.

## Resolution, retries, and review

Each published source question has a persistent resolution job. A job becomes due at its frozen `resolve_after` time. The resolver uses question metadata and original rules, not a new template chosen after the outcome.

A resolver can return `resolved`, `void`, `pending`, or `needs_review`. Valid outcomes flow through the same idempotent `Store.resolve` operation used by manual adjudication. Only registered option IDs are accepted, evidence must include a preregistered source host, and conflicting terminal outcomes are rejected.

Pending results and fetch errors receive exponential retry delays, capped by `retry_max_seconds`. `retry_base_seconds` controls the initial delay. Reaching `max_resolution_attempts` moves a job to review; it does not fabricate a No, force a void, or assign zero reward. A reviewed nonterminal job may be requeued after the source problem is understood.

Inspect pending and review counts alongside scores. Examples include a postponed game, a source record that changed identity, a missing final result, a partial response, or a source error. Review should follow the original criteria; if no valid result can be obtained, document a void resolution with evidence.

## Snapshots and restart behavior

Source fetching uses an HTTPS host allowlist, request/response limits, bounded network retries, and raw JSON snapshots keyed by SHA-256. Snapshot records include source URL, actual observation time, content hash, byte count, and local path. Keep the database and snapshot directory together in persistent private storage.

SQLite also retains candidate states, resolution jobs and attempt counts, and pipeline run reports. Reusing the same configuration and database makes collection repeatable and preserves due work. Do not replace the state database with an empty file each day or treat a previous file's existence as proof that a cycle succeeded.

Operational snapshots may have separate retention or redistribution constraints. They stay outside version control by default. Public question exports should omit private baselines; public data releases additionally require source-data review and a frozen study description.

## Private baseline stage

Before the forecast deadline, look for a reviewed exact market mapping. The contract must match event identity, option semantics, timing, and resolution criteria; a similar title is insufficient. Preserve the market observation time and contract mapping. If no eligible market record is available, the configured fixed internal model may supply a probability distribution.

Seal only a baseline that was actually produced in time and passed probability validation. The store also requires sealing before any evaluated episode for that question starts. Missing or failed baseline attempts stay missing. Baseline probabilities, model inputs, and raw source responses are private scoring records and must not be inserted into forecast-agent prompts or observations. The baseline-improvement reward cannot be calculated without a valid frozen reference. See [project design](PROJECT_DESIGN.md#private-baselines-and-rewards).

## Operating an unattended deployment

A complete deployment needs a persistent database and snapshot volume, one scheduled worker for that state, source/model credentials where applicable, bounded model budgets, and failure reporting. Source collection is useful without model credentials, but a complete live experiment also needs pre-deadline assignments and actual model submissions.

Review the exact CLI/configuration examples in the [README](../README.md) and use `python -m future_prediction_bench --help` for available commands. Source failures and review queues must remain visible to the operator; repeatedly running an empty or failed collection is not evidence of a healthy service.

The repository's CI workflow uses offline tests and synthetic data. It does not activate a live schedule, incur model costs, update model parameters, publish operational snapshots, or create an external repository. Before a public or persistent deployment, complete the [release checklist](RELEASE_CHECKLIST.md).

## Pilot acceptance

Validate at least two full real collection/resolution cohorts before treating a source as operationally established. Check that:

- New questions were genuinely unresolved at their deadlines and all generated text is English.
- Repeated collection does not duplicate or mutate published questions.
- Source evidence, question IDs, frozen rules, and timestamps can be traced end to end.
- Actual model forecasts and private baselines were submitted before their deadlines.
- Source failures remain pending or reviewable and never become fabricated outcomes.
- Final source results map to the correct frozen options and produce the expected scores/rewards.
- Train/test isolation, baseline privacy, coverage, unresolved/void rates, and operating costs are visible.

The first 30-50 events are a workflow pilot. Add domains and formal comparative statistics only after confirming source reliability and adequate independent event coverage.
