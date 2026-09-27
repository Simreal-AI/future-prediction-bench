# Project design

Future Prediction Bench has two uses of a shared event stream: evaluating an agent's probability forecasts after autonomous research, and collecting delayed outcome rewards for a future RL trainer. The first release targets English, multi-domain questions with a forecasting horizon of at most seven days.

The [README](../README.md) describes executable capabilities. This document also specifies requirements for future experiments; a requirement here does not imply that a hosted benchmark or parameter-training system already exists.

## Shared forecasting protocol

| Property | Design |
| --- | --- |
| Question types | Binary or categorical; categorical outcomes are mutually exclusive and collectively exhaustive |
| Model output | A probability for every option, with no required winning-option selection |
| Forecast horizon | `metadata.target_at - forecast_deadline`, greater than zero and at most 168 hours for automated pilot admission |
| Primary metric | `Brier = 0.5 * sum((p_i - y_i)^2)` |
| Default information condition | `no_consensus`; report `market_aware` separately |
| Default live-runner reward | Private baseline improvement; low-level environment/demo retain negative Brier for compatibility |
| Initial study | Small multi-domain live cohorts, followed by comparable model runs |
| Project language | English documentation, generated questions, fixtures, prompts, and messages |

Categorical questions should initially use three to five options. Several options may not be true simultaneously: multi-label events require a separate protocol. Numeric outcomes can be converted into frozen, exhaustive, nonoverlapping bins. Units, boundaries, rounding, revisions, and exceptions must be defined before prediction.

```json
{"action": "submit", "probabilities": {"home": 0.48, "draw": 0.27, "away": 0.25}}
```

Probability keys must match option IDs exactly. Values must be finite numeric values in `[0, 1]`; booleans and numeric strings are invalid. Their sum must be within `1e-6` of one. Validation never silently normalizes or fills a missing option. Zero and one are allowed.

## Questions, provenance, and time

A question binds stable question/event/cluster IDs, a schema version, domain, split, prompt, option IDs, timezone-aware timestamps, resolution criteria, and preregistered source URLs. Automated sources also attach a template version, source identity, target time, and raw-snapshot provenance. The current question schema is `0.1`; the package version can evolve independently.

```text
issued_at < forecast_deadline < outcome_not_before <= resolve_after
```

`outcome_not_before` bounds the earliest moment the answer could become known. `metadata.target_at` identifies the target instant or end of the event window. `resolve_after` is the first planned resolution attempt, not a guarantee that data will be available.

For the pilot, report horizon buckets `(0, 24]`, `(24, 72]`, and `(72, 168]` hours. Resolution latency is measured separately from target time to observed settlement and may exceed seven days. A source outage, missing value, or late publication does not itself establish a No outcome.

Published question payloads and resolution rules are immutable. A correction needs explicit adjudication or a new question version; existing predictions remain bound to the original question. Same-event variants and related clusters cannot cross train/dev/test splits. Automated source defaults use the test split; a separate train configuration is needed for RL collection. A formal study should freeze temporal split/cohort rules before publishing questions.

SQLite records and content hashes support local auditing. They do not constitute a tamper-proof public submission server, signed timestamp service, or independently attested historical snapshot.

## Research benchmark

A formal benchmark run should:

1. Register the common question cohort, forecasting windows, models, prompts, tool versions, sampling parameters, and budgets.
2. Provide an unresolved question and its public resolution rules to each assigned system.
3. Let the model choose analyst-tool actions and then submit a complete probability distribution before the deadline. Probability drafts and evidence notes do not receive outcome rewards.
4. Persist the actual observations returned to that model, including time, URL, text, filtering, and errors.
5. Freeze one submission per question and system configuration.
6. Resolve the event from its frozen rules and report scores together with coverage and pending/void rates.

`PredictionEnv` implements the interaction and storage contract. The default `self_research` mode requires at least one successful search that returns permitted evidence. That requirement establishes tool use, not research quality. The explicit `no_search` ablation has a separate reporting group.

The core budget is a shared maximum number of analyst-tool calls. The included runner also enforces model-step, requested-output-token, prompt-size, and wall-time limits. Currency cost and distributed concurrency accounting remain future work. Model/provider versions and these budgets are part of the system definition; a base-model name alone is insufficient for comparison.

Benchmark episodes are unique per question, policy ID, market mode, research mode, and reward mode. RL can collect repeated train-only episodes. Current report denominators cover created episodes, so a formal benchmark still needs complete, preregistered assignments to prevent selective participation.

## Information conditions

`no_consensus` attempts to filter known market/odds sources and obvious consensus-probability text. `market_aware` permits those sources and records detected exposure. The heuristic filter can miss mirrors, paraphrases, new domains, and indirect references, and can reject legitimate analysis. Neither a clean-looking audit nor a Brier score proves independent reasoning.

Other public forecast products, such as weather forecasts, should have an explicit predeclared access policy. If allowed, the evaluated system includes the model's ability to use those forecasts. Market-aware comparisons need synchronized, matched market data and their own coverage reporting.

Research tool access and private baseline calculation are separate. Agent-facing tool observations must never contain a privately stored baseline merely because reward code uses it.

## Private baselines and rewards

A question may have a private reference distribution, chosen before its prediction deadline and before any evaluated episode starts:

1. Prefer a market distribution only when the event identity, option mapping, target window, forecast cutoff, and resolution criteria strictly match.
2. When no eligible market baseline is available, a fixed internal model can generate the distribution with a recorded model/prompt/configuration and cutoff.
3. If neither has actually produced a valid distribution in time, record the baseline as missing. Do not substitute post-outcome data or pretend a model call occurred.

A market title match is insufficient. Initial integrations should use reviewed explicit mappings; fuzzy search can propose candidates but must not silently authorize a match. Market probabilities must be observed before the deadline, with the mapping, timestamp, and provenance preserved. Freeze any transformation for prices, spreads, missing contracts, or non-unit sums in advance rather than applying arbitrary normalization after seeing outcomes.

The fallback internal model should be fixed within an experiment. Its input must contain only information available before the deadline, and its probabilities must be validated and sealed on the same schedule as forecasts. It is a reference system, not a substitute for a synchronized human baseline.

Private baseline records are administrative/scoring data. Do not expose them in `PredictionEnv.reset`, research observations, exported public questions, or model prompts. Baseline selection cannot depend on which system eventually wins.

For a valid resolved question:

```text
negative_brier reward = -Brier(agent)
baseline_improvement reward = Brier(baseline) - Brier(agent)
```

Positive improvement means the agent outscored the reference on that outcome. Copying the baseline exactly yields zero improvement. Both distributions share the same outcome and scoring rule. Always retain the agent's raw Brier score independently of reward mode. The live runner defaults to baseline improvement and skips question assignments without a valid frozen baseline. The low-level environment rejects creating an improvement-mode episode without one. Missing baselines must not silently switch reward formulas or become zero. Negative-Brier runs must select that mode explicitly at the live runner; the low-level environment/demo keep it as their compatibility default.

A baseline shared within a same-question rollout group adds the same constant to each negative-Brier reward. It changes the absolute reference point but not the within-group ordering or centered advantage. Any claim that this alone changes GRPO learning needs separate evidence.

## Delayed-feedback environment

```text
Publish train question -> research rollout -> freeze probabilities and observations
    -> pending outcome -> authoritative resolution -> score and fill reward
    -> export audited trajectories -> future tokenizer/trainer integration
```

| State | Meaning | Outcome reward |
| --- | --- | --- |
| `active` | Assigned and still researching | None |
| `pending_reward` | Valid submission waiting for an outcome | None |
| `graded` | Outcome resolved and scores calculated | Selected reward, when its prerequisites are available |
| `invalid` | Invalid probabilities or unmet research requirement | Immediate protocol penalty of -1, without a fabricated outcome |
| `missed` | Assigned but deadline passed without a submission | None; penalized task loss is 1 once the question validly resolves |
| `void` | No valid outcome under the original rules | None |

Cancellation can be an outcome only if the original options and criteria say so. Otherwise a canceled event may remain pending or become void. Identical resolution submissions are idempotent; conflicting resolutions cannot silently overwrite prior results. Resolver access is administrative and is absent from the forecasting action space.

Training exports are restricted to train-split RL trajectories. Test questions never enter training because they have resolved. A future experiment may define a new historical training dataset, but must not continue claiming that reused data are unseen test data from the original experiment.

Text trajectory export is not a ready-to-train GRPO batch. It lacks token IDs, tokenizer/chat-template provenance, token-level observation masks, and behavior-policy log probabilities. A trainer must implement those requirements, reference-policy/loss configuration, and a strategy for delayed-policy staleness. Tool observations must not receive policy-gradient loss as if the model generated them.

Multiple rollouts share one event outcome. They can provide within-question policy variation but are not independent events. Near-identical rewards also offer weak within-group learning signals. Inspect event diversity, reward distributions, checkpoint lag, and group variation before expanding training.

## Automated event production

The collector and resolver are paired per source template. Collection discovers future events, preserves raw evidence, generates a frozen question, validates time and structure, deduplicates IDs, and publishes eligible records. Resolution fetches the specified source, checks the original event identity and rules, and returns a result, a pending reason, or an exception for review.

The initial adapter set and exact semantics appear in [SOURCES.md](SOURCES.md). The [automation guide](AUTOMATION_PLAN.md) explains execution and recovery. A small number of stable templates is preferable to treating arbitrary scraped news as automatically resolvable questions.

The pilot aims for 30-50 real events spanning multiple domains over short feedback cycles. Current adapter availability may cover fewer domains than the target study; expand coverage before making broad cross-domain claims. Weather, sports, public economic releases, and space/technology are possible extensions, subject to precise source rules and live verification. Domain weights in `pilot_config.json` are a planning proposal rather than an enforced experiment policy.

Preserve all published questions, including unresolved and void cases. Reporting only convenient resolved cases conceals resolution bias. Keep raw operational payloads private by default and review each source's retention and redistribution terms before a public data release.

## Scores and comparative reporting

Normalized Brier lies in `[0, 1]`; for binary probabilities it equals the conventional `(p - y)^2`. A uniform forecast for K outcomes has loss `0.5 * (1 - 1/K)`, so a common score range does not make all option counts equally difficult.

The auxiliary `clipped_log_loss` is `-log(max(p_true, 1e-15))`, with the epsilon explicitly reported. It is not raw log loss, which is infinite when the true option receives zero probability. Calibration plots and formal inferential reports remain future work.

Current summaries separate policy/configuration, benchmark/RL track, market condition, research condition, reward mode, and split. They average rollouts within each question and then weight questions equally. Invalid/missed assignments receive task loss 1 only on validly resolved questions. Pending and void cases do not acquire invented outcomes.

Formal comparisons should report paired differences on common questions, by K, domain, and horizon, with event/cluster-aware uncertainty and synchronized baseline coverage. Repeated rollouts must not narrow confidence intervals as if they were independent events. Predeclare missing-data rules, weighting, and multiple comparisons.

## Release and research milestones

| Milestone | Completion evidence |
| --- | --- |
| Protocol prototype | Binary/categorical validation, immutable submissions, delayed scoring, train/test isolation, offline tests |
| Automated collection | Source-specific discover/resolve adapters, persisted provenance, repeatable cycles, pending/error handling |
| First live pilot | Actual pre-deadline predictions, verifiable outcomes, measured coverage and operating costs |
| Comparable benchmark | Fixed assignments/budgets, internal or matched-market baselines, information conditions, stratified reports |
| Trainer integration | Token-level data and masks, policy provenance/staleness rules, tested parameter updates |
| Public release | English artifacts, reproducible setup/CI, reviewed repository identity/license, reviewed source-data distribution |

The code does not establish a live performance result merely by passing fixture tests. Complete at least two real collection/resolution cohorts and inspect failures before describing the automation as operationally validated. See the [release checklist](RELEASE_CHECKLIST.md) for repository publication tasks.
