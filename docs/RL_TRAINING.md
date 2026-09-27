# RL collection and preparation contract

The implemented layer collects model-driven analyst trajectories, joins delayed outcomes, prepares sequence advantages, and validates text-level generation masks. It does **not** update model weights. Internal reference forecasts and the policy being trained are separate systems: the reference supplies a private scoring baseline; a future weight trainer needs a policy with accessible weights or a supported training service.

## Run the offline smoke exercise

```bash
python -m future_prediction_bench rl-smoke --output runs/rl-smoke-example
```

Use a new output directory. Two synthetic train questions, one binary and one categorical, each collect four scripted trajectories through search, open, calculator, notebook, draft and submit. All eight forecasts remain pending until the fixture clock advances. Resolution then produces two prepared RLOO groups. A changed policy revision causes both groups to be quarantined. Forty analyst calls exercise the shared tool path.

Artifacts include `bench.sqlite`, `trajectories.jsonl`, `prepared_groups.json`, `calibration_probe.json`, and `report.json`. They are ignored operational files. No network, actual model inference, or optimizer is used. The mathematical calibration probe enumerates both outcomes exactly; its favorable/unfavorable signal is not a learned performance measurement.

## Collect real policy rollouts

First preregister a future **train** cohort disjoint by event/cluster and time from development and test. The default source configuration produces test questions and must not be relabeled after observing outcomes. Configure a train cohort prospectively. Evaluated benchmark models can be connected later, when a fresh unresolved evaluation cohort is available.

```bash
python -m future_prediction_bench forecast --db runs/train.sqlite \
  --track rl --rollouts-per-question 4 --temperature 0.7 \
  --policy-revision immutable-checkpoint-001 --collection-id cohort-001 \
  --baseline-config configs/baselines.local.json
```

`--policy-revision` must identify the actually served immutable weights. It is an operator commitment, not cryptographic proof of what a remote endpoint served. Sampling settings and tool configuration are also hashed. `--collection-id` and question identity determine stable group IDs; reruns skip already assigned sample indices, including failed ones. A conflicting collection configuration is rejected. Missing baselines prevent inference in the default improvement mode.

Group size is 2–64; begin experiments at 4 and compare 2/4/8 under equal total budgets. The CLI defaults grouped sampling to temperature 0.7 if unspecified and requires a positive temperature. A positive temperature does not guarantee distinct trajectories. The Python collection API leaves decoding control to the model adapter. Equal-reward groups are counted, not resampled after outcomes to manufacture a useful gradient.

Registration happens before actions and records question, group ID, intended group size, sample index, policy revision, configuration hash, information mode, reward mode and optional evidence-pack identifier. The identifier alone is still metadata. The separate `EvidencePack` module now seals recorded pre-cutoff search/open observations and offers closed exact-match replay through `EvidencePack.provider()`. The current live runner does not automatically create packs or substitute historical virtual clocks. Market tools have a separate provider and must be disabled or independently frozen in historical replay. Distinct live rollouts can see changing web responses, so comparisons should inspect actual observation times.

## Join outcomes and prepare groups

```bash
python -m future_prediction_bench prepare-rl --db runs/train.sqlite \
  --current-policy-revision immutable-checkpoint-001 \
  --advantage-method rloo --output runs/prepared.json
```

The trusted store exports only train-split RL trajectories. Preparation additionally checks complete unique sample membership, consistent question/configuration/policy/information conditions, valid prediction-time chronology, resolved outcomes available at the current cutoff, and reconstructable assistant/tool messages. A void, pending, missing, stale, or incompatible sample prevents that group from becoming a prepared group. Diagnostic reasons and counts remain visible. Invalid probability submissions retain their protocol penalty, but their group must still wait for the question's valid resolution.

The conservative first policy is exact revision matching. A week-old trajectory from a still-frozen policy may be usable, while a minute-old one from changed weights may not be. An asynchronous trainer could later admit older revisions with measured behavior probabilities and supported correction. This implementation does not invent ratios, assume negligible policy drift, or regenerate old evidence from the live web.

## Advantage choices

| Option | Sequence advantage | Role |
| --- | --- | --- |
| `rloo` (default) | `reward_i - mean(other rewards)` | First experimental baseline; no reward standard-deviation scaling. |
| `centered` | `reward_i - mean(all rewards)` | No-std group comparison; proportional to RLOO for fixed group size. |
| `standard_grpo` | `(reward_i - mean(all rewards)) / (population_std + 1e-8)` | Explicit calibration ablation. |

These are advantage estimators, not complete optimizer implementations. In particular, `centered` does not by itself implement all of Dr. GRPO, and choosing `standard_grpo` does not implement a token-clipped GRPO loss. The preserved terminal reward remains the selected proper-score reward. A fixed same-question private baseline cancels when computing these relative advantages; it remains useful as an external score comparison.

Do not add citation count, confidence, or self-judged reasoning quality to outcome reward without a separately justified experiment. Notes and drafts support the agent's work but do not produce dense success labels. Resource budgets are enforced independently of score. See [RL_RESEARCH.md](RL_RESEARCH.md) for direct probability-calibration evidence and current asynchronous/turn-level research.

## What a real optimizer still needs

Prepared samples retain exact per-turn requests and assistant responses, tool definitions, sequence reward and advantage, and message spans marked as generated or observed. Tool schemas are part of the conditioning input, not generated actions. Tool output stays in the context needed to score later assistant tokens. Host-synthesized termination is never an assistant target.

The output deliberately retains `trainer_ready=false`. To integrate a tokenizer, vLLM/verl-style runtime, or another trainer, implement and verify:

1. The exact deployed tokenizer and chat template, including tool-schema and tool-call formatting.
2. Token IDs and loss masks aligned with actual assistant generations and special-token boundaries.
3. Behavior-policy token log probabilities from the preserved policy, with matching conditioning context and sampling/truncation semantics.
4. The chosen optimizer, reference/KL configuration, sequence-versus-token loss reduction, clipping and policy-staleness handling.
5. Repeated-seed, prospective event-cluster evaluation of Brier, calibration, valid coverage and resource usage.

Missing log probabilities are marked unavailable. No API usage counter or reported output probability distribution can substitute for generation-token log probabilities. A GPU update and subsequent held-out improvement must be observed before claiming successful RL training.

The separate coding track can now fork several suffixes from one filesystem or full-VM checkpoint. Its branches inherit one prefix and are correlated observations of one task. The [branch-local RLOO preparation](BRANCH_ADVANTAGES.md) checks complete same-policy siblings and assigns the sibling reward contrast only to post-checkpoint policy actions. A future optimizer must retain the common prefix's behavior-policy context, attach each suffix's own token log probabilities and masks, and justify any separate shared-prefix credit estimator. Counting sibling suffixes as independent task examples or resetting their action/wall budget would bias a learning comparison. The current branch outputs remain text trajectories with `trainer_ready=false`; measured throughput changes apply to the environment only.

## Skills and recursive improvement

The separate [self-improvement module](SELF_IMPROVEMENT.md) hashes candidate artifacts, commits an independent future development contract, and checks paired cluster-level score improvement. It returns promotion eligibility, without modifying the agent, evaluator, baseline, or final test set. Candidate generation, trusted score export, comparison-slot persistence and the outer executor remain integration work. Improving a forecast draft inside an episode is distinct from improving a reusable skill, and both are distinct from a model parameter update.
