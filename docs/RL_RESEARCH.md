# RL research and experiment priorities

Evidence checked on **2026-09-22**. The recommendations below concern short-horizon, multi-domain probability forecasting with delayed real outcomes. They are design choices to test, not evidence that a new algorithm already outperforms FutureWorld or another forecaster on this project's events.

The primary reward remains `Brier(private_baseline, outcome) - Brier(agent, outcome)`, using the baseline frozen before the deadline. The agent predicts every option; it does not trade. The benchmark track measures a research system under a fixed protocol. The RL track collects train-only trajectories and prepares learning signals. Neither trajectory preparation nor a mathematical probe establishes successful parameter training.

## Recommended order

1. **Use grouped RLOO as the first learning-signal baseline.** Compare it with centered rewards without within-group standard-deviation scaling. Keep standard GRPO as an explicitly labeled ablation. Probability calibration is central here; results on deterministic answer checking do not establish suitability for stochastic forecasting. The relevant evidence is [Uncalibrated Reasoning](https://arxiv.org/abs/2508.11800) and the recent [proper-scoring-rule forecasting experiment](https://arxiv.org/abs/2608.28482).
2. **Preserve the delayed-feedback contract before increasing algorithm complexity.** Store immutable prediction-time actions and observations, exact policy revisions, complete rollout-group membership, and authoritative settlement provenance. Admit only complete, compatible, resolved training groups. A policy revision check is a conservative admission rule; it is not an importance-sampling correction.
3. **Allocate enough budget to distinct events.** Start with a small complete group, such as four trajectories, then compare groups of two, four, and eight under matched total rollout/token budgets. More trajectories expose policy variation within a question but cannot create additional independent real-world outcomes. Report the number of event clusters, group reward spread, and zero-advantage frequency.
4. **Add a real trainer only after token-level provenance exists.** This requires tokenizer/chat-template versions, token IDs, assistant-action masks, decoding settings, behavior-policy log probabilities, and a tested optimizer integration. The present text preparation is not that trainer.
5. **Evaluate asynchronous or turn-level methods after the baseline works.** SAO and Mu-GRPO motivate better treatment of stale trajectories; RTPO motivates stronger credit assignment. Their reported gains come from other task distributions. GSPO, SAPO, and AReaL remain candidate follow-up investigations after the data contract, supported implementation, and matched baseline are available; their names alone do not specify a working training recipe.

## What FutureWorld establishes and leaves open

[FutureWorld](https://arxiv.org/html/2604.26733v4) was first submitted on **2026-04-29**, with the reviewed version dated **2026-05-15**. It records live search trajectories, waits for realized outcomes, and replays completed trajectories with negative binary Brier reward and GRPO. It samples four trajectories per question, masks tool observations from policy loss, and uses 500 questions daily. Three small open models improve across eight training days; checkpoints are compared on a common live evaluation set.

Its reported pipeline discards questions lacking a retrievable answer at the scheduled attempt: **35.65%** across five observed days. Its RL training is binary, while its broader benchmark also uses selected-option/F1 and numeric scores. It does not establish the best optimizer, reliable learning across seven-day policy delays, or the quality of all-option categorical probabilities. Reported aggregate improvement also does not isolate how much comes from retrieval, probability calibration, formatting, or question-distribution adaptation.

For this project, pending settlement should remain a state with explicit retry/adjudication rules. The event horizon and publication delay are different clocks. Retaining pending records supports coverage analysis without inventing a negative outcome. These are project design conclusions, not claims that FutureWorld's experiment was invalid.

## Why standard GRPO needs a calibration ablation

[Bereket and Leskovec, 2025-08-15](https://arxiv.org/abs/2508.11800) find that standard GRPO can make probability reports overconfident on stochastic-outcome tasks; RLOO, PPO, and GRPO without group standard-deviation normalization perform better on calibration in their experiments. Their analysis also exhibits bias for Brier rewards. This does not say that every GRPO run becomes overconfident or that a particular replacement will win our live benchmark.

The following two-candidate calculation is a project diagnostic derived from the reward formula. Here `q` denotes a true Bernoulli outcome rate, `p` a central forecast, and `0 < delta < min(p, 1-p)`. Compare forecasts `p + delta` and `p - delta`, scored against the same outcome `Y`:

```text
reward_plus  = -(p + delta - Y)^2
reward_minus = -(p - delta - Y)^2

reward_plus - reward_minus = 4 * delta * (Y - p)

RLOO upper-candidate advantage:
    A_plus = reward_plus - reward_minus
    E[A_plus] = 4 * delta * (q - p)

Centered, no-std upper-candidate advantage:
    A_plus = (reward_plus - reward_minus) / 2
    E[A_plus] = 2 * delta * (q - p)

Standardized upper-candidate advantage, population std and epsilon -> 0:
    A_plus = +1 when Y = 1, and -1 when Y = 0
    E[A_plus] = 2*q - 1
```

At `p = q = 0.7`, the expected RLOO and centered advantages are zero, while the standardized upper-candidate advantage approaches `0.4`. Standardization removes the magnitude difference between a helpful outcome and a harmful one. A sample-standard-deviation convention changes the constant scale, not this directional problem; finite epsilon makes the final expression approximate.

This is an exact outcome expectation for a fixed pair of candidates, not a proof about the full stochastic LLM training dynamics. It can be checked deterministically by weighting the two possible outcomes, without Monte Carlo noise or a GPU run. An implemented probe verifies arithmetic and estimator behavior, not learned calibration.

## Baseline improvement and strict scoring incentives

For a fixed question and outcome, write the private baseline's Brier loss as `C` and trajectory `i`'s Brier loss as `L_i`. The baseline must be fixed independently of that trajectory's actions and prediction:

```text
r_i = C - L_i
r_i - mean(r) = mean(L) - L_i
r_i - mean(r_j, j != i) = mean(L_j, j != i) - L_i
```

Thus the question's baseline term cancels in both centered GRPO-style advantages and RLOO. It also leaves within-group reward standard deviation unchanged. Baseline-relative reward remains useful for the externally meaningful question, "Did this system improve on the available baseline?" It does not, by itself, give a grouped optimizer a different within-question gradient from negative Brier. This conclusion assumes exogenous outcomes, the same baseline in the group, and no action-dependent baseline selection.

Keep the market-first/private-fallback baseline on the scorer side. Benchmark exposure to public consensus is a separately recorded information condition. A baseline that depends on the submitted forecast, or is chosen after observing which comparison looks favorable, breaks the intended comparison.

The project's normalized categorical Brier score is `0.5 * sum((p_k - y_k)^2)`. It reduces to the conventional binary score and is strictly proper on an exhaustive, mutually exclusive option set. Subtracting the baseline score preserves the probability optimum because that term does not depend on the reported probabilities.

Report latency, token usage, and cost beside forecast quality, and impose matched research budgets. Adding action-dependent cost, length, confidence, citation-count, or reasoning-style rewards creates a different optimization target and may distort probability reporting or evidence acquisition. In particular, do not treat producing longer reasoning or matching a process rubric as evidence that the final distribution deserves a higher outcome reward. Any auxiliary objective needs its own preregistered trade-off and ablation. A variable-length policy-loss normalization is also a trainer choice requiring inspection; it is not part of the Brier scoring rule.

## Delayed replay and tool credit

An event can resolve days after its trajectory was generated. Calendar age is not the same as policy distance: an old trajectory may still match a frozen policy, while a recent one may already be stale after many updates. A strict same-revision gate is a reasonable initial design. It will limit reuse after policy updates; measure this exclusion rather than silently mixing revisions. To relax it, first obtain behavior log probabilities and a supported importance-ratio/staleness rule. Token clipping alone is not a general guarantee that arbitrarily stale prefixes remain safe to optimize, as the [Mu-GRPO study](https://arxiv.org/abs/2605.17570) illustrates.

The storage contract should retain:

- Question/event/cluster IDs, train split, cutoff, baseline snapshot, resolution rule/version, and settlement status.
- Group ID, intended group size, rollout ID, immutable policy revision, prompt/scaffold/tool versions, and sampling settings.
- The exact action/observation sequence, URLs, retrieval timestamps, errors, final probability vector, and termination reason.
- For a future optimizer: token IDs, chat-template mapping, assistant-action token mask, behavior log probabilities, and reference/loss configuration.

Tool responses are environment observations, not model actions. They condition later decisions but must not receive assistant-generation policy loss. A text-level role mask is only a preparation aid: it must later be translated and checked against the actual token sequence, including tool-call boundaries and special tokens.

With a terminal reward, query selection, evidence interpretation, and final probability generation receive a shared outcome signal. This weak credit assignment is a real research problem, but inserting an LLM judge's process score is not automatically a solution. A staged ablation should compare final-answer-only learning, full assistant-action learning, and a supported turn-aware estimator under the same data and budget. Process diagnostics can remain evaluation-only until validated against held-out outcome scores.

Branching or resampling old tool trajectories needs special care. A search executed after settlement can disclose the answer. RTPO-style counterfactual continuations would require a reproducible prediction-time evidence environment, or all branch rollouts must be completed before the deadline. A stored historical trace alone does not provide a replayable web environment.

## Primary reading set

These are author papers or primary preprints, with dates verified from their arXiv records. Newer publication dates do not make results more general. FutureWorld's details are summarized above; the remaining summaries are deliberately limited to measured findings and applicable design implications.

| Paper and date | Verified finding | Implication and limit for this project |
| --- | --- | --- |
| [FutureWorld](https://arxiv.org/abs/2604.26733), 2026-04-29; v4 2026-05-15 | Live tool trajectories, delayed real outcomes, and GRPO updates form an executable research loop. | Closest environment precedent; our categorical probabilities, pending retention, and optimizer comparisons need their own validation. |
| [Uncalibrated Reasoning: GRPO Induces Overconfidence for Stochastic Outcomes](https://arxiv.org/abs/2508.11800), 2025-08-15 | Experiments and analysis identify calibration problems caused by within-group std normalization. | Strongest direct reason to begin with RLOO/no-std advantages and include a calibration probe. |
| [How Proper Scoring Rules Shape LLM Forecasting](https://arxiv.org/abs/2608.28482), 2026-08-28 | Dr. GRPO uses mean-centered rewards without std scaling. On 965 held-out binary questions, the Brier-trained model's Brier score is 0.1648 versus base 0.1861; log training has the lowest ECE. | Supports Brier-oriented training plus separate calibration diagnostics. One seed per reward, one base model, 7–90-day horizons, and unequal realized token budgets limit generality. It is not a live multi-turn result for our events. |
| [Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning](https://arxiv.org/abs/2607.07508), 2026-07-08 | SAO combines single-rollout sampling, a trained critic, direct behavior-policy ratios, and double-sided clipping/masking. On its coding setup, SWE-Bench Verified rises from 27.0% with GRPO+DIS to 29.8% with SAO. | Candidate when research-rollout groups become a throughput bottleneck. Requires reliable behavior log probabilities and value training; its reasoning/coding and simulated writing results do not remove real-outcome delay. |
| [How Off-Policy Can GRPO Be? Mu-GRPO](https://arxiv.org/abs/2605.17570), 2026-05-17 | Across five models on math tasks, staged rollout reuse with relaxed clipping and negative-advantage veto matches or exceeds standard GRPO with about 2x training speedup. | Motivates stored behavior probabilities and explicit stale-prefix diagnostics. It does not license unrestricted replay of seven-day-old forecasting trajectories. |
| [RTPO: Reverse-Turn Policy Optimization for Stabilizing Agentic RL Training](https://arxiv.org/abs/2608.18682), 2026-08-19 | Reverse-turn sibling continuations address context mismatch, credit assignment, and policy drift in the paper's agentic tasks. Its strict convergence claim applies to a tabular formulation with fixed downstream policies. | A research direction for tool-level credit, not a drop-in guarantee for shared neural policies or changing live websites. Requires compatible branching and evidence snapshots. |
| [Hyperagents](https://arxiv.org/abs/2603.19461), 2026-03-19 | An editable task agent and meta-agent improve task performance and the mechanism producing later variants. The paper separates validation selection from held-out testing and measures meta-improvement with `imp@k`. | Supports a separately measured outer improvement loop. A rising training reward alone does not demonstrate improved ability to improve. Details belong in [SELF_IMPROVEMENT.md](SELF_IMPROVEMENT.md). |
| [Reflections on Trusting Trust, Revisited](https://arxiv.org/abs/2609.17817), 2026-09-15 | Proof-of-concept poisoned benchmarks contaminate self-modifying coding agents, including Hyperagents; effects can persist after later clean evolution. | Evidence for keeping evaluator rules, baseline selection, and final holdouts outside candidate-editable state. The measured attack is on coding systems, not evidence of an attack on this project. |

## Evidence needed before claiming improvement

Use a frozen chronological train/dev/test plan and event-cluster isolation. Repeated versions of the same event, neighboring numeric bins, and reworded questions are not independent holdouts. A test question does not become training data merely because it has resolved. Training on historical events needs evidence available at the original cutoff and does not retroactively create a clean live test.

Compare candidates on the same questions, information condition, baseline coverage, and resource budget. Report paired Brier improvement, raw Brier, calibration curves, clipped log loss, valid-submission rate, and pending/void/excluded coverage by domain, horizon, and option count. Average or otherwise aggregate repeated rollouts within a question before event/cluster-aware uncertainty calculations; do not use rollout count as the independent sample size.

The first decision is whether train-split collection yields enough diverse, resolvable, auditable events. The next is whether a real optimizer improves future held-out forecasts across repeated seeds/cohorts. Only after those results should a more complex optimizer or an outer self-improvement loop be credited with a performance gain.
