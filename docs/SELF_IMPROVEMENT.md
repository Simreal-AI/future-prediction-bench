# Self-improvement design and evaluation

Research checked against primary sources on **2026-09-22**. This repository supplies an analyst environment, trajectory interfaces, and an offline candidate evaluation primitive. It does not implement an autonomous self-modifying agent, a GEPA/Hyperagents/Dream-RSI reproduction, or a model-weight trainer. No forecasting gain from these mechanisms has been demonstrated here.

## What the research supports

| Primary source and verified version | Mechanism | Relevant design choice here |
| --- | --- | --- |
| [GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning, v2, 2026-02-14](https://arxiv.org/abs/2507.19457v2) | Reflect on execution trajectories, propose prompt revisions, and retain complementary candidates using a Pareto-oriented search. This is prompt optimization; its task-specific comparisons do not establish that prompt search universally replaces weight RL. | Version prompt and skill artifacts, preserve lineage, and compare proposals on independent outcomes. A reflection is a hypothesis about a useful change, not evidence of improvement. |
| [Hyperagents, v1, 2026-03-19](https://arxiv.org/abs/2603.19461v1) | An editable program combines a task agent with the meta agent that generates further changes. Its experiments initialize a frozen foundation model. The main experimental outer selection and evaluation procedures remain fixed. | Track which agent proposed a revision and which parent it changes. Improving forecast accuracy alone does not establish that the proposal mechanism improved. The current implementation records candidates but neither generates nor executes them. |
| [AREX: Towards a Recursively Self-Improving Agent for Deep Research, v3, 2026-09-01](https://arxiv.org/abs/2607.21461v3) | An inner research loop builds an answer; an outer loop verifies constraints and chooses targeted refinement or restart. An explicit context-update operation preserves evidence and unresolved constraints. The paper also trains models through mid-training and long-horizon RL. | Represent verified evidence, contrary evidence, unresolved claims, and next research steps in analyst state. Keep within-episode revision distinct from persistent adaptation and from weight optimization. Forecast confidence must not be substituted for a verified outcome. |
| [Dream-RSI: Recursive Self-Improvement through Evolving Worlds, v1, 2026-09-14](https://arxiv.org/abs/2609.14858v1) | Optimize an exploration policy by replaying recorded discovery trees, then use the selected policy in another online round. The underlying models, evaluator, and interfaces stay fixed. Replay only reveals recorded continuations. | Recorded research attempts could support bounded experiments with scheduling and stopping rules. A historical trace cannot reveal what a different search query would have returned, or what an unrecorded forecast would have scored. The current module does not simulate such counterfactuals. |

These are findings from the cited tasks, not measured results for future-event forecasting. Our application of them is a research design inference. In particular, no paper above makes the future outcome of an unresolved event available for immediate online learning.

Evaluation integrity also matters empirically: [Reflections on Trusting Trust, Revisited, 2026-09-15](https://arxiv.org/abs/2609.17817) studies benchmark poisoning of self-modifying coding agents and propagation into descendants. This motivates keeping the benchmark, reward code, private baseline store, and final holdout outside candidate editing and input privileges. It is not evidence that this repository has been compromised.

## Three different loops

1. **Episodic self-correction:** before the forecast deadline, the analyst gathers evidence, audits its claims, revises probabilities, and optionally compresses its working state. This changes one trajectory. Tool feedback can verify citations and chronology; it cannot verify the future outcome. There is no additional outcome reward for sounding more confident or writing a longer critique.
2. **Persistent skill and harness optimization:** after historical outcomes become available, a proposer extracts failure patterns and suggests a new prompt, skill, memory policy, or resource allocation configuration. The proposal becomes a separately identified candidate. Future paired development outcomes decide whether its forecasting behavior improves. Repeatedly revising the proposal generator would be a further meta-level experiment, requiring its own matched comparison of candidate-generation quality and cost.
3. **Weight RL:** a trainer uses actual model actions, their behavior-policy identity and probabilities where required, masked observations, and delayed verified rewards to update model weights. Exporting scored traces is preparation for a trainer; it is not training. Run prospective evaluation under a fixed checkpoint after an update. See [RL_RESEARCH.md](RL_RESEARCH.md) for the broader training design.

For all three, retain raw proper scores separately from any training reward transformation. Self-reported progress and model-written critiques are useful state, not ground-truth reward labels. Private baselines must remain evaluator-only even when a baseline-relative reward is later exported by an authorized training interface.

## Chronological experiment protocol

The intended lifecycle is historical training outcomes → candidate proposal and commitment → future paired development forecasts → delayed resolution → sealed comparison → reviewed candidate selection → a fresh future window. The final test split is reserved for reporting a frozen selected system. It is never a proposal source or a promotion set.

Before the development window starts, the evaluator commits its entire question cohort, event/cluster assignments, chronology, metric, aggregation, sample floor, minimum effect, bootstrap seed, interval confidence, and comparison budget. Candidate manifests commit artifact content hashes, parent lineage, creation time, and the latest task-specific training/reflection data cutoff. Both candidate and incumbent are fixed before `dev_start`. Use the same evidence cutoff, available tools, model sampling rules, and resource budget in a paired experiment; record these settings in a configuration artifact and enforce them in the external runner.

Both systems forecast every registered question during the development window and strictly before its deadline, matching the store's exclusive deadline convention. Once all outcomes have resolved and `dev_end` has passed, a trusted scorer supplies the paired normalized Brier scores. No answer text, prompt text, private baseline, raw metadata, test record, or unresolved outcome enters this API. The incumbent here is an explicitly evaluated agent version; it is not the benchmark's private baseline.

Every registered question is required. Missing runs, invalid distributions, unresolved or void events, and changed resolution rules block this primitive. It does not silently remove them or invent scores. A richer precommitted failure/void policy can be added later as a new contract version; it must not be invented after inspecting performance. The current strict policy favors auditability over throughput.

For question `q`, define `delta[q] = incumbent_brier[q] - candidate_brier[q]`; positive values favor the candidate. Average questions within each event, then events within each cluster, then weight clusters equally. This prevents paraphrases and densely sampled events from dominating the result. Cluster membership must reflect shared outcome dependence and be assigned before outcomes; IDs alone cannot establish independence.

The pure evaluator resamples whole cluster means with a local seeded generator. It returns a percentile interval with tail probability `(1 - confidence) / (2 * max_comparisons)`. A candidate is eligible only if the lower endpoint strictly exceeds the frozen minimum improvement. There must be at least 30 clusters by default, with an enforced configurable floor of 20. Sample floors and a bootstrap do not guarantee coverage: the interval is approximate and depends on representative, sufficiently independent clusters. Multiple-comparison adjustment is also approximate because bootstrap intervals are approximate.

The comparison budget covers a predeclared set of candidates, all frozen before that window. A trusted caller must persist each unique comparison slot and reject reuse. Neither changing the random seed until an interval passes nor trying unlimited candidates on one dev window is permitted by this protocol. An adaptive next-generation candidate needs fresh chronological outcomes. Store and scheduler enforcement are not implemented in this pure module.

## Pure Python API

`future_prediction_bench.improvement` has no third-party dependencies, file access, network access, database queries, model calls, code executor, training loop, or deployment action. Constructors return detached ordinary dictionaries; consumers revalidate exact field sets and content hashes. All validation errors raise `ValueError`.

```python
from future_prediction_bench.improvement import (
    create_candidate_manifest,
    create_evaluation_contract,
    evaluate_promotion,
    seal_dev_outcomes,
)

# These are illustrative trusted-exporter inputs, not database query helpers.
# Use at least 30 independent event clusters for the default promotion floor.
contract = create_evaluation_contract(
    frozen_at="2027-01-01T00:00:00Z",
    dev_start="2027-02-01T00:00:00Z",
    dev_end="2027-03-01T00:00:00Z",
    questions=[{
        "question_id": "q-1", "event_id": "event-1", "cluster_id": "cluster-1",
        "forecast_deadline": "2027-02-02T00:00:00Z",
        "outcome_not_before": "2027-02-03T00:00:00Z",
        "resolve_after": "2027-02-04T00:00:00Z",
    }],
)
trusted_commitment = contract["sha256"]  # Persist before the window starts.
incumbent = create_candidate_manifest(
    artifacts={"prompt": "Initial analyst instructions", "config": "model=offline; budget=4"},
    created_at="2027-01-02T00:00:00Z",
    training_data_cutoff="2026-12-31T00:00:00Z", contract=contract,
)
candidate = create_candidate_manifest(
    artifacts={"prompt": "Revised analyst instructions", "config": "model=offline; budget=4"},
    created_at="2027-01-03T00:00:00Z",
    training_data_cutoff="2026-12-31T00:00:00Z", contract=contract, parent=incumbent,
)
sealed = seal_dev_outcomes(
    contract=contract, incumbent=incumbent, candidate=candidate,
    sealed_at="2027-03-01T00:00:00Z",
    rows=[{
        "question_id": "q-1",
        "incumbent_submitted_at": "2027-02-01T12:00:00Z",
        "candidate_submitted_at": "2027-02-01T12:00:00Z",
        "resolved_at": "2027-02-04T00:00:00Z",
        "incumbent_brier": 0.3, "candidate_brier": 0.1,
    }],
)
decision = evaluate_promotion(
    contract=contract, incumbent=incumbent, candidate=candidate,
    sealed_outcomes=sealed, expected_contract_sha256=trusted_commitment,
)
assert decision["eligible"] is False  # One cluster is insufficient evidence.
assert decision["reason"] == "insufficient_clusters"
```

`questions` and `rows` accept exactly the keys in the example. Passing a whole database question, raw model prompt, a private baseline field, or a test field fails instead of creating an implicit data path. Artifact names map to caller-supplied text or bytes; a filename string is hashed as text and is never opened. Include the actual artifact contents. Manifests store only the hashes, not the original artifacts.

An incumbent can be re-registered for a new contract by calling `create_candidate_manifest` with the same artifacts and the previous manifest as `parent`. This creates an explicit lineage step for a changed evaluation registration. The new challenger is then its direct child. Preserve the complete archive externally, including rejected candidates and comparison results.

The seal is a reproducible integrity envelope, **not a digital signature, proof of chronological collection, or proof that a score is true**. Anyone who controls its inputs can fabricate another self-consistent envelope. Only `dev` is accepted as the envelope split; proving that the referenced questions actually came from that split remains the exporter's responsibility. A real integration must use the trusted store's immutable submissions and resolution records, retain the original committed hashes outside the candidate's control, verify artifact identity in the runner, and enforce split isolation. The helper does not claim to enforce operating-system privileges or authenticate an external dataset.

## What would establish improvement

Compare a frozen analyst, episodic correction only, historical skill revision, and any later weight-trained policy under matched budgets. Report raw Brier and log loss, event-cluster uncertainty, calibration, invalid-output rate, cost, and coverage on fresh chronological cohorts. Ablate each proposed mechanism. To claim improved ability to improve, hold the proposal budget and evaluation contract fixed and compare the quality of descendants generated by different proposer versions; a rising archive maximum is not sufficient evidence.

Offline replay can support inexpensive design exploration on legitimately available historical observations. Final evidence requires fresh paired forecasts made before their outcomes. This implementation supplies candidate identity and a limited statistical gate for that experiment, while leaving model execution, proposal generation, store authorization, and any weight training to separately specified integrations.
