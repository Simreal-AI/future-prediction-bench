# Project context

This document records the project's public design rationale. It is a curated summary, not a transcript of private planning conversations.

## Purpose

Future Prediction Bench evaluates systems that research genuinely unresolved events and return a probability for every possible outcome. The same event stream supports a research benchmark and an environment that supplies delayed rewards after real outcomes arrive.

The initial scope is multi-domain forecasting with a target no more than seven days after the prediction deadline. Binary and categorical questions share one probability protocol. Categorical options must be mutually exclusive and collectively exhaustive; multi-label events need a different protocol.

The project language is English: documentation, question templates, fixtures, prompts, diagnostics, and public metadata should be usable without local context. The public alpha source release is maintained by Simreal-AI.

## Why research conditions matter

A forecasting system's score reflects its model, tools, information access, research budget, base rates, and probability calibration. A high score alone does not identify which component improved.

The default `no_consensus` condition attempts to filter direct market and crowd probabilities from agent observations. A separate `market_aware` condition measures performance with those sources available. Filtering is heuristic and should be audited; it is not a proof that every indirect reference has been removed.

Private baselines serve a different purpose from agent evidence. A market baseline is eligible only when its event, options, target, deadline, and resolution criteria match the question. Otherwise, a fixed internal model can produce the baseline. Both must be recorded before the forecasting deadline, kept out of agent observations, and used only by scoring/reward code. Missing baselines remain missing.

## Why delayed feedback needs explicit state

An unresolved event has no outcome reward. A missing source value is not a negative outcome, a canceled event is not automatically a No, and a late publication does not change the original forecasting horizon. Pending, invalid, missed, resolved, and void states must remain distinguishable.

Stored research observations support replay and audit. Text trajectories alone do not implement parameter training: a trainer also needs token-level masks, behavior-policy log probabilities, checkpoint information, and a policy for stale rollouts.

## What evidence can support a claim

The first live cohorts test whether collection, forecasting, and resolution work reliably. Multiple rollouts for one event share one outcome; they do not create independent forecasting examples. Comparisons should use common questions, comparable deadlines, coverage reporting, and event-cluster-aware statistics.

No human-expert study is required to start. Without a suitable contemporaneous human comparison, the project should report system comparisons and baseline performance rather than claims of expert-level ability.

## Design influences

[FutureX](https://arxiv.org/abs/2508.11987), [ForecastBench](https://arxiv.org/abs/2409.19839), and [FutureWorld](https://arxiv.org/abs/2604.26733) inform the questions this project investigates. This is an independent implementation, not an official reproduction. See [references](REFERENCES.md) and the [project design](PROJECT_DESIGN.md).
