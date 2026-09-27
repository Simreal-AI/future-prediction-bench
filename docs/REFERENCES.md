# References and scope

These primary sources inform the project's methods. Links do not imply that this repository reproduces the original experiments, has integrated their code, or inherits their data rights. The project's own protocol is specified in [PROJECT_DESIGN.md](PROJECT_DESIGN.md).

## Forecasting research

- [FutureX paper](https://arxiv.org/abs/2508.11987) and [collection pipeline, section 3.2](https://arxiv.org/html/2508.11987v3#S3.SS2): a reference for dynamic forecasting, reusable source templates, and automated question/result collection. [FutureX-Eval repository](https://github.com/Futurex-ai/Futurex-Eval) is a separate artifact; do not assume an evaluation repository contains the complete collection service.
- [ForecastBench paper](https://arxiv.org/abs/2409.19839): a reference for live probability forecasting and comparative evaluation. This project has not collected a contemporaneous expert-forecaster baseline and makes no expert-level performance claim.
- [FutureWorld paper](https://arxiv.org/abs/2604.26733): a reference for collecting forecasts before outcomes are known and using later outcomes as training feedback. This project's shared binary/categorical protocol and implementation are independent design choices.
- [verl-tool repository](https://github.com/TIGER-AI-Lab/verl-tool): a possible future tool-agent training integration. This repository does not currently include that trainer or claim compatibility with a tested GPU training configuration.

The project intentionally omits copied performance tables, recruitment statistics, and claims about the current availability of external code. Confirm a source's version, license, and interface when integrating it, and record the exact dependency version or commit in the run configuration.

## Real-world RL and sandbox systems

The detailed state-boundary and measurement audit is in [MICROVM_RESEARCH.md](MICROVM_RESEARCH.md). Primary sources are [DeltaBox v2](https://arxiv.org/html/2605.22781v2), [Crab](https://arxiv.org/html/2604.28138), [SWE-MiniSandbox](https://arxiv.org/html/2602.11210), [RollArt v2](https://arxiv.org/html/2512.22560), [Collinear sandbox documentation](https://docs.collinear.ai/core-concepts/sandbox), and [Daytona persistence documentation](https://www.daytona.io/docs/en/persistence/). [Branching Policy Optimization](https://arxiv.org/html/2607.14171) motivates our separate [sibling-return preparation](BRANCH_ADVANTAGES.md); this project has not reproduced that paper's policy training.

The [official Crab implementation](https://github.com/open-agent-infra/crab)
is now public. Our [pinned real-guest probe](OFFICIAL_CRAB_MICROVM.md) invokes
its process monitor and policy, with a narrower full-QEMU backend. This
supersedes earlier source-availability observations.

## Allocation and recent checkpoint research

The [2026-09-27 implementation review](ROLLOUT_ALLOCATION_RESEARCH_20260927.md)
records exact execution boundaries and evidence for these primary sources:

- [Tree-GRPO paper](https://arxiv.org/abs/2509.21240) and
  [official code](https://github.com/AMAP-ML/Tree-GRPO): pinned advantage
  functions execute locally with real CPU PyTorch; no trainer executes.
- [VIP paper](https://arxiv.org/abs/2602.01601) and
  [official code](https://github.com/HieuNT91/VIP): pinned allocation module
  executes with real NumPy/SciPy; no predictor is trained.
- [TRACE](https://arxiv.org/html/2606.11119): equations 12–16 inform the
  independent bounded contrast allocator. No author code release is established.
- [KLPO official repository](https://github.com/yifanzhang-pro/KLPO):
  September 2026 technical-report/code source executes unchanged with real
  CPU PyTorch: exact MC-KL/reference gradient checks and four author toy
  updates. See the [source review](KLPO_SOURCE_REVIEW.md); no LLM or GPU
  training integration is established.
- [Exact checking of agent execution edits](https://arxiv.org/abs/2608.22928)
  and [author-linked code](https://github.com/eunomia-bpf/agent-check-restore-safety):
  19 pinned upstream history-realization tests passed; no sandbox guard is integrated.
- [Checkpoint handoff](https://arxiv.org/abs/2609.19636): recent paper review;
  no verified author code or replication here.
- [AReaL 2.0 report](https://arxiv.org/abs/2607.01120v2) and
  [official code](https://github.com/areal-project/AReaL): pinned 2.1.0
  capacity/recovery manager executes with normal imports; 45 official CPU
  tests pass. See [source review](AREAL_SOURCE_REVIEW.md). No GPU training
  or model-throughput result is established locally.
- [Waypoint author project](https://daplab.cs.columbia.edu/projects/waypoint/),
  [systems paper](https://arxiv.org/abs/2510.05556), and
  [official code](https://github.com/Alex-XJK/waypoint): the complete pinned
  package compiles and its image-store API passes actual Linux
  tmpfs/disk, lock, atomic-publication and retry checks. See
  [source review](WAYPOINT_SOURCE_REVIEW.md); no actual process restore or
  training-throughput claim is established by this byte-fixture example.
- [CRIU incremental-dump interface](https://criu.org/Incremental_dumps),
  [runc 1.4.3 implementation](https://github.com/opencontainers/runc/blob/v1.4.3/libcontainer/criu_linux.go),
  and [CRIU 4.2 memory dumping](https://github.com/checkpoint-restore/criu/blob/v4.2/criu/mem.c):
  primary sources for the actual parent/epoch
  [interleaved-write challenge](CRIU_PAIRED_EPOCH_CHALLENGE.md).

## Candidate official sources

The implemented adapters, source semantics, and operating requirements are documented in [SOURCES.md](SOURCES.md). The following links also support future source expansion; inclusion here does not mean an adapter has been implemented or validated in a live cohort.

- [USGS earthquake catalog API](https://earthquake.usgs.gov/fdsnws/event/1/): earthquake observations and catalog query semantics.
- [MLB scores](https://www.mlb.com/scores): official schedule/result context for baseball events.
- [NWS web API documentation](https://www.weather.gov/documentation/services-web-api): weather observations, metadata, and service behavior.
- [NBA games](https://www.nba.com/games): a candidate official sports schedule/result source.
- [U.S. Treasury daily interest rates](https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve) and [XML feed documentation](https://home.treasury.gov/treasury-daily-interest-rate-xml-feed): candidate fixed-date, fixed-tenor publication targets.
- [NASA events](https://www.nasa.gov/events/): candidate discovery only; event occurrence must be supported by mission-specific outcome evidence.

Source publication delays and revisions need explicit resolution rules. Our seven-day horizon is a project constraint, not a guarantee that any source will publish a valid result within seven days. Raw source payloads and research snapshots are operational records; confirm retention and redistribution rights before including them in a public data release.
