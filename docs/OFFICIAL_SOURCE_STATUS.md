# Official source availability for the six reference systems

Checked 2026-09-27 against the authors' papers, repositories, and project
announcements. A public paper, SDK, or landing page is not evidence that the
entire evaluated system is reproducible from released code. This table is
about source availability, not about our measured performance.

| Reference | Official source status | What this project uses |
| --- | --- | --- |
| [DeltaBox](https://github.com/dongyunpeng-sjtu/deltabox/blob/main/README.md) | Its public author repository explicitly says the kernel patch, userspace controller, and benchmark artifacts are **not yet public**; it is a project landing page. | An independent cooperative process/OverlayFS experiment with a narrower state contract. No DeltaBox code. |
| [Crab](https://github.com/open-agent-infra/crab) | The official MIT-licensed repository is public, including process inspection, checkpoint scheduling, and CRIU/ZFS backend code. Pinned at `9607d61a41dc44358cf078c4b438bfd971c8ee9d`. This supersedes our earlier no-release-found observation. | A [full-QEMU experiment](OFFICIAL_CRAB_MICROVM.md) executes its unmodified monitor/policy: 8 → 3 saves; 18.101 → 16.257 s median controlled workload. The [original process backend](OFFICIAL_CRAB_CRIU.md) executes actual runc/CRIU dumps and restores. The [composite recovery probe](CRAB_COMPOSITE_RECOVERY.md) additionally restores whole owned RAM, workspace and held FD/offset, then continues exact writes and computation: five bounded positive trials, including two public-script replays. The local nonforced unmount/rollback/remount candidate keeps the original backend unchanged. eBPF, concurrent-writer atomicity, graded software episodes through this checkpoint path and a trainer remain unverified. |
| [SWE-MiniSandbox](https://github.com/lblankl/SWE-MiniSandbox) | The author repository publicly contains the Linux namespace/chroot sandbox implementation and agent/training adapters. | A pinned [real SWE-bench Flask task](../examples/official_mini_sandbox/OFFICIAL_TASK_RUN.md) ran through upstream deployment/session/grade code with an operator-prepared cache: 59/60 base and 60/60 reference tests, strict rewards 0/1. This is one task, not a model rollout or measured speedup. Our guest-local namespace path remains independent. |
| [RollArt](https://arxiv.org/html/2512.22560v2) | The authors' [ROLL framework](https://github.com/alibaba/ROLL) is public and links the paper. We have not established that its full reported heterogeneous, serverless production deployment and evaluation configuration are released. | An independent same-host rollout scheduler; a separate [protocol, runner, manager, and sample guard](../examples/official_roll/README.md) executed pinned upstream methods under infrastructure/model doubles and checked real CPU token tensors. No GPU trainer integration. |
| [Collinear Environment-as-a-Service](https://blog.collinear.ai/p/rl-env-as-a-service) | This is a product architecture article rather than a research artifact release. The organization has [public simulation examples and tools](https://github.com/collinear-ai), but we did not verify an open release of the EaaS control/data-plane service described in the article. | An independent seeded CI-triage task world and local service-style queue; no Collinear platform code. |
| [Daytona](https://github.com/daytonaio/daytona) | Its older core repository remains public and forkable. The company [moved current production core development to a private codebase in June 2026](https://www.daytona.io/dotfiles/updates/daytona-is-going-closed-source); SDKs/docs remain public. | An independent reusable QEMU template; no Daytona code or hosted service. |

The public source bundle vendors none of these repositories. Local pinned
checkouts used for study stay under ignored `runs/` and are excluded from the
release. The [integration audit](OFFICIAL_CODE_INTEGRATION_AUDIT.md) records
which exact upstream files and interfaces were inspected and which bridge
tests actually ran.

Additional [allocation and checkpoint research](ROLLOUT_ALLOCATION_RESEARCH_20260927.md)
now records pinned Tree-GRPO CPU advantage execution, VIP CPU allocation,
independent TRACE equation-level allocation, and newer code-review sources.
These are separate from the six-system source table and do not establish
full trainer or GPU integration.

[KLPO](KLPO_SOURCE_REVIEW.md) additionally runs actual author CPU losses and
toy optimizer updates; [AReaL](AREAL_SOURCE_REVIEW.md) runs its original
staleness manager and 45 official CPU tests. Neither is connected to a
GPU trainer or claimed as measured LLM learning throughput.

[CubeCoW](CUBE_COW_SOURCE_REVIEW.md), from the pinned CubeSandbox v0.7.0
source, additionally executes its original filesystem APIs and seven author
tests on real XFS. The populated multi-process fixture verifies 92 full-file
witnesses with independent hashes. The [Pilot-Commit repository producer](PILOT_COMMIT_CODING_OUTCOMES.md)
connects genuine scripted Docker case outcomes to allocation and episode-budget
accounting. These are component and control-plane results, without a complete
CubeSandbox VMM or model-training result.

The [CubeCoW latency experiment](CUBE_COW_LATENCY.md) additionally executes 240 populated filesystem operations in twelve paired groups on native ARM/HVF. It retains both return and caller-durable boundaries, every trial and independent full-content/isolation verification. Its scope is filesystem primitives.
