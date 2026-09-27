# Research results and integration record

This page preserves the detailed experiment descriptions and command examples from the project homepage before the v0.19.1 release. Run every shell command below from the repository root. Measurements describe their specific fixtures, hardware, and trust boundaries; they are not formal model forecasting scores or evidence of RL learning. Start with the [README](../README.md) for the five-minute offline demo.

A probability forecasting benchmark and two RL environment directions: delayed real-event forecasting and independently verifiable real-world work. The forecasting scope is multi-domain events with a target no more than **seven days after the prediction deadline**.

| Track | Agent task | Result |
| --- | --- | --- |
| Research Benchmark | Search, read, research, and submit a probability for every option | Frozen forecasts, evidence traces, and outcome-based scores |
| Live RL Environment | Collect grouped analyst rollouts on train questions | Delayed rewards, policy-version checks, and RLOO/centered text batches for a future trainer |
| Real-World Task RL | Inspect and change an isolated software workspace, then submit it | Immediate hidden verification, auditable task rewards, and train-only text trajectories |

Binary and categorical questions use the same output format. Categorical options must be mutually exclusive and collectively exhaustive; the agent does not need to select one winning option.

```json
{"action": "submit", "probabilities": {"A": 0.50, "B": 0.30, "C": 0.20}}
```

## Status

This is an alpha research prototype published by [Simreal-AI](https://github.com/Simreal-AI/future-prediction-bench). It includes probability validation and scoring, SQLite persistence, immutable submissions, research observation snapshots, train/test isolation, automated MLB/USGS collection and resolution, private baseline sealing, an analyst toolkit, and grouped RL data preparation. It also includes a separate real-world task contract, isolated Docker and QEMU/HVF coding adapters, filesystem-only Docker branches, full-state VM checkpoints, bounded episode scheduling, a same-host interaction/verifier pipeline, closed point-in-time evidence replay, and stage timing. Pure offline candidate evaluation supports a future skill-improvement loop. Source-specific operating rules are in [SOURCES.md](SOURCES.md).

The [six-system source-status table](OFFICIAL_SOURCE_STATUS.md) and [official-code integration audit](OFFICIAL_CODE_INTEGRATION_AUDIT.md) distinguish available code, direct upstream API calls, and independently implemented analogues. A pinned [SWE-MiniSandbox Flask task run](../examples/official_mini_sandbox/OFFICIAL_TASK_RUN.md) used the official namespace deployment, SWE-ReX session, `post_init`, and SWE-bench grader with an explicitly operator-prepared dependency cache. Its host-private strict checker observed all 60 expected tests: the public base passed 59 and scored 0; the official reference patch passed 60 and scored 1. This is one task and no model rollout. A separate [ROLL protocol, runner, and manager guard](../examples/official_roll/README.md) executed pinned upstream runner, `ProxyEnvManager`, and sample-tensor construction with model/infrastructure doubles and real CPU PyTorch tensors. The guards reject unverified grades, missing behavior logprobs, and empty or corrupt training samples while preserving a genuinely verified zero. No ROLL model, optimizer, or delayed-reward refill was run. The main Docker/QEMU environments and local rollout scheduler do not import SWE-MiniSandbox or ROLL. No paper's full system or published training speedup is claimed here.

Fixture runs exercise the lifecycle with synthetic data. They are not live forecasting results or evidence of model learning. Text trajectory exports carry `trainer_ready=false`: no tokenizer integration, behavior-policy token log probabilities, GPU trainer, or RL parameter update is included. A hosted leaderboard and formal statistical reporting remain future work.

New [official Crab code execution](OFFICIAL_CRAB_MICROVM.md) uses the
unchanged upstream process monitor inside an actual x86-64 QEMU/TCG guest
and the unchanged checkpoint policy on the host, with no dependency doubles.
For a controlled stopped-worker/single-file workload, three alternating
pairs reduced captures **8 → 3** and median timed workload duration
**18.101 → 16.257 s (10.2% less)**, with 33 RAM/disk restore checks and six
killed-worker recoveries. Captures remain full QEMU snapshots; no upstream
CRIU/ZFS/eBPF backend, software-task grader, or model training runs in this
probe. Failed ARM calibration and the slower legacy-RPC control remain in
the evidence. Separate [Tree-GRPO CPU advantages](../examples/official_tree_grpo/README.md)
and [VIP CPU allocation](../examples/official_vip/README.md) execute pinned
upstream numerical code. The [allocation research note](ROLLOUT_ALLOCATION_RESEARCH_20260927.md)
also documents an independent TRACE allocator and recent implementation
reviews, distinguishing tested components from future GPU training work.

The [KLPO source example](../examples/official_klpo/README.md) also executes the
unchanged author loss and native loss adapter with real CPU autograd. Its
exact auxiliary-draw expectation agrees with Full-KL gradients within
`6.67e-16`; the author toy completed four actual CPU AdamW updates. These
checks are separate from our text trajectory collector and do not establish
LLM or GPU training. [Backend configuration](MICROVM_BACKENDS.md)
distinguishes measured ARM/HVF and x86/TCG execution from unmeasured KVM.

The [AReaL source example](../examples/official_areal/README.md) imports the
unchanged author package and executes its actual rollout staleness manager.
All 45 selected official CPU tests passed. The probe exercises admission,
acceptance, rejection, and recovery of the version-dependent capacity budget;
it is separate from our collector and does not measure model or GPU throughput.

The separate [original Crab process backend](OFFICIAL_CRAB_CRIU.md)
now executes real runc/CRIU dumps, pre-dump chains, and restores in a
disposable x86 VM. Three modes passed destruction, exact eight-MiB memory
recovery, and subsequent computation. Selected full dumps reduced restore
points from six to three; incremental chains reduced retained image bytes,
but residual stack dirtiness still selected every boundary. These are
bounded process experiments with actual image inventories, distinct from
full-VM snapshots, changing workspaces, and model training.

The [paired-epoch challenge](CRIU_PAIRED_EPOCH_CHALLENGE.md) reproduces
an interleaved-write loss in the original parent chain and verifies a local
complete-process-parent adapter on real CRIU recovery. The separate
[Waypoint example](../examples/official_waypoint/README.md) compiles the whole
unchanged author package and validates actual image-store publication,
reader locks and permission-failure retry on Linux. Its byte fixtures are
distinct from saved processes or measured rollout throughput.

The [Pilot-Commit integration](PILOT_COMMIT_SOURCE_REVIEW.md) calls the
unchanged author prompt selector through normal package imports and connects
it to our persistent rollout-budget planner. Eighteen original CPU tests,
240 exhaustive selector cases and 42 owned budget/preparation checks passed.
Actual SQLite reservations, revision checks and competing-worker claims were
tested; the supplied binary outcomes are owned fixtures, without model sampling
or training. The core does not import the external trainer by default.

The [repository-outcome integration](PILOT_COMMIT_CODING_OUTCOMES.md)
now connects that selector to genuine offline Docker repository episodes.
Four scripted control episodes execute all 14 cases each; actual case outputs
drive two pilot receipts and two subsequent commit reservations in one SQLite
ledger. Claimed work costs one episode unit, duplicate dispatch does not rerun,
and artifact changes are checked before further allocation. This verifies the
execution and accounting chain, without model generation or optimizer updates.

A separate [genuine ZFS capability probe](CRAB_ZFS_CAPABILITY.md) now
executes Crab's original filesystem worker inside a disposable x86 VM.
All 65,536 damaged file bytes recover exactly through actual snapshot and
rollback. Its single observed component timings are distinct from process
recovery, full software tasks, model rollout and training throughput.

The [composite checkpoint probe](CRAB_COMPOSITE_RECOVERY.md) additionally
restores whole owned RAM, workspace bytes and a held file descriptor, then
continues exact writes and computation. Five bounded trials pass, including
8/64 MiB public-script replays. Failed original controls remain recorded;
these correctness trials do not establish a new speedup ratio.

The [CubeCoW source review](CUBE_COW_SOURCE_REVIEW.md) executes the
unchanged original filesystem library on real XFS inside an ARM/HVF microVM.
Seven author tests pass without skip. A populated 8/64 MiB fixture then checks
snapshot and branch isolation, mutation, expansion, deleted-origin recovery
and cleanup through six separate processes and 92 complete-file witnesses.
An independent Python byte oracle agrees with every hash. This is the original
filesystem component, not a reproduction of the whole CubeSandbox VMM.

The [populated latency comparison](CUBE_COW_LATENCY.md) then records all
240 method trials across twelve balanced groups against explicit byte-copy.
For the 64 MiB caller-durable checkpoint-plus-fork operation, method medians
are 110.161 ms for copying and 1.739 ms for the original CoW API. Every sample
passes data and bidirectional isolation checks; the independent verifier
checks 2,720 complete-file witnesses. The [source and verifier](../examples/official_cubecow/latency/README.md)
are included. These timings measure warm-cache filesystem operations.

## Quick start

Use Python 3.10 or newer. Run these commands from the repository root; the core runtime uses the Python standard library.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
python -m future_prediction_bench demo --output runs/my-first-demo
python -m future_prediction_bench validate examples/questions.fixture.jsonl
python -m future_prediction_bench status --db runs/my-first-demo/bench.sqlite --mode fixture
```

With Docker running and a **locally cached** image containing Python 3, test the third track offline. Pass a local image ID or tag; this command never pulls an image or calls a model:

```bash
python -m future_prediction_bench realworld-code-smoke \
  --image sha256:<your-local-image-id> --output runs/my-code-smoke
```

The fixture repairs one broken Python function in a network-disabled container, freezes the submitted files, then checks two hidden cases whose expected outputs stay outside the candidate container. The resulting reward and stage timings are in `report.json`; trusted audit and train-only text trajectory are separate files. This demonstrates the environment protocol, not trained performance. See the [real-world environment contract](REALWORLD_ENV.md) for external repository tasks.

The [pinned Boltons repair example](../examples/realworld_boltons26/README.md) exercises the same adapter on an actual public repository revision and a documented upstream defect. It checks the source archive hash, keeps the source outside the published package, and supplies baseline/fixed scripted episodes. The fix is already public, so this remains an integration fixture rather than a held-out evaluation task.

A second [Humanize 4.15.0 repair fixture](../examples/realworld_humanize/README.md) pins a distinct public repository and the documented `naturalsize()` unit-boundary defect. Real Docker `RealWorldEnv` runs scored the baseline **0 (5/14 cases)** and scripted repair **1 (14/14)** on the same immutable image with host-side verification. A later [three-pair verifier-worker A/B](measurements/realworld_humanize_verifier_workers_ab_2026-09-25.json) kept actions, observations, all case outcomes, source, image, and rewards fixed: for the repair, median hidden verification was **2.190 s with one worker versus 1.098 s with four**, and complete episodes **3.222 versus 2.130 s**. This small solved-task timing check has no model or optimizer; the source archive remains outside the public package.

The real-world track also supports trusted filesystem checkpoints and sibling rollouts from one audited prefix. On macOS/APFS, copies use `clonefile` when supported; branches start fresh containers and inherit the original action and wall-time budget. A bounded scheduler can run either independent episodes or branch suffixes concurrently. A controlled four-suffix Boltons comparison measured medians of 14.742 s for cold serial, 14.197 s for shared-prefix serial, 8.409 s for cold parallel, and 8.559 s for shared-prefix parallel (two repetitions each). The short shared prefix did **not** improve the parallel case. These are environment timings without model inference or weight updates; see the [research and measurements](REALWORLD_RESEARCH.md#mechanisms-implemented-in-this-prototype) for exact scope and paper-by-paper boundaries.

`realworld-code-pipeline` separates interaction and hidden verification into bounded same-host queues. After submission freezes a workspace, the interaction worker begins another episode while a verifier worker grades the submitted one. On three scripted Boltons episodes per run (repaired, untouched, repaired), two repetitions gave median batch times of 9.828 s serial versus 8.156 s pipelined, or 1.205× graded-episode throughput on this host. This narrow fixture measures environment collection, not model training. Reproduce it with [benchmark_pipeline.py](../examples/realworld_boltons26/benchmark_pipeline.py); the [public timing report](measurements/pipeline_boltons_v0.6.0.json) and [environment contract](REALWORLD_ENV.md) give the exact scope. The verifier queue also supports trusted bounded priority. A [controlled straggler experiment](VERIFIER_SCHEDULING.md) reduced selected fast-job completion latency while leaving full-batch throughput essentially unchanged.

The experimental `realworld-code-microvm` path runs the same bounded coding actions inside a network-disabled ARM64 Linux VM on Apple Silicon, without host-directory mounts. QEMU `savevm`/`loadvm` restores CPU, RAM, devices, and the writable guest disk before each host-private verifier case. A pinned Boltons baseline scored `0` and its scripted repair scored `1` across 14 cases through the actual CLI. The final three-pair cold/warm experiment also restored a live process and RAM state; median graded environment episodes took 8.578 versus 3.324 seconds, a 2.581× steady-state ratio. Including the first warm setup, the three-episode total improved 1.657×. The adapter supports independent full-state VM branches through `RealWorldEnv`; a scripted two-child check graded `[1, 0]` after a parent post-checkpoint write and verified separate RAM and disk state. A separate two-sibling runtime benchmark measured 1.565× graded-branch throughput over parallel cold VMs. Trusted guest-local process/overlay experiments measured 0.988 ms with a shared mount namespace and 0.864 ms with separate mount namespaces per branch; neither provides full-VM restoration or an untrusted-agent sandbox. The [microVM reproduction guide](MICROVM_ENV.md) specifies the image builder, commands, measurement boundaries, and limitations. These measurements do not include model inference or weight training.

For the pinned Boltons task, an opt-in stateless verifier keeps exact expected outputs on the host but runs each hidden case in a fresh guest mount/PID namespace and overlay workspace after a full submitted-state restore. A task/verifier/helper digest contract and process-quiescence checks gate this narrower path. Seven rotated real-guest comparisons showed exact case-result parity and a **5.433× median ratio for 14-case grading** against full-VM restore per case. Six alternating complete `RealWorldEnv` episodes improved **1.152×** because boot and preparation still dominate. A reusable full-VM template also graded four isolated sibling branches **1.306×** faster than parallel cold VMs at steady state, or **1.125×** including template setup over three pairs. These are solved public-fixture environment timings, not RL training or broad sandbox performance claims; see the [performance comparison](PERFORMANCE_COMPARISON.md).

An opt-in prepared `RealWorldEnv` template removed repeated guest boot from complete episodes on the same fixture. In final three-pair, action-and-reward-parity runs, median default full-VM episodes fell from **8.117 s cold to 3.647 s prepared** (**2.226×**), or **1.800×** after charging template setup over six episodes. With the declared stateless verifier and generic helper preinstalled, medians fell from **7.029 s to 2.122 s** (**3.313×**), or **2.197×** including setup. A matched helper-only comparison measured a smaller **1.085×** steady-state ratio (**1.032×** including setup). These are scripted environment episodes on one public solved task, without policy inference or optimizer time; the [prepared template guide](PREPARED_MICROVM_ENV.md) records the exact state and trust boundaries.

A separate five-repetition replication of the prepared-plus-stateless condition, with **10 cold and 10 prepared complete episodes**, measured **6.824 s versus 1.935 s** medians (**3.525×** steady state) and **2.591×** including its one-time setup. These overall medians combine repaired and untouched branches with different scripted action lengths; the [prepared template guide](PREPARED_MICROVM_ENV.md) gives each branch's medians. Opening, every policy action observation, all 14 hidden cases, and rewards matched within every repaired/baseline pair. The [replication report](measurements/prepared_stateless_stability_5pair_v0.6.0.json) is separate from the earlier three-pair report; samples and ratios are not pooled.

The v2 solved Boltons fixture adds an opt-in, hash-bound `replace_text` action and removes a redundant full-VM restore from the declared stateless verification path. In a separate **40-episode, five-repetition** QEMU/HVF comparison, bounded replacement plus the prepared VM and preinstalled generic verifier helper measured **6.577 s cold / 1.728 s prepared** mixed-branch medians and a **3.800× ratio of total times** at steady state. Repaired branches alone measured **6.637 / 1.752 s**. Charging the 6.610-s template setup to this method's 10 prepared episodes gives **2.751×**. All 40 episodes retained the expected 14/14 repaired or 7/14 untouched cases, rewards, observations, source digests, and independent child state. This complete-episode ratio is separate from the requested checkpoint/restore latency tolerance relative to DeltaBox and excludes model inference and optimizer time. See the [v2 protocol and run instructions](SMALL_EDIT_MICROVM.md) and [curated report](measurements/prepared_small_edit_v0.7.0.json).

A later **80-episode controlled A/B** on the same frozen v2 task removed repeated reads of the sealed template disk during spawn while retaining a full SHA-256 check of every cloned child before boot. Across 20 prepared episodes per variant, complete graded time fell from **36.632 s to 31.280 s** (**14.6% less**, **1.171× throughput**); 20 matched prepared episode deltas all favored the child-check path. The five alternating pairs preserved opening and action observations, all case results/rewards, source hashes, and independent child state. Cold controls stayed close. This reduces host provisioning work, not QEMU `savevm`/`loadvm` latency; see the [performance comparison](PERFORMANCE_COMPARISON.md) and [per-arm A/B report](measurements/prepared_template_hash_ab_v0.8.0.json).

A [trajectory control plane](ROLLART_CONTROL_PLANE.md) now keeps multi-turn `RealWorldEnv` episodes alive across bounded actor, environment, and reward queues. On one pinned Docker fixture, two alternating repetitions per condition with two reward workers and two live slots measured 1.638×, 1.568×, and 1.575× valid graded episode throughput relative to an otherwise identical serial scheduler under 0, 2, and 5 second verifier delays. A one-reward-worker configuration regressed at the 2-second delay (0.951×); queue capacity and host load matter. A later nine-pair experiment moved a bounded feedback-readiness wait out of the reward worker, but its complete graded batches were slower in all three wait conditions by median ratio (0.935×, 0.946×, 0.938×); this option remains experimental. Rewards and host evidence digests matched in every paired arm. These are scripted environment-collection results, not GPU training results; there are no behavior token log probabilities or optimizer updates.

An opt-in [policy-revision fence](ROLLOUT_REVISION_FENCE.md) rejects stale generated actions before environment dispatch and prevents mixed-revision trajectories from receiving a reward or entering text export. Offline revision-flip tests cover regeneration, mid-episode abort, and verifier races. In a stable-revision Docker A/B, three alternating pairs produced 12 fully graded episodes with identical actions, evidence, and rewards; median two-episode batches were **5.876 s default versus 6.075 s fenced**. This is a lineage safeguard with no measured throughput gain.

A [durable submitted-rollout queue](DURABLE_ROLLOUT_SERVICE.md) hands a frozen Docker episode to an independent verifier process through SQLite WAL with expiring leases, heartbeat, bounded backlog, and token-fenced reward publication. A worker-kill probe after all 14 hidden cases but before reward commit regraded on attempt 2 without a duplicate reward. Three alternating real-Docker pairs kept all 12 graded episodes' action observations, 14 case results, source hashes, and rewards equal; median two-episode batches were **9.728 s in memory versus 9.890 s durable**. An opt-in queue-local generation fence now rejects stale reward publication, including A→B→A revision changes; a separate-process real-Docker smoke graded 14/14 cases at the bound generation. This adds process-death and revision-race protection, with no measured throughput gain or optimizer step.

An opt-in [parallel full-VM template provisioner](PARALLEL_TEMPLATE_PROVISION.md) overlaps independent child qcow2 clones and full SHA-256 checks, while preserving a no-boot-before-all-verified barrier and inode-aware failure cleanup. In two alternating four-child real-QEMU pairs, setup median fell from **1.247 to 0.916 s** (1.36×), while complete 14-case graded-batch medians were **11.517 versus 11.370 s** (1.013×). All 224 case results, sources, rewards, and sibling RAM/ext4 isolation matched. The small sample does not establish an end-to-end rollout gain; the prepared `RealWorldEnv` path retains its serial default.

A [Collinear-inspired CI-triage world](COLLINEAR_CI_TRIAGE.md) exercises fresh seeded state, a discoverable tool/step interface, before/after application-state diffs, and a host-only exact-state verifier. Thirty alternating pairs preserved state and reward parity; prepared-seed copies reduced median complete simulated graded episodes from **11.114 to 9.088 ms**. Five adversarial traces received zero false rewards. This is an in-process SQLite simulation, not Collinear's service, real repository verification, VM isolation, or RL training.

The [Crab-inspired recovery journal](SEMANTIC_VM_RECOVERY.md) avoids full QEMU snapshots only for bounded read-only turns whose guest workspace and process census are unchanged. A real VM-process-kill test restarted a fresh paused QEMU from the committed qcow2 snapshot, restored RAM and a live process, and preserved all 14 case results and reward. A later four-arm run also interrupted terminal verification after the first hidden case, resumed it from the submitted VM state on the same live adapter, and obtained the identical 14-case reward without resubmitting. The selective path saved 5 instead of 8 journal snapshots but took **8.895 s versus 8.688 s** for the every-turn control in that run. Prior runs changed the timing direction, so this is a correctness and checkpoint-traffic result, not a stable end-to-end speedup; it does not implement Crab's eBPF/ZFS/CRIU stack.

A [real-QEMU terminal crash test](TERMINAL_VM_CRASH_RECOVERY.md) killed the VM after the first hidden case and restarted a matching replacement from the submitted snapshot. Its 14 case records, action observations, final source, and reward exactly matched the uninterrupted control, with one submit and no result before recovery. A mismatched kernel was rejected before boot. The [snapshot transaction guard](SNAPSHOT_TRANSACTION_GUARD.md) separately rejects inherited tag collisions, cleans a tag after a deliberately lost `savevm` acknowledgement, and quarantines an uncertain disk when cleanup fails. These are pinned-fixture correctness checks; neither measures training speed or host power-loss recovery.

A stronger [host-coordinator restart check](TERMINAL_HOST_RESTART_RECOVERY.md) exits the Python coordinator after the first hidden case, kills its orphan QEMU under a trusted supervisor, and reconstructs the submitted episode in a new Python process. The replacement completed the same 14/14 cases and reward without a second submit; a live orphan and wrong kernel were rejected before restore. This covers one same-host process restart under pinned artifacts, not host power loss. A separate [cooperative guest failure stress](../examples/cooperative_realworld/FAILURE_STRESS.md) ran adversarial cleanup probes and 40 subsequent fully graded Boltons/Humanize branches without process or workspace leakage. The [current-source full-VM primitive report](measurements/microvm_primitives_currentguard_2026-09-25.json) measured 48.47 ms median 128-MiB QEMU restore and 67.39 ms median 512-MiB restore; these operations differ from DeltaBox's process-level warm-template restore and do not establish rollout or training speed.

A separate [turn-boundary overlap experiment](ASYNC_VM_CHECKPOINT.md) runs real QEMU `savevm` and a simulated model wait concurrently, then gates the next tool action or reward on a committed snapshot and observation-bound policy choice. In the current two-pair, 100 ms-per-turn test, six overlapped turns exposed **0.006826 s** of checkpoint gate wait while retaining exact observations, 14-case results, rewards, and snapshot counts. The 0 ms control exposed **0.528095 s** of gate wait. Whole graded-episode deltas varied, so this proves the completion gate and hidden critical-path work, not stable episode or model-training throughput.

The [experimental resident guest adapter](../examples/resident_guest_candidate/README.md) uses fresh mount/PID namespaces and private overlays inside one already-booted VM for a declared stateless verifier contract. Real QEMU smokes scored the scripted repair 14/14, the untouched baseline 7/14, and both an adversarial candidate timeout and an 8-MiB file-limit violation 13/14 with reward 0 in batch and sequential transports; later cases completed and the whole outer task workspace stayed at its pinned digest. Candidate cases also inherit CPU, address-space, process, file, FD, and core-dump limits. These are single-run validation smokes, not a randomized throughput comparison or full-VM C/R. The policy remains confined to bounded read, exact edit, and submit actions, and the guest root/operator are trusted.

A separate [cooperative guest checkpoint example](../examples/cooperative_guest_checkpoint/README.md) couples a frozen OverlayFS prefix to a forked, quiescent Python process inside a real VM. Ten templates produced 100 isolated siblings; guest restore-to-ready measured **0.470 ms p50**, and the 98 no-case branch cycles including scripted work and cleanup measured **1.026 ms p50**. Two branches preserved exact host-only 14-case reward parity and the entire outer workspace digest stayed unchanged. This is a narrow single-process contract, not full-VM recovery or a learned-policy throughput result. On the existing full-VM path, [cached validation](QEMU_RESTORE_VALIDATION.md) reduced repeated digest/listing work while still calling QEMU `loadvm`; its complete restore latency remained host-variable.

The [graded cooperative RealWorldEnv adapter](../examples/cooperative_realworld/README.md) connects that process/OverlayFS pattern to complete scripted coding episodes. In a five-repetition QEMU comparison, repaired and untouched branches matched a prepared full-VM child with an opt-in stateless verifier on common action observations, all 14 host-only case outcomes, and reward across **20 graded episodes**. Cooperative complete episodes had a **0.810 s median** versus **1.647 s** for the prepared full-VM child, including cleanup; all 10 paired differences favored the cooperative arm. Counting one-time preparation, the 10-episode totals were **14.917 vs 23.315 s**. This one-fixture comparison includes different guest transports and grading implementations, has different state/isolation contracts, and no model or optimizer; its difference cannot be attributed to the checkpoint primitive alone or called general training throughput. A separate [ext4-plus-overlay workspace A/B](WORKSPACE_OVERLAY_AB.md) measured **0.0703 ms** median preparation versus **4.517 ms** for tar extraction across 20 matched pairs in one booted guest. That 64.3× ratio covers workspace preparation only, outside grading and RL training.

The same [cooperative adapter on Humanize 4.15.0](../examples/cooperative_realworld_humanize/README.md) adds a second public repository fixture. Five alternating repetitions produced 10 matched baseline/repair pairs: common read/edit observations, all 14 host-only cases, final source, and reward agreed in every pair. Complete graded episodes including cleanup had **0.619 s cooperative / 1.223 s prepared full-VM** medians (1.975× ratio of medians), and every paired difference favored the cooperative path. The paths still have different state and trust boundaries, so this is a two-fixture environment comparison, not a full-VM checkpoint-latency or learned-policy speed claim.

An [experimental read-only virtio-serial RPC](VIRTIO_GRADED_AB.md) serves bounded policy file reads. An audit found that STOP and host socket close alone could leave QEMU reporting `host=on`; those earlier graded timings remain historical unsafe-prototype evidence. The hardened path stops the agent, closes and unlinks its host socket, reconciles QEMU's stale host bit through a trusted zero-byte guest open/close, and requires exact guest/host/chardev disconnection before and after snapshot operations. Snapshot save pauses the VM and fails closed if tag cleanup cannot be verified; action-port VM template export and branching are disabled. A [final-source three-pair real-QEMU A/B](measurements/virtio_hardened_checkpoint_ab_2026-09-25.json) under the UID/GID 65534 candidate contract preserved exact opening/action observations, all 14 hidden-case results, final source, and reward in all six episodes, with one save and 14 loads per episode. Median five-read time fell from **184.497 to 20.624 ms** (8.946×), but complete episodes took **7.625 s serial versus 7.786 s virtio**; all three paired virtio episodes were slower. This is an opt-in verified transport path, not a complete-episode or training-speed improvement. The default serial path remains available.

The [full-VM candidate-identity revision](GUEST_IDENTITY_INTEGRITY.md) runs visible and hidden candidate Python as guest UID/GID 65534 and binds that identity into the task. A three-arm real-QEMU probe retained the 7/14 baseline and 14/14 repair results; an imported candidate module that attempted to kill guest PID 1 resolved at reward 0 over all 14 cases. A closed-stdout hang now receives the bounded candidate timeout instead of leaving verification pending. This narrows a concrete reward-evasion path but is not a general guest-isolation proof.

A [two-child full-VM verifier-pool proof](../examples/realworld_boltons26/FULLVM_VERIFIER_POOL_RESULTS.md) parallelized the 14 host-private cases across independent cloned QEMU guests, restoring each child's submitted snapshot before each case. Five alternating pairs per repaired/untouched branch completed 20 graded episodes with exact task, action, case, source, and reward parity. On this 8-GB host the pool was slower: median complete episodes **7.212 s serial versus 7.788 s pool**, and verification **1.477 versus 2.092 s**. The maximum sampled sum of QEMU RSS rose from **442,832 to 932,688 KiB**. This is an opt-in infrastructure proof and a measured negative result, not a default verifier or training-speed claim.

The demo output directory must be new or empty. It creates three clearly labeled synthetic questions: two train questions with two scripted RL rollouts each, and one test question with a benchmark submission. All five submissions initially await outcomes; simulated resolution scores them and exports only the four train trajectories.

Artifacts include `bench.sqlite`, `questions.jsonl`, `pending_receipts.json`, `training_trajectories.jsonl`, and `report.json`. Operational `runs/` files are ignored by Git. The 2030 fixture clock and database mode prevent synthetic runs from being treated as live results.

## Automated collection and resolution

```text
Discover future events -> freeze eligible questions -> collect private baselines
    -> model researches and submits probabilities -> wait for real outcomes
    -> resolve from original source rules -> score -> export/report
```

Collectors generate questions from reusable source templates. Resolvers later fetch authoritative outcomes and map them to frozen option IDs. The pipeline validates the seven-day horizon, deduplicates published questions, preserves raw source snapshots, and stores retry/review state in SQLite.

The default [source configuration](../configs/sources.json) enables MLB sports and USGS geophysics questions in the test split. These public-source calls do not require model or search credentials:

```bash
python -m future_prediction_bench collect \
  --config configs/sources.json --db runs/live.sqlite
python -m future_prediction_bench resolve-due \
  --config configs/sources.json --db runs/live.sqlite
python -m future_prediction_bench export-questions \
  --db runs/live.sqlite --output runs/questions.public.jsonl
```

Run collection and due resolution together, or keep the worker running in a foreground process:

```bash
python -m future_prediction_bench cycle \
  --config configs/sources.json --db runs/live.sqlite
python -m future_prediction_bench worker \
  --config configs/sources.json --db runs/live.sqlite --interval-seconds 3600
```

`--max-cycles N` bounds a worker run. By default, cycle/worker only collect and resolve; `--with-forecasts` opts into configured model calls. No persistent worker is installed or started by setup. See the [automation guide](AUTOMATION_PLAN.md) for recovery and operation.

## Optional live model run

The included runner supports a Chat Completions-compatible model endpoint and Brave Search. Configure the environment variables in [`.env.example`](../.env.example), including a separately configured fixed fallback baseline. Model and search calls may incur provider charges. The CLI reads process environment variables; it does not automatically load `.env`.

```bash
cp .env.example .env
cp configs/baselines.example.json configs/baselines.local.json
# Fill in .env with your endpoint, model IDs, and credentials before continuing.
set -a
. ./.env
set +a
python -m future_prediction_bench forecast \
  --db runs/live.sqlite --limit 20 \
  --baseline-config configs/baselines.local.json
```

The baseline example has no active market mappings. With fallback baseline credentials configured, it performs an actual fixed-model prediction without search. To use a market baseline, add a reviewed exact mapping; a matching title alone is insufficient. The runner attempts to seal a baseline before starting the evaluated agent, then lets that agent choose its own research actions.

`forecast` defaults to `--track benchmark --research-mode self_research --market-mode no_consensus --reward-mode baseline_improvement`. Missing or invalid baselines cause explicit skips in improvement mode. For an explicitly separate raw-reward experiment, use `--reward-mode negative_brier`. `--research-mode no_search` does not need a Brave key.

For repeated train trajectories, use `--track rl` only with a separately configured, nonoverlapping train cohort. The default source configuration creates test questions, which RL rejects. The model runner enforces step/call, output-token allowance, prompt-size, and wall-time limits; see `forecast --help` for overrides. Provider-reported usage is recorded, but pricing and a real parameter trainer are not included.

## Integrating an agent

`PredictionEnv` is a small dictionary-action environment. Its versioned toolkit provides `search`, `open`, `calculator`, `notebook`, `draft`, and `submit`. In `market_aware` mode it also provides public `market_search` and `market_snapshot` tools. Notes cite source hashes from the current episode; probability drafts can be revised before final submission and receive no interim outcome reward. Public market candidates are not automatically treated as matching private baselines. See [analyst tools](ANALYST_TOOLS.md).

The following illustrates the integration contract with caller-supplied `question`, `provider`, and `model_adapter` objects:

```python
from future_prediction_bench.env import PredictionEnv
from future_prediction_bench.store import Store

store = Store("runs/live.sqlite")
try:
    store.add_question(question)
    env = PredictionEnv(store, provider, max_calls=8)
    observation = env.reset(
        question["question_id"],
        policy_id="model-checkpoint-and-config-v1",
        track="benchmark",
        market_mode="no_consensus",
    )
    while True:
        transition = env.step(model_adapter.next_action(observation))
        observation = transition["observation"]
        if transition["terminated"]:
            break
finally:
    store.close()
```

The default `self_research` mode requires at least one successful permitted search. Use `research_mode="no_search"` for the separately reported no-search ablation. Change `track` to `rl` only for train questions; repeated RL episodes are allowed, while benchmark episodes are unique per question and system configuration.

The default `no_consensus` condition filters known market/odds sources and obvious consensus-probability text. `market_aware` permits them and records detected exposure. These rules are heuristic, not a guarantee against every mirror or indirect disclosure. See [adapter contracts](ADAPTERS.md).

## Probabilities, outcomes, and private baselines

All option IDs must appear exactly once, all values must be finite probabilities, and their sum must be within `1e-6` of one. Validation never silently normalizes a distribution.

```text
Brier = 0.5 * sum((p_i - 1[outcome=i])^2)
negative_brier reward = -Brier(agent)
baseline_improvement reward = Brier(baseline) - Brier(agent)
```

For binary questions, normalized Brier equals `(p_yes - y)^2`. Raw agent Brier is always retained independently of the selected reward mode. The live runner defaults to baseline improvement; the low-level environment and synthetic demo retain negative Brier for compatibility. A private baseline uses a strictly matched market distribution when available, otherwise an explicitly configured fixed internal model. It must be generated and sealed before the deadline and before any evaluated episode starts, then stay hidden from forecasting agents. A missing baseline remains missing. See [baseline eligibility and rewards](PROJECT_DESIGN.md#private-baselines-and-rewards).

Valid submissions wait with `reward=null` until a real outcome arrives. Invalid submissions receive a protocol penalty without a fabricated outcome. Missing, delayed, canceled, and void events follow their original resolution rules; unavailable data must not be converted automatically to No or zero reward.

All timestamps have explicit timezones:

```text
issued_at < forecast_deadline < outcome_not_before <= resolve_after
```

Forecast horizon and resolution latency are separate. An event targeted within seven days may take longer to settle. Review detailed timing, state, and experimental requirements in the [project design](PROJECT_DESIGN.md).

## Administrative operations

A trusted operator can submit a reviewed resolution and export train trajectories:

```bash
python -m future_prediction_bench resolve \
  --db runs/live.sqlite --resolution path/to/reviewed-resolution.json
python -m future_prediction_bench export-training \
  --db runs/live.sqlite --output runs/train-trajectories.jsonl
```

Resolution JSON includes `question_id`, `status` (`resolved` or `void`), `outcome` (an option ID or `null`), `evidence_urls`, and `evidence_text`. At least one evidence host must match a preregistered source. That structural check is not proof that the evidence establishes the outcome. Repeated identical resolutions are idempotent; conflicting outcomes are rejected.

Current reports group by policy/configuration, track, market condition, research condition, reward mode, and split. They average rollouts within each question and then weight questions equally. Invalid/missed assignments receive task loss 1 only when the question validly resolves. Formal studies still need complete preregistered assignments, baseline coverage, stratified reporting, and event-cluster-aware uncertainty estimates.

## RL collection and self-improvement

Run a richer, entirely offline smoke exercise:

```bash
python -m future_prediction_bench rl-smoke --output runs/my-rl-smoke
```

It exercises eight scripted analyst trajectories across binary and categorical questions, keeps them pending before resolution, then prepares two four-sample RLOO groups. It also checks stale-policy exclusion and computes an exact toy probe of standard-GRPO calibration bias. This is an infrastructure and arithmetic check, not model training or a forecasting performance result.

For real grouped collection, use a separately registered **train cohort**, a deployed trainable policy, and explicit collection/checkpoint identifiers:

```bash
python -m future_prediction_bench forecast --db runs/train.sqlite \
  --track rl --rollouts-per-question 4 --temperature 0.7 \
  --policy-revision checkpoint-001 --collection-id train-cohort-001 \
  --baseline-config configs/baselines.local.json
# After authoritative outcomes arrive:
python -m future_prediction_bench prepare-rl --db runs/train.sqlite \
  --current-policy-revision checkpoint-001 --advantage-method rloo \
  --output runs/prepared-groups.json
```

The collector records intended group membership before inference; reruns skip existing assignments. Preparation requires complete resolved groups with matching policy/configuration, masks observations out of policy generation, and retains the exact tool-enabled requests. It rejects stale revisions instead of inventing importance ratios. The output still has `trainer_ready=false` until a real tokenizer, behavior log probabilities, and optimizer are integrated. [RL training guide](RL_TRAINING.md) explains the contract and supported ablations.

RLOO is the default advantage estimator; centered rewards without standard-deviation normalization and standard GRPO are explicit comparisons. [Research notes](RL_RESEARCH.md) explain why probability calibration and delayed policies matter more than adopting an optimizer by name. [Self-improvement notes](SELF_IMPROVEMENT.md) connect recent GEPA, Hyperagents, AREX, and Dream-RSI work to a tested, offline candidate-promotion guard. Autonomous proposal generation and parameter updates are future integrations.

For repeated historical research, `seal-evidence-pack` converts recorded successful search/open events into a hashed, point-in-time pack. `EvidencePack.provider()` can replay only exact recorded queries and URLs under a virtual pre-deadline clock; misses fail closed and no live provider is called. Market tools require their own frozen provider or must be disabled during replay. `evidence-replay-bench` measures only a synthetic slow-provider fixture. These mechanisms can reduce repeated evidence-fetch cost, but no end-to-end training speedup has been measured.

## Documentation and release references

- [Automation guide](AUTOMATION_PLAN.md): collection, resolution, retries, and operation.
- [Validation record](VALIDATION.md): executed tests, live endpoint checks, and remaining operational checks.
- [Source adapters](SOURCES.md): exact question templates and settlement rules.
- [Project design](PROJECT_DESIGN.md): protocols, private baselines, scoring, and research milestones.
- [Adapter contracts](ADAPTERS.md): model actions, research observations, and replay.
- [Analyst tools](ANALYST_TOOLS.md), [RL training contract](RL_TRAINING.md), [Qwen3-8B GPU validation plan](QWEN8B_GPU_RUNBOOK.md), [RL papers](RL_RESEARCH.md), and [self-improvement](SELF_IMPROVEMENT.md).
- [Real-world environment](REALWORLD_ENV.md), [microVM reproduction](MICROVM_ENV.md), [prepared RealWorldEnv templates](PREPARED_MICROVM_ENV.md), [guest mount namespaces](GUEST_MOUNT_NAMESPACE.md), [opt-in stateless verifier](STATELESS_VERIFIER.md), [branch-local RL preparation](BRANCH_ADVANTAGES.md), [performance comparison](PERFORMANCE_COMPARISON.md), [microVM source audit](MICROVM_RESEARCH.md), and [frontier research](REALWORLD_RESEARCH.md): the third track, infrastructure measurements, and limits of transferring published results.
- [Trajectory control plane](ROLLART_CONTROL_PLANE.md), [revision fence](ROLLOUT_REVISION_FENCE.md), [semantic VM recovery](SEMANTIC_VM_RECOVERY.md), [terminal VM crash recovery](TERMINAL_VM_CRASH_RECOVERY.md), [snapshot transaction guard](SNAPSHOT_TRANSACTION_GUARD.md), [asynchronous VM checkpoint gate](ASYNC_VM_CHECKPOINT.md), [resident guest candidate](../examples/resident_guest_candidate/README.md), [cooperative guest checkpoint](../examples/cooperative_guest_checkpoint/README.md), [graded cooperative RealWorldEnv](../examples/cooperative_realworld/README.md), [Humanize cooperative replication](../examples/cooperative_realworld_humanize/README.md), [workspace overlay A/B](WORKSPACE_OVERLAY_AB.md), [read-only virtio action transport](MICROVM_ENV.md#opt-in-read-only-virtio-action-transport), and [QEMU restore validation](QEMU_RESTORE_VALIDATION.md): measured scheduling, recovery, branch, and transport experiments with their exact trust boundaries.
- [Public project context](DISCUSSION_CONTEXT.md) and [references](REFERENCES.md): motivation and methodological sources.
- [Contributing](../CONTRIBUTING.md) and [release checklist](RELEASE_CHECKLIST.md): English project conventions, validation, and publication decisions.

`pilot_config.json` is a study-planning draft, not the executable source configuration. The public repository is [Simreal-AI/future-prediction-bench](https://github.com/Simreal-AI/future-prediction-bench). Project-owned code is distributed under the [MIT license](../LICENSE); upstream code, data, and source-service terms retain their independent licenses and rights.
