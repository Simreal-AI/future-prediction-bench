# Official implementation audit: sandbox and rollout code

Updated on 2026-09-27. This is a code-level provenance audit, not a claim
that this repository reproduces any upstream system in full.
The upstream source links below are pinned to exact commits fetched into
ignored local checkouts during this review. Checkout identities were
verified with Git. The checkouts are audit inputs, not copied into the public
package. The validation statements below distinguish source inspection,
direct upstream API execution, and full environment integration.

| Upstream | Official code and inspected revision | License finding |
| --- | --- | --- |
| SWE-MiniSandbox | [author repository](https://github.com/lblankl/SWE-MiniSandbox), [commit `381ada53ab35dadb342add33ff006f3157c22fb7`](https://github.com/lblankl/SWE-MiniSandbox/commit/381ada53ab35dadb342add33ff006f3157c22fb7) | [`sandboxdev/pyproject.toml` declares MIT](https://github.com/lblankl/SWE-MiniSandbox/blob/381ada53ab35dadb342add33ff006f3157c22fb7/sandboxdev/pyproject.toml); no repository-root `LICENSE` was visible in the [root listing](https://github.com/lblankl/SWE-MiniSandbox/tree/381ada53ab35dadb342add33ff006f3157c22fb7). Bundled upstream projects have their own licenses. This is insufficient basis for copying the whole repository into our release. |
| Alibaba ROLL | [official repository](https://github.com/alibaba/ROLL), [commit `192b1a01ea61c113b2deb543f7b115783038dff8`](https://github.com/alibaba/ROLL/commit/192b1a01ea61c113b2deb543f7b115783038dff8) | [Apache-2.0](https://github.com/alibaba/ROLL/blob/192b1a01ea61c113b2deb543f7b115783038dff8/LICENSE); its README notes third-party components and `NOTICE`. No upstream source is copied into this package. |
| Crab | [official repository](https://github.com/open-agent-infra/crab), [commit `9607d61a41dc44358cf078c4b438bfd971c8ee9d`](https://github.com/open-agent-infra/crab/commit/9607d61a41dc44358cf078c4b438bfd971c8ee9d) | [MIT](https://github.com/open-agent-infra/crab/blob/9607d61a41dc44358cf078c4b438bfd971c8ee9d/LICENSE); third-party notices apply to bundled components. No source is vendored. |
| Tree-GRPO | [official repository](https://github.com/AMAP-ML/Tree-GRPO), [commit `19bf3fa0c74d8b9ba70619a09a2fd480c31b9d59`](https://github.com/AMAP-ML/Tree-GRPO/commit/19bf3fa0c74d8b9ba70619a09a2fd480c31b9d59) | [Apache-2.0](https://github.com/AMAP-ML/Tree-GRPO/blob/19bf3fa0c74d8b9ba70619a09a2fd480c31b9d59/LICENSE). Exact function ASTs execute from an external checkout; no source is vendored. |
| VIP | [official repository](https://github.com/HieuNT91/VIP), [commit `f7bd18915467f50a0d8565b4f16afaef0741a96f`](https://github.com/HieuNT91/VIP/commit/f7bd18915467f50a0d8565b4f16afaef0741a96f) | No repository-root license file was established in the pinned checkout. The reviewed file is loaded externally; no source is copied into the release. |
| KLPO | [author repository](https://github.com/yifanzhang-pro/KLPO), [commit `30c0ae8c3fa8f56213d6b57bc88b18ebee8ed696`](https://github.com/yifanzhang-pro/KLPO/commit/30c0ae8c3fa8f56213d6b57bc88b18ebee8ed696) | [Apache-2.0](https://github.com/yifanzhang-pro/KLPO/blob/30c0ae8c3fa8f56213d6b57bc88b18ebee8ed696/LICENSE). The complete package is imported from the external checkout; no source is vendored. |
| AReaL | [author repository](https://github.com/areal-project/AReaL), [commit `2fad2d0e308fe631e70e971b97188ad5c5cc03cb`](https://github.com/areal-project/AReaL/commit/2fad2d0e308fe631e70e971b97188ad5c5cc03cb) | [Apache-2.0](https://github.com/areal-project/AReaL/blob/2fad2d0e308fe631e70e971b97188ad5c5cc03cb/LICENSE). Normal external-package imports; no source is vendored. |
| Waypoint | [author repository](https://github.com/Alex-XJK/waypoint), [commit `dcb6a7b0f4114f4732b9d7f1d1824dc5e4ff4a24`](https://github.com/Alex-XJK/waypoint/commit/dcb6a7b0f4114f4732b9d7f1d1824dc5e4ff4a24) | Apache-2.0 plus NOTICE; complete original package compiled externally, no source vendored. |
| Pilot-Commit | [author repository](https://github.com/databricks/pilot-commit), commit `6def20ea211fc936ed092a5a08624807c54df381` | Apache-2.0 plus NOTICE; normal external imports, no source vendored. |
| CubeCoW | [CubeSandbox v0.7.0 filesystem crate](https://github.com/TencentCloud/CubeSandbox/tree/d0081641c59822e4e5653b7462e914410b81910a/cubecow), commit `d0081641c59822e4e5653b7462e914410b81910a` | Apache-2.0 with the upstream repository's listed third-party exceptions; normal external Rust path dependency, no upstream library source vendored. |

## New allocation and filesystem components

The [Pilot-Commit integration](PILOT_COMMIT_SOURCE_REVIEW.md) executes the
unchanged selector through normal original package imports and connects it
to our guarded SQLite budget planner. Six original-selector planning calls,
42 owned budget/preparation tests, 18 unchanged official CPU tests and 240
exhaustive component configurations pass. Fixtures use owned terminal binary
outcomes; no model rollout, inference or optimizer ran. Reservation and claim
guarantees apply to callers sharing the same ledger and authoritative revisions.

The [repository outcome producer](PILOT_COMMIT_CODING_OUTCOMES.md) extends
this with actual DockerCodingAdapter verification, preclaimed episode costs
and authoritative SQLite evidence. Four genuine scripted Boltons episodes
execute 56 command cases. The original selector uses the mixed pilot outcomes
to allocate two commits; the final ledger has four spent episode units, no
reservations and no redispatch on retry. Ten additional owned tests exercise
the producer and revision guard. No model inference or optimizer executes.

The separate [Crab ZFS capability](CRAB_ZFS_CAPABILITY.md) executes the actual
filesystem checkpoint worker and direct original runtime rollback inside a
matching Linux/ZFS guest. Genuine damage and full 65,536-byte recovery pass.
It has one component timing observation, without a baseline, process recovery,
eBPF monitoring, software grader or training throughput claim.

The separate [composite recovery probe](CRAB_COMPOSITE_RECOVERY.md) now
combines original ZFS and runc/CRIU operations with a local nonforced
unmount/rollback/remount candidate. Five positive trials restore all owned
RAM, workspace bytes and held FD/device/inode/offset, followed by exact
continued writes and computation. Two failed original controls are preserved.
Two positive trials execute the public source-pinned preparation/runner chain.
The contract is a quiescent single-process writer; concurrent writer atomicity
and a comparative speedup are not established.

The [CubeCoW experiment](CUBE_COW_SOURCE_REVIEW.md) compiles the unchanged
full filesystem crate normally for native ARM64 GNU and executes seven
original reflink tests on real XFS without skips. A separate linked fixture
passes populated 8/64 MiB branch, snapshot, expansion and orphan-recovery
checks across six new processes and 92 whole-file byte/hash witnesses, with
independent Python expected hashes. This establishes the original component
and its filesystem behavior, not the complete CubeSandbox VMM or training.

## Crab: actual upstream monitor and policy, local full-VM backend

The new [Crab microVM audit](OFFICIAL_CRAB_MICROVM.md) supersedes the earlier
source-availability record. The pinned `crab/host_inspector/process_monitor.py`
runs unchanged inside a real x86-64 Linux QEMU/TCG guest; the pinned
`FaultToleranceCheckpointingPolicy` runs unchanged on the host. Neither uses
dependency doubles. The probe tracks one stopped worker's RAM and one file;
filesystem hashing is our code. An upstream filesystem-only request is
promoted explicitly to a full QEMU snapshot. Its source-bound report records
every decision, 33 successful RAM/disk restores, and six killed-worker
recoveries. An ARM known-write calibration failed and aborted before any skip.

At matched packed transport, three alternating pairs reduce captures 8 → 3
and median eight-action workload time 18.101 → 16.257 s. The earlier separate
RPC experiment regressed 46.246 → 51.223 s and remains published. These are
controlled capture workloads, not graded software episodes. That QEMU probe
does not run the original process checkpoint backend.

A separate [real runc/CRIU chain experiment](OFFICIAL_CRAB_CRIU.md) now imports
the complete unchanged Crab package and executes its actual scheduler,
process worker, pre-dump/final-dump calls, and runtime restore inside a
disposable x86-64 VM. All three modes passed actual process destruction,
byte-exact eight-MiB owned-memory recovery, and continued computation.
The first comparison selected 6/3/6 logical restore points for every-turn
full, selective full, and selective incremental respectively; incremental
pairs are two actual commands per point. All retained ancestors are counted.
Incremental residual stack dirtiness prevents a claim that both count and
image savings combine automatically. This is a controlled CPU process
experiment, separate from graded episodes or training. That process-only
probe excludes ZFS capture; the later composite probe above exercises it.
Upstream eBPF, cgroup-wide discovery, LLM proxy, and crash-safe upstream
manifest restoration remain unverified.

The [paired-epoch challenge](CRIU_PAIRED_EPOCH_CHALLENGE.md) additionally
reproduces a lost owned-memory byte when an explicit external writer runs
between genuine pre-dump and final-dump calls. The original chain restores
the old byte and fails the whole-allocation hash. A local runtime subclass
redirects only the next pre-dump to the preceding complete process image;
the same challenge then restores the changed byte, full hash, and continued
computation. The original source is unmodified. This bounded diagnostic is
separate from ordinary workload concurrency and the baseline timing table.

## Waypoint: real image-store publication and retry

The [source review](WAYPOINT_SOURCE_REVIEW.md) pins the whole original Go
repository and invokes its actual exported image-store API from an external
harness. The unchanged full author package compiles normally with verified
Go/modules in an offline, non-root Linux container. Two actual fixture cases
pass: tmpfs-to-disk copy/fsync/atomic symlink publication under reader leases,
and a genuine permission failure followed by successful retry. All 262,185
fixture bytes are verified. The final helper also passes 22 input/path guards.
The [public evidence](measurements/official_waypoint_imagestore_cpu_2026-09-27.json)
retains earlier failed attempts separately.

These are arbitrary owned image bytes, not a saved process. No CRIU restore,
detached flusher subprocess, incremental parent migration, simultaneous
flushers, power-loss recovery, agent rollout or trainer runs. The component
provides a verified basis for separating volatile and durable readiness;
our product runtime has not adopted its persistence path.

## Tree-GRPO and VIP: executed numerical components

The [Tree-GRPO bridge](../examples/official_tree_grpo/README.md) executes two
unmodified upstream advantage-function ASTs using real CPU PyTorch. Synthetic
scored tensors test both normalization modes and masking; eight optional
tests check the bridge when pinned source/dependencies are available. Other
module imports are omitted, not mocked. This does not run the distributed
trainer, collector, complete loss, or optimizer.

The [VIP probe](../examples/official_vip/README.md) executes the unmodified
`Allocator` using real NumPy/SciPy. Eight supplied success probabilities
produce `[5,11,12,12,12,4,4,4]` under an exact 64-rollout budget and 4–12 bounds.
Its predictor is not trained. The pinned package initializer imports a
missing `allocate_rollout` export; direct file execution does not establish
a working trainer entry point. Both probes use clean pinned checkouts,
record source hashes, and run without dependency doubles.

[Further research](ROLLOUT_ALLOCATION_RESEARCH_20260927.md) distinguishes the
independent TRACE allocator from these upstream calls, and records additional
source inspection and execution-edit tests without claiming deployed guards
or training gains.

## KLPO: actual CPU loss, gradient, and toy updates

The clean author checkout is pinned at
`30c0ae8c3fa8f56213d6b57bc88b18ebee8ed696`. The [project example](../examples/official_klpo/README.md)
imports the complete unchanged package and executes token/sequence MC-KL,
Full-KL references, and `klpo.molt.KLPOLoss` using real CPU PyTorch/autograd.
Exact enumeration of independent auxiliary draws agrees with reference
gradients within `6.67e-16`. Eight rejection guards, four optional project
tests, and 133 selected author tests passed; one optional backend test was
skipped. The unmodified author four-action toy completed four real AdamW
updates. Source hashes, [numerical evidence](measurements/official_klpo_loss_cpu_2026-09-27.json),
and [toy updates](measurements/official_klpo_cpu_toy_2026-09-27.json) are recorded.

This does not run the Molt backend, collect LLM responses, train in our sandbox,
or demonstrate faster learning. The [source review](KLPO_SOURCE_REVIEW.md)
records the required frozen sampler histories and auxiliary draws, the
current synchronous backend restriction, and predicted-KL budget limits.

## AReaL: actual capacity and recovery accounting

The [source review](AREAL_SOURCE_REVIEW.md) pins project version 2.1.0 and
the complete 504-file Python tree. Normal imports load 92 original modules,
including genuine namespace packages, without source extraction or doubles.
The unchanged `StalenessManager` executes its real enqueue/submit/accept/
reject and trainer-recovery bookkeeping. All **45 original CPU tests pass**.
The [public report](measurements/official_areal_staleness_cpu_2026-09-27.json)
binds to the actual hardened helper and records module/dependency hashes.

At supplied version 100, batch size 8, staleness 2 and concurrency 1000,
the recovery callback corrects the submission budget from 824 to 24. These
are capacity units, not measured concurrent LLM rollouts or a speed result.
The numerical version counter supplies a control-plane input; it loads no
policy weights. Existing exact revision fences in our trajectory control
plane remain intact. Integrating this admission accounting also requires
completion/rejection callbacks and actual token-level policy provenance;
it is not an alternative to those contracts. GPU inference, weight transfer,
loss computation and optimization have not executed in this example.

## SWE-MiniSandbox: what the implementation actually provides

The [official deployment implementation](https://github.com/lblankl/SWE-MiniSandbox/blob/381ada53ab35dadb342add33ff006f3157c22fb7/sandboxdev/swesandbox/sandbox_deployment.py)
defines `SandboxDeployment` and constructs `unshare --mount`, recursive/bind
mounts, `chroot`, and a per-sandbox `/tmp` tmpfs in its startup path. Its
[architecture guide](https://lblankl.github.io/SWE-MiniSandbox/guide/architecture/)
names the `SWEsbEnv`/SWE-Agent and SWE-Rex terminal integration, plus a SkyRL
generator that launches sandbox or container rollout processes. This is a
Linux-native environment preparation and isolation stack, not a live VM
checkpoint/restore API. The [paper](https://arxiv.org/abs/2602.11210)
reports preparation/storage comparisons; those ratios are not measurements of
our macOS/QEMU path.

Our `future_prediction_bench/guest_mount_namespace.py`,
`future_prediction_bench/guest_stateless_verifier.py`, and
`examples/resident_guest_candidate/guest_supervisor.py` independently use
Linux mount/PID namespaces and tmpfs/overlay views **inside a QEMU guest**.
The [guest namespace measurement](GUEST_MOUNT_NAMESPACE.md) isolates a
trusted fork/mount primitive; the [resident candidate](../examples/resident_guest_candidate/README.md)
adds a narrow graded `RealWorldEnv` task. This is a transferred mechanism,
not an import, fork, vendored copy, or reproduction of `SandboxDeployment`.
Our macOS host cannot invoke MiniSandbox's native Linux namespace setup
directly. A cached privileged Linux Docker container can exercise the pinned
upstream namespace and SWE-ReX terminal path, as described below. We have not
implemented its general multi-repository dependency pre-cache, full
SWE-Agent/SkyRL rollout wiring, or its published preparation/disk benchmark
with matched tasks. The QEMU guest and restricted candidate runtime also have
different trust and cost boundaries from an untrusted process running directly
on a Linux training host.

One **single-task official environment integration** now uses the pinned
`swesandbox` checkout for a real SWE-bench Verified Flask instance. It runs
the upstream deployment/session/`post_init`/grade code, with the public task
files in the actor sandbox and the expected tests and verifier patch confined
to fresh trusted verifier deployments. The [runbook](../examples/official_mini_sandbox/OFFICIAL_TASK_RUN.md)
and [source-bound result](measurements/official_mini_flask_5014_grade_2026-09-25.json)
record the one-run base/reference test discrimination and the explicit
operator-prepared dependency cache needed after the pinned upstream setup
produced an incomplete venv on this host. A general multi-repository
`RealWorldEnv` adapter, matched repair/action A/B against our Docker and guest
adapters, and SWE-Agent/SkyRL model rollouts remain unbuilt. Before including
upstream source in a public release, resolve repository-level and transitive
licenses; this bundle vendors no upstream checkout.

One narrow **direct API bridge** now exists at
[`examples/official_mini_sandbox/session_bridge.py`](../examples/official_mini_sandbox/session_bridge.py).
Given a deployment already initialized and attached by the official Linux
workflow, it imports the upstream `SandboxDeployment` and SWE-ReX `BashAction`
types, then awaits `deployment.runtime.run_in_session(BashAction(...))` for a
bounded operator-authored command. The exact call shape comes from the
[official deployment source](https://github.com/lblankl/SWE-MiniSandbox/blob/381ada53ab35dadb342add33ff006f3157c22fb7/sandboxdev/swesandbox/sandbox_deployment.py).
An offline fake-module contract test verifies that invocation and fail-closed
platform/dependency checks. The same bridge also ran against an **actual pinned
upstream** `SandboxDeployment` and SWE-ReX `LocalRuntime` in a cached arm64
Ubuntu container with `--privileged --network none`. The container created the
upstream `unshare --mount`/`chroot` session and executed an operator command
through `run_in_official_session()`. The session observed a different mount
namespace from its container host, `tmpfs` at `/tmp`, and absence of the
container's `/src` checkout; a `/tmp` file created by the session was absent
from the host sandbox tree. The recorded, source-bound
[`check_linux_session.py`](../examples/official_mini_sandbox/check_linux_session.py)
contains this exact primitive probe; the
[sanitized result](measurements/official_mini_linux_session_2026-09-25.json)
records the pinned source digest and observed namespace IDs. The
[probe README](../examples/official_mini_sandbox/README.md) identifies the
local-only image and dependencies needed to repeat it. Its config uses a
minimal terminal-bundle path to satisfy the upstream session precondition and
does not invoke `post_init()` or install a tool bundle. This bridge **does not**
build a task repository/virtualenv, expose a policy tool, run our 14-case
verifier, or measure MiniSandbox's preparation speed. The separate Flask task
run above exercises `post_init()` and native grading; the general
`RealWorldEnv` adapter remains unbuilt.

The [official-task preflight](../examples/official_mini_sandbox/OFFICIAL_TASK_PREFLIGHT.md)
selects SWE-bench Verified `pallets__flask-5014` and verifies its public base
commit. A disposable ARM64 Linux image supplied Python 3.11; an explicitly
operator-prepared cache supplied the pinned public task dependencies after an
unmodified upstream `post_init` produced an incomplete environment here.
The actual offline run observed **59/60 tests and reward 0** on the public
base, then **60/60 and reward 1** on the official reference patch. One fresh
deployment per arm took approximately 0.38 s for session creation,
6.25–6.89 s for `post_init`, and 4.16–4.21 s for trusted grading. These are
one-sample integration timings without a speed A/B. Upstream
`_calculate_reward` catches grader exceptions and turns them into zero, and
its SWE-bench evaluator has an instance skip list. Our [host-only strict grade wrapper](../examples/official_mini_sandbox/strict_grade.py)
calls `_calculate_reward_swebench` directly, rejects skip-listed or mismatched
instances, requires every expected case result, and raises an infrastructure
error instead of converting a missing installation or failed evaluation into
zero. Its interface tests and live Flask run check those gates at different
levels. No model generated the patch in this experiment.

## ROLL and RollArt: public framework versus paper deployment

The public ROLL checkout contains a real Ray-based agentic RL framework:
[`roll/pipeline/agentic/agent_runner/base.py`](https://github.com/alibaba/ROLL/blob/192b1a01ea61c113b2deb543f7b115783038dff8/roll/pipeline/agentic/agent_runner/base.py)
defines `AgentRunner.run_job(seed) -> EpisodeResult` and an
OpenAI-compatible proxy request; [`env_manager/traj_env_manager.py`](https://github.com/alibaba/ROLL/blob/192b1a01ea61c113b2deb543f7b115783038dff8/roll/pipeline/agentic/env_manager/traj_env_manager.py)
calls an environment's `reset(seed)` for `(observation, info)` and
`step(action=decoded_response)` for `(observation, reward, terminated,
truncated, info)`. Its
[`agentic_pipeline.py`](https://github.com/alibaba/ROLL/blob/192b1a01ea61c113b2deb543f7b115783038dff8/roll/pipeline/agentic/agentic_pipeline.py)
and [scheduler module](https://github.com/alibaba/ROLL/blob/192b1a01ea61c113b2deb543f7b115783038dff8/roll/distributed/scheduler/generate_scheduler.py)
coordinate rollout batches and training. The official [ROLL README](https://github.com/alibaba/ROLL)
links the RollArt paper, while the [RollArt v2 paper](https://arxiv.org/html/2512.22560v2)
describes a roughly 60,000-line research system with heterogeneous GPU
placement, serverless reward, and staleness-bounded weight updates. The
existence of ROLL source does **not** establish that every component, config,
or result of the paper's more than 3,000-GPU production deployment is public
and reproducible from that checkout.

Our `future_prediction_bench/disaggregated_rollout.py`,
`future_prediction_bench/trajectory_control.py`, and
`future_prediction_bench/durable_rollout.py` independently implement bounded
local actor/environment/verifier scheduling, policy revision fences, and a
restart-durable submitted-job queue. The
[local controlled A/B](ROLLART_CONTROL_PLANE.md) measured completed graded
Docker episodes under scripted actions and injected verifier waits. There is
no live ROLL or SkyRL **training-stack** run. The optional `swesandbox` import
is confined to the isolated terminal bridge above. A separate [ROLL GEMRunner protocol bridge](../examples/official_roll/README.md)
maps `RealWorldEnv`'s bounded JSON actions to the upstream five-value text
environment interface and emits a scalar only after trusted grading. An
optional source-level test executes the pinned, unmodified upstream
`AgentRunner`/`GEMRunner` and string-rendering utility with stubbed missing
imports and model responses: a graded submission scores 1, a not-yet-due
submission raises without a score, and the stock runner's max-step exit
returns 0 without any submission. Our local
[`guarded_runner.py`](../examples/official_roll/guarded_runner.py) subclass
rejects that false zero and an inference-error zero before returning an
`EpisodeResult`, while accepting a genuinely verified zero. The source-level
test exercises these paths against the pinned upstream methods. A local
config-loadable runner factory and
[`VerifiedProxyEnvManager`](../examples/official_roll/verified_proxy_env_manager.py)
then gate ROLL's sample-construction path. The pinned official
`ProxyEnvManager.__init__` and `run_rollout_loop` methods ran unchanged under
dependency/model stubs: the stock manager passed a pending false zero to
sample construction, while the guarded manager rejected it before any sample,
posted a `None` completion marker for the claimed episode, and surfaced the
error. A genuinely verified zero reached the sample constructor. A second
pinned-source probe also ran the upstream `formulate_rollouts` method on real
CPU PyTorch tensors using small `TensorDict`/`DataProto` doubles. The resulting
response/prompt masks, shifted logprobs, and rewards matched the synthetic
trajectory for verified zero and one. Our post-format
[`sample_guard.py`](../examples/official_roll/sample_guard.py) rejected the
upstream no-response placeholder even for a verified zero, plus malformed
sample fields, before the queue received it. The guard also requires a
recorded inference step to have one finite behavior logprob per response token;
the pinned upstream message tracker otherwise masks or zero-fills missing
logprobs. This is a
protocol/source integration check, not an installed ROLL worker or optimizer
run. ROLL's group queue can count `None` as a completed member, but this guard
does not refill a group or reassemble a delayed-reward trajectory. Our trajectory
objects have no token-level behavior log probabilities, optimizer loss masks,
Ray cluster, prefill/decode placement, actual model weight synchronization, or
full training loop. Therefore they are **not** native ROLL `DataProto` batches
and are not trainable in ROLL merely by renaming fields. This audit found no
complete ROLL integration or RollArt training-time reproduction.

A full ROLL integration should use an **installed upstream ROLL checkout**, not
copy its scheduler. Our bridge wraps one `RealWorldEnv` as ROLL's five-value
`reset(seed)`/`step(action)` environment, parses model output into a bounded
action JSON object, and invokes the trusted verifier after submit. The local
config-loadable classes match ROLL's import path shape, but no actual proxy,
Ray worker, tokenizer, or generated-token batch has run. The next step is a
real ROLL worker with frozen task and model revisions, correct sampled-token
log probabilities, and a delayed-reward or batch-refill policy.
Our `RealWorldEnv.step` returns `reward=None` until trusted `verify()`; an
unavailable or delayed result **must not** become zero reward in ROLL. The
wrapper should fail the sample closed or use an explicitly supported deferred
reward path with a durable job identity and revision fence. Our local runner
and manager guards exercise the fail-closed path at source level; they do not
implement deferred reward recovery. The acceptance gate
is a real upstream-ROLL process consuming our fixture through a supported
config, matching all 14 host-private case results/reward to the existing
adapter, retaining policy/observation token masks, and completing one actual
Qwen training step on a GPU host. None of that acceptance gate has run here.

## Claim ledger

| Claim | Current status |
| --- | --- |
| Official SWE-MiniSandbox source exists | Verified at the pinned author repository. |
| Official ROLL source exists | Verified at the pinned Alibaba repository. |
| Official Crab source exists and some code runs here | **Yes.** Original monitor/policy and original process/ZFS workers execute. Five bounded composite trials restore owned RAM/workspace/FD and continue work; the full eBPF/application stack remains unverified. |
| Official Tree-GRPO and VIP components run here | **Yes.** Exact CPU advantage functions and the direct allocator module execute with real numerical dependencies. Neither trainer runs. |
| Official KLPO loss and toy updates run here | **Yes.** Real CPU/autograd loss and reference gradients, the native loss adapter, and four unchanged author toy updates execute. No sandbox or LLM trainer runs. |
| Official AReaL capacity manager runs here | **Yes.** Normal original-package imports, capacity/recovery lifecycle and 45 original CPU tests pass. No policy weights, model inference or GPU trainer runs. |
| Official Waypoint image-store API runs here | **Yes.** Full original package compiles; actual tmpfs/disk, flock/fsync/publication and permission-retry fixtures pass. No actual CRIU or incremental parent migration is tested. |
| Official CubeCoW filesystem library runs here | **Yes.** Normal unchanged crate compilation, seven original no-skip XFS tests and the populated multi-process fixture pass. The full CubeSandbox VMM is not reproduced. |
| Original Pilot-Commit selector receives real repository outcomes | **Yes.** Four offline Docker control episodes execute 56 actual cases and drive preclaimed pilot/commit accounting. No model or trainer runs. |
| Our code directly invokes upstream code | **Yes, within separate measured scopes:** SWE-MiniSandbox/SWE-ReX session and one task; Crab process inspection, policy and actual runc/CRIU/ZFS workers; CubeCoW filesystem APIs; Pilot-Commit selection from repository outcomes; Tree-GRPO advantage functions; VIP allocator; KLPO loss and toy updates; AReaL capacity accounting. ROLL runner/manager methods still use dependency/model stubs. No full upstream trainer integration is established. |
| Our namespace mechanism is equivalent to MiniSandbox's full environment preparation | **No.** It is a narrower guest-local analogue. |
| Our local trajectory pipeline reproduces RollArt's training system or reported speedup | **No.** It only tests a local environment scheduling mechanism. |
| A public repository contains every component of RollArt's reported production deployment | **Not established** by the paper and inspected ROLL repository. |

The audit used official GitHub/paper/docs pages, exact-commit local Git
checkouts, source inspection, pinned ROLL runner/manager/CPU tensor probes,
actual Crab guest inspection/policy probes, Tree-GRPO/VIP/KLPO CPU execution, and a
privileged, network-isolated Linux smoke of the SWE-MiniSandbox session path.
Those results establish the narrow calls described above; they do not measure
the papers' reported speedups or establish a complete trainer integration.

The [populated CubeCoW latency trial](CUBE_COW_LATENCY.md) uses the complete original library, balanced method order and explicit byte-copy controls on native ARM/HVF. All twelve groups and 240 samples remain retained, including separate operation-return and caller-durable boundaries; 2,720 full-file witnesses are independently verified. The 64 MiB caller-durable checkpoint-plus-fork comparison has method medians 110.160542 ms and 1.739000 ms. This establishes populated filesystem primitive latency in the measured cohort.
