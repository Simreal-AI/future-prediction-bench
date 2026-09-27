# Revision fencing for local coding rollouts

## Transfer boundary

[RollArt](https://arxiv.org/html/2512.22560v2) advances trajectories independently through generation, environment steps, and rewards. Its distributed control plane also coordinates weight updates and admits trajectories within a bounded policy-version window. The local `TrajectoryControlPlane` already has bounded stage queues and a post-hoc freshness report, but a version update between action generation and environment dispatch could previously execute a stale action. The new opt-in `stale_revision_policy="fence"` checks the revision at actor completion, at environment dispatch, after an environment step, and before verification. It is an **exact-revision** rule, more conservative than RollArt's numeric staleness window; this project does not implement RollArt's weight synchronization, KV-cache recomputation, sample buffer, distributed worker placement, or training.

The [Collinear environment architecture](https://blog.collinear.ai/p/rl-env-as-a-service) separates environment data-plane steps from control-plane scheduling and policy-version routing. [Daytona's architecture](https://www.daytona.io/docs/en/architecture/) adds a persistent control plane that manages isolated sandbox lifecycles on runners. This module is a single-process scheduler over the project's existing Docker environment adapter. It has no remote environment service, cluster reconciliation, or restart-durable sandbox lease.

`future_prediction_bench/disaggregated_rollout.py` is a separate fixed-action actor/verifier batch path with bounded handoff queues and per-job policy IDs. It does not call a versioned generator between turns, so this action-dispatch fence belongs in `trajectory_control.py`. Neither path currently provides Collinear-style environment-as-a-service endpoints or Daytona-style lifecycle reconciliation.

## Contract

The default `report_only` mode preserves the prior behavior. In opt-in `fence` mode, an action generated under a revision that is no longer current is rejected before its environment step. If no action has yet been accepted, the scheduler requests regeneration under the new revision, up to `max_stale_regenerations` (default 2, maximum 16). Once any action has been accepted, a revision change ends the trajectory with `status="stale_policy_revision"` and `reward=null`; restarting it from a modified workspace would mix policy lineage. Exhausting the regeneration budget also ends it ungraded. Rejected action hashes and counts are retained for audit, while the applied-action list contains only dispatched actions.

The verifier is skipped when the revision is stale at its dispatch check. If the revision changes while verification runs, the returned evidence remains in the audit but the episode is marked stale with `reward=null`. `text_trajectories()` in fence mode also excludes an already graded audit whose action revision is no longer current at export time. A revision can still change between a check and an external environment or inference operation; the caller must provide immutable model weights for the requested revision and coordinate update handoff if it needs an atomic cross-process guarantee. Reports remain `trainer_ready=false`: visible event audits do not provide behavior token log probabilities or an optimizer.

## Verification

`tests/test_trajectory_control.py` covers first-action regeneration, a revision flip at environment dispatch, mid-episode abort without submission or reward, retry exhaustion, pre-verifier and mid-verifier flips, stable-revision reward/evidence parity, and stale graded-audit exclusion. `tests/test_benchmark_revision_fence.py` tests the A/B schedule and parity guards. Run them with:

```sh
python3 -m unittest discover -s tests -p test_trajectory_control.py -q
python3 -m unittest discover -s tests -p test_benchmark_revision_fence.py -q
```

The [Docker A/B harness](../examples/realworld_boltons26/benchmark_revision_fence.py) compares `report_only` and `fence` under a stable revision using the pinned public Boltons 26.0.0 v2 solved repair plus a baseline, the same 14 host-side regression cases, one worker per stage, two in-flight episodes, and three alternating arm pairs. Its clock covers manager and factory setup, all graded turns, verification, and cleanup. It requires exact applied-action, revision, reward, freshness, and evidence-hash parity. It tests CPU/Docker collection overhead, not model inference or RL training. Reproduce only with the pinned task archive and an already cached immutable ARM64 Python image:

```sh
PYTHONPATH=. python3 examples/realworld_boltons26/make_task_v2.py \
  --sdist /absolute/path/to/boltons-26.0.0.tar.gz \
  --output runs/revision-fence-task
PYTHONPATH=. python3 examples/realworld_boltons26/benchmark_revision_fence.py \
  --task-dir runs/revision-fence-task \
  --image sha256:8e8fad2e920379c34538b76ed81165e0c7c896605972a851830392892f282fb6 \
  --output runs/revision-fence-ab-20260925 --pairs 3
```

The local `runs/` directory is intentionally ignored; public measurement summaries exclude case payloads, case outputs, and machine-specific paths. Stable-revision timing is a correctness and overhead check, not an expected speedup. A substantive training-throughput comparison requires a real versioned actor, behavior log probabilities, a trainer, and controlled policy-update cadence.

### Measured stable-revision Docker result

On 2026-09-25, the pinned ARM64 image above ran three alternating A/B pairs, six batches, and 12 fully graded episodes. Every batch produced reward `[1.0, 0.0]` and exact per-episode equality for action hashes, policy revisions, freshness gates, and host verifier evidence hashes. Every stale-rejection count was zero, as expected with a stable revision. The primary clock included setup through cleanup, not only `manager.run()`.

| Pair | Default report-only batch | Revision-fence batch | Fence minus default |
| --- | ---: | ---: | ---: |
| 1 (default first) | 7.791 s | 5.644 s | −2.147 s |
| 2 (fence first) | 5.787 s | 6.075 s | +0.288 s |
| 3 (default first) | 5.876 s | 6.179 s | +0.304 s |
| Median | **5.876 s** | **6.075 s** | **+3.4% wall time** |

These medians correspond to 1,225 versus 1,185 valid graded episodes/hour at this two-episode batch size. The first default run was much slower than later runs; two of three pairs were slower with the fence. There is **no demonstrated speed gain**. The fence adds policy-lineage safety and had a small measured median overhead on this fixture. The [sanitized structured report](measurements/rollout_revision_fence_ab_2026-09-25.json) contains the exact source, task, fixture, image, and verifier hashes, all run timings, per-stage queue/work measurements, parity assertions, and the SHA-256 of the ignored raw report. Three pairs are insufficient to estimate a stable tail or isolate a few-percent scheduler cost from Docker host noise.
