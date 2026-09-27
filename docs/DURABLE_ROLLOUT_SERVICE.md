# Durable same-host rollout verification

`DurableDockerRolloutQueue` adds a cross-process handoff after a Docker coding
episode submits its frozen workspace. It is an environment service building
block for agentic RL rollouts. The policy still acts through `RealWorldEnv`;
the verifier remains a trusted host process. The queue does not infer actions,
compute token log probabilities, update weights, or implement a network API.

The design transfers the stage separation and trajectory-level asynchronous
work scheduling described in [RollArt](https://www.usenix.org/conference/osdi26/presentation/gao)
and the control-plane/data-plane split discussed by
[Collinear AI](https://blog.collinear.ai/p/rl-env-as-a-service). It does not
replicate RollArt's GPU prefill/decode placement, serverless reward models,
distributed scheduler, or reported training speedups. This implementation is
a local SQLite WAL service for one Docker daemon.

## Actual recovery boundary

1. The trusted actor creates a normal `RealWorldEnv` with a
   `DockerCodingAdapter`, takes policy actions, and calls `submit`. Submission
   stops the actor container and creates a content-addressed filesystem
   snapshot. It does not preserve process or `/tmp` state.
2. `enqueue_submitted(job_id, env, policy_revision=...)` validates the frozen
   task, submission status, snapshot digest, verifier/image/seed bindings,
   source-code bindings, and path separation, then commits one immutable job
   payload to SQLite. The caller may close the actor adapter after this call.
3. An independent worker calls `process_one(worker_id,
   revision_provider=...)`. A short SQLite write transaction claims the next
   eligible job, assigns a random lease token, and increments its attempt.
   Hidden cases run outside the database lock through the existing
   `DockerCodingAdapter.verify` logic. A heartbeat extends a live lease.
4. The worker restores the trusted submitted `RealWorldEnv` metering and
   audit state and calls its canonical `verify` method. A pending result
   transactionally advances its verifier-attempt counter, cooldown, and
   private audit events. Exhausting the frozen verification budget becomes
   terminal and ungraded; it never becomes reward zero. A trusted verifier
   `void` decision is likewise terminal, retains its host-only evidence, and
   cannot be retried or exported as a training reward.
5. The worker checks the current revision again, then publishes the final
   reward and evidence in one conditional SQL update. The update requires
   its unexpired token; a replacement or expired worker cannot commit.

`max_outstanding` bounds queued plus leased jobs (default 128), and producer
backpressure raises `QueueFull`. Priority 0 to 9 orders eligible jobs before
creation order. A pending or infrastructure result is requeued with `reward`
still null; exhausted or void jobs stop retrying without a reward. The worker
returns `status="exhausted"` when the verification budget is consumed, matching
the stored terminal state. One job ID binds one payload; the episode ID and submission digest
are also unique across jobs. An identical enqueue is idempotent, while a
conflicting or duplicated-episode enqueue is rejected. Non-fixture tasks
require the persistent `RealWorldTaskRegistry` entry created by the trusted
coordinator.

```python
queue = DurableDockerRolloutQueue("trusted/rollouts.sqlite", max_outstanding=64)
# env.status == "pending" after its Docker actor called submit.
queue.enqueue_submitted("episode-001", env, policy_revision=loaded_weight_sha)
env.adapter.close()

# In a separate host worker process, using the same trusted SQLite path:
result = queue.process_one(
    "verifier-01", revision_provider=lambda: current_loaded_weight_sha
)
```

The revision is an operator-supplied identity. The two-point check rejects a
revision change before verification or reward publication, but cannot prove
that the actor actually generated its actions with those weights or make a
remote model update atomic with the SQL commit. `trainer_ready` remains false
until an external trainer supplies token-level behavior likelihoods and its
own freshness policy.

### Optional atomic queue-local revision fence

The default `revision_provider` checks are advisory across the final check to
SQL commit: the provider can change in that interval. A trusted coordinator
can additionally declare each policy's current revision in the **same**
SQLite database and require the final reward transaction to match it:

```python
queue.set_current_revision("my-policy", loaded_weight_sha)
# The actor submits under that policy ID and the same immutable weight hash.
queue.enqueue_submitted(
    "episode-001", env, policy_revision=loaded_weight_sha,
    require_queue_revision=True,
)
result = queue.process_one(
    "verifier-01",
    revision_provider=lambda: current_loaded_weight_sha,
    atomic_revision_fence=True,
)
```

`enqueue_submitted(..., require_queue_revision=True)` binds the current
generation in its own SQLite write transaction and rejects a missing or
mismatched declaration. `set_current_revision` and the reward commit also
take immediate write transactions. The commit requires both the original
revision string **and its exact enqueue-time generation**; an A→B→A cycle
therefore cannot let an old episode publish reward. It stores the matched
revision and generation in `result["queue_revision_fence"]`. A worker that
requests atomic fencing for a legacy unbound submission, or sees a changed
generation, makes the job terminal `stale`, with null reward and no final
trajectory. The worker checks before running hidden cases and again inside
the reward transaction. A revision update **after** reward commit cannot revoke the
historical reward; `get(job_id)["queue_revision_fence_current"]` reports
whether both its revision and bound generation still match at read time. A
trainer must recheck freshness when consuming a result. The string-only
`queue_revision_matches_job` field is insufficient after A→B→A. The row is a
local, trusted declaration: it is not
atomic with external model loading, inference, or weight-file publication.
The caller still must attest that `policy_revision` names the weights that
actually generated the actor's actions. An enqueue-time
`require_queue_revision=True` binding automatically enforces the atomic fence
even if a worker omits its optional flag. Only legacy unbound jobs remain
advisory by default.

The durable payload retains the original submitted actor state, including
actor-side checkpoints and metrics. The final result's `final_state` is the
restored environment after verification; its `adapter_state` describes the
new verifier adapter and does **not** reproduce actor process state or
actor-side adapter metrics. The result also includes the visible text audit
trajectory when the frozen task split is `train`. It is not optimizer-ready.

The queue, task registry, and hidden verifier directories must be host-owned, separate from
the candidate workspace and each other. The worker rechecks the snapshot
digest, immutable local image ID, seed/verifier hashes, frozen task/reward
contract, and trusted source-file SHA-256 identities. Verifier expected
outputs are never mounted into the actor. Candidate code runs under the
existing Docker adapter's no-network, read-only verifier mount and resource
bounds; this is not a multi-tenant microVM security claim.

If a verifier process dies before committing, its lease expires and another
process rechecks the same frozen submission. Hidden cases **may execute
again**; only the final reward publication is at most once. The actor-to-queue
gap before `enqueue_submitted` returns is not recovered here. SQLite uses WAL
and FULL synchronous transactions, while the submitted filesystem snapshot
is not fsynced as a transaction with the database. Thus process-death recovery
is tested; host power loss, Docker daemon loss, disk corruption, remote hosts,
external side effects, and model-worker failures remain outside this proof.
The lease uses same-host wall-clock expiry; a large clock jump can delay or
accelerate reclaim, while the token still fences reward publication.
The current SQLite schema has a terminal `void` state. Queues created by the
earlier experimental schema are rejected on open; create a new queue and
resubmit from trusted frozen actor state rather than editing SQLite rows.

## Reproduce the full graded A/B and crash probe

Build a fresh versioned task using the SHA-pinned Boltons 26.0.0 sdist and
the current replace-text helper, then run:

```bash
python3 -m examples.realworld_boltons26.benchmark_durable_rollout \
  --task-dir runs/durable-rollout-task-v2-20260925 \
  --image sha256:8e8fad2e920379c34538b76ed81165e0c7c896605972a851830392892f282fb6 \
  --output runs/durable-rollout-final-threepair-v2-20260925 --pairs 3
```

The command's `--task-dir` and immutable local Docker image ID must point to
locally prepared inputs. The script refreshes only the task's time window;
it checks the pinned sdist, seed, scripted actions, and 14 host-side cases.
Each alternating pair compares two complete graded episodes (repair and
untouched baseline) against the existing in-memory actor/verifier pipeline,
with one actor and one verifier per arm. It compares reward, every case
result, workspace and verifier/image digests, policy action sequence, and
visible action-observation digest. Wall time includes process startup,
submission, case execution, and queue overhead. The separate fault probe
kills the verifier after all 14 cases finish but before the SQL reward commit;
a replacement must grade attempt 2 and reject the old token.

The source-bound, path-free result is
[`measurements/durable_rollout_ab_2026-09-25.json`](measurements/durable_rollout_ab_2026-09-25.json).
The raw run under `runs/` is local and not part of the public materials bundle.

On the measured macOS arm64 host, the three in-memory control batches took
9.728, 9.192, and 10.320 seconds; the three durable separate-process batches
took 9.890, 17.157, and 8.814 seconds. Their median complete two-episode
walls were **9.728 s** and **9.890 s**, respectively, or **0.984×** durable
throughput relative to the control. The 17.157-second durable tail and only
three pairs prevent a speedup claim. The mechanism adds process-death
recovery with near-parity median throughput in this small local fixture; it
does not demonstrate faster RL training. All 12 A/B episodes graded with
identical repair/baseline rewards (1/0), all 14 case results per episode,
snapshot source digests, and visible action-observation digests. The fault
probe killed a worker after verifying the repair but before committing;
attempt 2 regraded the same frozen submission, and the old token could not
publish another reward.

Those three-pair timings and source hashes describe the original advisory
mode before the optional atomic fence was added. They are retained as
historical evidence for that mode, not a performance measurement of the
atomic fence. The atomic mode also has focused revision-flip and A→B→A race
tests. Its separate, source-bound real-Docker smoke is recorded in
[`measurements/durable_atomic_revision_smoke_2026-09-25.json`](measurements/durable_atomic_revision_smoke_2026-09-25.json).
To repeat it with the locally prepared pinned fixture and immutable image:

```bash
python3 -m examples.realworld_boltons26.smoke_durable_atomic_revision \
  --task-dir runs/durable-rollout-task-v2-20260925 \
  --image sha256:8e8fad2e920379c34538b76ed81165e0c7c896605972a851830392892f282fb6 \
  --output runs/durable-atomic-revision-smoke-20260925
```

The smoke ran the repaired actor, closed its container, and graded the frozen
submission in a separate verifier process. All 14 Docker host-side cases
passed, the queue published reward 1.0 at revision generation 1, and the
final-source single complete episode took 5.623 seconds. This is a correctness smoke,
not a paired performance result or an observed training speedup. The
revision-flip and ABA cases use deterministic fault injection in the focused
tests rather than a Docker timing race.
