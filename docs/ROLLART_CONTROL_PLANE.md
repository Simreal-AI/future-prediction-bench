# Trajectory-level control plane for real coding episodes

[RollArt v2](https://arxiv.org/html/2512.22560) uses a per-trajectory
environment manager to alternate model generation and environment actions,
then sends completed trajectories to an independent reward stage. Its control
plane also coordinates model weight updates and a bounded policy-version
window. The implementation here transfers the **trajectory scheduling**
mechanism into this project's `RealWorldEnv` boundary. It is deliberately
same-host and keeps one worker pool and a bounded queue for each of actor
generation, environment operations, and trusted reward verification.

`future_prediction_bench/trajectory_control.py` contains the scheduler.
Every submitted `TrajectoryJob` owns a `RealWorldEnv` lease. Reset and steps
run in the environment pool. On each visible observation, the actor receives
only the task and visible history, plus the policy revision selected for that
action; verifier evidence never enters that history. A submitted episode moves
to the reward pool, while the next trajectory can continue its own turns.
Only the scheduler places work into stage queues, so workers cannot deadlock
each other by cyclic bounded-queue writes. `max_in_flight` bounds live episode
leases, and queue capacities bound the work waiting at each stage.

A completed graded trajectory records the exact revision ID passed to each
generation call. The report applies a strict same-revision gate against the
**current** revision and rejects mixed-revision and stale trajectories. A
caller-supplied generator must actually serve the requested immutable weights
or fail; merely passing a revision string cannot attest to model weights.
Freshness can be re-evaluated after a later update with `report()`. No
behavior token log probabilities, token masks, optimizer, weight sync, GPU
placement, remote RPC, or model checkpoint are implemented; therefore every
result remains `trainer_ready=false` even when the revision gate passes.
`text_trajectories()` returns detached, policy-visible event audits for graded
train episodes; it is not an optimizer batch.

An unavailable verifier returns `pending` with `reward=null` and keeps its
lease for `resume_pending()` in the same Python process. A verifier transport
error also cannot become a zero reward. The manager closes completed and
interrupted episodes, and `close()` explicitly releases any pending leases.
If another worker fails while an environment or verifier call is still
running, lease cleanup is requested immediately but deferred until that call
returns; the adapter's own timeout must bound that wait. This avoids closing
a container or VM while its active verifier still reads it. Fault-injection
tests cover a blocked verifier and an episode that opens after shutdown.
Restart-durable recovery is outside this module: it would require an
adapter-specific persistent sandbox reference and a trusted recovery journal.

## Controlled Docker measurement

The [benchmark script](../examples/realworld_boltons26/benchmark_trajectory_control.py)
uses the same frozen Boltons 26.0.0 source, local immutable Docker image,
scripted repair/baseline actions, 14 host-private command cases, one actor
worker, one environment worker, identically configured reward workers, and
one-item stage queues in each arm. Its only scheduling difference is
`max_in_flight=1` for serial versus 2–4 live slots for the pipeline. The first episode's trusted
verifier incurs an explicit 0, 2, or 5 second delay in both arms. A/B order
alternates by repetition. The script requires exact reward and host evidence
digest equality before writing the final summary. It reports completed valid
graded episodes per hour, per-stage queue-wait p95, individual completion
times, and full batch wall time.

To reproduce offline, prepare the pinned public task and use a locally cached
ARM64 Python image; these commands do not pull an image:

```sh
python3 examples/realworld_boltons26/make_task.py \
  --sdist /path/to/boltons-26.0.0.tar.gz --output runs/rollart-control-task
PYTHONPATH=. python3 examples/realworld_boltons26/benchmark_trajectory_control.py \
  --task-dir runs/rollart-control-task --image sha256:<cached-image-id> \
  --output runs/rollart-control-ab --repetitions 2 --episode-count 4
```

`make_task.py` verifies the source archive against the pinned SHA-256
`5566d6cfd5a1e873d8e8476496287a9f92979964611ad9a9cecb6b0ef29b1edd`.

On macOS 26.2/arm64 with Docker 28.3.3 and one cached, SHA-pinned ARM64
image, two alternating repetitions per delay produced the following **batch
wall-clock medians** for four graded episodes per arm. Each cell shows serial
→ pipeline seconds and the corresponding valid graded episode throughput
ratio. Both arms in each row had the same workers, actions, verifier, and
injected delay.

| Configuration | 0 s verifier delay | 2 s | 5 s |
| --- | ---: | ---: | ---: |
| 1 reward worker, 4 live slots | 15.117 → 13.380 (1.130×) | 19.550 → 20.551 (**0.951×**) | 29.659 → 25.635 (1.157×) |
| 2 reward workers, 2 live slots | 10.952 → 6.685 (1.638×) | 13.854 → 8.838 (1.568×) | 15.938 → 10.122 (1.575×) |

The second configuration's valid graded episode rates were 1,315 → 2,154,
1,039 → 1,629, and 903 → 1,423 episodes/hour for the three delays. Its
maximum observed per-run reward-queue p95 was **0.149 ms**, while its
environment-queue p95 reached **0.335 s**. In the first configuration, the
reward queue p95 reached **22.418 s** at the 5-second delay, and the 2-second
condition was slower than serial. Adding a second reward worker and limiting
live leases prevented that measured queue buildup. The two configurations ran
at different times with different host load, so their absolute wall times are
not a controlled reward-worker ablation. Within each configuration, the
serial/pipeline comparison held the worker resources fixed.

All 12 paired A/B comparisons had exact `[1, 0, 1, 0]` rewards and equal
host-side verifier evidence digests. The [sanitized measurement record](measurements/rollart_trajectory_control_ab_2026-09-25.json)
contains each paired wall time, per-stage queue p95, task/image provenance,
and measurement limits; it excludes raw case evidence and local paths. With
only two repetitions per delay, these are pilot throughput observations, not
stable tail estimates. An additive text-trajectory audit export was added
after the throughput runs and checked separately with a real-container smoke.

The measurements concern environment collection of a **public solved
fixture** with a fixed scripted policy. They neither demonstrate new agent
coding ability nor reproduce RollArt's published GPU training-time ratio.
Model inference latency and parameter training must be measured separately
once a real versioned actor and trainer are connected.

## Experimental short readiness timer: negative throughput result

`TrajectoryControlPlane` also accepts an optional host-side
`verification_delay_seconds(job_id)` callback for waits of at most 30 seconds.
The default `verification_wait_placement="scheduler"` holds a submitted
episode within the same `max_in_flight` bound until its monotonic readiness
deadline, without occupying a reward worker. `"worker"` waits until the
**same deadline** inside the reward worker and exists as a controlled
reference. Both arms run the same verifier only after readiness. A missing
callback adds no timer. This is useful for studying short known waits; days-
long forecast outcomes must use `pending` and later `resume_pending()` rather
than keep a live lease. The timer is opt-in and does not change reward
semantics, verifier availability, or policy revision checks.

The [real Docker A/B script](../examples/realworld_boltons26/benchmark_trajectory_wait.py)
ran three alternating pairs each at 0, 2, and 5 seconds of injected readiness
delay on the first of four graded Boltons episodes. Both arms used one actor,
one environment worker, one reward worker, one verifier case worker per
episode, four in-flight slots, one-item stage queues, and the same 14-case
host verifier. The primary clock covered factory setup, all action turns,
grading, and cleanup. All **9 pairs / 18 batches** agreed exactly on action
hashes, policy revisions, reward `[1, 0, 1, 0]`, freshness gates, and host
evidence hashes. The [sanitized report](measurements/rollart_readiness_ab_2026-09-25.json)
records each paired full-batch time and per-stage work/queue diagnostics.

| Injected delay | Worker-side median | Scheduler-side median | Worker / scheduler |
| --- | ---: | ---: | ---: |
| 0 s | 13.922 s | 14.888 s | 0.935× |
| 2 s | 14.898 s | 15.745 s | 0.946× |
| 5 s | 13.485 s | 14.373 s | 0.938× |

There is **no measured throughput improvement** on this fixture. Paired
differences changed sign within every delay group, including the 0-second
control. The first repair episode typically reached the reward worker after
several seconds of verifier queueing: its median submit-to-reward-start lag
was 4.687 seconds in the 5-second worker arm, so most of the configured wait
was already consumed before that worker started it. The timer adds scheduler
complexity and is retained only as an experimental bounded control for
workloads where a known short readiness wait actually blocks reward workers.
These injected waits do not stand in for measured model inference, natural
verifier latency, or RL training speed. Reproduce with a fresh pinned task and
the same locally cached Docker image:

```sh
python3 examples/realworld_boltons26/make_task.py \
  --sdist /path/to/boltons-26.0.0.tar.gz --output runs/rollart-readiness-task
PYTHONPATH=. python3 examples/realworld_boltons26/benchmark_trajectory_wait.py \
  --task-dir runs/rollart-readiness-task --image sha256:<cached-image-id> \
  --output runs/rollart-readiness-ab --repetitions 3 --episode-count 4 \
  --delays 0 2 5 --public-output docs/measurements/rollart_readiness_ab.json
```
