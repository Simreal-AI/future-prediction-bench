# Trusted verifier scheduling experiment

`run_disaggregated_coding_batch()` now accepts an optional, **operator-supplied**
`verifier_priority` on each job. It is an integer from 0 (highest) to 9
(lowest), with default 5. Waiting verification jobs are selected by priority
and then by actor handoff order. With equal or omitted priorities, the queue
retains its previous FIFO behavior. A job already inside `verify()` is never
preempted. The actor and verifier worker counts and queue capacities remain
bounded.

This priority is scheduler metadata. It is validated before constructing a
task-bound adapter, never added to the task, action, policy observation, or
adapter arguments, and is recorded in the host-side batch report for audit.
The scheduler does **not** estimate hidden test duration or inspect private
verifier outcomes. An operator needs a legitimate cost or urgency hint before
using priority on a real task. In training, a partial batch can make priority
change which rewards are available first; training-data admission must still
follow the complete-batch and policy-version rules.

## Reproduce the controlled experiment

```sh
PYTHONPATH=. python3 examples/realworld_boltons26/benchmark_verifier_priority.py \
  --output runs/verifier-priority --repetitions 3
```

This is a **synthetic environment fixture** with five distinct task IDs and
fixed, controlled verifier service times: two at 300 ms and three at 10 ms.
The first slow verification begins before all other jobs are handed off;
the second slow verification is queued ahead of the three fast jobs under
FIFO. Both conditions use exactly **one actor and one verifier worker**, the
same five jobs, queue capacities of five, and the same service times.
Condition order alternates across three paired repetitions. Rewards must
equal `[1, 0, 1, 0, 1]` in every run. The priority condition assigns the
second slow job priority 9 and the fast jobs priority 0. Repetitions reuse
the same fixture and are not independent outcome samples.

On the measured Apple Silicon host, the median time from batch start to
completion of the three fast jobs fell from **0.656 s** (FIFO) to **0.343 s**
(priority), a **1.91× latency ratio**. The median complete-batch wall time
was **0.673 s** versus **0.669 s**, a **1.006× throughput ratio** within
measurement noise. Priority improved waiting time for selected jobs; it did
not make the verifier faster or establish a throughput gain. The
[machine-readable report](measurements/verifier_priority_synthetic_v0.6.0.json)
contains every run's verification order, reward, and per-job actor, queue,
verifier, and completion timings.

The design can postpone low-priority work under sustained arrivals and has
no aging or preemption. This finite-batch API eventually drains all queued
jobs, but a continuously fed service would need an aging or deadline policy
before priority could be used without starvation controls. This experiment
measures only host-side environment scheduling, not policy generation,
model inference, or RL training speed.
