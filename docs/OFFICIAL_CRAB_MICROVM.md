# Pinned official Crab code in a real QEMU guest

On 2026-09-27, an x86-64 QEMU/TCG probe executed the **unmodified upstream
process monitor** in Linux and the **unmodified upstream checkpoint policy**
on the host. Its controlled eight-action workload reduced saves from **8 to
3**. Across three alternating pairs, median timed workload duration fell
from **18.101 to 16.257 seconds (10.2% less)**. This is a state-inspection
and capture experiment, not a graded software episode or a training result.

## Source and state boundaries

The [official Crab repository](https://github.com/open-agent-infra/crab) is
public. This probe pins commit
[`9607d61a41dc44358cf078c4b438bfd971c8ee9d`](https://github.com/open-agent-infra/crab/commit/9607d61a41dc44358cf078c4b438bfd971c8ee9d)
and refuses a different or dirty checkout. Source hashes are recorded in
the measurement. No upstream file is modified or vendored, and no
dependency double is used. The reviewed entry points are
[`process_monitor.py`](https://github.com/open-agent-infra/crab/blob/9607d61a41dc44358cf078c4b438bfd971c8ee9d/crab/host_inspector/process_monitor.py)
and [`FaultToleranceCheckpointingPolicy`](https://github.com/open-agent-infra/crab/blob/9607d61a41dc44358cf078c4b438bfd971c8ee9d/crab/scheduler.py).

The guest probe tracks **one stopped worker's writable mappings and one
file**. It uses upstream Linux soft-dirty inspection for RAM and our own
bounded file fingerprint for filesystem state. The worker is stopped between
commands; the probe service itself is outside this tracked workload. This
does not exercise upstream eBPF, cgroup-wide process discovery, the LLM proxy,
or a continuously active arbitrary agent.

Every accepted capture uses QEMU's full CPU/RAM/device/qcow2 snapshot. An
upstream filesystem-only request is explicitly **promoted to a full VM
snapshot**. In this pinned policy, a process-change decision also requests
filesystem capture. No CRIU/ZFS backend, process pre-dump chain, or
incremental process capture is run. Selective capture and incremental capture
are therefore **not combined in this experiment**.

## Measurements

The [packed-RPC report](measurements/official_crab_x86_packed_threepair_2026-09-27.json)
contains all six runs, individual actions, decisions, and restore checks.
Both conditions use the same packed guest-call transport and action sequence.
The selective arm additionally performs real inspection.

| Measured quantity | Every-turn control | Official selective policy |
| --- | ---: | ---: |
| Saves per eight-action workload, each of three runs | 8 | 3 |
| Median workload duration | 18.101 s | 16.257 s |
| Median sum of QEMU save calls per workload | 1.843 s | 0.809 s |

Five saves are avoided, a **62.5% capture-count reduction**. Median aggregate
save time falls by **56.1%**. Individual paired workload ratios are
**1.100×, 1.022×, and 1.309×**; the three-pair sample does not establish a
stable tail latency or a general throughput gain. The median guest inspector
duration is **10.498 ms**. The median full-VM restore call is **229.190 ms**;
these are different primitives and must not be called a millisecond-scale
full restore.

Timed workload duration includes actions, host/guest transport, inspection
and baseline bookkeeping, policy evaluation, and saves. The initial
**16.264-second setup**, verification challenges after capture, trial cleanup,
and VM teardown are excluded. Packed RPC includes several stages in one
wall-time measurement; zero separately attributed host inspection time does
**not** mean inspection is free. There is no model inference, software test
grader, GPU execution, or optimizer step.

## Correctness and failed controls

Before each of **33 saved-state checks**, the script corrupts both RAM and
disk, restores the actual QEMU snapshot, and checks the prior worker value,
file content, and worker identity. All checks pass. Each of six runs also
kills the tracked worker, verifies that missing-state inspection fails
closed, and restores it from a saved VM state.

A known-write calibration is mandatory before any selective run. The
[ARM64 capability result](measurements/official_crab_arm_capability_failure_2026-09-27.json)
failed this calibration and aborted **before the benchmark or any skip**.
That failure motivates the separate x86-64 guest rather than an unsupported
claim that soft-dirty detection works in the earlier ARM image.

The earlier [separate-RPC control](measurements/official_crab_x86_legacy_rpc_2026-09-27.json)
also reduced saves 8 → 3 but increased workload time **46.246 → 51.223 s**.
Its additional round trips outweighed the avoided captures. It remains
published, and is not pooled with the packed-RPC sample. Packed transport is
a local implementation improvement; the fair selective comparison is the
same-transport comparison above.

Reproduction commands and pinned asset inputs are in the
[example runbook](../examples/official_crab/README.md). Running the official
CRIU/ZFS/eBPF stack requires a suitable privileged Linux host; this local
probe does not prove that backend or its crash-recovery protocol. The earlier
[conservative journal](SEMANTIC_VM_RECOVERY.md) remains a separate experiment
with a different state contract and graded-episode measurement.
