# Opt-in stateless verifier experiment

This experiment speeds up grading for a narrow coding-task contract. The
normal `MicroVMCodingAdapter` restores a full QEMU VM before every hidden
case. The optional verifier instead runs each fixed Python case in a fresh
guest process, mount namespace, PID namespace, and overlayfs workspace view.
One trusted guest call can batch all 14 cases while the host retains the
expected return codes and exact stdout bytes. It is **not** a general
replacement for full-VM restore.

## Eligibility and state boundary

The operator must provide a task-specific
[`stateless_contract.json`](../examples/realworld_boltons26/stateless_contract.json)
bound to the host verifier SHA-256. It explicitly declares that cases do not
depend on live background processes or on writes from earlier cases, and
that unprivileged execution is valid. It also requires a quiescent submitted
VM state: no policy-created process may continue modifying the shared lower
workspace while cases run. The host rejects a missing contract,
a changed verifier, unsupported commands, or a declaration whose shape and
bindings do not match the pinned fixture. The operator remains responsible
for the semantic statelessness claim. The public Boltons fixture is the only
task validated here. Do not opt an arbitrary SWE task into this path based
solely on command shape.

The trusted host uploads the helper and **only Python case source**, never
expected outputs, to root-owned guest files before checkpointing. The case
source file is mode `0600` and checked by SHA-256 on each batch call. Policy
tools expose only the repository workspace, not these guest helper files.
Contract validation rejects source strings whose UTF-8/base64 representation
exceeds the guest command bound or whose combined upload exceeds 100 KiB,
before starting a VM. This opt-in path accepts at most 18 cases so the
worst-case per-case supervisor time fits under the host's 300-second guest
command cap. Exit code 124 is reserved for a case timeout and cannot be an
expected result. The stateless candidate receives 10 seconds per case; the
general full-VM runner can allow up to 25 seconds. The pinned Boltons cases
finish within the shorter bound, but a slow arbitrary task can receive a
different result under the two verifier modes and must not opt in without a
separately reviewed contract.
The trusted root helper launches Python with `-I -B` so workspace
`PYTHONPATH` and user site packages cannot initialize it before privilege
drop. Candidate subprocesses retain the normal task `PYTHONPATH` pointed at
their private overlay view.
Each case gets a fresh namespace PID 1, a private mount view, a read-only
bind mount of the base chroot and direct `/workspace`, private `/tmp`, and a
tmpfs-backed overlay workspace. The candidate runs as uid/gid 65534. A
worker-held liveness pipe and Linux parent-death signal guard keep the
namespace supervisor tied to its trusted worker. The candidate chroot in
this pinned image has **no mounted `/proc`**; the host checked that
`/proc/self` was absent from candidate code. We did not mount a host or
guest-global procfs into it.

This is still one shared Linux kernel inside one QEMU/HVF VM. It has no
comprehensive seccomp/cgroup policy or multi-tenant security audit. The
rootfs contains other paths besides the workspace, and future images could
introduce writable submounts requiring separate masking. The prototype's
probes cover `/workspace`, `/tmp`, `/var/tmp`, and `/dev/shm`, not every
possible path or kernel attack. The integrated adapter inventories the pinned
guest's persistent user processes at reset and rejects a new process before
submission, after helper upload, after restoring the submitted snapshot, or
after the batch. A detached writer started by the visible check caused a
pending/interrupted episode with no hidden cases or reward. This conservative
census is not a proof that arbitrary malicious code cannot evade it. Full-VM
restore remains the general verifier and the only path that restores arbitrary
guest RAM, CPU, devices, and pre-existing processes before every case.

## Reproduction and checks

Prepare the pinned Boltons task and ARM64 VM assets as described in
[MicroVM environment](MICROVM_ENV.md), then run:

```sh
python3 -m examples.realworld_boltons26.benchmark_stateless_verifier \
  --task-dir runs/boltons-task \
  --assets-dir runs/microvm-assets \
  --output runs/stateless-verifier-check \
  --repetitions 7
```

The benchmark refuses a reward or per-case return-code/stdout-hash mismatch
before recording timings. On the pinned defect, all three paths passed 7 of
14 cases before repair and 14 of 14 after repair; rewards were 0 and 1,
respectively. Both serial and batch stateless modes passed adversarial
three-case probes: overlay and `/tmp` writes were absent in the next case,
direct writes to `/workspace`, `/var/tmp`, and `/dev/shm` were denied, the
lower workspace remained unchanged, and a detached candidate sleeper was
gone after namespace PID 1 exited. These are fixture-specific checks, not
a proof of general sandbox security.

The public `realworld-code-microvm` command selects this path only with
`--stateless-verifier-contract` pointing to the exact task's host-authored
contract. Its default remains full-VM restore per hidden case. In the live
CLI probe, the original source received reward `0` (7/14 cases) and the
scripted repair received reward `1` (14/14). A separate nonprivileged visible
check started a detached writer; the submit failed closed with no reward and
no hidden grading. The [three-pair complete-episode comparison](measurements/microvm_stateless_episodes_v0.6.0.json)
alternated condition order and graded all six repair episodes `1` with 14/14
cases. Median verification time was **1.946 s** by default versus **0.451 s**
with the opt-in path. Median complete graded episodes were **8.371 s** versus
**7.267 s**: **1.152×**, including VM boot, scripted actions, helper upload,
quiescence checks, submission, and verification. No model inference or
optimizer time is included.

The guest batch response has a 64-KiB raw JSON bound. If a candidate emits
enough output to exceed it, the host reruns every case through the same
isolated helper one at a time and scores exact output bytes. A verbose wrong
answer therefore receives reward 0 rather than an infrastructure-pending
result. Evidence records `batch_fallback_used`, and the adapter counts
`stateless_batch_fallbacks`. The 5.43× benign-case batch ratio does not apply
to this fallback path; its wall time can approach the serial path.
In a [real-guest overflow correctness probe](measurements/stateless_overflow_fallback_v0.6.0.json),
an intentionally wrong candidate emitted 8 KiB of stdout per case. Both
full-VM grading and the stateless fallback returned reward `0` and exact
return-code/stdout-hash/pass parity across all 14 cases. This single probe
supports reward integrity for that failure mode, not a speed estimate.

## Same-host measurements

The [public measurement report](measurements/stateless_verifier_boltons_v0.6.0.json)
records seven rotated repetitions on one Apple M3 8-GB host running an
ARM64 QEMU/HVF guest. Every condition started from the same repaired VM
snapshot; that common initial `loadvm` is outside the timed interval. The
full-VM condition then calls `loadvm` before **each** of 14 cases; serial
stateless uses 14 host serial calls; batch stateless uses one host call but
still makes a new mount and PID namespace for every case. VM boot, helper
upload, model inference, and optimizer work are outside the timed interval.

| Grading path | Median for 14 exact host-checked cases | Ratio against full VM |
| --- | ---: | ---: |
| Full-VM restore per case | 1.825 s | 1.00× |
| Stateless, separate host call per case | 0.634 s | 2.88× |
| Stateless, one batched host call | 0.336 s | 5.43× |

This is a **graded-case wall-time** ratio for this declared stateless fixture,
not a checkpoint primitive or end-to-end RL training-speed ratio. Earlier
developmental runs on this host measured 4.963× and 6.475× with three
repetitions, and 6.195× with seven repetitions. They used evolving helper
or root-launcher revisions and are listed separately in the report, never
pooled with the final seven pairs. The full-VM path also varied within runs.
We do not treat 5.43× as a stable hardware guarantee or as a matched comparison
to DeltaBox's process-level checkpoint/restore figures.

An untimed guest-only diagnostic after the benchmark showed a 20.07-ms
median per case: 18.63 ms in candidate process startup and Python execution,
1.14 ms in outer supervision/cleanup, and 0.18 ms in namespace setup and
other non-candidate work. These components are nested clock intervals and
should not be added as independent operations; the candidate interpreter
dominates this fixture. A next scoped experiment could reuse a warmed Python
template, but it must preserve per-case file/process isolation and exact
reward parity before any speed claim.
