# Original Crab process checkpoint chains with real runc and CRIU

This experiment connects the original Crab checkpoint selector to its original
process worker and runc runtime inside a disposable x86-64 Linux VM. The
checkpoint backend executes real CRIU commands. The earlier
[QEMU checkpoint experiment](OFFICIAL_CRAB_MICROVM.md) is a separate result:
its whole-VM snapshots do not establish this process-level backend.

The implementation, fixed input manifest, and reproduction commands are in
[`examples/official_crab_criu/`](../examples/official_crab_criu/README.md).
On 2026-09-27, three cyclic mode-order runs at each of eight and 64 MiB passed
all **18 ordinary recovery challenges**, including exact owned-memory hashes
and continued execution. The 64-MiB incremental mode reduced median total
probe time by 36.6%; the eight-MiB incremental mode was 8.8% slower. These are
QEMU/TCG process-fixture measurements, not native-speed or training results.
The first single-run success and unsuccessful attempts remain separate records.

## Source and execution boundary

The guest receives the complete 52-file `crab/` Python tree and 52-file
`integrations/` Python tree from commit
`9607d61a41dc44358cf078c4b438bfd971c8ee9d`. Both complete trees are hashed before
normal package imports; the file counts are source coverage, not counts of
executed modules. The checkout remains unmodified. The probe uses `CRScheduler`,
`FaultToleranceCheckpointingPolicy`, the actual kernel soft-dirty monitor,
`AdapterProcessCWorker`, and `RuncRuntime`; it supplies no capturing runner or
replacement checkpoint function. The public wrapper creates a controlled
workload and feeds kernel observations into `scheduler.evaluate`.

In the [original scheduler](https://github.com/open-agent-infra/crab/blob/9607d61a41dc44358cf078c4b438bfd971c8ee9d/crab/scheduler.py),
the policy selects a checkpoint; the incremental resolver then chooses an
anchor or a parent-backed node. The
[process worker](https://github.com/open-agent-infra/crab/blob/9607d61a41dc44358cf078c4b438bfd971c8ee9d/crab/workers/process.py)
executes a pre-dump followed by a final dump when pairing is enabled. Its
[runc runtime](https://github.com/open-agent-infra/crab/blob/9607d61a41dc44358cf078c4b438bfd971c8ee9d/crab/runtime/runc.py)
converts those calls into actual `runc checkpoint` commands and restores the
selected final image through `runc restore`.

| Provenance field | Reviewed value |
| --- | --- |
| Original Crab Python tree SHA256 | `c6d0439e627c75ecc9aea47943212ece93a900605cbac924e99d823ee44b657b` |
| Original integrations Python tree SHA256 | `fe0edc09ea062f3495807d3da0f0d5a78e5855ec3f56c426f90d092c39bf897d` |
| Actual runc | 1.4.3, Alpine `runc-1.4.3-r1.apk` |
| Actual CRIU | 4.2, Alpine `criu-4.2-r0.apk` |
| Execution backend on the development machine | x86-64 QEMU/TCG on an ARM Mac |
| Workload | One static C process, eight or 64 MiB owned private anonymous memory |

The scheduler also requests filesystem protection when process state changes.
This probe records that request but executes only the process worker. The
workload's filesystem remains unchanged after startup. There is no ZFS
checkpoint, eBPF filesystem collection, graded repository episode, model
inference, optimizer update, or GPU training in this experiment.
Real CRIU still captures the mount metadata and tmpfs contents required for
container recovery. The exact GNU tar executable and ACL shared-library
dependency are included for that operation; the owned-memory hash does not
establish recovery of an independently changing workspace.

## Paired comparison

Every mode uses the same initial state and six boundaries: initialization,
idle, mutation of the first page, idle, mutation of the last page, and idle.
The C process blocks in `sigwaitinfo` between mutations. Reading its counters,
page maps, or memory does not wake it. No `runc exec` helper or connected host
pipe is part of the checkpointed process.

| Mode | Selection input | Actual process backend |
| --- | --- | --- |
| `every_turn_full` | Every boundary forces a process-change observation | One standalone full dump per selected boundary |
| `selective_full` | Original monitor's actual kernel observations | One standalone full dump per selected boundary |
| `selective_incremental` | Original monitor's actual kernel observations | Original pre-dump/final-dump pair with retained parent chain |

Intervals and forced time-based checkpoints are disabled for this short probe.
Incremental anchor interval is eight and maximum chain length is sixteen. The
actual selection count comes from recorded decisions; an idle label alone does
not prove that all process pages remained clean.

The full modes reset the selector's soft-dirty baseline after each successful
full dump. The incremental mode leaves tracking epochs to CRIU and never clears
them externally. The worker receives no mutation signal during either part of
a pair. This controlled sequence does not validate concurrent mutation during
pre-dump/final-dump or an entire application workload. A separate
[paired-epoch challenge](CRIU_PAIRED_EPOCH_CHALLENGE.md) deliberately injects a
write between those commands. It reproduces a failed original recovery and
tests a local parent adapter; that adapter is absent from the ordinary
comparisons below and no upstream source is changed.

## Parent images and accounting

For checkpoint `cp-N`, the original worker's final dump references that same
checkpoint's `pre_dump` directory. A non-anchor pre-dump references the previous
checkpoint's pre-dump. The experiment keeps the complete chain and every
previous final image; it performs no pruning or deduplication. Relative parent
paths are also required by
[CRIU's image-directory interface](https://criu.org/CLI/opt/--prev-images-dir).

```text
cp-0/pre_dump              anchor
cp-0/process → ../pre_dump
cp-1/pre_dump → ../../cp-0/pre_dump
cp-1/process → ../pre_dump
... all intervening pre-dumps retained ...
cp-5/pre_dump → ../../cp-4/pre_dump
cp-5/process → ../pre_dump
```

Every successful ordinary incremental run retained `cp-0` through `cp-5`; every
non-anchor pre-dump referenced its immediate predecessor. Recorded commands
confirm the links above. Restoring the
latest final image needs its referenced pre-dump ancestors; keeping only the
latest directory would remove required pages.

`checkpoint_count` counts selected logical restore points. An incremental
restore point executes **two runc checkpoint commands**: pre-dump and final
dump. Both command lists are retained in each row. `new_image_bytes` sums the
logical sizes of all regular `*.img` files produced under every selected
checkpoint, including pre-dumps and final dumps. The sum includes retained
ancestors and previous final images, excludes symlinks and logs, and is not a
measurement of allocated disk blocks, total directory size, or leaf-only
storage. Storage savings must be calculated from these complete per-mode sums.
Per-image path/size inventories and exact parent symlink targets are collected
after the mode wall timer, with their sum checked against the recorded byte
count. The host also retains SHA-bound CRIU log tails for each pre-dump, dump,
and restore command; this collection occurs outside the mode wall timer.

## Network and cgroup scope

The container retains a real network namespace. Actual Alpine
`iptables-restore` and `ip6tables-restore` binaries handle CRIU's default locking
path; no custom `org.criu.config`, active network-lock option, fake binary, or
network-lock bypass is used. Preflight records executable versions and an
actual read-only nftables probe. Cgroup handling is explicitly set to `ignore`
for this process probe, so recovery of resource-controller state is outside its
scope.

Stock [runc 1.4.3](https://github.com/opencontainers/runc/blob/v1.4.3/checkpoint.go)
sets the `EmptyNs` mask to `CLONE_NEWNET`: namespace creation remains enabled,
but network device, address, and route properties are not recovered. This
original default is distinct from removing the network namespace or bypassing
locking. The workload has no configured network or established connection;
their recovery is not claimed.

## Byte-exact recovery and continued execution

A mode passes only after successful checkpoint commands and this challenge:

1. Hash **all 8,388,608 or 67,108,864 bytes** of the owned anonymous allocation through
   `/proc/PID/mem`, in exact-length one MiB chunks. Confirm its original address,
   length, page size, private writable anonymous mapping, and both counters.
2. Deliver another mutation after the saved boundary. Require changed counters
   and a different full-region hash.
3. Destroy the running container and call the original runtime's restore.
4. Require the original counter pair and full-region SHA256 to return exactly,
   including pages that were never changed after initialization.
5. Deliver both mutation signals to the restored process. Require both separate
   counters to advance again, establishing continued execution after recovery.

The checker bounds the allocation to at most 64 MiB and rejects short memory or
page-map reads and changed PID identity. The hash covers the explicitly owned
allocation, not every VMA, kernel object, filesystem, device, or network state.
Hashes, saved/damaged/restored counters, original allocation identity, and the
continued-execution result remain in JSON even when a later step fails.

The final 64-MiB comparison and both repeated epoch diagnostics require a
complete, bounded identity JSON with exactly the address, byte length, and page
size before proceeding. Counter readiness additionally requires the x86-64
worker to be blocked in `rt_sigtimedwait` (syscall 128), rather than treating
file creation or counter equality alone as readiness. The earlier eight-MiB
comparison uses its separately pinned preceding checker. Each size's three
cyclic comparisons uses one frozen asset set; the two cohorts are not pooled.

The 64-MiB runs use a three-GiB guest disk. Before launching the ordinary
fixture, the checker requires actual free space of at least
`4 × 6 × owned_RAM_bytes + 64 MiB`: two full-dump modes plus two potential
full-sized images per incremental pair, with every parent and prior final
image retained. For 64 MiB this bound is 1,600 MiB. It is a conservative bound
for this fixed fixture, not a general application-storage guarantee.

## Timing interpretation

`checkpoint_ms` surrounds the original worker call and includes both commands
for a paired checkpoint. `restore_ms` surrounds the original runtime call.
Full-region hashing and the post-restore execution challenge are timed
separately, outside these primitive timers. They remain included in the mode's
total `wall_ms`, alongside inspection, launch, mutations, and destruction of the
pre-restore container. Final container cleanup runs after that wall timer and
is excluded. Image inventories and per-command log-tail collection are also
excluded. Guest setup is reported separately by the host driver.

All local VM timings are **QEMU/TCG emulation measurements**, not native x86
latencies. Reporting them in milliseconds does not establish millisecond
production recovery. Native Linux CPU and actual agent workloads are needed
before attributing training or rollout throughput gains.

## Repeated observed comparisons

The [curated measurement JSON](measurements/official_crab_criu_process_chain_2026-09-27.json)
contains all six successful ordinary runs, per-run values, source and asset
pins, actual commands, image inventories, and command log-tail records. At
each memory size, the three mode orders are `forward`, `rotate`, and `rotate2`,
so each mode occupies each sequential position once. Every mode in every run
passed exact saved/damaged/restored RAM checks and subsequent computation.
The tables report medians of three observations, not confidence intervals.
Read-only dirty-page diagnostics were disabled in these timed comparisons.

| Owned RAM | Mode | Restore points / checkpoint commands per run | Median all `*.img` bytes | Median mode wall (ms) | Median primitive restore (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| 8 MiB | `every_turn_full` | 6 / 6 | 50,533,216 | 3,005.512 | 507.293 |
| 8 MiB | `selective_full` | 3 / 3 | 25,254,358 | 1,951.626 | 488.299 |
| 8 MiB | `selective_incremental` | 6 / 12 | 8,546,235 | 3,269.563 | 317.188 |
| 64 MiB | `every_turn_full` | 6 / 6 | 402,831,786 | 9,885.774 | 2,121.535 |
| 64 MiB | `selective_full` | 3 / 3 | 201,403,600 | 6,960.774 | 2,092.548 |
| 64 MiB | `selective_incremental` | 6 / 12 | 67,266,784 | 6,268.938 | 624.074 |

Selective full dumping skips three idle restore points at both sizes. Against
the same-size every-turn full baseline, its median mode wall time is **35.1%
lower at eight MiB and 29.6% lower at 64 MiB**, with approximately 50% fewer
image bytes. Incremental mode keeps all six restore points and twelve commands.
It reduces image bytes by **83.1% and 83.3%**, respectively, but its total wall
time is **8.8% higher at eight MiB and 36.6% lower at 64 MiB**. Its primitive
restore median is 317.188 and 624.074 ms, respectively.

All three within-run 64-MiB comparisons favor each alternative over every-turn
full dumping: full/selective wall ratios are 1.44, 1.18, and 1.48;
full/incremental ratios are 1.58, 1.35, and 1.52. At eight MiB, incremental is
slightly faster in one cycle and slower in two. The larger-state gain is an
observed result of this stack and workload; it does not erase the small-state
overhead or establish general production speed. Selection-count and
incremental-image savings are separate effects, not a combined six-to-three
incremental result.

## First successful observed result, retained separately

The real run in `runs/official-crab-criu-gnutar-20260927/` passed all three modes.
It used the reviewed case-preserving guest assets, original GNU tar, a private
mount namespace, `MS_MOVE` of the ext4 mount onto `/`, and chroot through the
previously saved working directory `.`. Preflight checked the actual ext4 root,
real binaries and flags, CRIU capabilities, and soft-dirty clear/write behavior.
The [curated measurement JSON](measurements/official_crab_criu_process_chain_2026-09-27.json)
records the source pins, actual commands, observations, and verification fields.

| Mode | Logical restore points | runc checkpoint commands | All new `*.img` bytes | Mode wall time (ms) | Primitive restore (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `every_turn_full` | 6 | 6 | 50,508,626 | 2,968.484 | 505.191 |
| `selective_full` | 3 | 3 | 25,254,313 | 1,925.841 | 477.693 |
| `selective_incremental` | 6 | 12 | 8,546,235 | 3,005.971 | 304.178 |

In this single run, selective full checkpoints skipped the three idle
boundaries: **six to three restore points**, 50.0% fewer image bytes, and 35.1%
less measured mode wall time. The incremental mode retained six restore points
and executed twelve checkpoint commands. It produced **83.1% fewer image bytes**
than the every-turn full baseline and its measured primitive restore was 39.8%
shorter; its total mode wall time was 1.3% longer. These are separate observed
effects, not a combined reduction in both selection count and incremental
checkpoint cost.

All six incremental boundaries, including idle ones, still reported process
pages as soft-dirty. The result does not establish which writes caused those
signals. The probe correctly kept CRIU's tracking epoch intact rather than
clearing bits to obtain a desired skip count. Investigating this interaction is
required before claiming that selective and incremental savings compose.

For each of the three modes, all 8,388,608 owned RAM bytes had the same SHA256
before damage and after restore. The post-checkpoint damage changed that hash.
Counters followed `[42,12]` saved, `[43,12]` damaged, `[42,12]` restored, then
`[43,13]` after both continued-execution challenges. The restored process had a
new guest PID in every mode. This verifies actual destruction, recovery, and
subsequent computation, beyond successful command return codes.

Each mode ran once, sequentially, on the same x86-emulated guest. Host setup was
9.512 seconds and excluded from the per-mode table. The table is neither a
distribution of repeated trials nor evidence of native millisecond latency,
complete Crab filesystem/network recovery, graded agent rollout throughput, or
training acceleration. The repeated comparisons above extend this initial
correctness result; native Linux and actual agent measurements remain separate
validation tasks.

## Retained failure history

The earlier case-preserving run
`runs/official-crab-criu-linuxplugins-20260927/` passed preflight but failed its
first dump because the chroot's task root differed from its mount-namespace
root. The subsequent real root handoff passed that check and full mapping
collection, but BusyBox tar rejected CRIU's `--no-unquote` at the tmpfs archive
step. Both remain failed attempts with no successful checkpoint/restore claims.
The successful run uses the genuine GNU tool and its actual ACL dependency;
no backend flag was stripped or network lock bypassed.

The earlier CRIU 4.2 RPC configuration-parser failure is retained separately in
the [experiment README](../examples/official_crab_criu/README.md), alongside the
fixed-input reproduction workflow. The new successful result does not erase
those failures or alter the original upstream source pins.

The first 64-MiB attempt exhausted its 768-MiB disk while retaining all images.
An initial three-GiB series then exposed a startup race: the identity file could
exist before complete JSON was written. The checker now waits for complete
identity and quiescent signal-wait readiness, and all three orders were rerun
with the same frozen guard. These failed attempts and the two superseded
successful runs are retained under `larger_state_prior_attempts`; none enter
the six-run primary statistics.
