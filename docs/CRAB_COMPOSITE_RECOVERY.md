# Crab: bounded process and ZFS workspace recovery

Date: 2026-09-27. Five genuine composed-recovery trials pass: three predecessor
trials at 8, 8 and 64 MiB of private RAM, plus separate public-source replays
at 8 and 64 MiB. They include exact workspace restoration, strict held-FD
identity and a subsequent correct write and computation. Two original mounted
rollback controls still fail the ledger-content witness. This extends the
v0.18 filesystem component result and the retained failed compositions.
It is a correctness result, with no new speedup ratio.

## Actual source and execution

The unchanged [Crab source](https://github.com/open-agent-infra/crab/tree/9607d61a41dc44358cf078c4b438bfd971c8ee9d)
is pinned at `9607d61a41dc44358cf078c4b438bfd971c8ee9d`, under MIT.
Normal original package imports execute `DefaultCWorker`, `DefaultRWorker`,
the process/filesystem adapters, `LocalCheckpointManager` and `RuncRuntime`.
The original runtime, ZFS, runc and CRIU commands execute in a marked disposable
x86-64 QEMU/TCG guest; they are not replaced with doubles.

The candidate filesystem adapter adds **nonforced unmount → unchanged original
rollback → mount** before the original process restore. It calls the original
filesystem adapter through `super().restore()` and returns its step unchanged.
There is no global cache drop, corrective file rewrite, forced unmount or lazy
detach. Genuine runc/container absence and no live process executing the owned
worker inode are verified before the filesystem lifecycle.

## Recorded trials

| Trial | Private RAM (MiB) | Filesystem recovery | Complete recovery | Checkpoint (ms) | Composite restore (ms) |
| --- | ---: | --- | --- | ---: | ---: |
| original_rollback_8mib | 8 | Original rollback | Fail: damaged ledger | 833.666379 | 654.870654 |
| candidate_8mib_1 | 8 | Unmount / rollback / mount | Pass | 773.047608 | 843.683029 |
| candidate_8mib_2 | 8 | Unmount / rollback / mount | Pass | 825.764808 | 891.884272 |
| candidate_64mib_1 | 64 | Unmount / rollback / mount | Pass | 1602.900274 | 3202.825527 |
| original_rollback_64mib | 64 | Original rollback | Fail: damaged ledger | 1224.387297 | 2222.480247 |
| public_replay_8mib | 8 | Unmount / rollback / mount | Pass | 802.149781 | 790.654712 |
| public_replay_64mib | 64 | Unmount / rollback / mount | Pass | 1031.911167 | 2273.584384 |

The first five rows use the predecessor source and private asset assembly.
The last two execute the promoted public probe and hardened public offline
asset builder. These are distinct source and asset cohorts; their timings
are individual observations and are not pooled. Candidate restore includes the real
lifecycle, original rollback/process restore and after-filesystem observer
reads; the original controls omit unmount/mount. Both exclude boot and
subsequent full-RAM, FD and continued-execution checks.
Checkpoint excludes subsequent witness reads. Diagnostic wall time includes
the negative control and clone observations; host-driver wall time also
includes boot/setup. Failed original controls are not successful performance
baselines. No speedup or model-rollout latency improvement is inferred.

## What is restored and verified

The worker creates a complete anonymous private allocation with deterministic
bytes and two counters, and opens a deterministic 512-byte ledger at offset
64. It uses ordinary file write/read and fsync; the ledger is not mmap'ed.
A real full composite checkpoint captures both process and ZFS filesystem
state while the sole writer is observed in its signal wait and full RAM,
workspace and FD witnesses remain stable before and after capture.

The fixture changes RAM, writes eight bytes through the held FD, changes
another ledger byte, renames and changes the mode of a saved file, and adds
a new file. It genuinely deletes the process. A process-only negative control
restores RAM while retaining the damaged workspace in every recorded trial.
That failed negative control has its own explicit witness flag.

A genuine readonly clone of the exact original snapshot is checked at three
stages: before live damage, after live damage, and after negative process
restore/deletion. All three views retain the exact saved ledger and whole
workspace. The clone is genuinely unmounted and destroyed before positive
restore; its bytes are never copied into the target dataset as a repair.

The positive check requires all owned workspace paths, types, permission
modes and file bytes to match. It hashes every byte of the complete 8/64 MiB
owned RAM allocation after real damage and restoration. Held descriptor,
target, offset, inode, device, size and namespace PID must match the saved
values. Raw mount IDs change from 73 to 75 and remain separate kernel-local
namespace evidence. In each successful trial, device number 27 is actually
reused; equality is retained, not relaxed or generally guaranteed.

Finally, the restored process writes exactly `PH000042` once at the saved
offset, advances the FD to 72, then continues its second counter to 12.
The full 512-byte result is compared, and counters end at `[42, 12]`.
The memory/workspace/FD/identity comparisons and continuation all pass.
This covers the bounded fixture, not a whole-VM snapshot or arbitrary
concurrent-writer atomicity. Workspace equality does not claim timestamp,
extended-attribute, ownership or arbitrary metadata recovery.

## Why the controls remain visible

The original controls execute actual `zfs rollback -r` and original process
restore successfully, and recover RAM, FD identity and the identity file.
However, the ledger already has its exact damaged bytes immediately after
filesystem rollback, before positive process restore. Complete recovery is
therefore false and `composed_recovery_witness_mismatch` is retained.
The successful readonly clone views show that the saved snapshot content is
correct throughout the three observed stages. They do not alone identify
the mechanism behind stale live-dataset visibility.

Exact OpenZFS 2.4.4 release source matches Alpine's pinned source-archive
SHA-512. The actual kernel package comes from the original `zfs-lts` recipe
at `7ee0b2ca6e91f07bfdec60574fd28ea61b8e9555`, without added patch sources.
The source and shipped module lack the later `zfs_rezget()` data-page
invalidation in [OpenZFS #19013](https://github.com/openzfs/zfs/pull/19013),
merged on 2026-09-03. That absence is established; it is **not established as
the cause** of this no-ledger-mmap fixture's failure. The candidate is an
explicit lifecycle experiment rather than a patched ZFS kernel.

## Reproducible source and evidence

The [curated record](measurements/official_crab_zfs_composite_recovery_2026-09-27.json)
preserves all seven full guest results, including the five successes, two
failed controls, clone/negative/FD evidence, real command and CRIU log
observations, image inventories, source hashes and asset bindings. Host paths
and base64 source-transfer payloads are excluded. Operational VM disks,
checkpoint image binaries and external source are outside the public bundle.
The earlier [component record](measurements/official_crab_zfs_capability_2026-09-27.json)
retains the preceding composition failures separately.

The predecessor trials freeze five execution-source files and check their
original and frozen bytes before and after execution. The measured 8 MiB
probe SHA-256 is `8b281b5f712f50521c27f017530ecbf64d23bfcd335f8a52b73c730d631a1c91`;
the measured 64 MiB probe is `c9b554aa92ad230466963b79439e2b3e6ab98b90f08decd29cc836e2ec44d0b3`.
Worker C bytes are identical across these cohorts and the promoted example.

The public [probe](../examples/official_crab_criu/workspace_probe.py) and
[worker](../examples/official_crab_criu/workspace_worker.c) preserve that
64 MiB predecessor's recovery logic and strict contract. Promotion changes
documentation, normal package imports and public schema/provenance metadata.
The public probe SHA-256 is `07d7cafb3a88eb0664da2c0c7b3f9e0b005c44e5285afd5c94239e00a1390030`; worker SHA-256 is
`070f3896cd58bc19e348da16983c33012a7768ce805a825c728ddea972492a59`.
Both public-source replays execute those exact bytes and separately pass at
8 and 64 MiB. The public runner also freezes the host driver and shared runtime,
binding seven source files before and after execution. It retrieves the complete
negative and positive restore logs and attempt records, verifies their hashes,
then removes the successful run's owned VM disk. Failed runs retain their disks.

The public-source asset cohort has manifest SHA-256
`952c708d24f8112e71a5d97092ecb9302ab8f1631a2324c8c459f8e5c2384057`
and root-image SHA-256
`bede609ab77143f8cbb67441a7b6a6576133db1ec8b66388c51545aa96eff7e2`.
The record preserves all five asset hashes before and after both executions,
the actual offline-build report hash, and the predecessor asset bindings.
These new results are assigned to the public probe, not retroactively to an
earlier helper or image.

Use the [public workspace runner](../examples/official_crab_criu/run_workspace_microvm.py)
with assets from the [offline ZFS guest builder](../examples/official_crab_criu/prepare_zfs_guest.py).
The [example README](../examples/official_crab_criu/README.md) describes the
input-fetch, build and execution commands. Public replays use genuine ZFS
2.4.4-1 with the matching Linux 6.18.53-0-virt kernel and modules.

Only use this fixture in an operator-created, explicitly marked disposable
x86 Linux guest with genuine matched ZFS/runc/CRIU tools and an owned `fpb…`
parent dataset. The probe requires fresh output below `/tmp`, original source
pins and 8 or 64 MiB RAM scope. It is an experimental recovery example, not a
host setup command or a general production checkpoint API. No model,
repository grader, optimizer or GPU executes in these experiments.
