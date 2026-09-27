# Cooperative in-guest checkpoint experiment

This is a **trusted, single-process, in-guest**
[DeltaBox-inspired](https://arxiv.org/html/2605.22781) experiment on
the pinned Boltons 26.0.0 ARM64 QEMU fixture. It is not the DeltaBox kernel
patch, DeltaCR/CRIU, a durable checkpoint, or a full-VM sandbox boundary. It
is a reproducible research example, not the general `RealWorldEnv` runtime.

The guest first creates a private mount namespace and a writable OverlayFS
view of the pinned workspace. It commits a prefix file to that view and changes
an in-memory Python state variable from turn 0 to turn 17. At a quiescent
boundary it remounts the view read-only, then forks an idle template process.
The live parent changes its own heap to turn 999; each restored child still
starts with turn 17. For each branch the template forks again, the child creates
its own mount namespace and a fresh nested OverlayFS upper layer over the
frozen checkpoint view, then acknowledges readiness. The child performs either
the public `singularize("glass")` repair or an untouched baseline. Every child
checks the inherited prefix, original source digest, and absence of prior
branch markers. The template verifies its own heap and frozen source after
each child, and the guest hashes the complete outer ext4 workspace tree before
and after all branches to catch other-file contamination as well.

The host supplies the 14 pinned verifier *programs* to the guest, without their
expected outputs or rewards. The first repair and baseline branches execute
them; the host performs exact return-code/stdout comparison and removes all
per-case outputs before writing the report. The pinned fixture's stateless
case contract is used here. Cases in this prototype execute sequentially
within each branch's overlay and are not suitable for an arbitrary verifier
whose cases share writable state or leave background work behind. This run
does not pass through `RealWorldEnv` and does not contain policy inference or
optimizer steps.

Timing boundaries are deliberately narrow. Guest-clock checkpoint time starts
immediately before the read-only remount and ends when the forked template
acknowledges readiness. Guest-clock restore time includes fork, private mount
namespace creation, and nested OverlayFS mount, ending at child readiness.
Heap, prefix, source, and sibling-isolation validations complete after that
ready signal, before a branch result is accepted; their work is outside the
restore-to-ready number.
The separate no-case branch-cycle metric includes the scripted branch action,
reaping, and upper-layer cleanup. The first two branches' full verifier work
is excluded from that no-case distribution. The host also times QEMU
`load_snapshot` wrapper calls in the **same booted 512 MiB VM**, verifying both
RAM (`/tmp`) and ext4 markers after every load. That full-VM primitive
restores CPU, RAM, devices, and disk; its coverage is much broader than this
cooperative process/template operation, so their latency ratio is not a
same-contract speedup.

The final [QEMU report](../../docs/measurements/cooperative_guest_checkpoint_v0.9.0.json) has SHA-256
`b20b011dd4692398b2798e98a6f3b4caec94046544ff71c1190f7ffe848e0202`.
It records 10 independently established checkpoint templates and 100 restored
siblings in one booted guest. The raw samples, all 100 branch-source digests,
mount-namespace observations, and 30 full-VM RAM/ext4 checks are retained.
The first two branches had exact host-private grade parity: repaired **14/14,
reward 1.0**, untouched **7/14, reward 0.0**.

| Observed unit | Samples | p50 | p95 |
| --- | ---: | ---: | ---: |
| Guest checkpoint, read-only remount plus template fork to ready | 10 | 0.365 ms | 3.213 ms |
| Guest restore, fork plus private namespace/overlay to ready | 100 | 0.470 ms | 1.350 ms |
| Guest no-case branch cycle, including scripted action and cleanup | 98 | 1.026 ms | 2.895 ms |
| Host QEMU full-VM `load_snapshot` wrapper, 512 MiB | 30 | 103.795 ms | 274.949 ms |

The 10-checkpoint tail is only descriptive. The 14-case verifier makes the
first two guest branch cycles 318 ms and 271 ms; those two are
excluded from the no-case row. The complete host-observed guest experiment
call, including 10 checkpoints, 100 restores, the two 14-case checks, cleanup,
and serial framing, took **0.936 s**. VM boot took **5.976 s** separately. The
host's **12.261 s** total also includes helper transfer and 30 full-VM loads.
Those scopes cannot be divided into a model-training speedup. Linux reused
mount-namespace inode numbers after sequential children exited; the run
checks every live child namespace against the template namespace and verifies
fresh `unshare` calls plus per-branch filesystem isolation.

These measurements are from one host and should not be treated as a stable
cross-machine tail-latency estimate. In particular, the QEMU full-VM reference
varied materially between local runs.

Run offline checks first:

```sh
python3 -m unittest discover -s tests -p 'test_cooperative_guest_checkpoint.py' -v
```

Reproduce the QEMU run after preparing the pinned v2 task and assets as in
[the bounded-edit guide](../../docs/SMALL_EDIT_MICROVM.md). The inputs are
generated from the digest-checked public Boltons source distribution and are
kept under ignored `runs/`; no VM image or verifier output is bundled here.

```sh
PYTHONPATH=. python3 -m examples.cooperative_guest_checkpoint.benchmark \
  --task-dir runs/boltons-v2-task \
  --assets-dir runs/boltons-v2-assets \
  --output runs/cooperative-checkpoint-reproduction \
  --cycles 10 --restores 10 --vm-loads 30
```

The output directory must be new. QEMU/HVF and local UNIX sockets must be
permitted by the host. The report contains raw timing samples and asset
digests, but no per-case verifier outputs or personal filesystem paths.

Scope restrictions matter: the cooperative checkpoint assumes a single
quiescent Python worker, no independently persistent background processes,
and no arbitrary writable file descriptors held across the boundary. A child
can run trusted code as guest root; mount namespaces and OverlayFS alone do
not defend against hostile root. The frozen template and tmpfs upper layers
disappear if this guest is killed, so crash recovery still requires the
separate full-VM path or another durable artifact mechanism. The guest-clock
numbers cannot establish model-training throughput or transfer to a different
repository without checking its process and verifier contracts.
