# Cooperative guest checkpoint in graded RealWorldEnv episodes

This experimental adapter connects a process-and-filesystem checkpoint to
complete `RealWorldEnv` episodes for the pinned, already-solved Boltons 26.0.0
v2 repair fixture. The guest runs in one QEMU/HVF ARM64 Linux VM. A quiescent
Python template process inherits turn 17 in memory and a read-only OverlayFS
workspace with a committed prefix. For each episode it forks a child that
creates its own mount/PID namespace and writable nested OverlayFS view. The
child handles successive `read_file`, `replace_text`, and `submit` policy
actions, then runs 14 host-authored Python verifier programs in separate
per-case PID/mount namespaces over disposable tmpfs copies. This uses a copy
for the third layer because this guest kernel rejects three stacked OverlayFS
mounts. Expected outputs and reward stay on the host.

The exact [opt-in contract](boltons_contract.json) excludes tasks requiring
live background process state or shared case filesystem state. The host binds
the task, verifier, source file, assets, resident wire/edit/case helpers, this
guest service, per-case runner, and contract digests before boot. The service checks template
heap state and frozen source before and after each episode. Each child checks
the committed filesystem prefix and that a prior branch marker is absent;
the host checks the complete outer ext4 workspace tree between episodes.

The [A/B driver](benchmark.py) alternates repair and baseline episodes with a
prepared full-VM child control with an opt-in stateless verifier on the same
fixture, source, and 14 cases. The control restores a VM before grading, then
runs cases in separate guest namespaces rather than loading a VM for each
case. Both arms
share the same refreshed action and verification window and use the real host
clock. Their frozen task hashes differ because the cooperative arm declares
three tools and a shared-kernel runtime while the full-VM arm declares its
larger tool manifest and full-VM artifact binding. Each timed
episode includes child provisioning, `RealWorldEnv.reset`, policy actions,
verification, and cleanup. VM boot, helper installation, template creation,
and full-VM template preparation are measured separately. Exact read and edit
action observations, per-case return code/stdout digest/pass status, and reward
must agree. The opening observation declares different runtime/tool scope, and
`submit` declares the actual checkpoint kind in each condition. Those fields
are checked explicitly rather than erased from the report.

Generate the pinned v2 task and assets using
[the bounded-edit guide](../../docs/SMALL_EDIT_MICROVM.md). The input archive
and VM image are generated under ignored `runs/`, never bundled. Then run:

```sh
python3 -m pytest -q tests/test_cooperative_realworld.py
PYTHONPATH=. python3 -m examples.cooperative_realworld.benchmark \
  --task-dir runs/boltons-v2-task \
  --assets-dir runs/boltons-v2-assets \
  --output runs/cooperative-realworld-ab \
  --repetitions 5
```

The output directory must be new. The JSON report has no expected verifier
outputs or individual stdout. The ignored raw report retains per-case digests
for local parity debugging; `public-report.json` and the checked-in
[five-repetition measurement](../../docs/measurements/cooperative_realworld_ab_fivepair_2026-09-25.json)
retain only aggregate results and parity flags. The published
measurement has SHA-256
`f9e23eb1b4a143010469ff0ecf50de694b363db9140b9a3fee678de7fc512318`.
The earlier two-pair
[smoke report](../../docs/measurements/cooperative_realworld_ab_smoke_2026-09-25.json)
is retained as a separate initial check.

The five-repetition QEMU/HVF run completed 10 alternating repair/baseline
pairs, or 20 graded episodes. Every pair matched exact common read/edit
observations, all 14 host-only case outcomes, and rewards: repair 14/14 and
reward 1; baseline 7/14 and reward 0. The cooperative complete-episode median
was **0.810 s** versus **1.647 s** for the prepared full VM, a 2.035×
ratio of medians. All 10 paired time differences favored the cooperative arm;
their median was **0.849 s**. These times include child creation, actions,
verification, and child teardown. One-time setup took 6.886 and 6.471 s;
across this exact 20-episode run the setup-amortized totals were 14.917 and
23.315 s (1.563× ratio). The guest checkpoint itself took 0.797 ms in this
run and is only a primitive measurement. Outer-workspace integrity
audits were run between episodes and excluded from both episode timers.
The result is still one host and one solved fixture, without model inference
or training. The timed difference also includes child provisioning, distinct
guest transports, and distinct case-execution paths; it cannot be assigned
solely to the checkpoint primitive.

This runs no policy model, optimizer, or GPU
training. The cooperative branch shares the guest kernel and trust boundary;
it does not restore CPU/device state, arbitrary background processes, open
writable file descriptors, or a durable checkpoint after a VM crash. It is
therefore a narrow functional and timing comparison, not full-VM security or
crash-recovery equivalence. Guest root and the pinned fixture are trusted.
