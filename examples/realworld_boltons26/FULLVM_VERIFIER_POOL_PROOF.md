# Two-child full-VM verifier pool proof

This is an opt-in infrastructure experiment for the public, already-solved
Boltons v2 repository repair fixture. It does not evaluate a learned agent or
an RL optimizer. The production `MicroVMCodingAdapter` and `MicroVMRuntime`
remain unchanged. The proof subclass changes only the trusted host scheduler
after the policy submits its repair.

The serial arm restores the submitted full-VM snapshot before each of 14
host-private cases. The pool arm clones that snapshot into two independent
QEMU guests, assigns even and odd case indices to the children, restores each
child's snapshot before every assigned case, and reassembles results in the
original index order. Both arms call the same bounded Python case wrapper and
run the candidate as UID/GID 65534. Expected outputs stay on the host. Any
fork, child, cleanup, or result-index error keeps verification pending with
no reward. The submitted parent stays live but runs no hidden cases. This is
an environment scheduling proof, not a claim of equal sandbox security.

The benchmark first runs an excluded pool preflight that writes RAM (`/tmp`)
and writable ext4 markers in each child and checks that sibling and parent
VMs cannot see them. It then executes five alternating pairs for both the
scripted repair and untouched baseline: 20 measured `RealWorldEnv` episodes.
Each episode starts before the parent disk clone and ends after task reset,
scripted actions, host-private grading, source SHA attestation, VM teardown,
and disk cleanup. The report rejects any task, artifact-binding, action
observation, reward, evidence, case-order, or source-hash mismatch. Candidate
timeouts and truncated output abort the run rather than appearing as speed
improvements.

Run on a host with the pinned Boltons v2 task and ARM64 QEMU/HVF assets:

```bash
python3 -m examples.realworld_boltons26.benchmark_fullvm_verifier_pool \
  --task-dir runs/boltons-v2-task-static-20260925 \
  --assets-dir runs/microvm-assets-v2task-static-20260925 \
  --output runs/fullvm-pool-proof-example \
  --pairs 5 --memory-mib 128
```

The raw `report.json` lives under ignored `runs/`. It records the pinned v2
asset manifest binding and SHA-256 of the proof module, harness, core adapter,
and runtime; per-episode timings; full case evidence; and 100 ms samples of
the **sum of live parent and child QEMU process RSS**. That sum can double
count shared host pages and miss short peaks. The run aborts if sampling is
unavailable, sampled RSS exceeds 4 GiB, or macOS `memory_pressure -Q` reports
less than 10% free after an episode. The memory cap is 128–256 MiB per VM and
at most three live VMs; neither cap nor RSS samples prove the host is safe
under other workloads.

The harness reports full-episode and verifier-only medians. A shorter verifier
phase alone does not establish better training throughput: the full episode
includes the pool's clone, QEMU startup, and teardown costs. Results must be
reported as a negative finding if those costs erase the parallel case gain.
