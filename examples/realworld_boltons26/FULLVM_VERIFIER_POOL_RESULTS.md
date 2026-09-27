# Two-child full-VM verifier pool: measured negative result

The proof completed five alternating pairs for each of the scripted repair
and untouched baseline: 20 measured `RealWorldEnv` episodes, plus one excluded
isolation preflight. The pool did **not** accelerate this task. Its median
whole graded episode took **7.788 s**, versus **7.212 s** for the serial
full-VM verifier (8.0% slower). Verifier time rose from **1.477 s** to
**2.092 s** (41.6% slower). These figures include the same task, policy
actions, private cases, and final source attestation in both arms; no model
inference or optimizer step was run.

| Measure | Serial | Two-child pool |
| --- | ---: | ---: |
| Median whole graded episode, 10 episodes per arm | 7.212 s | 7.788 s |
| Median host-private verification | 1.477 s | 2.092 s |
| Repair episode median, 5 per arm | 7.210 s | 8.805 s |
| Untouched baseline episode median, 5 per arm | 7.213 s | 7.580 s |
| Maximum sampled sum of live QEMU RSS | 442,832 KiB | 932,688 KiB |

Across the ten pool verifications, the median fork call took **0.716 s**.
Within it, disk cloning took **0.184 s** and child start plus snapshot restore
took **0.308 s**. The two parallel worker lanes then took **1.094 s** in wall
time, and child teardown plus disk removal took **0.023 s**. The fork work
more than erased the case-level overlap on this eight-GB Mac. We should not
integrate this pool into the default verifier on the strength of this result.
The opt-in proof remains useful as a same-contract baseline for a future
resident-child or cheaper clone design.

All 20 episodes had one task SHA-256
`2fbca19203401982633ed6e7ce378058df3ca79901355428589b2daeb4b3eeb5`
and one artifact-binding SHA-256
`5f9b8dbfb23ea95fa887b27a3d71f3b2dc92e90666b0c1340c91f8c749e86cca`.
Within each branch, both arms produced identical policy observations, all 14
case results in original order, host-private evidence, final source SHA-256,
and reward (repair 1.0, baseline 0.0). Both used the existing UID/GID 65534
case wrapper and 14 full-VM case restores. The excluded preflight confirmed
independent child/parent RAM and writable ext4 markers. The parent VM remained
live during pool verification but executed no hidden case.

The raw local report is `runs/fullvm-pool-proof-final-20260925/report.json`
(SHA-256
`e6d2c6e0b3c7f3c4ce874dda87d7694fd9a0892ef3d2bc461707ce84005a4b81`).
It stays under ignored `runs/`; an earlier interrupted run is excluded. The
report binds the v2 asset manifest SHA-256
`3e853377e7a7fd53225be4b7aa905062b109d845745e8558ba52db6f01707bcd`
and records source hashes of the proof module
`0301da4f748d0437c44dcd3231dc8c6bed9ba10327ebdf63c9083c60a2b0d7ef`,
harness
`7217b61b243c429702423031c164858227c6a56296e38287f29c9bdcd8e6f639`,
core adapter, and runtime. The source and asset bindings were checked again
after the final episode. No QEMU process or qcow2 run disk remained afterward.
The publishable [structured measurement](../../docs/measurements/fullvm_verifier_pool_ab_2026-09-25.json)
contains the complete arm order, per-branch parity digests, timings, source
and asset bindings, and memory guard findings without the raw episode trace.

RSS is the maximum of 100 ms samples of the **sum** of live parent and child
QEMU process RSS, with 63–80 samples per episode. It can double-count shared
pages and miss brief peaks; it is not peak physical memory. The harness
aborted if sampling was unavailable, the sampled sum exceeded 4 GiB, or
macOS `memory_pressure -Q` fell below 10% free after an episode. None of
these guards fired. The pool is an infrastructure proof on one public solved
fixture, not a hardened sandbox, held-out evaluation, or claim about RL
training throughput.
