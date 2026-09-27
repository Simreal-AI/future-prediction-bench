# Populated CubeCoW snapshot/fork latency on native XFS

The unchanged CubeCoW library produced a 64 MiB snapshot and writable fork in
**1.314730 ms at API return**, versus **102.146354 ms** for two explicit full
copies in this cohort. With matching caller fsync completion, method medians
were **1.739000 ms versus 110.160542 ms**. These are filesystem primitive
measurements on a stated native guest, not model rollout or training speed.
All twelve groups appear below, including the smaller gains at 8 MiB.

[Complete evidence](measurements/official_cubecow_latency_2026-09-27.json)
retains all 240 raw method trials, full witness data, independently recalculated
statistics, build/source provenance and failed-attempt summaries.
[The public fixture and reproduction recipe](../examples/official_cubecow/latency/README.md)
contain the exact executed source, manifest and lock, plus the independently
executed Python verifier. The newly written manual reproduction recipe itself
has not yet built an ELF or repeated the guest measurement.

## Complete fixed result matrix

Times are method medians in milliseconds, with exactly eight measured pairs
per row. `Median ratio` is full-copy median divided by original median;
`Paired ratio` is the median of the eight paired full-copy/original ratios.
These distinct estimators must not be interchanged. The final column gives
more conservative, order-stratified descriptive bootstrap bounds using
`0.05/12` tail alpha; ordinary 95% bounds, all pairs and observed ranges also
remain in the evidence.

| MiB | Boundary | Operation | Original ms | Full-copy ms | Median ratio | Paired ratio | Adjusted paired bounds |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 8 | return | snapshot | 0.611271 | 1.310334 | 2.144 | 2.056 | 1.777–4.644 |
| 8 | return | fork | 0.649958 | 1.212729 | 1.866 | 1.965 | 1.671–2.679 |
| 8 | return | checkpoint+fork | 0.924937 | 2.292000 | 2.478 | 2.456 | 1.828–4.000 |
| 8 | caller-durable | snapshot | 0.590604 | 5.265333 | 8.915 | 9.661 | 6.995–13.339 |
| 8 | caller-durable | fork | 0.922063 | 5.406438 | 5.863 | 5.670 | 5.124–7.836 |
| 8 | caller-durable | checkpoint+fork | 1.312791 | 11.191750 | 8.525 | 9.060 | 5.052–10.610 |
| 64 | return | snapshot | 0.810667 | 35.602771 | 43.918 | 45.388 | 43.384–63.600 |
| 64 | return | fork | 0.969708 | 42.720334 | 44.055 | 42.087 | 26.263–57.106 |
| 64 | return | checkpoint+fork | 1.314730 | 102.146354 | 77.694 | 75.807 | 59.210–89.795 |
| 64 | caller-durable | snapshot | 0.901292 | 55.698687 | 61.799 | 57.508 | 47.221–77.632 |
| 64 | caller-durable | fork | 1.425021 | 61.128563 | 42.897 | 40.410 | 27.437–64.954 |
| 64 | caller-durable | checkpoint+fork | 1.739000 | 110.160542 | 63.347 | 66.081 | 49.299–79.237 |

Each group has two fixed warmup pairs and eight measured pairs, alternating
original-first/full-copy-first: four of each order in the measured set. There
are twelve groups, 48 warmup method trials and 192 measured method trials,
for 240 performed trials. All passed. No slow trial was dropped, and warmups
are excluded by the predeclared protocol rather than after inspecting times.
For eight measured values, the reported nearest-rank p95 is the maximum;
these samples do not provide a useful tail-performance guarantee.

The verifier uses 20,000 deterministic resamples of four pairs in each of the
two order strata. Bootstrap bounds are descriptive: eight serial pairs from
one guest do not establish independent population coverage or universal
speedup. The evidence preserves both ordinary bounds and the more conservative
twelve-group tail choice, together with every measured pair and observed
minimum/maximum ratio.

## Genuine original code and runtime

The library is the full normal Cargo path dependency from
[CubeSandbox v0.7.0](https://github.com/TencentCloud/CubeSandbox/tree/d0081641c59822e4e5653b7462e914410b81910a/cubecow),
commit `d0081641c59822e4e5653b7462e914410b81910a`.
The fixture invokes the author's constructor, `create_snapshot_from_volume`
and `create_volume_from_snapshot`, including genuine Linux `FICLONE`, without
patching or extracting selected library methods. All 34 original crate files,
106 original registry archive checksums and 5,214 direct vendor member bytes
were guarded by the frozen native build cohort.

A subsequent deeper GNU package-payload check, before the next cohort, found
a mismatch in the host-relocated `usr/include/linux/netfilter/xt_CONNMARK.h`,
consistent with a case collision against `xt_connmark.h` on the host filesystem.
The recorded compiler tree was unchanged during this trial, but that guard
proves stability rather than perfect reconstruction of every package member.
The actual executable, original library, runtime byte checks and timing samples
remain preserved. A fresh case-sensitive Linux assembly with complete
package-member verification is required for the next cohort; this result does
not claim that the earlier relocated sysroot is a complete byte-exact package
reconstruction.

The actual run used ARM64 HVF, one vCPU, 512 MiB guest RAM, a genuine
`6.18.52-0-virt` kernel and a 512 MiB owned regular backing file formatted as
XFS with `reflink=1`. QEMU had no network adapter or host-directory filesystem
share, and used normal flush handling rather than `cache=unsafe`. The native
GNU build used genuine Rust/Cargo 1.93.1, GCC 12.2 and glibc 2.36. The measured
ELF has SHA-256
`662740d496615e69829ab4bd2ed1688188b74b74f30392c2da5ecb6fdafae355`;
the actual GNU hash utility has SHA-256
`d72f4681a531bd27e3deaed5cbe4b1df0f87a4d69b9c12d5772712321405d709`.

The seven original reflink test bodies passed again on that genuine XFS
mount, without an unsupported-filesystem skip. The twelve other registered
configuration/S3 test names were filtered by the explicitly selected original
reflink test module, not represented as reflink coverage. The source, runtime
and frozen build-manifest hashes stayed unchanged across the accepted run.

## Comparison boundaries

Both methods start from fully populated nonzero 8/64 MiB source and prepared
reference files, with corresponding directory depths. The baseline performs
an explicit 64 KiB userspace read/write loop into a fresh independent file.
It does not use `std::fs::copy`, `copy_file_range`, a reflink-capable copy command,
a sparse placeholder or another hidden CoW path. A checkpoint plus fork
produces both the snapshot and writable branch under both methods.

At **API return**, the original timer includes its real name indexing,
metadata projection and directory-entry fsync calls. The copy wrapper mirrors
the relevant directory-entry fsync calls and has no engine index. The original
API treats those fsync errors as best-effort, while the baseline propagates
failures. Neither path adds destination data fsync at that boundary.
Consequently this is a component API comparison with explicit wrapper
cost differences, not two identical whole-VMM implementations.

At **caller-durable completion**, both paths additionally enforce `sync_all`
on every newly created data file and every ancestor directory through the
owned XFS mount root. The exact synced scope is recorded. Actual namespace
inspection rejects extra files or directories that would fall outside this
persistence contract; the original reflink namespace has no separately
persisted index. This is a caller fsync boundary, not a tested host power-loss
or storage-controller durability guarantee.

All whole-file hashes, source/reference warming, namespace inspection,
bidirectional mutation checks, restoration, cleanup, statvfs metrics and
report persistence are outside the primitive timer. Source warming before
every trial deliberately makes this a warm-cache experiment.

One inter-trial detail remains different: original deletion APIs issue
directory fsyncs during untimed cleanup, whereas the explicit-copy cleanup
unlinks without a separate directory fsync. Thus cleanup/writeback state is
not guaranteed identical between calls. The frozen result remains intact;
a fresh correction should add matching baseline cleanup fsyncs outside the
timers before stronger equivalence claims. This observed source-review limit
is recorded rather than repaired retrospectively in the measured fixture.

## Independent correctness and acceptance

A separate Python implementation generated every original byte using integer
division for the high-bit terms, independently applied the prefix/midpoint/
suffix patches, restored them and computed whole-file SHA-256 with `hashlib`.
It imported or executed no Rust writer/helper. The resulting eight digests
matched the previously independent oracle and the frozen fixture constants.

Acceptance verified the exact 240-sample chronology and twelve-group matrix,
all 2,720 whole-file witnesses, destination/source/reference sizes and hashes,
common XFS device, distinct protected and newly created inodes, both directions
of write isolation, full checkpoint/branch isolation, restored sources, exact
persistence paths, free-space headroom, original namespace counters and final
cleanup. Repeated witness lengths total 102,676,561,920 bytes; this is repeated
validation work, not unique data capacity or performance throughput.

The actual host record separately passed native resource, original unskipped
tests, owned-XFS setup/unmount, supervised process completion, guest/raw byte
hash and immutable input checks. The public verifier CLI passed on those exact
actual raw, host, build and original-source records. Its 26 focused controls
accepted the intact record and rejected copied-evidence mutations, without
changing the preserved actual input. Tests do not launch a kernel or pretend
to measure XFS performance. Unsupplied optional host/build provenance gates
are explicitly `unverified` in the portable verifier, not assumed successful.

Three earlier failed attempts remain in the evidence: an invalid driver
command-timeout configuration rejected before VM creation; a bounded-shell-line
rejection after genuine original tests but before benchmark launch; and a
Docker preparation failure before guest execution. None generated primitive
trials, none is counted as a successful timing result, and no Rust fixture
bytes were changed to remove a failed timed sample.

## Transfer and next experiments

This component establishes that original snapshot/fork operations can take
roughly millisecond-scale time on fully populated data in this native cohort.
It does not checkpoint process RAM, execute the complete original CubeVMM,
generate model tokens, run an optimizer, measure a complete graded episode,
or establish a GPU/training speedup. Multiworker concurrency, concurrent
writers, cold-cache storage and crash recovery remain separate experiments.

The author's
[original benchmark implementation](https://github.com/TencentCloud/CubeSandbox/blob/d0081641c59822e4e5653b7462e914410b81910a/cubecow/benches/reflink_ops.rs)
was reviewed: it already provides serial fanout, snapshot chains, concurrent
fanout and dirty-I/O interleave, and distinguishes latency-implied operations
per second from actual concurrent wall throughput. Those original benchmark
scenarios were not executed by this result. The next cohort should first match
untimed cleanup persistence, then run genuine dirty/concurrent workloads and
integrate branches with complete graded repository episodes; each additional
claim needs its own actual runtime and complete wall-throughput evidence.
