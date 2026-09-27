# Parallel full-VM template provisioning

`MicroVMTemplate.spawn(..., parallel_clone_verification=True)` is an opt-in
host-side scheduling change for a batch of two to four independent QEMU/HVF
children. A sealed qcow2 template already contains a full QEMU CPU, memory,
device, and writable-disk snapshot. Each child still gets a distinct qcow2
inode and a full SHA-256 comparison against the sealed template manifest.
The change overlaps the clone and hash work across children, then waits for
**all** checks before constructing or starting any QEMU child. The default
serial schedule remains available as the control arm.

This is a narrow, independently implemented response to the branch fan-out
cost studied in [DeltaBox](https://github.com/dongyunpeng-sjtu/deltabox).
It does not use DeltaBox's unreleased kernel patch or controller, does not
implement its millisecond live process checkpoint/restore, and does not
replace QEMU `savevm`/`loadvm`. It addresses template **provisioning** only;
the time to export the template and the cost of later hidden-case restores
remain separate.

The operational contract is fail-closed for this trusted-host fixture. A
child is not booted until every clone passes its own full digest, so a
corrupted sibling prevents the whole batch from starting. If any clone,
digest, startup, or restore fails, spawned processes are closed and all files
created by this call are removed. Each clone is first written into a private
staging directory. A no-replace hard link publishes the finished clone to the
requested path; cleanup checks the inode before unlinking. Thus a caller's
path that appears between the initial path check and publication is not
overwritten or deleted. Parent and sibling writable disks remain separate;
guest RAM and ext4 isolation are checked in the real-QEMU A/B. There is still
a path race against an adversarial host process capable of replacing an inode
between the identity check and unlink, or mutating bytes after digest
verification. This is a trusted-host mechanism, not a multi-tenant QEMU
security boundary.

Reproduce the solved Boltons fixture experiment after preparing the pinned
task and ARM64 guest assets described in [the microVM guide](MICROVM_ENV.md):

```bash
python3 -m examples.realworld_boltons26.benchmark_parallel_template_provision \
  --task-dir runs/boltons-v2-task-static-20260925 \
  --assets-dir runs/microvm-assets-v2task-static-20260925 \
  --output runs/parallel-template-provision-ab \
  --pairs 2 --branches 4 --grading-workers 4
```

The arms alternate serial/parallel order. Both use the same sealed template,
four independent QEMU children, four grading workers, the same scripted
repair/baseline branch choices, and the same 14 host-private Boltons cases
per branch. The harness checks exact final source digests, all case-result
hashes, rewards, sibling RAM/ext4 isolation, and template/verifier stability.
It reports child provisioning separately from the full graded batch. This
specific benchmark calls the low-level `MicroVMTemplate.spawn` API; it does
not measure RealWorldEnv, policy inference, GPU training, or optimizer work.
The prepared RealWorldEnv adapter continues to use the serial default; the
opt-in is deliberately not exposed there on this evidence.

## Real-QEMU result on 2026-09-25

The [source-bound report](measurements/parallel_template_provision_ab_2026-09-25.json)
(SHA-256 `d814e431d6099aad9376b73d70f70bcd73c1734c51260896867685d31656cc05`)
contains two AB/BA pairs on one ARM64 macOS host with 8 logical CPUs and
8 GiB RAM. Four 512 MiB guests ran per arm. The template was 167,837,696
bytes; preparing it once took 6.172 s and is excluded equally from both
steady-state arms. Host load average rose from 6.52 before the experiment
to 8.43 after the last arm, so CPU and memory-bandwidth contention matter.

| Pair | Serial setup | Parallel setup | Serial complete graded batch | Parallel complete graded batch |
| --- | ---: | ---: | ---: | ---: |
| 0, serial first | 1.501 s | 0.944 s | 12.882 s | 12.239 s |
| 1, parallel first | 0.994 s | 0.889 s | 10.151 s | 10.502 s |
| Median | **1.247 s** | **0.916 s** | **11.517 s** | **11.370 s** |

Parallel provisioning shortened the measured setup median by 0.331 s, a
**1.36× setup-time ratio** in these two pairs. The clone-plus-child-SHA
makespan was 0.493/0.319 s in the serial arms versus 0.213/0.227 s in the
parallel arms. Parallel child SHA work itself consumed more summed wall time
(0.818/0.830 s versus 0.486/0.302 s), consistent with host resource
contention. The complete graded-batch median ratio was only **1.013×**;
pair 1 was slower with parallel provisioning. This small sample does **not**
establish a complete-rollout throughput gain, let alone a 5× improvement.

All 16 branches completed 14 hidden cases each: 224 case results total. Each
arm produced rewards `[1, 0, 1, 0]`; pairwise final source digests, every
case-result hash, and rewards matched exactly. All sibling RAM and writable
ext4 separation probes passed. The benchmark left no QEMU child process or
branch disk behind. The full report binds the task, assets, verifier,
template, implementation, and benchmark source SHA-256 values, and contains
only path-free result hashes rather than hidden expected output text.

The focused test suite is `tests/test_parallel_template_provision.py`.
It uses a fake monitor to prove concurrent four-child provisioning and the
all-child preboot digest barrier, then injects a corrupted clone and a clone
exception after creating a partial private file to check that neither can
leave booted children or leaked target disks. It also races an unrelated file
into the requested pathname and verifies that the no-replace publication
preserves that file. Real-QEMU observations and any speed claim belong to the linked
measurement report, not to the fake-monitor tests.
