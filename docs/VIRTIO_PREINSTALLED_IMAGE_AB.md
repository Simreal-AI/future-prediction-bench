# Preinstalled read RPC in a new pinned guest image

**Current status (QEMU 11.1.1/HVF on this host): hardened action-port
retirement preserves exact graded parity, but no complete-episode speed gain
is established.** The first three-pair results below are historical
pre-barrier v0.11.0 measurements. A [negative control](measurements/unopened_action_port_snapshot_2026-09-25.json)
found that closing the Python socket left QEMU's port at `guest=off, host=on`.
The current adapter uses a bounded trusted guest open/close with zero
action-port read/write, then rescans processes and FDs and requires QEMU
`guest=off, host=off` plus a disconnected chardev. The runtime unlinks the
retired host listener and pauses QEMU to recheck the device immediately before
and after `savevm`. Three final-source alternating complete graded pairs
passed exact observations, 14 cases, source, and reward parity. The
[final-source report](measurements/virtio_hardened_checkpoint_ab_2026-09-25.json)
records **7.786 s virtio** versus **7.625 s serial** median complete episodes.

The first graded virtio-read experiment spent time uploading the guest
read-only RPC on every VM boot. This follow-up built a **new** ARM64 ext4
rootfs containing the exact agent source, then ran serial and virtio arms
from separate clones of that same new image. The existing pinned image and
its measurements were left intact.

## Build and runtime contract

`examples/realworld_boltons26/prepare_preinstalled_virtio_assets.py`
checks the v2 Boltons seed, 14-case verifier, Alpine kernel/initramfs and
module hashes, source rootfs hash, and the locally cached `linux/arm64`
Python image. It builds the ext4 rootfs offline using Docker's `mke2fs`
stage and places `future_prediction_bench/guest_action_rpc.py` at
`/fpb_guest_action_rpc.py` before converting the image to qcow2. The new
rootfs SHA-256 is
`2060b904c9764496cedf1e4508a37fd3b92ee1338018034b94e4df14f1ccacd6`.
The local new-image manifest records the helper SHA-256
`60e68593593051bfcdfa6e8e8898a3a4ff9134ce169626cd09f01c46b6761167`.

With `preinstalled_read_agent=True`, `MicroVMCodingAdapter` checks that
exact source hash on the host and checks the installed guest file hash on
every reset. A missing or different guest file fails setup before policy
actions. The adapter skips serial upload only after that check; the root-only
port mount, unprivileged candidate visible checks, read semantics, STOP,
guest process/FD quiescence, one-way socket retirement, and QEMU-side
disconnect attestation are otherwise the
same as the [graded read experiment](VIRTIO_GRADED_AB.md). The serial arm
never starts the agent even though its image contains the file. Neither arm
receives a live-service VM checkpoint.

The historical run built the new image and measured paired episodes with
these pinned inputs. Current source applies hardened two-stage retirement, so its
timings differ from the historical table:

```bash
python3 -m examples.realworld_boltons26.prepare_preinstalled_virtio_assets \
  --task-dir runs/boltons-v2-task-static-20260925 \
  --source-assets-dir runs/microvm-assets-v2task-static-20260925 \
  --output runs/microvm-assets-v2-preinstalled-virtio-20260925

python3 -m examples.realworld_boltons26.benchmark_virtio_graded \
  --task-dir runs/boltons-v2-task-static-20260925 \
  --assets-dir runs/microvm-assets-v2-preinstalled-virtio-20260925 \
  --output runs/virtio-graded-preinstalled-readheavy-threepair-20260925 \
  --pairs 3 --read-heavy --preinstalled-agent
```

## Complete-episode result

The [historical aggregate report](measurements/virtio_graded_preinstalled_readheavy_ab_2026-09-25.json)
contains three pre-barrier alternating pairs. Each episode used five distinct repository
reads followed by the exact same repair, visible check, and submit, consuming
the frozen eight-action budget. All six episodes had identical policy
observations, 14 hidden-case results, final source digest, and reward `1.0`.
Each saved one post-agent full-VM state and restored it 14 times. All three
virtio runs confirmed a stopped service, closed socket, unusable old client,
and rejected fork.

| Historical pre-barrier host-clock measure, median of three episodes per arm | Serial | Preinstalled virtio read |
| --- | ---: | ---: |
| Five policy reads, including symlink scans | 160.374 ms | 17.485 ms |
| Complete graded episode | 7.344 s | 7.328 s |

The five reads were **9.2× faster** in aggregate. The complete-episode
median differed by only **16 ms**, and the paired directions were mixed:
virtio was faster in pairs 1 and 3 and slower in pair 2. This is not reliable
evidence of complete-episode or RL rollout throughput improvement.

For the virtio arm, median guest-agent preparation (root-only port mount and
installed-file digest check) was **13.093 ms**; serial upload was **0 ms**;
start/PING was **20.074 ms**; and STOP, process/FD checks, unmount and socket
retirement were **93.901 ms**. These costs are included in the complete
episode. Full reset medians were **5.455 s** serial versus **5.530 s** virtio;
submit medians were **128.021 ms** versus **189.872 ms**. The process-baseline
scan is included in reset but not in the narrow preparation timer. The
one-time new-image build took **12.034 s** (9.274 s staging,
2.630 s ext4 creation, 0.096 s qcow2 conversion). It was excluded from both
arms' episode timing and is common to them; spread over the six measured
episodes it is **2.006 s per episode**. That six-episode amortization is a
cost accounting example, not a steady-state production rate.

## Earlier-source two-stage complete-episode result

The earlier [source-pinned report](measurements/virtio_two_stage_retirement_ab_2026-09-25.json)
uses three alternating pairs and the same preinstalled image with the
UID/GID 65534 candidate binding. Every episode passed the same 14 hidden
cases, saved one submitted VM snapshot, and restored it 14 times. The median
five-read time was **164.229 ms serial versus 17.427 ms virtio**. The median
complete episode was **7.273 s serial versus 7.558 s virtio**, with only one
of three pairs favoring virtio. The trusted reopen helper cost a median
**20.763 ms**; the full STOP/quiesce/reopen/attest sequence cost **161.285 ms**.
The helper has no fixed 200 ms hold: it opens, yields once, closes, and the
QEMU state check decides whether to continue. These results show restored
correctness and faster reads, not a complete-episode or RL-training speedup.

The current QEMU/HVF fixture has no model inference or optimizer in the
timing path. The result does not measure GPU training, disaggregated rollout
throughput, or the cost of building larger task images.

## Final-source hardened snapshot regression

The [new sanitized report](measurements/virtio_hardened_checkpoint_ab_2026-09-25.json)
binds the frozen runtime and template source hashes as well as the adapter,
guest agent, benchmark, task, verifier, and rootfs. It uses the same pinned
preinstalled image in both arms and the same five-read, eight-action workload.
All six episodes completed with exact opening and action observations, all 14
host-only case results, final source digest, and reward `1.0` equal. Each saved
one submitted VM snapshot and restored it 14 times; none used a virtio read
fallback or ended pending. The hardened runtime unlinks the retired host
listener, pauses the VM around `savevm`, checks the exact disconnected QEMU
port before and after saving, and cleans up an uncertain tag before a reward
could be published.

| Final-source host-clock measure, median of three episodes per arm | Serial | Preinstalled virtio read |
| --- | ---: | ---: |
| Five policy reads, including symlink scans | 184.497 ms | 20.624 ms |
| Complete graded episode | 7.625 s | 7.786 s |

The five reads were **8.946× faster**, while the complete virtio episode was
**0.161 s slower** by median. Paired virtio-minus-serial differences were
`+0.027`, `+0.221`, and `+0.249 s`, so all three complete virtio episodes were
slower. The trusted reopen helper cost a median **24.582 ms** and total
retirement cost **180.682 ms**. The earlier two-stage measurements above were
made with a prior runtime source hash and remain historical. This regression
establishes correctness under the hardened gate on one host and one scripted
fixture; it does not show a whole-episode or RL-training speedup.
