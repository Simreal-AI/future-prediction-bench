# Opt-in virtio reads in a graded RealWorldEnv episode

**Current status (QEMU 11.1.1/HVF on this host): complete graded parity holds
with the hardened action-port snapshot gate.** Closing the Python host socket
alone left `fpb.control` at `guest=off, host=on`. The adapter briefly opens
and closes the port from trusted guest code, without reading or writing port
bytes, then requires QEMU to report `guest=off, host=off` and a disconnected
chardev. The runtime also unlinks the retired host listener and pauses QEMU
while it rechecks disconnection immediately before and after `savevm`. The
[final-source three-pair report](measurements/virtio_hardened_checkpoint_ab_2026-09-25.json)
records exact observations, 14-case results, final source, and reward parity.
Five reads were faster, but the complete virtio episode median was **0.161 s
slower**; all three paired complete episodes were slower. The earlier
[two-stage report](measurements/virtio_two_stage_retirement_ab_2026-09-25.json)
is historical earlier-source evidence. The [negative control](measurements/unopened_action_port_snapshot_2026-09-25.json)
records why QEMU-side attestation and trusted guest reconciliation are needed.

The `MicroVMCodingAdapter` can now route its `read_file` policy action over the
bounded read-only virtio-serial RPC, while the same `RealWorldEnv` episode still
uses serial for edits, visible checks, submission, and host-private grading.
The default remains `read_transport="serial_shell"`. The opt-in requires a
`MicroVMRuntime(enable_action_port=True)` and adds the guest-agent source hash
and read transport to the frozen task artifact binding.

## State transition and scope

The adapter retains the whole-workspace symlink scan and `_relative_file`
validation before each read. The guest agent opens the requested path with
`O_NOFOLLOW`, hashes the complete regular file, and returns the same 16 KiB
preview, digest, path, and truncation fields. Missing and non-regular files
produce the existing policy-level missing-file error. A regular file above
the agent's 50 MB cap falls back to the serial read implementation. RPC
malformation, timeout, or an untrusted file state interrupts the episode
without a policy loss or reward.

For this opt-in mode, candidate-visible Python checks run as UID/GID 65534.
The virtio device is mode `0600` and bound into the trusted guest chroot for
the root-owned agent. At submit, the host sends `STOP`, waits for the agent
PID to exit, unmounts the chroot device, checks the live guest process census
against the pre-agent baseline, scans guest file descriptors for any remaining
port holder, and closes the host socket. The runtime now also requires QEMU
itself to report the exact port as `guest=off, host=off` and the chardev as
disconnected before enabling `save_snapshot` or `load_snapshot`; it rechecks
at each operation. The retired host `action.sock` listener is unlinked and
must remain absent. Save pauses the VM, rechecks exact disconnection before
and after `savevm`, and deletes an uncertain new tag or closes the runtime if
safe cleanup cannot be proved. When QEMU initially retains `host=on`, the adapter gives
it one bounded read-ready interval through a root-owned guest open/close,
then rescans processes and port FDs before final attestation. Any failure
interrupts without a snapshot or reward. The old client cannot be used or reconnected, and
`fork_snapshot` remains disabled for this opt-in runtime. The narrow
unopened-port HMP experiment restored RAM and the ext4 workspace but did not
establish a live-service checkpoint or connection-epoch restoration.

This is a trusted research guest, not a hardened multi-tenant isolation
boundary. The process and FD checks are a submit-time quiescence gate, not a
general defense against an adversarial guest kernel.

## Real guest evidence

The historical v0.11.0 run used the pinned Boltons repair A/B with separate
qcow2 disks and alternating arm order. Its numeric results require the
archived v0.11.0 source; current source applies two-stage retirement:

```bash
python3 -m examples.realworld_boltons26.benchmark_virtio_graded \
  --task-dir runs/boltons-v2-task-static-20260925 \
  --assets-dir runs/microvm-assets-v2task-static-20260925 \
  --output runs/virtio-graded-final-threepair-20260925 \
  --pairs 3
```

The [public aggregate report](measurements/virtio_graded_realworld_ab_2026-09-25.json)
records the pre-barrier three alternating pairs, six graded episodes, and the exact source
hashes. Every policy observation, each of the 14 hidden-case results, the
final source digest, and reward `1.0` matched across arms. Each episode saved
one full-VM snapshot and loaded it 14 times. All three opt-in runs confirmed
STOP acknowledgement, closed host socket, unusable old client, unavailable
runtime socket, and rejected fork.

| Historical pre-barrier host-clock measure, median of three episodes per arm | Serial | Opt-in virtio read |
| --- | ---: | ---: |
| One policy `read_file`, including whole-workspace symlink scan | 70.222 ms | 3.662 ms |
| Whole graded episode, including VM boot, setup, repair, and 14 cases | 7.355 s | 7.679 s |

The read action was **19.2× faster** in this one-read workload. Two of the
three complete opt-in episodes were slower, and the median was slower. The opt-in agent cost a median
**133.569 ms** to upload/mount, **21.890 ms** to start and ping, and
**92.763 ms** to stop/quiesce/unmount/retire; those costs dominate one
accelerated read. These are end-to-end implementation timings, not pure
virtio wire latency. They do not show an RL rollout or trainer throughput
improvement.

The adversarial [real-guest probe](../examples/realworld_boltons26/probe_virtio_graded_security.py)
ran candidate-visible Python as UID 65534. It could not open the port, then
spawned a background process. Submit was interrupted with no QEMU snapshot
and no reward. The [public report](measurements/virtio_graded_realworld_ab_2026-09-25.json)
contains only aggregate parity and timing, not private verifier output.

The [preinstalled-image follow-up](VIRTIO_PREINSTALLED_IMAGE_AB.md) built a
new rootfs hash with the agent included and measured both arms on that same
new image. The numbers above use the original per-episode upload path.

## Five-read complete-episode check

The same frozen task allows eight actions. A second scripted workload reads
five distinct repository files before the same exact repair, visible check,
and submit, consuming that full budget. It uses `--read-heavy` with the same
command and a new output directory. Three alternating pairs again had exact
action observations, 14-case results, source digest, and reward parity. The
[public aggregate report](measurements/virtio_graded_readheavy_ab_2026-09-25.json)
records the five paths, each pair, and source hashes.

| Historical pre-barrier host-clock measure, median of three episodes per arm | Serial | Opt-in virtio read |
| --- | ---: | ---: |
| Total of five policy `read_file` actions | 172.599 ms | 15.847 ms |
| Whole graded episode | 7.279 s | 7.493 s |

The five reads were **10.9× faster** in aggregate, but all three opt-in
complete episodes were still slower. The read savings of about 157 ms did
not cover the service lifecycle and other setup costs in this workload.

## Earlier-source two-stage graded run

The earlier [source-pinned three-pair report](measurements/virtio_two_stage_retirement_ab_2026-09-25.json)
uses the preinstalled image, five reads, the UID 65534 candidate
binding, and a zero-fixed-delay trusted reopen. In all six complete episodes,
one submitted snapshot and 14 restores produced reward `1.0`; every policy
observation, hidden case result, and final source digest matched. The median
five-read time was **164.229 ms serial versus 17.427 ms virtio**, while the
median complete episode was **7.273 s serial versus 7.558 s virtio**. Pair
directions were mixed, and this sample gives no complete-episode speedup
claim. The guest reopen helper itself cost a median **20.763 ms**, while the
entire STOP/quiesce/reopen/attest retirement cost **161.285 ms**. Its runtime
source predates the listener-unlink and paused-save hardening below.

## Final-source hardened snapshot regression

The [new sanitized report](measurements/virtio_hardened_checkpoint_ab_2026-09-25.json)
binds the frozen runtime, template, adapter, guest agent, benchmark, task,
verifier, and rootfs hashes. Three alternating pairs used the same preinstalled
image and eight scripted actions: five distinct reads, one exact repair, one
visible check, and submit. All six episodes graded `14/14`, returned reward
`1.0`, saved once, and restored 14 times. Opening and action observations,
every hidden-case result, and the final source digest matched. There were no
virtio read fallbacks or pending episodes. The retired host socket and client
remained unusable, and action-port branch forking stayed disabled.

| Final-source host-clock measure, median of three episodes per arm | Serial | Opt-in virtio read |
| --- | ---: | ---: |
| Total of five policy `read_file` actions | 184.497 ms | 20.624 ms |
| Complete graded episode | 7.625 s | 7.786 s |

The five reads were **8.946× faster** in aggregate. Virtio's complete-episode
median was **0.161 s slower**, with paired virtio-minus-serial differences of
`+0.027`, `+0.221`, and `+0.249 s`. Its trusted reopen helper cost a median
**24.582 ms** and total STOP/quiesce/reopen/attest retirement cost **180.682 ms**.
This regression confirms graded correctness after hardening, not a complete
episode or RL-training speedup. The measured time contains no model inference
or optimizer work.
