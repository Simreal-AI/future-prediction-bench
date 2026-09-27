# Full-state ARM64 microVM coding environment

`MicroVMCodingAdapter` runs a real Linux guest under QEMU/HVF on Apple Silicon. The VM has no virtual NIC or host-directory mount. The policy receives the same bounded `list_files`, `read_file`, `write_file`, fixed `run_visible_checks`, and `submit` actions as the Docker coding environment. After submission, QEMU `savevm` freezes guest CPU, RAM, devices, and a dedicated writable qcow2 disk; `loadvm` restores that state before each host-checked hidden case. Hidden expected outputs and the verifier specification stay on the host. This is a real full-VM checkpoint, unlike the Docker adapter's filesystem branch snapshots. The measurements below used a MacBook Pro with an Apple M3 chip and 8 GB of memory, QEMU 11.1.1, and Python 3.12.1 on the host.

This experimental adapter is for locally trusted research workloads. It has no network interface, but QEMU/HVF and the guest image have not been audited as a hardened multi-tenant security boundary. Trusted setup code runs as guest root; candidate-visible and hidden Python run as UID/GID 65534 under the current task binding. A policy cannot choose an arbitrary shell command through the environment API; operator-fixed Python checks can execute policy-edited code inside the VM.

## Reproduce the real repository sample

Requirements are an Apple Silicon macOS host with QEMU/HVF, Docker for *building* the guest disk, and one locally cached `linux/arm64` image containing Python 3.12 and `mke2fs`. The asset builder pins Alpine 3.24 ARM64 `vmlinuz-virt`, `initramfs-virt`, and `modloop-virt` by SHA-256 and creates a 256 MiB ext4 root filesystem containing Python and a [pinned Boltons 26.0.0 task](../examples/realworld_boltons26/README.md). The image-building Docker container has no network; the final VM does not depend on Docker. Generated images, source archives, reports, and snapshots live under ignored `runs/` and are not included in the release source archive.

Create a fresh Boltons task, build a guest image, then run a scripted repair through the full `RealWorldEnv` action and reward protocol:

```bash
python3 examples/realworld_boltons26/make_task.py --output runs/boltons-vm-task
python3 -m examples.realworld_boltons26.prepare_microvm \
  --task-dir runs/boltons-vm-task --output runs/boltons-vm-assets \
  --image sha256:<cached-arm64-python312-and-mke2fs-image-id>
python3 -m future_prediction_bench realworld-code-microvm \
  --task runs/boltons-vm-task/task.json \
  --verifier runs/boltons-vm-task/verifier \
  --assets runs/boltons-vm-assets \
  --actions runs/boltons-vm-task/actions.solution.jsonl \
  --visible-check '["python3","-B","-c","import boltons.strutils"]' \
  --output runs/boltons-vm-episode
```

The task and output paths must be new or empty. The generated fixture has a 30-minute action window; regenerate it if that deadline has passed. To avoid fetching Alpine assets again, pass `--alpine-source-dir` with the three previously downloaded, digest-matching files. The asset builder checks the local image architecture and stores its immutable image SHA-256 in a manifest. It also checks the pinned Alpine downloads before building. The official Boltons source archive is separately verified by its pinned digest in `make_task.py`. This preparatory trust chain should be tightened with signed upstream artifacts for a production service.

Run the controlled cold-start versus warm-restore experiment:

```bash
python3 -m examples.realworld_boltons26.microvm_benchmark \
  --task-dir runs/boltons-vm-task --assets-dir runs/boltons-vm-assets \
  --output runs/boltons-vm-benchmark --repetitions 3
```

The benchmark first proves that the unmodified source gets reward 0, the scripted repair gets reward 1 across all 14 host-side cases, and restoring the pristine VM reintroduces the original bug. It also starts a guest process and writes RAM state before a checkpoint, kills the process and changes that state, then verifies that the process and original RAM state return after restore. It alternates cold and warm condition order within each pair. Each measured episode applies the public repair, snapshots the submitted guest, and restores that snapshot before *every* hidden case. The cold condition copies a pristine disk and boots a new VM. The warm condition reuses one already booted VM and loads a full-state pristine snapshot. Reports include per-episode wall time, QEMU snapshot timing, a steady-state median ratio, and an amortized ratio including the one-time warm setup.

On one Apple Silicon host, the final three paired repetitions yielded 8.606/8.578/8.147 seconds cold and 3.324/3.956/2.116 seconds warm. Medians were **8.578 versus 3.324 seconds**, a **2.581× environment-episode throughput gain** after warm setup. Warm setup took 5.890 seconds; including it, the three-episode total was 25.331 seconds cold versus 15.286 seconds warm, a **1.657× gain**. The [public measurement report](measurements/microvm_boltons_v0.6.0.json) preserves the complete timing and case-hash summary without guest binaries or local paths. These numbers cover sandbox setup, a scripted public fix, and hidden grading. They do not measure model inference, optimizer updates, policy improvement, or independent repository generalization.

## Independent sibling VM forks

`MicroVMRuntime.fork_snapshot()` pauses one warm parent, clones its qcow2 snapshot into separate writable child disks, starts each child paused, restores the saved CPU/RAM/device/disk state, and resumes parent and children. To compare these independent siblings with equally independent cold VMs on the pinned asset manifest v2, run:

```bash
python3 -m examples.realworld_boltons26.microvm_fork_benchmark \
  --task-dir runs/boltons-vm-task --assets-dir runs/boltons-vm-assets \
  --output runs/boltons-vm-fork-benchmark \
  --repetitions 3 --branch-count 2 --grading-workers 2
```

Both conditions use the same APFS `clonefile`-or-copy helper for their child disks; the cold condition boots its two VMs **in parallel**. Both keep two children live and grade their 14 host-private cases with two workers. The repaired and untouched branches receive `[1, 0]`; unique guest `/tmp` and ext4 markers remain absent from the sibling and parent, and the warm children inherit the parent's pre-snapshot RAM marker. The parent remains idle during cold measurements, so both conditions carry its background resource cost. Per-condition wall time includes cloning, VM startup or restore, isolation probes, grading, and cleanup; post-run disk hashing is excluded.

Three alternating-order pairs on one Apple Silicon host measured **15.885/15.260/13.376 s** for parallel cold VMs and **12.181/9.748/7.868 s** for forked siblings. Median batch times were **15.260 versus 9.748 s**, a **1.565×** ratio in graded environment-branch throughput. One-time parent clone, boot, and snapshot setup took **6.397 s**; including it, total time across the three two-branch batches was **44.521 s cold** versus **36.194 s forked**, a **1.230×** amortized ratio. The [sanitized per-run report](measurements/microvm_fork_boltons_v0.6.0.json) preserves every branch reward, isolation check, timing, and child disk digest. This direct runtime experiment uses the public solved Boltons fixture and does not invoke `RealWorldEnv` or its action deadline. It measures neither model inference nor training, and it does not implement Daytona's live remote sandbox fork service.

### Serial versus parallel sibling startup

The optional `parallel_children=True` path starts and restores up to four independent child QEMU processes concurrently after the parent's qcow2 disk has been cloned and verified. The default remains serial. This matched benchmark uses the same warm parent and pinned assets for both conditions, alternates condition order within each pair, and keeps two siblings live for identical isolation probes and two-worker grading:

```bash
python3 -m examples.realworld_boltons26.microvm_parallel_fork_benchmark \
  --task-dir runs/boltons-real-repo-bound-20260924 \
  --assets-dir runs/microvm-assets-v2-20260925 \
  --output runs/boltons-vm-parallel-fork-20260925 \
  --repetitions 3 --branch-count 2 --grading-workers 2
```

Three paired runs measured child **start plus snapshot restore** medians of **0.679 versus 0.490 s** (serial versus parallel, **1.388×**). Complete branch setup, including disk cloning, fork, and inherited-RAM check, had **1.119 versus 0.885 s** medians (**1.264×**). The full two-branch batch, including isolation checks, 14 host-private cases per branch, and VM cleanup, had **7.694 versus 7.252 s** medians (**1.061×**). One-time parent setup was **5.869 s** and is reported separately; adding it to all three batches yields a **1.064×** aggregate ratio. Every batch returned rewards `[1, 0]`, passed sibling/parent RAM and disk marker checks, and retained the host-private verifier digest. The [sanitized report](measurements/microvm_parallel_fork_boltons_v0.6.0.json) contains all per-run times, branch rewards, and disk digests. These are environment measurements on one solved fixture, not model-training speedups or millisecond checkpoint/restore results.

## Full-VM branches through RealWorldEnv

The [`check_microvm_branch_env.py`](../examples/realworld_boltons26/check_microvm_branch_env.py) integration check exercises the actual trusted-host `RealWorldEnv.create_branch_checkpoint()` and `fork_from_checkpoint()` API. It refreshes the **fixture's** expired action timestamps for the local run, then freezes the parent at an audited action boundary. The parent writes a file *after* the checkpoint. Two sequential child VMs restore the earlier snapshot: neither sees that later parent file, both inherit the earlier RAM and ext4 markers, and each gets its own CPU/RAM and writable disk state. The public repair goes into only one child. Full-file SHA-256 checks confirm that the baseline child and parent retain the original source. Both children submit through the normal environment action protocol and run all 14 host-private verifier cases.

```bash
python3 -m examples.realworld_boltons26.check_microvm_branch_env \
  --task-dir runs/boltons-vm-task --assets-dir runs/boltons-vm-assets \
  --output runs/boltons-vm-realworld-branches
```

On the prepared pinned assets, the repaired and baseline branches returned **[1.0, 0.0]**; parent and sibling `/tmp` and ext4 markers remained independent. The final scripted correctness run took **12.629 s** on the development host, including VM setup, branch actions, and both sets of hidden cases. It also prepared sibling-local advantages of `+1/-1` for the two post-checkpoint suffixes, with the common prefix masked and `trainer_ready=false`. This is one correctness run, not a paired throughput benchmark or model update. The [path-free report](measurements/microvm_realworld_branches_v0.6.0.json) records the task and snapshot hashes, case counts, rewards, isolation checks, and local advantages. The checkpoint disk digest is an audit value for the qcow2 container at creation; QEMU can subsequently change its container bytes without changing the named frozen snapshot, so repeated fan-out validates the snapshot tag and restores it in each child rather than requiring the live parent's whole-file digest to remain equal.

## QEMU snapshot primitive latency

The [primitive benchmark](../examples/realworld_boltons26/benchmark_vm_primitives.py) times individual HMP `savevm` and `loadvm` command roundtrips, separately from VM boot, policy actions, verifier cases, and complete episodes. It also times a monitor-only `info status` roundtrip. Each memory size gets two saves, three warm-up loads, and 20 measured loads; every load is checked to restore both a guest `/tmp` RAM marker and an ext4 workspace marker. QEMU's `info snapshots` supplies the displayed `VM_SIZE`.

```bash
python3 -m examples.realworld_boltons26.benchmark_vm_primitives \
  --task-dir runs/boltons-vm-task --assets-dir runs/boltons-vm-assets \
  --output runs/boltons-vm-primitives --memory-mib 128 256 512 \
  --measured-loads 20 --warmup-loads 3 --save-count 2
```

One consolidated sweep on the same Apple Silicon host gave the following **milliseconds**. The p95 uses nearest rank; save p50 has only two observations and is descriptive, not a stable percentile estimate.

| Guest RAM | QEMU `VM_SIZE` | `savevm` p50 (n=2) | `loadvm` p50 / p95 / min / max (n=20) | Monitor-only p50 (n=20) |
| --- | ---: | ---: | ---: | ---: |
| 128 MiB | 78.6 MiB | 90.09 | 48.13 / 63.92 / 40.04 / 75.44 | 0.104 |
| 256 MiB | 82.6 MiB | 123.54 | 51.11 / 69.05 / 45.78 / 70.82 | 0.104 |
| 512 MiB | 87.7 MiB | 218.59 | 81.95 / 244.16 / 61.82 / 247.88 | 0.115 |

The allocated RAM differs from the saved state size, and a separate 512 MiB run had a 161 ms load median, so host variability is material. The monitor baseline is reported alongside the primitive time and is **not subtracted** from it. The [measurement report](measurements/microvm_primitives_v0.6.0.json) records every sample, two-sample save ranges, full `load_snapshot` wrapper time, snapshot listing, and all 23 successful state resets per size. These QEMU full-VM timings are not DeltaBox-style incremental container checkpoint/restore timings and do not establish RL training acceleration.

Current `save_snapshot` checks QEMU's snapshot table before saving because HMP `savevm` would replace an existing tag, including one inherited in a cloned qcow2. If a save fails after it may have written a tag, the runtime tries to delete that uncommitted tag and closes the VM. If deletion cannot be verified, a durable sidecar marker bars this runtime from reopening or cloning the affected disk; marker-write failure is reported separately. The [real-QEMU transaction probe](SNAPSHOT_TRANSACTION_GUARD.md) checks this failure boundary. These guards add host monitor work around the `savevm` command. The primitive table above predates this guard and times the QEMU `savevm` command itself, so it is not a current-source wrapper-latency measurement. A fresh full-episode A/B is required before claiming any throughput effect from the guard.

A [current-source QEMU/HVF rerun](measurements/microvm_primitives_currentguard_2026-09-25.json) on the pinned asset-manifest-v2 Boltons guest measured the guarded public `save_snapshot` wrapper as well as the individual HMP commands. At **128 MiB**, two wrapper saves took **80.63–113.73 ms** (median **97.18 ms**); 20 wrapper loads had **48.47 ms median / 69.00 ms p95**. At **512 MiB**, two wrapper saves took **210.16–217.97 ms** (median **214.07 ms**); 20 wrapper loads had **67.39 ms median / 222.92 ms p95**, including one **331.63 ms** outlier. Every one of the 23 warm-up and measured loads per size restored both RAM and ext4 state. The save wrapper performed `info snapshots`, `savevm`, and `info snapshots`; the corresponding HMP-only save medians were **94.40 ms** and **213.60 ms**. Monitor-only roundtrip medians were **0.104 ms** and **0.112 ms**. These are primitive latencies from one host session, with two saves per size, not complete graded episodes or a controlled before/after comparison with the older table. The report binds the exact runtime, benchmark, asset manifest, source archive, and seed workspace digests.

## Guest-local fork and overlayfs branch experiment

For a narrower warm-process branch primitive, [`benchmark_guest_cow.py`](../examples/realworld_boltons26/benchmark_guest_cow.py) boots one pinned VM, loads the Boltons module into a trusted guest Python process, then runs [`guest_cow_branch.py`](../future_prediction_bench/guest_cow_branch.py) inside that process. Linux `fork()` gives each child a copy-on-write Python heap. Two overlayfs mounts share the same lower repository but use separate upper/work directories: one child applies the public `glass` repair and returns `glass`, while the other remains at the failing `glas` baseline. Both inherit counter 7, their mutations remain separate, and the lower source hash is unchanged. The subsequent repeated trial creates an overlay, forks a ready child, reaps it, then unmounts and removes that overlay's upper/work files.

```bash
python3 -m examples.realworld_boltons26.benchmark_guest_cow \
  --assets-dir runs/boltons-vm-assets \
  --output runs/boltons-guest-cow-benchmark --repetitions 200
```

The guest uses `perf_counter_ns()` around each operation; serial transport and VM boot are outside these millisecond samples. One 200-repetition run on the Apple Silicon host produced these guest-clock results:

| Guest-local operation | Median | p95 |
| --- | ---: | ---: |
| Create directories and mount overlay | 0.058 ms | 0.121 ms |
| `fork()` through child-ready pipe handshake | 0.170 ms | 0.435 ms |
| Reap the child | 0.163 ms | 0.270 ms |
| Unmount and remove the overlay view | 0.575 ms | 1.216 ms |
| Complete measured branch cycle | **0.988 ms** | **1.881 ms** |

The complete [measurement report](measurements/guest_cow_branch_v0.6.0.json) includes min/max values, asset and program digests, and correctness results. VM boot still took **5.671 s**; the guest experiment call took **0.342 s** for all 200 cycles. These guest-local branches share one VM and one Linux kernel, and their overlay mount paths are accessible in a shared mount namespace. They accept only this trusted scripted workload. They do not implement a secure untrusted-agent sandbox, host-private reward grading per branch, or a full CPU/RAM/device VM checkpoint. Their sub-millisecond operation is therefore **not comparable as a full-state restore** to the 48–82 ms QEMU `loadvm` medians above or to checkpoint/restore claims in other systems. Integrating such branches into RL would require separate containment and a policy action protocol before throughput or training gains can be claimed.

A subsequent [private mount-namespace branch experiment](GUEST_MOUNT_NAMESPACE.md) adds `unshare(CLONE_NEWNS)`, private mount propagation, and one tmpfs/overlay view per trusted child. Its final 200-cycle run measured **0.864 ms median / 0.990 ms p95** for a complete guest-local branch cycle, with distinct mount namespace IDs and an unchanged shared lower source. This uses a different guest image/program revision from the earlier COW run, so the two medians are not a controlled speed comparison. It is still shared-kernel trusted execution, not an independent full VM or an agent-facing grader.

## Reusable full-VM snapshot templates

[`MicroVMTemplate`](../future_prediction_bench/microvm_template.py) exports a running parent's QEMU `savevm` checkpoint while the parent is paused, copies its qcow2 to a read-only template, records a SHA-256 digest and the exact snapshot tag in a host-only manifest, then resumes the parent. Reopening the manifest requires the previously returned template ID. Every spawn checks the manifest, template file type, kernel, initramfs, and read-only module disk digests; it then gives each child an independent writable qcow2 clone and hashes **every cloned child disk** against the frozen template digest before any child boots. Up to four children start in parallel and load the same CPU/RAM/device/disk snapshot. A source-template change after the final clone does not alter those verified child bytes; a later spawn rejects a mismatched clone. The parent need not remain alive between batches. A failed export resumes the parent and removes partial template files; a failed spawn closes children and removes only its new disks. The read-only mode guards against accidental writes; the digest detects changes before a cloned child boots, but this is not a signed or tamper-proof artifact store.

The [paired benchmark](../examples/realworld_boltons26/microvm_template_benchmark.py) closes the parent **before** spawning any template children. It writes RAM and ext4 markers after export; children inherit the earlier RAM marker and see neither later marker. Each pair compares two parallel cold VMs with two parallel template children, followed by two-worker grading of the same 14 host-private Boltons cases. Each batch returns `[1, 0]` for repaired and unmodified branches; sibling RAM and ext4 markers remain independent.

```bash
python3 -m examples.realworld_boltons26.microvm_template_benchmark \
  --task-dir runs/boltons-vm-task --assets-dir runs/boltons-vm-assets \
  --output runs/boltons-vm-template-benchmark \
  --repetitions 2 --branch-count 2 --grading-workers 2
```

On the pinned Apple Silicon host, two alternating-order pairs took **10.938/10.928 s** for parallel cold batches and **6.288/6.259 s** for template batches. Median batch time was **10.933 versus 6.273 s**, a **1.743×** graded environment-branch throughput ratio. The one-time parent clone, boot, template export, post-export correctness probes, and closure took **6.126 s**; template export itself took **0.504 s**. Charging all setup to the two template batches yields **1.171×** total throughput relative to cold startup. Per-batch setup was **5.627/5.565 s** cold versus **0.775/0.860 s** from the template; grading remained about five seconds in both conditions. The [path-free measurement report](measurements/microvm_template_boltons_v0.6.0.json) records each batch, reward, isolation check, and child disk digest. Two pairs on one solved repository fixture are a small infrastructure sample, not a model-training result. Template spawn is QEMU disk clone plus full-VM restore, not a live hypervisor fork or a millisecond full-state branch.

A later [three-pair four-sibling run](measurements/microvm_template_four_siblings_v0.6.0.json) used the same pinned assets and four grading workers. Every condition returned `[1, 0, 1, 0]` with independent RAM and ext4 markers. Parallel cold batches took **15.063 s** median and reusable-template batches **11.532 s** median, a **1.306×** steady-state ratio; after the **6.039 s** one-time template setup, the observed three-pair ratio was **1.125×**. The smaller benefit at four simultaneous VMs is a reminder to measure fan-out on the actual 8-GB host rather than extrapolating from two children.

The [prepared RealWorldEnv integration](PREPARED_MICROVM_ENV.md) binds a clean template to this exact Boltons task, verifies no policy or background state was inherited, and runs complete `RealWorldEnv` episodes from independent children. With the default full-VM hidden-case verifier, three alternating repair/baseline pairs yielded **8.117 versus 3.647 s** median episode time (cold versus prepared, **2.226×**); charging the **6.100 s** template preparation to all six episodes yielded **1.800×**. Every cold/prepared pair had identical opening observations, every policy action observation, and exact 14-case evidence. A separately bound stateless contract plus helper-only preinstallation measured **7.029 versus 2.122 s** median cold/prepared episode time (**3.313×**) and **2.197×** including its one-time setup across six episodes. Its case-isolation contract is narrower than full VM restore per case, and neither result measures model training.

## Task-declared stateless hidden cases

The experimental [stateless verifier](STATELESS_VERIFIER.md) targets the repeated *grading* restores shown above. Under a host-authored, checksum-bound contract for an explicitly stateless Python task, it loads the submitted VM once and executes each hidden case in its own guest mount/PID namespace with a tmpfs/overlay workspace. One trusted guest call can batch all case executions while exact expected stdout and return codes remain on the host. The `realworld-code-microvm` CLI selects this path only when passed `--stateless-verifier-contract` with the exact task's contract; otherwise it restores the full VM before every case. A submitted snapshot is inspected for new persistent processes before grading. The final pinned-fixture test measured **5.433×** for the 14-case stage and **1.152×** over complete graded `RealWorldEnv` episodes, including setup and scripted actions. The linked guide gives exact parity, contamination, timing, and security limits; the optional path is neither a full-VM checkpoint replacement nor a model-training result.

## What is implemented

### Opt-in read-only virtio action transport

An experimental `enable_action_port=True` option adds a dedicated [QEMU socket chardev and virtio-serial port](https://www.qemu.org/docs/master/system/qemu-manpage.html) to the pinned ARM64 coding guest. Linux exposes this named port under `/dev/virtio-ports/`, the same [documented device convention](https://www.qemu.org/docs/master/interop/qemu-ga.html) used by QEMU Guest Agent; this experiment runs its own small read-only process, not QEMU Guest Agent. A persistent guest Python process answers framed `PING` and `READ_FILE` requests. Frames have a size bound, session ID, and sequence number; the host enforces an absolute call deadline and one caller at a time. The read operation returns the same bounded file preview and SHA-256 observation as the serial path on the pinned symlink-free fixture. It cannot write files, submit an episode, execute a policy-chosen shell command, receive hidden expected outputs, or calculate reward. The default runtime still uses the serial console. The opt-in runtime rejects `save_snapshot` and `load_snapshot` while the port is active, and rejects `fork_snapshot` and template export throughout its lifetime. A submitted-state save/load requires STOP, guest process/file-descriptor quiescence, one-way host socket retirement and listener unlinking, and **QEMU-side `guest=off,host=off` with a disconnected chardev**. This host initially kept `host=on` after host socket close; the current two-stage path uses a trusted bounded guest open/close to let QEMU observe the closed host connection. Save pauses the VM and rechecks the disconnected topology before and after `savevm`; load rechecks before and after `loadvm`. An uncertain save tag is deleted or the runtime is closed. Any unexpected state interrupts before snapshot or reward. The [final-source graded A/B](measurements/virtio_hardened_checkpoint_ab_2026-09-25.json) validates parity but finds no complete-episode speedup; no live connection restore or connection-epoch protocol is claimed.

Run the paired probe against a locally prepared pinned VM asset directory:

```bash
python3 -m examples.realworld_boltons26.benchmark_virtio_action \
  --assets-dir runs/boltons-vm-assets \
  --output runs/boltons-virtio-action \
  --repetitions 100 --warmup 10
```

One real QEMU/HVF run alternated condition order on the same booted VM. For 100 measured pairs, the host-clock medians were **2.890 ms** for a serial shell `true` versus **0.230 ms** for virtio `PING`, and **70.447 ms** for serial shell file observation versus **0.511 ms** for virtio file observation. The corresponding p95 values were **3.302/0.420 ms** and **78.002/0.852 ms**. The complete bounded read observations matched exactly before timing, after each of all **110 pairs** (10 warmup and 100 measured), and after timing; the source digest stayed fixed. The [path-free report](measurements/virtio_action_readonly_2026-09-25.json) records sample counts, tails, asset hashes, and setup. VM boot took **5.719 s** and was excluded from the per-action samples.

These are end-to-end *implementations* of read operations, not isolated virtio-serial wire latency: the serial path starts `sha256sum`, `wc`, `head`, and `base64` for each read, while the persistent Python process hashes and encodes in memory. The standalone probe omits the adapter's separate whole-workspace symlink scan and is not itself a policy-tool replacement. The later [graded adapter](VIRTIO_GRADED_AB.md) retains that scan and routes only read actions through the agent; writes, checks, submission, and grading use their existing paths. The [preinstalled-image A/B](VIRTIO_PREINSTALLED_IMAGE_AB.md) found a much faster read stage but no reliable whole-episode gain. The guest agent shares the VM's kernel and is suitable only for the pinned trusted research sample.

The runtime pins and rechecks the guest kernel/initramfs digests, checks each read-only module disk before restore, requires an independent qcow2 disk per VM, uses short private host UNIX sockets for serial and QEMU monitor control, limits command output and duration, and records `savevm`/`loadvm` timing. It identifies the SquashFS module disk and ext4 work disk by filesystem magic inside the guest rather than trusting QEMU's device order. The adapter keeps verifier expectations on the host and treats VM/transport failure as pending infrastructure, never as a policy reward of zero.

This is a local QEMU/HVF implementation. [Firecracker](https://github.com/firecracker-microvm/firecracker/blob/main/docs/getting-started.md) requires Linux/KVM and is not the hypervisor running on this macOS host. QEMU's [ARM `virt` machine](https://www.qemu.org/docs/master/system/arm/virt) supplies the hardware-virtualized ARM64 guest here. The environment currently has no virtual network or broader analyst web tools inside the coding VM; forecasting research tools live in the separate prediction track. The trainer integration remains a future step: exported text trajectories deliberately say `trainer_ready=false` because token-level behavior log probabilities and a weight-updating optimizer are not present.
