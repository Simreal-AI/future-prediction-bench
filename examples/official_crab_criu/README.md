# Original Crab process checkpoint chains inside a disposable Linux VM

This experiment executes the unmodified Crab scheduler, fault-tolerance
policy, kernel soft-dirty monitor, process worker, and runc runtime from
commit `9607d61a41dc44358cf078c4b438bfd971c8ee9d`. It calls the real runc and
CRIU binaries. It does not use a capturing command runner or substitute
Python functions for either backend.

The experiment compares three modes on the same private-memory workload:

| Mode | Checkpoint selection | Process image backend |
| --- | --- | --- |
| `every_turn_full` | Every boundary | Standalone full CRIU dump |
| `selective_full` | Original change-driven Crab policy | Standalone full CRIU dump |
| `selective_incremental` | Original change-driven Crab policy | Original pre-dump plus final-dump chain |

## Repeated actual results: 2026-09-27

Three cyclic mode-order runs at each of eight and 64 MiB completed all **18
ordinary recovery challenges**. Each destroyed the original process, recovered
every owned RAM byte with the saved SHA256, proved a distinct damaged state,
and advanced both counters again after restore. The
[measurement record](../../docs/measurements/official_crab_criu_process_chain_2026-09-27.json)
retains all six runs; the
[method/results note](../../docs/OFFICIAL_CRAB_CRIU.md) defines the timers and
complete retained-image accounting.

| Owned RAM | Mode | Restore points / checkpoint commands per run | Median all `*.img` bytes | Median wall / primitive restore (ms) |
| --- | --- | ---: | ---: | ---: |
| 8 MiB | `every_turn_full` | 6 / 6 | 50,533,216 | 3,005.512 / 507.293 |
| 8 MiB | `selective_full` | 3 / 3 | 25,254,358 | 1,951.626 / 488.299 |
| 8 MiB | `selective_incremental` | 6 / 12 | 8,546,235 | 3,269.563 / 317.188 |
| 64 MiB | `every_turn_full` | 6 / 6 | 402,831,786 | 9,885.774 / 2,121.535 |
| 64 MiB | `selective_full` | 3 / 3 | 201,403,600 | 6,960.774 / 2,092.548 |
| 64 MiB | `selective_incremental` | 6 / 12 | 67,266,784 | 6,268.938 / 624.074 |

At 64 MiB, selective full and incremental median wall times are 29.6% and
36.6% below every-turn full dumping; all three paired cycles favor both.
At eight MiB, selective full is 35.1% faster while incremental is **8.8%
slower** in median total wall time. Incremental image volume falls by 83.1%
and 83.3%, respectively, but all six restore points remain because the actual
monitor still reports dirty pages at idle boundaries. It executes twelve
checkpoint commands and retains every parent and prior final image. Selection
and image-volume savings therefore remain separate measured effects.

Each size rotates modes through all three sequential positions, with one
frozen guest asset set per size. The 64-MiB series uses the complete-identity
and quiescent signal-wait guard described below; the eight-MiB series retains
its preceding checker pin. These cohorts are not pooled. All timings are
x86 QEMU/TCG on an ARM Mac, not native-speed, graded rollout, or training
measurements. Dirty-page diagnostics were disabled in the timed series.

## First actual successful run, retained separately

The eight-MiB probe passed all three real runc/CRIU modes in
`runs/official-crab-criu-gnutar-20260927/`. This was one sequential x86 QEMU/TCG
comparison on an ARM Mac, using the original source and reviewed real binaries.
The [measurement record](../../docs/measurements/official_crab_criu_process_chain_2026-09-27.json)
and [detailed method/results note](../../docs/OFFICIAL_CRAB_CRIU.md) retain the
counts, commands, source pins, and exact RAM verification.

| Mode | Restore points / checkpoint commands | All `*.img` bytes | Mode wall / primitive restore (ms) |
| --- | ---: | ---: | ---: |
| `every_turn_full` | 6 / 6 | 50,508,626 | 2,968.484 / 505.191 |
| `selective_full` | 3 / 3 | 25,254,313 | 1,925.841 / 477.693 |
| `selective_incremental` | 6 / 12 | 8,546,235 | 3,005.971 / 304.178 |

Selective full checkpoints skipped three idle boundaries. Incremental
checkpointing kept all six restore points because the actual dirty-page monitor
still reported changes at its idle boundaries; both commands and all parent
images are included. Therefore the run demonstrates separate selection and
image-volume effects, not a combined six-to-three incremental speedup.
Incremental image volume was 83.1% lower than every-turn full dumps, but its
total mode wall time was 1.3% longer in this single comparison.

Every mode destroyed the original process, recovered the full 8,388,608-byte
owned allocation with identical saved/restored SHA256, verified a distinct
damaged state, and advanced both counters again after restore. These are real
process recovery results. The repeated series above extends this initial
single-sample result. Full ZFS/eBPF recovery, native speed, graded rollout, and
training throughput remain separate validation; earlier failures are retained.

A separate [interleaved-write diagnostic](../../docs/CRIU_PAIRED_EPOCH_CHALLENGE.md)
completed an initial eight-MiB run plus readiness-guarded repeats at eight and
64 MiB. Each run's zero-write control passed; a witnessed write after the
original pre-dump was lost by the later original chain and failed the complete
owned-region hash. A local runtime subclass using the previous completed
process image as the next pre-dump's parent recovered byte 107, the full saved
hash, and continued execution in all three candidate trials. The fixture writer
and caller hook are explicit diagnostic deviations. This candidate is absent
from the ordinary timing tables and leaves all upstream source unchanged.
Each original failed recovery remains recorded; a passing expected diagnostic
is not a claim that all actual recoveries passed.

## Workload and recovery checks

The controlled C process owns anonymous writable memory and blocks in
`sigwaitinfo` between actions. Two signals change different pages. The
observer reads counters through `/proc/PID/mem`; observations never wake
the worker. Each mode records actual kernel change signals, decisions,
commands, new image bytes, checkpoint latency, and restore latency. It
then changes RAM beyond the saved boundary, destroys the running
container, and restores with original `RuncRuntime.restore_process`.
Both saved counters must reappear exactly, and SHA256 of every byte of
the worker's owned anonymous allocation (at most 64 MiB) must match its
pre-damage hash, including unmodified pages. The observer reads exact-length
1 MiB chunks through `/proc/PID/mem` without waking or modifying the worker.
The post-checkpoint damage must also change the full-region hash before
the container is destroyed. Hashes, counter pairs, the allocation's original
address and length, and verification times are recorded. Hashing is
outside checkpoint/restore timers and included in the total probe wall time;
this verifies the owned allocation, not all process or device state.
After the saved-state hash matches, both mutation signals must again advance
their separate counters. This continued-execution challenge is recorded
separately, outside the restore timer and inside total wall time.

The current checker waits for a complete identity JSON, bounded to 1,024 bytes,
with exactly the positive integer address, length, and page-size fields.
Counter equality also requires the worker to be quiescent in x86-64
`rt_sigtimedwait` (syscall 128). File creation alone is not a ready signal.
The final 64-MiB series and both repeated epoch diagnostics use this guard;
the earlier eight-MiB series remains labeled with its preceding checker pin.
Per-image inventories and parent symlink targets are collected after the mode
wall timer and must agree with its complete image-byte sum. Per-command dump,
pre-dump, and restore log tails are retained with SHA256 on the host outside
that timer. Final container cleanup is also excluded from mode wall time.

The original process worker performs `pre_dump(new, parent=previous)`
followed by `checkpoint_process(new, parent=new)`. Consequently the final
image links to its own `../pre_dump`; that pre-dump links to the previous
checkpoint's pre-dump. The script retains every ancestor for the full
experiment. It does not prune or deduplicate parent images.
The incremental mode leaves soft-dirty resets to CRIU's `track_mem`
epoch; it never clears those bits externally. The standalone full-dump
modes reset their selector baseline only after the full dump commits.
The controlled worker receives no mutation signal during a checkpoint
pair. The linked paired-epoch diagnostic separately challenges one controlled
interleaving; arbitrary concurrent applications remain unvalidated.
The container retains its real network namespace and original CRIU
default network-lock path. Genuine Alpine `iptables-restore` and
`ip6tables-restore` executables perform locking; there is no fake command
or network-lock bypass. The probe adds no `org.criu.config` annotation
and no active configuration option. Preflight verifies both real
executables and requires any default runc CRIU configuration to contain
only comments/blank lines. Result rows record this default backend and
the absence of a custom configuration.
Stock runc 1.4.3 also defaults its CRIU `EmptyNs` mask to the network
namespace because runc does not manage network device configuration.
The namespace still exists and is locked; this original default omits
device, address, and route properties from recovery. This probe never
claims restoration of a configured network or established connections.
"Process-only" describes this probe's workload and the absence of a separate
Crab/ZFS filesystem worker. Real CRIU still captures the mount metadata and
tmpfs contents required to restore the container; genuine GNU tar is required
for that backend operation. The owned-memory hash does not establish recovery
of an independently changing workspace.

## Prerequisites and provenance

Run only in an operator-created, disposable x86-64 Linux VM as guest root.
The default scripts require these reviewed Alpine binaries:

| Input | Version | SHA256 |
| --- | --- | --- |
| `runc-1.4.3-r1.apk` | Alpine v3.24/community | `1fb8426d88027efcb8f62a7919c5fbc58e2be819b46c42e4a792cbe18ea81d8c` |
| Extracted `usr/bin/runc` | 1.4.3 | `09653b0f4c473d6c855a0cd522dcc27a69b34b2403214ff67ebbbf22b360ec9c` |
| `criu-4.2-r0.apk` | Alpine edge/testing | `c502eac15bfd61b6ba031cc52dd85cdc213b913467f8a536d2e00910135c5650` |
| Extracted `usr/sbin/criu` | 4.2 | `7603b91bf98249ccb5b956c11c1b14e746309c21b90669893e29b84eb6d8795b` |
| `tar-1.35-r5.apk` | GNU tar 1.35, Alpine v3.24/main | `5dad2fe0c7d18394dd7ffb64682760ce63718a7a1552b661f6e63036ce8b4958` |
| Extracted `bin/tar` | Original GNU tar executable | `2d3e170780a649c3a4cd8dd3e86960644b8a66ee7a76d39d9ecabe157bf98617` |
| `acl-libs-2.3.2-r1.apk` | Actual `libacl.so.1` dependency provider | `3f02851d586c25d97ba8b5b845615aacc209ae7eaeb6f7843eca76d85da2249d` |

The original Crab package's 52 Python files and the upstream
`integrations/` tree's 52 Python files are each checked before import.
Crab's eager package initialization imports its original request
classification integration, so both unmodified trees are required.
Sorted paths relative to the upstream checkout root and file bytes,
separated by NUL bytes, hash to:

- `crab/`: `c6d0439e627c75ecc9aea47943212ece93a900605cbac924e99d823ee44b657b`
- `integrations/`: `fe0edc09ea062f3495807d3da0f0d5a78e5855ec3f56c426f90d092c39bf897d`

Copy both complete unmodified trees and the upstream license into the
guest; do not replace eager imports or copy selected functions into this
probe. Both source hashes and counts appear in the result.

The guest needs Python 3.11+, the runc/CRIU shared-library dependency
closure, and a static C worker. The script builds the worker with
`gcc -static`, or accepts a previously reviewed static binary through
`--worker-binary`. Its SHA256 is recorded in each result. GCC is needed
only to build that binary; checkpointing itself needs no compiler and no
GPU. A native x86 Linux CPU host is needed for meaningful native-speed
measurements; x86 QEMU/TCG on an ARM Mac can establish execution and
correctness, but its timings are emulation measurements.

Create the following regular file **inside the disposable VM**:

```text
/etc/fpb-disposable-criu-guest
```

Its exact content, including the final newline, is:

```text
Future Prediction Bench disposable CRIU guest v1
```

The marker records the operator's guest setup. It is not VM attestation
and does not make a host execution safe. These scripts refuse macOS,
non-x86 Linux, non-root execution, and absent/mismatched markers.

## Automated public reproduction

Run these commands from the project root with Python 3.12 or newer, `curl`,
Git, QEMU utilities (`qemu-img`, `qemu-system-x86_64`), and Docker available.
The offline filesystem builder additionally needs an already cached immutable
**ARM64 Linux image containing Python 3.12 and `mke2fs`**. It never pulls an image. These host
commands build and start a disposable guest; the CRIU checker executes as
root inside that guest. They do not install CRIU on the host or change host
kernel settings.

The public [`package_inputs.json`](package_inputs.json) contains all **35
APK URLs, versions, sizes, and actual SHA-256 pins**. It has no machine paths
or credentials. The fetcher also fixes five kernel/Python/musl base inputs
and the official PyYAML 6.0.3 source archive, for **41 downloaded files**.
No private `runs/` inputs are required:

```bash
python3 -m examples.official_crab_criu.fetch_inputs \
  --source-output runs/criu-public-source-inputs \
  --packages-output runs/criu-public-apk-inputs --workers 4

git clone https://github.com/open-agent-infra/crab.git runs/criu-public-crab
git -C runs/criu-public-crab checkout 9607d61a41dc44358cf078c4b438bfd971c8ee9d

python3 -m examples.realworld_boltons26.make_task_v2 \
  --output runs/criu-public-boltons-task

python3 -m examples.official_crab_criu.prepare_guest \
  --source runs/criu-public-source-inputs \
  --packages runs/criu-public-apk-inputs \
  --task-dir runs/criu-public-boltons-task \
  --crab-source runs/criu-public-crab \
  --output runs/criu-public-guest-assets \
  --build-image sha256:<your-cached-arm64-mke2fs-image-id> --disk-mib 3072

python3 -m examples.official_crab_criu.run_microvm \
  --assets runs/criu-public-guest-assets \
  --output runs/criu-public-8m-forward --memory-mib 8 --mode-order forward

python3 -m examples.official_crab_criu.run_microvm \
  --assets runs/criu-public-guest-assets \
  --output runs/criu-public-64m-forward --memory-mib 64 --mode-order forward
```

For the three-order comparison, repeat each run with `--mode-order rotate`
and `--mode-order rotate2`, using a new host output directory each time. For
the controlled epoch diagnostic, choose `--probe-program paired_epoch` in a
separate run; its expected negative result is interpreted separately from the
ordinary three-mode comparison.

All output directories must be new. The fetcher's source and APK directories
must also be disjoint. It accepts no arbitrary URL or credential option;
requests use fixed primary HTTPS sources and at most four concurrent `curl`
processes. Each private partial file is checked for exact size and SHA-256
before atomic publication. Only a fully verified batch publishes the builder's
`pinned-package-inputs.json`. Failed batches retain verified files for diagnosis
but remove their own partial files. The bundled package manifest is itself
hash-bound to the fetcher.

To recheck already obtained inputs without making a network request:

```bash
python3 -m examples.official_crab_criu.fetch_inputs \
  --source-output runs/criu-public-source-inputs \
  --packages-output runs/criu-public-apk-inputs --verify-only
```

The builder verifies archive pins, actual APK identities/architectures, both
upstream Python trees, and the copied package digests. It installs the **real
pure-Python PyYAML package**, preserving its optional compiled-backend fallback,
and copies upstream licenses. It excludes unrelated fixture data and terminal
databases whose names collide on case-insensitive APFS. Guest-absolute symlinks
are rewritten relative to the staging root; safe tar extraction never writes
host-root paths. The default filesystem is 512 MiB; the command deliberately
uses three GiB for the compiler and all retained 64-MiB images. An earlier
768-MiB attempt exhausted its disk. Before running, the current ordinary
checker requires actual free space of at least `4 × 6 × RAM_bytes + 64 MiB`
(1,600 MiB at 64 MiB); the three-trial epoch probe requires
`3 × 2 × 6 × RAM_bytes + 64 MiB` (2,368 MiB at 64 MiB). No ancestor is deleted
to satisfy those checks. The Boltons seed is
only an asset input here; this experiment does not grade a Boltons repair.

GNU tar is a genuine additional runtime dependency, not a compatibility wrapper.
CRIU 4.2 invokes flags such as `--no-unquote` when saving tmpfs contents; BusyBox
tar does not support that operation. The fixed Alpine graph provides GNU tar
1.35 and its actual `acl-libs` dependency; those reviewed records depend on the
already included musl and do not require a guessed `libattr` package. The builder
requires the exact regular `/bin/tar` payload, its x86-64 ELF header and SHA256,
and the original ACL shared library. It rejects an earlier PATH entry that could
shadow that executable. BusyBox's original bytes remain unchanged. Guest
preflight separately checks real GNU identity and executes a bounded archive
round trip with the actual CRIU-required flags; no flag is stripped or bypassed.

The iptables archive contains **115 regular xtables plugins and six symlinks**,
including **nine casefold pairs with different payload bytes**. Extracting them
directly on case-insensitive APFS would collapse distinct Linux names such as
`libxt_MARK.so` and `libxt_mark.so`. The builder excludes **all 121 plugin
entries** from host staging, keeps their exact original path/type/hash/alias
records with hash-named byte objects, and reconstructs every entry in a real
case-sensitive Linux tmpfs. Native Python verifies all names, file bytes,
permission bits, and original link targets before native `mke2fs` assembles
the guest filesystem. No guest plugin or binary is executed by this builder.
Unhandled casefold collisions elsewhere, including implicit directory-prefix
collisions, fail before that archive is extracted. The asset directory retains
`xtables-manifest.json` and `linux-stage-verification.json`, whose hashes are
bound in the final asset manifest.

The focused guard tests are dependency-free. The following optional command
additionally compares every entry with the actual pinned APK and performs
real offline Linux reconstruction using the cached image:

```bash
FPB_CRIU_APK_DIR=runs/criu-public-apk-inputs \
FPB_CRIU_BUILD_IMAGE=sha256:<your-cached-arm64-python-mke2fs-image-id> \
python3 -m unittest discover -s tests -p test_criu_case_sensitive_builder.py -v
```

The APK graph deliberately combines Alpine v3.24 dependencies with
edge/testing CRIU. Exact pins reproduce those bytes; they do not establish
general compatibility, supply a production distribution, or guarantee that
upstream mirrors will retain the files indefinitely. If a pinned URL disappears,
the fetch fails rather than selecting a newer package. The asset manifest
records this scope, sources, hashes, and guest marker. The host driver returns
nonzero unless actual guest preflight and all three modes succeed; inspect its
`result.json`, `guest-result.json`, and retained runtime logs for the outcome.

## Alpine emergency-shell guest setup

The existing x86 microVM boot helper mounts the writable root at
`/mnt/root` and loads the pinned Alpine modloop. The original runc
cgroupfs manager needs a writable guest cgroup hierarchy; no systemd is
used. Before chroot, provision `/proc`, `/sys`, `/dev`, `/run`, and `/tmp`
inside that root. If `/sys/fs/cgroup` is not already mounted, mount the
guest's cgroup2 hierarchy there, then bind the guest `/sys` into the
chroot. Bind `/proc` and `/dev` from the guest likewise.

For a freshly booted disposable guest whose root is `/mnt/root`:

```sh
mkdir -p /sys/fs/cgroup /mnt/root/proc /mnt/root/sys /mnt/root/dev /mnt/root/run /mnt/root/tmp
mount -t cgroup2 none /sys/fs/cgroup
mount --bind /proc /mnt/root/proc
mount --bind /sys /mnt/root/sys
mount --bind /dev /mnt/root/dev
```

If the cgroup mount already exists, reuse it; do not run the first mount
again blindly. CRIU's diagnostics may require the guest's `unix_diag`,
`inet_diag`, `packet_diag`, and `netlink_diag` modules. Load available
modules from the pinned modloop in the disposable guest and keep any
failure in the experiment log. The preflight checks the actual kernel;
the kernel version alone never proves support.
The reviewed Alpine iptables executables use the nftables backend.
The guest needs `nf_tables`, `x_tables`, `nft_compat`, and the `xt_mark`
match support. The preflight records real executable versions and a
read-only `nft list ruleset` result; the actual checkpoint still must
prove that network locking succeeds.

For the automated host path, first build the pinned offline guest with
`prepare_guest.py`, then use the host-side `run_microvm.py` driver. The
driver performs the guest mounts and executes this checker through the
guest's serial interface, saving the returned JSON on the host. These
wrappers do not change the process checkpoint backend inside the VM.
The driver creates a private mount namespace, makes mount propagation private,
saves its working directory on the real ext4 root, moves that mount onto `/`
with `MS_MOVE`, then chroots through the saved `.`. A plain chroot would retain
the initramfs mount-namespace root and CRIU rejects that mismatch; pivoting the
initial rootfs is not the selected handoff. The control shell remains intact
outside the private namespace. Inside it, genuine BusyBox installs only missing
applet symlinks; the existing original GNU tar binary remains selected.
Preflight verifies the actual root file descriptor's mount ID, the ext4 root,
and GNU tar before the checkpoint chain runs.

Place the original checkout at `/opt/fpb/crab` and this directory at
`/opt/fpb/probe`, or adjust the paths below. Ensure adequate free guest
disk space for the compiler and all retained images. Eight MiB per
worker is a useful initial correctness run; increasing memory makes the
unchanged-page saving easier to measure.

Use the automated `run_microvm` command above to perform that checked root
handoff and execute preflight and the three-mode probe. Mount preparation or a
plain chroot alone is insufficient for the real process-chain experiment.

Use a new output directory each time. Failed commands remain failures;
the script preserves partial images/logs for diagnosis and deletes only
its named running runc container during cleanup. It never installs
packages, writes sysctls, creates a ZFS pool, or edits the operator's
checkout.

## Scope and next validation

The preflight requires actual `criu check`, the `mem_dirty_track` feature
check, and a real anonymous-page soft-dirty clear/write round trip. It
also records available kernel configuration and executable hashes.
Passing preflight does not prove checkpoint/restore. A successful
`result.json` requires all three actual modes and exact restored RAM.

This probe has an unchanged filesystem after startup. Crab's scheduler
also requests a filesystem checkpoint when process state changes; this
experiment deliberately executes only the process worker and labels
that scope in every result. It therefore does **not** prove the complete
Crab ZFS/eBPF backend, filesystem recovery, running arbitrary agents,
safe concurrent mutation during a pre-dump/final-dump pair, complete
graded episodes, or GPU training acceleration. Those need separate
validation and must not be inferred from this probe's image sizes or
primitive latency.

The original full installer additionally requires Ubuntu x86-64,
Docker, runc, CRIU, ZFS, compiler tools, and the built eBPF inspector.
Run that stack in a separate disposable Linux VM or native host; do not
run its host-root installation script on the Mac. Check the package
candidate for CRIU first: the current Ubuntu package index lists CRIU
for other suites but does not list the exact `criu` package in Noble.

Primary sources: [original Crab runtime and workers](https://github.com/open-agent-infra/crab/tree/9607d61a41dc44358cf078c4b438bfd971c8ee9d),
[runc checkpoint integration tests](https://github.com/opencontainers/runc/blob/main/tests/integration/checkpoint.bats),
[CRIU kernel requirements](https://criu.org/Linux_kernel),
[CRIU image directories and parent chains](https://criu.org/CLI/opt/--prev-images-dir),
[CRIU statistics](https://criu.org/Statistics), and
[Alpine package repository](https://dl-cdn.alpinelinux.org/alpine/edge/testing/x86_64/).

## Reproduced CRIU 4.2 RPC configuration failure

A real prior attempt selected `network-lock nftables` through runc's
supported OCI annotation/config interface. CRIU 4.2 crashed before
creating `dump.log`; the guest kernel reported a fault at address
`0x18`, and runc reported RPC EOF. This was a failure, not a successful
checkpoint measurement.

The original [CRIU 4.2 RPC service](https://github.com/checkpoint-restore/criu/blob/v4.2/criu/cr-service.c)
parses that file through `parse_options(0, NULL, ...)`. The
[4.2 configuration parser](https://github.com/checkpoint-restore/criu/blob/v4.2/criu/config.c)
then checks `argv[optind]` without a NULL guard after a `network-lock`
option. Its NULL argument-vector dereference matches the observed
`0x18` access. The official [4.2.1 parser](https://github.com/checkpoint-restore/criu/blob/v4.2.1/criu/config.c)
adds checks for `argv` and `optind < argc` at that same point.

The current experiment keeps the pinned original 4.2 binary, omits
that option entirely, and supplies the real executables required by
its default network-lock path. Explicitly configuring even
`network-lock iptables` would enter the same broken parser branch and
is deliberately avoided. No upstream source or binary is patched.
Testing an unmodified official 4.2.1 build is a separate future input
revision with new binary pins.

## Other retained failures before the successful run

The initial host staging collapsed distinct xtables plugin filenames on
case-insensitive APFS, despite both originals existing in the pinned APK. The
builder now reconstructs all 121 original entries on Linux and verifies every
file and symlink before filesystem assembly. A later ordinary chroot passed
preflight but CRIU rejected its mismatched task/mount-namespace roots. The
private `MS_MOVE`/saved-cwd handoff corrected that root arrangement. The next
dump collected the full memory mapping but failed when BusyBox tar rejected
CRIU's required `--no-unquote` flag. Supplying the genuine pinned GNU tar and
actual ACL dependency resolved that runtime requirement.

These remain failed experiments, not successful latency samples. The first
complete three-mode pass and its precise scope are recorded above and in
[OFFICIAL_CRAB_CRIU.md](../../docs/OFFICIAL_CRAB_CRIU.md).

## Retained larger-state attempts

The first 64-MiB comparison ran out of space on its 768-MiB disk. In an early
three-GiB series, one startup read found an identity file before its JSON write
completed. The complete bounded JSON and syscall-128 readiness guard addressed
that race, then all three cyclic orders were rerun with one frozen asset set.
The failed runs and two superseded successful runs remain in the measurement
record's `larger_state_prior_attempts`, outside the primary six-run statistics.
The eight-MiB incremental slowdown remains in the primary results above.
