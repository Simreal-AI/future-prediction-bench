# Guest-local mount-namespace branches

This experimental path narrows the sharing left by the earlier [guest-local fork and overlayfs experiment](MICROVM_ENV.md#guest-local-fork-and-overlayfs-branch-experiment), taking the per-instance mount-view idea from [SWE-MiniSandbox](https://arxiv.org/abs/2602.11210) into this VM's trusted branch process. A trusted host program runs inside one pinned, network-disabled ARM64 QEMU/HVF VM. It warms the Python Boltons module once, then forks branch processes. Each child calls [`unshare(CLONE_NEWNS)`](https://man7.org/linux/man-pages/man2/unshare.2.html), makes [mount propagation private](https://man7.org/linux/man-pages/man7/mount_namespaces.7.html), mounts its own tmpfs for [overlayfs upper/work files](https://docs.kernel.org/filesystems/overlayfs.html), and mounts an overlay over the shared read-only-in-practice source tree at the **same pathname**. The parent and sibling processes retain their separate mount views. Branch heap changes use Linux fork copy-on-write memory.

The correctness check holds two children alive at once. One patches the public Boltons `glass` repair in its overlay and returns `glass`; the other reads the untouched source and returns `glas`. Both inherit the warmed counter value `7` and change it independently. The host mounts `/proc` inside the chroot before timing, and each child reports a different `/proc/self/ns/mnt` identifier from the parent and sibling. Their upper tmpfs and overlay device IDs also differ, the parent mountpoint stays empty, and the shared lower source hash remains unchanged. The repeated trial creates a fresh namespace and two mounts, writes a marker in the child overlay, releases the child, reaps it, and checks that the marker and mounts did not appear in the parent view. Failed setup or cleanup fails the run rather than counting as a successful branch.

Reproduce with the pinned VM assets prepared in [the microVM guide](MICROVM_ENV.md):

```sh
python3 -m examples.realworld_boltons26.benchmark_guest_mount_namespace \
  --assets-dir runs/microvm-assets-v2-20260925 \
  --output runs/guest-mount-namespace-reproduction \
  --repetitions 200
```

The host validates the pristine qcow2 hash against its manifest, uploads the exact trusted guest program with a SHA-256 check, runs it in the already-booted VM, and parses one framed report. The [200-cycle report](measurements/guest_mount_namespace_v0.6.0.json) contains guest-clock p50/p95/min/max, separate host setup and call times, correctness evidence, and asset/program digests. `fork_to_ready` includes the guest's namespace/mount setup, marker write, source read/hash, baseline call, and one-byte handshake; `release_to_reap` includes unmount and IPC. Individual syscall timings isolate `unshare`, private propagation, tmpfs mount, overlay mount, and unmount operations. The measured cycle excludes VM boot, model inference, hidden grading, and optimizer work.

| Final 200-cycle guest-clock measure | Median | p95 |
| --- | ---: | ---: |
| `unshare(CLONE_NEWNS)` | 0.012 ms | 0.018 ms |
| Mount private tmpfs upper | 0.013 ms | 0.017 ms |
| Mount overlay view | 0.022 ms | 0.034 ms |
| Fork to ready, including marker/source exercise | 0.692 ms | 0.809 ms |
| Release through unmount and reap | 0.123 ms | 0.143 ms |
| Complete branch cycle | **0.864 ms** | **0.990 ms** |

The same run took **5.653 s** to boot the VM, **0.259 s** for the 200-cycle guest program call, and **6.136 s** overall from the host clock. The guest-clock cycle distribution is measured after boot; the host call includes serial transport and report encoding. These are different timing scopes.

This is **not** a secure untrusted-agent sandbox. All branches share one guest kernel, and a root process with unrestricted `/proc`, mount, or signal authority can inspect or interfere with sibling processes. It is also not a full VM CPU/RAM/device checkpoint, and it is not yet a real-world task adapter with policy actions or per-branch host-private grading. Its milliseconds cannot be compared as equivalent operations with full QEMU `loadvm` or published checkpoint/restore numbers. The experiment demonstrates that private mount views can be added to the trusted guest-local COW primitive while retaining millisecond-scale branch setup on this host.
