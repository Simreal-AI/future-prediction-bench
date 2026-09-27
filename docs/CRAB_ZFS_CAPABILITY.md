# Crab: genuine ZFS filesystem capability

Date: 2026-09-27. This new result extends the earlier process-only runc/CRIU
work to actual ZFS filesystem snapshot/rollback. It is one bounded component
experiment, not a complete agent checkpoint or a training speed comparison.

The unchanged [Crab source](https://github.com/open-agent-infra/crab/tree/9607d61a41dc44358cf078c4b438bfd971c8ee9d)
is pinned at `9607d61a41dc44358cf078c4b438bfd971c8ee9d`, under MIT. Normal
imports call the actual `AdapterFileSystemCWorker` and `RuncRuntime` methods;
the runtime, module loader and ZFS commands are not replaced with doubles.
External source and operational guest images remain outside the public bundle.

## Actual recovery

An owned 512 MiB file vdev is created inside a disposable guest, with no host
block device, guest network or host directory mount. Genuine ZFS creates the
pool and dataset. The fixture writes and fsyncs a deterministic 65,536-byte
file, calls the original worker to create a snapshot, genuinely changes the
file and fsyncs it, then calls original `restore_filesystem` for rollback.

| Witness | Actual result |
| --- | --- |
| Snapshot command | `zfs snapshot fpbcrab/caps@cp0`, executed successfully |
| Restore command | `zfs rollback -r fpbcrab/caps@cp0`, executed successfully |
| Saved whole-file SHA-256 | `8b0a3e846d9808a933047cf1e7976c0ecffb9d4d746de63128092ebaaffe9028` |
| Damaged whole-file SHA-256 | `45ecf79eb02406560b5735265feb43ef07180eb0c4c2ade50461f1831d276ac9` |
| Restored whole-file SHA-256 | Exact saved hash; all 65,536 bytes also compared |
| Original worker checkpoint | 81.211406 ms, one observation |
| Original runtime rollback | 108.350542 ms, one observation |

These component timers include the original Python adapter/runtime call and
its real command subprocesses. They exclude guest boot, package assembly,
pool/dataset preparation, file verification and final pool destruction.
Snapshot metadata records written bytes 163,840 and used bytes 0 at that
instant; these ZFS fields are not CRIU image sizes or network-traffic metrics.
Actual pool destruction and guest sync also succeed.

## Matching kernel and actual tools

Execution uses x86-64 QEMU/TCG with 1,024 MiB guest RAM, stock Alpine Linux
`6.18.53-0-virt` and stock ZFS userspace/kernel version `2.4.4-1`. Nineteen
genuine APKs, totaling 84,622,181 bytes, are bound to the recorded APKINDEX,
compressed control checksums, control-bound payload SHA-256 and whole package
SHA-256. The source tree, GNU tar and 121 case-sensitive xtables paths are
verified during offline image assembly.

The initrd contains genuine matching modules. The original kmod 34.2 binary
actually regenerates the binary dependency indexes using its `depmod` alias;
the resulting indexes and tool SHA-256 are recorded. Actual `virtio_pci` and
`virtio_blk` discover the owned disk. Actual ZFS/SPL modules load with a bounded
128 MiB ARC setting. The shipped module cohort omits legacy `ip_tables` names;
the original default iptables tools report `nf_tables`, whose real modules
and read-only ruleset check pass. No CRIU network-lock bypass is used.

The guest initial mount namespace binds its owned ext4 temporary directory at
`/tmp`, making the same file-vdev absolute path visible to kernel-side file
opening and the private namespace used by the probe. The existing correct
private `MS_MOVE`/chroot root contract remains. Actual `criu check`, memory
dirty tracking, original GNU tar and default network-tool prerequisites pass.
The new kernel still has `CONFIG_USERFAULTFD` disabled; this result does not
enable lazy-pages.

## Evidence and limits

The [curated measurement](measurements/official_crab_zfs_capability_2026-09-27.json)
preserves the full guest result, original commands, actual setup observations,
source/asset hashes and preceding failed-attempt records. Host paths and base64
source-transfer commands are excluded. Failed attempts include missing binary
dependency indexes, missing disk discovery, an unavailable legacy module and
a file-vdev path failure; none is counted as successful recovery.

Two subsequent actual full process-plus-filesystem attempts are retained as
failed composition evidence. Original composite checkpoint/restore commands
succeed, and eight-MiB RAM plus held FD/offset recover. Renamed/added files
also roll back, but the mutable ledger still has its damaged bytes. A second
attempt observes this mismatch immediately after original filesystem rollback,
before original process restore. This locates the failed witness in filesystem
rollback visibility; its underlying cause is still under investigation. No
equality requirement is relaxed and neither attempt is counted as complete
recovery or a performance improvement. The process-only negative controls
recover RAM while demonstrably retaining the damaged workspace.

The matching `.53` environment is a separate cohort from the earlier `.52`
process timing experiments. The component result above has one successful
observation, without a fair baseline or speedup ratio. Its primitive times do
not include a restored process/held FD, repository grader, model generation or
GPU training.

The subsequent [composite recovery experiment](CRAB_COMPOSITE_RECOVERY.md)
now passes five bounded whole-RAM/workspace/held-FD recovery trials, including
two executions of the public reproduction scripts. It retains the failed
original rollback controls and adds a local nonforced unmount/rollback/remount
sequence without patching the upstream runtime or relaxing byte, inode, device
or offset checks. Its larger timers belong to that separate full recovery
contract; they do not replace the single-file capability timings above. The
public scripts prepare source-pinned guest assets and execute the bounded
probe. General concurrent-writer atomicity, eBPF and native KVM performance are
not established by either experiment.
