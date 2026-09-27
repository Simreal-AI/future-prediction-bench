# QEMU snapshot transaction guard

The full-VM coding adapter now checks QEMU's snapshot table before `savevm`.
QEMU's monitor command replaces a snapshot with the same tag, including a tag
inherited from a cloned qcow2. The adapter rejects that collision before the
save. After an attempted save it checks that the new tag is visible. If the
monitor command or its acknowledgement fails, it tries to remove any visible
uncommitted tag, verifies its absence, and closes the VM because the failed
write may have changed qcow2 state. If removal cannot be verified, it returns
`vm_snapshot_cleanup_unverified` and writes a durable `.fpb-unsafe` marker
beside the disk. The runtime refuses to reopen or clone a marked disk; callers
must discard it after inspection. If the marker itself cannot be written, the
runtime closes and reports `vm_snapshot_disk_quarantine_failed`, which still
requires the caller to discard or externally isolate the disk.
The marker is tied to the disk pathname: a raw host rename or copy that omits
the sidecar is outside this managed-path guard. Host operators must keep the
disk and marker together until the disk is discarded.
The separately guarded retired-action-port path still requires QEMU-side
port disconnection before and after snapshot operations.

The [real-QEMU fault probe](../examples/realworld_boltons26/probe_snapshot_transaction.py)
used pinned Boltons v2 assets and actual QEMU/HVF monitor commands. It saved a
committed tag, rejected a duplicate without changing QEMU's tag table, and
restored RAM and ext4 markers. A host-injected lost acknowledgement occurred
*after* QEMU completed a real `savevm orphan`; the adapter removed that tag,
closed the VM, and left the committed tag intact. A new VM restored the
committed RAM and ext4 state. In a second disk, an injected `delvm` failure
left the orphan tag visible, wrote the quarantine marker, and caused both a
new runtime start and a managed qcow2 clone to reject the disk before use.
The [path-free report](measurements/snapshot_transaction_guard_2026-09-25.json)
binds the source and pinned assets and records every check.

Reproduce with a fresh output directory:

```bash
python3 -m examples.realworld_boltons26.probe_snapshot_transaction \
  --assets-dir runs/microvm-assets-v2-20260925 \
  --output runs/snapshot-transaction-reproduction
```

This tests QEMU monitor response loss and deletion failure on one local VM.
It does not prove atomicity under host power loss or storage failure. The
historical primitive timings in [MicroVM environment](MICROVM_ENV.md) predate
the extra monitor calls; no current-source latency or RL-training improvement
is claimed from this correctness guard. The related
[terminal VM crash recovery](TERMINAL_VM_CRASH_RECOVERY.md) tests replacement
of QEMU during hidden grading, a separate failure boundary.
