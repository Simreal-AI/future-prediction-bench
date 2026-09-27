# Terminal verification after a host coordinator restart

The full-VM coding verifier now has an opt-in recovery path for a **Python host
coordinator process** that dies after a committed terminal `submit`. A new
process constructs a fresh `MicroVMRuntime`, `MicroVMCodingAdapter`, and
`CodingVMRecoveryJournal`. The journal restores the submitted QEMU snapshot
and reruns host-private hidden cases. Neither `reset` nor `submit` runs again.
The existing [VM-only recovery path](TERMINAL_VM_CRASH_RECOVERY.md) keeps the
host adapter alive; this path reconstructs it from bounded durable metadata.

The operator first calls `journal.begin()` and then
`journal.enable_host_restart(original_task)` before `submit_and_verify()`.
The adapter records the complete validated task digest at `reset`;
`enable_host_restart` rejects a different prompt or contract even when the
VM artifact binding is identical.
The context is published durably immediately before the terminal attempt,
so a prior VM-process replacement cannot leave a stale QEMU PID in it.
The context file is mode `0600` and contains the frozen task and artifact
digests, exact runtime and adapter configuration, disk device/inode identity,
source-code and latest committed action-manifest digests, and the old QEMU
PID. It contains no hidden expected
outputs. The original task and private `verifier/verify.json` remain on the
trusted host. On restart, the caller creates the same runtime and adapter,
without calling `reset`, then uses:

```python
fresh_journal = CodingVMRecoveryJournal(fresh_adapter, journal_directory,
                                        mode="every_turn")
terminal = fresh_journal.resume_verification_after_host_restart(
    original_task, now=clock())
```

The reopen path requires a canonical context, the same task, code and asset
bindings, the same disk inode, an absent old QEMU process, a valid action
manifest and terminal attempt, and a committed `terminal-submit.json`. It
starts the replacement QEMU paused, checks its read-only disk hashes, and
restores `submitted` with full snapshot validation. If a durable terminal
result already exists, it validates and returns that result without booting
QEMU. A missing submit marker or an ambiguous earlier submit cannot be
recovered this way: the reward stays unavailable and the caller must not
submit a second time. A failed replacement start or restore also yields no
reward; a later retry uses another fresh runtime and adapter.

The old QEMU must be terminated by a trusted supervisor before reopening the
qcow2. The API conservatively refuses a live or recycled recorded PID. QEMU's
normal image lock is a further single-writer check, but this prototype does
not supply a distributed lease or recovery after host power loss. The
context and verifier directory must remain host-private. The supported
contract is the serial-shell, per-hidden-case full-VM verifier; stateless
namespace batches and virtio action-port mode are rejected. This does not
replay an in-flight policy action, restore external services, or implement
Crab's eBPF classifier and ZFS/CRIU backend. It makes no checkpoint-latency,
episode-throughput, or RL training speed claim.

The original disk-seed digest cannot equal the writable qcow2 bytes after
repair and grading, so recovery checks the original device/inode, pinned
read-only assets, and the committed `submitted` snapshot tag instead. A
malicious host that edits the same qcow2 inode or rewrites host-private
records is outside this integrity boundary. A recycled old PID is rejected
conservatively, even when the original QEMU has exited.

The [cross-process QEMU probe](../examples/realworld_boltons26/check_terminal_host_restart_recovery.py)
uses two pinned Boltons v2 arms. One coordinator exits via `os._exit` after
the first hidden case; the parent kills the verified orphan QEMU, then starts
a new Python coordinator. It checks the durable submit boundary, rejects a
live orphan and a mismatched kernel before boot, and compares all 14 hidden
case results, action observations, final source hash, and reward with the
uninterrupted control. Run on macOS with QEMU/HVF and the pinned local assets:

```bash
python3 -m examples.realworld_boltons26.check_terminal_host_restart_recovery \
  --task-dir runs/boltons-v2-task-static-20260925 \
  --assets-dir runs/microvm-assets-v2task-static-20260925 \
  --output runs/terminal-host-restart-reproduction
```

Focused fault tests, including disk replacement, changed task, live orphan,
missing submit marker, corrupt result and idempotent result reload:

```bash
python3 -m pytest -q tests/test_host_restart_recovery.py \
  tests/test_semantic_vm_recovery.py
```

## One pinned real-QEMU validation

On 2026-09-25, two separate-process Boltons v2 runs each passed **14/14**
hidden cases and earned reward **1.0**. The uninterrupted control and
host-kill/restart arms produced identical action observations, all 14
per-case result records, and final source SHA-256. The original coordinator
made one `submit` call; the new coordinator made zero. At the crash boundary,
`terminal-submit.json` existed and `terminal-result.json` did not. The
supervisor confirmed and killed the orphan QEMU; a live orphan and a wrong
kernel were each rejected before replacement boot. The guest `/tmp` marker
written after the first hidden case was absent after restoring `submitted`.
No model or optimizer ran. The [path-free measurement report](measurements/terminal_host_restart_recovery_2026-09-25.json)
has SHA-256 `73af63f1ef01fc56e0722858e613e470e3baf953045a44d0289fd35d38fc01f6`.
The recorded recovery-module SHA-256 is
`6727df7c448e928d73b4ac7fdc9dc03b9f3216cc8a3fc19e044ba7f01bb1c8f0`.
The recorded coding-adapter SHA-256 is
`ece372bed0b5de29da5b3bcf770851a1749c58a32ea514bb995bdc7df2942a7e`.
