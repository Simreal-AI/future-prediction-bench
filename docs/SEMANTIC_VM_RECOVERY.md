# Conservative turn-aligned QEMU recovery experiment

The [Crab paper](https://arxiv.org/html/2604.28138) identifies a real
semantic gap: an agent tool call alone does not reveal whether files, live
processes, or memory changed. Its implementation uses eBPF inspection and
ZFS/CRIU checkpoint backends, with host-level scheduling and overlap during
LLM wait. The journal described in this document does **not** replicate those
mechanisms. The narrower
transfer implemented by
[`CodingVMRecoveryJournal`](../future_prediction_bench/semantic_vm_recovery.py)
uses a trusted, bounded QEMU coding adapter, guest file-tree fingerprinting,
live-process census, full-VM `savevm`/`loadvm`, and a durable host manifest.

A separate [2026-09-27 official-code experiment](OFFICIAL_CRAB_MICROVM.md)
now executes Crab's unmodified process monitor and checkpoint policy against
one stopped worker in an actual x86 guest. Its eight-action capture workload
reduces saves 8 → 3 and median time 18.101 → 16.257 s across three pairs.
It is not this journal's graded task, has no upstream CRIU/ZFS/eBPF backend,
and cannot replace this document's recovery or throughput evidence.

## Recovery contract

The operator exclusively controls one active QEMU adapter. An initial full
CPU/RAM/device/qcow2 checkpoint is required. Only successful `read_file` and
`list_files` turns can skip the next checkpoint. The classifier inspects the
guest workspace tree (file bytes, type, ownership, links, and timestamps) and
the live PID/start-time/executable set before and after each candidate turn.
If either probe fails, differs, or observes a process set other than the
initial baseline, the turn receives a full-VM checkpoint. `write_file`,
`replace_text`, `run_visible_checks`, and explicitly recorded opaque host
guest work always receive a full checkpoint. Once an executable or opaque
turn has run, later reads cannot skip one: an unchanged PID set cannot prove
that a long-lived process did not modify private memory.

For a skipped turn, the host manifest stores the bounded action and a digest
of its observation. Recovery loads the last committed full-VM tag and replays
those reads, rejecting observation divergence as an infrastructure error.
Each replayed read is also bracketed by file-tree and process inspection;
matching visible output alone cannot release a hidden guest mutation. Merely
opening a journal leaves the reward gate closed until the committed snapshot
has been loaded and the read replay verified. An interruption anywhere from
guest action through manifest publication also keeps that gate closed.
The action is copied into canonical JSON before execution so later caller
mutation cannot change its replay record.
Every manifest is written to a private temporary file, fsynced, atomically
renamed, and followed by a directory fsync. A QEMU snapshot is saved and the
qcow2 host descriptor is fsynced **before** the manifest can name its tag.
The manifest also binds the task artifacts, adapter source, runtime source,
journal source, read-only disk contents, and dedicated disk path. A failed save or partial action blocks
further turns and terminal grading until recovery. A snapshot saved immediately
before a coordinator failure but not manifest-committed is an orphan; recovery uses the previous committed
tag and the operator retries the uncommitted action.

This contract assumes no out-of-band writer or network side effect. The
guest has no NIC. The file/process probes are not an atomic eBPF event stream
and cannot identify arbitrary kernel state changes, short-lived concurrent
processes, or semantic memory changes in a persistent process. `savevm` and
the manifest do not establish host power-loss atomicity. Recovery here
restores sandbox state and read observations; it does not restore the
`RealWorldEnv` event log or an actual agent/LLM process. The code is therefore
an experimental, trusted-host recovery primitive, not a general transparent
Crab replacement or an adversarial multi-tenant boundary.

## Retrying terminal verification

Before calling `submit`, the journal durably records a private
`terminal-attempt.json` tied to the committed manifest. If `submit` then
fails before its response can be validated, that episode stays terminal:
`recover()` and a second submit are refused, including from a fresh journal
object. This sacrifices availability because a fixed-name `submitted`
snapshot may already have been created. After a validated `submit`, the
adapter has a full QEMU snapshot named `submitted`. The journal fsyncs that
qcow2 and publishes a private
`terminal-submit.json` record bound to the last action manifest, frozen task
and code identities, submission observation, and a stable `terminal_id`.
No further policy turns are accepted. Only then does it call the host-private
verifier. A resolved grade is written to a create-once, fsynced and validated
`terminal-result.json` before the API returns any reward. Repeating
`submit_and_verify()` or calling `resume_verification()` returns that same
record without a second submit or verifier run.

If verification is interrupted or returns pending, call
`resume_verification(now=...)` with the **same live adapter**, serializing
calls through the single coordinator that owns the adapter. In full-VM
verifier mode, the adapter reloads `submitted` before each hidden case. In
the opt-in stateless verifier mode, it reloads `submitted` once before each
batch, then runs each case in a separate guest mount/PID namespace and
overlay workspace. A partial previous verification attempt can therefore
be restarted within the declared isolation contract. Already completed
cases may be run again; only a fully resolved result is published. If the submit
completed but terminal-marker publication failed, the live journal retains
the validated submission and can publish the marker on retry without calling
`submit` again. Missing, changed, or malformed marker/result records fail
closed. The private terminal directory is never mounted into the guest or
shown to the evaluated policy.

If the **QEMU process** dies after `terminal-submit.json` has committed while
the host adapter survives, the narrower
[`resume_verification_after_vm_crash()`](TERMINAL_VM_CRASH_RECOVERY.md)
path can restart a matching VM from the `submitted` full-state snapshot and
retry grading without another submit. This addition currently has
fault-injected fake-VM tests and one pinned real-QEMU crash/replacement
validation. An absent terminal submit marker still blocks that restart and reward.

This is a bounded same-host-adapter retry, not coordinator-process recovery. A new
journal object may read the records only while the original submitted adapter
remains alive. The adapter's `submitted`, `snapshot`, and cached `verified`
fields are in host memory; a fresh host process cannot attach to the old
submitted VM through this API. An interruption before the `submit` response
can be validated has no terminal marker and permanently closes this
episode's policy/reward path: a fixed-name orphan snapshot may already exist.
The `terminal_id` lets a downstream trainer deduplicate reward delivery;
this module alone cannot enforce exactly-once consumption outside its API.
As elsewhere in this experiment, fsyncs do not establish host power-loss
atomicity.

## Pinned real-QEMU check

Build the pinned Boltons v2 fixture and ARM64 guest as described in
[the microVM guide](MICROVM_ENV.md), then run:

```bash
python3 -m examples.realworld_boltons26.benchmark_semantic_recovery \
  --task-dir runs/boltons-v2-task-static-20260925 \
  --assets-dir runs/microvm-assets-v2task-static-20260925 \
  --output runs/semantic-recovery-reproduction \
  --terminal-interruption
```

The script clones a separate qcow2 for each arm; keeps expected outputs on
the host; executes identical read/list/edit/read/list/visible-check turns;
restores before submission; and grades 14 cases using the full-VM verifier.
The strict run compares every action observation, every case result, final
source SHA, and reward against the every-turn control. A third arm injects a
failure exactly after a real `savevm` and before manifest publication,
reopens the journal, checks that the orphaned edit disappeared, and retries it.
It then checkpoints a live `sleep` process and an initramfs `/tmp` RAM marker,
kills the QEMU process, starts a replacement QEMU paused on the same qcow2,
loads the committed snapshot, and checks both guest states before grading.
Replacement QEMU boot/load failure can be retried while the reward gate stays
closed; the replacement must match the frozen read-only disk hashes.
With `--terminal-interruption`, a fourth arm interrupts after executing the
first real hidden case, leaves a guest-RAM marker, and resumes verification
on the same live adapter. The run asserts that the marker disappears after
the submitted-state restore and that all 14 case results, action observations,
source hash, and reward match the every-turn control. It also checks that a
duplicate terminal call returns the same result.
The third arm tests an actual **VM process** crash; the host adapter/coordinator remain
alive. The public report omits private case outputs; the exact comparison
happens in memory. This is a solved public integration
fixture, not a held-out SWE result.

Five exploratory same-host executions on 2026-09-25 produced:

| Paired run | Every-turn snapshots | Selective snapshots | Complete scripted and graded VM time, every-turn → selective |
| --- | ---: | ---: | ---: |
| Initial run (count/reward parity) | 8 | 5 | 19.986 → 16.146 s |
| Strict per-case parity run | 8 | 5 | 9.390 → 10.981 s |
| VM process-crash and strict parity run | 8 | 5 | 9.104 → 10.140 s |
| Final hardened-inspector / VM-crash run | 8 | 5 | 9.104 → 12.376 s |
| Final action-to-commit guard / frozen-action VM-crash run | 8 | 5 | 9.365 → 9.158 s |

The classifier avoided **3 of 8 full-VM journal saves (37.5%)** in every
run. Both arms scored 14/14, reward 1; the latter four runs also established
exact case-result and observation parity. The initial run checked case counts
and reward only, so it is not evidence of per-case parity. Selective guest
inspections took 0.903 s in the second run, 0.790 s in the first crash run,
0.989 s with the hardened inspector, and 0.704 s in the final guard run.
Whole-episode speed changed sign between runs, so **there is no demonstrated
throughput gain**. The crash arm also passed the manifest-injection and real
QEMU process-kill recovery checks, but its timing is not a fair speed A/B.
These samples also lack policy inference, GPU execution, and optimizer work. The measurable
result is correct selective checkpoint traffic reduction under this limited
contract, with inspection cost currently capable of outweighing it.

The later [terminal-interruption pair](measurements/semantic_vm_terminal_resume_2026-09-25.json)
shows the immediate cost problem: journal snapshot time fell from **0.619 s**
to **0.405 s** when three saves were skipped, a **0.214 s** reduction. The
selective arm spent **0.708 s** inspecting guest files and processes across
the read/list turns. Its inspection work exceeded the saved snapshot time by
about **0.494 s**, while observed complete graded time was **0.207 s slower**
(8.688 versus 8.895 s). The difference between those two deltas reflects
other stage and host variation; it is not evidence of hidden acceleration.
On this exact trace, an equally strict replacement inspector would need to
bring its total work below the roughly 0.214 s snapshot saving, or overlap
that work off the critical path, before the skip decision could be expected
to improve wall time. No such replacement has been validated for this exact
graded journal trace. The later official-code probe has a different controlled
state contract, transport, and workload; its positive result is separate.

The raw ignored reports are
`runs/semantic-recovery-qemu-final-20260925/report.json`,
`runs/semantic-recovery-qemu-parity-20260925/report.json`,
`runs/semantic-recovery-vm-crash-20260925/report.json`, and
`runs/semantic-recovery-hardened-final-20260925/report.json`, and
`runs/semantic-recovery-frozenaction-20260925/report.json`. The earlier sanitized
public [report](measurements/semantic_vm_recovery_finalguard_2026-09-25.json)
has SHA-256 `a2bfe01da9444245fe0c2ef4c6a9aa193e1134687ae80e6caa9f1fa1c852b115`.
The subsequent fresh four-arm terminal-interruption run is in the path-free
[terminal resume report](measurements/semantic_vm_terminal_resume_2026-09-25.json)
(SHA-256 `b63ac12fd593cbba727961e9a002b775d006c7629cd0699aa087e9fc8f3b47b3`).
All four arms scored 14/14 with reward 1 and strict case/observation parity;
the interrupted arm restored guest RAM state before regrading. Every-turn
versus selective saved 8 versus 5 journal snapshots and took 8.688 versus
8.895 seconds in this one run. The interrupted arm took 9.508 seconds;
these values are correctness evidence and do not establish a speedup.
The unit suite in
[`tests/test_semantic_vm_recovery.py`](../tests/test_semantic_vm_recovery.py)
covers skipped-read replay, nominal reads with actual file/process mutation,
opaque process/RAM restoration, orphan snapshot rollback, replay divergence,
malformed manifests, cancellation after a partial action, terminal reward
gating, and failed-restart retry. Its fake VM proves control logic, while the pinned
run proves those specific QEMU state transitions. Neither proves correctness
for arbitrary shell actions or host power failure.
