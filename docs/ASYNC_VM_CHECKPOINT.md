# Turn-boundary VM checkpoint overlap

The [Crab paper](https://arxiv.org/html/2604.28138v1) places checkpoint work
inside the wait for a model response and withholds that response until the
checkpoint has committed. This experiment implements that **completion gate**
for the project's trusted QEMU coding adapter. It builds on the full-VM
`savevm` and durable host manifest in
[`CodingVMRecoveryJournal`](../future_prediction_bench/semantic_vm_recovery.py).

[`AsyncVMCheckpointCoordinator`](../future_prediction_bench/async_vm_checkpoint.py)
first executes one bounded tool action and captures its observation. It starts
the same full QEMU checkpoint on a worker and passes an immutable serialized
observation to the policy callback. The callback can produce a proposed next
action while QEMU saves. The coordinator waits for **both** results, verifies
that the policy echoed the observation digest, and fsyncs a boundary record
binding the observation, chosen action, checkpoint tag, and recovery manifest.
Only then does it release the action. The terminal `submit` path checks the
latest committed boundary before invoking the host-private grader. A failed
step, checkpoint, policy call, or boundary publication halts the coordinator;
the worker is awaited before the VM is closed. An injected failure after
`savevm` but before manifest publication cannot release the policy action or
receive a reward.
The requested action is copied into canonical JSON before execution, so a
caller mutation while the worker runs cannot alter the committed action.
Cancellation anywhere after the guest action begins marks the journal as
requiring recovery and halts this coordinator. If a checkpoint worker was
submitted, it is joined before the exclusive gate is released.

The serial A/B arm uses the same action, QEMU save, fsync, manifest, policy
callback, and grader. It waits for the checkpoint before starting the policy
callback. The overlap arm starts the policy callback while the checkpoint
runs. Both arms checkpoint **every** nonterminal turn, so a difference in
checkpoint selection does not confound this timing comparison.

## Reproduce the pinned QEMU experiment

Build the public Boltons v2 task and ARM64 guest as described in
[the microVM guide](MICROVM_ENV.md). Then run, with an empty output directory:

```bash
python3 -m examples.realworld_boltons26.benchmark_async_vm_checkpoint \
  --task-dir runs/boltons-v2-task \
  --assets-dir runs/boltons-v2-assets \
  --output runs/async-checkpoint-reproduction \
  --wait-ms 100 --pairs 2
```

`--pairs 2` reverses serial/overlap order in the second pair. Each arm gets an
independent qcow2 clone, executes the same read, exact repair, visible check,
and submit sequence, and grades all 14 hidden cases from VM snapshots. The
script rejects any difference in per-action observation digests, each hidden
case result, final source hash, checkpoint counts, or reward. It measures the
entire graded episode, checkpoint work, policy wait, actual checkpoint/policy
time intersection, and exposed completion-gate delay. The policy callback in
this benchmark uses `time.sleep`; it is an **inference-latency surrogate**, not
an LLM run or an RL training throughput result.

## Same-host result on 2026-09-25

The final-source experiment ran on macOS QEMU/HVF at 128 MiB. Every arm made
four journal snapshots (initial plus three turn boundaries) and five QEMU
snapshots including submission. All arms produced identical action
observations, all 14 hidden case results, final source hash, and reward 1.
The local task JSON SHA-256 was
`7c411f32f13ec31d028b7d90982af8df6a7869d9baf765681530644a970fdc14`;
the guest asset-manifest SHA-256 was
`3e853377e7a7fd53225be4b7aa905062b109d845745e8558ba52db6f01707bcd`.
The benchmark checks the manifest against the task's source distribution,
seed workspace, and actual qcow2/module disk digests before starting. A newly
generated task has new timestamps and a different task JSON digest.

| Simulated wait per turn | Serial graded episodes | Overlapped graded episodes | Exposed checkpoint gate across six overlapped turns |
| --- | --- | --- | ---: |
| 100 ms | 8.337, 8.441 s | 8.051, 7.939 s | 0.006826 s |
| 0 ms | 8.168, 8.706 s | 8.051, 8.863 s | 0.528095 s |

With 100 ms available, actual checkpoint/policy time intersection totaled
0.523 s over six overlapped turns. At zero wait, intersection was 0.00014 s
and the gate exposed the checkpoint work. This directly verifies the
completion-gate mechanism. Whole graded episodes varied: the two 100 ms
paired differences were +0.286 s and +0.503 s (serial minus overlap); the
two zero-wait differences were +0.117 s and -0.157 s. Two pairs do not
establish a stable whole-episode throughput gain, especially while VM boot
and the 14-case grader remain serial.

Sanitized reports with per-turn timings are
[`100 ms`](measurements/async_vm_checkpoint_100ms_2026-09-25.json) and
[`0 ms`](measurements/async_vm_checkpoint_0ms_2026-09-25.json), SHA-256
`29a087c2ce1a693d63ae78e2f1b1dada01a00641b56d314be790b56faa76db24`
and `d8738c5dd7a3b7b4c8523e1ae4aac75a362bc4e0716428ac3a72d2d1885a0a9c`
respectively. These reports exclude per-case outputs; the benchmark compares
them in memory before declaring parity.

## Scope and limits

This is one QEMU VM per arm under an exclusive, trusted host coordinator.
The callback receives only an observation string and digest, and must not
capture or call the VM adapter itself. `savevm` stores full VM state, including
CPU, RAM, devices, and the writable qcow2. It cannot run concurrently with
another guest command on the same VM. The completion gate enforces that rule
for actions issued through this coordinator.
There is no built-in deadline for a blocked policy callback; the caller must
provide and enforce one outside this experimental coordinator. The underlying
journal supports a same-live-adapter terminal-verifier retry after a durable
submission record; this coordinator does not expose that retry as an
automatic recovery operation.

Crab's eBPF change inspector, filesystem/process-specific ZFS/CRIU snapshots,
host-wide urgency scheduler, LLM proxy, and agent-process recovery are outside
this implementation. The journal's existing durability limit also applies:
its fsync and atomic manifest narrow the publication window but do not prove
atomic recovery after host power loss. The simulated wait isolates overlap
behavior; a real policy service and a larger multi-host workload are needed
to establish training throughput.
