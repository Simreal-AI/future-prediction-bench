# AReaL: original asynchronous rollout capacity control, executed on CPU

Reviewed and executed on 2026-09-27. The unchanged original
`StalenessManager` passed all **45 official CPU tests**, with no skipped
tests. A separate control-plane probe demonstrated concurrency blocking,
accepted/rejected lifecycle accounting, version advancement, and recovery
budget correction. This provides a concrete reusable admission component
for our future training pipeline. It is not a measured GPU training or
model-generation speedup.

## Exact upstream source

Repository: [areal-project/AReaL](https://github.com/areal-project/AReaL).
The clean reviewed checkout is pinned to
[`2fad2d0e308fe631e70e971b97188ad5c5cc03cb`](https://github.com/areal-project/AReaL/tree/2fad2d0e308fe631e70e971b97188ad5c5cc03cb),
committed 2026-09-27 09:49:39 +08:00. Its declared project version is 2.1.0.
The license is Apache-2.0; this project imports the original code rather
than publishing a modified copy of it.

The upstream July 2026 report,
[Next-Generation Agentic Reinforcement Learning Systems Enable Self-Evolving Agents](https://arxiv.org/abs/2607.01120v2),
connects deployment trajectories, data governance, and an evolution control
plane. The current repository contains the AReaL 2.x service architecture;
the code paths below establish which parts we actually examined and ran.

| Source | SHA256 |
| --- | --- |
| `areal/infra/staleness_manager.py` | `ade1bd484e86d666421716159d6a675ca010c5b5acb8fca05f989389c7f4269d` |
| `areal/api/io_struct.py` | `57643e1e318291ee38f6b7633678dcb535da3bbb330ffcc5c1caefb4566b9359` |
| `areal/infra/workflow_executor.py` | `eee33608c94575e1a777f870d53e44e4cdf6bd449a0a17041f7d7752871ea0e6` |
| `areal/v2/inference_service/controller/controller.py` | `10a1bd6062b11c4e0042397d3eff41311f3bea35f38e8cda360c06c8efcf8325` |
| `areal/trainer/rl_trainer.py` | `62beff00ecd44dca988ffef3c3c4160afae4e488cecd64361a3dd90eaae3a559` |
| `tests/test_staleness_manager.py` | `0b6164520fd63f158d122b6f47dd4659a14982e29b02b74aba04acc57c2afeb8` |

The complete **504-file** `areal/**/*.py` source tree hashes to
`ed205b771ed629a5ca656ade06efe3cc4ad5f245346e89e2e1cd1d1c27926af4`.
The canonical digest concatenates sorted paths relative to the checkout
root, a NUL byte, the unchanged file bytes, and another NUL byte for each
file. The probe checks both this digest and a clean Git checkout before and
after execution. The hardened helper executed for the recorded 45-test
measurement has SHA256
`edf8616b6f5b5e1bd7be2ec807a82c201b0485c9f7abfef75c2328f86d86ca62`:
it rejects source/output overlap before creating any directory
and disables bytecode/pytest cache writes. Tiny guard checks passed for equal,
descendant, ancestor, and symlink-overlap paths without importing AReaL or
Torch. The official suite was then rerun with this exact hardened helper;
the public evidence binds to that completed execution.

## What the original component does

The original [capacity manager](https://github.com/areal-project/AReaL/blob/2fad2d0e308fe631e70e971b97188ad5c5cc03cb/areal/infra/staleness_manager.py)
combines two budgets. Let `v` be the current weight version, `s` the permitted
rollout lead, `B` the consumer batch size, `C` the concurrency limit, `A` the
cumulative accepted count, and `R` the running count. For positive configured
limits, its result is:

```text
capacity = min(C - R, (s + v + 1) * B - A - R)
pending_limit = (s + 1) * B
```

The implementation guards its counters with a real `threading.Lock`. It
increments pending input accounting when enqueued, moves a task from
enqueued to running when submitted, and moves it from running to accepted
or rejected when completed. Accepted means usable rollout data, not a
positive reward. Rejection frees running capacity without increasing the
accepted count.

This is **aggregate admission control**. The class does not inspect the age
of each returned trajectory, token log probabilities, or individual weight
versions. It is not, by itself, a completed-sample freshness filter. It also
does not perform an atomic reserve-if-capacity-positive operation: callers
must coordinate admission and callbacks correctly.

The [original dispatcher](https://github.com/areal-project/AReaL/blob/2fad2d0e308fe631e70e971b97188ad5c5cc03cb/areal/infra/workflow_executor.py)
supplies that context. Its producer loop submits only when capacity is
positive; a separate result thread collects completions and wakes the
producer. It has pause/resume and background-error propagation. Its internal
pending-input deque is unbounded, so the upstream pending limit and the
caller-side queue limits still matter. Importing this dispatcher was part
of our real package import; executing its asynchronous infrastructure was
not part of the capacity-manager test.

The [v2 inference controller](https://github.com/areal-project/AReaL/blob/2fad2d0e308fe631e70e971b97188ad5c5cc03cb/areal/v2/inference_service/controller/controller.py)
creates the same original manager with itself as the version provider and
passes it into `WorkflowExecutor`. The
[trainer](https://github.com/areal-project/AReaL/blob/2fad2d0e308fe631e70e971b97188ad5c5cc03cb/areal/trainer/rl_trainer.py)
calls `on_version_recovered` after recovery. That callback offsets the
accepted count by `recovered_version * B`, keeping restart admission within
the intended window. The trainer's weight-update path publishes the new
version after updating weights; the counter therefore represents a training
commit rather than an arbitrary queue tick.

The upstream [asynchronous RL guide](https://github.com/areal-project/AReaL/blob/2fad2d0e308fe631e70e971b97188ad5c5cc03cb/docs/en/algorithms/async.md)
also pairs asynchronous generation with an off-policy loss and log-probability
recomputation. A capacity manager does not supply those learning signals.

## Observed execution

The probe used a normal Python import:

```python
from areal.infra.staleness_manager import StalenessManager
```

It executed the original `areal/__init__.py` and real dependency imports;
**92 original AReaL modules** loaded from the verified checkout. The result
records their source paths and individual hashes, including genuine upstream
namespace packages. There was no AST extraction, package-initializer bypass,
synthetic module, command-capturing replacement, or patched original source.

The probe uses an explicit integer version counter as a control-plane input.
The unchanged official tests use their author's `MockVersionProvider` integer
counter. Neither should be interpreted as an executed model or weight server.

| Event in the control-plane probe | Observed capacity |
| --- | ---: |
| Initial: concurrency 3, batch size 2, permitted lead 1 | 3 |
| Three tasks submitted | 0 |
| Two accepted and one rejected | 2 |
| Version counter advanced from 0 to 1 | 3 |
| Recovery at version 100, batch size 8, permitted lead 2, before callback | 824 |
| Same recovery state after original `on_version_recovered(100)` | 24 |

The recovery pair demonstrates a prevented submission burst under these
specified inputs; **824 to 24 is an admission-budget result, not a speedup**.

The unchanged `tests/test_staleness_manager.py` suite finished with **45
passed, 0 failed, 0 errors, and 0 skipped**. Coverage includes concurrent
counter operations, capacity reads, lifecycle transitions, synchronous
mode, negative capacity, large versions, and recovered-version accounting.
Pytest reported 4.39 seconds; the child process took 5.45 seconds including
startup, and the wrapper took 11.35 seconds including the probe and separate
test-process imports. These times describe this CPU test
run and are not a rollout or training benchmark.

The run used macOS arm64, CPython 3.12.1, and an observed dependency mix
recorded in the
[public measurement](measurements/official_areal_staleness_cpu_2026-09-27.json).
Missing dependencies were installed into an isolated ignored `runs` directory;
the host installation and this project's dependency list were not changed.
The existing host Torch 2.11.0, Transformers 5.5.4, and aiohttp 3.9.5 differ
from the upstream declared training ranges. Passing this pure counter/control
suite establishes that narrow execution result, not compatibility of the full
training stack or a reproduction of `uv.lock`.

## Concrete integration into our rollout pipeline

Our `trajectory_control.py` already separates actor generation, environment
interaction, and trusted reward verification. It has bounded queues and an
exact immutable-revision fence. The first reusable AReaL component is an
optional **admission gate before a new episode is opened**, with the
following integration contract:

1. Supply a trusted monotonic optimizer-update version mapped to the actual
   immutable policy revision. Advance it only after new weights commit. Do
   not increment it merely to unlock a full queue.
2. Keep the scheduler as the sole producer. Read positive original capacity
   and perform `on_rollout_submitted` in that serialized admission path;
   keep our existing bounded queues. Each job is enqueued and submitted once.
3. Call `on_rollout_accepted` once when a completed, trusted, usable training
   rollout is admitted. A correctly graded zero-reward example can be accepted.
   Call `on_rollout_rejected` once for discarded data or infrastructure failure.
   Pending verification remains pending rather than becoming a fake rejection
   or zero reward. Hardware leases and statistical rollout accounting remain
   separate lifecycles.
4. On real trainer recovery, restore both the actual version and the original
   manager's accepted-count offset before admitting work. Align batch size and
   callback units with logical rollout groups rather than counting every
   agent turn or tensor row as a separate sample.
5. Preserve our current exact revision fence. This numerical budget does not
   authorize mixed-revision action trajectories. Relaxing that fence would
   require verified per-token behavior log probabilities, per-token versions,
   and an off-policy objective with separately validated weight synchronization.

These hooks are a source-grounded integration plan, not a claim that the core
pipeline now runs AReaL or performs RL training. This change adds the executable
original-source probe and evidence only. The CPU result justifies implementing
the optional gate; the next performance experiment must exercise real
generation, environment execution, verification, and model updates together.

## Reproduce the source probe

Use a Python environment that can normally import the pinned AReaL package.
The authors document a non-CUDA development installation with
`uv sync --group dev`; the recorded local run above used the explicit observed
CPU dependency mix instead. No GPU is needed for the capacity-manager checks.

```bash
git clone https://github.com/areal-project/AReaL runs/official-areal
git -C runs/official-areal checkout 2fad2d0e308fe631e70e971b97188ad5c5cc03cb
python3 examples/official_areal/check_staleness.py \
  --source runs/official-areal \
  --output runs/official-areal-cpu-check \
  --upstream-tests
```

The output directory must be new and cannot contain or be contained by the
source checkout, including through canonical symlink paths. The probe validates the clean source pin,
uses normal package imports, runs the exact original official test file, and
writes JSON, the captured test log, and JUnit XML. A failed import or failed
test produces a failed result; it never substitutes a local implementation.
Imported bytecode writing and pytest cache writing are disabled.
