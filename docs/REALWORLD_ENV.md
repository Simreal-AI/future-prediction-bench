# Real-world RL environment contract

The third track gives an agent a bounded, stateful task and verifies its final work with a separate trusted adapter. `DockerCodingAdapter` lets the agent edit an isolated repository workspace and run a fixed visible check; separate containers execute hidden command cases. The experimental `MicroVMCodingAdapter` exposes the same bounded actions inside an offline ARM64 Linux VM, then restores a full QEMU snapshot for every hidden case. In both backends, the trusted host compares outputs with expected results. This track has its own task, action, and reward contract; it does not use forecast probabilities or Brier scoring.

The core lives in `future_prediction_bench.realworld`. It has no code executor or network client. `RealWorldEnv` accepts a task specification, one trusted adapter instance, and optional clocks. The adapter implements:

```python
reset(task, *, now) -> dict  # Initial policy-visible observation.
step(action, *, now) -> {"observation": dict, "terminated": bool}
verify(*, now) -> {"status": "pending" | "resolved" | "void", ...}
get_state() -> dict  # Trusted diagnostics, never a policy observation.
```

The host calls `env.reset(policy_id)`, then `env.step(action)` until the adapter submits or the action budget expires. Every valid adapter result is an observation with no reward. A terminal step moves the episode to `pending`. Only the trusted host calls `env.verify()`; a resolved result becomes `graded`, a void result remains unscored, and an unavailable or failed verifier remains `pending`. The core records timestamped, hashed policy actions, visible observations, and private verifier results with separate visibility and loss-mask fields. `env.get_state()` includes private audit and adapter state and must not be passed to the agent.

## Frozen task specification

`validate_task()` makes a detached copy and stamps it with `task_sha256` and `reward_contract_sha256`. A task must contain:

| Field | Purpose |
| --- | --- |
| `schema_version` | `realworld-0.1` |
| `task_id`, `event_id`, `cluster_id`, `split` | Immutable identity and train/dev/test assignment |
| `prompt`, `tool_manifest` | Policy-visible task and allowed action names, descriptions, optional parameter schemas |
| `issued_at`, `action_deadline` | Action window; the deadline is exclusive |
| `outcome_not_before`, `verify_after` | Earliest legitimate outcome and first verifier attempt |
| `reward_contract` | ID, description, and finite `min_reward`/`max_reward` within `[-1, 1]` |
| `budgets` | `max_actions`, `max_wall_seconds`, optional verification cooldown and attempt limit |
| `is_fixture` | Explicitly labels controlled test tasks |

The task may also include `adapter_id`, `adapter_version`, and private `metadata`. The agent's reset observation contains only an allowlist and excludes metadata. The environment checks action names against the frozen manifest; the adapter validates each action's arguments and effects. The action window and verification window are independent: a coding task can become verifiable immediately after submission, even if its action deadline is later. `verify()` cannot call the adapter before terminal submission or before `verify_after`. A resolved response must include a reward within the frozen bounds, nonempty evidence, and an `available_at` timestamp no earlier than submission and no later than verification.

`RealWorldEnv` is currently an in-memory host boundary. The caller must persist its state and audit record and schedule pending verification. `RealWorldTaskRegistry` persists frozen task definitions in SQLite, rejects edits to a registered task, and prevents matching event or cluster IDs from crossing train/dev/test splits. The scripted real-task CLI requires this registry. Repository-family and near-duplicate detection beyond declared IDs still need independent dataset review. A task digest is an integrity checksum, not proof that an external source or verifier is correct.

## Docker coding task

`DockerCodingAdapter` is constructed with an operator-controlled seed directory, hidden verifier directory, locally available image, and output directory. These directories must be disjoint. One adapter instance serves one episode. It exposes `list_files`, `read_file`, `write_file`, `run_visible_checks`, and `submit`. The agent has no direct shell action: the visible check is a fixed operator-supplied command. Agent-written code invoked by that check can execute inside the isolated container, so the seed and image must contain no secrets or host Git metadata. Reads do not create workspace snapshots. Writes and visible checks checkpoint changed workspace content, and submission freezes a workspace snapshot.

The actor container starts from a locally resolved image SHA-256 ID with network disabled, a read-only root filesystem, limited CPU/memory/processes, and an isolated writable workspace mount. The hidden verifier is never mounted in any candidate container. Before task registration, the runner binds the seed workspace, verifier, image ID, and visible command hashes into the frozen task. Reset rechecks the binding; grading rechecks the verifier and submitted workspace. After submission, each operator-defined command case runs in a new container with the frozen workspace mounted read-only. The host compares its exact exit code and stdout with expected results kept in `verifier/verify.json` on the trusted host. The adapter returns `0` or `1` and hashes the submitted workspace, verifier, image, and per-case outputs. A candidate timeout or output-limit violation is a failed case when its container can be cleaned up; Docker infrastructure failure remains pending. These boundaries need validation on each deployment host; this prototype does not claim a general sandbox security guarantee.

`verify.json` uses the following form. Each `argv` is operator-selected and the expected result never enters the candidate container. Independent cases may run in parallel through `--verifier-workers 1..8`; each retains its own container:

```json
{"kind":"command_cases_v1","cases":[{"argv":["python3","-B","-c","from math_utils import add; print(add(2, 3))"],"expected_stdout":"5\n","expected_returncode":0}]}
```

For a fully offline end-to-end check, run `python -m future_prediction_bench realworld-code-smoke --image sha256:<local-image-id> --output runs/code-smoke`. To run an externally supplied task with scripted actions, use `realworld-code-run --task task.json --seed repo-seed --verifier trusted-verifier --image sha256:<local-image-id> --actions actions.jsonl --visible-check '["python3","-B","-m","pytest"]' --registry runs/realworld-registry.sqlite --output runs/external-code-task`. The task JSON must satisfy the frozen schema above; action JSONL contains one action object per line. This entry point intentionally does not connect a model. An online policy can call `RealWorldEnv` directly through the same action protocol.

The host must call `DockerCodingAdapter.close()` in a `finally` block, including after setup errors or missed action budgets, to stop any remaining actor container. A Docker timeout interrupts the episode without assigning a policy reward; an operator must resolve that infrastructure condition rather than silently count it as policy failure.

A scripted fixture can test the complete lifecycle in seconds. Its `is_fixture=true` label must remain attached to exports, and passing its hidden checks is not evidence of improved real-world task performance. Live repository tasks require pinned source revisions, independent verifier versions, licensing review, and train/dev/test separation by repository lineage or task family. A generated task or synthetic verifier should carry its own provenance instead of being described as a real external result. Future-event forecasting remains a separate track: no amount of sandbox acceleration reveals a physical outcome before it occurs.

## Full-state microVM adapter

`MicroVMCodingAdapter` is a second implementation of the same `RealWorldEnv` interface. Its operator-prepared ARM64 Linux image holds Python and the source workspace on an ext4 disk. A separate matching Alpine module image is attached read-only. The host launches QEMU/HVF with no NIC or host-directory mount, identifies the disks by filesystem magic, and only exposes bounded file operations and one fixed Python check to the policy. At `submit`, the host saves a full QEMU CPU/RAM/device/disk snapshot. For each hidden `command_cases_v1` case, it restores that snapshot, runs the fixed Python command in the guest, and compares exact stdout and return code with host-only expectations. A VM error remains pending rather than becoming a zero reward. The interface does not permit a policy-selected shell command.

The VM adapter also implements the trusted branch interface. `env.create_branch_checkpoint()` saves a full-state QEMU snapshot with a private tag. The host makes a child adapter with `parent_adapter.branch_adapter(new_child_disk_path)`, then calls `env.fork_from_checkpoint(ref, child_adapter, branch_id=...)`. Each child receives a separate qcow2 and QEMU process restored from the tag; its policy-visible opening omits trusted lineage metadata. A [scripted integration check](../examples/realworld_boltons26/check_microvm_branch_env.py) forks two children after a parent write made *after* the checkpoint, confirms independent RAM and ext4 markers, and grades the repaired/baseline children `[1.0, 0.0]` over 14 cases each. The checkpoint's recorded qcow2 digest is an audit value at creation time, not an equality check against the live parent's later qcow2 bytes: QEMU may change container metadata during stop/continue. The runtime verifies the snapshot tag, pinned artifacts, byte-for-byte child disk clone, and successful `loadvm` before exposing a child. See the [microVM guide](MICROVM_ENV.md) for the report and limits.

The source package includes a pinned Boltons image builder, full-VM benchmark, and `realworld-code-microvm` CLI. On this development host, the actual CLI graded the unmodified repository `0.0` and the public scripted repair `1.0` over 14 cases. A three-pair cold/warm experiment measured 8.578 versus 3.324 seconds median graded environment episode time and confirmed guest process/RAM restoration; see the [microVM guide](MICROVM_ENV.md) for the setup, full report, and amortized comparison. This experiment does not contain model generation, token log probabilities, a trainer, or parameter updates. It is an experimental local VM backend, not an audited multi-tenant isolation service.

### Opt-in namespaced hidden-case verifier

The CLI can run an explicitly declared stateless verifier instead of a full VM restore for every hidden case. This is available only with `--stateless-verifier-contract`; omitting that flag preserves the full-VM default. The path expects the exact operator-authored `task.json`, its sibling `verifier/verify.json`, and a task-specific contract. For the validated Boltons fixture, the invocation is:

```sh
python3 -m future_prediction_bench realworld-code-microvm \
  --task runs/boltons-task/task.json \
  --verifier runs/boltons-task/verifier \
  --assets runs/microvm-assets \
  --actions runs/boltons-task/actions.solution.jsonl \
  --visible-check '["python3","-B","-c","import boltons.strutils"]' \
  --stateless-verifier-contract examples/realworld_boltons26/stateless_contract.json \
  --output runs/boltons-stateless-episode
```

The frozen task binding includes normalized task content plus the verifier, contract, and guest-helper SHA-256 digests. The host rechecks them at reset, submission, and grading. The contract declares independent Python cases, unprivileged execution, and a quiescent submitted state. After policy actions, the host uploads the isolated root helper and case **source code only** to guest-private files, then snapshots the VM. Expected stdout and return codes remain on the host. Before grading, the host restores that submitted snapshot once and checks its process census, then starts the one-call hidden-case batch. Each case in that batch runs in a fresh guest mount and PID namespace with a private overlay workspace; a guest or verification failure leaves reward pending rather than assigning zero. See the [stateless verifier experiment](STATELESS_VERIFIER.md) for its precise filesystem and process limits.

This mode currently requires the pinned guest image's two BusyBox baseline processes, including PID 1. The host compares guest user-space PID, start tick, and executable identities with the exact baseline at reset, before and after helper preparation, and after restoring the submitted snapshot. A changed process census before submission interrupts the episode; a process captured in the submitted snapshot or appearing during grading leaves reward pending. The baseline is trusted only because the image is operator-built. A daemon already present in that baseline or a process that escapes the scanned view can defeat this check. Candidate-visible Python and full-VM hidden-case Python now run as UID/GID 65534 under a versioned task binding, while the trusted wrapper and guest control still run as root; see the [real-guest identity probe](GUEST_IDENTITY_INTEGRITY.md). This is not a general isolation or arbitrary untrusted SWE safety guarantee. Operators must validate the contract and task behavior before opting in. Full-VM per-case grading remains the general path for tasks with cross-case or live-process dependencies.

In an [alternating three-pair full-episode comparison](measurements/microvm_stateless_episodes_v0.6.0.json) on the same pinned scripted Boltons repair, both paths graded all 14 cases correctly in every run. Median hidden verification wall time fell from 1.946 to 0.451 seconds, while median action time rose from 0.851 to 1.339 seconds because of helper upload and extra checks. Median complete graded environment episode time fell from 8.371 to 7.267 seconds (1.15×). These runs include VM startup and environment actions, but no model inference or optimizer training. A separate [grader-only measurement](measurements/stateless_verifier_boltons_v0.6.0.json) isolates the 14-case verification path; its 5.43× ratio must not be presented as full-episode or training speed.

## Shared-prefix branches and bounded episode scheduling

The trusted host can call `env.create_branch_checkpoint()` at an active action boundary, then `env.fork_from_checkpoint(ref, new_adapter, branch_id=...)` for independent suffixes. The policy cannot request or inspect a checkpoint. For the **Docker** adapter, the actor stops before the workspace is hashed and frozen, and each branch starts a fresh container. On macOS/APFS it attempts `clonefile` copy-on-write for regular files; on other filesystems it makes real copies. These filesystem snapshots are content-addressed, verified before reuse, and never hard-linked to a mutable branch. The **VM** adapter instead restores full guest state into a separate QEMU process and disk as described above. Both kinds of branch retain the frozen task, policy ID, prefix event hashes, action count, original wall-clock budget, and absolute deadline. Each branch gets a unique episode ID, verifier run, and final reward. A failed snapshot or restore is an infrastructure error, never a policy reward of zero.

The Docker branch is **filesystem-only** restore into a new sandbox: it does not preserve process memory, open sockets, or `/tmp`. Its fixed visible check destroys and restarts the actor container before a workspace checkpoint, so a background process cannot become hidden branch state. The VM branch preserves CPU, RAM, devices, and disk at its snapshot boundary. Both adapters need `close()` in a `finally` block, and generated disks and run artifacts need an operator retention policy.

`run_coding_branches()` executes one scripted prefix and a bounded set of suffixes. `run_coding_batch()` executes independent complete episodes with at most `max_workers` active and at most twice that many submitted jobs; verification remains synchronous inside each worker. `run_disaggregated_coding_batch()` instead uses separate same-host interaction and verification worker pools. Once `submit` stops the interactive container and freezes the workspace, the episode enters a bounded verifier queue and an interaction worker can start another task. Both queues have explicit capacity limits and block producers when full. The verifier queue optionally orders waiting jobs by trusted operator priority, with FIFO handoff order for equal priorities; it cannot interrupt a running verifier. The [straggler scheduling experiment](VERIFIER_SCHEDULING.md) measures the latency effect and starvation limit. The report records queue wait, execution, verification, and handoff time, queue high-water marks, and graded episodes per second. These runners keep per-episode outputs separate and infrastructure errors unscored.

This is a local two-stage **environment** pipeline. It accepts operator-supplied actions and has no model inference, token log probabilities, GPU trainer, weight synchronization, remote sandbox service, or process-level checkpointing. A pending result after one verifier attempt remains ungraded; a later retry would need a separate durable host scheduler. Branches sharing one prefix are correlated samples from one task and must not be counted as independent tasks in evaluation or policy-gradient statistics. Behavior-policy token log probabilities and suffix-specific masks are still absent.

The CLI accepts operator-authored JSON manifests:

```bash
python -m future_prediction_bench realworld-code-batch \
  --jobs jobs.json --max-workers 4 --output runs/code-batch
python -m future_prediction_bench realworld-code-pipeline \
  --jobs jobs.json --actor-workers 2 --verification-workers 2 \
  --actor-queue-capacity 2 --verifier-queue-capacity 2 \
  --output runs/code-pipeline
python -m future_prediction_bench realworld-code-branches \
  --task task.json --seed repo-seed --verifier trusted-verifier \
  --image sha256:<local-image-id> --visible-check '["python3","-B","-m","pytest"]' \
  --plan branches.json --policy-id scripted-policy --max-workers 4 \
  --output runs/code-branches
```

`jobs.json` is a JSON array of objects with `task`, `seed_dir`, `verifier_dir`, `image`, `actions`, and an explicit `policy_id`; `task` may be a JSON path and `actions` a JSONL path relative to the manifest. Optional `verifier_workers` controls parallel hidden cases **within** one episode; `--verification-workers` controls concurrent episode verifiers in the pipeline. `branches.json` has `prefix_actions` (an array or JSONL path) and `branches`, an array of `{ "branch_id": "candidate-1", "actions": [...] }`. A persistent `--registry` is required for non-fixture tasks. The [Boltons example](../examples/realworld_boltons26/README.md) provides a fully offline controlled comparison.

## Training and speed measurement

`export_trajectory()` returns only graded train episodes. It includes the frozen task/reward hashes, policy ID, submission and reward-availability times, final reward, and policy-visible action/observation audit. Hidden verifier events are excluded. The record explicitly sets `trainer_ready=false`: it has no token IDs, exact chat-template masks, behavior-policy token log probabilities, optimizer, or stale-policy correction. Tool observations have loss mask `0`; only agent actions have loss mask `1` at this text-audit level.

For sibling branches, the trusted host can call [`prepare_sibling_advantages()`](BRANCH_ADVANTAGES.md) with the frozen planned cohort size, exact policy ID, and preparation cutoff. It checks shared checkpoint/task/prefix lineage and computes leave-one-out reward advantages for **post-checkpoint policy actions only**. It rejects incomplete or prematurely resolved groups and still returns `trainer_ready=false`. Shared-prefix actions are context, not separate independent samples or automatic recipients of the suffix-local advantage.

The core measures reset, action, and verifier wall time, action count, verifier attempts, pending attempts, and next verification time. The coding adapter additionally measures Docker startup, file reads/writes, visible checks, checkpoints, and verifier execution. A training-speed claim should report valid graded episodes per hour, p50/p95 stage latency, reward wait, model generation time, tokens, optimizer time, and total cost on the same task mix and compute budget. Compare serial and concurrent execution before adding snapshot or caching machinery. Fast fixture verification tests infrastructure; faster RL training requires an actual trainable policy and measured updates on independent held-out real tasks.
