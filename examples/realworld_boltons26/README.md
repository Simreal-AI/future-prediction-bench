# Pinned public-repository repair integration task

This example starts from [Boltons 26.0.0](https://github.com/mahmoud/boltons/commit/fb464991b718ca7bfabc14555c2947f25e7c79c9), commit `fb464991b718ca7bfabc14555c2947f25e7c79c9`. The official [PyPI source distribution](https://pypi.org/project/boltons/26.0.0/) has SHA-256 `5566d6cfd5a1e873d8e8476496287a9f92979964611ad9a9cecb6b0ef29b1edd`; the upstream repository uses the [BSD-3-Clause license](https://github.com/mahmoud/boltons/blob/master/LICENSE). The source archive is downloaded or supplied locally, checked against that digest, and unpacked under ignored `runs/`. It is **not** vendored into this repository.

The task reproduces a real upstream bug: `singularize("glass")` returns `"glas"` at the pinned revision. [Upstream PR #418](https://github.com/mahmoud/boltons/pull/418) describes the fix. Because the repair is already public, this task is marked `is_fixture=true` and is only an integration check. Its publicly generated cases are hidden from the agent's tool surface during a run, but they are not a secret held-out benchmark.

From the repository root, create a task with live timestamps. The default builder fetches only the pinned PyPI source archive over HTTPS. For a fully offline setup, supply an already downloaded `--sdist` path:

```bash
python3 examples/realworld_boltons26/make_task.py \
  --output runs/boltons-task
```

The output contains `seed/`, trusted `verifier/verify.json`, `task.json`, and two scripted action files. With Docker running and a locally cached Python 3 image, grade the untouched source and the scripted repair:

```bash
python3 -m future_prediction_bench realworld-code-run \
  --task runs/boltons-task/task.json --seed runs/boltons-task/seed \
  --verifier runs/boltons-task/verifier --image sha256:<local-image-id> \
  --actions runs/boltons-task/actions.baseline.jsonl \
  --visible-check '["python3","-B","-c","from boltons.strutils import singularize"]' \
  --registry runs/boltons-task/registry.sqlite --output runs/boltons-baseline

python3 -m future_prediction_bench realworld-code-run \
  --task runs/boltons-task/task.json --seed runs/boltons-task/seed \
  --verifier runs/boltons-task/verifier --image sha256:<local-image-id> \
  --actions runs/boltons-task/actions.solution.jsonl \
  --visible-check '["python3","-B","-c","from boltons.strutils import singularize"]' \
  --registry runs/boltons-task/registry.sqlite --output runs/boltons-solution
```

The first run should receive reward `0` and the second `1`. A real policy can produce the same JSON actions instead of replaying these files. A credible learning experiment needs additional independent repositories, verifier review, no solution leakage in the evaluation cohort, and an actual trainable-policy optimizer.

To compare one versus four independent hidden-case containers on the same host and task, run `python3 examples/realworld_boltons26/benchmark_verifier.py --task-root runs/boltons-task --image sha256:<local-image-id> --output runs/boltons-verifier-bench`. It writes individual reports and a median summary. This isolates verifier work only; it does not measure model generation, gradient updates, or live event settlement.

To compare shared-prefix filesystem branches with full replay, create a **fresh** task because its action deadline is fixed at generation, then run:

```bash
python3 -m examples.realworld_boltons26.benchmark_branching \
  --task-dir runs/boltons-task --image sha256:<local-image-id> \
  --output runs/boltons-branch-bench --repetitions 2 --branch-count 4
```

This script checks that repaired and unrepaired suffixes receive the same rewards in all four conditions: cold serial, shared-prefix serial, cold parallel, and shared-prefix parallel. It fixes one verifier worker per episode so the comparison does not mix in hidden-case worker-count changes. On one macOS/APFS host, the two-repetition medians were 14.742, 14.197, 8.409, and 8.559 seconds respectively. The short prefix saved little; ordinary concurrency provided most of the throughput gain. These are small environment measurements on a solved fixture, not a result for agent learning or process-level checkpoint systems.

For a **full CPU/RAM/disk checkpoint** in a real ARM64 Linux VM, use [the microVM guide](../../docs/MICROVM_ENV.md). `prepare_microvm.py` builds the pinned guest from this seed and a locally cached arm64 Python 3.12 image; `realworld-code-microvm` runs the same action protocol with host-private grading; `microvm_benchmark.py` compares complete cold episodes with warm full-state restores. The generated guest images and reports remain outside the source archive under ignored `runs/`.

To compare serial complete episodes with separate, bounded interaction and verification queues, generate a **fresh** task within its 30-minute action window and run:

```bash
python3 -m examples.realworld_boltons26.benchmark_pipeline \
  --task-dir runs/boltons-task --image sha256:<local-image-id> \
  --output runs/boltons-pipeline-bench --repetitions 2 --episode-count 3
```

The script alternates condition order and checks rewards `[1, 0, 1]` from scripted repaired, untouched, and repaired episodes. It fixes one interaction worker, one episode verifier worker, one hidden-case worker per episode, and one queued verifier slot. On one host, two-repetition median batch times were **9.828 s serial** and **8.156 s pipelined**, a **1.205×** ratio in graded environment episode throughput. This is stage overlap on a public solved fixture; it does not measure model inference, weight updates, training wall time, or held-out task performance. Individual batch reports and `benchmark.json` retain the timings and rewards.
