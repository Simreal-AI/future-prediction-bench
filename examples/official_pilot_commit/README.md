# Original Pilot-Commit components and product allocation

The probes execute the authors' unchanged [Pilot-Commit implementation](https://github.com/databricks/pilot-commit/tree/6def20ea211fc936ed092a5a08624807c54df381) through normal imports, including the original `verl` initializer. The product probe then calls the original `select_prompts` through an explicit adapter into our standard-library SQLite budget planner. Original source is external; this directory contains our probes, not a vendored trainer.

The recorded CPU component run passed 18 unchanged author data-protocol tests, eight component fixture groups, 240 exhaustive binary subset checks, and six original-selector/product-planner calls. Separately, 42 owned tests cover the product budget, provenance, concurrent reservation/claim, and input preparation contracts. See [the source review](../../docs/PILOT_COMMIT_SOURCE_REVIEW.md) and [curated component evidence](../../docs/measurements/official_pilot_commit_cpu_2026-09-27.json).

A separate genuine repository integration now connects real Docker verification to selection and dispatch. Four fixed-actor Boltons episodes produced rewards `[0, 1, 0, 1]`; the original selector used the two pilot receipts to reserve two actual commit jobs. The final shared ledger records four spent units and no outstanding reservations. An attempted duplicate dispatch made zero provider calls. See [the coding outcome guide](../../docs/PILOT_COMMIT_CODING_OUTCOMES.md) and [complete case/cost/budget evidence](../../docs/measurements/official_pilot_coding_outcomes_2026-09-27.json).

## Reproduce the recorded fixture

Use a fresh Python process from the project root and an external clean checkout at commit `6def20ea211fc936ed092a5a08624807c54df381`. Supply isolated genuine dependency directories containing the recorded TensorDict 0.9.1, PyVers 0.1.0 and Ray 2.58.0 distributions. The dependency byte hashes are intentionally tied to the recorded macOS ARM64 fixture; Linux wheels require a separately reviewed pin set. Torch 2.11.0, NumPy 2.2.6, Transformers 5.5.4 and pytest 9.0.2 were present in the recorded runtime; their versions are reported, but their distributions are not byte-pinned by these probes.

Replace the example paths with distinct existing source/dependency directories and a new, nonexistent output directory. Paths must have no symlink ancestors and output must have an owned parent. The checker rejects overlaps and performs source/dependency verification before importing author code or creating output. It does not install dependencies.

```sh
python3 -m examples.official_pilot_commit.check_components \
  --source /path/to/original-pilot-commit \
  --dependency /path/to/pilot-component-dependencies \
  --dependency /path/to/additional-isolated-dependencies \
  --output /path/to/new-component-output \
  --official-tests

python3 -m examples.official_pilot_commit.check_planner \
  --source /path/to/original-pilot-commit \
  --dependency /path/to/pilot-component-dependencies \
  --dependency /path/to/additional-isolated-dependencies \
  --output /path/to/new-planner-output \
  --official-tests

python3 -m unittest discover -s tests -p test_pilot_commit.py -q
```

`--official-tests` enables the unchanged author CPU test file; omitting it does not establish the reported 18-test result. Each probe writes `result.json` with actual task/row IDs, alignment, boundary observations and hashes. The planner probe also writes a real SQLite ledger. Its binary outcomes and verifier registry are explicitly owned fixtures. Full local logs can contain paths; the curated public record replaces only command paths and includes the original raw-record digest.

## Product contract

`future_prediction_bench.pilot_commit.plan_pilot_commit` accepts an explicit selector callback and caller-owned trusted evidence verifier. Accepted receipt batches charge all spent pilot statuses; an over-budget batch is rejected atomically, so callers must control pilot spending before submission. The planner admits only current task/policy revisions with verified integer binary outcomes, validates the original four category semantics, subtracts independent exclusions from `keep`, and reserves bounded floor/cap allocations transactionally. Consumers must successfully claim a reservation with current revisions before dispatch; repeated claims do not authorize dispatch twice. All consumers sharing a budget must use the same SQLite file and epoch.

This is an allocation control plane. The CPU component fixture does not generate tokens, execute software tasks, reuse optimizer data, update a policy, or measure model rollout/training speed. It must not treat an unresolved forecast probability or continuous forecast score as a binary outcome. The original replay buffer is studied and exercised here, but is not exposed as a guarded product replay API; its documented boundary behavior remains visible in the evidence.

## Reproduce genuine coding outcomes

`CodingOutcomeProducer` constructs the existing `DockerCodingAdapter` and `RealWorldEnv`, claims each reservation before actor/provider work, and rederives rewards from exact host-owned cases. Planning checks matching persisted outcomes and spent reservations on the same SQLite ledger; already spent pilots are not charged twice. The generic provider returns visible repository actions, never authoritative rewards. Provider, infrastructure and unknown failures remain charged and unresolved; confirmed candidate command limits follow the inherited verifier's binary failure contract. The recorded run had no such limits or infrastructure failures.

Supply the same external author checkout and isolated dependency cohort above, a running Docker daemon, a pinned cached Boltons 26.0.0 source archive, and a locally cached Python utility image. The exact recorded ARM64 image ID is `sha256:8e8fad2e920379c34538b76ed81165e0c7c896605972a851830392892f282fb6`; it is a local image identity, not a downloadable registry address. A different image/platform requires a separately reviewed cohort. No dependencies are installed or images pulled during the checker. Create fresh task timestamps immediately before execution and use new disjoint outputs.

```sh
python3 -B -m examples.realworld_boltons26.make_task \
  --output runs/new-pilot-coding-task \
  --sdist /path/to/pinned/boltons-26.0.0.tar.gz

python3 -B -m examples.official_pilot_commit.check_coding_outcomes \
  --task-dir runs/new-pilot-coding-task \
  --image sha256:8e8fad2e920379c34538b76ed81165e0c7c896605972a851830392892f282fb6 \
  --source /path/to/original-pilot-commit \
  --dependency /path/to/pilot-component-dependencies \
  --dependency /path/to/additional-isolated-dependencies \
  --output runs/new-pilot-coding-outcomes

python3 -B -m unittest discover -s tests -p 'test_pilot_commit*.py' -v
```

The checker retains `result.json`, `budget.sqlite3`, frozen driver/producer source, per-episode `outcome.json` and visible `trajectory.json`. All 56 hidden-case commands in the recorded run exited 0; unchanged episodes each failed seven stdout comparisons and repairs passed all 14 cases. Exact nanosecond costs are preserved as observed environment costs, without an acceleration comparison. The 10 added producer tests mock Docker transport; their results are separate from the four actual Docker episodes and combine with the 42 existing tests for 52 owned tests.

The repaired defect and action files are public integration controls. No model inference, optimizer update, secret held-out evaluation or learning/speed improvement is claimed. Policy hashes bind declared controller/prompt/configuration rather than attested model weights; provider identity remains outside public evidence. Text trajectories are explicitly `trainer_ready=false`. The [coding outcome guide](../../docs/PILOT_COMMIT_CODING_OUTCOMES.md) details source pins, read-only audit, failure semantics and portability limits.
