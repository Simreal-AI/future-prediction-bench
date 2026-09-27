# Genuine repository outcomes connected to Pilot-Commit allocation

On 2026-09-27, four offline Docker episodes on the pinned Boltons repository completed through the guarded coding producer and the same SQLite budget ledger. The two initial episodes produced genuine binary rewards `[0, 1]`; the unchanged author `select_prompts` retained the task, and two reserved commit episodes then produced `[0, 1]`. The final budget was **4 spent, 0 reserved, 0 remaining**, and retrying an already spent reservation made **zero provider calls**.

This closes the connection between repository verification, trusted reward receipts, allocation and actual dispatch. The actors in this control run replay fixed action files. It is an execution and provenance result, not a model rollout, learning or acceleration result.

The [full curated measurement](measurements/official_pilot_coding_outcomes_2026-09-27.json) includes every case's actual stdout hash and exit code, complete outcome/cost records, both plans, all budget snapshots, final SQLite rows, policy/source/artifact bindings and raw-file digests. The [original component review](PILOT_COMMIT_SOURCE_REVIEW.md) covers the earlier CPU selection/replay fixtures separately.

## What actually executed

The task starts from [Boltons 26.0.0 at commit `fb464991b718ca7bfabc14555c2947f25e7c79c9`](https://github.com/mahmoud/boltons/commit/fb464991b718ca7bfabc14555c2947f25e7c79c9), whose source distribution has SHA-256 `5566d6cfd5a1e873d8e8476496287a9f92979964611ad9a9cecb6b0ef29b1edd`. It repairs the already public [singularize defect in PR #418](https://github.com/mahmoud/boltons/pull/418). This is a known repair integration fixture, marked `is_fixture=true`; its published verifier cases are hidden from the actor's tool surface during execution but are not a secret evaluation set.

The baseline submits the unchanged repository. The repair control reads and replaces `boltons/strutils.py`, runs the fixed visible import check and submits. Both use the normal `RealWorldEnv` and `DockerCodingAdapter`; the producer constructs the adapter itself and accepts no model-provided pass flag or arbitrary grading callback. Expected results stay on the host. Each hidden case runs candidate code in a separate offline container with a read-only submitted workspace, dropped capabilities, bounded resources and no new privileges. Image pulls are disabled.

| Phase | Fixed actor | Hidden cases passed | Binary reward | Provider calls / actions | Episode wall (s) | Verifier component (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Pilot 0 | Unchanged baseline | 7 / 14 | 0 | 1 / 1 | 5.194853 | 2.344880 |
| Pilot 1 | Public repair | 14 / 14 | 1 | 4 / 4 | 2.803777 | 2.137085 |
| Commit 2 | Unchanged baseline | 7 / 14 | 0 | 1 / 1 | 2.550114 | 2.163496 |
| Commit 3 | Public repair | 14 / 14 | 1 | 4 / 4 | 3.184371 | 2.497613 |

All 56 hidden-case commands exited with code 0. The two zero rewards came from seven actual stdout mismatches each, at case indices `0, 2, 4, 6, 8, 10, 12`, rather than an infrastructure error. Rewards are rederived from **all 14 exact stdout SHA-256 comparisons plus exit codes**; a partial case set cannot earn a reward. The submitted workspace hashes were independently rechecked against the retained snapshot trees.

These wall times are observations from one sequential control run. They do not form a paired performance comparison. Costs preserve integer nanoseconds for reset, fixed-provider calls, actions, verification, cleanup and total episode wall time. The wall scope includes final read-only revision checks but excludes admission, claim and persistence. Budget units remain dispatched episodes; nanoseconds and scripted provider calls are not token or inference costs.

## Original selection and actual budget flow

The checker normally imports [`recipe.pc.utils.select_prompts`](https://github.com/databricks/pilot-commit/blob/6def20ea211fc936ed092a5a08624807c54df381/recipe/pc/utils.py), including the original `verl` initializer, from an unchanged external checkout at commit `6def20ea211fc936ed092a5a08624807c54df381`. No author function is extracted or replaced. Its 450 Python-file tree digest is `9ea69afcf931fb1a1c4a19573fc3150379632e7018ebf99885a3934ad93cd264`, unchanged and Git-clean before and after the run.

There was one actual original-selector call. Canonical receipt ordering supplied prompt indices `[0, 0]` and rewards `[1, 0]`, with lower threshold `0.125`, upper threshold `0.25` and exclusion threshold `1.0`. The output was `keep=[0]` and three empty remaining categories. The task mean was 0.5. The commit plan retained exactly `boltons-26-singularize-ss-v1`, referenced exactly the two initial verified receipts and reserved two new jobs under the configured floor and cap of two.

| Boundary | Total | Spent | Reserved | Remaining |
| --- | ---: | ---: | ---: | ---: |
| Initial pilots reserved, explicitly unselected | 4 | 0 | 2 | 2 |
| Both actual pilots completed | 4 | 2 | 0 | 2 |
| Original selection and commit reservation completed | 4 | 2 | 2 | 0 |
| Both actual commits completed | 4 | 4 | 0 | 0 |

The initial plan is labeled `unselected-coding-initial-pilots-v1`; it does not pretend that allocation has already happened. Every dispatch must successfully claim an existing reservation against the exact task and policy revision before environment reset, actor execution or provider calls. Each completed episode consumes one reserved unit. Selection verifies the already spent pilot receipts through the producer's persisted outcomes and matching ledger rows, then reserves new jobs on the **same** ledger without charging those pilots a second time. The unchanged planner core still owns transactional selection and reservation.

A final attempted redispatch of Pilot 0 returned `dispatched=false`, `receipt=null` and `provider_calls=0`. The final database has exactly four spent reservations, four matching one-unit receipts, four authoritative coding outcomes and two plans. Source control flow and tests establish the claim-before-execution ordering; the SQLite schema records final state, not a timestamped transition history.

## Trusted producer boundary

`future_prediction_bench.pilot_commit_coding.CodingOutcomeProducer` connects the existing environment and ledger. It binds the validated task, original grader source, seed tree, verifier tree, local image and declared policy revision; freezes its source before execution; and checks those bindings before dispatch, before planning and after execution. A provider receives detached visible observations and returns repository tool actions. A provider-supplied `reward`, `passed`, `verification` or `terminal_reward` is rejected.

The producer stores an outcome before completing its ledger receipt. A receipt becomes eligible only when the canonical stored outcome matches its exact execution, case contract and hashes **and** the original ledger contains the corresponding spent reservation/receipt with the same owner. Interrupted/provider/infrastructure/unknown failures remain charged and unresolved without a reward. The inherited verifier treats a candidate command timeout or output limit as a binary task failure only after cleanup of the owned container is confirmed; an infrastructure exception is not silently converted to zero. Neither limit condition occurred in this run.

`CodingPolicyBinding` hashes the controller, prompt and provider configuration. These are declared bindings, not attestation of remote model weights. The production interface allows a caller-injected generic inference provider; this checker uses only fixed scripted controls. Raw provider configuration and identity are not part of public evidence.

## Reproduction

Use a fresh Python process from the project root, a running Docker daemon, the pinned cached Boltons source archive and a clean external Pilot-Commit checkout at the commit above. Supply the isolated dependency directories described in [the example README](../examples/official_pilot_commit/README.md): PyVers 0.1.0, TensorDict 0.9.1 and Ray 2.58.0 with the recorded macOS ARM64 distribution hashes. These pins do not imply that all other imported host libraries are byte-pinned. The Python interpreter must support the normal original package import and its dependencies; the checker installs nothing.

The recorded local ARM64 utility image ID is `sha256:8e8fad2e920379c34538b76ed81165e0c7c896605972a851830392892f282fb6`. It must already be cached for this exact cohort. A local image ID is not a registry download URL. A different image or Linux dependency cohort needs a separately reviewed binding; do not present it as byte-identical reproduction.

Replace the placeholder input paths below with existing inputs. Use new, disjoint task/output directories. Source, dependency and checker output paths must have no symlink ancestors; output must have an owned parent. Generate the task immediately before the checker because the generator's action deadline is fixed approximately 31 minutes after issuance. Fresh timestamps change the task revision, IDs and raw-file digests; compare structural outcomes and verification bindings rather than expecting identical run IDs.

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

The checker writes `result.json`, `budget.sqlite3`, a frozen driver, a producer manifest/source snapshot and one `outcome.json` plus visible `trajectory.json` per graded episode. Its success requires genuine producer-verified terminal receipts, allocated commit execution, source/dependency/task preservation, final budget consistency and the duplicate-dispatch witness. Keep unsuccessful raw outputs for diagnosis. The 52 current owned tests comprise 42 planner/preparation tests and 10 coding-producer boundary tests; the latter mock Docker transport and do not establish actual execution. The four-episode run above supplies the actual Docker integration evidence separately.

## Evidence audit and scope

An independent standard-library audit opened SQLite read-only and did not import the producer or execute Docker. It recomputed every hidden-case result, every visible event digest and action sequence, all four submitted workspace trees, task/policy bindings, canonical receipt/execution/outcome/plan/reservation identities, original source/dependency digests and joins between authoritative outcomes and spent rows. SQLite integrity passed, and all inspected raw artifact bytes remained unchanged.

The raw result file digest is `461415d9b37e867e50c2c26c5929170c55bcb8f192b9138c7191470cb1df62ab`; the final raw SQLite digest is `f388a15bec7441bf8b60fa2cfb5c95209957b811ea5019014304437ebac18d2e`. The curated record distinguishes exact file hashes from canonical JSON hashes. It publishes complete hidden-case stdout hashes and exit codes; this verifier did not persist raw hidden stdout text. Visible trajectory summaries preserve every event's digest and metadata plus payload hashes; large repository text/action bodies remain in local raw trajectories. Public evidence contains no absolute host paths or private provider identity.

This is one solved public task with fixed actors and one selection decision. It establishes the authentic outcome-to-dispatch path and budget conservation. It does not establish multi-task allocation savings, unseen repair capability, tokenized trainer-ready trajectories, model sampling, policy gradients, optimizer execution, off-policy replay effectiveness or GPU/training speed. All exported text trajectories retain `trainer_ready=false`.
