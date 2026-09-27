# Pilot-Commit: original execution and guarded product integration

Date: 2026-09-27. The completed integration is an allocation control plane backed by a persistent SQLite budget. An explicit adapter actually calls the authors' unchanged prompt selector. The core planner and ledger are our code; the author trainer, replay buffer, tokenizer and policy optimizer are not presented as our implementations.

## Source and preservation

The [Pilot-Commit paper](https://arxiv.org/abs/2605.26606) links the [author implementation](https://github.com/databricks/pilot-commit). We executed commit [`6def20ea211fc936ed092a5a08624807c54df381`](https://github.com/databricks/pilot-commit/tree/6def20ea211fc936ed092a5a08624807c54df381), under Apache-2.0 with its NOTICE and original third-party notices. The external checkout remained Git-clean before components, after components and after product integration. We do not redistribute the external source in this example.

Normal package imports executed the original `verl` initializer and 11 original Python modules. Real `DataProto`, TensorDict, NumPy arrays and CPU tensors were used. No function was extracted with AST, package initializer bypassed, source patched, or upstream module replaced with a stub. A canonical digest of all 450 original Python files stayed `9ea69afcf931fb1a1c4a19573fc3150379632e7018ebf99885a3934ad93cd264`. The digest processes sorted relative paths, each path followed by NUL and the raw SHA-256 digest of its file bytes.

| Original source | SHA-256 | Exercised behavior |
| --- | --- | --- |
| [`recipe/pc/utils.py`](https://github.com/databricks/pilot-commit/blob/6def20ea211fc936ed092a5a08624807c54df381/recipe/pc/utils.py) | `67d28124bdc5162e2137345402cb20e93547825b442eadbd8c745bb10dbea2d7` | `select_prompts`, original prompt extraction and UID reduction |
| [`recipe/pc/replay_buffer.py`](https://github.com/databricks/pilot-commit/blob/6def20ea211fc936ed092a5a08624807c54df381/recipe/pc/replay_buffer.py) | `b70b368a1c46213419b9672865bc599a6c5cbf2d3d0ffdcb2598673dd65c5066` | add/sample/pop/flush, variance selection and strict step-age boundaries |
| [`verl/protocol.py`](https://github.com/databricks/pilot-commit/blob/6def20ea211fc936ed092a5a08624807c54df381/verl/protocol.py) | `eb9569fb199297ce8262e0869621dc9bb282a4087fb663c1f80e0a12026e480a` | Genuine DataProto batch and token/metadata alignment |
| [`tests/test_protocol_on_cpu.py`](https://github.com/databricks/pilot-commit/blob/6def20ea211fc936ed092a5a08624807c54df381/tests/test_protocol_on_cpu.py) | `b246081693321bc3f7ecf66a20790108fc73fcf8543f304631d2c2496324bbae` | 18 unchanged author CPU tests, zero failures/errors/skips |

Three genuine isolated dependency distributions were checked by every RECORD-listed file except `.pyc`, using the same relative-path/digest encoding. The imports must originate in the declared dependency directories, and these pins are checked again after integration.

| Distribution | Version | Files | Recorded macOS ARM64 distribution digest |
| --- | --- | ---: | --- |
| PyVers | 0.1.0 | 9 | `53acaacdd0fee27e102753fe7d41f58452421339e59b71d0cfd275a0b1a65e74` |
| TensorDict | 0.9.1 | 47 | `43cbe1005a2efe7f63d21863cdd8e4e4a711a6bdb9b00d4ec4bfe47311b77938` |
| Ray | 2.58.0 | 2767 | `308f3875c8ef7e5f5b885c158183ed66d6e76ed49a5b5cdc1e737777e6657165` |

The recorded runtime used Python 3.12.1, Torch 2.11.0, NumPy 2.2.6, Transformers 5.5.4 and pytest 9.0.2. The latter host distributions are version-recorded, not fully byte-pinned. A fresh Linux or other-platform fixture needs reviewed dependency hashes; passing this fixture is not a cross-platform trainer compatibility claim.

## What actually passed

The [curated public record](measurements/official_pilot_commit_cpu_2026-09-27.json) preserves all substantive component and integration fields from the successful fresh process. It includes the original private raw-record SHA-256, `b312009934cef86dacb5fcf852c59086821274f75168643fc142f7f1b371e1ab`, and replaces only command paths. Local JUnit and complete logs remain in ignored run storage and are not packaged with hostnames or private paths.

The original component execution passed eight fixture groups: inclusive selection boundaries, empty selection, variance-ranked real batch and step provenance, strict step-age eviction, pop ordering and alignment, empty replay, original prompt extraction, and binary maximum-variance subset selection. The last group checked all 240 small binary configurations against an independent exhaustive combinatorial oracle. All 18 author data-protocol CPU tests passed without skips.

The product execution made six actual calls to original `select_prompts`: first planning, an identical retry through a reopened ledger, a different overlapping request with no remaining budget, an exclusion-overlap case, a small floor/cap case and an all-past-costs case. A deny-all verifier case made no selector call. Every callback input and actual returned category ID is recorded.

In the primary fixture, four tasks supplied eight explicitly owned resolved binary pilot outcomes: `always-fails` = 00, `mixed-a` = 01, `mixed-b` = 10 and `always-passes` = 11. With total budget 16, the unchanged selector retained exactly `mixed-a` and `mixed-b`; the product reserved four further units for each. A repeated request with reversed input order returned the exact stored plan. After one claimed commit failed, the final budget was spent 9, reserved 7, remaining 0. The new overlapping request allocated zero extra units. The small budget case spent four pilots and allocated three units to `mixed-a`, zero to `mixed-b`, preserving the two-unit floor for a task that is actually activated.

Separately, 42 owned tests passed: 24 budget/provenance/concurrency contracts and 18 preparation guards. This is separate from the 18 unchanged author tests and the 240 oracle configurations. Concurrency tests use two independent handles on one SQLite file, proving that simultaneous plans reserve 8 and 0 units instead of 8 and 8, and that two competing workers receive one dispatch claim. The public record binds the owned test source and local log digests.

## Guarded product behavior

Immutable task revisions, pilot receipts, plans and reservations preserve IDs and evidence references. An exact integer 0 or 1 with a caller-verified terminal outcome is eligible; a probability, float `1.0`, boolean, NaN, unresolved outcome, stale policy or changed task is not. The verifier is an explicit caller-owned authority callback, not cryptographic attestation supplied by this module. A production consumer must bind it to authentic terminal task results and provide authoritative current revisions.

Accepted pilot receipt batches charge all statuses once before selection, including failed, stale, excluded and untrusted pilots. A batch whose costs exceed the remaining epoch budget is rejected atomically; the caller must control pilot spending before supplying receipts, because rejecting an over-budget batch cannot undo external work already performed. Duplicate receipt IDs with equal content deduplicate; conflicting identities fail. A bad callback cannot refund the already recorded pilot cost. Selector output must contain exactly the four original categories with known, unique integer category IDs and evidence-consistent semantics: strict `too_correct > 1 - upper`, strict `too_incorrect < lower`, inclusive `keep`, and independent `exclude_too_easy >= exclude`. Exclusions can overlap `keep`; the planner subtracts them before allocating.

The ledger uses SQLite `BEGIN IMMEDIATE` transactions. Identical request ID and content returns the stored plan; the same ID with different content fails. Outstanding reservations are included in every remaining-budget calculation. Stable IDs alone do not enforce budget: the shared transactional ledger does. These guarantees apply to consumers sharing the same SQLite file and explicit budget epoch, not to separate files/epochs, external model-provider quotas or callers that bypass the ledger.

Allocation is deterministic over sorted task IDs, respects each task's floor/cap and explicitly records unallocated remainder. A plan is rejected before enumeration if it would materialize more than 65,536 reservations; a one-billion-unit test confirms that zero reservation objects are constructed and actual pilot costs remain charged. Dispatch must claim a reservation against current task and policy revisions. The first owner receives `True`; repeated claims return `False`, competing owners fail, and claimed uncertain work is neither automatically retried nor refunded. Completion moves one reserved unit to spent even on failure. Only unclaimed pending work can be cancelled and released.

The core uses the standard library and an explicit selector backend; installing or importing the product planner does not import the external trainer. Its exact executed source digest is `5be5b5b37e385d37cc1ef9111c913c6e4e01e87ed614ddb65356eb76aa9ec012`. It controls future allocation; it neither supplies token/optimizer data nor exposes the original replay buffer as a guarded product API. No step-age contract is implemented in this planner: eligibility requires exact current policy/task revisions.

## Boundary observations retained

The unchanged source produced these observations, which the evidence keeps visible:

- `select_prompts` kept a NaN reward input rather than rejecting it. The product rejects nonfinite/binary-invalid evidence before selection.
- `ReplayBuffer.add` retained two prompts with configured maximum size one; add alone does not enforce capacity.
- The sample-all branch returned eight rows when the smaller response request implied four.
- A collapsed all-zero/all-one pool produced floating indices and a real TensorDict `IndexError` during replay sampling. The 240 helper optima do not prove every returned index is valid for downstream indexing.
- A missing prompt pop raised the original assertion as expected.
- Step-age acceptance alone retained `policy-6` data for a `policy-7` consumer. The upstream off-policy training algorithm was not executed; the product requires explicit exact revision provenance.

These observations are not patched out of the original source. The product uses only the original selector through its own validated boundary. It must not convert continuous forecasting scores or unresolved forecast probabilities into binary rewards to gain eligibility.

## Reproduction and prior work

[Example commands and input requirements](../examples/official_pilot_commit/README.md) use fresh outputs and separate source/dependency paths. The component checker rejects symlink ancestors, source/dependency/output overlaps, existing outputs and a writable parent not owned by the current user before normal author imports or output writes. Bytecode is redirected away from the original checkout. The author-test subprocess has a bounded timeout and diagnostics retain bounded traceback tails.

Earlier component probes used older wrapper revisions and remain separate ignored records; their timings are not pooled with this product run. Setup initially encountered a missing PyVers dependency; after installing the genuine distribution into isolated ignored storage, normal imports succeeded. This setup observation has no standalone raw result and is not counted as an author-code failure. An owned preparation-guard test also caught a directory-file-descriptor handling defect in our checker; it was repaired before the promoted checker and final successful run. The promoted component checker has unchanged bytes from the hardened candidate (`c0c9d48325728543b74121de1f1a4ac91aa14425904cc7224686c1f1505273bb`). The executed planner probe digest is `ffe30e51ea3f03f34f4ef4dcd79fefe436db11a9166f5264c2b6f2a5b72e1667`.

No model inference, real environment rollout, software verifier execution, optimizer update or full RL trainer ran in these fixtures. Wall times in the records are fixture diagnostics; no rollout speedup, GPU training speedup or learning improvement is inferred from them. The next runtime integration must bring authentic task outcomes and dispatch through this ledger before an end-to-end allocation experiment can make those claims.
