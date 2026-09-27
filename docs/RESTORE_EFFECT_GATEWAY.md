# Experimental host operation journal around VM restore

[`HostOperationJournal`](../future_prediction_bench/restore_effect_gateway.py)
keeps an external-operation ledger on the host, outside the restorable guest
disk. The opt-in `FencedMicroVMRuntime` facade checks this ledger before
QEMU snapshot/restore calls. **Nine focused unit tests** exercise real
SQLite and a fake VM; no real provider, QEMU integration, process-kill
recovery, or performance comparison has run for this facade.

This is an independent, narrow transfer from
[execution-edit safety research](https://arxiv.org/abs/2608.22928) and its
[author-linked implementation](https://github.com/eunomia-bpf/agent-check-restore-safety).
It does not import that checker, enforce its full semantics, or implement
safe fork/merge. The separately executed upstream history tests are described
in the [research review](ROLLOUT_ALLOCATION_RESEARCH_20260927.md).

## What the implementation checks

Before invoking a trusted effect callback, the journal commits a pending
record with the stable call ID, route, and canonical payload hash. A completed
call replays its stored JSON result without dispatching again. Reusing an ID
with a different route or payload fails. An episode cannot attach to another
episode's ledger.

A pending call means the provider outcome is unknown. It blocks snapshot,
restore, and blind redispatch until an authoritative read-only provider
lookup returns a committed result. An unknown lookup remains pending. A
SQLite writer transaction excludes new dispatches while the VM edit callback
runs. Fork is refused because this implementation has no branch identity
rule for external effects.

Provider keys use `fpb-op-v1:` followed by SHA-256 of canonical JSON
`[episode_id, call_id]`. Pair encoding avoids the earlier delimiter ambiguity
between, for example, `("a:b", "c")` and `("a", "b:c")`. Dispatch and status
lookup use the same versioned key. The ledger persists its key-format version.
Unversioned legacy ledgers with pending effects **cannot migrate automatically**:
their provider calls may exist under an earlier key and require manual
provider reconciliation using that original key. Empty or completed-only
legacy ledgers can adopt the new format, since completed calls replay stored
results without querying or dispatching under a new key.

## Required operating contract

The coordinator owns the journal and raw runtime. Every external effect must
use `dispatch` with a stable call ID for the same logical operation after
restore. Provider lookup and receipt semantics must be authoritative. Direct
runtime calls or alternate guest network/device egress bypass the facade;
this Python wrapper is not an adversarial egress boundary. The journal must
remain outside every restore domain, and effect callbacks must not recursively
invoke fenced VM edits or dispatch while a VM edit holds its writer lock.

SQLite WAL with `synchronous=FULL` supplies the journal's transaction
mechanism; reopening is tested, while abrupt host-process death and power-loss
durability are not established by these tests. The module supplies no actual
payment, email, market order, or other provider integration. Future real-guest
validation must demonstrate the sole-effect path, stable IDs after restore,
authoritative lost-response reconciliation, and crash behavior before making
an operational recovery claim.

Run the focused tests from the project root:

```bash
python3 -m unittest discover -s tests -p test_restore_effect_gateway.py -v
```

The tests cover stored-result replay, lost responses, conflict rejection,
dispatch/restore exclusion, fork refusal, key-pair ambiguity, and legacy
key-format migration gates. They establish those control paths, not exactly-once
delivery across arbitrary providers or a trained-agent result.
