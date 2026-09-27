# Collinear-inspired seeded CI task-world experiment

Collinear's official documentation describes a clean workspace with seed data
for each rollout, a [`GET /tools` and `POST /step` interface](https://docs.collinear.ai/api-reference/tool-server-protocol),
[before/after application-state diffs](https://docs.collinear.ai/simulation-lab/understanding-results),
and [programmatic verification](https://docs.collinear.ai/simulation-lab/verifiers-and-reward-models).
Its [sandbox documentation](https://docs.collinear.ai/core-concepts/sandbox)
uses isolated Docker networks and containers locally and supports remote
Daytona execution. The [CI triage example](../examples/collinear_ci_triage/README.md)
transfers the task-world and verifier ideas into this project without copying
Collinear's service or its isolation layer.

## Contract and threat model

The simulated task has 300 historical runs, one failed release-branch run, and
one unrelated healthy main-branch run. The target run timed out after 87 seconds
under a 60-second job limit. The operator prompt asks for a 120-second limit
on that exact job and a rerun of its original commit. The agent can discover
four tools: inspect the run, update a job timeout, queue a rerun, and write a
triage note. The interface uses Collinear's request/response *shapes* in
process; it is not an HTTP service. All SQL statements use parameters, and
each rollout is bounded to 16 `POST /step` attempts and a 4096-byte action
envelope. Malformed attempts consume the budget and receive trace entries with
unchanged state digests and empty diffs.

The host creates a separate SQLite file for every rollout. It captures the
semantic application state before and after each step, records SHA-256 digests
and structured row diffs in an in-memory run artifact, and checks the final database
against an exact expected state. A successful grade requires the target
timeout change, a subsequent passing rerun for the same commit, and no extra
changes to any job, note, run, or metadata row. The verifier runs outside the
tool interface; neither its result nor expected state enters an agent-facing
tool response. This is a public, auditable verifier, so “host-only” describes
the runtime data boundary, not secrecy of the published source.

The negative suite checks a persuasive note without the fix, a forged trace
claiming a passing rerun, a rerun before the fix, a timeout broadened to five
minutes, and a fix on the wrong branch. All five receive reward 0. Two
simultaneously live sibling rollouts start at the same digest, use distinct
database paths, and reach the same final digest independently; changing one
does not change the other. This tests logical rollout state, not container or
VM security isolation. The task executes no untrusted repository code.

## Measured seed materialization

The baseline builds a fresh SQLite schema and inserts all seed rows for every
rollout. The candidate builds the same seed once, then copies that prepared
database into each new private rollout directory. **Both** arms start fresh
and execute the same three scripted tools and the same host verifier. A run
includes setup, tool discovery, three steps, verification, and directory
teardown. The step clock includes application-state snapshots and row diffs.
The two conditions alternate order in each pair; the first three pairs are
warmups and 30 subsequent pairs are measured. The one-time seed build is
reported separately and charged to all 30 prepared episodes in the
amortized ratio.

On the development host, the [path-free 30-pair report](measurements/collinear_ci_triage_2026-09-25.json)
records these medians and nearest-rank p95 values:

| Metric | SQL reseed baseline | Prepared seed copy |
| --- | ---: | ---: |
| Fresh setup median / p95 | 3.022 / 5.101 ms | 1.219 / 2.291 ms |
| Step median / p95, 90 calls per arm | 1.558 / 1.833 ms | 1.552 / 1.790 ms |
| Host verification median / p95 | 2.011 / 2.113 ms | 1.976 / 2.059 ms |
| Complete graded episode median / p95 | 11.114 / 13.736 ms | 9.088 / 10.744 ms |

Summing all 30 measured episodes gives **344.287 ms** for SQL reseeding and
**279.096 ms** for copying, a **1.234×** same-host simulated episode-time
ratio. Charging the **2.104 ms** one-time seed build to the prepared arm gives
**281.199 ms** and a **1.224×** ratio. Both arms have identical initial and
final application-state digests and reward 1 in every measured pair; five
adversarial checks produce no false rewards. The per-step cost is essentially
unchanged, so the observed difference comes from seed materialization.

This is a small synthetic SQLite workload on one host. It is **not** a
Collinear benchmark, a real CI deployment, a Docker/VM checkpoint result, or
an RL-training throughput measurement. It does not measure model inference,
hidden-test execution on an actual repository, distributed tool servers, or
multi-tenant safety. A real deployment would need isolated services, resource
limits, durable event provenance, and independent held-out task families.
