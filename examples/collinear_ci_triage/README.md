# Simulated CI triage task world

This example implements a small, stateful CI workflow inspired by Collinear
Simulation Lab's [fresh rollout sandboxes](https://docs.collinear.ai/core-concepts/sandbox),
[tool-server protocol](https://docs.collinear.ai/api-reference/tool-server-protocol),
[application-state diffs](https://docs.collinear.ai/simulation-lab/understanding-results),
and [programmatic verifiers](https://docs.collinear.ai/simulation-lab/verifiers-and-reward-models).
It is our own **in-process SQLite simulation**. It does not start Collinear,
Docker, a VM, a CI service, or a policy model.

The task prompt is: “Investigate failed CI run 1842 for `northstar/api`
`release/2.4`. If the job timeout caused the failure, set only that job's
timeout to 120000 ms and queue a rerun of the same commit. A note alone is not
a fix.” The `inspect_failure` tool exposes the simulated log and current
configuration. `update_job_timeout`, `queue_rerun`, and `post_triage_note`
change application state. The harness offers the shapes `GET /tools` and
`POST /step` through an in-process method; no HTTP listener is started.
Malformed step attempts consume the 16-attempt rollout budget and are logged
with unchanged state digests and empty diffs.

Every rollout gets a distinct temporary SQLite file with the same 302 seeded
CI runs. The host stores a digest of the initial state and before/after state
digests plus structured row diffs for each step. A host-only verifier requires
the exact target configuration, a successful rerun after the change, and no
other state changes. A claim written only in a triage note, a forged text
trace, an early rerun, a wrong-branch change, or a broad timeout all score 0.
The tool response does not contain the reward or verifier checks.

Run the tests and paired measurement from the project root:

```bash
python3 -m unittest discover -s tests -p 'test_collinear_ci_triage.py' -v
python3 -m examples.collinear_ci_triage.benchmark \
  --output /tmp/collinear-ci-triage-report.json \
  --repetitions 30 --warmup 3
```

The [method and measured result](../../docs/COLLINEAR_CI_TRIAGE.md) explain
the setup baseline, amortization, parity gates, and limits. These SQLite files
provide logical state separation only. The simulated agent is not isolated
from the host process or operating system, so this example must not be used
as a sandbox for untrusted code.
