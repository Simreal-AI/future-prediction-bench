# Cooperative checkpoint failure stress

The original cooperative checkpoint A/B runs used only the pinned Boltons and
Humanize verifier cases. They established ordinary grading parity, but did
not exercise a verifier candidate that leaves a detached descendant, creates
an out-of-workspace symlink, or closes stdout before its lifetime ends. Those
are meaningful gaps for a reusable process checkpoint: one surviving process
or writable handle could affect a later branch while the template appears
quiescent.

The [real-QEMU stress driver](stress_qemu.py) injects those faults into one
trusted diagnostic branch of each pinned repository, then reuses the same
frozen template for 20 alternating, normally graded `RealWorldEnv` branches.
The diagnostic branch is not a policy episode and does not produce reward.
Its 14 bounded case programs are host-authored; no hidden expected output is
sent to the guest. The first case forks a descendant, moves it to a detached
session, changes its process name, acknowledges that it ran over a pipe, then
leaves it sleeping. The nested PID namespace must terminate it when its init
exits. The second case makes a symlink from its private workspace to a path
outside the workspace and attempts a write; the following case checks that
the symlink did not persist. A fourth case closes stdout and sleeps for 30
seconds; the existing candidate deadline must score return code 124 after
about 10 seconds and clean up. The remaining cases verify that the same batch
continues.

After the fault batch and each ordinary branch, the driver checks the guest's
`/proc` for the named descendant, checks for leftover root-side resident/case
pool directories, attests the installed service helper, and hashes the whole
outer workspace. Every ordinary branch gets a fresh process nonce and must
match the repository's stable per-case return-code/stdout-digest/pass results,
submitted source digest, and reward. A failed cleanup or transport withholds
a successful report.

The [sanitized QEMU 11.1.1/HVF report](../../docs/measurements/cooperative_failure_stress_2026-09-25.json)
has SHA-256 `22d5c037be3bd2eebb1882f2c5267bf67802eaebfe3f07cdca965090c140f36c`.
Both pinned guests passed the fault batch and 20 subsequent graded branches:

| Repository | Repair branches | Baseline branches | Unique branch nonces | Reward and cases |
| --- | ---: | ---: | ---: | --- |
| Boltons 26.0.0 | 10 | 10 | 21 | Repair 1.0, 14/14; baseline 0.0, 7/14 |
| Humanize 4.15.0 | 10 | 10 | 21 | Repair 1.0, 14/14; baseline 0.0, 5/14 |

Every post-branch census was empty, and the outer workspace and installed
helper digests stayed unchanged. The stdout-close fault returned case code
124 in both guests, so it did not turn into an infrastructure-pending reward.
This exercise evaluates 40 graded branches and two 14-case diagnostics; it
does not benchmark checkpoint speed or model training. The raw per-branch
case digests remain under ignored `runs/`; the public report contains no
candidate stdout, verifier expectations, local paths, or branch nonces.

This is targeted evidence, not a comprehensive sandbox proof. The process
census looks for the deliberately named descendant after each case batch and
branch; it does not prove the absence of every possible process or transient
effect. The symlink probe covers one out-of-workspace target. The fixture and
guest root are trusted, and the cooperative branches share one guest kernel;
they do not inherit the isolation or crash-recovery guarantees of full VM
snapshots. The test does demonstrate that this detached descendant, denied
write, and timeout do not contaminate subsequent graded branches on the two
measured images.

Run with freshly built, SHA-pinned task and guest assets from the
[Boltons setup](../../docs/SMALL_EDIT_MICROVM.md) and the
[Humanize setup](../cooperative_realworld_humanize/README.md):

```sh
python3 -m pytest -q tests/test_cooperative_failure_stress.py
python3 -m examples.cooperative_realworld.stress_qemu \
  --boltons-task runs/boltons-v2-task \
  --boltons-assets runs/boltons-v2-assets \
  --humanize-task runs/humanize-v2-task \
  --humanize-assets runs/humanize-v2-assets \
  --output runs/cooperative-failure-stress \
  --branches 20
```

The output directory must be new. The driver checks both rootfs digests
against the published pins and writes a raw and sanitized report into the
ignored output directory. The checked-in sanitized report is bound to the
exact host/runtime/guest helper hashes used for this run.
