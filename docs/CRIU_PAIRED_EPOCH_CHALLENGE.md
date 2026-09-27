# Interleaved-write challenge for CRIU paired checkpoint epochs

Three real three-trial diagnostics completed on 2026-09-27: an initial eight-MiB
run and readiness-guarded repeats at eight and 64 MiB. Every run reproduced loss
of an externally injected byte in the original parent chain, while its local
previous-complete-image parent candidate recovered the entire saved allocation
and continued execution. These are controlled QEMU/TCG results, not general
concurrent-write safety or a training/performance claim. The ordinary
comparison deliberately keeps its worker quiescent during each pair.

The implementation is
[`paired_epoch_probe.py`](../examples/official_crab_criu/paired_epoch_probe.py).
It requires the same marked disposable x86 Linux guest, whole original Crab and
integration source pins, real binary/kernel preflight, and static C worker as
the [process-chain experiment](OFFICIAL_CRAB_CRIU.md). It refuses to run without
an explicit optional `runtime_factory` hook in `check_chain.execute_mode`.
The normal checker uses its original runtime when that hook is not supplied.

## Repeated observed outcomes

The [curated diagnostic record](measurements/official_crab_criu_paired_epoch_2026-09-27.json)
contains the first run and both complete repeats, including negative original
recovery evidence, actual commands, parent links, injection witnesses, and
per-command CRIU log-tail records.

| Run | Owned allocation | Zero-write control | Original interleaved write | Local complete-image parent candidate |
| --- | ---: | --- | --- | --- |
| `official-crab-criu-paired-epoch1-20260927` | 8 MiB | Exact hash and progress passed; byte 23 | Failed full hash; restored byte 23 instead of 107 | Exact hash and progress passed; byte 107 |
| `official-crab-criu-paired-ready8-20260927` | 8 MiB | Exact hash and progress passed; byte 23 | Failed full hash; restored byte 23 instead of 107 | Exact hash and progress passed; byte 107 |
| `official-crab-criu-paired-ready64-20260927` | 64 MiB | Exact hash and progress passed; byte 23 | Failed full hash; restored byte 23 instead of 107 | Exact hash and progress passed; byte 107 |

In every write trial, the live target changed from 23 to 107 after the first
genuine pre-dump, with the actual present-page soft-dirty bit changing from
false to true. The original restore later recovered 23 and a different
full-region hash. The candidate recovered 107, the saved hash of all 8,388,608
or 67,108,864 owned bytes, and both post-restore counter increments. Each trial
retained its full chain and executed six logical pairs, or twelve genuine
runc checkpoint commands. The three candidate successes are repetitions of
one controlled fixture, not three distinct concurrency scenarios.

Both later repeats use the same frozen three-GiB assets and complete bounded
identity plus quiescent x86-64 `rt_sigtimedwait` (syscall 128) readiness guard.
The first run retains its preceding checker pin separately. The diagnostic
script is unchanged across all three runs. In every run,
`expected_diagnostic_passed=true` and `all_actual_recoveries_passed=false`:
the original challenged recovery actually fails. These results are not pooled
with the ordinary mode-timing comparisons, whose runtime is unmodified.

## First observed diagnostic result

The actual run is `runs/official-crab-criu-paired-epoch1-20260927/`. The
[curated diagnostic record](measurements/official_crab_criu_paired_epoch_2026-09-27.json)
retains the original negative case alongside the local candidate, actual
commands, source pins, and memory witnesses.

| Trial | Fixture writes | Target before / after injection / after restore | Full 8,388,608-byte recovery | Continued execution |
| --- | ---: | --- | --- | --- |
| `original_zero_write_control` | 0 | 23 / 23 / 23 | Passed | Passed |
| `original_interleaved_write` | 1 | 23 / 107 / 23 | **Failed: hash mismatch** | Not attempted after mismatch |
| `complete_process_parent_candidate` | 1 | 23 / 107 / 107 | Passed | Passed |

In both write trials, the target page was present before and after the write,
and its actual kernel soft-dirty bit changed from false to true. The original
challenge performed six actual restore-point pairs, then executed a real
restore. Its target reverted to 23 and its full-region saved/restored SHA256s
differed. Ordinary counters still returned correctly; checking only those
counters would have missed the lost write.

The local candidate used the same single interleaved write and six pairs. Each
non-anchor pre-dump command referenced the preceding completed `process`
directory; every current final dump still referenced its own `pre_dump`. Its
restored byte remained 107, full-region SHA256 matched the saved value, and
both counters advanced again after recovery. Each trial retained all ancestors
and executed twelve real runc checkpoint commands.

`expected_diagnostic_passed=true` means the control, observed original loss,
and candidate recovery matched this diagnostic's expectations.
`all_actual_recoveries_passed=false` remains explicit because the original
interleaved-write trial failed recovery. No successful recovery label is
assigned to that negative case. The diagnostic script was unchanged during
this run, with SHA256
`6cb40616322dcabf358b2780eeb74fb1f6831c8534d307a7fc5ecfa67450f98c`.

This first sample and both repeats use one controlled writer at one fixed
boundary, one memory byte, and the pinned x86 Linux/runc/CRIU stack under
emulation. They do not prove all concurrent writes, chain resets, retention
pruning, arbitrary applications, filesystem/network recovery, or GPU training.
Broader challenges remain separate validation. No author source file was
patched, and the candidate is a local adapter rather than an upstream fix.

## Hypothesis and original parent semantics

The [original process worker](https://github.com/open-agent-infra/crab/blob/9607d61a41dc44358cf078c4b438bfd971c8ee9d/crab/workers/process.py)
produces pre-dump A, then final A referencing pre-dump A. The next pre-dump B
references pre-dump A rather than final A. The
[runtime](https://github.com/open-agent-infra/crab/blob/9607d61a41dc44358cf078c4b438bfd971c8ee9d/crab/runtime/runc.py)
preserves these paths in actual runc commands. In
[runc's CRIU RPC code](https://github.com/opencontainers/runc/blob/v1.4.3/libcontainer/criu_linux.go),
supplying a parent image enables memory tracking.

The suspected mechanism is an epoch mismatch: a write after pre-dump A may be
captured only in final A, then excluded from pre-dump B after final A resets
tracking. B would still inherit the older pre-dump A bytes. The observed loss
above is consistent with this parent/epoch explanation. It establishes this
specific controlled failure, not a general audit of all upstream execution
paths or an author-accepted fix.

## Explicit diagnostic deviations

The probe constructs a local subclass of the genuine original `RuncRuntime`.
Its `pre_dump_process` first calls `super()` and requires the actual runc/CRIU
pre-dump to execute. Before returning control to the original process worker
for the matching final dump, an **external fixture memory writer** performs
exactly one `/proc/PID/mem` write: byte 23 becomes 107 at the first byte of the
owned allocation's second page. It never modifies the first or last pages used
by the normal counters. The target is derived from the original worker's own
address/length identity, checked against its actual private writable anonymous
mapping, and bounded to at most 64 MiB. No arbitrary PID or address is accepted.

The witness records original PID/start time, allocation identity, target
address/page index, exact live bytes before/after, actual pagemap entry, present
and soft-dirty bits, and successful pre-dump command. A short read/write,
unexpected initial byte, missing dirty bit, or changed PID identity makes the
diagnostic inconclusive. The writer does not clear CRIU's tracking bits.
After actual restore, a read-only runtime wrapper records the recovered target
byte before the ordinary checker hashes all owned memory and cleans up.

The caller hook and external writer are deliberate fixture changes. Upstream
source files, selector, process worker, command runner, runc, and CRIU remain
unmodified. This is neither an unmodified paper reproduction nor a performance,
rollout, or training speed claim.

## Three actual trials

| Trial | External one-byte write | Next pre-dump parent | Required diagnostic outcome |
| --- | --- | --- | --- |
| `original_zero_write_control` | Disabled; read-only observations | Original previous pre-dump | Full RAM recovery and continued execution pass; target remains 23 |
| `original_interleaved_write` | One write after the first actual pre-dump | Original previous pre-dump | If the hypothesis is reproduced, restored target is 23 while saved live RAM contains 107, and full-region hash fails |
| `complete_process_parent_candidate` | The same one write | Previous completed **process** image | Candidate succeeds only if target 107, full-region hash, and continued execution all pass |

The candidate changes only the next pre-dump's parent resolution in the local
runtime subclass. An anchor still has no previous parent; every current final
dump still references its own current pre-dump. The original backend executes
the generated commands. All parent images and original command/status evidence
remain available. If CRIU cannot use that complete image safely, its failure is
reported; there is no fallback that strips flags or bypasses restore checks.

The zero-write control disables only the external fixture writer: the existing
worker still performs its ordinary counter mutations. A failed control stops
the diagnostic before further trials. `actual_recovery_passed` always reflects
the ordinary mode result. Separate `expected_diagnostic_passed` is true only
when the control succeeds, the original write challenge demonstrably loses the
target with a full-region hash mismatch, and the proposed parent candidate
recovers exactly. An expected negative challenge is never labeled a successful
restore. If the original challenge recovers correctly, the status is
`hypothesis_not_reproduced`.

## Reproduction boundary

After the optional caller hook and diagnostic file have been explicitly
included in new guest assets, invoke inside the same correctly rooted,
preflight-validated disposable guest:

```bash
PATH=/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
LD_LIBRARY_PATH=/usr/local/lib:/usr/lib:/lib \
python3.12 -B /opt/fpb/probe/paired_epoch_probe.py \
  --crab-source /opt/fpb/crab --output /tmp/fpb-paired-epoch-new --memory-mib 8
```

Use a new output directory. The actual host root handoff, fixed inputs, and
guest marker are documented in the
[CRIU reproduction README](../examples/official_crab_criu/README.md).
For assets assembled by that current public builder, the host driver selects
the diagnostic explicitly:

```bash
python3 -m examples.official_crab_criu.run_microvm \
  --assets runs/criu-public-guest-assets \
  --output runs/criu-public-paired64 --memory-mib 64 --probe-program paired_epoch
```

The 64-MiB run uses a three-GiB disk, with enough actual free capacity for all
three trials and every retained ancestor. The current checker waits for the
complete bounded identity JSON and quiescent signal-wait readiness rather than
file creation alone. These later guards are explicitly recorded in the repeat
results; the historical first sample keeps its own pin.
`result.json` retains each complete ordinary mode result plus the injection,
restored-byte, command, and parent-override witnesses. The observed outcomes
above come from actual guest execution and full-region memory verification,
not syntax checks or inferred command success.
Before executing the trials, the probe checks actual free space against three
trials times two possible full-sized images times six boundaries, plus a
64 MiB reserve. It keeps all images and refuses insufficient capacity; an
out-of-space failure cannot count as the expected memory-loss diagnosis.
At 64 MiB the required bound is 2,368 MiB. Per-image inventories and exact
parent link targets are retained even after the intentional hash failure;
the original negative evidence is not discarded during cleanup. Inventory
collection and host per-command log-tail retrieval occur outside the mode wall
timer. No speed claim is made from this diagnostic's timings.
