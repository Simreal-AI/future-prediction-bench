# Bounded text edits in complete microVM episodes

The optional `replace_text` action edits one exact fragment in an existing file. It is available only when a frozen real-world task declares the tool and binds the exact host implementation and compressed guest program by SHA-256. The original Boltons fixture and its default verifier path remain supported. The separate v2 Boltons fixture exposes both `write_file` and `replace_text` so that the two repairs can be compared on the same task, seed, verifier, and VM assets.

An action supplies a relative workspace path, the expected SHA-256 of the **entire current file**, one nonempty `old_text`, and `new_text`:

```json
{"action":"replace_text","path":"boltons/strutils.py","expected_file_sha256":"<64 lowercase hex digits>","old_text":"<exact fragment>","new_text":"<replacement>"}
```

The adapter accepts at most 600 bytes of compact UTF-8 JSON for the complete action. The target must be a regular file of at most 64 KiB with one link. The helper walks relative path components without following symlinks, checks the full-file hash and exactly one occurrence (including overlapping occurrences), and writes the result through a same-directory temporary file and atomic rename. It retains file mode and ownership or aborts before replacement. Missing or changed files, wrong hashes, ambiguous matches, and size limits produce bounded conflict observations. Unsafe paths and I/O failures do not become a scored task failure. Metadata checks around the rename detect common concurrent changes; they are not a strict compare-and-swap against a hostile concurrent writer.

The VM receives only base64-encoded action data and a fixed, compressed trusted helper. User-controlled text is never inserted as shell or Python source. The guest payload is committed as fixed bytes, so Python minor versions do not regenerate a different program. Its source and decompressed-program hashes are checked against the frozen v2 task. The optional VM path is validated on an Apple Silicon host with Python 3.12 and a pinned ARM64 Linux guest; the core benchmark supports Python 3.10 or newer.

## Reproduce the paired experiment

First build the v1 pinned microVM task and assets using [the microVM setup guide](MICROVM_ENV.md), or reuse an already verified v1 asset directory. The Boltons source archive stays under ignored `runs/`; it is not part of the release bundle. Generate a distinct v2 task from the digest-checked pinned source distribution, then rebind the checked VM assets to its v2 task ID. `make_task_v2` downloads the pinned distribution by default; pass `--sdist <local-archive>` for an offline run.

```bash
python3 -m examples.realworld_boltons26.make_task_v2 \
  --output runs/boltons-v2-task
python3 -m examples.realworld_boltons26.rebind_assets_v2 \
  --task-dir runs/boltons-v2-task \
  --v1-assets-dir runs/boltons-vm-assets \
  --output runs/boltons-v2-assets
python3 -m examples.realworld_boltons26.benchmark_small_edit_v2 \
  --task-dir runs/boltons-v2-task \
  --assets-dir runs/boltons-v2-assets \
  --stateless-contract examples/realworld_boltons26/stateless_contract_v2.json \
  --preinstall-stateless-helper \
  --output runs/boltons-v2-small-edit --repetitions 5
```

Each repetition runs independent cold and prepared `RealWorldEnv` episodes for both edit methods and both repaired/untouched branches. Method, branch, and condition order alternate. Every repair must pass all 14 host-private cases and receive reward `1`; every untouched baseline must retain the expected seven passes and reward `0`. The harness checks the same opening and all action observations within each cold/prepared method pair, the common read/check/submit observations across edit methods, all per-case results, rewards, final source hashes, frozen artifact digests, and independent child RAM/ext4 markers. The measured episode wall time starts before child disk provisioning and ends after grading and final attestation. The one-time parent boot, helper installation, and template export are charged separately in setup-adjusted ratios.

Preinstalling the **generic** stateless verifier helper is part of the prepared condition in this command; cold episodes start from the pristine image. Hidden case source is uploaded only after policy submission, and expected outputs remain on the host. This is a composed prepared-template plus task-declared stateless-verifier experiment, with a narrower per-case isolation contract than a full-VM restore before each case. Its guest runner allows 10 seconds per hidden case; the general full-VM Python runner may allow up to 25 seconds. The pinned Boltons cases complete within the narrower bound, but this opt-in mode is not a drop-in replacement for arbitrary slow tests. Policy inference, optimizer updates, and trained-model quality are outside its timing boundary. The task is a publicly solved integration fixture, so these runs do not establish performance across unseen repositories.

## Five-repetition result on the pinned host

The [curated report](measurements/prepared_small_edit_v0.7.0.json) contains all 40 complete graded episodes from five alternated repetitions. Each method ran five repaired and five untouched episodes per condition. Every repaired episode passed 14/14 host-checked cases with reward `1`; each untouched episode passed 7/14 with reward `0`. Each attempt used one full-VM restore before the namespaced hidden batch, no fallback, and a fresh child with separate RAM and ext4 markers. Source digests, common action observations, every hidden case result, and reward matched across the method/condition comparisons. The cases and repair are intentionally public in this solved integration fixture; expected outputs are withheld from the policy during each episode, not secret benchmark answers.

| Scripted method | Repair-only median cold → prepared | Mixed repair/baseline median cold → prepared | Steady-state ratio of mixed total times | Setup-adjusted ratio for this method's 10 prepared episodes |
| --- | ---: | ---: | ---: | ---: |
| Full-file `write_file` | 7.138 → 2.527 s | 6.825 → 1.987 s | 3.398× | 2.564× |
| Bounded `replace_text` | 6.637 → 1.752 s | 6.577 → 1.728 s | **3.800×** | **2.751×** |

Template preparation took 6.610 s once. Across both methods, charging that one setup to the 20 prepared episodes gives a 3.048× total-time ratio against 20 cold episodes. The repair edit action itself took 0.529/0.601 s for full writes versus 0.051/0.052 s for bounded replacement (cold/prepared medians); these are action timings, not whole-episode speedups. The mixed medians combine branches with different scripted action lengths, so the repair-only medians are reported separately. This episode experiment does not answer the separate 5–6× paper-relative checkpoint/restore latency tolerance. The opt-in process census is checked before snapshot, before grading, and after grading, but its live-process scan is best-effort rather than an adversarial atomic freeze.
