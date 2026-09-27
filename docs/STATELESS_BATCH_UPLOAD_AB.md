# Bounded stateless case upload: same-contract A/B

This experiment changes one trusted-host preparation step in the Boltons v2
prepared-microVM plus stateless-verifier path. The baseline uploads the
host-private Python case source with separate truncate, data, and digest shell
calls. The experimental arm uses one bounded shell command to write the same
source, set mode `0600`, and print its SHA-256. The host checks that digest
before submission. Expected outputs remain on the host. A request longer than
3,800 bytes falls back to the unchanged serial uploader.

[`benchmark_stateless_batch_upload_ab.py`](../examples/realworld_boltons26/benchmark_stateless_batch_upload_ab.py)
prepares one clean VM template, then runs three alternating serial/one-command
pairs for both a scripted repair and an untouched baseline. All 12 episodes
use the same effective task SHA, template, 14 private cases, stateless evidence
kind, and guest UID/GID 65534. The experiment requires exact action-observation,
case-result, source-hash, and reward parity. The repair scores 14/14 and reward
1; the untouched baseline scores 7/14 and reward 0. Full episode wall time
includes child provisioning, actions, grading, final-source attestation, VM
close, and child-disk removal. Template setup is recorded separately and
charged to both arms in the amortized comparison.

The published task file contains an older replace-text helper binding. The
benchmark independently verifies the current pinned public helper bytes and
refreshes only the in-memory task metadata for *both* arms. The report records
the original task-file digest, original refreshed task digest and helper digest,
and the effective task and helper digests. This is one new frozen contract; its
timings must not be compared directly with earlier reports that used the older
task SHA.

The [sanitized measurement](measurements/stateless_batch_upload_ab_2026-09-25.json)
shows a paired upload-stage saving in 5 of 6 pairs, with a median of about
**4.97 ms**. Complete-episode totals were 10.71 s serial and 10.52 s
one-command across six episodes per arm (1.018×); charging the shared 6.59 s
template preparation to each arm gives 1.011×. Pair-level complete-episode
differences changed sign, so these data do **not** establish a whole-episode
speedup. The repeated ~0.3–0.4 s submit snapshot and ~0.35–0.38 s guest batch
verification stages are larger future targets. This was one Apple Silicon host,
one public solved fixture, and scripted policy actions; no model inference or
RL optimization was measured.

With the locally generated pinned fixture and QEMU assets, reproduce with:

```bash
python3 -m pytest -q tests/test_stateless_batch_upload_ab.py
python3 -m examples.realworld_boltons26.benchmark_stateless_batch_upload_ab \
  --task-dir runs/boltons-v2-task-pinned2-20260925 \
  --assets-dir runs/microvm-assets-v2task-pinned2-20260925 \
  --contract examples/realworld_boltons26/stateless_contract_v2.json \
  --output runs/stateless-upload-ab-new-run
```

The `runs/` fixture and VM image are local generated inputs, not part of the
source archive. Reproduction needs a compatible ARM64 QEMU/HVF host and those
verified assets. The script writes a path-free public report and keeps private
case specifications out of it.
