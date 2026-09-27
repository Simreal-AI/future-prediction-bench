# Humanize cooperative checkpoint experiment

This opt-in experiment runs **Humanize 4.15.0** under the same logical coding
task and 14 host-private verifier cases in two environments: a cooperative
in-guest process/OverlayFS checkpoint and a prepared QEMU full-VM template.
Both arms start from the same SHA-pinned ARM64 guest image, execute the same
`read_file` → `replace_text` → `submit` actions, and use the same prompt,
reward contract, action window, source bytes, and verifier. The experimental
task IDs are the same, but the runtime-specific artifact bindings make their
task hashes different. The full-VM arm advertises more tools; this scripted
comparison exercises only the three common actions.

The solved fixture is public, not held out. It is built from the pinned
Humanize 4.15.0 source distribution (SHA-256
`1dd098483eb1c7ee8e32eb2e99ad1910baefa4b75c3aff3a82f4d78688993b10`).
The pristine source tree digest is
`b42ad06044a4bab6906fc95b16c4fa1348a8c82f36cbe490f18b9a929293cfd2`;
the 14-case verifier file digest is
`4e36079f09ed50c6b9d1c23aed746e9d6ffca22cf62878411717b8329dcb7305`.
The measured pristine Humanize qcow2 digest is
`7d5fe2be77059694b546988a5cbebda0d1fe9dc00ffa00b5fb038f72843bbeba`.
The v2 fixture adds an exact SHA-bound `replace_text` tool while preserving
the v1 prompt, 14 case programs and expected outputs, and all-or-nothing
reward. The expected outputs stay on the trusted host. Candidate case programs
run as UID/GID 65534 in bounded guest namespaces; case source, not expected
output or reward, crosses the guest bridge.

The cooperative arm forks a frozen Python process into a writable OverlayFS
episode branch. For each verifier case, a bounded copy of the submitted
workspace is made on tmpfs because this guest cannot nest a third OverlayFS
layer. It retains the resident helper's per-case PID namespace, unprivileged
UID, resource caps, output cap, deadline, and cleanup. The pinned Humanize tree
has 197 entries before cooperative markers, so the copy bound is 256 entries
and 2 MiB of files; the branch tmpfs is capped at 64 MiB. The prepared control
seals full VM CPU/RAM/device/ext4 state before actions, spawns a fresh child VM
for every graded episode, and runs the same 14 cases with its host-private
stateless verifier inside that child. It checks the submitted source SHA on
the guest after grading. The outer guest tree is audited before and after
cooperative episodes, and every branch receives a fresh process nonce.

This is **not** security equivalence. The cooperative template shares a guest
kernel and trusts guest root; it cannot recover from guest-kernel failure or
provide the full-VM isolation boundary. Neither experiment trains a model.

## Measured result

The source-bound, sanitized [five-repetition report](../../docs/measurements/cooperative_humanize_ab_fivepair_2026-09-25.json)
records 10 alternating repair/baseline pairs (20 complete graded episodes)
under QEMU 11.1.1/HVF on the same host. Every pair matched on the common
read/edit observations, submitted source digest, all 14 hidden-case results,
and reward. Repair passed 14/14 and scored 1.0 in both arms; baseline passed
5/14 and scored 0.0 in both arms. No pair fell back or returned pending.

| Timing scope | Cooperative | Prepared full VM |
| --- | ---: | ---: |
| Median complete graded episode, 10 per arm | 0.619122 s | 1.222544 s |
| 95th observed complete episode | 0.645112 s | 1.426444 s |
| One-time setup | 6.413528 s | 6.282130 s |
| Setup plus 10 graded episodes | 12.564683 s | 18.609390 s |

The steady-state median ratio is 1.9746× full-VM/cooperative; the ratio of
setup-plus-ten totals is about 1.481×. Episode timing includes branch or child
creation, actions, all 14 verifier cases, and teardown. It excludes the
one-time guest image build and the separate read-only outer-tree/source
attestations. The full-VM arm's prepared-template setup and the cooperative
service upload/start are separately charged above. This is a single pinned
repository and scripted workload, so it does not establish a general speedup
or training throughput claim. The earlier Boltons measurements retain their
own source-bound results and were not rerun or changed by this experiment.

The raw per-case report stays under ignored `runs/`; the public JSON contains
aggregate parity booleans and source hashes without case code, candidate
stdout, expected outputs, or local paths. Its SHA-256 is
`f717494419a7dd5bf42ed723b114bb84585b8f42acc78e7e947e7f2f8a42c9f9`.

## Reproduce

Use an Apple Silicon host with QEMU/HVF, Docker, `qemu-img`, and Python 3.12.
Supply a local copy of the SHA-pinned Humanize source archive and a prior
pinned ARM64 boot-assets directory with `vmlinuz-virt`, `initramfs-virt`,
`modloop-virt-padded.raw`, and its manifest naming the cached immutable
`linux/arm64` Python 3.12 Docker image. The builder checks every digest and
does no network access. Each command below writes into ignored `runs/`.

```sh
python3 -m examples.cooperative_realworld_humanize.make_task_v2 \
  --sdist /path/to/humanize-4.15.0.tar.gz \
  --output runs/humanize-v2-task

python3 -m examples.cooperative_realworld_humanize.prepare_microvm \
  --task-dir runs/humanize-v2-task \
  --source-assets-dir /path/to/pinned-arm64-boot-assets \
  --output runs/humanize-v2-assets

python3 -m examples.cooperative_realworld_humanize.benchmark \
  --task-dir runs/humanize-v2-task \
  --assets-dir runs/humanize-v2-assets \
  --output runs/humanize-cooperative-ab \
  --repetitions 5
```

The asset builder accepts only the exact image digest and source tree listed
in its code, and the prepared template separately pins the resulting qcow2
digest. Rebuilding ext4/qcow2 can change image bytes even from the same source;
an independently rebuilt image needs a reviewed, versioned pin update before
the prepared path will run. The report names all measured source and artifact
digests so this change cannot be silently treated as the published result.
