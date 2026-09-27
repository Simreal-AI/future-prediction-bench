# Humanize repository repair fixture

This is a second, distinct Python repository task for the generic Docker
`RealWorldEnv` adapter. It pins [Humanize 4.15.0](https://github.com/python-humanize/humanize/releases/tag/4.15.0)
at commit `2ddb5903cdc1c7e6eb6b083f4f99f73db50aecd9` and the
[PyPI source distribution](https://pypi.org/project/humanize/4.15.0/) at
SHA-256 `1dd098483eb1c7ee8e32eb2e99ad1910baefa4b75c3aff3a82f4d78688993b10`.
The [upstream repair PR #329](https://github.com/python-humanize/humanize/pull/329)
documents the defect: `naturalsize(999999)` returned `1000.0 kB` after
rounding, instead of carrying into `1.0 MB`. The task builder applies the
same narrow carry condition to the earlier pinned release. Humanize is MIT
licensed; the archive remains a separate upstream download and is not bundled
as project source.

`make_task.py` verifies the downloaded archive hash before extracting it,
rejects unsafe tar members, records the extracted workspace digest for a
pre-episode integrity check, and creates a seed workspace, an independent
host verifier directory, a task, and baseline/repair action files. The task
allows a policy to read or replace a repository file, run a fixed visible
import check, and submit. The verifier executes 14 decimal, binary, GNU,
negative, custom-format, and ordinary-value cases in host-controlled Docker
containers. It is outside the policy workspace and is never mounted in the
actor container. Its construction is public in this repository, so
“host-private” describes the execution boundary, not secret test knowledge.

Build and test without network access using an already downloaded archive:

```sh
python3 -m examples.realworld_humanize.make_task \
  --sdist /path/to/humanize-4.15.0.tar.gz \
  --output runs/humanize-task
FPB_HUMANIZE_SDIST=/path/to/humanize-4.15.0.tar.gz \
  python3 -m pytest -q tests/test_realworld_humanize_fixture.py
python3 -m examples.realworld_humanize.smoke \
  --task-dir runs/humanize-task --image sha256:<locally-cached-python-image-id> \
  --output runs/humanize-docker-smoke \
  --public-output docs/measurements/realworld_humanize_docker_smoke_2026-09-25.json
```

Omitting `--sdist` downloads only the hardcoded PyPI archive URL and still
checks the same SHA-256. Docker runs require a locally cached image; the
adapter disables network pulls and binds the resolved immutable image ID into
the task. The raw run directory contains trusted case evidence and should
stay private. The published smoke report contains only status, aggregate case
counts, workspace/evidence hashes, and provenance.

The offline source check yielded **5/14** passing cases before repair and
**14/14** after repair. The real Docker `RealWorldEnv` smoke independently
graded one baseline episode at **0.0 (5/14)** and one scripted repair episode
at **1.0 (14/14)** on the same SHA-pinned image; see the
[path-free measurement record](../../docs/measurements/realworld_humanize_docker_smoke_2026-09-25.json).
This fixture is a **public solved integration test**:
scripted actions demonstrate reward separation and adapter portability. It
does not measure learned repair ability, held-out generalization, policy
inference, optimizer updates, or episode throughput.

For a small matched verifier-scheduling check, run 12 fresh Docker episodes
with the same pinned archive and cached image:

```bash
python3 -m examples.realworld_humanize.benchmark_verifier_workers \
  --sdist /path/to/humanize-4.15.0.tar.gz \
  --image sha256:<locally-cached-python-image-id> \
  --output runs/humanize-verifier-workers-ab \
  --public-output docs/measurements/realworld_humanize_verifier_workers_ab_2026-09-25.json
```

The [three-pair, alternating-order report](../../docs/measurements/realworld_humanize_verifier_workers_ab_2026-09-25.json)
keeps all 14 hidden-case results and action observations equal within each
baseline/repair pair. On the scripted repair, one versus four verifier
workers took **2.190 versus 1.098 s** median verification time and
**3.222 versus 2.130 s** complete episode time. This extends the scheduling
check to a second solved repository; it has no model generation or optimizer
and is too small to estimate broad training throughput.
