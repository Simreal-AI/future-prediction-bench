# One real SWE-bench task in upstream MiniSandbox

The [source-bound measurement](../../docs/measurements/official_mini_flask_5014_grade_2026-09-25.json)
records a live offline run of the official SWE-bench Verified instance
`pallets__flask-5014` on pinned
[SWE-MiniSandbox commit `381ada53`](https://github.com/lblankl/SWE-MiniSandbox/commit/381ada53ab35dadb342add33ff006f3157c22fb7).
The actor used the upstream `SandboxDeployment`, SWE-ReX Bash session,
`post_init`, and versioned cache paths. Separate fresh verifier deployments
used upstream `_calculate_reward_swebench` through our fail-closed
`strict_grade.py`. The actor did not receive the test patch or expected test
names. It ran with Docker networking disabled and could not see the host's
`/private` or `/src` mounts from its chroot.

| Arm | Expected cases observed | Passed | Failed | Trusted reward | Grade time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unmodified base | 60 | 59 | 1 | 0 | 4.205 s |
| Official reference patch | 60 | 60 | 0 | 1 | 4.159 s |

Each arm has **one** sample. Fresh deployment creation took about 0.38 s and
`post_init` took 6.25–6.89 s per arm. The one-time image build, Python
package downloads, and cache repair are excluded from those timings. These
numbers establish real task discrimination and end-to-end grader wiring; they
are not a throughput, checkpoint, or RL-training speedup result.

## What the experiment had to repair

The first unmodified upstream `post_init` returned and produced a venv tar,
but the venv lacked Flask, pytest, and chardet. Upstream SWE-ReX logged a
Bashlex heredoc warning during setup. The later trusted grade failed when
`setup_env_swebench()` ran `python3 -m pip install chardet` and guest DNS could
not resolve the package index. A guest check also failed to resolve
`pypi.org`, while the outer disposable container resolved it. The strict
wrapper raised `AdapterInfrastructureError`; it did not assign a zero reward
for that infrastructure failure. We did not establish whether the heredoc
warning or the missing guest resolver caused each skipped installation
command, so this run does not claim unmodified upstream setup succeeded.

`prepare_flask_5014_cache.py` is the explicit operator repair. Outside the
actor chroot, it calls the pinned upstream requirement generator for the
task's exact environment-setup commit, installs its 61 non-comment
requirements and six upstream Flask 2.3 pins in a clean Python 3.11 venv,
installs the public base source from a writable copy, checks imports, and
atomically replaces the invalid venv cache. Its public wheelhouse contains
only `chardet==5.1.0`, `setuptools==70.0.0`, and `wheel==0.40.0`. The
per-deployment runner copies those public wheels into `/wheels` in the
sandbox, checks `PIP_NO_INDEX=1` and `PIP_FIND_LINKS=/wheels` in the same
persistent session used by the grader, and clears `PYTHONPATH` so Python
3.11 cannot import the host runner's Python 3.12 wheels. It also sets
`PAGER=cat` and `GIT_PAGER=cat`; the unmodified upstream evaluation script
invokes `git show`, which otherwise waited for an interactive pager in our
direct-deployment harness.

## Reproduction inputs

The upstream checkout, its dependencies, the source checkouts, and the
prepared Linux image are **not** included in the release bundle. This run
used an ARM64 Debian 12 image derived from a cached local base image, with
Python 3.11.2 and Git 2.39.5 installed through Debian packages. The
derived image's digest is in the measurement file; because its base image
was local, the image name alone is not a portable reproduction recipe.
`/opt/deps` supplied pinned upstream SWE-ReX, SWE-bench, R2E-Gym, and
MiniSandbox Python dependencies to the Python 3.12 host runner. The upstream
source itself was mounted at `/src`, not copied into this repository. A
compatible prepared Linux image needs root, `unshare`, `mount`, `chroot`,
Bash, tar, Git, Python 3.11 with `venv`, and the upstream Python 3.12 runner
dependencies. Use a disposable privileged container for the test.

The public Flask base checkout must be exactly
`7ee9ceb71e868944a46e1ff00b506772a53a4f1d`; the separate environment
requirements checkout must be exactly
`182ce3dd15dfa3537391c3efaf9c3ff407d134d4`. The upstream parquet
file's SHA-256 must be
`43ed5a3d1d98da36472c1ade65ddd2085d7b4ff694fcaf6a023a07c5c1f32f21`.
`task_asset_preflight.py`, `prepare_flask_5014_cache.py`, and
`run_official_flask_5014.py` check these bindings. Keep the parquet and
environment source under host-only `/private` mounts; never place them in the
actor's repository archive. The report includes hashes of the official
reference and test patches, but neither patch content nor test output.

With the project root as the current directory, a prepared image and
dependencies, and the two pinned Flask checkouts in the ignored cache, the
invocation shape is:

```sh
UPSTREAM="$PWD/runs/upstream-swe-mini-381ada53"
CACHE="$PWD/runs/official-mini-task-cache"
LINUX_DEPS="$PWD/runs/upstream-mini-linux-deps-381ada53"
DATASET="$UPSTREAM/dataset/SWE-bench/SWE-bench_Verified/data/test-00000-of-00001.parquet"
IMAGE="fpb-mini-flask:py311"
PYTHONPATH_IN_CONTAINER="/project:/opt/deps:/src/SWE-ReX/src:/src/sandboxdev:/src/R2E-Gym/src:/src/SWE-bench:/src/SWE-smith"
MOUNTS=(
  -v "$PWD:/project:ro"
  -v "$UPSTREAM:/src:ro"
  -v "$LINUX_DEPS:/opt/deps:ro"
  -v "$DATASET:/private/verified.parquet:ro"
  -v "$CACHE/source/flask:/private/base_source:ro"
  -v "$CACHE/env_source/flask:/private/env_source:ro"
  -v "$CACHE:/work/cache"
)

# One-time public-package preparation; network is required for pip.
docker run --rm "${MOUNTS[@]}" \
  -e "PYTHONPATH=$PYTHONPATH_IN_CONTAINER" "$IMAGE" \
  python3 -m examples.official_mini_sandbox.prepare_flask_5014_cache

# Actor and both trusted verifier deployments run without network.
docker run --rm --init --privileged --network none "${MOUNTS[@]}" \
  -e "PYTHONPATH=$PYTHONPATH_IN_CONTAINER" "$IMAGE" \
  timeout 600s python3 -m examples.official_mini_sandbox.run_official_flask_5014
```

The runner prints only a summary with counts, rewards, timings, and log
digests. The public measurement was curated from that summary and also binds
the script, checker, environment, image, and wheelhouse hashes. A failed
setup, incomplete parsed case map, timeout, or unexpected pass/fail outcome
must remain an error rather than a model reward.
