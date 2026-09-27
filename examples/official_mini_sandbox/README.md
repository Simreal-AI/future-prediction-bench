# Pinned upstream MiniSandbox terminal probe

`session_bridge.py` calls the real SWE-MiniSandbox/SWE-ReX terminal API when
given an already initialized official `SandboxDeployment`. The
`check_linux_session.py` probe creates that upstream deployment and Bash
session in Linux, then checks a private mount namespace, chroot visibility,
and `/tmp` tmpfs. It does **not** prepare a coding task, build a venv, or run
the private grader.

The upstream checkout tested here is
[`381ada53ab35dadb342add33ff006f3157c22fb7`](https://github.com/lblankl/SWE-MiniSandbox/commit/381ada53ab35dadb342add33ff006f3157c22fb7).
The script verifies its Git HEAD and a digest of 312 Python files from the
MiniSandbox, SWE-ReX, SWE-bench, R2E-Gym, and SWE-smith source trees before
running. The pinned checkout must be mounted at `/src`, and this project's
`examples` directory must be mounted at `/project/examples`.

The [observed result](../../docs/measurements/official_mini_linux_session_2026-09-25.json)
was produced in a **local-only** `quantbench-ubuntu-test:0.7.0` arm64 Ubuntu
image with Python 3.12, Bash, `unshare`, `mount`, `chroot`, and coreutils. The
image and the ignored `/opt/deps` installation are not shipped in this
repository, so this is not yet a turnkey Docker reproduction. The installed
Python set included `bashlex==0.18`, `pexpect==4.9.0`,
`pydantic==2.13.5`, `ray==2.58.0`, the dependencies declared by the pinned
SWE-bench and SWE-ReX source trees, and their transitive dependencies. The
upstream source was supplied on `PYTHONPATH`, not copied into this package.

Given an equivalent **disposable privileged Linux** image and installed
dependencies, the invocation is:

```sh
docker run --rm --init --privileged --network none \
  -v "$UPSTREAM_CHECKOUT:/src:ro" \
  -v "$LINUX_PYTHON_DEPS:/opt/deps:ro" \
  -v "$PROJECT_ROOT/examples:/project/examples:ro" \
  -e PYTHONPATH=/project:/opt/deps:/src/SWE-ReX/src:/src/sandboxdev:/src/R2E-Gym/src:/src/SWE-bench:/src/SWE-smith \
  "$PREPARED_LINUX_IMAGE" timeout 50s python3 -m examples.official_mini_sandbox.check_linux_session \
  --conda-env /opt/quantbench-venv
```

The path passed to `--conda-env` must exist in the image and contain the
Python environment the upstream startup code mounts. This smoke command
disables network access while running the operator session. The first
dependency installation requires network access separately. It uses the
upstream's exact `SandboxDeploymentConfig.get_deployment()`, `start()`,
`startup()`, and SWE-ReX session APIs; a minimal terminal-bundle path only
satisfies the config's nonempty-bundle requirement. It does not call
`post_init()` or reproduce the paper's setup/storage measurement.

A subsequent [real Flask task run](OFFICIAL_TASK_RUN.md) advanced beyond that
terminal probe. Using pinned upstream SWE-bench Verified instance
`pallets__flask-5014`, an operator-prepared Python 3.11 cache, and the real
upstream deployment and grader, it observed all 60 expected tests: baseline
reward 0 (59 pass, 1 fail) and official patch reward 1 (60 pass). The
[source-bound result](../../docs/measurements/official_mini_flask_5014_grade_2026-09-25.json)
also records the host's setup repair and per-stage timings. This is one
task/one sample per arm, not an RL rollout or speedup measurement.

A full RealWorldEnv coding adapter still needs a way to prepare Boltons or
Humanize within upstream MiniSandbox. The pinned deployment's constructor
maps Python versions from SWE-bench/SWE-smith task metadata, and neither of
our task repos is in that map. It also needs a persistent host-to-session
action transport and a submitted-workspace handoff to our host-private
14-case grader. This probe does not treat a SWE-bench placeholder dataset as
that integration.
