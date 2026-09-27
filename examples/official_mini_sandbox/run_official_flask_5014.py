"""Run one pinned SWE-bench Verified task through upstream MiniSandbox.

This is a host-private integration experiment, not a policy rollout. The task
parquet and environment source are mounted outside the actor's chroot. The
actor deployment receives redacted task metadata. After it is closed, two
fresh verifier deployments evaluate the unmodified base and the official
reference patch. Neither the test patch nor test output is written to the
public report.

Run only in a disposable, privileged Linux container with the pinned upstream
checkout and a Python 3.11 runtime. Network is used only for first-time pip
environment preparation; it is not needed for the actor's commands.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shutil
from types import SimpleNamespace
import sys
import tarfile
import time

from examples.official_mini_sandbox.check_linux_session import _check_pinned_source
from examples.official_mini_sandbox.strict_grade import grade_flask_5014
from examples.official_mini_sandbox.task_asset_preflight import (
    BASE_COMMIT,
    DATASET,
    INSTANCE_ID,
    REPO,
    UPSTREAM_SHA,
    VERSION,
)
from examples.official_mini_sandbox.session_bridge import run_in_official_session


ENV_SETUP_COMMIT = "182ce3dd15dfa3537391c3efaf9c3ff407d134d4"
EXPECTED_DATASET_SHA256 = "43ed5a3d1d98da36472c1ade65ddd2085d7b4ff694fcaf6a023a07c5c1f32f21"


class _NoOpHooks:
    def __getattr__(self, name):
        if not name.startswith("on_"):
            raise AttributeError(name)
        return lambda *args, **kwargs: None


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(path: Path) -> str:
    import subprocess
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def _task_row(dataset: Path) -> dict:
    import pyarrow.parquet as parquet
    if _sha(dataset) != EXPECTED_DATASET_SHA256:
        raise RuntimeError("Official task parquet digest mismatch")
    rows = [row for row in parquet.read_table(dataset).to_pylist()
            if row["instance_id"] == INSTANCE_ID]
    if len(rows) != 1:
        raise RuntimeError("Expected one official Flask task")
    row = rows[0]
    if (row["repo"], row["base_commit"], row["version"],
            row["environment_setup_commit"]) != (
                REPO, BASE_COMMIT, VERSION, ENV_SETUP_COMMIT):
        raise RuntimeError("Official task metadata differs")
    if not row.get("patch") or not row.get("test_patch"):
        raise RuntimeError("Official patches missing")
    return row


def _prepare_official_cache(cache: Path, source_base: Path, source_env: Path) -> dict:
    if _git_head(source_base) != BASE_COMMIT or _git_head(source_env) != ENV_SETUP_COMMIT:
        raise RuntimeError("Public Flask source checkouts are not pinned")
    cache.mkdir(parents=True, exist_ok=True)
    # The pinned upstream get_requirements_by_commit reads this exact local
    # cache location. The environment commit stays outside the actor chroot.
    request_cache = Path("/tmp/swe_repos") / f"pallets__flask__{ENV_SETUP_COMMIT}"
    request_cache.parent.mkdir(parents=True, exist_ok=True)
    if request_cache.is_symlink():
        if request_cache.resolve() != source_env.resolve():
            raise RuntimeError("Conflicting upstream request cache")
    elif request_cache.exists():
        if _git_head(request_cache) != ENV_SETUP_COMMIT:
            raise RuntimeError("Conflicting upstream request checkout")
    else:
        request_cache.symlink_to(source_env.resolve(), target_is_directory=True)

    git_tar = cache / "git" / REPO / VERSION / INSTANCE_ID / "testbed.tar.gz"
    if not git_tar.exists():
        git_tar.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(git_tar, "w:gz") as tar:
            tar.add(source_base, arcname="")
    py_bin = cache / "conda/3.11/miniconda3/bin/python"
    py_bin.parent.mkdir(parents=True, exist_ok=True)
    if not py_bin.exists():
        py_bin.symlink_to("/usr/bin/python3.11")
    if os.path.realpath(py_bin) != "/usr/bin/python3.11":
        raise RuntimeError("Official Python 3.11 cache points elsewhere")
    return {"git_cache_sha256": _sha(git_tar), "git_cache_bytes": git_tar.stat().st_size}


def _make_deployment(ds: dict, role: str, *, upstream: Path, cache: Path):
    importlib.import_module("swerex.deployment.config")
    sandbox = importlib.import_module("swesandbox.sandbox_deployment")
    terminal = upstream / "SWE-agent/tools/terminal"
    if not (terminal / "config.yaml").is_file():
        raise RuntimeError("Official terminal bundle missing")
    config = sandbox.SandboxDeploymentConfig(
        ds=ds,
        data_type="swebench",
        root_base=str(cache / "sandboxes"),
        root_dir=str(cache / "sandboxes" / role),
        shared_venv=str(cache / "shared_venv"),
        git_base_path=str(cache / "git"),
        tool_path=str(upstream / "SWE-agent/tools"),
        conda_env=str(cache / "conda"),
        bundles=[SimpleNamespace(path=terminal)],
        cmd_list=["unset PIP_CONSTRAINT; export PIP_DISABLE_PIP_VERSION_CHECK=1 "
                  "PIP_NO_INDEX=1 PIP_FIND_LINKS=/wheels PYTHONPATH= "
                  "PAGER=cat GIT_PAGER=cat"],
    )
    deployment = config.get_deployment()
    return deployment


def _open_deployment(deployment, *, cache: Path) -> dict:
    runtime_types = importlib.import_module("swerex.runtime.abstract")
    upstream_utils = importlib.import_module("swesandbox.utils")
    # The official startup bind-mounts host /opt but not this sandbox-local
    # directory. Public wheels are copied here so its pip commands can run
    # without guest DNS; no task dataset or test patch enters the namespace.
    wheelhouse = cache / "wheelhouse"
    if not wheelhouse.is_dir() or not any(wheelhouse.iterdir()):
        raise RuntimeError("Prepared public wheelhouse is missing")
    shutil.copytree(wheelhouse, Path(deployment.root_dir) / "wheels", dirs_exist_ok=True)
    started = time.perf_counter()
    asyncio.run(deployment.start())
    asyncio.run(deployment.runtime.create_session(
        runtime_types.CreateSandboxBashSessionRequest(
            startup_timeout=60, startup_cmd=deployment.startup(),
        )
    ))
    session_seconds = time.perf_counter() - started
    repo_tar = cache / "git" / REPO / VERSION / INSTANCE_ID / "testbed.tar.gz"
    upstream_utils.tar_extract(str(repo_tar),
                               str(Path(deployment.root_dir) / "testbed"), threads=2)
    post_started = time.perf_counter()
    deployment.post_init(SimpleNamespace(_chook=_NoOpHooks()))
    post_seconds = time.perf_counter() - post_started
    vpy = "/work/cache/shared_venv/pallets/flask/2.3/default/venv/bin/python"
    package_check = asyncio.run(run_in_official_session(
        deployment,
        operator_command=(
            "test \"$PIP_NO_INDEX\" = 1 && "
            "test \"$PIP_FIND_LINKS\" = /wheels && "
            "test -z \"$PYTHONPATH\" && "
            "test \"$PAGER\" = cat && test \"$GIT_PAGER\" = cat && "
            "test -d /wheels && "
            f"{vpy} -c 'import flask, pytest, chardet, werkzeug; "
            "import importlib.metadata as m; "
            "print(\"FPB_PACKAGES_OK pytest=\"+m.version(\"pytest\")+"
            "\" chardet=\"+m.version(\"chardet\"))'"
        ),
        timeout_seconds=30,
    ))
    if "FPB_PACKAGES_OK pytest=7.3.0 chardet=5.1.0" not in package_check:
        raise RuntimeError("Prepared official task environment failed import/version check")
    return {"session_seconds": session_seconds, "post_init_seconds": post_seconds}


def _close_deployment(deployment) -> None:
    runtime = getattr(deployment, "_runtime", None)
    if runtime is not None:
        asyncio.run(runtime.close())
        deployment._runtime = None
    shutil.rmtree(deployment.root_dir, ignore_errors=True)


def run(upstream: Path, dataset: Path, source_base: Path,
        source_env: Path, cache: Path) -> dict:
    if sys.platform != "linux" or os.geteuid() != 0:
        raise RuntimeError("Requires root in a disposable privileged Linux container")
    for path, target in ((upstream, Path("/src")), (dataset.parent, Path("/private")),
                         (cache, Path("/work/cache"))):
        if path.resolve() != target:
            raise RuntimeError(f"Expected mount path {target}")
    _check_pinned_source(upstream)
    if INSTANCE_ID in importlib.import_module("swesandbox.swe_bench_instance_map").instance_to_skip:
        raise RuntimeError("Official task is in upstream's skip list")
    row = _task_row(dataset)
    cache_info = _prepare_official_cache(cache, source_base, source_env)
    public_ds = {key: row[key] for key in (
        "instance_id", "repo", "version", "base_commit",
        "environment_setup_commit", "problem_statement",
    )}
    public_ds.update(patch="", test_patch="", FAIL_TO_PASS="[]", PASS_TO_PASS="[]")

    actor = _make_deployment(public_ds, "actor", upstream=upstream, cache=cache)
    try:
        actor_setup = _open_deployment(actor, cache=cache)
        actor_observation = asyncio.run(run_in_official_session(
            actor,
            operator_command=(
                "test ! -e /private && test ! -e /src && "
                "test -z \"$(find /testbed -maxdepth 2 -name 'test_*5014*' -print -quit)\" && "
                f"cd /testbed && git merge-base --is-ancestor {BASE_COMMIT} HEAD && "
                f"test \"$(git rev-parse HEAD^{{tree}})\" = \"$(git rev-parse {BASE_COMMIT}^{{tree}})\" && "
                "git diff --quiet && "
                "printf 'FPB_ACTOR_HEAD=%s\\nFPB_ACTOR_TREE=%s\\n' "
                "\"$(git rev-parse HEAD)\" \"$(git rev-parse HEAD^{tree})\" && "
                "/work/cache/shared_venv/pallets/flask/2.3/default/venv/bin/python --version"
            ),
            timeout_seconds=30,
        ))
        actor_head = re.search(r"FPB_ACTOR_HEAD=([0-9a-f]{40})", actor_observation)
        actor_tree = re.search(r"FPB_ACTOR_TREE=([0-9a-f]{40})", actor_observation)
        if not actor_head or not actor_tree or "Python 3.11" not in actor_observation:
            raise RuntimeError("Actor task startup check failed")
    finally:
        _close_deployment(actor)

    grades = {}
    for role in ("baseline", "official_patch"):
        verifier = _make_deployment(row, role, upstream=upstream, cache=cache)
        try:
            setup = _open_deployment(verifier, cache=cache)
            apply_patch = importlib.import_module("swesandbox.utils").apply_patch
            if role == "official_patch":
                apply_patch(sandbox_root=verifier.root_dir, git_folder="testbed",
                            patch_str=row["patch"], instance_id=INSTANCE_ID)
            apply_patch(sandbox_root=verifier.root_dir, git_folder="testbed",
                        patch_str=row["test_patch"], instance_id=INSTANCE_ID)
            started = time.perf_counter()
            result = grade_flask_5014(verifier, timeout_seconds=300)
            grade_seconds = time.perf_counter() - started
            grades[role] = {
                "reward": result.reward,
                "expected_cases": result.expected_cases,
                "passed_cases": result.passed_cases,
                "failed_cases": result.failed_cases,
                "log_sha256": result.log_sha256,
                "grade_seconds": grade_seconds,
                **setup,
            }
        finally:
            _close_deployment(verifier)
    if grades["baseline"]["reward"] != 0.0 or grades["official_patch"]["reward"] != 1.0:
        raise RuntimeError("Official task did not discriminate baseline and reference patch")
    venv_tar = cache / "shared_venv" / REPO / VERSION / "default/venv.tar.gz"
    return {
        "scope": "single_official_flask_task_baseline_vs_reference_no_policy_rollout",
        "upstream_sha": UPSTREAM_SHA,
        "instance_id": INSTANCE_ID,
        "base_commit": BASE_COMMIT,
        "environment_setup_commit": ENV_SETUP_COMMIT,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "actor_hidden_mounts_absent": True,
        "actor_task_metadata_redacted": True,
        "noninteractive_pager": "cat",
        "actor_setup_head": actor_head.group(1),
        "actor_setup_tree": actor_tree.group(1),
        "actor_session_seconds": actor_setup["session_seconds"],
        "actor_post_init_seconds": actor_setup["post_init_seconds"],
        "cache": {**cache_info,
                  "venv_cache_sha256": _sha(venv_tar),
                  "venv_cache_bytes": venv_tar.stat().st_size},
        "grades": grades,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, default=Path("/src"))
    parser.add_argument("--dataset", type=Path, default=Path("/private/verified.parquet"))
    parser.add_argument("--source-base", type=Path, default=Path("/private/base_source"))
    parser.add_argument("--source-env", type=Path, default=Path("/private/env_source"))
    parser.add_argument("--cache-root", type=Path, default=Path("/work/cache"))
    args = parser.parse_args()
    print(json.dumps(run(args.upstream_root, args.dataset, args.source_base,
                         args.source_env, args.cache_root), sort_keys=True))


if __name__ == "__main__":
    main()
