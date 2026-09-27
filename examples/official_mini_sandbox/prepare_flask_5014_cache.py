"""Prepare the pinned Flask task's Linux Python 3.11 cache outside its chroot.

The official MiniSandbox environment builder was observed to cache a venv
without Flask/pytest/chardet when its multiline installation action encountered
guest DNS failure. This explicit trusted-host preparation uses the unmodified
upstream requirement generator and version pins, then validates imports before
atomically replacing that invalid cache. It is an operator repair, not a claim
that upstream's unmodified setup succeeds on this host.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

from examples.official_mini_sandbox.check_linux_session import _check_pinned_source
from examples.official_mini_sandbox.run_official_flask_5014 import (
    _prepare_official_cache,
    _task_row,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_logged(argv: list[str], log_path: Path, *, env: dict[str, str]) -> None:
    with log_path.open("a", encoding="utf-8") as log:
        log.write("command: " + " ".join(argv) + "\n")
        log.flush()
        result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, env=env)
    if result.returncode:
        raise RuntimeError(f"Cache preparation command failed ({result.returncode}); inspect {log_path}")


def prepare(upstream: Path, dataset: Path, source_base: Path, source_env: Path,
            cache: Path, *, requirements_only: bool = False) -> dict:
    if sys.platform != "linux":
        raise RuntimeError("Prepare in Linux using the target Python 3.11 runtime")
    _check_pinned_source(upstream)
    row = _task_row(dataset)
    git_info = _prepare_official_cache(cache, source_base, source_env)
    requirements_fn = importlib.import_module("swesandbox.swebench_utils.python")
    constants = importlib.import_module("swebench.harness.constants")
    requirements = requirements_fn.get_requirements(row)
    req_file = cache / "official_requirements_flask_5014.txt"
    req_file.write_text(requirements, encoding="utf-8")
    pins = constants.MAP_REPO_VERSION_TO_SPECS["pallets/flask"]["2.3"]["pip_packages"]
    result = {
        "scope": "operator_prepared_official_flask_task_cache",
        "requirements_sha256": _sha(req_file),
        "requirements_noncomment_lines": sum(bool(x.strip()) and not x.lstrip().startswith("#")
                                         for x in requirements.splitlines()),
        "official_pip_pins": pins,
        **git_info,
    }
    if requirements_only:
        return result

    venv_source = cache / "shared_venv/pallets/flask/2.3/default/venv"
    venv_tar = venv_source.with_suffix(".tar.gz")
    if venv_source.exists():
        shutil.rmtree(venv_source)
    python311 = cache / "conda/3.11/miniconda3/bin/python"
    subprocess.run([str(python311), "-m", "venv", str(venv_source)], check=True)
    vpy = venv_source / "bin/python"
    log_path = cache / "operator_env_prepare.log"
    log_path.write_text("", encoding="utf-8")
    env = dict(os.environ, PIP_INDEX_URL="https://pypi.org/simple",
               PIP_DISABLE_PIP_VERSION_CHECK="1")
    # The host's /opt/deps contains Python 3.12 wheels for MiniSandbox itself;
    # never let pip or the target Python 3.11 venv satisfy task requirements
    # from that external tree.
    env.pop("PYTHONPATH", None)
    _run_logged([str(vpy), "-m", "pip", "install", "-r", str(req_file)], log_path, env=env)
    _run_logged([str(vpy), "-m", "pip", "install", *pins], log_path, env=env)
    install_copy = cache / "public_source_install"
    if install_copy.exists():
        shutil.rmtree(install_copy)
    shutil.copytree(source_base, install_copy, ignore=shutil.ignore_patterns(".git"))
    try:
        _run_logged([str(vpy), "-m", "pip", "install", str(install_copy)], log_path, env=env)
    finally:
        shutil.rmtree(install_copy)
    _run_logged([str(vpy), "-c", "import flask, pytest, chardet, werkzeug"], log_path, env=env)
    wheelhouse = cache / "wheelhouse"
    wheelhouse.mkdir(parents=True, exist_ok=True)
    _run_logged([str(vpy), "-m", "pip", "download", "--no-deps", "--dest", str(wheelhouse),
                 "setuptools==70.0.0", "chardet==5.1.0", "wheel==0.40.0"],
                log_path, env=env)
    temp_tar = venv_tar.with_name("venv.tar.gz.tmp")
    with tarfile.open(temp_tar, "w:gz") as tar:
        tar.add(venv_source, arcname="")
    temp_tar.replace(venv_tar)
    shutil.rmtree(venv_source)
    result.update({
        "venv_cache_sha256": _sha(venv_tar),
        "venv_cache_bytes": venv_tar.stat().st_size,
        "wheelhouse_files": sorted(path.name for path in wheelhouse.iterdir() if path.is_file()),
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, default=Path("/src"))
    parser.add_argument("--dataset", type=Path, default=Path("/private/verified.parquet"))
    parser.add_argument("--source-base", type=Path, default=Path("/private/base_source"))
    parser.add_argument("--source-env", type=Path, default=Path("/private/env_source"))
    parser.add_argument("--cache-root", type=Path, default=Path("/work/cache"))
    parser.add_argument("--requirements-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.upstream_root, args.dataset, args.source_base,
                             args.source_env, args.cache_root,
                             requirements_only=args.requirements_only), sort_keys=True))


if __name__ == "__main__":
    main()
