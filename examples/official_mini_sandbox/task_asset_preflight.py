"""Audit local assets for one real, pinned SWE-bench Verified MiniSandbox task.

This is a read-only feasibility check. It neither prepares a task nor grades a
patch. The private test patch is read only to confirm it exists, and is never
included in the report. In training, keep the dataset and grader host-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from examples.official_mini_sandbox.check_linux_session import (
    PINNED_SOURCE_TREE_SHA256,
    _check_pinned_source,
)


UPSTREAM_SHA = "381ada53ab35dadb342add33ff006f3157c22fb7"
INSTANCE_ID = "pallets__flask-5014"
REPO = "pallets/flask"
BASE_COMMIT = "7ee9ceb71e868944a46e1ff00b506772a53a4f1d"
VERSION = "2.3"
PYTHON_VERSION = "3.11"
DATASET = Path("dataset/SWE-bench/SWE-bench_Verified/data/test-00000-of-00001.parquet")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(upstream_root: Path, cache_root: Path) -> dict:
    upstream_root = upstream_root.resolve(strict=True)
    cache_root = cache_root.resolve()
    _check_pinned_source(upstream_root)
    dataset = upstream_root / DATASET
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("Install pyarrow to inspect the local task parquet") from exc
    table = parquet.read_table(dataset)
    matches = [row for row in table.to_pylist() if row["instance_id"] == INSTANCE_ID]
    if len(matches) != 1:
        raise ValueError("Expected exactly one real SWE-bench Verified instance")
    row = matches[0]
    if (row["repo"], row["base_commit"], row["version"]) != (
        REPO, BASE_COMMIT, VERSION,
    ):
        raise ValueError("Instance metadata differs from the pinned task")
    if not row["patch"] or not row["test_patch"] or not row["FAIL_TO_PASS"]:
        raise ValueError("Pinned task is missing its solution or verification fields")
    git_rel = Path("git") / REPO / VERSION / INSTANCE_ID / "testbed.tar.gz"
    venv_rel = Path("shared_venv") / REPO / VERSION / "default" / "venv.tar.gz"
    python_rel = Path("conda") / PYTHON_VERSION / "miniconda3/bin/python"
    required = {
        "git_checkout_cache": git_rel,
        "versioned_venv_cache": venv_rel,
        "python_3_11_base_runtime": python_rel,
    }
    assets = {
        name: {"relative_path": path.as_posix(), "present": (cache_root / path).is_file()}
        for name, path in required.items()
    }
    source = cache_root / "source/flask"
    source_commit = None
    source_clean = False
    if (source / ".git").is_dir():
        result = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
        )
        if result.returncode == 0:
            source_commit = result.stdout.strip()
            status = subprocess.run(
                ["git", "-C", str(source), "status", "--porcelain=v1"],
                capture_output=True, text=True, check=False,
            )
            source_clean = status.returncode == 0 and not status.stdout.strip()
    tools = upstream_root / "SWE-agent/tools/terminal/config.yaml"
    return {
        "scope": "read_only_official_task_asset_preflight_no_grade",
        "upstream_git_sha": UPSTREAM_SHA,
        "upstream_python_tree_sha256": PINNED_SOURCE_TREE_SHA256,
        "preflight_script_sha256": _sha256(Path(__file__)),
        "dataset_sha256": _sha256(dataset),
        "instance_id": INSTANCE_ID,
        "repo": REPO,
        "base_commit": BASE_COMMIT,
        "version": VERSION,
        "official_python_version": PYTHON_VERSION,
        "official_terminal_bundle_present": tools.is_file(),
        "public_source_checkout": {
            "relative_path": "source/flask",
            "base_commit_verified": source_commit == BASE_COMMIT and source_clean,
        },
        "required_cache_assets": assets,
        "all_declared_cache_files_present": tools.is_file() and all(
            asset["present"] for asset in assets.values()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.upstream_root, args.cache_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
