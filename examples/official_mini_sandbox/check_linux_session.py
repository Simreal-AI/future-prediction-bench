"""Exercise the pinned upstream MiniSandbox terminal API inside Linux.

This is a direct upstream API and isolation smoke, not task preparation or
graded RealWorldEnv integration. Run it only inside a disposable, privileged
Linux container with the upstream checkout mounted at /src and its Python
dependencies installed. The container should have networking disabled.
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
import sys
import tempfile
from types import SimpleNamespace

from examples.official_mini_sandbox.session_bridge import run_in_official_session


PINNED_UPSTREAM_SHA = "381ada53ab35dadb342add33ff006f3157c22fb7"
PINNED_SOURCE_TREE_SHA256 = "605342a1abd452ad04b8f9d00deb993d9b73c4afac61a268f229816af63b9d84"
SOURCE_TREES = (
    "sandboxdev/swesandbox",
    "SWE-ReX/src/swerex",
    "SWE-bench/swebench",
    "R2E-Gym/src/r2egym",
    "SWE-smith/swesmith",
)


def _check_pinned_source(upstream_root: Path) -> None:
    # Read Git's symbolic HEAD directly: the cached Linux image intentionally
    # has no Git binary. The source digest also detects edits in Python files.
    git_dir = upstream_root / ".git"
    head = (git_dir / "HEAD").read_text(encoding="ascii").strip()
    if head.startswith("ref: "):
        ref = head[5:]
        if not ref.startswith("refs/") or ".." in Path(ref).parts:
            raise RuntimeError("Invalid upstream Git HEAD reference")
        sha = (git_dir / ref).read_text(encoding="ascii").strip()
    else:
        sha = head
    if sha != PINNED_UPSTREAM_SHA:
        raise RuntimeError(f"Upstream checkout must be {PINNED_UPSTREAM_SHA}")
    paths = sorted(
        path for tree in SOURCE_TREES
        for path in (upstream_root / tree).rglob("*.py")
        if "__pycache__" not in path.parts
    )
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(upstream_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    if digest.hexdigest() != PINNED_SOURCE_TREE_SHA256:
        raise RuntimeError("Pinned upstream Python source differs from the tested checkout")


async def _probe(upstream_root: Path, conda_env: Path, workspace: Path) -> dict:
    # The pinned checkout has a circular import when the sandbox module loads
    # before SWE-ReX's config registration.
    importlib.import_module("swerex.deployment.config")
    deployment_module = importlib.import_module("swesandbox.sandbox_deployment")
    runtime_module = importlib.import_module("swerex.runtime.abstract")
    constants = importlib.import_module("swebench.harness.constants")

    repo = "redis/redis"
    version = next(iter(constants.MAP_REPO_VERSION_TO_SPECS[repo]))
    tool_path = upstream_root / "SWE-agent" / "tools"
    if not (tool_path / "terminal" / "config.yaml").is_file():
        raise RuntimeError("Pinned upstream terminal bundle is missing")
    if not conda_env.is_dir():
        raise RuntimeError("The container Python environment to mount is missing")

    # The config demands a bundle; this minimal shape is sufficient for a
    # session-only probe. It does not install tools or build the task venv.
    terminal_bundle = SimpleNamespace(path=tool_path / "terminal")
    config = deployment_module.SandboxDeploymentConfig(
        ds={"repo": repo, "version": version, "instance_id": "official-api-probe"},
        root_base=str(workspace),
        root_dir=str(workspace / "sandbox"),
        shared_venv=str(workspace / "shared"),
        git_base_path=str(workspace / "git"),
        tool_path=str(tool_path),
        conda_env=str(conda_env),
        bundles=[terminal_bundle],
    )
    deployment = config.get_deployment()
    try:
        await deployment.start()
        host_mount_namespace = os.readlink("/proc/self/ns/mnt")
        await deployment.runtime.create_session(
            runtime_module.CreateSandboxBashSessionRequest(
                startup_timeout=15, startup_cmd=deployment.startup()
            )
        )
        output = await run_in_official_session(
            deployment,
            operator_command=(
                "test ! -e /src && printf 'inside-mini-ok\\n' && "
                "stat -f -c %T /tmp && readlink /proc/self/ns/mnt && "
                "touch /tmp/only_inside"
            ),
            timeout_seconds=10,
        )
        fields = output.strip().splitlines()
        if len(fields) != 3 or fields[0] != "inside-mini-ok" or fields[1] != "tmpfs":
            raise AssertionError(f"Unexpected upstream session observation: {output!r}")
        if not re.fullmatch(r"mnt:\[\d+\]", fields[2]):
            raise AssertionError(f"Mount namespace observation missing: {output!r}")
        if fields[2] == host_mount_namespace:
            raise AssertionError("Upstream session did not enter a new mount namespace")
        if (workspace / "sandbox" / "tmp" / "only_inside").exists():
            raise AssertionError("Guest /tmp marker leaked to the host sandbox tree")
        return {
            "upstream_sha": PINNED_UPSTREAM_SHA,
            "upstream_python_tree_sha256": PINNED_SOURCE_TREE_SHA256,
            "upstream_deployment_class": type(deployment).__module__,
            "operator_output": fields[0],
            "host_mount_namespace": host_mount_namespace,
            "session_mount_namespace": fields[2],
            "private_tmpfs": True,
            "host_only_source_hidden": True,
            "guest_tmp_marker_absent_on_host": True,
            "scope": "official_terminal_isolation_smoke_only",
        }
    finally:
        runtime = getattr(deployment, "_runtime", None)
        try:
            if runtime is not None:
                await runtime.close()
        finally:
            # Upstream's __del__ calls asyncio.run(self.stop()). Clearing an
            # already closed runtime avoids a second asynchronous cleanup in
            # the active event loop. main() removes the disposable workspace.
            deployment._runtime = None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, default=Path("/src"))
    parser.add_argument("--conda-env", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "linux" or os.geteuid() != 0:
        raise RuntimeError("Run inside a disposable privileged Linux container as root")
    upstream_root = args.upstream_root.resolve()
    if upstream_root != Path("/src"):
        raise RuntimeError("Mount the pinned upstream checkout at /src for the isolation probe")
    _check_pinned_source(upstream_root)
    workspace = Path(tempfile.mkdtemp(prefix="official-mini-probe-"))
    try:
        report = asyncio.run(_probe(upstream_root, args.conda_env, workspace))
        print(json.dumps(report, sort_keys=True))
    finally:
        shutil.rmtree(workspace)


if __name__ == "__main__":
    main()
