"""Per-case tmpfs copy for a two-layer cooperative OverlayFS branch.

Linux limits nested OverlayFS depth on this guest. The frozen workspace and
episode branch already use two layers, so a third per-case overlay fails. This
runner keeps the resident helper's PID isolation, unprivileged execution,
resource limits, deadlines, stdout cap, and cleanup, but substitutes a bounded
workspace copy on the branch's tmpfs for the per-case overlay.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path


def _load_resident():
    spec = importlib.util.spec_from_file_location(
        "fpb_resident_case_runner_for_cooperative", "/fpb_resident_case_runner.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("resident_case_runner_missing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


resident = _load_resident()


def _namespace_case(libc, code, case_pool):
    resident._mount(libc, None, "/", flags=resident.MS_REC | resident.MS_PRIVATE)
    target = Path(case_pool) / "merged"
    # Both pinned fixtures are symlink-free and below 2 MiB. Humanize has 197
    # tree entries before the two cooperative markers, so retain a finite cap
    # while allowing its exact source tree.
    source = Path("/workspace")
    files = list(source.rglob("*"))
    if (len(files) > 256
            or any(path.is_symlink() or not (path.is_file() or path.is_dir())
                   for path in files)
            or sum(path.stat().st_size for path in files if path.is_file()) >
               2 * 1024 * 1024):
        raise RuntimeError("cooperative_case_workspace_exceeds_pinned_bound")
    shutil.copytree(source, target, dirs_exist_ok=True, symlinks=False)
    resident._mount(libc, os.fsencode(target), "/workspace",
                    flags=resident.MS_BIND | resident.MS_REC)
    Path("/workspace").chmod(0o777)
    return resident._capture(code)


resident._namespace_case = _namespace_case
run_one = resident.run_one
