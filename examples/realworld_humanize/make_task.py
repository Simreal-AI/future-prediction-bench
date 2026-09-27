"""Build a SHA-pinned public Humanize 4.15.0 coding task and host verifier.

The defect and repair are documented in upstream pull request #329. This is a
public solved repair fixture for environment integration, not a held-out task.
The task builder source is public; the verifier files remain on the trusted
host and are never mounted in the policy container during an episode.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import tarfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.request import urlopen

from future_prediction_bench.coding_env import _workspace_digest


VERSION = "4.15.0"
SOURCE_COMMIT = "2ddb5903cdc1c7e6eb6b083f4f99f73db50aecd9"
ARCHIVE_SHA256 = "1dd098483eb1c7ee8e32eb2e99ad1910baefa4b75c3aff3a82f4d78688993b10"
ARCHIVE_URL = (
    "https://files.pythonhosted.org/packages/ba/66/"
    "a3921783d54be8a6870ac4ccffcd15c4dc0dd7fcce51c6d63b8c63935276/"
    "humanize-4.15.0.tar.gz"
)
UPSTREAM_REPAIR = "https://github.com/python-humanize/humanize/pull/329"
SOURCE_FILE = "src/humanize/filesize.py"
VISIBLE_CHECK = (
    "python3", "-B", "-c",
    "import sys; sys.path.insert(0, 'src'); "
    "from humanize import naturalsize; print(naturalsize(1000))",
)


def _json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def _archive_bytes(sdist):
    if sdist is not None:
        with Path(sdist).open("rb") as source:
            data = source.read(2_000_001)
    else:
        with urlopen(ARCHIVE_URL, timeout=30) as response:
            data = response.read(2_000_001)
    if len(data) > 2_000_000 or hashlib.sha256(data).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("Humanize 4.15.0 source archive SHA-256/size mismatch")
    return data


def _extract(data, seed):
    root = f"humanize-{VERSION}"
    total = 0
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        members = archive.getmembers()
        if not 1 <= len(members) <= 500:
            raise ValueError("Unexpected source archive member count")
        for member in members:
            parts = Path(member.name).parts
            if (not parts or parts[0] != root or ".." in parts
                    or not (member.isdir() or member.isfile())):
                raise ValueError("Unsafe Humanize archive member")
            target = seed.joinpath(*parts[1:])
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            total += member.size
            if total > 10_000_000:
                raise ValueError("Source archive exceeds unpacked size limit")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
    if not (seed / "LICENCE").is_file() or not (seed / SOURCE_FILE).is_file():
        raise ValueError("Pinned Humanize source and license are missing")


def _repair(original):
    anchor = "    exp = int(min(log(abs_bytes, base), len(suffix)))\n"
    if original.count(anchor) != 1:
        raise ValueError("Pinned filesize.py no longer has the upstream repair anchor")
    guard = (
        "    # Carry a rounded mantissa into the next available unit.\n"
        "    if exp < len(suffix) and abs(float(format % (abs_bytes / (base**exp)))) >= base:\n"
        "        exp += 1\n"
    )
    repaired = original.replace(anchor, anchor + guard)
    if len(repaired.encode("utf-8")) > 65_536:
        raise ValueError("Repair exceeds the adapter's write-file limit")
    return repaired


def _cases():
    # Values and expected outputs are host-authored. Their files stay outside
    # the policy workspace even though this fixture builder is public.
    return [
        (999_999, {}, "1.0 MB"),
        (999_999_999, {}, "1.0 GB"),
        (999_999_999_999, {}, "1.0 TB"),
        (1024**2 - 1, {"binary": True}, "1.0 MiB"),
        (1024**3 - 1, {"binary": True}, "1.0 GiB"),
        (1024**2 - 1, {"gnu": True}, "1.0M"),
        (-999_999, {}, "-1.0 MB"),
        (-1024**2 + 1, {"binary": True}, "-1.0 MiB"),
        (999_999, {"format": "%.2f"}, "1.00 MB"),
        (999_999, {"format": "%.3f"}, "999.999 kB"),
        (999_500, {}, "999.5 kB"),
        (1_000, {}, "1.0 kB"),
        (1_024, {"binary": True}, "1.0 KiB"),
        (1_000_000, {}, "1.0 MB"),
    ]


def make_task(output, *, sdist=None):
    output = Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output directory must be new or empty")
    data = _archive_bytes(sdist)
    seed, verifier = output / "seed", output / "verifier"
    seed.mkdir(parents=True)
    verifier.mkdir(parents=True)
    _extract(data, seed)
    seed_workspace_sha256 = _workspace_digest(seed)
    (output / f"humanize-{VERSION}.tar.gz").write_bytes(data)

    cases = []
    for value, options, expected in _cases():
        code = ("import sys; sys.path.insert(0, 'src'); "
                "from humanize import naturalsize; "
                f"print(naturalsize({value!r}, **{options!r}))")
        cases.append({"argv": ["python3", "-B", "-c", code],
                      "expected_stdout": expected + "\n", "expected_returncode": 0})
    _json(verifier / "verify.json", {"kind": "command_cases_v1", "cases": cases})

    original = (seed / SOURCE_FILE).read_text(encoding="utf-8")
    repaired = _repair(original)
    now = datetime.now(timezone.utc)
    task = {
        "schema_version": "realworld-0.1",
        "task_id": "humanize-4150-naturalsize-rounding-v1",
        "event_id": "humanize-4150-naturalsize-rounding",
        "cluster_id": "humanize-filesize-naturalsize",
        "split": "train",
        "prompt": (
            "In the pinned Humanize 4.15.0 repository, naturalsize() can print "
            "a rounded mantissa equal to the unit base while keeping the smaller "
            "suffix (for example, 999999 bytes becomes '1000.0 kB'). Repair the "
            "unit rollover for decimal, binary, and GNU modes without breaking "
            "ordinary values; then submit the workspace."
        ),
        "issued_at": (now - timedelta(minutes=1)).isoformat(),
        "action_deadline": (now + timedelta(minutes=30)).isoformat(),
        "outcome_not_before": (now - timedelta(minutes=1)).isoformat(),
        "verify_after": (now - timedelta(minutes=1)).isoformat(),
        "tool_manifest": [
            {"name": "read_file", "description": "Read a repository file."},
            {"name": "write_file", "description": "Replace a repository file."},
            {"name": "run_visible_checks", "description": "Run the fixed public import check."},
            {"name": "submit", "description": "Freeze and submit the repository workspace."},
        ],
        "reward_contract": {
            "id": "humanize4150-naturalsize-host-cases-v1",
            "description": "One point if every host-side command case matches its expected output.",
            "min_reward": 0, "max_reward": 1,
        },
        "budgets": {"max_actions": 8, "max_wall_seconds": 900},
        "is_fixture": True,
        "adapter_id": "docker_coding", "adapter_version": "0.1",
        "metadata": {
            "provenance": "public_upstream_repair_example",
            "repository": "https://github.com/python-humanize/humanize",
            "source_tag": VERSION,
            "source_commit": SOURCE_COMMIT,
            "source_sdist_sha256": ARCHIVE_SHA256,
            "source_workspace_sha256": seed_workspace_sha256,
            "upstream_fix": UPSTREAM_REPAIR,
            "license": "MIT",
        },
    }
    _json(output / "task.json", task)
    (output / "actions.baseline.jsonl").write_text('{"action":"submit"}\n', encoding="utf-8")
    actions = [
        {"action": "read_file", "path": SOURCE_FILE},
        {"action": "write_file", "path": SOURCE_FILE, "content": repaired},
        {"action": "run_visible_checks"},
        {"action": "submit"},
    ]
    (output / "actions.solution.jsonl").write_text(
        "".join(json.dumps(action, ensure_ascii=False) + "\n" for action in actions),
        encoding="utf-8")
    return {"task_id": task["task_id"], "source_commit": SOURCE_COMMIT,
            "source_sdist_sha256": ARCHIVE_SHA256,
            "baseline_actions": str(output / "actions.baseline.jsonl"),
            "solution_actions": str(output / "actions.solution.jsonl"),
            "host_cases": len(cases),
            "classification": "public_solved_repository_integration_fixture"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sdist", help="Optional local SHA-pinned Humanize source archive")
    args = parser.parse_args()
    print(json.dumps(make_task(args.output, sdist=args.sdist), indent=2))
