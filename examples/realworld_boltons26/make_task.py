"""Build a reproducible, public real-repository coding integration task.

The upstream defect is already fixed publicly. This example checks environment
integrity and grading, not generalization to unseen repair tasks.
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
from urllib.parse import urlparse
from urllib.request import urlopen


VERSION = "26.0.0"
ARCHIVE_SHA256 = "5566d6cfd5a1e873d8e8476496287a9f92979964611ad9a9cecb6b0ef29b1edd"
SOURCE_COMMIT = "fb464991b718ca7bfabc14555c2947f25e7c79c9"
METADATA_URL = f"https://pypi.org/pypi/boltons/{VERSION}/json"


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _archive_bytes(local_path):
    if local_path is not None:
        return Path(local_path).read_bytes()
    with urlopen(METADATA_URL, timeout=15) as response:
        metadata = json.load(response)
    releases = [entry for entry in metadata["urls"] if entry.get("packagetype") == "sdist"
                and entry.get("filename") == f"boltons-{VERSION}.tar.gz"
                and entry.get("digests", {}).get("sha256") == ARCHIVE_SHA256]
    if len(releases) != 1:
        raise ValueError("Pinned source distribution is unavailable")
    url = releases[0]["url"]
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "files.pythonhosted.org":
        raise ValueError("Unexpected source distribution host")
    with urlopen(url, timeout=30) as response:
        data = response.read(5_000_001)
    if len(data) > 5_000_000:
        raise ValueError("Source distribution exceeds size limit")
    return data


def _extract_pinned_archive(data, seed):
    if hashlib.sha256(data).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("Source distribution SHA-256 mismatch")
    expected_root = f"boltons-{VERSION}"
    total = 0
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        members = archive.getmembers()
        if not 1 <= len(members) <= 500:
            raise ValueError("Unexpected source distribution file count")
        for member in members:
            parts = Path(member.name).parts
            if (not parts or parts[0] != expected_root or ".." in parts
                    or not (member.isfile() or member.isdir())):
                raise ValueError("Unsafe source distribution member")
            target = seed.joinpath(*parts[1:])
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            total += member.size
            if total > 20_000_000:
                raise ValueError("Source distribution exceeds unpacked size limit")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
    if not (seed / "LICENSE").is_file() or not (seed / "boltons" / "strutils.py").is_file():
        raise ValueError("Pinned repository files are missing")


def make_task(output, *, sdist=None):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be new or empty")
    data = _archive_bytes(sdist)
    if hashlib.sha256(data).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("Source distribution SHA-256 mismatch")
    seed = output / "seed"
    verifier = output / "verifier"
    seed.mkdir(parents=True)
    verifier.mkdir(parents=True)
    _extract_pinned_archive(data, seed)
    (output / f"boltons-{VERSION}.tar.gz").write_bytes(data)

    checks = [("glass", "glass"), ("glasses", "glass"),
              ("boss", "boss"), ("bosses", "boss"),
              ("class", "class"), ("classes", "class"),
              ("kiss", "kiss"), ("kisses", "kiss"),
              ("address", "address"), ("addresses", "address"),
              ("business", "business"), ("businesses", "business"),
              ("GLASS", "GLASS"), ("Glasses", "Glass")]
    cases = [{"argv": ["python3", "-B", "-c",
                       f"from boltons.strutils import singularize; print(singularize({word!r}))"],
              "expected_stdout": expected + "\n", "expected_returncode": 0}
             for word, expected in checks]
    _write_json(verifier / "verify.json", {"kind": "command_cases_v1", "cases": cases})

    now = datetime.now(timezone.utc)
    task = {
        "schema_version": "realworld-0.1", "task_id": "boltons-26-singularize-ss-v1",
        "event_id": "boltons-26-singularize-ss", "cluster_id": "boltons-strutils-singularize",
        "split": "train",
        "prompt": "In the pinned Boltons repository, singularize() incorrectly removes the final s from already singular words ending in ss (for example, glass). Repair this without breaking existing plural forms such as glasses, and submit the workspace.",
        "issued_at": (now - timedelta(minutes=1)).isoformat(),
        "action_deadline": (now + timedelta(minutes=30)).isoformat(),
        "outcome_not_before": (now - timedelta(minutes=1)).isoformat(),
        "verify_after": (now - timedelta(minutes=1)).isoformat(),
        "tool_manifest": [{"name": name, "description": description} for name, description in (
            ("list_files", "List repository files."), ("read_file", "Read a repository file."),
            ("write_file", "Replace a repository file."),
            ("run_visible_checks", "Run the fixed visible import check."),
            ("submit", "Freeze and submit the repository workspace."))],
        "reward_contract": {"id": "boltons26-singularize-host-cases-v1",
                            "description": "One point if every held-out command case matches its expected result.",
                            "min_reward": 0, "max_reward": 1},
        "budgets": {"max_actions": 8, "max_wall_seconds": 900},
        "is_fixture": True, "adapter_id": "docker_coding", "adapter_version": "0.1",
        "metadata": {"provenance": "public_upstream_repair_example",
                     "repository": "https://github.com/mahmoud/boltons",
                     "source_commit": SOURCE_COMMIT,
                     "source_sdist_sha256": ARCHIVE_SHA256,
                     "upstream_fix": "https://github.com/mahmoud/boltons/pull/418",
                     "license": "BSD-3-Clause"},
    }
    _write_json(output / "task.json", task)
    (output / "actions.baseline.jsonl").write_text('{"action":"submit"}\n', encoding="utf-8")

    original = (seed / "boltons" / "strutils.py").read_text(encoding="utf-8")
    anchor = "    else:\n        singular = word[:-1]\n    return _match_case(orig_word, singular)"
    replacement = "    elif word.endswith('ss'):\n        singular = word\n" + anchor
    if original.count(anchor) != 1:
        raise ValueError("Pinned source no longer matches repair example")
    repaired = original.replace(anchor, replacement)
    actions = [{"action": "read_file", "path": "boltons/strutils.py"},
               {"action": "write_file", "path": "boltons/strutils.py", "content": repaired},
               {"action": "run_visible_checks"}, {"action": "submit"}]
    (output / "actions.solution.jsonl").write_text(
        "".join(json.dumps(action, ensure_ascii=False) + "\n" for action in actions), encoding="utf-8")
    return {"source_commit": SOURCE_COMMIT, "source_sdist_sha256": ARCHIVE_SHA256,
            "task": str(output / "task.json"), "seed": str(seed),
            "verifier": str(verifier), "baseline_actions": str(output / "actions.baseline.jsonl"),
            "solution_actions": str(output / "actions.solution.jsonl"),
            "classification": "public_real_repository_integration_fixture"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sdist", help="Optional already downloaded pinned source distribution")
    arguments = parser.parse_args()
    print(json.dumps(make_task(arguments.output, sdist=arguments.sdist), indent=2))
