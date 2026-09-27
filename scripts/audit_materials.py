"""Audit a local materials bundle before any public release.

The script checks the nested source ZIP against the allowlisted checkout,
validates the mirrored materials and checksums, scans for common private data,
checks relative Markdown links, and optionally runs tests from an extracted ZIP.
It prints only finding categories and archive member names, never matched text.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

try:
    from . import package_materials
except ImportError:  # Direct `python scripts/audit_materials.py` execution.
    import package_materials


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".md", ".json", ".jsonl", ".toml", ".txt", ".yml", ".c", ".go", ".rs", ".lock"}
PRIVATE_PATTERNS = {
    "personal_absolute_path": re.compile(rb"/(?:Users/[^/\s\"']+|private/var/folders/)[^\s\"']*"),
    "credential_value": re.compile(
        rb"(?:sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
        rb"AKIA[0-9A-Z]{16}|Bearer[ \t]+[A-Za-z0-9._-]{20,})"
    ),
}
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
SHA256 = re.compile(r"(?:sha256:)?[a-f0-9]{64}\Z")
CJK_PROSE = re.compile(r"[\u3400-\u9fff]")


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_name(name: str) -> bool:
    parts = PurePosixPath(name).parts
    return bool(parts) and not name.startswith("/") and ":" not in parts[0] and "\\" not in name and all(
        part not in {"", ".", ".."} for part in parts
    )


def _members(zipped: zipfile.ZipFile, errors: list[str], label: str) -> dict[str, bytes]:
    names = zipped.namelist()
    if len(names) != len(set(names)):
        errors.append(f"{label}: duplicate archive members")
    data: dict[str, bytes] = {}
    for info in zipped.infolist():
        if info.is_dir():
            continue
        if not _safe_name(info.filename):
            errors.append(f"{label}: unsafe member name")
            continue
        if (info.external_attr >> 16) & 0o170000 == 0o120000:
            errors.append(f"{label}: symbolic-link member {info.filename}")
            continue
        if info.file_size > 10_000_000:
            errors.append(f"{label}: oversized member {info.filename}")
            continue
        data[info.filename] = zipped.read(info)
    return data


def _audit_text(name: str, data: bytes, errors: list[str]) -> None:
    for label, pattern in PRIVATE_PATTERNS.items():
        if pattern.search(data):
            errors.append(f"{name}: {label}")
    if name.endswith(".md") and CJK_PROSE.search(data.decode("utf-8")):
        errors.append(f"{name}: non-English prose")
    if name.endswith(".json"):
        try:
            obj = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError):
            errors.append(f"{name}: invalid JSON")
            return
        if "/docs/measurements/" in name:
            _audit_digests(name, obj, errors)
            filename = PurePosixPath(name).name
            if filename.startswith(("microvm_", "guest_", "stateless_", "prepared_")):
                _audit_asset_binding(name, obj, errors)


def _audit_asset_binding(name: str, obj: object, errors: list[str]) -> None:
    if not isinstance(obj, dict) or not isinstance(obj.get("asset_binding"), dict):
        errors.append(f"{name}: missing asset binding")
        return
    binding = obj["asset_binding"]
    schema = binding.get("manifest_schema_version")
    if schema not in {"boltons-microvm-assets-v1", "boltons-microvm-assets-v2"}:
        errors.append(f"{name}: unknown asset manifest schema")
    if not isinstance(binding.get("rootfs_seed_sha256"), str):
        errors.append(f"{name}: missing rootfs seed digest")
    if schema == "boltons-microvm-assets-v2":
        for key in ("source_sdist_sha256", "seed_workspace_sha256"):
            if not isinstance(binding.get(key), str):
                errors.append(f"{name}: missing {key}")
    elif schema == "boltons-microvm-assets-v1":
        if binding.get("source_sdist_sha256") is not None or binding.get("seed_workspace_sha256") is not None:
            errors.append(f"{name}: unsupported v1 source binding")
        if not binding.get("provenance_note"):
            errors.append(f"{name}: missing v1 provenance note")


def _audit_digests(name: str, obj: object, errors: list[str], path: str = "") -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            child = path + "/" + key
            if key.endswith("sha256") and isinstance(value, str) and not SHA256.fullmatch(value):
                errors.append(f"{name}: malformed digest at {child}")
            _audit_digests(name, value, errors, child)
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            _audit_digests(name, value, errors, path + f"/{index}")


def _audit_links(members: dict[str, bytes], errors: list[str]) -> None:
    for name, data in members.items():
        if not name.endswith(".md"):
            continue
        text = data.decode("utf-8")
        for match in MARKDOWN_LINK.finditer(text):
            target = match.group(1).split("#", 1)[0].strip("<>")
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            dest = PurePosixPath(name).parent.joinpath(target)
            normalized: list[str] = []
            escaped = False
            for part in dest.parts:
                if part == "..":
                    if len(normalized) > 1:
                        normalized.pop()
                    else:
                        escaped = True
                elif part != ".":
                    normalized.append(part)
            if escaped:
                errors.append(f"{name}: relative link escapes source root")
                continue
            if "/".join(normalized) not in members:
                errors.append(f"{name}: missing relative link target")


def _audit_distributions(
    version: str, expected_paths: dict[str, Path], errors: list[str]
) -> dict[str, str]:
    dist = ROOT / "dist"
    sdist_path = dist / f"future_prediction_bench-{version}.tar.gz"
    wheel_path = dist / f"future_prediction_bench-{version}-py3-none-any.whl"
    statuses = {"sdist": "missing", "wheel": "missing"}
    expected_package = {
        rel for rel in expected_paths if rel.startswith("future_prediction_bench/") and rel.endswith(".py")
    }
    if sdist_path.is_file():
        with tarfile.open(sdist_path, "r:gz") as archive:
            actual: dict[str, bytes] = {}
            prefix = f"future_prediction_bench-{version}/"
            for member in archive.getmembers():
                if member.isdir():
                    continue
                if not member.isfile() or not member.name.startswith(prefix):
                    errors.append("sdist: unsafe or unexpected archive member")
                    continue
                rel = member.name[len(prefix):]
                if not _safe_name(rel):
                    errors.append("sdist: unsafe archive name")
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    errors.append("sdist: unreadable member")
                    continue
                actual[rel] = extracted.read()
        allowed_generated = {"PKG-INFO", "setup.cfg"}
        unexpected = {
            rel for rel in actual if rel not in expected_paths
            and rel not in allowed_generated
            and not rel.startswith("future_prediction_bench.egg-info/")
        }
        if unexpected:
            errors.append("sdist: unreviewed source members")
        if {rel for rel in actual if rel.startswith("future_prediction_bench/") and rel.endswith(".py")} != expected_package:
            errors.append("sdist: package module list differs from manifest")
        if "scripts/public_manifest.json" not in actual:
            errors.append("sdist: public manifest is missing")
        expected_sdist = set(expected_paths) - {
            ".gitignore", ".github/workflows/ci.yml", "runs/.gitkeep"
        }
        if expected_sdist - set(actual):
            errors.append("sdist: reviewed source files are missing")
        for rel in set(actual) & set(expected_paths):
            if actual[rel] != expected_paths[rel].read_bytes():
                errors.append(f"sdist: checkout mismatch {rel}")
        statuses["sdist"] = "checked"
    else:
        errors.append("sdist: build artifact is missing")

    if wheel_path.is_file():
        with zipfile.ZipFile(wheel_path) as archive:
            actual = _members(archive, errors, "wheel")
        wheel_package = {
            rel for rel in actual if rel.startswith("future_prediction_bench/") and rel.endswith(".py")
        }
        if wheel_package != expected_package:
            errors.append("wheel: package module list differs from manifest")
        for rel in wheel_package & expected_package:
            if actual[rel] != expected_paths[rel].read_bytes():
                errors.append(f"wheel: checkout mismatch {rel}")
        unexpected = {
            rel for rel in actual if not rel.startswith("future_prediction_bench/")
            and not rel.startswith(f"future_prediction_bench-{version}.dist-info/")
        }
        if unexpected:
            errors.append("wheel: unreviewed member")
        statuses["wheel"] = "checked"
    else:
        errors.append("wheel: build artifact is missing")
    return statuses


def audit(
    outer_path: Path, run_tests: bool = True, check_distributions: bool = False,
    briefing_path: str | Path | None = None,
) -> dict[str, object]:
    errors: list[str] = []
    version = package_materials._version()
    source_name = f"future-prediction-bench-v{version}.zip"
    base = "future-prediction-bench-materials/"
    source_member = base + "github/" + source_name
    briefing = package_materials._briefing_file(briefing_path)
    briefing_member = base + "briefing/" + package_materials.BRIEFING_NAME
    expected_paths = {path.relative_to(ROOT).as_posix(): path for path in package_materials._public_files()}
    expected_source_names = {"future-prediction-bench/" + path for path in expected_paths}
    expected_outer_names = {
        base + "README.md", base + "github/PUBLISHING.md", source_member,
        base + "SHA256SUMS.txt",
    } | {base + "project-materials/" + path for path in expected_paths}
    if briefing is not None:
        expected_outer_names.add(briefing_member)

    with zipfile.ZipFile(outer_path) as outer:
        outer_members = _members(outer, errors, "materials")
    if set(outer_members) != expected_outer_names:
        errors.append("materials: member list differs from public allowlist")
    briefing_bytes = briefing.read_bytes() if briefing is not None else None
    if briefing_bytes is not None:
        if outer_members.get(briefing_member) != briefing_bytes:
            errors.append("materials: Chinese briefing differs from reviewed file")
        for label, pattern in PRIVATE_PATTERNS.items():
            if pattern.search(briefing_bytes):
                errors.append(f"{briefing_member}: {label}")
    source_bytes = outer_members.get(source_member)
    if source_bytes is None:
        errors.append("materials: nested source ZIP is missing")
        source_members: dict[str, bytes] = {}
    else:
        standalone_source = outer_path.parent / source_name
        if not standalone_source.is_file() or standalone_source.read_bytes() != source_bytes:
            errors.append("materials: standalone source ZIP differs from nested source")
        with tempfile.TemporaryFile() as source_file:
            source_file.write(source_bytes)
            source_file.seek(0)
            with zipfile.ZipFile(source_file) as zipped:
                source_members = _members(zipped, errors, "source")
    if set(source_members) != expected_source_names:
        errors.append("source: member list differs from public allowlist")

    for rel, path in expected_paths.items():
        source_key = "future-prediction-bench/" + rel
        mirror_key = base + "project-materials/" + rel
        current = path.read_bytes()
        if source_members.get(source_key) != current:
            errors.append(f"source: checkout mismatch {rel}")
        if outer_members.get(mirror_key) != current:
            errors.append(f"materials: checkout mismatch {rel}")
        if rel.startswith("runs/") and rel != "runs/.gitkeep":
            errors.append("source: operational run included")
        if Path(rel).suffix.lower() in {".qcow2", ".sqlite", ".sqlite3", ".db", ".pdf", ".pyc"}:
            errors.append(f"source: unexpected binary {rel}")
        if Path(rel).suffix.lower() in TEXT_SUFFIXES or rel in {".env.example", ".gitignore"}:
            _audit_text(source_key, current, errors)

    checksums = outer_members.get(base + "SHA256SUMS.txt", b"").decode("utf-8")
    expected_checksums = [f"{_digest(source_bytes)}  github/{source_name}"] if source_bytes else []
    if briefing_bytes is not None:
        expected_checksums.append(
            f"{_digest(briefing_bytes)}  briefing/{package_materials.BRIEFING_NAME}"
        )
    expected_checksums += [
        f"{_digest(path.read_bytes())}  project-materials/{rel}"
        for rel, path in sorted(expected_paths.items())
    ]
    if checksums != "\n".join(expected_checksums) + "\n":
        errors.append("materials: SHA256SUMS mismatch")
    outer_sha_file = outer_path.with_name(outer_path.name + ".sha256")
    if not outer_sha_file.is_file() or outer_sha_file.read_text() != f"{_digest(outer_path.read_bytes())}  {outer_path.name}\n":
        errors.append("materials: outer ZIP checksum mismatch")

    _audit_links(source_members, errors)
    distribution_status = (
        _audit_distributions(version, expected_paths, errors)
        if check_distributions else {"sdist": "skipped", "wheel": "skipped"}
    )
    tests_output = "skipped"
    cli_output = "skipped"
    if run_tests and not errors and source_bytes:
        with tempfile.TemporaryDirectory(prefix="future-prediction-audit-") as temp:
            dest = Path(temp)
            with zipfile.ZipFile(io.BytesIO(source_bytes)) as source:
                source.extractall(dest)
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            process = subprocess.run(
                [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"],
                cwd=dest / "future-prediction-bench", env=env,
                capture_output=True, text=True, timeout=180,
            )
            tests_output = "pass" if process.returncode == 0 else "fail"
            if process.returncode:
                errors.append("source: extracted archive test suite failed")
            else:
                cli = subprocess.run(
                    [sys.executable, "-m", "future_prediction_bench", "--help"],
                    cwd=dest / "future-prediction-bench", env=env,
                    capture_output=True, text=True, timeout=30,
                )
                if cli.returncode:
                    errors.append("source: extracted archive CLI smoke test failed")
                    cli_output = "fail"
                else:
                    cli_output = "pass"
    return {
        "status": "pass" if not errors else "fail",
        "version": version,
        "source_files": len(expected_paths),
        "supplemental_briefing": briefing is not None,
        "extracted_tests": tests_output,
        "extracted_cli": cli_output,
        "distributions": distribution_status,
        "materials_sha256": _digest(outer_path.read_bytes()),
        "findings": sorted(set(errors)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("materials_zip", type=Path)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--check-distributions", action="store_true")
    parser.add_argument("--briefing", type=Path,
                        help="Explicitly audit the reviewed reports/project-highlights-zh.md supplement")
    args = parser.parse_args()
    report = audit(args.materials_zip, run_tests=not args.skip_tests,
                   check_distributions=args.check_distributions,
                   briefing_path=args.briefing)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
