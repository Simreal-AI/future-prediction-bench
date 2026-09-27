"""Build an allowlisted source archive nested in a reviewable materials bundle.

This creates local ZIP files only. It never publishes a repository or includes
operational runs, private configuration, downloaded source archives, or PDFs.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
BRIEFING_NAME = "project-highlights-zh.md"


def _briefing_file(requested: str | Path | None = None) -> Path | None:
    """Accept only an explicitly requested, reviewed supplemental briefing."""
    if requested is None:
        return None
    path = Path(requested)
    if path.is_absolute():
        try:
            path = path.relative_to(ROOT)
        except ValueError:
            raise ValueError("Supplemental briefing must stay within the project root") from None
    if path.as_posix() != f"reports/{BRIEFING_NAME}":
        raise ValueError("Supplemental briefing must be reports/project-highlights-zh.md")
    path = ROOT / path
    if (not path.is_file() or path.is_symlink() or path.parent.is_symlink()
            or not path.resolve().is_relative_to(ROOT)):
        raise ValueError("Reviewed supplemental briefing is missing or unsafe")
    return path


def _version():
    import re
    content = (ROOT / "future_prediction_bench" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__ = "([0-9]+\.[0-9]+\.[0-9]+)"$', content, re.MULTILINE)
    if match is None:
        raise ValueError("Package version not found")
    return match.group(1)


def _candidate_public_files():
    fixed = [".env.example", ".gitignore", ".github/workflows/ci.yml",
             "CHANGELOG.md", "CONTRIBUTING.md", "LICENSE", "MANIFEST.in", "README.md",
             "THIRD_PARTY_NOTICES.md", "pilot_config.json",
             "pyproject.toml", "configs/baselines.example.json", "configs/sources.json",
             "examples/questions.fixture.jsonl",
             "examples/realworld_boltons26/stateless_contract.json",
             "examples/realworld_boltons26/stateless_contract_v2.json",
             "examples/resident_guest_candidate/boltons_contract_local.json",
             "examples/cooperative_realworld/boltons_contract.json",
             "scripts/public_manifest.json", "runs/.gitkeep"]
    files = [ROOT / name for name in fixed]
    for folder, suffixes in (("docs", {".md", ".json"}), ("examples", {".py", ".md", ".json", ".jsonl", ".c", ".go"}),
                             ("examples/official_cubecow", {".rs", ".toml", ".lock"}),
                             ("future_prediction_bench", {".py"}),
                             ("tests", {".py"}), ("scripts", {".py"})):
        files.extend(path for path in (ROOT / folder).rglob("*") if path.is_file()
                     and path.suffix in suffixes and "__pycache__" not in path.parts)
    return sorted(set(files))


def _public_files():
    manifest = json.loads((ROOT / "scripts" / "public_manifest.json").read_text(encoding="utf-8"))
    names = manifest.get("files")
    if manifest.get("schema_version") != 1 or not isinstance(names, list) or not all(
            isinstance(name, str) and name and not name.startswith("/")
            and "\\" not in name and ":" not in PurePosixPath(name).parts[0]
            and ".." not in PurePosixPath(name).parts for name in names):
        raise ValueError("Invalid public file manifest")
    if names != sorted(set(names)) or "scripts/public_manifest.json" not in names:
        raise ValueError("Public file manifest must be sorted, unique, and self-contained")
    unique = [ROOT / name for name in names]
    missing = [str(path) for path in unique if not path.is_file()]
    if missing:
        raise ValueError(f"Missing public files: {missing}")
    unsafe = [str(path) for path in unique if path.is_symlink() or not path.resolve().is_relative_to(ROOT)]
    if unsafe:
        raise ValueError(f"Public file is a symlink or leaves project root: {unsafe}")
    candidates = {path.relative_to(ROOT).as_posix() for path in _candidate_public_files()}
    unlisted = sorted(candidates - set(names))
    if unlisted:
        raise ValueError(f"Unreviewed public candidate files: {unlisted}")
    outside_candidates = sorted(set(names) - candidates)
    if outside_candidates:
        raise ValueError(f"Public manifest includes files outside reviewed candidate roots: {outside_candidates}")
    return unique


def _add(zipped, path, name):
    _add_bytes(zipped, path.read_bytes(), name)


def _add_bytes(zipped, data, name):
    info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    zipped.writestr(info, data)


def _bytes_zip(files, prefix):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        for path in files:
            _add(zipped, path, prefix + path.relative_to(ROOT).as_posix())
    return stream.getvalue()


def build_materials(
    briefing_path: str | Path | None = None, output_dir: Path | None = None
) -> dict[str, object]:
    version = _version()
    day = datetime.now().strftime("%Y-%m-%d")
    files = _public_files()
    briefing = _briefing_file(briefing_path)
    out = output_dir if output_dir is not None else ROOT / "dist"
    out.mkdir(parents=True, exist_ok=True)
    source_name = f"future-prediction-bench-v{version}.zip"
    source_bytes = _bytes_zip(files, "future-prediction-bench/")
    source_path = out / source_name
    source_path.write_bytes(source_bytes)
    base = "future-prediction-bench-materials/"
    intro = ("# Future Prediction Bench materials\n\n"
             "The `github/` ZIP is the allowlisted public source archive. "
             "`project-materials/` contains the same reviewable files. "
             "Consult `github/PUBLISHING.md` for release verification.\n")
    if briefing is not None:
        intro += "\n`briefing/` contains an explicitly included Chinese stakeholder summary.\n"
    publishing = ("# Release verification\n\n"
                  "Use `docs/RELEASE_CHECKLIST.md` in the source archive to verify a release. "
                  "This bundle includes curated public measurement reports but excludes raw operational runs, "
                  "private baseline configuration, credentials, and downloaded third-party source.\n")
    checksums = [f"{hashlib.sha256(source_bytes).hexdigest()}  github/{source_name}"]
    if briefing is not None:
        checksums.append(f"{hashlib.sha256(briefing.read_bytes()).hexdigest()}  briefing/{BRIEFING_NAME}")
    checksums.extend(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  project-materials/{path.relative_to(ROOT).as_posix()}"
                     for path in files)
    outer_path = out / f"future-prediction-bench-materials-v{version}-{day}.zip"
    with zipfile.ZipFile(outer_path, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        _add_bytes(zipped, intro.encode(), base + "README.md")
        _add_bytes(zipped, publishing.encode(), base + "github/PUBLISHING.md")
        _add_bytes(zipped, source_bytes, base + "github/" + source_name)
        if briefing is not None:
            _add(zipped, briefing, base + "briefing/" + BRIEFING_NAME)
        _add_bytes(zipped, ("\n".join(checksums) + "\n").encode(), base + "SHA256SUMS.txt")
        for path in files:
            _add(zipped, path, base + "project-materials/" + path.relative_to(ROOT).as_posix())
    outer_sha = hashlib.sha256(outer_path.read_bytes()).hexdigest()
    (out / (outer_path.name + ".sha256")).write_text(f"{outer_sha}  {outer_path.name}\n")
    return {"version": version, "public_file_count": len(files),
            "source_zip": str(source_path), "materials_zip": str(outer_path),
            "materials_sha256": outer_sha, "supplemental_briefing": briefing is not None}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--briefing", type=Path,
                        help="Opt in to reports/project-highlights-zh.md as a separate supplement")
    parser.add_argument("--output-dir", type=Path, help="Archive output directory (default: dist)")
    args = parser.parse_args(argv)
    print(json.dumps(build_materials(args.briefing, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
