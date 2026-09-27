"""Prepare or build the exact populated fixture against external original code.

Only a fresh owned output directory is written. This script never installs,
downloads, formats, mounts, runs the fixture, or substitutes an upstream API.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import time
import tomllib

HERE = Path(__file__).resolve().parent
COMMIT = "d0081641c59822e4e5653b7462e914410b81910a"
MAIN_SHA = "9512f947ff915b4ad234bf238780b418f30311a8a471d2eb98c084116da60574"
LOCK_SHA = "1bc25eab0f64a3e71c3c781a0250733e5c9d12d5105e0d58cc09ae49b75b50dd"
MIB = 1048576


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(MIB), b""):
            h.update(block)
    return h.hexdigest()


def canonical(path, *, must_exist=True):
    lexical = Path(path).absolute()
    require(not any(p.is_symlink() for p in (lexical, *lexical.parents)), "symlink path/ancestor unsupported")
    return lexical.resolve(strict=must_exist)


def require_separate(output, inputs):
    require(not output.exists() and not output.is_symlink(), "fresh output required")
    require(output.parent.is_dir(), "output parent must already exist")
    for root in inputs:
        require(not output.is_relative_to(root) and not root.is_relative_to(output), "output/input overlap")


def inventory(root, *, max_files=6000):
    """Hash regular files only; never follow links or read special files."""
    records = {}
    total = 0
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs:
            p = Path(directory) / name
            require(not p.is_symlink(), "input directory symlink unsupported")
        for name in files:
            p = Path(directory) / name
            mode = p.lstat().st_mode
            require(stat.S_ISREG(mode), "input nonregular file unsupported")
            total += p.stat().st_size
            require(total <= 512 * MIB and len(records) < max_files, "input inventory exceeds bound")
            records[p.relative_to(root).as_posix()] = sha_file(p)
    return dict(sorted(records.items()))


def compiler_inventory(root):
    """Preserve genuine compiler/sysroot links as link text, without following.

    Debian's original sysroot includes absolute loader/library symlinks. They
    are input records, not destinations used by this helper to read or write.
    Runtime dependency attestation remains the operator's responsibility.
    """
    records, total = {}, 0
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in [*dirs, *files]:
            p = Path(directory) / name
            relative = p.relative_to(root).as_posix()
            mode = p.lstat().st_mode
            if stat.S_ISLNK(mode):
                target = os.readlink(p)
                require(len(target) <= 4096 and "\0" not in target, "bounded compiler symlink required")
                records[relative] = {"kind": "symlink", "target": target}
            elif stat.S_ISREG(mode):
                total += p.stat().st_size
                require(total <= 512 * MIB, "compiler input bytes exceed bound")
                records[relative] = {"kind": "regular_file", "sha256": sha_file(p)}
            else:
                require(stat.S_ISDIR(mode), "compiler special file unsupported")
            require(len(records) <= 10000, "compiler input count exceeds bound")
    return dict(sorted(records.items()))


def source_state(source, pins):
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    def git(*args):
        p = subprocess.run(["git", "-C", str(source), *args], capture_output=True, text=True, env=env, timeout=30)
        require(p.returncode == 0, "original Git checkout unavailable")
        return p.stdout
    require(git("rev-parse", "--show-toplevel").strip() == str(source), "source must be the checkout root")
    require(git("rev-parse", "HEAD").strip() == COMMIT, "wrong official source commit")
    require(not git("status", "--porcelain", "--untracked-files=all"), "clean original checkout required")
    actual = inventory(source / "cubecow", max_files=64)
    require(actual == pins["upstream_crate_files_sha256"], "original crate byte inventory changed")
    return {"commit": COMMIT, "Git_clean": True, "crate_files_sha256": actual}


def registry_packages(lock):
    packages = tomllib.loads(Path(lock).read_text())["package"]
    result = {}
    for package in packages:
        if "checksum" not in package:
            continue
        identifier = package["name"] + "-" + package["version"]
        require(re.fullmatch(r"[A-Za-z0-9_.+-]+", identifier) is not None, "invalid package identifier")
        require(package["source"] == "registry+https://github.com/rust-lang/crates.io-index", "unexpected registry source")
        require(identifier not in result and re.fullmatch(r"[0-9a-f]{64}", package["checksum"]), "invalid/duplicate registry package")
        result[identifier] = package["checksum"]
    require(len(result) == 106, "exact106 registry archives required")
    return result


def vendor_state(vendor, archives, packages):
    """Compare every vendored byte to the SHA-pinned original .crate member."""
    require({p.name for p in vendor.iterdir()} == set(packages), "exact vendor package cohort required")
    require({p.name for p in archives.glob("*.crate")} == {name + ".crate" for name in packages}, "exact archive cohort required")
    raw_inventory = inventory(vendor)
    declared_files, archive_hashes = {}, {}
    payload_total = 0
    for identifier, checksum in sorted(packages.items()):
        archive_path = archives / (identifier + ".crate")
        require(archive_path.is_file() and not archive_path.is_symlink(), "regular original crate archive required")
        require(archive_path.stat().st_size <= 64 * MIB and sha_file(archive_path) == checksum, "original crate archive SHA mismatch")
        archive_hashes[archive_path.name] = checksum
        checksum_path = vendor / identifier / ".cargo-checksum.json"
        manifest = json.loads(checksum_path.read_text())
        require(manifest.get("package") == checksum and type(manifest.get("files")) is dict, "vendor package checksum changed")
        member_hashes = {}
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive:
                path = PurePosixPath(member.name)
                require(path.parts and path.parts[0] == identifier and ".." not in path.parts and not path.is_absolute(), "crate member path escape")
                if member.isdir():
                    continue
                require(member.isfile() and len(path.parts) >= 2, "nonregular crate member unsupported")
                relative = PurePosixPath(*path.parts[1:]).as_posix()
                require(relative not in member_hashes and relative != ".cargo-checksum.json", "duplicate/reserved crate member")
                payload_total += member.size
                require(0 <= member.size <= 64 * MIB and payload_total <= 512 * MIB, "crate payload exceeds bound")
                stream = archive.extractfile(member)
                require(stream is not None, "crate member unreadable")
                digest = hashlib.sha256()
                with stream:
                    for block in iter(lambda: stream.read(MIB), b""):
                        digest.update(block)
                expected = digest.hexdigest()
                require(raw_inventory.get(identifier + "/" + relative) == expected, "vendor/original archive bytes differ")
                member_hashes[relative] = expected
                declared_files[identifier + "/" + relative] = expected
        require(manifest["files"] == member_hashes, "vendor checksum-map differs from original archive")
        declared_files[identifier + "/.cargo-checksum.json"] = raw_inventory[identifier + "/.cargo-checksum.json"]
    require(raw_inventory == dict(sorted(declared_files.items())), "extra/unverified vendor files")
    require(len(raw_inventory) - len(packages) == 5214, "exact5214 original vendor files required")
    digest = hashlib.sha256()
    for name, value in sorted(raw_inventory.items()):
        digest.update(name.encode() + b"\0" + bytes.fromhex(value))
    return {"archive_sha256": archive_hashes, "vendor_file_count_including_checksum_maps": len(raw_inventory),
            "vendor_sha256": digest.hexdigest(), "vendor_digest_serialization": "sorted relative path UTF8+NUL+raw SHA256; checksum maps included",
            "original_vendor_files_compared_to_archive_members": 5214}


def generated_manifest(source):
    quoted = json.dumps(str(source / "cubecow"), ensure_ascii=False)
    return ("[package]\nname = \"fpb-cubecow-populated-fixture\"\nversion = \"0.1.0\"\nedition = \"2021\"\npublish = false\n\n"
            "[workspace]\n\n[dependencies]\ncubecow = { path = " + quoted + " }\nserde_json = \"=1.0.149\"\n\n"
            "[profile.release]\nopt-level = 3\ndebug = false\n")


def native_toolchain(cargo, rustc):
    require(platform.system() == "Linux", "actual build requires native Linux GNU")
    result = {}
    for name, path in (("cargo", cargo), ("rustc", rustc)):
        require(path.is_file() and os.access(path, os.X_OK), "executable " + name + " required")
        p = subprocess.run([str(path), "-vV"], capture_output=True, text=True, timeout=30)
        require(p.returncode == 0 and p.stdout.startswith(name + " 1.93.1 "), "genuine " + name + "1.93.1 required")
        result[name] = {"path": str(path), "sha256": sha_file(path), "version_output": p.stdout}
    hosts = re.findall(r"^host: (.+)$", result["rustc"]["version_output"], re.M)
    require(len(hosts) == 1 and hosts[0] in ("aarch64-unknown-linux-gnu", "x86_64-unknown-linux-gnu"), "Linux GNU Rust host required")
    expected_machine = "aarch64" if hosts[0].startswith("aarch64") else "x86_64"
    require(platform.machine() == expected_machine, "native architecture mismatch")
    libc = ctypes.CDLL(None)
    require(hasattr(libc, "gnu_get_libc_version"), "GNU libc runtime required")
    libc.gnu_get_libc_version.restype = ctypes.c_char_p
    result["host"] = hosts[0]
    result["glibc_version"] = libc.gnu_get_libc_version().decode("ascii")
    result["declarations_are_not_distribution_signature_attestation"] = True
    return result


def prepare_build(*, source, vendor, archives, output, cargo=None, rustc=None,
                  toolchain_root=None, sysroot=None, prepare_only=False,
                  rustflags=None, timeout=600):
    source, vendor, archives = [canonical(p) for p in (source, vendor, archives)]
    output = canonical(output, must_exist=False)
    require_separate(output, (source, vendor, archives, HERE))
    require(type(timeout) is int and 1 <= timeout <= 900, "bounded compile timeout required")
    pins_file = HERE / "source_pins.json"
    pins = json.loads(pins_file.read_text())
    require(pins.get("upstream_commit") == COMMIT and pins.get("executed_fixture_source_sha256") == MAIN_SHA and pins.get("executed_fixture_lock_sha256") == LOCK_SHA, "public pin contract changed")
    require(sha_file(HERE / "src/main.rs") == MAIN_SHA and sha_file(HERE / "Cargo.lock") == LOCK_SHA, "executed fixture source/lock changed")
    before = {"source": source_state(source, pins), "vendor": vendor_state(vendor, archives, registry_packages(HERE / "Cargo.lock")),
              "public_inputs_sha256": {name: sha_file(HERE / name) for name in ("src/main.rs", "Cargo.lock", "source_pins.json", "prepare_build.py")}}
    require(registry_packages(source / "cubecow/Cargo.lock") == registry_packages(HERE / "Cargo.lock"), "fixture must preserve every original registry pin")
    toolchain = None
    protected = {}
    if toolchain_root is not None:
        protected["toolchain"] = canonical(toolchain_root)
    if sysroot is not None:
        protected["GNU_sysroot"] = canonical(sysroot)
    if protected:
        require_separate(output, protected.values())
        before["protected_compiler_input_records"] = {name: compiler_inventory(path) for name, path in protected.items()}
    if not prepare_only:
        require(cargo is not None and rustc is not None and toolchain_root is not None, "explicit original cargo/rustc and owned toolchain root required")
        cargo, rustc = canonical(cargo), canonical(rustc)
        toolchain_root = canonical(toolchain_root)
        require(cargo.is_relative_to(toolchain_root) and rustc.is_relative_to(toolchain_root), "executables must belong to supplied toolchain root")
        toolchain = native_toolchain(cargo, rustc)
    output.mkdir()
    fixture = output / "fixture"
    (fixture / "src").mkdir(parents=True)
    shutil.copyfile(HERE / "src/main.rs", fixture / "src/main.rs")
    shutil.copyfile(HERE / "Cargo.lock", fixture / "Cargo.lock")
    (fixture / "Cargo.toml").write_text(generated_manifest(source))
    home = output / "cargo-home"
    home.mkdir()
    config = "[source.crates-io]\nreplace-with = \"vendored-sources\"\n[source.vendored-sources]\ndirectory = " + json.dumps(str(vendor), ensure_ascii=False) + "\n[net]\noffline = true\n"
    (home / "config.toml").write_text(config)
    (output / "temp").mkdir()
    (output / "tool-home").mkdir()
    report = {"schema_version": "public-cubecow-populated-offline-build-v1", "prepared": True, "compiled": False,
        "fixture_runtime_executed": False, "prepare_only": prepare_only, "upstream_commit": COMMIT,
        "before": before, "native_toolchain": toolchain, "commands": [],
        "output_directory": str(output), "normal_original_path_dependency": str(source / "cubecow"),
        "compiler_input_signature_attestation": False,
        "snapshot_or_training_speedup_claimed": False}
    def save():
        (output / "build-result.json").write_text(json.dumps(report, indent=2) + "\n")
    save()
    try:
        require(sha_file(fixture / "src/main.rs") == MAIN_SHA and sha_file(fixture / "Cargo.lock") == LOCK_SHA, "prepared input byte mismatch")
        if not prepare_only:
            env = {name: os.environ[name] for name in ("PATH", "LD_LIBRARY_PATH", "GCC_EXEC_PREFIX", "COMPILER_PATH", "LIBRARY_PATH") if name in os.environ}
            env.update(HOME=str(output / "tool-home"), CARGO_HOME=str(home), CARGO_TARGET_DIR=str(output / "target"),
                       TMPDIR=str(output / "temp"), RUSTC=str(rustc), CARGO_NET_OFFLINE="true", CARGO_BUILD_JOBS="2")
            flags = os.environ.get("RUSTFLAGS", "") if rustflags is None else rustflags
            require(type(flags) is str and len(flags) <= 16384 and "\0" not in flags, "bounded explicit compiler flags required")
            env["RUSTFLAGS"] = flags
            report["operator_compiler_flags"] = flags
            commands = [[str(cargo), "metadata", "--manifest-path", str(fixture / "Cargo.toml"), "--format-version", "1", "--offline", "--locked"],
                        [str(cargo), "build", "--manifest-path", str(fixture / "Cargo.toml"), "--release", "--offline", "--locked", "--jobs", "2"]]
            for index, command in enumerate(commands):
                started = time.perf_counter()
                try:
                    p = subprocess.run(command, cwd=fixture, env=env, capture_output=True, timeout=timeout)
                    stdout, stderr, returncode = p.stdout, p.stderr, p.returncode
                    error = None
                except subprocess.TimeoutExpired as exc:
                    stdout, stderr, returncode = exc.stdout or b"", exc.stderr or b"", None
                    error = "compile_command_timeout"
                (output / f"command-{index}.stdout.log").write_bytes(stdout)
                (output / f"command-{index}.stderr.log").write_bytes(stderr)
                report["commands"].append({"argv": command, "returncode": returncode, "error": error,
                    "wall_ms": (time.perf_counter() - started) * 1000,
                    "stdout_sha256": hashlib.sha256(stdout).hexdigest(), "stderr_sha256": hashlib.sha256(stderr).hexdigest()})
                save()
                require(returncode == 0, error or "actual offline Cargo command failed")
                if index == 0:
                    metadata = json.loads(stdout)
                    original = [p for p in metadata["packages"] if p["name"] == "cubecow"]
                    require(len(original) == 1 and original[0]["source"] is None and Path(original[0]["manifest_path"]).resolve() == source / "cubecow/Cargo.toml", "normal full original library path dependency required")
                    report["actual_original_library_metadata"] = original[0]
            # No explicit --target: native GNU flags must also reach host
            # build scripts and proc macros, as in the actual frozen build.
            binary = output / "target/release/fpb-cubecow-populated-fixture"
            require(binary.is_file() and not binary.is_symlink(), "native fixture ELF required")
            with binary.open("rb") as stream:
                header = stream.read(20)
            expected_machine = 183 if toolchain["host"].startswith("aarch64") else 62
            require(len(header) == 20 and header[:6] == b"\x7fELF\x02\x01" and int.from_bytes(header[18:20], "little") == expected_machine, "native GNU architecture ELF required")
            report["fixture_executable"] = {"path": str(binary), "bytes": binary.stat().st_size, "sha256": sha_file(binary)}
            report["compiled"] = True
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        try:
            after = {"source": source_state(source, pins), "vendor": vendor_state(vendor, archives, registry_packages(HERE / "Cargo.lock")),
                "public_inputs_sha256": {name: sha_file(HERE / name) for name in before["public_inputs_sha256"]}}
            if protected:
                after["protected_compiler_input_records"] = {name: compiler_inventory(path) for name, path in protected.items()}
            report["after"] = after
            require(after == before, "original source/vendor/public inputs changed")
            require(sha_file(fixture / "src/main.rs") == MAIN_SHA and sha_file(fixture / "Cargo.lock") == LOCK_SHA, "Cargo modified the exact fixture source/lock")
            if toolchain is not None:
                require(sha_file(cargo) == toolchain["cargo"]["sha256"] and sha_file(rustc) == toolchain["rustc"]["sha256"], "toolchain executables changed")
            report["all_original_inputs_unchanged"] = True
        except Exception as exc:
            report["compiled"] = False
            report["preservation_error"] = {"type": type(exc).__name__, "message": str(exc)}
        save()
    require(report.get("all_original_inputs_unchanged") is True, "final input guard failed")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "vendor", "archives", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cargo", type=Path)
    parser.add_argument("--rustc", type=Path)
    parser.add_argument("--toolchain-root", type=Path, help="read-only owned standalone Rust distribution; required for actual builds")
    parser.add_argument("--sysroot", type=Path, help="optional relocated GNU compiler/CRT root, protected and rehashed before/after")
    parser.add_argument("--rustflags", help="explicit GNU linker/sysroot flags; otherwise uses caller RUSTFLAGS")
    parser.add_argument("--prepare-only", action="store_true", help="validate/copy exact inputs without invoking Cargo or executing the fixture")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    result = prepare_build(**vars(args))
    print(json.dumps({"prepared": result["prepared"], "compiled": result["compiled"], "fixture_runtime_executed": False,
                      "output": str(args.output), "fixture_executable": result.get("fixture_executable")}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("build error: " + str(error), file=sys.stderr)
        raise SystemExit(1)
