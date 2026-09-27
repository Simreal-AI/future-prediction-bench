"""Run genuine Waypoint image-store code on bounded Linux byte fixtures.

The author package is imported normally and mounted read-only. No CRIU,
agent rollout, model, or trainer runs. --prepare-only never invokes Docker/Go.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import tarfile
import time
from urllib.parse import urlsplit
import uuid

COMMIT = "dcb6a7b0f4114f4732b9d7f1d1824dc5e4ff4a24"
GO_TREE_SHA256 = "94c8943f9e0ae8631aa5c08b2df3951974e92d9930513c9773e41fbc332f59ac"
GO_MODULE_COUNT = 29
GO_VERSION = "go1.27.1"
TOOLCHAIN_NAME = GO_VERSION + ".linux-arm64.tar.gz"
TOOLCHAIN_SHA256 = "3450b45a3f9ee8568792736a5c5e70a1f2e9b36c35a8f74958c03e51d7d92bec"
DEFAULT_IMAGE = "sha256:8e8fad2e920379c34538b76ed81165e0c7c896605972a851830392892f282fb6"
SOURCE_FILES = {
    "pkg/waypoint/imagestore.go": "7da668ecaf5c22e5d9e293547aa95601264cdf7a881ba8dc86dd306bc3ce3300",
    "go.mod": "96c01ebb7fba1b9bc243ba028af4aa55754425c0d1a24cebbdc28530e49dee64",
    "go.sum": "73ad92b4a3264581a00aee3dd7dee6dee3f64e014fe11d278e130e2303a05209",
    "LICENSE": "cddf00b77e2d22068bde8e501d95bca1e3de615e0b4e905298e5752b04deea11",
    "NOTICE": "098349e723ae12a0c1babfcf87b7cefa222b75b9dea863d1c92ea3cb2e44f63a",
}
# Fixed public inputs, reviewed against the author's untouched go.sum.
MODULE_INPUTS = {
    "github.com/creack/pty/@v/v1.1.24.mod": "a3979fd32c00f35d17eb3b97d0b3cc6f4efb057bf78271361fa773531e601531",
    "github.com/creack/pty/@v/v1.1.24.info": "1c916a1e65da9cf22d6b223a900d750b8f174ec093cc66969cc3e1fc4d141569",
    "github.com/creack/pty/@v/v1.1.24.zip": "754e25253e76a5583b80d57d3add3afe68fc4d9f2a490968a9d1eda8c8fd8815",
    "golang.org/x/sys/@v/v0.42.0.mod": "57f4393ea18d5446a12363b35c23a616d843fa1669c7121a70a2bc3a9677d665",
    "golang.org/x/sys/@v/v0.42.0.info": "4f952cb1f57499e8605892ce833af159f1edd800d2ad09a25463674aa636756e",
    "golang.org/x/sys/@v/v0.42.0.zip": "99df0ad90183debc80aee0b7489648574c6baa0c1cf5da37aaf591cf2e2d426a",
}


def sha(path):
    digest = hashlib.sha256()
    # This also prevents a cached target from being followed after validation.
    fd = os.open(Path(path), os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("regular_file_required_for_sha256")
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def reject_overlap(source, writable):
    source, writable = Path(source).resolve(), Path(writable).resolve()
    if source == writable or source in writable.parents or writable in source.parents:
        raise ValueError("source_and_writable_paths_must_not_overlap")


def owned_directory_path(path):
    """Check writable ancestry without following any existing symlink."""
    path = Path(os.path.abspath(path))
    nearest = None
    for part in (path, *path.parents):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode):
            raise ValueError("writable_path_symlink_rejected")
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError("writable_directory_path_required")
        if nearest is None:
            nearest = info
    if nearest is None or nearest.st_uid != os.getuid():
        raise ValueError("owned_writable_parent_required")
    return path


def existing_input(path, expected):
    """Validate an existing fixed input; never replace or follow it."""
    path = Path(path)
    owned_directory_path(path.parent)
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
        raise ValueError("owned_regular_public_input_required")
    if sha(path) != expected:
        raise ValueError("public_input_sha256_mismatch")
    return info.st_size


def checked_paths(source, output, cache):
    """Reject overlap and unowned/existing output before creating anything."""
    source = Path(source).resolve(strict=True)
    output, cache = (owned_directory_path(path) for path in (output, cache))
    for left, right in ((source, output), (source, cache), (output, cache)):
        reject_overlap(left, right)
    if os.path.lexists(output):
        raise ValueError("new_nonexistent_output_directory_required")
    # Reject every unsafe cached target before writing even the output folder.
    existing_input(cache / TOOLCHAIN_NAME, TOOLCHAIN_SHA256)
    for relative, expected in MODULE_INPUTS.items():
        existing_input(cache / "module-proxy" / relative, expected)
    return source, output, cache


def checked_source(source):
    source = Path(source).resolve(strict=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True)
    if head != COMMIT or status:
        raise ValueError("clean_pinned_official_waypoint_checkout_required")
    digest = hashlib.sha256()
    modules = sorted(source.rglob("*.go"))
    for path in modules:
        if path.is_symlink():
            raise ValueError("source_go_symlink_rejected")
        digest.update(path.relative_to(source).as_posix().encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    if len(modules) != GO_MODULE_COUNT or digest.hexdigest() != GO_TREE_SHA256:
        raise ValueError("official_go_source_tree_pin_mismatch")
    actual = {name: sha(source / name) for name in SOURCE_FILES}
    if actual != SOURCE_FILES:
        raise ValueError("official_source_or_license_pin_mismatch")
    return {"repository": "https://github.com/Alex-XJK/waypoint", "commit": head,
            "clean": True, "go_module_count": len(modules),
            "canonical_go_tree_sha256": digest.hexdigest(),
            "source_sha256": actual, "license": "Apache-2.0", "notice_present": True}


def download_checked(url, path, expected):
    path = Path(path)
    parts = urlsplit(url)
    if parts.scheme != "https" or not parts.hostname or parts.username or parts.password:
        raise ValueError("credential_free_https_public_url_required")
    cached_bytes = existing_input(path, expected)
    if cached_bytes is not None:
        return {"url": url, "sha256": expected, "bytes": cached_bytes}
    path.parent.mkdir(parents=True, exist_ok=True)
    owned_directory_path(path.parent)
    partial = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
    fd = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        # curl writes the already-open owned descriptor, rather than reopening
        # a target pathname. --disable must be its first argument.
        with os.fdopen(fd, "wb") as stream:
            subprocess.run(["curl", "--disable", "--fail", "--location", "--silent",
                "--show-error", "--proto", "=https", "--proto-redir", "=https",
                "--max-time", "180", url], stdout=stream, check=True, timeout=190)
            stream.flush()
            os.fsync(stream.fileno())
        if sha(partial) != expected:
            raise ValueError("public_download_sha256_mismatch")
        # Hard-link publication fails if another target appeared; no existing
        # verified or unverified cached input is overwritten.
        os.link(partial, path, follow_symlinks=False)
    finally:
        partial.unlink(missing_ok=True)
    size = existing_input(path, expected)
    return {"url": url, "sha256": expected, "bytes": size}


def prepare_inputs(cache):
    records = {TOOLCHAIN_NAME: download_checked("https://go.dev/dl/" + TOOLCHAIN_NAME,
        cache / TOOLCHAIN_NAME, TOOLCHAIN_SHA256)}
    for relative, expected in MODULE_INPUTS.items():
        records["module-proxy/" + relative] = download_checked(
            "https://proxy.golang.org/" + relative, cache / "module-proxy" / relative, expected)
    return records


def extract_toolchain(archive, output):
    """Extract a checksum-verified distribution only into a new owned dir."""
    output.mkdir()
    with tarfile.open(archive, "r:gz") as stream:
        members = stream.getmembers()
        for item in members:
            name = PurePosixPath(item.name)
            if (name.is_absolute() or ".." in name.parts or not name.parts or
                    name.parts[0] != "go" or not (item.isfile() or item.isdir())):
                raise ValueError("toolchain_archive_member_not_regular_or_safe")
        stream.extractall(output, members=members, filter="data")


def run(args):
    source, output, cache = checked_paths(args.source, args.output, args.cache)
    if re.fullmatch(r"sha256:[0-9a-f]{64}", args.docker_image) is None:
        raise ValueError("cached_linux_arm64_image_digest_required")
    provenance = checked_source(source)
    output.mkdir(parents=True, exist_ok=False)
    inputs = prepare_inputs(cache)
    here = Path(__file__).resolve().parent
    result = {"schema_version": "official-waypoint-imagestore-check-v1",
        "source": provenance, "public_inputs": inputs,
        "harness_sha256": sha(here / "harness.go"), "checker_sha256": sha(__file__),
        "toolchain_archive_sha256": TOOLCHAIN_SHA256,
        "actual_criu_run": False, "model_rollout_run": False, "trainer_run": False,
        "speedup_claim": False, "upstream_tests_run": False,
        "scope": "full_unchanged_author_package_import_owned_images_store_fixtures"}
    if args.prepare_only:
        result.update({"status": "prepared_not_executed", "package_compiled": False,
                       "fixture_passed": False})
        (output / "prepared.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    inspected = subprocess.check_output(["docker", "image", "inspect", "--format",
        "{{.Id}} {{.Architecture}} {{.Os}}", args.docker_image], text=True).strip().split()
    if inspected != [args.docker_image, "arm64", "linux"]:
        raise ValueError("cached_docker_image_platform_or_digest_mismatch")
    result["linux_container_image"] = {"id": inspected[0], "architecture": inspected[1],
                                       "os": inspected[2], "privileged": False, "uid": 65534}
    toolchain = output / "toolchain"
    extract_toolchain(cache / TOOLCHAIN_NAME, toolchain)
    module = output / "external-module"
    module.mkdir()
    shutil.copyfile(here / "harness.go", module / "main.go")
    (module / "go.mod").write_text("module fpb-waypoint-fixture\n\ngo 1.25.0\n\nrequire (\n"
        " github.com/Alex-XJK/waypoint v0.0.0\n golang.org/x/sys v0.42.0\n)\n\n"
        "replace github.com/Alex-XJK/waypoint => /source\n")
    shutil.copyfile(source / "go.sum", module / "go.sum")
    # Read-only mounts + local module proxy; no service, trainer, or privileged API.
    docker = ["docker", "run", "--rm", "--pull", "never", "--platform", "linux/arm64",
        "--network", "none", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--user", "65534:65534", "--pids-limit", "128", "--cpus", "1",
        "--workdir", "/module"]
    for host, guest in ((source, "/source"), (module, "/module"),
                        (toolchain / "go", "/go"), (cache / "module-proxy", "/proxy")):
        if "," in str(host):
            raise ValueError("docker_bind_path_comma_rejected")
        docker += ["--mount", f"type=bind,src={host},dst={guest},readonly"]
    for value in ("GOTOOLCHAIN=local", "GOMAXPROCS=1", "CGO_ENABLED=0", "GOENV=off",
                  "GOWORK=off", "GOCACHE=/tmp/fpb-go-cache", "GOPATH=/tmp/fpb-go-path",
                  "GOPROXY=file:///proxy", "GOSUMDB=off"):
        docker += ["--env", value]
    script = ("set -eu; /go/bin/go build -mod=readonly -buildvcs=false -p=1 "
              "-o /tmp/fpb-waypoint-harness .; /go/bin/go mod verify >&2; "
              "/tmp/fpb-waypoint-harness")
    docker += ["--entrypoint", "/bin/sh", args.docker_image, "-c", script]
    started = time.monotonic()
    completed = subprocess.run(docker, text=True, capture_output=True, timeout=600, check=False)
    result["build_and_fixture_wall_ms"] = (time.monotonic() - started) * 1000
    result["container_returncode"] = completed.returncode
    (output / "container-stdout.json").write_text(completed.stdout)
    (output / "container-stderr.txt").write_text(completed.stderr)
    result["container_stdout_sha256"] = sha(output / "container-stdout.json")
    result["container_stderr_sha256"] = sha(output / "container-stderr.txt")
    result["source_after"] = checked_source(source)
    result["module_verify_passed"] = "all modules verified" in completed.stderr
    try:
        fixture = json.loads(completed.stdout)
    except json.JSONDecodeError:
        fixture = None
    if isinstance(fixture, dict) and fixture.get("schema_version") == "official-waypoint-imagestore-fixture-v1":
        result["fixture"] = fixture
        result["package_compiled"] = True
    else:
        result["package_compiled"] = False
    if completed.returncode != 0:
        result.update({"status": "real_compile_or_fixture_failed", "fixture_passed": False,
                       "diagnostic": completed.stderr[-12000:],
                       "fixture_error": fixture.get("error") if isinstance(fixture, dict) else None})
    else:
        result["fixture_passed"] = (isinstance(fixture, dict) and fixture.get("passed") is True and
            fixture.get("go_version") == GO_VERSION and fixture.get("goos") == "linux" and
            fixture.get("goarch") == "arm64" and fixture.get("uid") == 65534 and
            result["module_verify_passed"])
        result["status"] = "passed" if result["fixture_passed"] else "real_fixture_validation_failed"
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--docker-image", default=DEFAULT_IMAGE)
    parser.add_argument("--prepare-only", action="store_true")
    observed = run(parser.parse_args())
    print(json.dumps(observed, indent=2))
    raise SystemExit(0 if observed.get("fixture_passed") or observed.get("status") ==
                     "prepared_not_executed" else 1)
