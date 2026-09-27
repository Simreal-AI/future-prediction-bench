"""Execute AReaL's pinned original capacity manager through normal imports.

This CPU control-plane probe advances an explicit version counter. It does
not load policy weights, generate model tokens, or perform RL optimization.
The complete unmodified AReaL package and real dependencies must import.
"""

import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import PackageNotFoundError, version
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from threading import Lock
import time


COMMIT = "2fad2d0e308fe631e70e971b97188ad5c5cc03cb"
PYTHON_COUNT = 504
PYTHON_SHA256 = "ed205b771ed629a5ca656ade06efe3cc4ad5f245346e89e2e1cd1d1c27926af4"
TEST_SHA256 = "0b6164520fd63f158d122b6f47dd4659a14982e29b02b74aba04acc57c2afeb8"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_separate_output(source, output):
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("source_and_output_must_not_overlap")


def checked_source(source):
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    changes = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain"], text=True)
    if commit != COMMIT or changes:
        raise ValueError("clean_pinned_original_areal_checkout_required")
    paths = sorted((source / "areal").rglob("*.py"))
    digest = hashlib.sha256()
    for path in paths:
        if path.is_symlink():
            raise ValueError("source_symlink_rejected")
        digest.update(path.relative_to(source).as_posix().encode() + b"\0" +
                      path.read_bytes() + b"\0")
    if len(paths) != PYTHON_COUNT or digest.hexdigest() != PYTHON_SHA256:
        raise ValueError("original_areal_python_source_pin_mismatch")
    if sha256(source / "tests/test_staleness_manager.py") != TEST_SHA256:
        raise ValueError("original_areal_test_pin_mismatch")


class VersionCounter:
    """An explicit control-plane input, not an inference engine substitute."""

    def __init__(self, initial=0):
        self._version = initial
        self._lock = Lock()

    def get_version(self):
        with self._lock:
            return self._version

    def advance(self):
        with self._lock:
            self._version += 1


def run_probe(manager_type):
    counter = VersionCounter()
    manager = manager_type(version_provider=counter,
        max_concurrent_rollouts=3, consumer_batch_size=2, max_staleness=1)
    snapshots = []

    def record(label, expected_capacity):
        capacity = manager.get_capacity()
        snapshot = {"event": label, "version": counter.get_version(),
                    "capacity": capacity, "stats": asdict(manager.get_stats())}
        snapshots.append(snapshot)
        if capacity != expected_capacity:
            raise AssertionError("capacity_lifecycle_mismatch: " + json.dumps(snapshot))

    record("initial", 3)
    for _ in range(3):
        manager.on_rollout_enqueued()
        manager.on_rollout_submitted()
    record("three_submitted", 0)
    manager.on_rollout_accepted()
    manager.on_rollout_accepted()
    manager.on_rollout_rejected()
    record("two_accepted_one_rejected", 2)
    counter.advance()
    record("version_counter_advanced", 3)
    if manager.get_pending_limit() != 4:
        raise AssertionError("pending_limit_mismatch")

    recovered_version = 100
    recovered = manager_type(version_provider=VersionCounter(recovered_version),
        max_concurrent_rollouts=1000, consumer_batch_size=8, max_staleness=2)
    before = recovered.get_capacity()
    recovered.on_version_recovered(recovered_version)
    after = recovered.get_capacity()
    if (before, after) != (824, 24):
        raise AssertionError("recovery_capacity_mismatch")
    return {"lifecycle": snapshots, "pending_limit": manager.get_pending_limit(),
        "recovery": {"recovered_version": recovered_version,
            "capacity_before_recovery_callback": before,
            "capacity_after_recovery_callback": after,
            "stats_after_recovery": asdict(recovered.get_stats())},
        "control_plane_checks_passed": True}


def main(args):
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    require_separate_output(source, output)
    if output.exists():
        raise ValueError("new_output_directory_required")
    output.mkdir(parents=True)
    report = {"schema_version": "official-areal-staleness-cpu-v1",
        "commit": COMMIT, "areal_python_sha256": PYTHON_SHA256,
        "areal_python_file_count": PYTHON_COUNT, "official_test_sha256": TEST_SHA256,
        "probe_sha256": sha256(Path(__file__)), "platform": platform.platform(),
        "python": sys.version, "scope": "original_capacity_control_CPU_only",
        "inference_executed": False, "gradient_update_executed": False,
        "gpu_speedup_measured": False, "passed": False}
    started = time.perf_counter()
    try:
        checked_source(source)
        # Use the actual upstream package initializer and all actual imports.
        # No AST extraction, synthetic modules, or replacements are installed.
        os.environ.setdefault("AREAL_CACHE_DIR", str(output / "cache"))
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(source))
        from areal.infra.staleness_manager import StalenessManager

        class_path = Path(inspect.getfile(StalenessManager)).resolve()
        if class_path != source / "areal/infra/staleness_manager.py":
            raise ValueError("capacity_manager_not_imported_from_pinned_source")
        report["class_source_path"] = class_path.relative_to(source).as_posix()
        report["class_source_sha256"] = sha256(class_path)
        report.update(run_probe(StalenessManager))
        loaded = {}
        for name, module in sorted(sys.modules.items()):
            if name == "areal" or name.startswith("areal."):
                file = getattr(module, "__file__", None)
                if file is None:
                    # Upstream areal.utils is a genuine namespace package.
                    namespace_paths = [Path(path).resolve() for path in
                                       getattr(module, "__path__", ())]
                    if not namespace_paths or any(not path.is_relative_to(source)
                                                  for path in namespace_paths):
                        raise ValueError("original_namespace_outside_source: " + name)
                    loaded[name] = {"namespace_paths": [path.relative_to(source).as_posix()
                                                       for path in namespace_paths]}
                    continue
                path = Path(file).resolve()
                if not path.is_relative_to(source) or path.suffix != ".py":
                    raise ValueError("original_module_outside_source: " + name)
                loaded[name] = {"path": path.relative_to(source).as_posix(),
                                "sha256": sha256(path)}
        report["loaded_original_modules"] = loaded
        report["loaded_original_module_count"] = len(loaded)
        dependencies = {}
        for name in ("torch", "numpy", "transformers", "pytest", "hydra-core",
                     "omegaconf", "torchdata", "ray", "openai", "aiohttp",
                     "numba", "llvmlite", "colorlog", "uvloop"):
            try:
                dependencies[name] = version(name)
            except PackageNotFoundError:
                dependencies[name] = None
        report["observed_dependency_versions"] = dependencies
        if args.upstream_tests:
            command = [sys.executable, "-m", "pytest",
                str(source / "tests/test_staleness_manager.py"), "-q",
                "-p", "no:cacheprovider",
                "--junitxml=" + str(output / "official-tests.xml")]
            environment = dict(os.environ)
            environment["PYTHONPATH"] = os.pathsep.join([str(source)] +
                [str(Path(path).resolve()) for path in
                 environment.get("PYTHONPATH", "").split(os.pathsep) if path])
            environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
            test_start = time.perf_counter()
            test = subprocess.run(command, cwd=source, env=environment,
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, timeout=120)
            (output / "official-tests.log").write_text(test.stdout)
            report["official_tests"] = {"command": command, "returncode": test.returncode,
                "wall_ms": (time.perf_counter() - test_start) * 1000,
                "log_sha256": sha256(output / "official-tests.log")}
            if test.returncode != 0:
                raise RuntimeError("unchanged_official_cpu_tests_failed")
        checked_source(source)
        report["passed"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    report["wall_ms"] = (time.perf_counter() - started) * 1000
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--upstream-tests", action="store_true")
    result = main(parser.parse_args())
    print(json.dumps({key: value for key, value in result.items()
                      if key != "loaded_original_modules"}, indent=2))
    raise SystemExit(0 if result["passed"] else 2)
