"""Untimed guest diagnostic for the opt-in stateless verifier.

The host runs this after the comparative benchmark. It instruments the same
`run_one` implementation by wrapping two functions, without changing the
timed verifier path or uploading any expected outputs to the guest.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import re
import sys
import time


spec = importlib.util.spec_from_file_location(
    "fpb_guest_stateless_verifier", "/fpb_guest_stateless_verifier.py")
if spec is None or spec.loader is None:
    raise RuntimeError("Trusted guest helper is missing")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def _summary(values):
    values = sorted(values)
    count = len(values)
    median = (values[(count - 1) // 2] + values[count // 2]) / 2
    p95 = values[min(count - 1, (95 * count + 99) // 100 - 1)]
    return {"count": count, "median_ms": round(median / 1e6, 6),
            "p95_ms": round(p95 / 1e6, 6),
            "sum_ms": round(sum(values) / 1e6, 6)}


def profile_batch(digest):
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Expected case-code SHA-256")
    raw = helper.BATCH_FILE.read_bytes()
    if not 0 < len(raw) <= helper.MAX_BATCH_FILE or hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("Trusted case code differs")
    codes = json.loads(raw)
    if not isinstance(codes, list) or not 1 <= len(codes) <= 32:
        raise ValueError("Invalid trusted case-code list")
    original_capture = helper._capture_candidate
    original_namespace = helper._namespace_case

    def capture(code, workspace, temp_dir):
        started = time.perf_counter_ns()
        result = original_capture(code, workspace, temp_dir)
        result["_candidate_ns"] = time.perf_counter_ns() - started
        return result

    def namespace(libc, code, pool):
        started = time.perf_counter_ns()
        result = original_namespace(libc, code, pool)
        result["_namespace_ns"] = time.perf_counter_ns() - started
        return result

    helper._capture_candidate = capture
    helper._namespace_case = namespace
    total_ns, namespace_ns, candidate_ns, outputs = [], [], [], []
    for encoded in codes:
        started = time.perf_counter_ns()
        result = helper.run_one(encoded)
        total_ns.append(time.perf_counter_ns() - started)
        namespace_ns.append(result.pop("_namespace_ns"))
        candidate_ns.append(result.pop("_candidate_ns"))
        output = base64.b64decode(result["stdout_b64"], validate=True)
        outputs.append({"returncode": result["return_code"],
                        "stdout_sha256": hashlib.sha256(output).hexdigest(),
                        "truncated": result["truncated"]})
    outer_ns = [max(0, total - private) for total, private in zip(total_ns, namespace_ns)]
    mount_and_launch_ns = [max(0, private - candidate)
                           for private, candidate in zip(namespace_ns, candidate_ns)]
    return {"kind": "guest_stateless_case_profile_v1", "case_count": len(codes),
            "per_case": {"whole_run_one": _summary(total_ns),
                         "outer_supervision_and_cleanup": _summary(outer_ns),
                         "namespace_setup_and_non_candidate_work": _summary(mount_and_launch_ns),
                         "candidate_process_and_python_execution": _summary(candidate_ns)},
            "outputs": outputs}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: guest_stateless_profile.py SHA256")
    report = profile_batch(sys.argv[1])
    encoded = base64.b64encode(json.dumps(report, separators=(",", ":")).encode()).decode()
    print("FPB_STATELESS_PROFILE=" + encoded, flush=True)
