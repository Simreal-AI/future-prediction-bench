"""Host-private, opt-in stateless verifier for a pinned coding fixture.

This experimental fast path is valid only when the operator supplies a
checksum-bound contract declaring that hidden cases need neither a preceding
case's filesystem changes nor live background-process state. It does not
replace MicroVMCodingAdapter's full-VM verifier for general tasks.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from pathlib import Path

from .http import strict_json_loads
from .microvm_runtime import _sha256_file


GUEST_PROGRAM = Path(__file__).with_name("guest_stateless_verifier.py")
GUEST_TARGET = "/mnt/root/fpb_guest_stateless_verifier.py"
BATCH_TARGET = "/mnt/root/fpb_stateless_cases.json"
GUEST_PYTHON = "/usr/local/bin/python3.12"
GUEST_ENV = "LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/aarch64-linux-gnu"
MAX_CASES = 18  # 18 * (10 s candidate + 5 s supervisor) < 300 s host cap.
CONTRACT_KEYS = {"kind", "task_id", "verifier_sha256",
                 "requires_live_background_process_state",
                 "requires_shared_case_filesystem_state",
                 "requires_quiescent_submitted_state",
                 "allow_unprivileged_case_execution"}
BATCH_OVERFLOW_MARKER = "FPB_STATELESS_BATCH_RESULT_OVERSIZE=1"


class StatelessBatchOverflow(RuntimeError):
    """The guest's bounded batch transport cannot hold all captured stdout."""


def validate_stateless_contract(task_dir, contract_path):
    """Fail closed unless a task-specific, host-authored declaration matches."""
    task_dir, contract_path = Path(task_dir).resolve(), Path(contract_path)
    if contract_path.is_symlink() or not contract_path.is_file():
        raise ValueError("Stateless verifier contract must be a regular file")
    verifier_file = task_dir / "verifier" / "verify.json"
    if verifier_file.is_symlink() or not verifier_file.is_file():
        raise ValueError("Host verifier must be a regular file")
    contract = strict_json_loads(contract_path.read_text(encoding="utf-8"))
    task = strict_json_loads((task_dir / "task.json").read_text(encoding="utf-8"))
    if (not isinstance(contract, dict) or set(contract) != CONTRACT_KEYS
            or contract["kind"] != "stateless_python_cases_overlay_v1"
            or contract["task_id"] != task.get("task_id")
            or contract["verifier_sha256"] != _sha256_file(verifier_file)
            or contract["requires_live_background_process_state"] is not False
            or contract["requires_shared_case_filesystem_state"] is not False
            or contract["requires_quiescent_submitted_state"] is not True
            or contract["allow_unprivileged_case_execution"] is not True):
        raise ValueError("Task does not declare the exact stateless verifier contract")
    specification = strict_json_loads(verifier_file.read_text(encoding="utf-8"))
    if (not isinstance(specification, dict) or set(specification) != {"kind", "cases"}
            or specification["kind"] != "command_cases_v1"
            or not isinstance(specification["cases"], list)
            or not 1 <= len(specification["cases"]) <= MAX_CASES):
        raise ValueError("Unsupported host verifier specification")
    cases = specification["cases"]
    codes = []
    for case in cases:
        if (not isinstance(case, dict)
                or set(case) != {"argv", "expected_stdout", "expected_returncode"}
                or not isinstance(case["expected_stdout"], str)
                or len(case["expected_stdout"].encode("utf-8")) > 12000
                or type(case["expected_returncode"]) is not int
                or not 0 <= case["expected_returncode"] <= 123):
            raise ValueError("Unsupported hidden case")
        codes.append(validate_python_argv(case["argv"]))
    # The guest consumes the same compact JSON payload prepared below. Check
    # its aggregate bound before starting a VM, not after a policy submits.
    _batch_payload(codes)
    return contract, task, cases


def validate_python_argv(argv):
    if (not isinstance(argv, (list, tuple)) or len(argv) != 4
            or tuple(argv[:3]) != ("python3", "-B", "-c")
            or not isinstance(argv[3], str) or not argv[3]
            or len(argv[3]) > 2048 or "\x00" in argv[3]):
        raise ValueError("Only bounded Python -B -c cases are supported")
    # The guest receives this exact source as base64. The stricter host bound
    # leaves room for the shell command and serial framing inside a canonical
    # 4096-byte TTY input line. Character count alone lets multibyte UTF-8 pass host
    # validation only to fail after the episode has already started.
    if len(base64.b64encode(argv[3].encode("utf-8"))) > 3600:
        raise ValueError("Python case exceeds guest source transport bound")
    return argv[3]


def _batch_payload(codes):
    encoded = [base64.b64encode(code.encode("utf-8")).decode("ascii")
               for code in codes]
    data = json.dumps(encoded, separators=(",", ":")).encode("ascii")
    if not 0 < len(data) <= 100_000:
        raise ValueError("Batch case code exceeds upload bound")
    return data


class StatelessCaseVerifier:
    """Trusted host driver; all expected output remains in the host process."""

    def __init__(self, runtime, task_dir, contract_path):
        self.runtime = runtime
        self.contract, self.task, self.cases = validate_stateless_contract(
            task_dir, contract_path)
        self.verifier_file = Path(task_dir).resolve() / "verifier" / "verify.json"
        self.contract_file = Path(contract_path)
        self.contract_sha256 = _sha256_file(self.contract_file)
        self.installed = False
        self.program_sha256 = None
        self.batch_code_sha256 = None
        self.batch_fallback_used = False

    def _check_host_contract(self):
        try:
            changed = (self.verifier_file.is_symlink() or self.contract_file.is_symlink()
                       or _sha256_file(self.verifier_file) != self.contract["verifier_sha256"]
                       or _sha256_file(self.contract_file) != self.contract_sha256)
        except OSError as exc:
            raise RuntimeError("Host stateless verifier contract is unavailable") from exc
        if changed:
            raise RuntimeError("Host stateless verifier contract changed after binding")

    def _required(self, command, *, timeout=30):
        result = self.runtime.run_shell(command, timeout=timeout)
        if result["return_code"] != 0:
            raise RuntimeError("Trusted guest stateless verifier command failed: "
                               + result["stdout"][-400:])
        return result["stdout"]

    def install(self):
        if self.installed:
            return self.program_sha256
        self._required("modprobe overlay")
        self._required("mkdir -p /mnt/root/dev /mnt/root/tmp && chmod 1777 /mnt/root/tmp && "
                       "(test -c /mnt/root/dev/null || mknod -m 666 /mnt/root/dev/null c 1 3)")
        source = GUEST_PROGRAM.read_bytes()
        if not 0 < len(source) <= 30000:
            raise ValueError("Trusted guest program exceeds upload bound")
        self._required(f": > {GUEST_TARGET}")
        # 2700 raw bytes become 3600 base64 characters. The complete command
        # plus runtime framing stays below the guest canonical TTY line cap.
        for offset in range(0, len(source), 2700):
            block = base64.b64encode(source[offset:offset + 2700]).decode("ascii")
            self._required(f"printf '%s' '{block}' | base64 -d >> {GUEST_TARGET}")
        digest = hashlib.sha256(source).hexdigest()
        if self._required(f"sha256sum {GUEST_TARGET}").split()[0] != digest:
            raise RuntimeError("Uploaded guest program digest differs")
        self.program_sha256 = digest
        self.installed = True
        return digest

    def run_case(self, argv):
        if not self.installed:
            raise RuntimeError("Install the trusted guest helper before grading")
        code = validate_python_argv(argv)
        encoded = base64.b64encode(code.encode("utf-8")).decode("ascii")
        command = (f"{GUEST_ENV} chroot /mnt/root {GUEST_PYTHON} -I -B "
                   f"/fpb_guest_stateless_verifier.py '{encoded}'")
        if len(command.encode("utf-8")) > 3800:
            raise ValueError("Guest case command exceeds transport bound")
        output = self._required(command, timeout=25)
        match = re.fullmatch(r"FPB_STATELESS_RESULT=([A-Za-z0-9+/=]+)\n?", output)
        if not match:
            raise RuntimeError("Guest stateless result framing failed")
        payload = self._decode_result(match.group(1))
        return self._parse_case_result(payload)

    @staticmethod
    def _decode_result(encoded):
        try:
            return strict_json_loads(base64.b64decode(encoded, validate=True).decode())
        except (ValueError, UnicodeError) as exc:
            raise RuntimeError("Guest stateless result is invalid") from exc

    @staticmethod
    def _parse_case_result(payload):
        if (not isinstance(payload, dict)
                or set(payload) != {"return_code", "stdout_b64", "truncated"}
                or type(payload["return_code"]) is not int
                or not -255 <= payload["return_code"] <= 255
                or not isinstance(payload["stdout_b64"], str)
                or len(payload["stdout_b64"]) > 16000
                or type(payload["truncated"]) is not bool):
            raise RuntimeError("Guest stateless case result is invalid")
        try:
            stdout = base64.b64decode(payload["stdout_b64"], validate=True)
            if len(stdout) > 12000:
                raise ValueError("Guest case stdout exceeds bound")
        except ValueError as exc:
            raise RuntimeError("Guest stateless case result is invalid") from exc
        return {"return_code": payload["return_code"], "stdout_bytes": stdout,
                "truncated": payload["truncated"]}

    def prepare_batch_codes(self, codes):
        """Upload trusted case source only; never put expected outputs in the VM."""
        self._check_host_contract()
        if not self.installed:
            raise RuntimeError("Install the trusted guest helper first")
        if (not isinstance(codes, (list, tuple)) or not 1 <= len(codes) <= MAX_CASES
                or any(not isinstance(code, str) or not code or len(code) > 2048
                       or "\x00" in code
                       or len(base64.b64encode(code.encode("utf-8"))) > 3600
                       for code in codes)):
            raise ValueError("Batch needs 1..18 bounded Python source strings")
        data = _batch_payload(codes)
        self._required(f": > {BATCH_TARGET} && chmod 600 {BATCH_TARGET}")
        for offset in range(0, len(data), 2700):
            block = base64.b64encode(data[offset:offset + 2700]).decode("ascii")
            self._required(f"printf '%s' '{block}' | base64 -d >> {BATCH_TARGET}")
        digest = hashlib.sha256(data).hexdigest()
        if self._required(f"sha256sum {BATCH_TARGET}").split()[0] != digest:
            raise RuntimeError("Uploaded guest case code digest differs")
        self.batch_code_sha256 = digest
        return digest

    def prepare_batch(self):
        """Upload the contract-bound verifier's case code before checkpointing."""
        return self.prepare_batch_codes(
            [validate_python_argv(case["argv"]) for case in self.cases])

    def run_batch(self, *, expected_count=None):
        if not self.installed or self.batch_code_sha256 is None:
            raise RuntimeError("Upload fixed case code before batch grading")
        command = (f"{GUEST_ENV} chroot /mnt/root {GUEST_PYTHON} -I -B "
                   "/fpb_guest_stateless_verifier.py --batch " + self.batch_code_sha256)
        output = self._required(command, timeout=min(300, len(self.cases) * 15 + 10))
        if output.rstrip("\n") == BATCH_OVERFLOW_MARKER:
            raise StatelessBatchOverflow("Guest batch response exceeded transport bound")
        match = re.fullmatch(r"FPB_STATELESS_BATCH_RESULT=([A-Za-z0-9+/=]+)\n?", output)
        if not match:
            raise RuntimeError("Guest stateless batch result framing failed")
        payload = self._decode_result(match.group(1))
        if expected_count is None:
            expected_count = len(self.cases)
        if not isinstance(payload, list) or len(payload) != expected_count:
            raise RuntimeError("Guest stateless batch case count differs")
        return [self._parse_case_result(item) for item in payload]

    def _score(self, outputs):
        results = []
        for case, actual in zip(self.cases, outputs, strict=True):
            expected = case["expected_stdout"].encode("utf-8")
            results.append({"passed": (not actual["truncated"]
                                       and actual["return_code"] == case["expected_returncode"]
                                       and actual["stdout_bytes"] == expected),
                            "returncode": actual["return_code"],
                            "stdout_sha256": hashlib.sha256(actual["stdout_bytes"]).hexdigest()})
        return {"reward": 1.0 if all(item["passed"] for item in results) else 0.0,
                "passed_cases": sum(item["passed"] for item in results),
                "case_count": len(results), "case_results": results}

    def grade(self):
        self._check_host_contract()
        return self._score([self.run_case(case["argv"]) for case in self.cases])

    def grade_batch(self):
        self._check_host_contract()
        self.batch_fallback_used = False
        try:
            outputs = self.run_batch()
        except StatelessBatchOverflow:
            # Every case still runs in its own mount/PID namespace. Rerun via
            # bounded one-case serial calls so wrong, verbose candidate output
            # becomes a scored failure instead of an untrainable pending item.
            self.batch_fallback_used = True
            outputs = [self.run_case(case["argv"]) for case in self.cases]
        return self._score(outputs)
