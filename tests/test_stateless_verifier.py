"""Offline gates for the opt-in, host-private stateless verifier path."""

import base64
import contextlib
import hashlib
import io
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from future_prediction_bench import guest_stateless_verifier
from future_prediction_bench.stateless_verifier import (
    BATCH_OVERFLOW_MARKER, GUEST_PROGRAM, StatelessCaseVerifier,
    validate_stateless_contract,
    validate_python_argv,
)


class FakeGuest:
    def __init__(self):
        self.commands = []
        self.batch_data = b""
        self.outputs = {"print('private')": b"private\n",
                        "print('second')": b"second\n"}

    def run_shell(self, command, *, timeout):
        self.commands.append(command)
        if command.startswith(": > /mnt/root/fpb_stateless_cases.json"):
            self.batch_data = b""
            return {"return_code": 0, "stdout": ""}
        if command.endswith(">> /mnt/root/fpb_stateless_cases.json"):
            encoded = re.search(r"printf '%s' '([^']+)'", command).group(1)
            self.batch_data += base64.b64decode(encoded)
            return {"return_code": 0, "stdout": ""}
        if command.startswith("sha256sum /mnt/root/fpb_stateless_cases.json"):
            digest = hashlib.sha256(self.batch_data).hexdigest()
            return {"return_code": 0, "stdout": digest + "  file\n"}
        if command.startswith("sha256sum /mnt/root/fpb_guest_stateless_verifier.py"):
            digest = hashlib.sha256(GUEST_PROGRAM.read_bytes()).hexdigest()
            return {"return_code": 0, "stdout": digest + "  file\n"}
        if "/fpb_guest_stateless_verifier.py --batch " in command:
            cases = json.loads(self.batch_data)
            payloads = []
            for encoded in cases:
                code = base64.b64decode(encoded).decode()
                stdout = self.outputs.get(code, b"")
                payloads.append({"return_code": 0,
                                 "stdout_b64": base64.b64encode(stdout).decode(),
                                 "truncated": False})
            marker = base64.b64encode(json.dumps(payloads).encode()).decode()
            return {"return_code": 0, "stdout": "FPB_STATELESS_BATCH_RESULT=" + marker}
        if "/fpb_guest_stateless_verifier.py '" in command:
            encoded = command.rsplit("'", 2)[1]
            code = base64.b64decode(encoded).decode()
            stdout = self.outputs.get(code, b"")
            payload = {"return_code": 0, "stdout_b64": base64.b64encode(stdout).decode(),
                       "truncated": False}
            marker = base64.b64encode(json.dumps(payload).encode()).decode()
            return {"return_code": 0, "stdout": "FPB_STATELESS_RESULT=" + marker}
        return {"return_code": 0, "stdout": ""}


class StatelessVerifierTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "verifier").mkdir()
        self.cases = [
            {"argv": ["python3", "-B", "-c", "print('private')"],
             "expected_stdout": "private\n", "expected_returncode": 0},
            {"argv": ["python3", "-B", "-c", "print('second')"],
             "expected_stdout": "second\n", "expected_returncode": 0},
        ]
        self.verifier = self.root / "verifier" / "verify.json"
        self.verifier.write_text(json.dumps({"kind": "command_cases_v1", "cases": self.cases}))
        (self.root / "task.json").write_text(json.dumps({"task_id": "fixture"}))
        self.contract = self.root / "contract.json"
        self.valid_contract = {
            "kind": "stateless_python_cases_overlay_v1", "task_id": "fixture",
            "verifier_sha256": hashlib.sha256(self.verifier.read_bytes()).hexdigest(),
            "requires_live_background_process_state": False,
            "requires_shared_case_filesystem_state": False,
            "requires_quiescent_submitted_state": True,
            "allow_unprivileged_case_execution": True,
        }
        self.contract.write_text(json.dumps(self.valid_contract))

    def test_rejects_missing_or_expanded_contract(self):
        for key, value in (
            ("requires_live_background_process_state", True),
            ("requires_shared_case_filesystem_state", True),
            ("requires_quiescent_submitted_state", False),
            ("allow_unprivileged_case_execution", False),
            ("verifier_sha256", "0" * 64),
            ("task_id", "another-task"),
        ):
            with self.subTest(key=key):
                bad = dict(self.valid_contract)
                bad[key] = value
                self.contract.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    validate_stateless_contract(self.root, self.contract)
        self.contract.write_text(json.dumps(self.valid_contract))
        self.verifier.write_text(self.verifier.read_text() + " ")
        with self.assertRaises(ValueError):
            validate_stateless_contract(self.root, self.contract)

    def test_host_keeps_expectations_private_and_compares_exact_bytes(self):
        guest = FakeGuest()
        verifier = StatelessCaseVerifier(guest, self.root, self.contract)
        verifier.install()
        result = verifier.grade()
        self.assertEqual((result["reward"], result["passed_cases"]), (1.0, 2))
        self.assertNotIn("expected_stdout", "".join(guest.commands))
        self.assertNotIn("second\\n", "".join(guest.commands))
        self.assertTrue(any("modprobe overlay" in command for command in guest.commands))
        self.assertTrue(any("/mnt/root/tmp" in command for command in guest.commands))
        self.assertTrue(all(len(command.encode("utf-8")) <= 3800
                            for command in guest.commands))
        self.assertEqual(sum("/fpb_guest_stateless_verifier.py '" in command
                             for command in guest.commands), 2)
        self.assertTrue(all(" -I -B /fpb_guest_stateless_verifier.py" in command
                            for command in guest.commands
                            if "/fpb_guest_stateless_verifier.py '" in command))
        verifier.prepare_batch()
        self.assertEqual(verifier.grade_batch(), result)
        self.assertFalse(verifier.batch_fallback_used)
        uploaded = json.loads(guest.batch_data)
        self.assertEqual([base64.b64decode(code).decode() for code in uploaded],
                         [case["argv"][3] for case in self.cases])
        self.assertNotIn(b"expected_stdout", guest.batch_data)
        self.assertEqual(sum("/fpb_guest_stateless_verifier.py --batch " in command
                             for command in guest.commands), 1)
        self.assertTrue(any(" -I -B /fpb_guest_stateless_verifier.py --batch " in command
                            for command in guest.commands))
        # Larger source uploads still reconstruct the exact case-code JSON
        # while staying within the serial shell transport's line bound.
        long_codes = ["# " + "界" * 350 + "\nprint(1)" for _ in range(10)]
        verifier.prepare_batch_codes(long_codes)
        self.assertEqual([base64.b64decode(code).decode("utf-8")
                          for code in json.loads(guest.batch_data)], long_codes)
        self.assertTrue(all(len(command.encode("utf-8")) <= 3800
                            for command in guest.commands))
        # The host checks exact UTF-8 bytes, not a lossy-decoded observation.
        spec = {"kind": "command_cases_v1", "cases": [dict(self.cases[0])]}
        spec["cases"][0]["expected_stdout"] = "different\n"
        self.verifier.write_text(json.dumps(spec))
        contract = dict(self.valid_contract)
        contract["verifier_sha256"] = hashlib.sha256(self.verifier.read_bytes()).hexdigest()
        self.contract.write_text(json.dumps(contract))
        self.assertEqual(StatelessCaseVerifier(FakeGuest(), self.root, self.contract)
                         .cases[0]["expected_stdout"], "different\n")

    def test_bounded_python_case_contract(self):
        self.assertEqual(validate_python_argv(["python3", "-B", "-c", "print(1)"]),
                         "print(1)")
        for bad in (["sh", "-c", "print(1)"],
                    ["python3", "-B", "-c", ""],
                    ["python3", "-B", "-c", "x" * 2049],
                    ["python3", "-B", "-c", "# " + "界" * 1100 + "\nprint(1)"]):
            with self.assertRaises(ValueError):
                validate_python_argv(bad)

    def test_multibyte_case_rejected_before_guest_work(self):
        too_many_bytes = "# " + "界" * 1100 + "\nprint(1)"
        spec = {"kind": "command_cases_v1", "cases": [
            {"argv": ["python3", "-B", "-c", too_many_bytes],
             "expected_stdout": "1\n", "expected_returncode": 0}]}
        self.verifier.write_text(json.dumps(spec))
        contract = dict(self.valid_contract)
        contract["verifier_sha256"] = hashlib.sha256(self.verifier.read_bytes()).hexdigest()
        self.contract.write_text(json.dumps(contract))
        with self.assertRaisesRegex(ValueError, "transport bound"):
            StatelessCaseVerifier(FakeGuest(), self.root, self.contract)

    def test_excessive_case_group_rejected_before_guest_work(self):
        code = "# " + "界" * 790 + "\nprint(1)"
        self.assertEqual(validate_python_argv(["python3", "-B", "-c", code]), code)
        cases = [{"argv": ["python3", "-B", "-c", code],
                  "expected_stdout": "1\n", "expected_returncode": 0}
                 for _ in range(32)]
        self.verifier.write_text(json.dumps({"kind": "command_cases_v1", "cases": cases}))
        contract = dict(self.valid_contract)
        contract["verifier_sha256"] = hashlib.sha256(self.verifier.read_bytes()).hexdigest()
        self.contract.write_text(json.dumps(contract))
        guest = FakeGuest()
        with self.assertRaisesRegex(ValueError, "Unsupported host verifier specification"):
            StatelessCaseVerifier(guest, self.root, self.contract)
        self.assertEqual(guest.commands, [])

    def test_reserved_timeout_return_code_cannot_be_an_expected_result(self):
        spec = {"kind": "command_cases_v1", "cases": [
            {"argv": ["python3", "-B", "-c", "print('private')"],
             "expected_stdout": "", "expected_returncode": 124}]}
        self.verifier.write_text(json.dumps(spec))
        contract = dict(self.valid_contract)
        contract["verifier_sha256"] = hashlib.sha256(self.verifier.read_bytes()).hexdigest()
        self.contract.write_text(json.dumps(contract))
        with self.assertRaisesRegex(ValueError, "Unsupported hidden case"):
            StatelessCaseVerifier(FakeGuest(), self.root, self.contract)

    def test_oversized_guest_batch_requests_exact_serial_fallback(self):
        class OverflowGuest(FakeGuest):
            def run_shell(self, command, *, timeout):
                if "/fpb_guest_stateless_verifier.py --batch " in command:
                    self.commands.append(command)
                    return {"return_code": 0, "stdout": BATCH_OVERFLOW_MARKER}
                return super().run_shell(command, timeout=timeout)

        cases = [dict(self.cases[0]) for _ in range(14)]
        self.verifier.write_text(json.dumps({"kind": "command_cases_v1", "cases": cases}))
        contract = dict(self.valid_contract)
        contract["verifier_sha256"] = hashlib.sha256(self.verifier.read_bytes()).hexdigest()
        self.contract.write_text(json.dumps(contract))
        guest = OverflowGuest()
        guest.outputs["print('private')"] = b"x" * 8000
        verifier = StatelessCaseVerifier(guest, self.root, self.contract)
        verifier.install()
        verifier.prepare_batch()
        result = verifier.grade_batch()
        self.assertEqual((result["reward"], result["passed_cases"], result["case_count"]),
                         (0.0, 0, 14))
        self.assertTrue(verifier.batch_fallback_used)
        self.assertEqual(sum("/fpb_guest_stateless_verifier.py '" in command
                             for command in guest.commands), 14)
        self.assertEqual(sum("--batch" in command for command in guest.commands), 1)

    def test_guest_batch_emits_oversize_marker_without_crashing(self):
        codes = [base64.b64encode(b"print(1)").decode("ascii") for _ in range(14)]
        batch_path = self.root / "cases.json"
        raw = json.dumps(codes).encode("ascii")
        batch_path.write_bytes(raw)
        batch_path.chmod(0o600)
        digest = hashlib.sha256(raw).hexdigest()
        verbose = {"return_code": 0,
                   "stdout_b64": base64.b64encode(b"x" * 8000).decode("ascii"),
                   "truncated": False}
        output = io.StringIO()
        with patch.object(guest_stateless_verifier, "BATCH_FILE", batch_path), \
                patch.object(guest_stateless_verifier, "run_one", return_value=verbose), \
                contextlib.redirect_stdout(output):
            guest_stateless_verifier.main(["--batch", digest])
        self.assertEqual(output.getvalue(), BATCH_OVERFLOW_MARKER + "\n")

    def test_host_contract_rechecked_at_grading(self):
        guest = FakeGuest()
        verifier = StatelessCaseVerifier(guest, self.root, self.contract)
        verifier.install()
        verifier.prepare_batch()
        self.verifier.write_text(self.verifier.read_text() + " ")
        with self.assertRaises(RuntimeError):
            verifier.grade()
        with self.assertRaises(RuntimeError):
            verifier.grade_batch()


if __name__ == "__main__":
    unittest.main()
