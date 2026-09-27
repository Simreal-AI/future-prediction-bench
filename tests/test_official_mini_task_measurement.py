"""Keep the published live MiniSandbox result bound to its tested checker."""

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs/measurements/official_mini_flask_5014_grade_2026-09-25.json"


class OfficialMiniTaskMeasurementTests(unittest.TestCase):
    def test_live_grade_report_matches_checker_sources_and_case_map(self):
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        source_files = {
            "run_script_sha256": "examples/official_mini_sandbox/run_official_flask_5014.py",
            "strict_grade_sha256": "examples/official_mini_sandbox/strict_grade.py",
            "session_bridge_sha256": "examples/official_mini_sandbox/session_bridge.py",
        }
        for key, relative in source_files.items():
            with self.subTest(source=relative):
                self.assertEqual(
                    report["source_binding"][key],
                    hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(),
                )
        self.assertEqual(report["sample_count"], {"baseline": 1, "official_patch": 1})
        self.assertEqual(report["task"]["fail_to_pass_cases"], 1)
        self.assertEqual(report["task"]["pass_to_pass_cases"], 59)
        for arm, expected in {
            "baseline": (0, 59, 1),
            "official_patch": (1, 60, 0),
        }.items():
            with self.subTest(arm=arm):
                grade = report["grades"][arm]
                self.assertEqual(
                    (grade["reward"], grade["passed_cases"], grade["failed_cases"]),
                    expected,
                )
                self.assertEqual(grade["expected_cases_observed"], 60)


if __name__ == "__main__":
    unittest.main()
