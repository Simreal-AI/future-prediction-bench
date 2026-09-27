"""Host acceptance checks for the guest-local COW measurement contract."""

import base64
import json
import unittest

from examples.realworld_boltons26.benchmark_guest_cow import _parse_guest_report
from future_prediction_bench.guest_cow_branch import _summarize_ns


def _framed(report):
    encoded = base64.b64encode(json.dumps(report).encode()).decode()
    return "FPB_COW_RESULT:" + encoded + "\n"


class GuestCOWContractTests(unittest.TestCase):
    def _report(self):
        timing = {"count": 10, "median_ms": 1.0, "p95_ms": 2.0,
                  "min_ms": 0.5, "max_ms": 3.0}
        return {"kind": "guest_linux_fork_overlayfs_microbenchmark_v1",
                "repetitions": 10,
                "correctness": {"parent_glass": "glas",
                                "fix_branch": {"glass": "glass"},
                                "baseline_branch": {"glass": "glas"},
                                "parent_counter": 7,
                                "shared_lower_sha256_unchanged": True,
                                "isolated_overlay_writes": True},
                "timings": {name: dict(timing) for name in (
                    "overlay_create", "fork_ready", "process_reap",
                    "overlay_rollback", "rollback_total", "combined_branch_cycle")}}

    def test_guest_clock_summary_reports_median_and_tail(self):
        summary = _summarize_ns([4_000_000, 1_000_000, 3_000_000, 2_000_000])
        self.assertEqual(summary, {"count": 4, "median_ms": 2.5,
                                   "p95_ms": 4.0, "min_ms": 1.0, "max_ms": 4.0})

    def test_host_accepts_only_verified_baseline_fix_and_full_timing_contract(self):
        report = self._report()
        self.assertEqual(_parse_guest_report(_framed(report), 10), report)
        report["correctness"]["isolated_overlay_writes"] = False
        with self.assertRaisesRegex(RuntimeError, "isolation"):
            _parse_guest_report(_framed(report), 10)
        report = self._report()
        report["timings"]["combined_branch_cycle"]["count"] = 9
        with self.assertRaisesRegex(RuntimeError, "timing"):
            _parse_guest_report(_framed(report), 10)

    def test_host_rejects_missing_or_duplicate_result_markers(self):
        report = self._report()
        with self.assertRaisesRegex(RuntimeError, "framing"):
            _parse_guest_report("nothing", 10)
        with self.assertRaisesRegex(RuntimeError, "framing"):
            _parse_guest_report(_framed(report) * 2, 10)


if __name__ == "__main__":
    unittest.main()
