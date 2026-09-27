"""Contract and syscall-order checks for the private mount-view experiment."""

import base64
import ctypes
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from examples.realworld_boltons26.benchmark_guest_mount_namespace import _parse_report
from future_prediction_bench import guest_mount_namespace as guest


class _FakeMountAPI:
    def __init__(self, *, unshare_result=0):
        self.calls = []
        self.unshare_result = unshare_result

    def unshare(self, flags):
        self.calls.append(("unshare", flags))
        return self.unshare_result

    def mount(self, source, target, filesystem, flags, options):
        self.calls.append(("mount", source, target, filesystem, flags, options))
        return 0


def _framed(report):
    return "FPB_MOUNT_NS_RESULT:" + base64.b64encode(json.dumps(report).encode()).decode() + "\n"


def _report():
    timing = {"count": 10, "median_ms": 0.5, "p95_ms": 1.0,
              "min_ms": 0.1, "max_ms": 2.0}
    return {"kind": "guest_linux_mount_namespace_overlayfs_microbenchmark_v1",
            "repetitions": 10,
            "correctness": {"parent_namespace": "mnt:[1]",
                            "parent_upper_device": 10,
                            "fix_branch": {"glass": "glass",
                                           "mount_namespace": "mnt:[2]",
                                           "upper_device": 11},
                            "baseline_branch": {"glass": "glas",
                                                "mount_namespace": "mnt:[3]",
                                                "upper_device": 12},
                            "distinct_child_mount_namespaces": True,
                            "parent_view_empty": True,
                            "shared_lower_sha256_unchanged": True,
                            "parent_heap_unchanged": True},
            "timings": {name: dict(timing) for name in (
                "unshare_ns", "private_propagation_ns", "tmpfs_mount_ns",
                "overlay_mount_ns", "cleanup_mounts_ns", "fork_to_ready_ns",
                "release_to_reap_ns", "complete_cycle_ns")}}


class GuestMountNamespaceTests(unittest.TestCase):
    def test_mount_namespace_is_created_before_private_overlay(self):
        fake = _FakeMountAPI()
        with tempfile.TemporaryDirectory() as root:
            upper, view = Path(root) / "upper", Path(root) / "view"
            upper.mkdir()
            view.mkdir()
            timings = guest._setup_private_view(fake, upper, view)
            self.assertEqual(fake.calls[0], ("unshare", guest.CLONE_NEWNS))
            self.assertEqual(fake.calls[1][:5],
                             ("mount", None, b"/", None, guest.MS_REC | guest.MS_PRIVATE))
            self.assertEqual(fake.calls[2][1:4],
                             (b"tmpfs", bytes(upper), b"tmpfs"))
            self.assertEqual(fake.calls[2][4],
                             guest.MS_NOSUID | guest.MS_NODEV | guest.MS_NOEXEC)
            self.assertEqual(fake.calls[2][5], b"size=64m,mode=0700")
            self.assertEqual(fake.calls[3][1], b"overlay")
            self.assertEqual(fake.calls[3][2], bytes(view))
            self.assertEqual(set(timings), {"unshare_ns", "private_propagation_ns",
                                            "tmpfs_mount_ns", "overlay_mount_ns"})

    def test_unshare_failure_stops_before_mounting(self):
        fake = _FakeMountAPI(unshare_result=-1)
        with tempfile.TemporaryDirectory() as root:
            with patch.object(ctypes, "get_errno", return_value=1):
                with self.assertRaises(OSError):
                    guest._setup_private_view(fake, Path(root) / "upper",
                                              Path(root) / "view")
        self.assertEqual(fake.calls, [("unshare", guest.CLONE_NEWNS)])

    def test_host_accepts_only_complete_verified_report(self):
        report = _report()
        self.assertEqual(_parse_report(_framed(report), 10), report)
        report["correctness"]["parent_view_empty"] = False
        with self.assertRaisesRegex(RuntimeError, "isolation"):
            _parse_report(_framed(report), 10)
        report = _report()
        report["timings"]["complete_cycle_ns"]["count"] = 9
        with self.assertRaisesRegex(RuntimeError, "timing"):
            _parse_report(_framed(report), 10)
        report = _report()
        report["correctness"]["baseline_branch"]["mount_namespace"] = "mnt:[2]"
        with self.assertRaisesRegex(RuntimeError, "identifiers"):
            _parse_report(_framed(report), 10)

    def test_host_rejects_duplicate_or_missing_frame(self):
        with self.assertRaisesRegex(RuntimeError, "framing"):
            _parse_report("no guest result", 10)
        with self.assertRaisesRegex(RuntimeError, "framing"):
            _parse_report(_framed(_report()) * 2, 10)

    def test_timing_summary_has_tail(self):
        self.assertEqual(guest._summary([1_000_000, 2_000_000, 3_000_000, 4_000_000]),
                         {"count": 4, "median_ms": 2.5, "p95_ms": 4.0,
                          "min_ms": 1.0, "max_ms": 4.0})

    def test_published_measurement_binds_current_guest_program(self):
        root = Path(__file__).resolve().parents[1]
        report = json.loads((root / "docs/measurements/guest_mount_namespace_v0.6.0.json")
                            .read_text(encoding="utf-8"))
        source = root / "future_prediction_bench/guest_mount_namespace.py"
        self.assertEqual(report["assets"]["guest_program_sha256"],
                         hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertEqual(report["guest"]["repetitions"], 200)
        self.assertTrue(report["guest"]["correctness"]["distinct_child_mount_namespaces"])


if __name__ == "__main__":
    unittest.main()
