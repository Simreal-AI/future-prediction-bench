"""Offline checks for the SHA-pinned Humanize repository repair fixture."""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path

from examples.realworld_humanize.make_task import (
    ARCHIVE_SHA256, SOURCE_FILE, _cases, _extract, _repair, make_task,
)
from examples.realworld_humanize.smoke import smoke
from future_prediction_bench.realworld import validate_task
from future_prediction_bench.coding_env import _workspace_digest


class HumanizeFixtureTests(unittest.TestCase):
    def test_repair_is_anchor_bound_and_carries_rounded_unit(self):
        anchor = "    exp = int(min(log(abs_bytes, base), len(suffix)))\n"
        original = anchor + "    return format % (bytes_ / (base**exp)) + suffix[exp - 1]\n"
        fixed = _repair(original)
        self.assertIn("abs(float(format % (abs_bytes / (base**exp)))) >= base", fixed)
        self.assertEqual(fixed.count("exp += 1"), 1)
        with self.assertRaisesRegex(ValueError, "repair anchor"):
            _repair("return old_result\n")
        with self.assertRaisesRegex(ValueError, "repair anchor"):
            _repair(anchor + anchor)

    def test_cases_cover_boundary_and_nonboundary(self):
        cases = _cases()
        self.assertEqual(len(cases), 14)
        self.assertEqual(len({(value, tuple(sorted(options.items()))) for value, options, _ in cases}), 14)
        self.assertTrue(any(value < 0 for value, _, _ in cases))
        self.assertTrue(any(options.get("binary") for _, options, _ in cases))
        self.assertTrue(any(options.get("gnu") for _, options, _ in cases))
        self.assertTrue(any(options.get("format") == "%.3f" for _, options, _ in cases))

    def test_extract_rejects_path_traversal(self):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            payload = b"not source"
            member = tarfile.TarInfo("humanize-4.15.0/../../escape")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                _extract(buffer.getvalue(), Path(tmp) / "seed")
            self.assertFalse((Path(tmp) / "escape").exists())

    @unittest.skipUnless(os.environ.get("FPB_HUMANIZE_SDIST"),
                         "Set FPB_HUMANIZE_SDIST to run the pinned-source integration check")
    def test_pinned_archive_baseline_and_repair_subprocesses(self):
        archive = Path(os.environ["FPB_HUMANIZE_SDIST"])
        self.assertEqual(__import__("hashlib").sha256(archive.read_bytes()).hexdigest(), ARCHIVE_SHA256)
        with tempfile.TemporaryDirectory() as tmp:
            task_root = Path(tmp) / "task"
            make_task(task_root, sdist=archive)
            task = json.loads((task_root / "task.json").read_text(encoding="utf-8"))
            self.assertEqual(validate_task(task)["task_id"], task["task_id"])
            self.assertEqual(task["metadata"]["source_sdist_sha256"], ARCHIVE_SHA256)
            self.assertEqual(task["metadata"]["source_workspace_sha256"],
                             _workspace_digest(task_root / "seed"))
            self.assertTrue((task_root / "seed" / SOURCE_FILE).is_file())
            actions = [json.loads(line) for line in
                       (task_root / "actions.solution.jsonl").read_text(encoding="utf-8").splitlines()]
            fixed = Path(tmp) / "fixed"
            shutil.copytree(task_root / "seed", fixed)
            (fixed / actions[1]["path"]).write_text(actions[1]["content"], encoding="utf-8")
            cases = json.loads((task_root / "verifier/verify.json").read_text(encoding="utf-8"))["cases"]
            counts = []
            for root in (task_root / "seed", fixed):
                count = 0
                for case in cases:
                    process = subprocess.run(case["argv"], cwd=root,
                                             capture_output=True, text=True, timeout=5)
                    count += (process.returncode == case["expected_returncode"]
                              and process.stdout == case["expected_stdout"])
                counts.append(count)
            self.assertEqual(counts, [5, 14])
            (task_root / "seed" / SOURCE_FILE).write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "pinned Humanize archive"):
                smoke(task_dir=task_root, image="unused-image", output=Path(tmp) / "untouched-output")


if __name__ == "__main__":
    unittest.main()
