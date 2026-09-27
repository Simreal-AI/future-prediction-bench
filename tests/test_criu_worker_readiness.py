"""Exercise the actual empty/partial identity-file startup race."""
import json
from pathlib import Path
import tempfile
import threading
import unittest

from examples.official_crab_criu.check_chain import wait_identity_file


class WorkerReadinessTest(unittest.TestCase):
    def test_partial_file_is_observed_before_writer_completes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "identity.json"
            path.write_bytes(b"{")
            partial_observed = threading.Event()
            failures = []
            expected = {"address": 4096, "bytes": 1048576, "page_size": 4096}

            class ObservedPath:
                def __getattr__(self, name):
                    return getattr(path, name)

                def read_bytes(self):
                    raw = path.read_bytes()
                    if raw == b"{":
                        partial_observed.set()
                    return raw

            def finish_write():
                if not partial_observed.wait(2):
                    failures.append("reader did not observe the partial file")
                    return
                path.write_text(json.dumps(expected))

            writer = threading.Thread(target=finish_write)
            writer.start()
            try:
                self.assertEqual(wait_identity_file(ObservedPath()), expected)
            finally:
                writer.join(3)
            self.assertFalse(writer.is_alive())
            self.assertEqual(failures, [])
            self.assertTrue(partial_observed.is_set())
