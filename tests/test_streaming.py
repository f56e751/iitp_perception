import json
import os
import sys
import time
import unittest
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _setup  # noqa: F401  (configures sys.path for project imports)

import streaming


class TestBuildRecord(unittest.TestCase):
    def test_schema_and_types(self):
        rec = streaming.build_record(
            timestamp=1.5,
            elapsed_s=0.08,
            positions=[(0.1, 0.2, 0.3), (1, 2, 3)],
            class_names=["metal", "transparent"],
            confidences=[0.9, 0.5],
        )
        self.assertEqual(
            set(rec),
            {"timestamp", "elapsed_s", "positions", "class_names", "confidences"},
        )
        self.assertEqual(rec["positions"][0], [0.1, 0.2, 0.3])
        self.assertEqual(rec["class_names"], ["metal", "transparent"])
        self.assertEqual(rec["confidences"], [0.9, 0.5])
        # JSON-serializable end to end.
        self.assertEqual(json.loads(json.dumps(rec)), rec)

    def test_parallel_arrays_aligned(self):
        rec = streaming.build_record(0, 0, [(1, 1, 1)], ["metal"], [0.7])
        self.assertEqual(len(rec["positions"]), len(rec["class_names"]))
        self.assertEqual(len(rec["positions"]), len(rec["confidences"]))


class TestServerEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = streaming.start_server(0, fps=30)  # ephemeral port
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def _url(self, path):
        return f"http://127.0.0.1:{self.port}{path}"

    def test_latest_returns_published_record(self):
        rec = streaming.build_record(2.0, 0.05, [(1, 2, 3)], ["metal"], [0.8])
        streaming.publish_detections(rec)
        with urllib.request.urlopen(self._url("/detections"), timeout=2) as r:
            got = json.loads(r.read().decode())
        self.assertEqual(got, rec)

    def test_stream_pushes_new_record(self):
        # Unique record so we can distinguish it from any stale frame the
        # stream may emit first (handler pushes latest-on-seq-change).
        rec = streaming.build_record(3.0, 0.04, [(4, 5, 6)], ["transparent"], [0.6])
        with urllib.request.urlopen(self._url("/detections/stream"), timeout=3) as r:
            streaming.publish_detections(rec)
            seen = []
            for _ in range(5):
                line = r.readline().strip()
                if line:
                    seen.append(json.loads(line.decode()))
                    if seen[-1] == rec:
                        break
            self.assertIn(rec, seen)

    def test_unknown_path_404(self):
        try:
            urllib.request.urlopen(self._url("/nope"), timeout=2)
            self.fail("expected HTTP 404")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 404)


if __name__ == "__main__":
    unittest.main()
