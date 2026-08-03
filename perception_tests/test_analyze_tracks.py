import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _setup  # noqa: F401  (configures sys.path for project imports)
from _setup import score_row, write_track_csv

import analyze_tracks as at


def run_main(input_dir):
    argv = ["analyze_tracks", "-i", input_dir]
    with mock.patch.object(sys, "argv", argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = at.main()
    return rc, buf.getvalue()


class TestAnalyze(unittest.TestCase):
    def _setup_dir(self, d, t0_cls, t1_cls):
        objects = os.path.join(d, "objects")
        os.makedirs(objects)
        rows0 = [score_row(f"{i:06d}.jpg", 0, (0, 0, 10, 10), t0_cls) for i in range(3)]
        rows1 = [score_row(f"{i:06d}.jpg", 1, (50, 50, 60, 60), t1_cls) for i in range(3)]
        write_track_csv(os.path.join(objects, "track_0.csv"), 0, rows0)
        write_track_csv(os.path.join(objects, "track_1.csv"), 1, rows1)
        labels = {"0": "metal", "1": "transparent"}
        with open(os.path.join(objects, "true_labels.json"), "w") as f:
            json.dump(labels, f)

    def test_all_correct_is_full_accuracy(self):
        with tempfile.TemporaryDirectory() as d:
            self._setup_dir(d, "metal", "transparent")  # predictions match truth
            rc, out = run_main(d)
            self.assertEqual(rc, 0)
            self.assertIn("Tracks labeled: 2", out)
            self.assertIn("Per-frame accuracy:  6/6 = 100.0%", out)

    def test_wrong_predictions_lower_accuracy(self):
        with tempfile.TemporaryDirectory() as d:
            # track_1 predicts metal but its true label is transparent.
            self._setup_dir(d, "metal", "metal")
            rc, out = run_main(d)
            self.assertEqual(rc, 0)
            self.assertIn("Per-frame accuracy:  3/6 = 50.0%", out)

    def test_missing_labels_returns_error(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "objects"))
            rc, _ = run_main(d)
            self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
