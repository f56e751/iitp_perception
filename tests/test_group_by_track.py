import contextlib
import csv
import io
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _setup  # noqa: F401  (configures sys.path for project imports)
from _setup import score_row, write_scores

try:
    import group_by_track as gbt
    HAS_GBT = True
except Exception as _e:  # pragma: no cover - depends on cv2 availability
    HAS_GBT = False
    _GBT_ERR = _e


@unittest.skipUnless(HAS_GBT, "group_by_track import failed (needs cv2)")
class TestPureHelpers(unittest.TestCase):
    def test_centroid(self):
        row = {"x1": "0", "y1": "0", "x2": "10", "y2": "20"}
        self.assertEqual(gbt.centroid(row), (5.0, 10.0))

    def test_distance(self):
        self.assertEqual(gbt.distance((0, 0), (3, 4)), 5.0)

    def test_is_ahead_any_always_true(self):
        self.assertTrue(gbt.is_ahead((0, 0), (0, 0), "any", 30))

    def test_is_ahead_up_requires_min_progress(self):
        self.assertTrue(gbt.is_ahead((0, 100), (0, 60), "up", 30))   # moved 40 up
        self.assertFalse(gbt.is_ahead((0, 100), (0, 80), "up", 30))  # moved 20 up


def run_main(input_dir, *extra):
    argv = ["group_by_track", "-i", input_dir, "--no-annotate", *extra]
    with mock.patch.object(sys, "argv", argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = gbt.main()
    return rc, buf.getvalue()


@unittest.skipUnless(HAS_GBT, "group_by_track import failed (needs cv2)")
class TestTracking(unittest.TestCase):
    def test_two_static_tracks(self):
        left = (280, 180, 350, 240)
        right = (560, 250, 640, 270)
        rows = []
        for i in range(3):
            img = f"{i:06d}.jpg"
            rows.append(score_row(img, 0, left, "metal"))
            rows.append(score_row(img, 1, right, "transparent"))

        with tempfile.TemporaryDirectory() as d:
            write_scores(os.path.join(d, "scores.csv"), rows)
            rc, _ = run_main(d)
            self.assertEqual(rc, 0)

            objects = os.path.join(d, "objects")
            self.assertTrue(os.path.exists(os.path.join(objects, "track_0.csv")))
            self.assertTrue(os.path.exists(os.path.join(objects, "track_1.csv")))

            with open(os.path.join(objects, "summary.csv")) as f:
                summary = list(csv.DictReader(f))
            self.assertEqual(len(summary), 2)
            # No motion across frames -> both flagged static.
            self.assertTrue(all(r["kind"] == "static" for r in summary))


if __name__ == "__main__":
    unittest.main()
