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

import group_by_object as gbo


def run_main(input_dir, *extra):
    argv = ["group_by_object", "-i", input_dir, *extra]
    with mock.patch.object(sys, "argv", argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = gbo.main()
    return rc, buf.getvalue()


class TestIoU(unittest.TestCase):
    def test_identical_boxes(self):
        self.assertAlmostEqual(gbo.iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)

    def test_disjoint_boxes(self):
        self.assertEqual(gbo.iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)

    def test_half_overlap(self):
        # inter = 5*10 = 50, union = 100 + 100 - 50 = 150
        self.assertAlmostEqual(gbo.iou((0, 0, 10, 10), (5, 0, 15, 10)), 50 / 150)


class TestClustering(unittest.TestCase):
    def test_two_static_objects_split_into_two(self):
        left = (280, 180, 350, 240)   # center x = 315
        right = (560, 250, 640, 270)  # center x = 600
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
            self.assertTrue(os.path.exists(os.path.join(objects, "object_0.csv")))
            self.assertTrue(os.path.exists(os.path.join(objects, "object_1.csv")))
            self.assertFalse(os.path.exists(os.path.join(objects, "object_2.csv")))

            # object_0 is the leftmost cluster (sorted by mean x-center).
            with open(os.path.join(objects, "object_0.csv")) as f:
                rows0 = list(csv.DictReader(f))
            with open(os.path.join(objects, "object_1.csv")) as f:
                rows1 = list(csv.DictReader(f))

            self.assertEqual(len(rows0), 3)
            self.assertEqual(len(rows1), 3)
            self.assertTrue(all(r["predicted_class"] == "metal" for r in rows0))
            self.assertTrue(all(r["predicted_class"] == "transparent" for r in rows1))

    def test_missing_scores_returns_error(self):
        with tempfile.TemporaryDirectory() as d:
            rc, _ = run_main(d)
            self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
