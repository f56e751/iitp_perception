import contextlib
import csv
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _setup  # noqa: F401  (configures sys.path for project imports)
from _setup import score_row, write_scores, write_track_csv

try:
    import apply_corrections as ac
    HAS_AC = True
except Exception as _e:  # pragma: no cover - depends on cv2 availability
    HAS_AC = False
    _AC_ERR = _e

A = (280, 180, 350, 240)
B = (560, 250, 640, 270)


def build_dir(d, corrections):
    """Two objects (box_id 0 -> track 0, box_id 1 -> track 1) over 3 frames."""
    objects = os.path.join(d, "objects")
    os.makedirs(objects)
    score_rows, t0, t1 = [], [], []
    for i in range(3):
        img = f"{i:06d}.jpg"
        r0 = score_row(img, 0, A, "metal")
        r1 = score_row(img, 1, B, "transparent")
        score_rows += [r0, r1]
        t0.append(r0)
        t1.append(r1)
    write_scores(os.path.join(d, "scores.csv"), score_rows)
    write_track_csv(os.path.join(objects, "track_0.csv"), 0, t0)
    write_track_csv(os.path.join(objects, "track_1.csv"), 1, t1)
    with open(os.path.join(objects, "corrections.json"), "w") as f:
        json.dump(corrections, f)
    return objects


def run_main(input_dir):
    argv = ["apply_corrections", "-i", input_dir, "--no-annotate"]
    with mock.patch.object(sys, "argv", argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = ac.main()
    return rc, buf.getvalue()


def read_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


@unittest.skipUnless(HAS_AC, "apply_corrections import failed (needs cv2)")
class TestApplyCorrections(unittest.TestCase):
    def test_pure_helpers(self):
        self.assertEqual(ac.centroid({"x1": "0", "y1": "0", "x2": "10", "y2": "20"}), (5.0, 10.0))
        self.assertEqual(ac.distance((0, 0), (3, 4)), 5.0)

    def test_load_original_assignments(self):
        with tempfile.TemporaryDirectory() as d:
            objects = build_dir(d, {})
            assigns = ac.load_original_assignments(__import__("pathlib").Path(objects))
            self.assertEqual(assigns[("000000.jpg", "0")], 0)
            self.assertEqual(assigns[("000000.jpg", "1")], 1)

    def test_delete_track(self):
        with tempfile.TemporaryDirectory() as d:
            objects = build_dir(d, {"delete_tracks": [1]})
            rc, _ = run_main(d)
            self.assertEqual(rc, 0)
            self.assertTrue(os.path.exists(os.path.join(objects, "track_0.csv")))
            self.assertFalse(os.path.exists(os.path.join(objects, "track_1.csv")))
            summary = read_rows(os.path.join(objects, "summary.csv"))
            self.assertEqual([r["track_id"] for r in summary], ["0"])

    def test_reassignment_moves_detection(self):
        with tempfile.TemporaryDirectory() as d:
            objects = build_dir(
                d,
                {"reassignments": [{"frame": "000000.jpg", "box_id": 0, "to_track": 1}]},
            )
            rc, _ = run_main(d)
            self.assertEqual(rc, 0)
            t0 = read_rows(os.path.join(objects, "track_0.csv"))
            t1 = read_rows(os.path.join(objects, "track_1.csv"))
            # box_id 0 from frame 0 moved to track 1: track 0 loses one row, track 1 gains one.
            self.assertEqual(len(t0), 2)
            self.assertEqual(len(t1), 4)


if __name__ == "__main__":
    unittest.main()
