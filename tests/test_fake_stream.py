import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _setup  # noqa: F401  (configures sys.path for project imports)

import fake_stream


class TestRandomDetections(unittest.TestCase):
    def test_arrays_aligned_and_in_range(self):
        rng = random.Random(0)
        for _ in range(50):
            positions, class_names, confidences = fake_stream.random_detections(3, rng)
            self.assertEqual(len(positions), len(class_names))
            self.assertEqual(len(positions), len(confidences))
            self.assertLessEqual(len(positions), 3)
            for p in positions:
                self.assertEqual(len(p), 3)
            for c in class_names:
                self.assertIn(c, fake_stream.CLASSES)
            for conf in confidences:
                self.assertTrue(0.2 <= conf <= 0.95)

    def test_record_is_buildable(self):
        rng = random.Random(1)
        pos, cls, conf = fake_stream.random_detections(3, rng)
        import streaming
        rec = streaming.build_record(1.0, 0.05, pos, cls, conf)
        self.assertEqual(rec["schema_version"], streaming.SCHEMA_VERSION)
        self.assertEqual(len(rec["positions"]), len(rec["confidences"]))


if __name__ == "__main__":
    unittest.main()
