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
            boxes, class_names, confidences = fake_stream.random_detections(3, rng)
            self.assertEqual(len(boxes), len(class_names))
            self.assertEqual(len(boxes), len(confidences))
            self.assertLessEqual(len(boxes), 3)
            for box in boxes:
                self.assertEqual(len(box), 4)
                self.assertTrue(all(len(point) == 3 for point in box))
            for c in class_names:
                self.assertIn(c, fake_stream.CLASSES)
            for conf in confidences:
                self.assertTrue(0.2 <= conf <= 0.95)

    def test_record_is_buildable(self):
        rng = random.Random(1)
        boxes, cls, conf = fake_stream.random_detections(3, rng)
        import streaming
        rec = streaming.build_record(1.0, 0.05, boxes, cls, conf)
        self.assertEqual(rec["schema_version"], streaming.SCHEMA_VERSION)
        self.assertEqual(len(rec["bounding_boxes"]), len(rec["confidences"]))


if __name__ == "__main__":
    unittest.main()
