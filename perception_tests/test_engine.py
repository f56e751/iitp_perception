"""Unit tests for the detector engine's pure post-processing helpers.

Importing iitp_object_detector pulls in torch / supervision / grounding_dino,
so these only run inside the grounded_sam image; on a bare host they skip.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _setup  # noqa: F401  (configures sys.path for project imports)

try:
    import numpy as np
    import iitp_object_detector as eng
    HAS_ENG = True
except Exception as _e:  # pragma: no cover
    HAS_ENG = False
    _ENG_ERR = _e


@unittest.skipUnless(HAS_ENG, "iitp_object_detector import failed (needs torch/grounding_dino)")
class TestIoUxyxy(unittest.TestCase):
    def test_identical(self):
        box = np.array([0, 0, 10, 10], dtype=float)
        boxes = np.array([[0, 0, 10, 10]], dtype=float)
        self.assertAlmostEqual(float(eng._iou_xyxy(box, boxes)[0]), 1.0, places=5)

    def test_disjoint(self):
        box = np.array([0, 0, 10, 10], dtype=float)
        boxes = np.array([[20, 20, 30, 30]], dtype=float)
        self.assertAlmostEqual(float(eng._iou_xyxy(box, boxes)[0]), 0.0, places=5)

    def test_half_overlap(self):
        box = np.array([0, 0, 10, 10], dtype=float)
        boxes = np.array([[5, 0, 15, 10]], dtype=float)
        self.assertAlmostEqual(float(eng._iou_xyxy(box, boxes)[0]), 50 / 150, places=4)


@unittest.skipUnless(HAS_ENG, "iitp_object_detector import failed (needs torch/grounding_dino)")
class TestNmsByLabel(unittest.TestCase):
    def test_empty_input(self):
        boxes = np.zeros((0, 4), dtype=float)
        conf = np.zeros((0,), dtype=float)
        kb, ks, kl, ki = eng.nms_by_label(conf, boxes, [], iou_thresh=0.5, method="hard")
        self.assertEqual(len(kb), 0)
        self.assertEqual(kl, [])

    def test_overlapping_same_label_suppressed(self):
        boxes = np.array([[0, 0, 10, 10], [1, 0, 11, 10]], dtype=float)  # IoU ~0.82
        conf = np.array([0.9, 0.5], dtype=float)
        kb, ks, kl, ki = eng.nms_by_label(conf, boxes, ["a", "a"], iou_thresh=0.5, method="hard")
        self.assertEqual(len(kb), 1)
        self.assertAlmostEqual(float(ks[0]), 0.9, places=5)

    def test_different_labels_both_kept(self):
        boxes = np.array([[0, 0, 10, 10], [1, 0, 11, 10]], dtype=float)
        conf = np.array([0.9, 0.5], dtype=float)
        kb, ks, kl, ki = eng.nms_by_label(conf, boxes, ["a", "b"], iou_thresh=0.5, method="hard")
        self.assertEqual(len(kb), 2)
        self.assertEqual(sorted(kl), ["a", "b"])

    def test_results_sorted_by_score_desc(self):
        boxes = np.array([[0, 0, 10, 10], [100, 100, 110, 110]], dtype=float)
        conf = np.array([0.3, 0.9], dtype=float)
        kb, ks, kl, ki = eng.nms_by_label(conf, boxes, ["a", "a"], iou_thresh=0.5, method="hard")
        self.assertEqual(len(kb), 2)
        self.assertGreaterEqual(float(ks[0]), float(ks[1]))


@unittest.skipUnless(HAS_ENG, "iitp_object_detector import failed (needs torch/grounding_dino)")
class TestGeometryAndMask(unittest.TestCase):
    def test_offset_to_bbox_clamps_to_image(self):
        self.assertEqual(eng.offset_to_bbox([5, 5, 15, 15], 10, 20, 20), [0, 0, 20, 20])

    def test_merge_bounding_boxes_envelope(self):
        merged = eng.merge_bounding_boxes([[0, 0, 10, 10], [5, 5, 20, 20]])
        self.assertEqual(list(merged), [0, 0, 20, 20])

    def test_bbox_from_mask(self):
        mask = np.zeros((5, 5), dtype=bool)
        mask[1:3, 1:3] = True
        self.assertEqual(eng._bbox_from_mask(mask), [1, 1, 2, 2])

    def test_bbox_from_empty_mask_is_none(self):
        self.assertIsNone(eng._bbox_from_mask(np.zeros((5, 5), dtype=bool)))


if __name__ == "__main__":
    unittest.main()
