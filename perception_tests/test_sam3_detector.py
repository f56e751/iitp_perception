"""Unit tests for the SAM3 backend's adapter layer.

These cover the pure mapping between the vendored runtime's instance dicts and
this repo's detection contract, so they run without tensorrt/pycuda or the
TensorRT engines.  The engine itself is exercised end-to-end by running
main.py --backend sam3 against the camera.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _setup  # noqa: F401  (configures sys.path for project imports)

import numpy as np

import sam3_detector


def project(u, v, _depth):
    """Stand-in projection: 100 pixels = 1 m, no offset."""
    return (float(u) / 100.0, float(v) / 100.0, 0.0)


def instance(category, score, box):
    return {"category": category, "score": score, "bbox_xyxy": list(box)}


class TestNormalizeCategory(unittest.TestCase):
    def test_cardboard_is_lowercased(self):
        self.assertEqual(sam3_detector.normalize_category("Cardboard"), "cardboard")

    def test_other_categories_pass_through(self):
        self.assertEqual(sam3_detector.normalize_category("metal"), "metal")
        self.assertEqual(sam3_detector.normalize_category("transparent"), "transparent")


class TestInstancesToDetections(unittest.TestCase):
    def test_empty_input(self):
        boxes, names, conf, (xyxy, ids, labels) = sam3_detector.instances_to_detections(
            [], project, None
        )
        self.assertEqual(boxes, [])
        self.assertEqual(names, [])
        self.assertEqual(conf, [])
        self.assertEqual(xyxy.shape, (0, 4))
        self.assertEqual(len(ids), 0)
        self.assertEqual(labels, [])

    def test_corners_are_clockwise_from_top_left(self):
        boxes, _, _, _ = sam3_detector.instances_to_detections(
            [instance("metal", 0.9, (10, 20, 30, 40))], project, None
        )
        self.assertEqual(
            boxes[0],
            [(0.1, 0.2, 0.0), (0.3, 0.2, 0.0), (0.3, 0.4, 0.0), (0.1, 0.4, 0.0)],
        )

    def test_sorted_by_score_descending(self):
        instances = [
            instance("metal", 0.5, (0, 0, 10, 10)),
            instance("Cardboard", 0.9, (20, 20, 30, 30)),
            instance("transparent", 0.7, (40, 40, 50, 50)),
        ]
        _, names, conf, _ = sam3_detector.instances_to_detections(
            instances, project, None
        )
        self.assertEqual(names, ["cardboard", "transparent", "metal"])
        self.assertEqual(conf, [0.9, 0.7, 0.5])

    def test_label_text_matches_stream_format(self):
        _, _, _, (_, _, labels) = sam3_detector.instances_to_detections(
            [instance("metal", 0.9, (10, 20, 30, 40))], project, None
        )
        self.assertEqual(labels[0], "metal 0.90 (+0.20,+0.30)m")

    def test_class_id_is_stable_per_category(self):
        one = sam3_detector.instances_to_detections(
            [instance("metal", 0.9, (0, 0, 10, 10))], project, None
        )[3][1]
        two = sam3_detector.instances_to_detections(
            [
                instance("Cardboard", 0.95, (20, 20, 30, 30)),
                instance("metal", 0.9, (0, 0, 10, 10)),
            ],
            project,
            None,
        )[3][1]
        # metal keeps its id (and therefore its colour) as the frame changes.
        self.assertEqual(int(one[0]), int(two[1]))

    def test_overlay_boxes_stay_in_crop_pixels(self):
        _, _, _, (xyxy, _, _) = sam3_detector.instances_to_detections(
            [instance("metal", 0.9, (10, 20, 30, 40))], project, None
        )
        np.testing.assert_allclose(xyxy[0], [10, 20, 30, 40])


class TestDefaultArgs(unittest.TestCase):
    def test_engine_paths_live_under_weights(self):
        args = sam3_detector.default_args()
        for path in (args.vision_engine, args.text_engine, args.decoder_engine):
            self.assertIn(os.path.join("weights", "SAM3-trt"), path)
        self.assertTrue(args.tokenizer.endswith(os.path.join("usls", "tokenizer.json")))

    def test_overrides_win(self):
        args = sam3_detector.default_args(score_thr=0.75, temporal_vote=True)
        self.assertEqual(args.score_thr, 0.75)
        self.assertTrue(args.temporal_vote)

    def test_temporal_vote_off_by_default(self):
        self.assertFalse(sam3_detector.default_args().temporal_vote)


class TestRequireRuntimeFiles(unittest.TestCase):
    def test_missing_files_name_the_build_command(self):
        args = sam3_detector.default_args(
            vision_engine="/nope/vision.engine",
            text_engine="/nope/text.engine",
            decoder_engine="/nope/decoder.engine",
            tokenizer="/nope/tokenizer.json",
        )
        with self.assertRaises(FileNotFoundError) as ctx:
            sam3_detector.require_runtime_files(args)
        message = str(ctx.exception)
        self.assertIn("/nope/vision.engine", message)
        self.assertIn("wrap_sam3_trt.sh", message)

    def test_existing_files_pass(self):
        args = sam3_detector.default_args(
            vision_engine=__file__,
            text_engine=__file__,
            decoder_engine=__file__,
            tokenizer=__file__,
        )
        sam3_detector.require_runtime_files(args)  # must not raise


class FakeSegmenter:
    """Stands in for the vendored RuntimeSegmenter."""

    def __init__(self, instances):
        self.instances = instances
        self.seen_image = None
        self.seen_frame_delta = None

    def infer(self, image, frame_delta=1.0, **_kwargs):
        self.seen_image = image
        self.seen_frame_delta = frame_delta
        return {
            "instances": self.instances,
            "timing_ms": {"total": 12.5},
        }


try:
    import cv2  # noqa: F401
    import PIL.Image  # noqa: F401
    HAS_IMAGING = True
except ImportError:  # pragma: no cover
    HAS_IMAGING = False


@unittest.skipUnless(HAS_IMAGING, "needs cv2 + pillow")
class TestSam3DetectorDetect(unittest.TestCase):
    def setUp(self):
        self.fake = FakeSegmenter([instance("metal", 0.9, (10, 20, 30, 40))])
        self.detector = sam3_detector.Sam3Detector(segmenter=self.fake)

    def test_returns_the_engine_contract_triple(self):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        boxes, names, conf = self.detector.detect(frame, None, project)
        self.assertEqual(names, ["metal"])
        self.assertEqual(conf, [0.9])
        self.assertEqual(len(boxes[0]), 4)

    def test_frame_is_converted_bgr_to_rgb(self):
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        frame[:, :, 0] = 255  # pure blue in BGR
        self.detector.detect(frame, None, project)
        self.assertEqual(self.fake.seen_image.getpixel((0, 0)), (0, 0, 255))

    def test_overlay_and_timing_recorded(self):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        self.detector.detect(frame, None, project)
        xyxy, ids, labels = self.detector.last_overlay
        np.testing.assert_allclose(xyxy[0], [10, 20, 30, 40])
        self.assertEqual(labels, ["metal 0.90 (+0.20,+0.30)m"])
        self.assertEqual(self.detector.last_timing_ms["total"], 12.5)

    def test_empty_frame_clears_the_overlay(self):
        self.fake.instances = []
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        boxes, names, conf = self.detector.detect(frame, None, project)
        self.assertEqual((boxes, names, conf), ([], [], []))
        self.assertEqual(len(self.detector.last_overlay[0]), 0)

    def test_frame_delta_is_forwarded_to_the_temporal_voter(self):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        self.detector.detect(frame, None, project, frame_delta=2.5)
        self.assertEqual(self.fake.seen_frame_delta, 2.5)


if __name__ == "__main__":
    unittest.main()
