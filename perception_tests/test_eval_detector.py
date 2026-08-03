"""Unit tests for eval_detector's per-class scoring helpers.

eval_detector imports torch / grounding_dino / iitp_object_detector, so these
run only inside the grounded_sam image; on a bare host they skip.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _setup  # noqa: F401  (configures sys.path for project imports)

try:
    import torch
    import eval_detector as ed
    HAS_ED = True
except Exception as _e:  # pragma: no cover
    HAS_ED = False
    _ED_ERR = _e


@unittest.skipUnless(HAS_ED, "eval_detector import failed (needs torch/grounding_dino)")
class TestClassSpans(unittest.TestCase):
    def test_three_class_spans(self):
        # Separators at [CLS]/"."/[SEP] token positions for 3 classes.
        sep_idx = [0, 4, 8, 12]
        self.assertEqual(ed.class_spans(sep_idx, 3), [(1, 4), (5, 8), (9, 12)])


@unittest.skipUnless(HAS_ED, "eval_detector import failed (needs torch/grounding_dino)")
class TestPerClassScores(unittest.TestCase):
    def test_max_pool_over_spans(self):
        row = torch.arange(13, dtype=torch.float)  # [0,1,2,...,12]
        spans = [(1, 4), (5, 8), (9, 12)]
        self.assertEqual(ed.per_class_scores(row, spans), [3.0, 7.0, 11.0])

    def test_zero_width_span_scores_zero(self):
        row = torch.arange(13, dtype=torch.float)
        self.assertEqual(ed.per_class_scores(row, [(5, 5)]), [0.0])


if __name__ == "__main__":
    unittest.main()
