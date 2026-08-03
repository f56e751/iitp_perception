import unittest

from bbox_projection import project_bounding_boxes


class TestBoundingBoxProjection(unittest.TestCase):
    def test_preserves_all_four_corners_clockwise(self):
        def project(u, v, _depth):
            return (u / 10.0, v / 10.0, 0.0)

        self.assertEqual(
            project_bounding_boxes([[10, 20, 30, 40]], project, None),
            [[(1.0, 2.0, 0.0), (3.0, 2.0, 0.0),
              (3.0, 4.0, 0.0), (1.0, 4.0, 0.0)]],
        )


if __name__ == "__main__":
    unittest.main()
