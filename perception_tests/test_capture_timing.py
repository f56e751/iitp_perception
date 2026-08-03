import unittest

from capture_timing import capture_age_seconds


class TestCaptureTiming(unittest.TestCase):
    def test_global_timestamp_returns_frame_age(self):
        self.assertAlmostEqual(
            capture_age_seconds(100.125, 100000.0, is_global_time=True),
            0.125,
        )

    def test_hardware_clock_is_not_compared_to_epoch(self):
        self.assertIsNone(
            capture_age_seconds(100.125, 2500.0, is_global_time=False)
        )

    def test_rejects_negative_or_stale_age(self):
        self.assertIsNone(
            capture_age_seconds(100.0, 100100.0, is_global_time=True)
        )
        self.assertIsNone(
            capture_age_seconds(100.0, 97000.0, is_global_time=True)
        )

    def test_rejects_non_finite_age(self):
        for timestamp in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(timestamp=timestamp):
                self.assertIsNone(
                    capture_age_seconds(100.0, timestamp, is_global_time=True)
                )


if __name__ == "__main__":
    unittest.main()
