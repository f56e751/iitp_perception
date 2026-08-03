"""Pure helpers for validating RealSense capture timestamps."""

import math

MAX_CAPTURE_AGE_S = 2.0


def capture_age_seconds(
    inference_start_s,
    frame_timestamp_ms,
    *,
    is_global_time,
    max_age_s=MAX_CAPTURE_AGE_S,
):
    """Return frame-to-inference age, or ``None`` for an unusable timestamp."""
    if not is_global_time:
        return None
    age = float(inference_start_s) - float(frame_timestamp_ms) / 1000.0
    if not math.isfinite(age) or age < 0.0 or age > float(max_age_s):
        return None
    return age
