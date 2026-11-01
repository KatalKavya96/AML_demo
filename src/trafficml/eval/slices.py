"""Boolean masks over target times for sliced evaluation (peak vs off-peak, weekday vs weekend).

Slices are defined on the *target* time (anchor + h): the time being forecast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Weekday peak windows (local hours, [start, end)). Confirm against EDA daily profiles.
PEAK_HOURS = ((7, 10), (16, 19))


def target_times(index: pd.DatetimeIndex, anchors: np.ndarray, horizon: int) -> pd.DatetimeIndex:
    steps = anchors[:, None] + np.arange(1, horizon + 1)
    return index[steps.ravel()]


def time_slices(
    index: pd.DatetimeIndex, anchors: np.ndarray, horizon: int
) -> dict[str, np.ndarray]:
    """Masks of shape (S, horizon, 1), broadcastable over sensors."""
    times = target_times(index, anchors, horizon)
    weekday = np.asarray(times.day_of_week < 5)
    hour = np.asarray(times.hour)
    in_peak_hours = np.zeros(len(times), dtype=bool)
    for start, end in PEAK_HOURS:
        in_peak_hours |= (hour >= start) & (hour < end)
    peak = weekday & in_peak_hours
    shape = (len(anchors), horizon, 1)
    return {
        "all": np.ones(shape, dtype=bool),
        "peak": peak.reshape(shape),
        "off_peak": (~peak).reshape(shape),
        "weekday": weekday.reshape(shape),
        "weekend": (~weekday).reshape(shape),
    }
