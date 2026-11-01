"""Missing-value handling: 0 mph in METR-LA means a missing reading or sensor outage."""

from __future__ import annotations

import numpy as np

NULL_VALUE = 0.0


def valid_mask(x: np.ndarray, null_value: float = NULL_VALUE) -> np.ndarray:
    """True where ``x`` holds a real reading (not NaN, not the null value)."""
    return ~np.isnan(x) & (x != null_value)


def to_nan(x: np.ndarray, null_value: float = NULL_VALUE) -> np.ndarray:
    """Float copy of ``x`` with missing readings as NaN, so they cannot pass as zero traffic."""
    out = np.asarray(x, dtype=np.float32).copy()
    out[~valid_mask(out, null_value)] = np.nan
    return out
