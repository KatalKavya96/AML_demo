"""Masked forecast metrics: points whose true value is missing (0 or NaN) are never scored."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trafficml.data.masking import NULL_VALUE, valid_mask


def _masked(y_true: np.ndarray, y_pred: np.ndarray, null_value: float) -> tuple[np.ndarray, ...]:
    m = valid_mask(y_true, null_value)
    return y_true[m], y_pred[m]


def masked_mae(y_true, y_pred, null_value: float = NULL_VALUE) -> float:
    t, p = _masked(np.asarray(y_true, float), np.asarray(y_pred, float), null_value)
    return float(np.mean(np.abs(p - t))) if t.size else float("nan")


def masked_rmse(y_true, y_pred, null_value: float = NULL_VALUE) -> float:
    t, p = _masked(np.asarray(y_true, float), np.asarray(y_pred, float), null_value)
    return float(np.sqrt(np.mean(np.square(p - t)))) if t.size else float("nan")


def masked_mape(y_true, y_pred, null_value: float = NULL_VALUE) -> float:
    """Mean absolute percentage error, in percent."""
    t, p = _masked(np.asarray(y_true, float), np.asarray(y_pred, float), null_value)
    return float(np.mean(np.abs((p - t) / t)) * 100) if t.size else float("nan")


def horizon_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, null_value: float = NULL_VALUE
) -> pd.DataFrame:
    """MAE / RMSE / MAPE per horizon for arrays shaped (samples, horizon, sensors).

    Horizon h is 1-based (h=3 -> 15 min ahead).
    """
    rows = []
    for i in range(y_true.shape[1]):
        t, p = y_true[:, i], y_pred[:, i]
        rows.append(
            {
                "h": i + 1,
                "minutes": 5 * (i + 1),
                "MAE": masked_mae(t, p, null_value),
                "RMSE": masked_rmse(t, p, null_value),
                "MAPE": masked_mape(t, p, null_value),
            }
        )
    return pd.DataFrame(rows)
