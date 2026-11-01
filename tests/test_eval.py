import numpy as np
import pandas as pd
import pytest

from trafficml.eval.predictions import load_predictions, save_predictions, save_targets
from trafficml.eval.significance import diebold_mariano, holm, loss_differential
from trafficml.eval.slices import time_slices


def test_prediction_roundtrip_and_validation(tmp_path):
    anchors = np.arange(11, 15)
    y_true = np.full((4, 12, 3), 50.0, dtype=np.float32)
    y_true[0, 0, 0] = np.nan
    save_targets("test", anchors, ["a", "b", "c"], y_true, tmp_path)

    pred = y_true + 1
    pred[0, 0, 0] = np.nan  # NaN allowed on an unscored point
    save_predictions("m", "test", anchors, pred, tmp_path)
    np.testing.assert_array_equal(load_predictions("m", "test", tmp_path), pred)

    bad = pred.copy()
    bad[1, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        save_predictions("m", "test", anchors, bad, tmp_path)
    with pytest.raises(ValueError, match="anchors"):
        save_predictions("m", "test", anchors + 1, pred, tmp_path)


def test_peak_slice_is_weekday_rush_hour_of_target_time():
    index = pd.date_range("2012-06-04 06:00", periods=48, freq="5min")  # Monday
    s = time_slices(index, anchors=np.array([11]), horizon=12)  # targets 07:00..07:55
    assert s["peak"].all() and not s["off_peak"].any()
    weekend = pd.date_range("2012-06-09 06:00", periods=48, freq="5min")  # Saturday
    assert not time_slices(weekend, np.array([11]), 12)["peak"].any()


def test_loss_differential_ignores_missing_targets():
    y = np.array([[10.0, np.nan], [np.nan, np.nan]])
    d = loss_differential(y, np.array([[12.0, 0.0], [0.0, 0.0]]), np.array([[11.0, 0.0], [0, 0]]))
    np.testing.assert_array_equal(d, [1.0])


def test_diebold_mariano_detects_shift_only():
    rng = np.random.default_rng(0)
    assert diebold_mariano(rng.normal(0, 1, 5000), h=3)["p_value"] > 0.01
    assert diebold_mariano(rng.normal(0.2, 1, 5000), h=3)["p_value"] < 1e-6


def test_holm_adjustment():
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
