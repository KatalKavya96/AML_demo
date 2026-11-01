import numpy as np

from trafficml.data.masking import to_nan, valid_mask
from trafficml.eval.metrics import horizon_metrics, masked_mae, masked_mape, masked_rmse


def test_missing_targets_are_not_scored():
    y_true = np.array([60.0, 0.0, 40.0, np.nan])
    y_pred = np.array([50.0, 99.0, 50.0, 99.0])
    assert masked_mae(y_true, y_pred) == 10.0
    assert masked_rmse(y_true, y_pred) == 10.0
    assert np.isclose(masked_mape(y_true, y_pred), (10 / 60 + 10 / 40) / 2 * 100)


def test_all_missing_gives_nan():
    assert np.isnan(masked_mae(np.zeros(3), np.ones(3)))


def test_horizon_metrics_shape_and_values():
    y_true = np.full((4, 12, 3), 50.0)
    y_pred = y_true + np.arange(1, 13)[None, :, None]
    y_true[0, 0, 0] = 0.0
    out = horizon_metrics(y_true, y_pred)
    assert list(out["h"]) == list(range(1, 13))
    np.testing.assert_allclose(out["MAE"], np.arange(1, 13))
    assert out.loc[out.h == 12, "minutes"].item() == 60


def test_masking_helpers():
    x = np.array([0.0, 55.0, np.nan])
    np.testing.assert_array_equal(valid_mask(x), [False, True, False])
    assert np.isnan(to_nan(x)[[0, 2]]).all()
