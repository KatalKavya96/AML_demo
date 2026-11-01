import numpy as np

from trafficml.data.splits import anchor_indices, chronological_split, make_windows

METR_LA_STEPS = 34272


def test_dcrnn_split_sizes():
    s = chronological_split(METR_LA_STEPS)
    assert len(anchor_indices(METR_LA_STEPS)) == 34249
    assert len(s.train) == 23974
    assert len(s.val_tune) + len(s.val_stack) == 3425
    assert len(s.test) == 6850


def test_splits_are_ordered_disjoint_and_complete():
    s = chronological_split(METR_LA_STEPS)
    parts = [a for _, a in s.items()]
    for earlier, later in zip(parts, parts[1:], strict=False):
        assert earlier.max() < later.min()
    np.testing.assert_array_equal(np.concatenate(parts), anchor_indices(METR_LA_STEPS))


def test_targets_stay_inside_series_and_windows_are_causal():
    s = chronological_split(METR_LA_STEPS)
    assert s.train.min() - 11 == 0
    assert s.test.max() + 12 == METR_LA_STEPS - 1

    data = np.arange(100, dtype=float)[:, None]  # value == time index
    x, y = make_windows(data, np.array([11, 50]))
    assert x.shape == (2, 12, 1) and y.shape == (2, 12, 1)
    np.testing.assert_array_equal(x[1, :, 0], np.arange(39, 51))
    np.testing.assert_array_equal(y[1, :, 0], np.arange(51, 63))
    assert (x[:, -1, 0] < y[:, 0, 0]).all()
