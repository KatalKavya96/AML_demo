"""Chronological sample windows and splits, following DCRNN's generate_training_data.

A sample is anchored at time index t: inputs are t-11..t and targets t+1..t+12.
Validation is halved in time: ``val_tune`` for tuning / early stopping, ``val_stack`` reserved
for training the Phase 2 hybrid stacker.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Splits:
    train: np.ndarray
    val_tune: np.ndarray
    val_stack: np.ndarray
    test: np.ndarray

    def items(self) -> list[tuple[str, np.ndarray]]:
        return [
            ("train", self.train),
            ("val_tune", self.val_tune),
            ("val_stack", self.val_stack),
            ("test", self.test),
        ]


def anchor_indices(num_steps: int, history: int = 12, horizon: int = 12) -> np.ndarray:
    """Anchors t with a full input window (t-history+1..t) and target window (t+1..t+horizon)."""
    return np.arange(history - 1, num_steps - horizon)


def chronological_split(
    num_steps: int,
    history: int = 12,
    horizon: int = 12,
    train_frac: float = 0.7,
    test_frac: float = 0.2,
) -> Splits:
    """DCRNN 70/10/20 split by sample, with validation halved in time."""
    anchors = anchor_indices(num_steps, history, horizon)
    n = len(anchors)
    # Python's round, as in DCRNN, so split sizes match the literature exactly.
    n_test = round(n * test_frac)
    n_train = round(n * train_frac)
    n_val = n - n_test - n_train
    val = anchors[n_train : n_train + n_val]
    half = n_val // 2
    return Splits(
        train=anchors[:n_train],
        val_tune=val[:half],
        val_stack=val[half:],
        test=anchors[-n_test:],
    )


def make_windows(
    data: np.ndarray, anchors: np.ndarray, history: int = 12, horizon: int = 12
) -> tuple[np.ndarray, np.ndarray]:
    """Slice (T, N) data into x of shape (S, history, N) and y of shape (S, horizon, N)."""
    x_offsets = np.arange(-(history - 1), 1)
    y_offsets = np.arange(1, horizon + 1)
    return data[anchors[:, None] + x_offsets], data[anchors[:, None] + y_offsets]


def split_time_ranges(index: pd.DatetimeIndex, splits: Splits) -> pd.DataFrame:
    """First/last anchor timestamp and sample count of each split, for the report."""
    rows = [
        {"split": name, "first_anchor": index[a[0]], "last_anchor": index[a[-1]], "samples": len(a)}
        for name, a in splits.items()
    ]
    return pd.DataFrame(rows)
