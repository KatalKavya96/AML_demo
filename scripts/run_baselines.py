"""Fit M0 (persistence) and M1 (historical average); save targets and predictions per split.

Usage: python scripts/run_baselines.py
"""

from __future__ import annotations

from _common import load_config, load_series

from trafficml.data.splits import make_windows
from trafficml.eval.predictions import save_predictions, save_targets
from trafficml.models.baselines import HistoricalAverage, persistence

EVAL_SPLITS = ("val_tune", "val_stack", "test")


def main() -> None:
    cfg = load_config()
    speed, values, ids, splits = load_series(cfg)
    history, horizon = cfg["windows"]["history"], cfg["windows"]["horizon"]

    # Training timestamps = every step that is an input or target of a training sample.
    train_end = splits.train[-1] + horizon + 1
    ha = HistoricalAverage().fit(values[:train_end], speed.index[:train_end])

    for split, anchors in splits.items():
        if split not in EVAL_SPLITS:
            continue
        x, y = make_windows(values, anchors, history, horizon)
        save_targets(split, anchors, ids, y)
        ha_pred = ha.predict(anchors, speed.index, horizon)
        save_predictions("M1_ha", split, anchors, ha_pred)
        save_predictions("M0_persistence", split, anchors, persistence(x, horizon, ha_pred))
        print(f"{split}: {len(anchors)} samples saved")


if __name__ == "__main__":
    main()
