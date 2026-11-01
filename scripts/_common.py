"""Shared config and data loading for scripts."""

from __future__ import annotations

from pathlib import Path

import yaml

from trafficml.data.load import load_sensor_ids, load_speed
from trafficml.data.masking import to_nan
from trafficml.data.splits import chronological_split

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = ROOT / "configs" / "data.yaml") -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    cfg["paths"] = {k: ROOT / v for k, v in cfg["paths"].items()}
    return cfg


def load_series(cfg: dict):
    """(speed DataFrame, float32 values with NaN for missing, sensor_ids, Splits)."""
    ids = load_sensor_ids(cfg["paths"]["sensor_ids"])
    speed = load_speed(cfg["paths"]["speed_h5"], ids)
    w, s = cfg["windows"], cfg["split"]
    splits = chronological_split(
        len(speed), w["history"], w["horizon"], s["train_frac"], s["test_frac"]
    )
    return speed, to_nan(speed.to_numpy(), cfg["series"]["null_value"]), ids, splits
