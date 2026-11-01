"""Week 1 data check: adjacency vs DCRNN, sensor order, split dates, missing rate.

Usage: python scripts/check_data.py [--config configs/data.yaml]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from trafficml.data.graph import build_adjacency
from trafficml.data.load import load_dcrnn_adj, load_distances, load_sensor_ids, load_speed
from trafficml.data.masking import valid_mask
from trafficml.data.splits import chronological_split, split_time_ranges

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=ROOT / "configs" / "data.yaml")
    cfg = yaml.safe_load(Path(parser.parse_args().config).read_text())
    paths = {k: ROOT / v for k, v in cfg["paths"].items()}

    ids = load_sensor_ids(paths["sensor_ids"])
    adj = build_adjacency(load_distances(paths["distances"]), ids, cfg["graph"]["normalized_k"])
    _, _, dcrnn_adj = load_dcrnn_adj(paths["dcrnn_adj"])
    off_diag = adj.copy()
    np.fill_diagonal(off_diag, 0)
    print(f"sensors: {len(ids)}")
    print(f"adjacency matches DCRNN: {np.allclose(adj, dcrnn_adj, rtol=1e-5, atol=1e-6)}")
    print(f"directed edges (excl. self-loops): {int((off_diag > 0).sum())}")
    print(f"symmetric: {np.allclose(adj, adj.T)}")

    if not paths["speed_h5"].exists():
        print(f"\n{paths['speed_h5'].relative_to(ROOT)} missing; skipping series checks.")
        return

    speed = load_speed(paths["speed_h5"], ids)
    w = cfg["windows"]
    s = cfg["split"]
    splits = chronological_split(
        len(speed), w["history"], w["horizon"], s["train_frac"], s["test_frac"]
    )
    values = speed.to_numpy()
    print(f"\nseries: {speed.index[0]} -> {speed.index[-1]}, shape {values.shape}")
    print(f"missing rate (0 mph or NaN): {1 - valid_mask(values).mean():.2%}")
    print(split_time_ranges(speed.index, splits).to_string(index=False))


if __name__ == "__main__":
    main()
