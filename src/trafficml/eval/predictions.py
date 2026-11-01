"""Standard on-disk format for forecasts, shared by every model (Phase 1 and Phase 2).

Per split, targets are stored once:   results/preds/targets_{split}.npz
    anchors (S,), sensor_ids (N,), y_true (S, H, N) float32 with NaN = missing (not scored)
Per model and split:                 results/preds/{model}_{split}.npz
    anchors (S,), y_pred (S, H, N) float32

Arrays instead of a long table: a test split is 6,850 x 12 x 207 = 17M points.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

PRED_DIR = Path(__file__).resolve().parents[3] / "results" / "preds"


def save_targets(split: str, anchors, sensor_ids, y_true, pred_dir: Path = PRED_DIR) -> Path:
    pred_dir.mkdir(parents=True, exist_ok=True)
    path = pred_dir / f"targets_{split}.npz"
    np.savez_compressed(
        path,
        anchors=np.asarray(anchors),
        sensor_ids=np.asarray(sensor_ids),
        y_true=np.asarray(y_true, dtype=np.float32),
    )
    return path


def save_predictions(model: str, split: str, anchors, y_pred, pred_dir: Path = PRED_DIR) -> Path:
    y_pred = np.asarray(y_pred, dtype=np.float32)
    targets = load_targets(split, pred_dir)
    if not np.array_equal(targets["anchors"], anchors) or y_pred.shape != targets["y_true"].shape:
        raise ValueError(f"{model}/{split}: anchors or shape differ from the saved targets")
    scored = ~np.isnan(targets["y_true"])
    if np.isnan(y_pred[scored]).any():
        raise ValueError(f"{model}/{split}: NaN predictions on scored points")
    path = pred_dir / f"{model}_{split}.npz"
    np.savez_compressed(path, anchors=np.asarray(anchors), y_pred=y_pred)
    return path


def load_targets(split: str, pred_dir: Path = PRED_DIR) -> dict[str, np.ndarray]:
    with np.load(pred_dir / f"targets_{split}.npz") as f:
        return {k: f[k] for k in f.files}


def load_predictions(model: str, split: str, pred_dir: Path = PRED_DIR) -> np.ndarray:
    with np.load(pred_dir / f"{model}_{split}.npz") as f:
        return f["y_pred"]


def available_models(split: str, pred_dir: Path = PRED_DIR) -> list[str]:
    suffix = f"_{split}.npz"
    return sorted(
        p.name[: -len(suffix)]
        for p in pred_dir.glob(f"*{suffix}")
        if not p.name.startswith("targets_")
    )
