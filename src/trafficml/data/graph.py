"""Adjacency construction from road-network distances (DCRNN Gaussian-kernel method)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def distance_matrix(distances: pd.DataFrame, sensor_ids: list[str]) -> np.ndarray:
    """N x N distance matrix over ``sensor_ids``; inf where no road distance is given.

    Only pairs with both endpoints in ``sensor_ids`` are used; the CSV covers more sensors.
    """
    idx = {s: i for i, s in enumerate(sensor_ids)}
    n = len(sensor_ids)
    dist = np.full((n, n), np.inf, dtype=np.float32)
    inside = distances["from"].isin(idx) & distances["to"].isin(idx)
    sub = distances.loc[inside]
    dist[sub["from"].map(idx).to_numpy(), sub["to"].map(idx).to_numpy()] = sub["cost"].to_numpy()
    return dist


def gaussian_kernel_adjacency(dist: np.ndarray, normalized_k: float = 0.1) -> np.ndarray:
    """W_ij = exp(-(d_ij / sigma)^2), zeroed below ``normalized_k``.

    sigma is the std of the finite entries of ``dist`` (not of the raw CSV column).
    The result is directed (d_ij != d_ji) and has 1 on the diagonal (d_ii = 0).
    """
    sigma = dist[~np.isinf(dist)].std()
    adj = np.exp(-np.square(dist / sigma))
    adj[adj < normalized_k] = 0
    return adj


def build_adjacency(
    distances: pd.DataFrame, sensor_ids: list[str], normalized_k: float = 0.1
) -> np.ndarray:
    return gaussian_kernel_adjacency(distance_matrix(distances, sensor_ids), normalized_k)


def without_self_loops(adj: np.ndarray) -> np.ndarray:
    """Copy of ``adj`` with a zero diagonal, for neighbour features that must exclude the node."""
    out = adj.copy()
    np.fill_diagonal(out, 0)
    return out
