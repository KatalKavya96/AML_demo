import numpy as np
import pandas as pd
from conftest import GRAPH_DIR, require

from trafficml.data.graph import (
    build_adjacency,
    distance_matrix,
    gaussian_kernel_adjacency,
    without_self_loops,
)
from trafficml.data.load import load_dcrnn_adj, load_distances, load_sensor_ids


def _toy_distances() -> pd.DataFrame:
    # "x" is not one of our sensors and must be ignored, including for sigma.
    rows = [
        ("a", "a", 0.0), ("b", "b", 0.0), ("c", "c", 0.0),
        ("a", "b", 1000.0), ("b", "c", 1500.0), ("c", "a", 9000.0),
        ("a", "x", 50000.0), ("x", "a", 50000.0),
    ]
    return pd.DataFrame(rows, columns=["from", "to", "cost"])


def test_distance_matrix_keeps_only_listed_sensors():
    dist = distance_matrix(_toy_distances(), ["a", "b", "c"])
    assert dist.shape == (3, 3)
    assert dist[0, 1] == 1000.0
    assert np.isinf(dist[1, 0])  # directed: no b -> a entry
    assert np.isinf(dist).sum() == 3


def test_sigma_uses_finite_entries_of_filtered_matrix():
    dist = distance_matrix(_toy_distances(), ["a", "b", "c"])
    sigma = np.array([0, 0, 0, 1000, 1500, 9000], dtype=np.float32).std()
    expected = np.exp(-np.square(dist / sigma))
    expected[expected < 0.1] = 0
    np.testing.assert_allclose(gaussian_kernel_adjacency(dist), expected)


def test_adjacency_diagonal_threshold_and_direction():
    adj = build_adjacency(_toy_distances(), ["a", "b", "c"])
    np.testing.assert_allclose(np.diag(adj), 1.0)
    assert adj[0, 1] > 0 and adj[1, 0] == 0
    assert adj[2, 0] == 0  # 9 km is below the kernel threshold
    assert np.all(np.diag(without_self_loops(adj)) == 0)


def test_adjacency_matches_dcrnn_pickle():
    ids = load_sensor_ids(require(GRAPH_DIR / "graph_sensor_ids.txt"))
    distances = load_distances(require(GRAPH_DIR / "distances_la_2012.csv"))
    pkl_ids, id_to_ind, pkl_adj = load_dcrnn_adj(require(GRAPH_DIR / "adj_mx.pkl"))

    assert pkl_ids == ids
    assert all(id_to_ind[s] == i for i, s in enumerate(ids))
    ours = build_adjacency(distances, ids)
    assert ours.shape == (207, 207)
    np.testing.assert_allclose(ours, pkl_adj, rtol=1e-5, atol=1e-6)
