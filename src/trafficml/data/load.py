"""Loaders for the METR-LA speed series and sensor-graph files."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import tables

# Only the numpy classes needed to rebuild arrays may be unpickled; pickles can execute code.
_ALLOWED_PICKLE_GLOBALS = {
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "scalar"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
}


class _NumpyOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if (module, name) not in _ALLOWED_PICKLE_GLOBALS:
            raise pickle.UnpicklingError(f"blocked global in pickle: {module}.{name}")
        return super().find_class(module, name)


def load_dcrnn_adj(path: str | Path) -> tuple[list[str], dict[str, int], np.ndarray]:
    """Load DCRNN's adj_mx.pkl (Python 2 pickle) -> (sensor_ids, sensor_id_to_ind, adj_mx)."""
    with open(path, "rb") as f:
        sensor_ids, sensor_id_to_ind, adj_mx = _NumpyOnlyUnpickler(f, encoding="latin1").load()
    return [str(s) for s in sensor_ids], {str(k): v for k, v in sensor_id_to_ind.items()}, adj_mx


def load_sensor_ids(path: str | Path) -> list[str]:
    return Path(path).read_text().strip().split(",")


def load_distances(path: str | Path) -> pd.DataFrame:
    """Road-network distances in metres: columns from, to (str), cost (float)."""
    return pd.read_csv(path, dtype={"from": str, "to": str, "cost": float})


def load_locations(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"sensor_id": str})
    return df[["sensor_id", "latitude", "longitude"]]


def load_speed(path: str | Path, sensor_ids: list[str] | None = None) -> pd.DataFrame:
    """Load metr-la.h5 as a (time x sensor) DataFrame; 0 mph marks a missing reading.

    If ``sensor_ids`` is given, checks the column order matches it.
    The file was written by pandas 0.15 under Python 2, which current ``pd.read_hdf`` cannot
    parse, so the arrays are read directly with PyTables.
    """
    with tables.open_file(path, mode="r") as f:
        columns = [c.decode() for c in f.root.df.block0_items.read()]
        if columns != [c.decode() for c in f.root.df.axis0.read()]:
            raise ValueError("block0_items and axis0 disagree; unexpected HDF5 layout")
        index = pd.DatetimeIndex(f.root.df.axis1.read().astype("datetime64[ns]"))
        df = pd.DataFrame(f.root.df.block0_values.read(), index=index, columns=columns)
    if sensor_ids is not None and list(df.columns) != list(sensor_ids):
        raise ValueError("speed columns do not match sensor_ids order")
    steps = df.index.to_series().diff().dropna()
    if not (steps == pd.Timedelta(minutes=5)).all():
        bad = steps[steps != pd.Timedelta(minutes=5)]
        raise ValueError(f"non-5-minute steps in index, e.g. {bad.head().to_dict()}")
    return df
