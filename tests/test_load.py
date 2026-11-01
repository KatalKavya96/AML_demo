import os
import pickle

import pytest

from trafficml.data.load import load_dcrnn_adj


class _Evil:
    def __reduce__(self):
        return (os.getcwd, ())


def test_unpickler_blocks_non_numpy_globals(tmp_path):
    path = tmp_path / "evil.pkl"
    path.write_bytes(pickle.dumps(_Evil()))
    with pytest.raises(pickle.UnpicklingError, match="blocked"):
        load_dcrnn_adj(path)
