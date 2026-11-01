from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GRAPH_DIR = ROOT / "data" / "raw" / "sensor_graph"
SPEED_H5 = ROOT / "data" / "raw" / "metr-la.h5"


def require(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"{path.relative_to(ROOT)} not downloaded (run scripts/download_data.sh)")
    return path
