#!/usr/bin/env bash
# Downloads the METR-LA speed series (DCRNN Google Drive folder, via gdown) and the
# sensor graph files from the DCRNN repository into data/raw/.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GRAPH_DIR="$ROOT/data/raw/sensor_graph"
BASE="https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph"
mkdir -p "$GRAPH_DIR"

for f in adj_mx.pkl distances_la_2012.csv graph_sensor_ids.txt graph_sensor_locations.csv; do
  if [[ -f "$GRAPH_DIR/$f" ]]; then
    echo "exists  $f"
  else
    curl -sSfL -o "$GRAPH_DIR/$f" "$BASE/$f"
    echo "fetched $f"
  fi
done

DRIVE_FOLDER="https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX"
if [[ -f "$ROOT/data/raw/metr-la.h5" ]]; then
  echo "exists  metr-la.h5"
else
  TMP="$ROOT/data/raw/_gdrive"
  gdown --folder "$DRIVE_FOLDER" -O "$TMP"   # also fetches pems-bay.h5
  mv "$TMP"/*.h5 "$ROOT/data/raw/" && rmdir "$TMP"
  echo "fetched metr-la.h5, pems-bay.h5"
fi
