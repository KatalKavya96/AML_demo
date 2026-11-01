# T072 · Project 4 — Traffic Speed Forecasting on METR-LA

Phase 1 (machine learning): segment-level ML baselines (persistence, historical average, Ridge,
multi-output Random Forest, XGBoost per segment / global / graph-aware) on METR-LA, with masked
horizon-wise metrics and a leakage-proof chronological protocol. See [PHASE1_PLAN.md](PHASE1_PLAN.md).

Signal control is out of scope for Phase 1 and, in Phase 2, is simulated only.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
./scripts/download_data.sh          # metr-la.h5 (Google Drive) + sensor graph (DCRNN repo)

```

## Check

```bash
pytest                              # unit tests (real-data tests skip until files are present)
python scripts/check_data.py        # adjacency vs DCRNN, split dates, missing rate
```

## Run

```bash
python scripts/run_baselines.py     # M0 persistence, M1 historical average -> results/preds/
python scripts/evaluate.py --dm M1_ha M0_persistence   # metrics, leaderboard, DM test
```

Outputs: `results/leaderboard_{split}.csv` (MAE / RMSE / MAPE at 15, 30, 60 min per slice),
`results/tables/metrics_{split}.csv` (all 12 horizons), `results/tables/dm_*.csv`.

## Layout

| Path | Contents |
|---|---|
| `src/trafficml/data` | loaders, adjacency construction, masking, chronological splits |
| `src/trafficml/models` | M0 persistence, M1 historical average |
| `src/trafficml/eval` | masked metrics per horizon, prediction format, time slices, Diebold–Mariano + Holm |
| `configs/` | data paths, window and split settings |
| `scripts/` | data download, checks, baselines, evaluation |
| `tests/` | split, masking, metric, adjacency and unpickling tests |
| `results/` | leaderboard, saved predictions, tables, figures |

## Author contributions

| Member | Contribution |
|---|---|
| Rudraksh | Repo, data pipeline, splits, masking, adjacency, evaluation harness, significance tests, A5/A6 |
| Anshuman Mehta | EDA, shockwave analysis, congestion maps, literature table |
| Vridhi Chaudhary | Feature pipeline, graph features, leakage tests, A1/A2/A3 |
| Bineet Keshari | Baselines M0–M3, tuning, prediction format, A7 |
| Kavya Katal | XGBoost M4–M6, SHAP, A3/A4 |
