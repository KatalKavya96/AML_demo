# Phase 1 Plan — Traffic Speed Forecasting on METR-LA

**Team T072 · Section C · Project 4: Multi-Agent Traffic Flow Control via Graph Attention Networks and LWR Models**
Rudraksh (2401010395) · Anshuman Mehta (2401010082) · Vridhi Chaudhary (2401010336) · Bineet Keshari (2401010130) · Kavya Katal (2401020031)

Scope: **Phase 1 = Machine Learning only.** GAT / deep learning, SUMO and signal control are Phase 2. Signal control is only ever *simulated*; we never claim live traffic control.

Status: merged plan after review rounds (v4, 25 Oct 2026). Week 1 progress is tracked in §17.

---

## 0. Hypothesis Phase 1 tests

> **H1.** Independent segment-level ML models (XGBoost / Random Forest) forecast well at short horizons but degrade at long horizons and at congestion onset, because they cannot see how congestion propagates between road links. Adding **graph-neighbour features** from the road graph improves them, mostly at h = 12 and at onset.

H1 is a hypothesis, not a conclusion. Whatever the result, the graph-aware tree model (M6) is framed as **evidence that graph information matters**, not as the fix. Even if H1 holds, three limits remain, and they are the Phase 2 motivation:

1. M6's neighbour weights are **fixed and hand-built**; a GAT learns attention weights that **change with the traffic state**.
2. M6 sees only 1–2 hops.
3. Residual errors at congestion onset show what is still unexplained.

---

## 1. Rubric → deliverables

| Rubric item (CSAI PDF) | What we deliver |
|---|---|
| Literature Review | Structured table of papers: method, dataset, result, limitation, relevance to us |
| Dataset Quality & EDA | Missingness/outage audit, temporal + spatial analysis, measured congestion-propagation speed |
| Feature Engineering | Causal lag / rolling / cyclic / profile / missingness / graph-neighbour features, all leakage-tested |
| Theoretical Rigor | Formulation, LWR physics, tree-ensemble maths, metric properties, time-series validation, significance testing |
| Model Application | M0–M6, horizon-wise metrics, peak vs off-peak, ablations A1–A7 |
| GitHub & Code Quality | `src/` package, configs, unit tests (incl. leakage), CI, README, reproducible scripts |
| Presentation | Each member presents the part they own |
| Viva Voce | Shared question bank + mock vivas; **everyone must explain everything** |
| Report | IEEE LaTeX (IEEEtran) |

Individual contribution must be visible (PDF §7): an **author-contributions table** goes in the report, the README and the final slide, in addition to git history.

Internal targets, **to confirm with faculty** (the PDF doesn't specify them): report 6–8 pages, 12–15 slides, 15–20 papers.

---

## 2. Dataset — METR-LA

- 207 loop detectors, Los Angeles County highways (Jagadish et al., 2014; released with DCRNN, Li et al., 2018).
- 5-minute speed readings (mph), 1 Mar – 27 Jun 2012 → **34,272 timesteps**.
- **Speed only. No flow, no occupancy.** The brief mentions both; we state this openly as a limitation. Optional extension: PeMS04/08 (flow + occupancy + speed), allowed by the "comparable or more complex dataset" rule.
- **0 = missing / sensor outage** (~8% of entries). Masked everywhere, never treated as zero traffic, never scored.

Files (`data/raw/`, gitignored):

| File | Content |
|---|---|
| `metr-la.h5` | pandas HDF5, DatetimeIndex × 207 sensor columns (needs `tables`) |
| `sensor_graph/distances_la_2012.csv` | `from,to,cost` road-network distances in metres; **contains more sensors than our 207**, includes self-pairs at 0 |
| `sensor_graph/graph_sensor_ids.txt` | the 207 sensor IDs, in column order |
| `sensor_graph/graph_sensor_locations.csv` | lat/lon for maps |
| `sensor_graph/adj_mx.pkl` | DCRNN's pre-built adjacency — used **only to verify ours** |

### Adjacency matrix — we construct it (brief: "construct the adjacency matrix from the supplied graph")

DCRNN method, implemented in `src/trafficml/data/graph.py`:

1. Start an N×N distance matrix at **∞**.
2. Fill it **only for pairs where both sensors are in the 207 `sensor_ids`**.
3. σ = std of the **finite entries of that filtered matrix** (not the raw CSV column).
4. W_ij = exp(−(d_ij/σ)²); set W_ij = 0 where W_ij < κ = 0.1.

Unit test: `np.allclose(ours, adj_mx.pkl)`. The pickle is loaded with a **restricted unpickler** that only allows numpy array classes (pickles can execute code).

Facts to know: W is **directed** (d_ij ≠ d_ji); the diagonal is 1 (self-pairs are at distance 0). Neighbour features exclude the diagonal. Which index is upstream is **checked empirically** (sign of the cross-correlation lag), not assumed.

Sanity check: h5 column order must equal `graph_sensor_ids.txt` order.

---

## 3. Problem formulation

- Road graph G = (V, E, W), |V| = N = 207; speed matrix X ∈ ℝ^{T×N}.
- Learn f : X_{t−11:t} → X̂_{t+1:t+12} (1 h history → 1 h ahead).
- Report h = 3, 6, 12 (15 / 30 / 60 min) in the main table, plus the **full h = 1…12 error curve**.

### Multi-step strategies (all three explained in the report)

| Strategy | Meaning | Used by |
|---|---|---|
| Recursive | one 1-step model, fed its own predictions | not used (error accumulates); explained only |
| **Direct** | one model per horizon | M4 (all 12 horizons), M5, M6 |
| **MIMO** | one model outputs all horizons jointly | M3 Random Forest (brief: "multi-output Random Forest") |

### Equal-history rule

Recent observations are limited to the **same 12-step window for every model**, including the Phase 2 GAT. Time features and the train-only historical profile are exogenous inputs that **the GAT will also receive**. In the report we justify fairness by "the GAT receives the same profile features", *not* by "the GAT could learn it" (a weight-shared GAT with only time-of-day input cannot easily learn per-sensor profiles). Ablation A7 measures what the profile contributes.

---

## 4. Leakage-proof data protocol

1. **Chronological split by sample window, DCRNN convention** (`generate_training_data`), so our numbers are comparable with the literature. Anchors t ∈ [11, T−12) → 34,249 samples. `n_test = round(0.2n) = 6,850`, `n_train = round(0.7n) = 23,974`, `n_val = 3,425`.
2. **Validation is split in time into two halves** (decided now, for the Phase 2 hybrid — see §7):
   - `val_tune` (first half, ≈ 23–29 May): hyperparameter tuning and early stopping. **Caveat:** only ~6 days, including a weekend and Memorial Day, so about three normal working days — tuning on it is noisy. When candidate settings are close, decide with `TimeSeriesSplit` inside train instead.
   - `val_stack` (second half, ≈ 29 May–4 Jun): **reserved** — models predict on it and the predictions are saved, but no Phase 1 decision uses it. Phase 2 trains the stacker here.

   Approximate anchor ranges (assuming the series starts 1 Mar 00:00 — confirmed in Week 1):

   | Split | Anchors (prediction times) |
   |---|---|
   | train | 1 Mar 00:55 → 23 May 06:40 |
   | val_tune | 23 May 06:45 → 29 May ~05:20 (contains Memorial Day, 28 May) |
   | val_stack | 29 May → 4 Jun ~04:05 |
   | test | 4 Jun → 27 Jun ~22:55 |
3. **Never split randomly.** Ablation A5 does it once on purpose to show how optimistic it is.
4. **Everything statistical is fitted on train timestamps only:** scalers, historical profiles, free-flow speed, encodings.
5. **Historical profile for training rows is leave-one-week-out**, because a train-only time-of-week mean at the target time would otherwise contain that row's own target (~1/12 of the weight). Val/test rows use the full-train profile.
6. **Input imputation is causal only:** forward fill with a limit + missingness indicators. No interpolation that looks ahead.
7. **Targets are never imputed.** Direct models drop masked targets per horizon. The MIMO Random Forest **drops rows with any masked target** (scikit-learn rejects NaN in y); the share dropped is reported and the bias (training skews away from outage periods) is stated in the report.
8. **Test is used exactly once**, for the final table.
9. **The scored test index is saved** — the exact (t, sensor, h) points with a valid target. Every model, including the Phase 2 GAT, is scored on identical points.
10. **Unit tests enforce this:** split order and sizes, no future values in features, scaler fitted on train only, masking applied, adjacency matches DCRNN.

---

## 5. EDA plan

**A. Data quality**
- Shape, time range, 5-minute frequency, duplicate or missing timestamps.
- **Daylight saving time (11 Mar 2012):** are timestamps local or UTC? Look for a 1-hour gap or a shift in the daily profile around that date.
- Print the exact split dates.
- Missing rate overall, per sensor, per hour; sensor × time missingness heatmap.
- Outage run-length distribution (short dropouts vs multi-day failures); list of mostly-dead sensors (feeds M4's fallback rule).
- Speed distribution, the ~70 mph cap, implausible values.

**B. Temporal**
- Daily profile weekday vs weekend; weekly profile; STL on sample sensors.
- ACF/PACF (justifies 12 lags, shows 288-step daily seasonality); ADF + KPSS.
- **Data-driven peak windows** — start from weekdays 07:00–10:00 and 16:00–19:00 and confirm from profiles.
- Distribution shift between splits (mean speed by hour, train vs test).

**C. Spatial and graph**
- Graph statistics: nodes, edges, sparsity, degree distribution, connected components, directedness.
- Sensor map coloured by mean speed and by missing rate.
- Congestion snapshots: weekday 08:00 and 17:30.
- Correlation vs graph distance.
- **Shockwave speed, done properly:** at 15–20 km/h a 5-min step covers only ~1.3–1.7 km, so single adjacent pairs give lags of 1–2 steps — too coarse. Instead build **sensor chains of 5–10 km from the directed graph** (METR-LA has no freeway/direction labels), cross-correlate speed drops along each chain, and **regress lag on distance**; the slope gives the propagation speed. Compare with the LWR backward-wave speed. The same analysis confirms which adjacency direction is upstream.

---

## 6. Feature engineering

Long format: one row per (anchor t, sensor). Critical path: **v0 (lags + time + profile) by end of Week 1, graph features in Week 2.**

| Group | Features |
|---|---|
| Lags | speed at t, t−1 … t−11 |
| Window stats | mean, std, min, max, last − first, 1-step and 3-step differences |
| Missingness | per-lag missing flag, count missing in window, steps since last valid reading **capped at 12** |
| Cyclic time | sin/cos time of day (period 288), sin/cos day of week, is_weekend. **No `is_holiday`** — train has no positive examples (Memorial Day falls in val). |
| Historical profile | train-only time-of-week mean **at the target time t+h** (horizon-specific: direct models get one column, the MIMO RF gets 12), plus current deviation from the profile at t. Leave-one-week-out on training rows. |
| Segment static | sensor ID (categorical, pooled models), free-flow speed (train 85th percentile), lat/lon, in/out-degree |
| Congestion state | congestion index 1 − v/v_ff, is_congested (e.g. < 60% of free flow) |
| **Graph neighbours** (M6 only) | W-weighted mean of 1-hop upstream lags, 1-hop downstream lags, 2-hop mean, neighbour − self difference, neighbours' minimum speed. **Diagonal excluded.** |

Viva link: Σ_j W_ij x_j is one diffusion step over the graph — the operation DCRNN is built on.
Trees need no scaling; Ridge uses a StandardScaler fitted on train only. XGBoost handles NaN natively; RF uses causal fill + indicators (or scikit-learn ≥ 1.4 NaN support in X).

---

## 7. Models

| ID | Model | Strategy | Purpose |
|---|---|---|---|
| M0 | Persistence (last valid value) | — | Floor; sets expectations |
| M1 | Historical Average (time of week) | — | Standard literature baseline |
| M2 | Ridge (pooled) | direct | Linear reference |
| M3 | Random Forest, multi-output, pooled | **MIMO** | Required by the brief |
| M4 | **XGBoost per segment** | **direct, all 12 horizons** (207 × 12 = 2,484 models, ~24k rows each, `tree_method="hist"`) | Brief's main baseline: independent segments |
| M5 | XGBoost global (pooled + sensor features) | direct | Information sharing vs M4 |
| M6 | **XGBoost global + graph-neighbour features** | direct | **Graph evidence**; identical to M5 except the graph features |

Notes:
- **M4 fallback:** sensors with fewer than a minimum number of valid training rows use M5's prediction; the count of fallback sensors is reported. M4 runtime is **timed in Week 3**, not assumed ("tens of minutes" is the guess).
- **Tuning:** on `val_tune` only (Optuna or randomized search), early stopping for XGBoost on `val_tune`. XGB: max_depth, learning_rate, n_estimators, min_child_weight, subsample, colsample_bytree, λ, γ. RF: n_estimators, max_depth, max_features, min_samples_leaf, max_samples.
- **Compute (8 GB RAM Mac):** ~5M train rows × ~60 features float32 ≈ 1.2 GB — fits, but RF and pooled XGB need float32 everywhere, `max_samples` for RF, and strided anchors if memory is tight. Every such shortcut is reported.
- Optional if time allows: ARIMA or LightGBM on a subset of sensors.

### Saved predictions (set up for the Phase 2 Hybrid bonus)

Every model writes `results/preds/{model}_{split}.npz` (`anchors`, `y_pred` of shape samples × 12 × 207) for `split ∈ {val_tune, val_stack, test}`; targets are stored once per split in `targets_{split}.npz` (`anchors`, `sensor_ids`, `y_true` with NaN = not scored). Arrays rather than a long table, because a test split is 17M points (8 GB RAM). `save_predictions` refuses files whose anchors or shape differ from the targets, or that have NaN on a scored point.

**Stacker training data is decided:** the Phase 2 hybrid (e.g. XGBoost on GAT residuals, or trees on GAT embeddings) is trained on **`val_stack`** and evaluated on test. Fallback if `val_stack` (~6 days) proves too small: rolling-origin out-of-fold predictions on the last weeks of train (more compute).

---

## 8. Evaluation protocol

- **Metrics:** masked MAE, RMSE, MAPE (score only where the true value ≠ 0) at h = 3, 6, 12 and the h = 1…12 curve.
- **Slices:** peak vs off-peak (required), weekday vs weekend, free-flow vs congested, **congestion onset** (first 30 min of a speed drop), per-sensor error map.
- **Diebold–Mariano test — main comparison M5 vs M6** (graph features are the only difference):
  - loss differential d_t = mean over sensors of (|e_M5| − |e_M6|) at each timestamp,
  - HAC (Newey–West) variance with lag ≥ h − 1,
  - run separately per horizon.
- **Holm correction** across all ablation tests *and* horizons.
- Day-level block-bootstrap 95% CIs for headline metrics.
- **Interpretability:** XGBoost gain + TreeSHAP; do neighbour features rank high at h = 12?
- **Residual ACF** (leftover structure the model missed).
- **Congestion maps for the forecast comparison (deliverable):** predicted vs actual maps for M4 and M6 at a weekday 17:30 timestamp, h = 12.
- **Literature numbers** (HA, ARIMA, SVR, FC-LSTM, DCRNN, Graph WaveNet on METR-LA) appear only as context in the related-work table, each verified against its source. **No target numbers for our models** — expectations come from M0/M1.

---

## 9. Ablations

| ID | Ablation | Owner |
|---|---|---|
| A1 | History window 3 / 6 / 12 steps | Vridhi |
| A2 | With vs without cyclic time features | Vridhi |
| A3 | **No neighbours → upstream only → downstream only → 1-hop → 1+2-hop** (key) | Kavya + Vridhi |
| A4 | Per-segment (M4) vs global (M5) | Kavya |
| A5 | Chronological vs random split (leakage demo) | Rudraksh |
| A6 | Masking vs zero-filling outages | Rudraksh |
| A7 | With vs without historical-profile features | Bineet |

---

## 10. Theoretical Rigor (one report subsection each)

1. **LWR model:** ∂ρ/∂t + ∂q/∂x = 0, q = ρ·v(ρ); Greenshields v = v_f(1 − ρ/ρ_j); characteristic speed dq/dρ; shockwave speed w = (q₂ − q₁)/(ρ₂ − ρ₁). Why congestion travels upstream and why independent-segment models miss it.
2. **Graph construction:** Gaussian kernel, threshold κ, sparsity, directedness, neighbour feature as one diffusion step.
3. **Random Forest:** bagging variance ρσ² + (1 − ρ)σ²/B; `max_features` lowers ρ; **why OOB error is invalid for autocorrelated series**; why MIMO forces dropping rows with masked targets.
4. **Gradient boosting / XGBoost:** second-order Taylor objective, Ω = γT + ½λ‖w‖², leaf weight w* = −G/(H + λ), split gain.
5. **Per-segment vs global:** bias–variance, data per model.
6. **Recursive vs direct vs MIMO** multi-step strategies.
7. **Metrics:** MAE → median, RMSE → mean and large errors, MAPE asymmetric and unstable at low speed (Hyndman & Koehler, 2006).
8. **Time-series validation:** why i.i.d. CV fails, rolling origin, purge gaps; leave-one-week-out profiles as a leakage fix.
9. **Diebold–Mariano** with HAC variance, block bootstrap, Holm correction.

---

## 11. Literature review list

- **Traffic flow theory:** Lighthill & Whitham (1955); Richards (1956); Greenshields (1935); Daganzo (1994), Cell Transmission Model.
- **Classical forecasting:** Williams & Hoel (2003), seasonal ARIMA; Castro-Neto et al. (2009), SVR; Vlahogianni et al. (2014), survey.
- **ML methods:** Breiman (2001); Friedman (2001); Chen & Guestrin (2016); Lundberg & Lee (2017).
- **Spatio-temporal DL (motivation):** Ma et al. (2015), LSTM; Li et al. (2018), DCRNN; Yu et al. (2018), STGCN; Wu et al. (2019), Graph WaveNet; Shao et al. (2022), STID; Jiang & Luo (2022), GNN traffic survey.
- **Brief's pre-reads:** Veličković et al. (2018), GAT; Zheng et al. (2019), CityFlow.
- **Evaluation:** Hyndman & Koehler (2006); Bergmeir & Benítez (2012); Diebold & Mariano (1995).

---

## 12. Repository structure

```
AML/                              (repo root)
├── PHASE1_PLAN.md
├── README.md                     setup, data download, reproduce, contributions table
├── requirements.txt, pyproject.toml
├── configs/data.yaml
├── data/                         gitignored: raw/, processed/
├── scripts/                      download_data.sh, check_data.py, (build_features.py, train.py, evaluate.py)
├── src/trafficml/
│   ├── data/      load.py, graph.py, splits.py, masking.py
│   ├── features/  lags.py, temporal.py, profiles.py, graph_feats.py
│   ├── models/    baselines.py, ridge.py, rf.py, xgb.py
│   ├── eval/      metrics.py, slices.py, significance.py
│   └── viz/       maps.py, plots.py
├── notebooks/                    thin; call src/
├── tests/
├── results/                      leaderboard.csv, preds/, tables/, figures/
├── report/                       IEEE LaTeX
└── slides/
```

Practices: ruff via pre-commit, type hints, fixed seeds, pytest + GitHub Actions CI, one branch per feature, PRs reviewed by another member, **each member commits their own work**.

---

## 13. Roles

| Member | Owns | Report / slides section |
|---|---|---|
| **Rudraksh** (lead) | Repo, loader, splits, masking, adjacency build, evaluation harness, peak/off-peak slicing, DM + bootstrap + Holm, A5, A6, integration + CI | Methodology & evaluation protocol |
| **Anshuman Mehta** | EDA (quality, DST, temporal, spatial, chain shockwave analysis); then prediction congestion maps + literature table (Weeks 3–4) | Dataset & EDA, related work |
| **Vridhi Chaudhary** | Feature pipeline (v0 Week 1, graph features Week 2), leakage tests, A1, A2, A3 (with Kavya) | Feature engineering |
| **Bineet Keshari** | M0–M3, tuning, saved-prediction format, A7 | Baselines & RF theory |
| **Kavya Katal** | M4–M6, SHAP, A3 (with Vridhi), A4 | XGBoost theory & results |

Everyone summarises 3–4 papers and writes the theory for their own part.

---

## 14. Timeline

Both phases are evaluated after mid-sems and dates aren't fixed, so Phase 2 may follow Phase 1 closely. Aim for a **minimum complete Phase 1 by the end of Week 3**.

| Week | Dates | Milestones |
|---|---|---|
| W1 | 26 Oct – 1 Nov | Repo, CI, loader, splits, masking, metrics, adjacency build + tests ✔ skeleton; download h5; confirm split dates + DST; **feature v0**; M0/M1; papers assigned |
| W2 | 2 – 8 Nov | Full EDA; graph features; leakage tests; literature table started; M2 |
| W3 | 9 – 15 Nov | M3–M6 trained + tuned; M4 runtime timed; **first full leaderboard (minimum complete Phase 1)** |
| W4 | 16 – 22 Nov | Ablations A1–A7, slices, DM + Holm, bootstrap, SHAP, prediction congestion maps |
| W5 | 23 – 29 Nov | IEEE report draft, slides, README, clean-up; **start Phase 2 loader + GAT prototype** |
| W6 | 30 Nov – 6 Dec | Buffer, final edits, **mock vivas** (each member questioned on others' parts) |

---

## 15. Risks

| Risk | Mitigation |
|---|---|
| No flow/occupancy in METR-LA | State it; optional PeMS04/08 appendix |
| 8 GB RAM | float32, RF `max_samples`, strided anchors; report shortcuts |
| M4 (2,484 models) slower than expected | Time it in W3; parallelise per sensor; fall back to h = 3/6/12 for M4 only if necessary and say so |
| RF drops many rows (masked targets) | Report share dropped; compare with direct models on identical scored points |
| Leakage found late | Leakage tests written in W1–W2 |
| Neighbour features don't help | Still a finding; analyse why (adjacency, horizon); still motivates learned attention |
| `val_stack` too small for the stacker | Rolling-origin OOF fallback |
| Single point of knowledge | Rotating code review + mock vivas |

---

## 16. Viva question bank (grow to ~40)

- Why is 0 treated as missing? What goes wrong if you zero-fill? (A6)
- Why can't you split randomly? Why is OOB error invalid here?
- How did you build the adjacency matrix? Why is it directed? What is σ computed over?
- Derive the XGBoost leaf weight. What do λ and γ do?
- Why does RF reduce variance? Role of `max_features`?
- Recursive vs direct vs MIMO — which did you use where, and why?
- Why does the RF drop rows with any masked target, and what bias does that cause?
- Explain LWR and shockwaves. Which way does congestion travel, how fast, and how did you measure it?
- Why is the historical profile leave-one-week-out for training rows?
- Is the profile a violation of the equal-history rule? (No: it's train-fitted, and the GAT gets it too; A7 measures it.)
- MAPE at low speeds — why report MAE and RMSE as well?
- Per-segment vs global — when is each better?
- How do you know the M6 gain is significant? Explain the DM setup and the Holm correction.
- What gap does a GAT fix that M6 doesn't?
- How will Phase 1 feed the Phase 2 hybrid? (saved predictions, `val_stack`)

---

## 17. Progress log

**Week 1 (25 Oct 2026)**
- ✔ Environment: `.venv` with `requirements.txt`; package installed with `pip install -e .`.
- ✔ Data: `scripts/download_data.sh` fetches `metr-la.h5` + `pems-bay.h5` (Google Drive, via gdown) and the sensor graph (DCRNN repo).
- ✔ The 2012 HDF5 file cannot be read by current `pd.read_hdf`; `load_speed` reads the arrays with PyTables instead.
- ✔ `scripts/check_data.py`: adjacency rebuilt from distances matches DCRNN (207 sensors, 1,515 directed edges excluding self-loops, not symmetric); column order matches; series 2012-03-01 00:00 → 2012-06-27 23:55, 34,272 × 207; **8.11 % missing**. Split anchors exactly as in §4. *It does not check daylight saving* — that is the EDA item in §5 (Anshuman).
- ✔ Evaluation harness: prediction format, masked metrics per horizon, peak/off-peak and weekday/weekend slices, Diebold–Mariano (Newey–West, lag = max(h−1, rule of thumb)) with Holm, leaderboard (`scripts/evaluate.py`).
- ✔ M0 persistence and M1 historical average (`scripts/run_baselines.py`). Test MAE at 15/30/60 min: M0 3.48 / 4.17 / 5.32, M1 4.19 (all horizons; DCRNN reports 4.16 for its HA). Peak MAE is much higher (M0 4.56 / 5.81 / 7.78).
- ✔ 20 unit tests, ruff clean.
- ☐ Feature v0 (Vridhi), EDA incl. DST check (Anshuman), git repo + GitHub remote + CI.
