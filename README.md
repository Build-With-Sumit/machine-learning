# Machine Learning — production-shaped TensorFlow POCs

Small, **end-to-end** TensorFlow / Keras models on **real operational data** from
a live inside-sales system — built the honest way: leakage-safe preprocessing,
cross-validation, class-imbalance handling, and metrics reported with error bars.
Not notebook toys, and no cherry-picked numbers.

Each POC is a complete `train.py`: load → engineer features → train → evaluate
with cross-validation → persist a `.keras` model + `metrics.json`. Every folder
runs standalone — real data is de-identified and kept private, so a **synthetic
fallback** with the same signal ships in its place.

| # | POC | Problem family | Key pieces | Headline result |
|---|-----|----------------|------------|-----------------|
| 1 | [Call connect-rate](01_call_connect_rate) | Tabular classification | `Normalization`+`StringLookup` DNN **vs** `HistGradientBoosting` | 5-fold CV ROC-AUC **0.60**, AP **0.55** (base 0.41); `hour`+`rep` drive it |
| 2 | [Time-series forecaster](02_timeseries_forecast) | Sequence forecasting | `LSTM` → Dense, sliding windows | RMSE **19% better** than the persistence baseline |

## 1 · Call connect-rate classifier (flagship)

**Real problem:** the inside-sales team makes hundreds of calls a day and only
~41% connect. Predicting *at dial time* whether a call will be answered lets reps
dial at better times and prioritize numbers more likely to pick up.

**What it does:** binary classification of `answered` from **dial-time features
only** — rep, direction, line, hour-of-day, day-of-week, counterpart-repeat.
Post-call columns (duration, disposition) are deliberately **excluded as
leakage**. It trains a Keras tabular DNN *and* a scikit-learn
`HistGradientBoostingClassifier` on the same folds — because on small structured
data, gradient-boosted trees are the honest baseline to beat.

**Result:** both land at **ROC-AUC ≈ 0.60** (a ~34% precision lift over the 0.41
base rate), and permutation importance gives a clean, actionable finding:
**`hour` and `rep` drive connect rate; which line/account you use doesn't.**

## 2 · LSTM time-series forecaster

Univariate sequence forecasting on the canonical public **AirPassengers**
benchmark (trend + strong seasonality) — a self-contained stand-in for the real
operational series it targets (daily call volume for capacity planning). Sliding-
window framing → `LSTM(32)` → `Dense(1)`, MinMax scaling fit on the train split
only, chronological (no-shuffle) split, one-step test error **plus** a recursive
12-month forward forecast, evaluated against a naive persistence baseline (which
it beats by ~19%). The pipeline is identical for the real series; only the CSV
source changes.

## Run it

```bash
python -m venv .venv                       # Python 3.11 (TF has no 3.14 wheels)
.venv/Scripts/python -m pip install -r requirements.txt   # or bin/ on *nix

.venv/Scripts/python 01_call_connect_rate/train.py
.venv/Scripts/python 02_timeseries_forecast/train.py
```

Each prints its metrics and writes a `.keras` model + `metrics.json` into its
`artifacts/` folder (gitignored — regenerated on every run). POC 1 uses a local
de-identified snapshot if present, else a synthetic sample; POC 2 embeds its
public dataset. **No private data needed to run either.**

## The engineering discipline (why the numbers are honest)

These are the details that separate a real model from a misleading demo — and
they're documented, not hidden:

- **Leakage control** — features must be knowable *at prediction time*; the
  vectorizer/scaler is fit on the training fold only; time series split
  chronologically.
- **Cross-validation** — small n, so results are a mean ± std over 5 stratified
  folds, not one lucky split.
- **Right metrics** — ROC-AUC and average-precision against the base rate, never
  raw accuracy (which the class balance would make meaningless).
- **Baseline-relative** — the forecaster is judged against naive persistence; a
  score that can't beat "guess the last value" is worthless.

Full engineer-facing walkthroughs in [`docs/`](docs/) — including a from-scratch
[explainer](docs/explainer.html) for readers with no ML background.

## Stack
TensorFlow 2.17 / Keras 3 · scikit-learn (splits, metrics, gradient boosting) ·
pandas / NumPy · matplotlib. CPU-only, Python 3.11. MIT licensed.
