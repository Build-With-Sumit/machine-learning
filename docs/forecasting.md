# POC 2 — LSTM time-series forecaster — `02_timeseries_forecast/train.py`

> **Audience:** engineer, no ML background. Self-contained — no dependency on the
> other POC or any private data. A different problem family (forecasting) on the
> same underlying training loop.

## The task

**Forecasting:** given the last 12 months of a monthly series, predict month 13.
Then do it *recursively* to project a year into the future. This is regression
(predict a real number), not classification, so the loss and metrics differ from
POC 1 — but `model.fit()` is the same loop underneath.

## Why AirPassengers

```python
SERIES = np.array([112, 118, 132, ... 432])   # 144 monthly totals, 1949–1960
```

It's the canonical public forecasting benchmark, embedded directly in the file so
the POC runs with **zero external data**. It has the two features that make
forecasting interesting — a steady **trend** and a strong repeating 12-month
**seasonality** — and the same shape as the real operational series it stands in
for: **daily inside-sales call volume** for capacity planning. The identical code
runs on the real series; only the CSV source changes.

## Manufacturing training examples — sliding windows

```python
def make_windows(x, window):
    for i in range(len(x) - window):
        X.append(x[i:i + window])     # 12 consecutive months
        y.append(x[i + window])       # the 13th = the answer
    return np.array(X)[..., None], np.array(y)
```

Supervised learning needs `(input, answer)` pairs, but a time series is one long
list. The **sliding window** manufactures the pairs: slide a 12-month window
along the series; each window is an input, the next month is its label. The
trailing `[..., None]` reshapes to `(samples, timesteps, 1)` — an LSTM expects a
feature axis, and we have one feature per timestep (univariate).

## Leakage-safety in a time series — two subtle spots

Time-series leakage is sneakier, because "the future" must never touch "the past":

1. **Scale on train-only stats.** MinMax scaling uses `min`/`max` from the
   **training portion only**, then applies them to everything. Global min/max
   would leak the future's range. `unscale()` reverses it so errors report in real
   units.
2. **Split chronologically, never randomly.** Random shuffling would train on 1960
   to predict 1955. Instead the last 24 months are held out as the test set; test
   windows may use the tail of training data as legitimate *history*.

## The model — why an LSTM

```python
keras.Sequential([keras.Input(shape=(WINDOW, 1)), layers.LSTM(32), layers.Dense(1)])
```

An **LSTM** (Long Short-Term Memory) reads a sequence step-by-step and carries an
internal **memory cell** with learned gates deciding what to remember, forget, and
output — which lets it track trend and seasonal position, something a plain
`Dense` layer (an unordered bag of 12 inputs) can't do as naturally. `LSTM(32)`
outputs a 32-dim summary after the last step; `Dense(1)` maps it to the predicted
next month. Loss is **MSE** (regression), metric **MAE** (average miss in real
units).

## Two kinds of prediction

**One-step-ahead** (measured vs a baseline): predict each held-out month from real
history → **RMSE ≈ 42.7**. But a raw error is meaningless without a baseline. The
honest one is **persistence** ("next = last value"), which scores **RMSE ≈ 52.7**.
The LSTM beats it by **~19%** — that's the real result. Always quote a forecast
error against a naive baseline.

**Recursive multi-step** (the actual forecast): to project past the data, the
model **eats its own predictions** — predict +1, append it, predict +2 from that,
and so on. Errors compound the further out you go, which is why long-horizon
forecasts fan out in uncertainty. A saved `forecast.png` shows the series, the
one-step test fit, and the 12-month projection — visual proof the forecast is
*seasonal*, not a flat line.

## What this POC demonstrates

The complete, correct forecasting pipeline — windowing, leakage-safe scaling,
chronological splits, an LSTM, baseline-relative evaluation, recursive
projection — on a benchmark anyone can reproduce, structured so pointing it at the
real call-volume series is a one-line data swap.

---

**Back to:** [README.md](README.md) (docs index) · [connect-rate.md](connect-rate.md)
