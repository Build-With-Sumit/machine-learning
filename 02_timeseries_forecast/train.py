"""
POC 2 — Time-series forecasting with a TensorFlow / Keras LSTM.

Univariate sequence forecasting: given the previous WINDOW months of a monthly
series, predict the next month. Trained on the canonical public AirPassengers
benchmark (144 monthly totals, 1949-1960) so the whole POC is self-contained and
reproducible with no private data.

Why this dataset stands in for a real use case: it has exactly the shape of the
operational series we care about -- daily inside-sales call volume -- an upward
trend with strong seasonality. The identical pipeline (windowing -> scaling ->
LSTM -> recursive multi-step forecast) is what runs on the real call-volume
series for capacity planning; only the CSV source changes.

Pipeline:
  * MinMax scaling fit on the TRAIN portion only (no leakage)
  * sliding-window supervised framing (WINDOW -> 1)
  * keras Sequential: LSTM(32) -> Dense(1)
  * chronological train/test split (last 24 months held out)
  * one-step test RMSE/MAE + a recursive 12-month forward forecast
  * a PNG plot written to artifacts/

Run:  .venv/Scripts/python.exe 02_timeseries_forecast/train.py
"""
from __future__ import annotations
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error, mean_absolute_error

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ART = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(ART, exist_ok=True)
SEED = 42
keras.utils.set_random_seed(SEED)

WINDOW = 12          # one year of history per prediction
TEST_MONTHS = 24     # last two years held out
FORECAST_H = 12      # recursive forecast horizon

# Canonical AirPassengers monthly totals (thousands), 1949-01 .. 1960-12.
SERIES = np.array([
    112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
    115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
    145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
    171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
    196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
    204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,
    242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
    284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
    315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,
    340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
    360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,
    417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432,
], dtype="float32")


def make_windows(x, window):
    X, y = [], []
    for i in range(len(x) - window):
        X.append(x[i:i + window])
        y.append(x[i + window])
    return np.array(X)[..., None], np.array(y)


def build_model():
    m = keras.Sequential([
        keras.Input(shape=(WINDOW, 1)),
        layers.LSTM(32),
        layers.Dense(1),
    ], name="lstm_forecaster")
    m.compile(optimizer=keras.optimizers.Adam(5e-3), loss="mse", metrics=["mae"])
    return m


def main():
    n = len(SERIES)
    split = n - TEST_MONTHS
    print(f"[data] AirPassengers: {n} months | train={split} test={TEST_MONTHS}")

    # scale on train only
    tr_min, tr_max = SERIES[:split].min(), SERIES[:split].max()
    scaled = (SERIES - tr_min) / (tr_max - tr_min)

    Xtr, ytr = make_windows(scaled[:split], WINDOW)
    # test windows use the tail of train as history (chronological, no leakage)
    Xte, yte = make_windows(scaled[split - WINDOW:], WINDOW)

    model = build_model()
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=30,
                                       restore_best_weights=True)
    model.fit(Xtr, ytr, validation_split=0.15, epochs=400, batch_size=8,
              callbacks=[es], verbose=0)

    # one-step-ahead test error (inverse-scaled to real passenger counts)
    def unscale(v):
        return v * (tr_max - tr_min) + tr_min

    p = model.predict(Xte, verbose=0).ravel()
    p_real, y_real = unscale(p), unscale(yte)
    rmse = np.sqrt(mean_squared_error(y_real, p_real))
    mae = mean_absolute_error(y_real, p_real)
    naive_rmse = np.sqrt(mean_squared_error(y_real[1:], y_real[:-1]))  # persistence
    print("\n===== ONE-STEP-AHEAD TEST =====")
    print(f"LSTM   RMSE={rmse:6.1f}  MAE={mae:6.1f}  (passengers, thousands)")
    print(f"naive  RMSE={naive_rmse:6.1f}   <- last-value baseline")
    print(f"LSTM beats naive by {100*(1-rmse/naive_rmse):4.1f}%")

    # recursive multi-step forecast beyond the series
    hist = list(scaled[-WINDOW:])
    fc = []
    for _ in range(FORECAST_H):
        nxt = model.predict(np.array(hist[-WINDOW:])[None, :, None],
                            verbose=0).ravel()[0]
        fc.append(nxt); hist.append(nxt)
    fc_real = unscale(np.array(fc))
    print(f"\n[forecast] next {FORECAST_H} months (thousands): "
          f"{np.round(fc_real).astype(int).tolist()}")

    # plot
    plt.figure(figsize=(10, 4))
    plt.plot(range(n), SERIES, label="actual", color="#1f77b4")
    test_idx = range(split, n)
    plt.plot(test_idx, p_real, "--", label="LSTM one-step (test)", color="#ff7f0e")
    plt.plot(range(n, n + FORECAST_H), fc_real, ":o", ms=3,
             label="LSTM 12-mo forecast", color="#2ca02c")
    plt.axvline(split, color="gray", ls=":", lw=1)
    plt.title("LSTM univariate forecast — AirPassengers benchmark")
    plt.xlabel("month index"); plt.ylabel("passengers (000s)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(ART, "forecast.png"), dpi=110)

    model.save(os.path.join(ART, "lstm_forecaster.keras"))
    with open(os.path.join(ART, "metrics.json"), "w") as f:
        json.dump({"test_rmse": float(rmse), "test_mae": float(mae),
                   "naive_rmse": float(naive_rmse),
                   "beat_naive_pct": float(100 * (1 - rmse / naive_rmse))},
                  f, indent=2)
    print(f"\n[saved] {ART}/lstm_forecaster.keras + forecast.png")


if __name__ == "__main__":
    main()
