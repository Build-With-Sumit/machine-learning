"""
POC 1 — sales-call connect-rate classification (real business data).

Task: predict whether a call will be ANSWERED, from features knowable at dial
time (which rep, inbound/outbound, which line, hour-of-day, day-of-week, how
often we've dialed this counterpart). Business use: connect-rate is real — it
varies ~26%->49% by rep and by time-of-day — so a model that ranks *when* and
*by whom* a call is likely to connect helps the inside-sales team dial smarter.

A standard tabular-DNN pattern (Keras Normalization + StringLookup), run
head-to-head against a gradient-boosted-trees baseline (scikit-learn
HistGradientBoosting) — because on small structured data, trees are the honest
thing to beat.

  * balanced ~41% label, so we can report ROC-AUC/AP without the imbalance games
  * stratified 5-fold CV for a stable estimate on n~559
  * Keras in-graph preprocessing (Normalization + StringLookup) so the saved
    model eats a raw feature dict
  * permutation importance to show which dial-time features actually carry signal

Run:  .venv/Scripts/python.exe 01_call_connect_rate/train.py
"""
from __future__ import annotations
import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.inspection import permutation_importance

sys.path.insert(0, os.path.dirname(__file__))
from data import load_calls, NUMERIC, CATEGORICAL  # noqa: E402

ART = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(ART, exist_ok=True)
SEED = 42
keras.utils.set_random_seed(SEED)


# ---------------------------------------------------------------- Keras DNN ---
def build_dnn(df_train) -> keras.Model:
    inputs, encoded = {}, []

    num_in = keras.Input(shape=(len(NUMERIC),), name="numeric")
    norm = layers.Normalization(name="norm")
    norm.adapt(df_train[NUMERIC].values.astype("float32"))
    inputs["numeric"] = num_in
    encoded.append(norm(num_in))

    for col in CATEGORICAL:
        cin = keras.Input(shape=(1,), dtype=tf.string, name=col)
        lookup = layers.StringLookup(output_mode="one_hot")
        lookup.adapt(tf.constant(df_train[col].values))
        inputs[col] = cin
        encoded.append(layers.Reshape((-1,))(lookup(cin)))

    x = layers.Concatenate()(encoded)
    x = layers.Dense(24, activation="relu",
                     kernel_regularizer=keras.regularizers.l2(1e-3))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(12, activation="relu",
                     kernel_regularizer=keras.regularizers.l2(1e-3))(x)
    out = layers.Dense(1, activation="sigmoid", name="answered")(x)

    model = keras.Model(inputs, out, name="connect_rate_dnn")
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy", metrics=[keras.metrics.AUC(name="auc")])
    return model


def to_inputs(df):
    d = {"numeric": tf.constant(df[NUMERIC].values.astype("float32"))}
    for col in CATEGORICAL:
        d[col] = tf.constant(df[col].values.reshape(-1, 1), dtype=tf.string)
    return d


def class_weight(y):
    pos = y.sum(); neg = len(y) - pos
    return {0: len(y) / (2 * neg), 1: len(y) / (2 * pos)}


def train_dnn(tr, va):
    y_tr = tr["answered"].values.astype("float32")
    y_va = va["answered"].values.astype("float32")
    model = build_dnn(tr)
    es = keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                       patience=15, restore_best_weights=True)
    model.fit(to_inputs(tr), y_tr, validation_data=(to_inputs(va), y_va),
              epochs=150, batch_size=16, class_weight=class_weight(y_tr),
              callbacks=[es], verbose=0)
    return model


# ------------------------------------------------------ Gradient-boosted trees ---
def gb_pipeline():
    """OrdinalEncode categoricals, tell HGB which columns are categorical."""
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    cat_idx = [NUMERIC.__len__() + i for i in range(len(CATEGORICAL))]
    clf = HistGradientBoostingClassifier(
        categorical_features=cat_idx, learning_rate=0.06, max_iter=250,
        max_depth=3, l2_regularization=1.0, class_weight="balanced",
        random_state=SEED)
    return enc, clf


def gb_matrix(df, enc, fit):
    cats = enc.fit_transform(df[CATEGORICAL]) if fit else enc.transform(df[CATEGORICAL])
    return np.hstack([df[NUMERIC].values.astype("float32"), cats])


# ---------------------------------------------------------------------- main ---
def main():
    df = load_calls()
    y = df["answered"].values.astype("int")
    print(f"[data] {len(df)} calls | {int(y.sum())} answered "
          f"({y.mean():.1%}) | {len(NUMERIC)} numeric + {len(CATEGORICAL)} categorical")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    dnn_auc, dnn_ap, gb_auc, gb_ap = [], [], [], []
    for k, (tr_i, va_i) in enumerate(skf.split(df, y), 1):
        tr, va = df.iloc[tr_i], df.iloc[va_i]

        dnn = train_dnn(tr, va)
        p_dnn = dnn.predict(to_inputs(va), verbose=0).ravel()

        enc, gb = gb_pipeline()
        gb.fit(gb_matrix(tr, enc, fit=True), tr["answered"].values)
        p_gb = gb.predict_proba(gb_matrix(va, enc, fit=False))[:, 1]

        a_dnn = roc_auc_score(va["answered"], p_dnn); p1 = average_precision_score(va["answered"], p_dnn)
        a_gb = roc_auc_score(va["answered"], p_gb);  p2 = average_precision_score(va["answered"], p_gb)
        dnn_auc.append(a_dnn); dnn_ap.append(p1); gb_auc.append(a_gb); gb_ap.append(p2)
        print(f"  fold {k}: DNN AUC={a_dnn:.3f} AP={p1:.3f}  |  GBT AUC={a_gb:.3f} AP={p2:.3f}")

    print(f"\n[cv] DNN  ROC-AUC = {np.mean(dnn_auc):.3f} +/- {np.std(dnn_auc):.3f}  AP = {np.mean(dnn_ap):.3f}")
    print(f"[cv] GBT  ROC-AUC = {np.mean(gb_auc):.3f} +/- {np.std(gb_auc):.3f}  AP = {np.mean(gb_ap):.3f}")
    print(f"[cv] baseline (base rate) = {y.mean():.3f}")

    # ---- final DNN on a held-out split, saved ---------------------------
    tr, te = train_test_split(df, test_size=0.2, stratify=y, random_state=SEED)
    tr2, va = train_test_split(tr, test_size=0.2, stratify=tr["answered"], random_state=SEED)
    model = train_dnn(tr2, va)
    p = model.predict(to_inputs(te), verbose=0).ravel()
    print("\n===== HELD-OUT TEST (20%) — DNN =====")
    print(classification_report(te["answered"], (p >= 0.5).astype(int),
          target_names=["missed", "answered"], zero_division=0))

    # ---- which dial-time features matter (GBT permutation importance) ---
    enc, gb = gb_pipeline()
    gb.fit(gb_matrix(tr, enc, fit=True), tr["answered"].values)
    imp = permutation_importance(gb, gb_matrix(te, enc, fit=False), te["answered"].values,
                                 n_repeats=20, random_state=SEED, scoring="roc_auc")
    feats = NUMERIC + CATEGORICAL
    order = np.argsort(imp.importances_mean)[::-1]
    print("===== FEATURE IMPORTANCE (GBT, permutation on AUC) =====")
    for i in order:
        print(f"  {feats[i]:18} {imp.importances_mean[i]:+.3f}")

    model.save(os.path.join(ART, "connect_rate_dnn.keras"))
    with open(os.path.join(ART, "metrics.json"), "w") as f:
        json.dump({
            "dnn_cv_roc_auc_mean": float(np.mean(dnn_auc)), "dnn_cv_roc_auc_std": float(np.std(dnn_auc)),
            "dnn_cv_avg_precision": float(np.mean(dnn_ap)),
            "gbt_cv_roc_auc_mean": float(np.mean(gb_auc)), "gbt_cv_roc_auc_std": float(np.std(gb_auc)),
            "gbt_cv_avg_precision": float(np.mean(gb_ap)),
            "positive_rate": float(y.mean()), "n": int(len(df)),
            "top_feature": feats[order[0]],
        }, f, indent=2)
    print(f"\n[saved] {ART}/connect_rate_dnn.keras")


if __name__ == "__main__":
    main()
