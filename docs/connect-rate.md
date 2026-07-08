# POC 1 — Call connect-rate classifier — `01_call_connect_rate/train.py`

> **Audience:** an engineer who reads code but hasn't built ML. This is the
> flagship POC, on real operational data. See [explainer.html](explainer.html)
> first if terms like *gradient* or *cross-validation* aren't yet familiar.

## The task & why it's worth doing

The inside-sales team makes hundreds of calls a day, and **only ~41% connect.**
If we can predict *at dial time* whether a call will be answered, reps can dial at
better times and prioritize numbers more likely to pick up. So: binary
classification, label = `answered` (1/0), from features known **before** the call
is placed.

The label is **balanced (~41% positive)** — no imbalance tricks needed — and the
signal turns out to be real and interpretable.

## The data (and how privacy is handled)

The source is a live call log. It contains phone numbers and rep identities, so
the real data is **de-identified** before it ever leaves the source system — rep
identity becomes an opaque hash, the counterpart phone number becomes a
repeat-count and the raw number is dropped — and the resulting snapshot is
**private, never committed**. If no snapshot is present, [`data.py`](../01_call_connect_rate/data.py)
generates a **synthetic sample** with the same rep/hour connect-rate signal, so
this repo runs standalone with zero private data.

## Leakage is the whole game here

A call row also records `duration`, talk-time, answer-time, and a disposition
code. **Every one of those is known only *after* the call** — the disposition is
basically the label written in words. Training on them would give a near-perfect
score that collapses in production, because at dial time you don't have them yet.
So the loader keeps **only dial-time features**:

| kind | features | knowable before dialing? |
|---|---|---|
| numeric | `hour`, `dow`, `counterpart_repeat` | yes |
| categorical | `account`, `direction`, `rep`, `number_name` | yes |
| **excluded (leakage)** | duration, talk-time, answer-time, disposition | **no — post-call** |

This is the single most important modeling decision, and it's pure engineering
discipline: *a feature you won't have at prediction time cannot be a feature.*

## Two models, same folds — DNN vs trees

On small structured data, **gradient-boosted trees are the honest thing to
beat** — they're the workhorse for tabular problems. So this POC trains two
models on the identical CV folds:

1. **Tabular DNN** — `Normalization` on the numeric block, a `StringLookup`
   one-hot per categorical, concatenate → two `Dense` layers → sigmoid.
   Preprocessing lives *inside* the Keras graph, so the saved model consumes a
   raw feature dict (no separate scaler/encoder to keep in sync at serving time).
2. **Gradient-boosted trees** — scikit-learn `HistGradientBoostingClassifier`
   with the categoricals declared natively, `class_weight="balanced"`, shallow
   trees + L2 to avoid overfitting n≈559.

Both are scored with **stratified 5-fold cross-validation** (mean ± std) on
ROC-AUC and average precision.

## The result

```
[cv] DNN  ROC-AUC = 0.605 ± 0.061   AP = 0.548
[cv] GBT  ROC-AUC = 0.599 ± 0.042   AP = 0.531
[cv] baseline (base rate) = 0.408
```

Both land at **ROC-AUC ≈ 0.60** — meaningfully above the 0.5 coin flip — and
**AP ≈ 0.53–0.55 vs a 0.41 base rate** (a ~34% precision lift). The DNN edges the
trees slightly; they agree, which is reassuring on a small sample.

## What actually drives a connect — permutation importance

The best part is *why* it works. Permutation importance on the trees (shuffle one
feature, measure the AUC drop) ranks the dial-time features:

```
hour                +0.118     <- time-of-day matters most
rep                 +0.086     <- which rep dials
dow                 +0.069     <- day-of-week
counterpart_repeat  +0.028
direction           +0.001
number_name         +0.000     <- which line: no signal
account            -0.025      <- redundant with number_name
```

A clean, actionable business finding, not a black box: **when you call and who
calls drive connect rate; which line/account you use doesn't.** The model
recovers the same pattern the raw data shows (connect rate swings 26%→49% by rep
and up to 71% by hour) — which is exactly what a trustworthy model should do.

## Honest limitations

- **n ≈ 559, a few weeks of history.** Enough for a credible balanced classifier;
  the same data will support a call-volume *forecaster* (see [POC 2](forecasting.md))
  once months accumulate.
- Dial-time features are thin (no caller demographics), so ROC-AUC ~0.60 is
  likely near the ceiling for this feature set — honestly reported, not inflated.
- Retrain as the log grows; the pipeline is unchanged, only the snapshot.

---

**Next:** [forecasting.md](forecasting.md) — a different problem family: predicting
the next value in a time series with an LSTM.
