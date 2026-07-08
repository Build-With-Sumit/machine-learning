# docs — engineer-facing documentation

Written for **software engineers who don't do ML**. Two layers: a concepts doc,
and code-level walkthroughs.

## Read in this order

1. **[explainer.html](explainer.html)** — *concepts, no code.* What ML actually
   is, framed as engineering: a model is a function whose body you didn't write;
   training is an optimization loop; overfitting is hardcoding your test cases.
   Open it in a browser (self-contained, styled, with diagrams). Start here if
   *tensor*, *gradient*, or *backprop* aren't yet muscle memory.

2. **[connect-rate.md](connect-rate.md)** — *flagship, real data.* The tabular
   pattern on a live sales-call log: dial-time features, post-call leakage
   exclusion, a Keras DNN vs gradient-boosted-trees head-to-head, and permutation
   importance. The most detailed walkthrough.

3. **[forecasting.md](forecasting.md)** — *different family, sequences.* Sliding-
   window framing, leakage-safe scaling, chronological splits, an LSTM, and
   baseline-relative evaluation. Self-contained.

## Concept → where it's explained

| Concept | Doc |
|---|---|
| model / training / inference (the paradigm flip) | explainer §1 |
| tensor, weights, neuron, layer, forward pass | explainer §2 |
| loss, gradient, backprop, optimizer, epoch, batch | explainer §3 |
| **data leakage** (feature-time + split-time) | [connect-rate](connect-rate.md), [forecasting](forecasting.md) |
| in-graph preprocessing / train-serve skew | [connect-rate](connect-rate.md) |
| **Normalization** (z-score) / **one-hot** categoricals | [connect-rate](connect-rate.md) |
| **gradient-boosted trees** (DNN-vs-trees) | [connect-rate](connect-rate.md) |
| **permutation importance** (which feature matters) | [connect-rate](connect-rate.md) |
| **cross-validation** (why, on small n) | [connect-rate](connect-rate.md) |
| ROC-AUC, average precision, why not accuracy | [connect-rate](connect-rate.md), explainer §5 |
| **overfitting** + defenses (L2, dropout, early stopping) | explainer §5 |
| **LSTM** / memory cell / sequences | [forecasting](forecasting.md), explainer §4 |
| sliding-window framing, recursive forecast | [forecasting](forecasting.md) |
| **baseline-relative** evaluation (persistence) | [forecasting](forecasting.md) |
| TensorFlow vs Keras (engine vs API) | explainer §6 |

## Run the code alongside the docs

```bash
python -m venv .venv                                   # Python 3.11
.venv/Scripts/python -m pip install -r requirements.txt

.venv/Scripts/python 01_call_connect_rate/train.py
.venv/Scripts/python 02_timeseries_forecast/train.py
```

Each prints its per-fold metrics and writes a `.keras` model + `metrics.json` to
its `artifacts/` folder (gitignored — regenerated on every run). No private data
needed: POC 1 falls back to a synthetic sample, POC 2 embeds its dataset.
