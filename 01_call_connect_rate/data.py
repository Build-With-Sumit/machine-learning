"""Data loader for POC 1 — sales-call connect-rate classification.

Target: `answered` (1 if the call connected, else 0) — a balanced ~41% label.
Features are everything knowable AT DIAL TIME (no post-call leakage):

    numeric      hour, dow, counterpart_repeat
    categorical  account, direction, rep (hashed), number_name

Reads a local de-identified snapshot CSV (`data/aircall_features.csv`) if one is
present; otherwise generates a synthetic sample with the same rep/hour
connect-rate signal, so this folder runs standalone with no private data. (The
real snapshot is produced internally from the call log and is not part of this
public repo.)
"""
from __future__ import annotations
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT = os.path.join(HERE, "data", "aircall_features.csv")

NUMERIC = ["hour", "dow", "counterpart_repeat"]
CATEGORICAL = ["account", "direction", "rep", "number_name"]


def _synthetic(n: int = 560, seed: int = 7) -> pd.DataFrame:
    """Stand-in with the real dataset's signal: connect rate varies by rep and
    by hour-of-day, most calls outbound. Lets the folder run without prod data."""
    import random
    rng = random.Random(seed)
    # per-rep base connect propensity (mirrors the real 0.26–0.49 spread)
    reps = {f"r{ i :08x}": p for i, p in
            zip(range(7), [0.26, 0.35, 0.45, 0.46, 0.48, 0.49, 0.33])}
    rep_ids = list(reps)
    # hour-of-day multiplier (afternoon connects better)
    def hour_mult(h):
        return 1.25 if 12 <= h <= 16 else (0.6 if (h < 8 or h > 19) else 1.0)
    rows = []
    for _ in range(n):
        rep = rng.choices(rep_ids, weights=[17, 12, 7, 6, 7, 5, 3])[0]
        account = rng.choices(["empmonitor", "globussoft"], weights=[8, 2])[0]
        number_name = "EMPMonitor" if account == "empmonitor" else "PAS & ADSGPT - USA"
        direction = rng.choices(["outbound", "inbound"], weights=[20, 1])[0]
        hour = rng.randint(0, 23)
        dow = rng.randint(0, 6)
        rep_cnt = rng.choices([1, 2, 3, 4], weights=[6, 3, 2, 1])[0]
        p = min(0.95, reps[rep] * hour_mult(hour) * (1.05 if direction == "inbound" else 1.0))
        answered = 1 if rng.random() < p else 0
        rows.append(dict(answered=answered, account=account, direction=direction,
                         rep=rep, number_name=number_name, hour=hour, dow=dow,
                         counterpart_repeat=rep_cnt))
    return pd.DataFrame(rows)


def load_calls() -> pd.DataFrame:
    """One row per call with dial-time features + the `answered` label."""
    if os.path.exists(SNAPSHOT):
        df = pd.read_csv(SNAPSHOT)
    else:
        print(f"[data] no snapshot at {SNAPSHOT} -> using synthetic sample")
        df = _synthetic()

    # coerce + fill: numerics numeric, categoricals string, no NaNs into the graph
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    for c in CATEGORICAL:
        df[c] = df[c].fillna("").astype(str)
    df["answered"] = pd.to_numeric(df["answered"], errors="coerce").fillna(0).astype(int)
    return df.reset_index(drop=True)


if __name__ == "__main__":
    d = load_calls()
    print(f"rows={len(d)}  answered={int(d.answered.sum())} "
          f"({d.answered.mean():.1%})")
    print(d.groupby("rep")["answered"].agg(["mean", "count"]).to_string())
