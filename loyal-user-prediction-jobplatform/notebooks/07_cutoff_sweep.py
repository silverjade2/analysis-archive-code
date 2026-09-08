"""07 — Cutoff sweep (data for the interactive widget in the article).

Takes the v2 cutoff of every user (features_v2_timecut.csv: cutoff_day) and shifts it by an
offset in days. Negative offsets look at the user EARLIER than the moment before consent;
positive offsets let events AFTER the target decision leak into the features; "snapshot"
is the original notebook's table (everyone cut at the query date, i.e. v1 + preference fields).

For every offset: LightGBM 5-fold out-of-fold AUC (same model / seed as 06) and the top-5
normalized gain importances (fit on all rows, one-hot columns folded back to their source
feature as in 05). Cutoffs never go before the profile-completion day (the modeling population
is defined by having a profile) and never past the snapshot day.

Outputs:
  outputs/results/cutoff_sweep.csv   long format: offset, auc, seen_after_consent, top features
  outputs/results/cutoff_sweep.json  the same, in the shape the widget reads
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RES, N_DAYS, NOTEBOOK_FEATURES, PREF_FEATURES, TARGET, CAT_COLS
from features import build_features, load

OFFSETS = [-30, -14, -7, -3, -1, 0, 1, 2, 3, 5, 7, 14, 30, 60, 90, 180, 365, "snapshot"]
COLS = NOTEBOOK_FEATURES + PREF_FEATURES
TOP_N = 5


def encode(df, cols):
    cats = [c for c in cols if c in CAT_COLS]
    X = pd.get_dummies(df[cols], columns=cats, dtype=int)
    X.columns = [c.replace(" ", "_") for c in X.columns]
    return X


def make_model():
    return LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=31, random_state=123, n_jobs=-1, verbose=-1)


users, *rest = load()
base = pd.read_csv(DATA / "features_v2_timecut.csv")["cutoff_day"].values
y = users[TARGET].values
pos = y == 1
consent_day = users["consent_day"].values
profile_day = users["profile_day"].values
oracle = roc_auc_score(y, users["truth_p_consent"].values)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

rows, points = [], []
for off in OFFSETS:
    if off == "snapshot":
        cutoff = np.full(len(users), N_DAYS - 1)
    else:
        cutoff = np.clip(base + off, profile_day, N_DAYS - 1)
    f = build_features(cutoff, users, *rest)
    X = encode(f, COLS)
    oof = cross_val_predict(make_model(), X, y, cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, oof)
    seen = (cutoff[pos] >= consent_day[pos]).mean()      # positives whose window includes the consent day
    m = make_model().fit(X, y)
    gain = pd.Series(m.booster_.feature_importance("gain"), index=X.columns)
    src = gain.index.to_series().map(lambda c: next((k for k in COLS if c == k or c.startswith(k + "_")), c))
    imp = gain.groupby(src.values).sum().sort_values(ascending=False)
    imp = imp / imp.sum()
    top = imp.head(TOP_N)
    label = "snapshot" if off == "snapshot" else f"{off:+d}d"
    print(f"{label:>9}  AUC {auc:.4f}  seen-after-consent {seen:.2f}  top: "
          + ", ".join(f"{k} {v:.2f}" for k, v in top.items()))
    for rank, (k, v) in enumerate(top.items(), 1):
        rows.append({"offset": label, "auc": round(auc, 4), "seen_after_consent": round(seen, 4),
                     "rank": rank, "feature": k, "importance": round(v, 4),
                     "login_importance": round(imp.get("login_counts", 0.0), 4)})
    points.append({"offset": None if off == "snapshot" else off, "label": label, "auc": round(auc, 4),
                   "seenAfterConsent": round(float(seen), 4),
                   "loginImportance": round(float(imp.get("login_counts", 0.0)), 4),
                   "topFeatures": [{"feature": k, "importance": round(float(v), 4)} for k, v in top.items()]})

pd.DataFrame(rows).to_csv(RES / "cutoff_sweep.csv", index=False)
out = {"meta": {"oracleAuc": round(oracle, 4), "users": int(len(users)), "positives": int(pos.sum()),
                "folds": 5, "model": "LightGBM (n_estimators=200, num_leaves=31)"},
       "points": points}
(RES / "cutoff_sweep.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(f"\noracle AUC {oracle:.4f}; saved {len(points)} offsets")
