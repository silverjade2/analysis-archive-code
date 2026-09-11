"""cutoff sweep. 글의 인터랙티브 위젯용 데이터.

모든 유저의 v2 cutoff를 일 단위 offset만큼 옮긴다. 음의 offset은 동의 직전보다 더 이른 시점에서 유저를 보는
것이고, 양의 offset은 타깃 결정 이후의 이벤트가 feature로 새어 들어가게 하는 것이다. "snapshot"은 원래
노트북의 테이블, 즉 전원을 조회 시점에서 자른 v1에 선호 정보를 더한 것이다. offset마다 06과 같은 모델과 seed로
LightGBM 5-fold out-of-fold AUC를 구하고, 전체 행으로 적합한 모델의 상위 5개 gain importance를 05처럼 one-hot
열을 원래 feature로 합산해 기록한다. cutoff는 04와 같은 하한이라 프로필 완성 전날보다 앞서지 않고, snapshot
당일을 넘지 않는다.
"""

import json

import numpy as np
import pandas as pd
from common import CAT_COLS, DATA, N_DAYS, NOTEBOOK_FEATURES, PREF_FEATURES, RES, TARGET
from features import build_features, load
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

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
        cutoff = np.clip(base + off, profile_day - 1, N_DAYS - 1)
    f = build_features(cutoff, users, *rest)
    X = encode(f, COLS)
    oof = cross_val_predict(make_model(), X, y, cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, oof)
    seen = (cutoff[pos] >= consent_day[pos]).mean()  # 관측 창에 동의일이 포함된 양성 비율
    m = make_model().fit(X, y)
    gain = pd.Series(m.booster_.feature_importance("gain"), index=X.columns)
    src = gain.index.to_series().map(lambda c: next((k for k in COLS if c == k or c.startswith(k + "_")), c))
    imp = gain.groupby(src.values).sum().sort_values(ascending=False)
    imp = imp / imp.sum()
    top = imp.head(TOP_N)
    label = "snapshot" if off == "snapshot" else f"{off:+d}d"
    print(
        f"{label:>9}  AUC {auc:.4f}  seen-after-consent {seen:.2f}  top: "
        + ", ".join(f"{k} {v:.2f}" for k, v in top.items())
    )
    for rank, (k, v) in enumerate(top.items(), 1):
        rows.append(
            {
                "offset": label,
                "auc": round(auc, 4),
                "seen_after_consent": round(seen, 4),
                "rank": rank,
                "feature": k,
                "importance": round(v, 4),
                "login_importance": round(imp.get("login_counts", 0.0), 4),
            }
        )
    points.append(
        {
            "offset": None if off == "snapshot" else off,
            "label": label,
            "auc": round(auc, 4),
            "seenAfterConsent": round(float(seen), 4),
            "loginImportance": round(float(imp.get("login_counts", 0.0)), 4),
            "topFeatures": [{"feature": k, "importance": round(float(v), 4)} for k, v in top.items()],
        }
    )

pd.DataFrame(rows).to_csv(RES / "cutoff_sweep.csv", index=False)
out = {
    "meta": {
        "oracleAuc": round(oracle, 4),
        "users": int(len(users)),
        "positives": int(pos.sum()),
        "folds": 5,
        "model": "LightGBM (n_estimators=200, num_leaves=31)",
    },
    "points": points,
}
(RES / "cutoff_sweep.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(f"\noracle AUC {oracle:.4f}; saved {len(points)} offsets")
