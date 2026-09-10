"""05. 모델 비교, v1 대 v2.

원래 노트북의 파이프라인을 재현한다. 95/5 무작위 분할(random_state=786) 후 95%에 대해
10-fold stratified CV(session_id=123)를 돌려 Logistic Regression, Random Forest, XGBoost,
LightGBM을 비교한다. feature 테이블 세 가지에 대해 실행한다.

  v1_asis   snapshot feature, 노트북 feature 집합 (실제로 했던 것)
  v1_pref   snapshot feature + 선호 정보 (있었지만 쓰지 않은 것)
  v2_pref   시간 절단 feature + 선호 정보 (했어야 하는 것)

이후 95% 분할에 LightGBM을 적합해 5% holdout을 채점하고 gain importance를 저장한다.
"""
import sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RES, FIG, NOTEBOOK_FEATURES, PREF_FEATURES, TARGET, CAT_COLS

VARIANTS = {
    "v1_asis": ("features_v1_snapshot.csv", NOTEBOOK_FEATURES),
    "v1_pref": ("features_v1_snapshot.csv", NOTEBOOK_FEATURES + PREF_FEATURES),
    "v2_pref": ("features_v2_timecut.csv", NOTEBOOK_FEATURES + PREF_FEATURES),
}
MODELS = {
    "Logistic Regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=123),
    "XGBoost": lambda: XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                     tree_method="hist", random_state=123, n_jobs=-1, verbosity=0),
    "LightGBM": lambda: LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=31,
                                       random_state=123, n_jobs=-1, verbose=-1),
}
SCORING = {"Accuracy": "accuracy", "AUC": "roc_auc", "Recall": "recall", "Prec.": "precision", "F1": "f1"}


def encode(df, cols):
    X = df[cols].copy()
    cats = [c for c in cols if c in CAT_COLS]
    X = pd.get_dummies(X, columns=cats, drop_first=False, dtype=int)
    X.columns = [c.replace(" ", "_") for c in X.columns]
    return X


rows, holdout, importances = [], [], {}
for vname, (fname, cols) in VARIANTS.items():
    df = pd.read_csv(DATA / fname)
    train = df.sample(frac=0.95, random_state=786)
    test = df.drop(train.index)
    Xtr, ytr = encode(train, cols), train[TARGET].values
    Xte = encode(test, cols).reindex(columns=Xtr.columns, fill_value=0)
    yte = test[TARGET].values
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
    print(f"\n== {vname}: train {len(train):,} × {Xtr.shape[1]} cols, test {len(test):,}")
    for mname, make in MODELS.items():
        t0 = time.time()
        r = cross_validate(make(), Xtr, ytr, cv=cv, scoring=SCORING, n_jobs=1)
        rec = {"variant": vname, "model": mname, **{k: r[f"test_{k}"].mean() for k in SCORING},
               "TT (Sec)": r["fit_time"].mean()}
        rows.append(rec)
        print(f"  {mname:20s} " + " ".join(f"{k} {rec[k]:.4f}" for k in SCORING) + f"  ({time.time()-t0:.0f}s)")
    # 95% 전체에 LightGBM 적합 → holdout + importance
    lgbm = MODELS["LightGBM"]().fit(Xtr, ytr)
    p = lgbm.predict_proba(Xte)[:, 1]
    yhat = (p >= 0.5).astype(int)
    holdout.append({"variant": vname, "Accuracy": accuracy_score(yte, yhat), "AUC": roc_auc_score(yte, p),
                    "Recall": recall_score(yte, yhat), "Prec.": precision_score(yte, yhat),
                    "F1": f1_score(yte, yhat)})
    gain = pd.Series(lgbm.booster_.feature_importance("gain"), index=Xtr.columns)
    # one-hot 열을 원래 feature로 되돌려 합산
    src = gain.index.to_series().map(lambda c: next((k for k in cols if c == k or c.startswith(k + "_")), c))
    imp = gain.groupby(src.values).sum().sort_values(ascending=False)
    imp = imp / imp.sum()
    importances[vname] = imp
    imp.rename("importance").to_csv(RES / f"importance_{vname}.csv")

comp = pd.DataFrame(rows)
comp.to_csv(RES / "model_comparison.csv", index=False)
pd.DataFrame(holdout).to_csv(RES / "holdout_lightgbm.csv", index=False)
print("\n", comp.round(4).to_string(index=False))
print("\nhold-out (LightGBM):\n", pd.DataFrame(holdout).round(4).to_string(index=False))

# ---------------------------------------------------------------- 그림
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=False)
for ax, vname, title in zip(axes, ["v1_asis", "v2_pref"],
                            ["v1 — snapshot (as-was)", "v2 — time cut (+ preference fields)"]):
    top = importances[vname].head(10)[::-1]
    ax.hlines(top.index, 0, top.values, color="#2b6cb0", lw=1.5)
    ax.plot(top.values, top.index, "o", color="#2b6cb0")
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_xlabel("normalized gain importance (LightGBM)")
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, max(0.45, top.values.max() * 1.15))
fig.suptitle("Top-10 feature importance: what the model learned", fontsize=12, x=0.02, ha="left")
fig.tight_layout()
fig.savefig(FIG / "fig5_importance_v1_vs_v2.png", dpi=150)

fig, ax = plt.subplots(figsize=(8, 4))
piv = comp.pivot(index="model", columns="variant", values="AUC").loc[list(MODELS)]
piv.plot.bar(ax=ax, color=["#a0aec0", "#2b6cb0", "#c05621"], width=0.75)
ax.set_ylim(0.5, 1.0)
ax.set_ylabel("10-fold CV AUC")
ax.set_xlabel("")
ax.tick_params(axis="x", rotation=0)
ax.grid(axis="y", alpha=0.3)
ax.legend(title="", frameon=False)
for cont in ax.containers:
    ax.bar_label(cont, fmt="%.3f", fontsize=8, padding=2)
ax.set_title("AUC by feature construction — the snapshot inflates every model", loc="left", fontsize=11)
fig.tight_layout()
fig.savefig(FIG / "fig4_auc_by_variant.png", dpi=150)
print("figures saved")
