"""06. 넛지 리스트 비교.

모델의 운영 용도는 주간 리스트였다. 아직 동의하지 않은 유저를 채점해 상위 10%를 마케팅에
넘긴다. 이 스크립트는 그 리스트를 v1(snapshot) 점수와 v2(시간 절단) 점수로 각각 만들고
(LightGBM out-of-fold 확률 사용), 각 리스트에 누가 오르는지 본다.

가상 데이터라 각 유저의 진짜 동의 성향(truth_p_consent)도 알고 있으므로 두 리스트를 정답
기준으로 채점할 수 있다. 좋은 넛지 리스트는 단순히 활동적인 유저가 아니라 전환에 가까운
유저로 채워져야 한다.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from lightgbm import LGBMClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RES, FIG, NOTEBOOK_FEATURES, PREF_FEATURES, TARGET, CAT_COLS

TOP = 0.10


def encode(df, cols):
    cats = [c for c in cols if c in CAT_COLS]
    X = pd.get_dummies(df[cols], columns=cats, dtype=int)
    X.columns = [c.replace(" ", "_") for c in X.columns]
    return X


users = pd.read_csv(DATA / "users.csv")
v1 = pd.read_csv(DATA / "features_v1_snapshot.csv")
v2 = pd.read_csv(DATA / "features_v2_timecut.csv")
y = v1[TARGET].values
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
scores = {}
for name, df, cols in [("v1", v1, NOTEBOOK_FEATURES), ("v2", v2, NOTEBOOK_FEATURES + PREF_FEATURES)]:
    X = encode(df, cols)
    m = LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=31, random_state=123, n_jobs=-1, verbose=-1)
    scores[name] = cross_val_predict(m, X, y, cv=cv, method="predict_proba")[:, 1]
    print(f"{name}: out-of-fold scores done")

from sklearn.metrics import roc_auc_score
oracle = roc_auc_score(y, users["truth_p_consent"].values)
print(f"oracle AUC (true propensity): {oracle:.4f}   v1 OOF AUC {roc_auc_score(y, scores['v1']):.4f}   v2 OOF AUC {roc_auc_score(y, scores['v2']):.4f}")
pd.DataFrame({"model": ["oracle (true propensity)", "v1 snapshot", "v2 time cut"],
              "AUC": [oracle, roc_auc_score(y, scores["v1"]), roc_auc_score(y, scores["v2"])]}).round(4).to_csv(RES / "oof_auc_vs_oracle.csv", index=False)

neg = np.flatnonzero(y == 0)
k = int(len(neg) * TOP)
lists = {name: neg[np.argsort(-s[neg])[:k]] for name, s in scores.items()}
overlap = len(np.intersect1d(lists["v1"], lists["v2"])) / k


def describe(idx, label):
    u = users.iloc[idx]
    f = v1.iloc[idx]  # 현재 시점의 snapshot 사실
    return {
        "list": label,
        "n": len(idx),
        "logged in ≤ 30d (snapshot)": (f["days_since_last_login"] <= 30).mean(),
        "season joiner": u["truth_season_joiner"].mean(),
        "preference complete": u["truth_pref_complete"].mean(),
        "salary left at default": (u["pref_salary_default_yn"] == "Y").mean(),
        "mean true consent propensity": u["truth_p_consent"].mean(),
    }


rows = [describe(lists["v1"], "v1 top 10% (snapshot score)"),
        describe(lists["v2"], "v2 top 10% (time-cut score)"),
        describe(neg, "all non-consented users")]
out = pd.DataFrame(rows).round(4)
out.to_csv(RES / "nudge_list_comparison.csv", index=False)
print(f"\noverlap between the two top-10% lists: {overlap:.1%}\n")
print(out.to_string(index=False))

# 정답 지표의 무작위 리스트 기준선
rng = np.random.default_rng(0)
rand_p = users.iloc[rng.choice(neg, k, replace=False)]["truth_p_consent"].mean()
pd.DataFrame({"metric": ["overlap between v1 and v2 top-10% lists",
                         "random 10% list: mean true consent propensity"],
              "value": [overlap, rand_p]}).round(4).to_csv(RES / "nudge_list_overlap.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
metrics = ["logged in ≤ 30d (snapshot)", "preference complete", "salary left at default", "season joiner"]
x = np.arange(len(metrics))
w = 0.26
for i, (lab, col) in enumerate([("v1 list", "#a0aec0"), ("v2 list", "#2b6cb0"), ("all non-consented", "#e2e8f0")]):
    axes[0].bar(x + (i - 1) * w, out.iloc[i][metrics].values.astype(float), w, label=lab, color=col)
axes[0].set_xticks(x, ["logged in\n≤ 30 days", "preference\ncomplete", "salary left\nat default", "season\njoiner"], fontsize=9)
axes[0].set_ylim(0, 1)
axes[0].legend(frameon=False, fontsize=9)
axes[0].set_title("Who is on the list", loc="left", fontsize=11)
axes[0].grid(axis="y", alpha=0.3)
vals = [rand_p, out.iloc[0]["mean true consent propensity"], out.iloc[1]["mean true consent propensity"]]
axes[1].bar(["random 10%", "v1 list", "v2 list"], vals, color=["#e2e8f0", "#a0aec0", "#2b6cb0"], width=0.6)
for i, v in enumerate(vals):
    axes[1].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
axes[1].set_ylim(0, 1)
axes[1].set_ylabel("mean true consent propensity")
axes[1].set_title("Scored against the planted truth", loc="left", fontsize=11)
axes[1].grid(axis="y", alpha=0.3)
fig.suptitle(f"Nudge lists: top 10% of non-consented users — overlap {overlap:.0%}", x=0.02, ha="left", fontsize=12)
fig.tight_layout()
fig.savefig(FIG / "fig6_nudge_lists.png", dpi=150)
print("figure saved")
