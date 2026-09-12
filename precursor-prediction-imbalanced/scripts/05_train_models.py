"""LightGBM x 불균형 처리 3종 (없음 / class_weight / scale_pos_weight)

- 시간 분할. day 150부터 test, 경계 7일 purge (day 143~149는 target window가 test를 봄)
- 랜덤 분할 안 씀. 인접일 행은 window가 거의 겹치는 준중복이라 성능 부풀려짐
- 지표는 PR-AUC 중심. 양성 0.36%에서 ROC-AUC는 후함
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from common import setup_font
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

setup_font()

SEED = 42
TEST_START_DAY = 150
PURGE_GAP = 7  # target window 길이
base = Path(__file__).resolve().parents[1]

table = pd.read_csv(base / "data" / "features.csv", parse_dates=["event_date"])
feature_cols = [c for c in table.columns if c not in ("device_id", "event_date", "day", "target")]

train = table[table["day"] <= TEST_START_DAY - 1 - PURGE_GAP]
test = table[table["day"] >= TEST_START_DAY]
X_tr, y_tr = train[feature_cols], train["target"]
X_te, y_te = test[feature_cols], test["target"]

print(f"train: {len(train):,}행 (day 13~{train['day'].max()}), positive {y_tr.sum()} ({y_tr.mean():.4%})")
print(f"test : {len(test):,}행 (day {TEST_START_DAY}~{test['day'].max()}), positive {y_te.sum()} ({y_te.mean():.4%})")

lgbm_params = dict(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=30,
    random_state=SEED,
    verbosity=-1,
    n_jobs=-1,
)
n_neg, n_pos = (y_tr == 0).sum(), (y_tr == 1).sum()
variants = {
    "no_handling": LGBMClassifier(**lgbm_params),
    "class_weight": LGBMClassifier(**lgbm_params, class_weight="balanced"),
    "scale_pos_weight": LGBMClassifier(**lgbm_params, scale_pos_weight=n_neg / n_pos),
}

results = []
curves = {}
for name, model in variants.items():
    model.fit(X_tr, y_tr)
    score = model.predict_proba(X_te)[:, 1]
    results.append(
        {
            "variant": name,
            "pr_auc": average_precision_score(y_te, score),
            "roc_auc": roc_auc_score(y_te, score),
        }
    )
    curves[name] = precision_recall_curve(y_te, score)
    joblib.dump(model, base / "data" / f"model_{name}.joblib")

res = pd.DataFrame(results).set_index("variant")
print("\n=== 불균형 처리 비교 (테스트: 마지막 30일) ===")
print(res.round(4).to_string())
best = res["pr_auc"].idxmax()
print(f"\nPR-AUC 최고: {best}")

labels = {
    "no_handling": "아무것도 안 함",
    "class_weight": "class_weight='balanced'",
    "scale_pos_weight": f"scale_pos_weight={n_neg / n_pos:.0f}",
}
colors = {"no_handling": "#B4B2A9", "class_weight": "#5B6B7A", "scale_pos_weight": "#9E3D22"}

fig, ax = plt.subplots(figsize=(8, 5.5))
for name in variants:
    prec, rec, _ = curves[name]
    ax.plot(rec, prec, color=colors[name], lw=1.8, label=f"{labels[name]} (PR-AUC {res.loc[name, 'pr_auc']:.3f})")
ax.axhline(y_te.mean(), color="black", ls=":", lw=1, label=f"무작위 baseline (양성 비율 {y_te.mean():.3%})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_ylim(0, 1.02)
ax.set_title("PR curve, 불균형 처리 3종 (test 구간)")
ax.legend(loc="lower left", fontsize=9)
fig.tight_layout()
fig.savefig(base / "outputs" / "figures" / "fig4_pr_curves.png", dpi=150)
print("saved: figures/fig4_pr_curves.png, data/model_*.joblib")

res_dir = base / "outputs" / "results"
res_dir.mkdir(parents=True, exist_ok=True)
res.round(4).reset_index().to_csv(res_dir / "model_compare.csv", index=False)
pd.DataFrame(
    {
        "metric": [
            "train_rows",
            "train_day_max",
            "train_positive",
            "test_rows",
            "test_day_min",
            "test_positive",
            "scale_pos_weight",
        ],
        "value": [
            len(train),
            int(train["day"].max()),
            int(y_tr.sum()),
            len(test),
            TEST_START_DAY,
            int(y_te.sum()),
            round(n_neg / n_pos, 2),
        ],
    }
).to_csv(res_dir / "split_stats.csv", index=False)
