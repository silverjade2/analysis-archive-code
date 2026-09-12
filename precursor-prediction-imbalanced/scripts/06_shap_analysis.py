"""SHAP. 대상은 05의 no_handling (PR-AUC 최고)

- 진짜 신호(W3, u1_std)가 위에 오는지, 함정 W7을 얼마나 쓰는지
- W7은 기여 방향도. 교란군에 이벤트가 없어서 "W7 높음 = 안전"으로 배웠을 수 있음
- W7 제거 ablation
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from common import setup_font
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score

setup_font()

SEED = 42
TEST_START_DAY = 150
PURGE_GAP = 7
base = Path(__file__).resolve().parents[1]

table = pd.read_csv(base / "data" / "features.csv", parse_dates=["event_date"])
feature_cols = [c for c in table.columns if c not in ("device_id", "event_date", "day", "target")]
train = table[table["day"] <= TEST_START_DAY - 1 - PURGE_GAP]
test = table[table["day"] >= TEST_START_DAY]
X_te, y_te = test[feature_cols], test["target"]

model = joblib.load(base / "data" / "model_no_handling.joblib")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_te)
if isinstance(shap_values, list):  # 구버전 shap은 [neg, pos] 리스트
    shap_values = shap_values[1]

mean_abs = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_cols)
ranking = mean_abs.sort_values(ascending=False)
print("=== mean|SHAP| 상위 15 ===")
for i, (name, v) in enumerate(ranking.head(15).items(), 1):
    print(f"{i:2d}. {name:<16s} {v:.4f}")


def rank_of(prefix: str) -> list[str]:
    return [f"{i + 1}위 {n}" for i, n in enumerate(ranking.index) if n.startswith(prefix)]


print("\n진짜 신호 W3:", ", ".join(rank_of("w3")))
print("진짜 신호 사용량:", ", ".join(rank_of("u1")))
print("함정 신호 W7:", ", ".join(rank_of("w7")))

fig = plt.figure()
shap.summary_plot(shap_values, X_te, max_display=15, show=False)
plt.title("SHAP summary (test 구간, no_handling)", fontsize=12)
plt.tight_layout()
plt.savefig(base / "outputs" / "figures" / "fig5_shap_summary.png", dpi=150, bbox_inches="tight")
plt.close("all")

w7_feat = max((f for f in feature_cols if f.startswith("w7")), key=lambda f: mean_abs[f])
idx = feature_cols.index(w7_feat)
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(X_te[w7_feat], shap_values[:, idx], s=6, alpha=0.3, color="#9E3D22")
ax.axhline(0, color="black", lw=0.8)
ax.set_xlabel(f"{w7_feat} (14일 W7 발생 건수)" if "14d" in w7_feat else w7_feat)
ax.set_ylabel("SHAP 값 (양수 = 위험 쪽 기여)")
ax.set_title(f"함정 신호 {w7_feat}의 SHAP 기여")
fig.tight_layout()
fig.savefig(base / "outputs" / "figures" / "fig6_shap_w7.png", dpi=150)

# 05 파라미터 복사. 05 바꾸면 여기도 손으로
no_w7 = [c for c in feature_cols if not c.startswith("w7")]
ablated = LGBMClassifier(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=30,
    random_state=SEED,
    verbosity=-1,
    n_jobs=-1,
)
ablated.fit(train[no_w7], train["target"])
pr_full = average_precision_score(y_te, model.predict_proba(X_te)[:, 1])
pr_ablated = average_precision_score(y_te, ablated.predict_proba(test[no_w7])[:, 1])
print("\n=== W7 제거 ablation ===")
print(f"전체 피처 PR-AUC: {pr_full:.4f}")
print(f"W7 제거 PR-AUC:   {pr_ablated:.4f} (차이 {pr_ablated - pr_full:+.4f})")
print("saved: figures/fig5_shap_summary.png, fig6_shap_w7.png")

res_dir = base / "outputs" / "results"
res_dir.mkdir(parents=True, exist_ok=True)
ranking.rename("mean_abs_shap").rename_axis("feature").reset_index().assign(rank=lambda d: d.index + 1).round(4).to_csv(
    res_dir / "shap_ranking.csv", index=False
)
pd.DataFrame(
    {"metric": ["pr_auc_full", "pr_auc_no_w7", "diff"], "value": [pr_full, pr_ablated, pr_ablated - pr_full]}
).round(4).to_csv(res_dir / "ablation_w7.csv", index=False)
