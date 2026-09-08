# SHAP 해석 — 모델이 어떤 신호를 쓰는가
#
# 확인할 것:
#   1. 심어둔 진짜 신호(W3 7일 빈도/추세, 사용량 표준편차)가 중요도 상위에 오는가
#   2. 함정 신호(W7 상시 높음 — 이벤트와 무관한 교란)를 모델이 얼마나 쓰는가
#      + W7 기여의 "방향" 확인: 교란군은 이벤트가 없으므로 W7 높음 → 안전 쪽으로
#        학습됐을 가능성이 있다 (인과가 아니라 코호트 구성의 산물)
#   3. W7 피처를 제거하고 재학습했을 때 성능이 얼마나 변하는가 (제거 가능성 검증)
#
# 대상 모델: PR-AUC 최고였던 no_handling (05의 산출물)
# 실행: .venv/bin/python notebooks/06_shap_analysis.py
# 출력: outputs/figures/fig5_shap_summary.png, fig6_shap_w7.png

from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score

matplotlib.rcParams["font.family"] = "AppleGothic"
matplotlib.rcParams["axes.unicode_minus"] = False

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

# ── SHAP 값 계산 (테스트 구간) ────────────────────────────────────
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_te)
if isinstance(shap_values, list):  # 구버전 호환: [음성, 양성] 리스트로 오는 경우
    shap_values = shap_values[1]

mean_abs = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_cols)
ranking = mean_abs.sort_values(ascending=False)
print("=== mean|SHAP| 상위 15 ===")
for i, (name, v) in enumerate(ranking.head(15).items(), 1):
    print(f"{i:2d}. {name:<16s} {v:.4f}")

def rank_of(prefix: str) -> list[str]:
    return [f"{i + 1}위 {n}" for i, n in enumerate(ranking.index) if n.startswith(prefix)]

print("\n진짜 신호 — W3:", ", ".join(rank_of("w3")))
print("진짜 신호 — 사용량:", ", ".join(rank_of("u1")))
print("함정 신호 — W7:", ", ".join(rank_of("w7")))

# ── fig5: summary plot ───────────────────────────────────────────
fig = plt.figure()
shap.summary_plot(shap_values, X_te, max_display=15, show=False)
plt.title("SHAP summary — 테스트 구간, no_handling 모델", fontsize=12)
plt.tight_layout()
plt.savefig(base / "outputs" / "figures" / "fig5_shap_summary.png", dpi=150, bbox_inches="tight")
plt.close("all")

# ── fig6: W7 기여의 방향 — 교란 신호가 어떻게 쓰이는지 ────────────
w7_feat = max((f for f in feature_cols if f.startswith("w7")), key=lambda f: mean_abs[f])
idx = feature_cols.index(w7_feat)
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(X_te[w7_feat], shap_values[:, idx], s=6, alpha=0.3, color="tab:orange")
ax.axhline(0, color="black", lw=0.8)
ax.set_xlabel(f"{w7_feat} (14일 W7 발생 건수)" if "14d" in w7_feat else w7_feat)
ax.set_ylabel("SHAP 값 (양수 = 위험 쪽 기여)")
ax.set_title(f"함정 신호의 사용 방식 — {w7_feat}의 SHAP 기여")
fig.tight_layout()
fig.savefig(base / "outputs" / "figures" / "fig6_shap_w7.png", dpi=150)

# ── W7 제거 재학습 (ablation) ────────────────────────────────────
no_w7 = [c for c in feature_cols if not c.startswith("w7")]
ablated = LGBMClassifier(
    n_estimators=400, learning_rate=0.05, num_leaves=31,
    min_child_samples=30, random_state=SEED, verbosity=-1, n_jobs=-1,
)
ablated.fit(train[no_w7], train["target"])
pr_full = average_precision_score(y_te, model.predict_proba(X_te)[:, 1])
pr_ablated = average_precision_score(y_te, ablated.predict_proba(test[no_w7])[:, 1])
print(f"\n=== W7 제거 ablation ===")
print(f"전체 피처 PR-AUC: {pr_full:.4f}")
print(f"W7 제거 PR-AUC:   {pr_ablated:.4f} (차이 {pr_ablated - pr_full:+.4f})")
print("saved: figures/fig5_shap_summary.png, fig6_shap_w7.png")
