# LightGBM 이진 분류 — 불균형 처리 3가지 비교
#
# 분할: 시간 기준 분할. 마지막 30일(day 150~)을 테스트로 쓴다.
#   - 랜덤 분할 금지 이유:
#     (a) 같은 기기의 인접한 날짜 행들은 트레일링 윈도우가 대부분 겹치는 준중복 행이다.
#         랜덤 분할하면 사실상 같은 행이 train/test 양쪽에 들어가 성능이 부풀려진다.
#     (b) 실제 운용은 "과거로 학습해 미래를 예측"하는 문제다. 랜덤 분할은 미래 데이터로
#         학습해 과거를 맞추는 평가가 되어 배포 후 성능을 과대추정한다.
#     (c) 한 이벤트의 전조 구간(7일, 최대 7행)이 양쪽에 흩어지면 이벤트 단위 누수가 된다.
#   - 퍼지(purge) 갭: train은 day <= 142까지만 쓴다. day 143~149 행의 타깃 윈도우(t+1~t+7)는
#     테스트 구간(150~)을 들여다보므로 경계의 7일을 버려 라벨을 통한 누수를 차단한다.
#
# 불균형 처리 비교: (1) 아무것도 안 함 (2) class_weight='balanced' (3) scale_pos_weight
# 평가: PR-AUC 중심(양성 0.36%에서는 ROC-AUC가 후하게 나오므로), ROC-AUC 병기
#
# 실행: .venv/bin/python notebooks/05_train_models.py
# 출력: outputs/figures/fig4_pr_curves.png, data/model_*.joblib

from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

matplotlib.rcParams["font.family"] = "AppleGothic"
matplotlib.rcParams["axes.unicode_minus"] = False

SEED = 42
TEST_START_DAY = 150          # 마지막 30일
PURGE_GAP = 7                 # 타깃 윈도우 길이만큼 경계 제거
base = Path(__file__).resolve().parents[1]

table = pd.read_csv(base / "data" / "features.csv", parse_dates=["event_date"])
feature_cols = [c for c in table.columns if c not in ("device_id", "event_date", "day", "target")]

train = table[table["day"] <= TEST_START_DAY - 1 - PURGE_GAP]
test = table[table["day"] >= TEST_START_DAY]
X_tr, y_tr = train[feature_cols], train["target"]
X_te, y_te = test[feature_cols], test["target"]

print(f"train: {len(train):,}행 (day 13~{train['day'].max()}), "
      f"positive {y_tr.sum()} ({y_tr.mean():.4%})")
print(f"test : {len(test):,}행 (day {TEST_START_DAY}~{test['day'].max()}), "
      f"positive {y_te.sum()} ({y_te.mean():.4%})")

common = dict(
    n_estimators=400, learning_rate=0.05, num_leaves=31,
    min_child_samples=30, random_state=SEED, verbosity=-1, n_jobs=-1,
)
n_neg, n_pos = (y_tr == 0).sum(), (y_tr == 1).sum()
variants = {
    "no_handling": LGBMClassifier(**common),
    "class_weight": LGBMClassifier(**common, class_weight="balanced"),
    "scale_pos_weight": LGBMClassifier(**common, scale_pos_weight=n_neg / n_pos),
}

results = []
curves = {}
for name, model in variants.items():
    model.fit(X_tr, y_tr)
    score = model.predict_proba(X_te)[:, 1]
    results.append({
        "variant": name,
        "pr_auc": average_precision_score(y_te, score),
        "roc_auc": roc_auc_score(y_te, score),
    })
    curves[name] = precision_recall_curve(y_te, score)
    joblib.dump(model, base / "data" / f"model_{name}.joblib")

res = pd.DataFrame(results).set_index("variant")
print("\n=== 불균형 처리 비교 (테스트: 마지막 30일) ===")
print(res.round(4).to_string())
best = res["pr_auc"].idxmax()
print(f"\nPR-AUC 최고: {best}")

# ── PR 커브 플롯 ─────────────────────────────────────────────────
labels = {
    "no_handling": "아무것도 안 함",
    "class_weight": "class_weight='balanced'",
    "scale_pos_weight": f"scale_pos_weight={n_neg / n_pos:.0f}",
}
colors = {"no_handling": "tab:gray", "class_weight": "tab:blue", "scale_pos_weight": "tab:red"}

fig, ax = plt.subplots(figsize=(8, 5.5))
for name in variants:
    prec, rec, _ = curves[name]
    ax.plot(rec, prec, color=colors[name], lw=1.8,
            label=f"{labels[name]} (PR-AUC {res.loc[name, 'pr_auc']:.3f})")
ax.axhline(y_te.mean(), color="black", ls=":", lw=1,
           label=f"무작위 baseline (양성 비율 {y_te.mean():.3%})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_ylim(0, 1.02)
ax.set_title("Precision-Recall curve — 불균형 처리 3가지 비교 (테스트 구간)")
ax.legend(loc="lower left", fontsize=9)
fig.tight_layout()
fig.savefig(base / "outputs" / "figures" / "fig4_pr_curves.png", dpi=150)
print(f"saved: figures/fig4_pr_curves.png, data/model_*.joblib")
