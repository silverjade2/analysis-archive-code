# 3호 글 — 동일 LightGBM으로 3버전 학습, 지표 4종 비교
#
# 확인할 것:
#   1. ROC-AUC는 세 버전에서 거의 같고 (조건부 분포 동일 → 설계상 그래야 함)
#      PR-AUC / F1 / precision@top10%는 크게 갈리는가 — "착시"의 정량 확인
#   2. 양성 3%에서 precision@top10%의 상한: 유병률 0.03이면 top 10%에 양성을
#      전부 몰아넣어도 정밀도는 0.3을 못 넘는다 (지표의 산술적 한계)
#   3. calibration: 예측 확률이 실제 빈도와 맞는가 (유병률별 차이)
#
# 실행: .venv/bin/python notebooks/02_metric_illusion.py
# 출력: outputs/figures/fig2_roc_curves.png    (ROC 3버전 겹치기)
#       outputs/figures/fig3_pr_curves.png     (PR 3버전 겹치기)
#       outputs/figures/fig4_calibration.png   (calibration 곡선)
#       outputs/figures/fig5_threshold_cm.png  (양성 3%: 임계값별 혼동행렬 변화)
#       outputs/figures/fig6_precision_at_k.png (precision@top-k% 커브)

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

matplotlib.rcParams["font.family"] = "AppleGothic"
matplotlib.rcParams["axes.unicode_minus"] = False

SEED = 42
VERSIONS = {"pos3": "양성 3%", "pos10": "양성 10%", "pos30": "양성 30%"}
COLORS = {"pos3": "tab:red", "pos10": "tab:orange", "pos30": "tab:blue"}
base = Path(__file__).resolve().parents[1]
fig_dir = base / "outputs" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)


def precision_at_top_frac(y_true: np.ndarray, prob: np.ndarray, frac: float) -> float:
    k = max(1, int(len(prob) * frac))
    top = np.argsort(prob)[::-1][:k]
    return y_true[top].mean()


results = {}
for name in VERSIONS:
    df = pd.read_csv(base / "data" / f"imbalance_{name}.csv")
    X = df.drop(columns="y").to_numpy()
    y = df["y"].to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y
    )
    model = LGBMClassifier(random_state=SEED, verbose=-1).fit(X_tr, y_tr)
    prob = model.predict_proba(X_te)[:, 1]
    results[name] = {
        "y": y_te,
        "prob": prob,
        "prevalence": y_te.mean(),
        "roc_auc": roc_auc_score(y_te, prob),
        "pr_auc": average_precision_score(y_te, prob),
        "f1": f1_score(y_te, (prob >= 0.5).astype(int)),
        "p_at_10": precision_at_top_frac(y_te, prob, 0.10),
    }

# ── 비교표 ────────────────────────────────────────────────────────
table = pd.DataFrame(
    [
        {"버전": VERSIONS[n], "유병률": f"{r['prevalence']:.1%}",
         "ROC-AUC": round(r["roc_auc"], 3), "PR-AUC": round(r["pr_auc"], 3),
         "F1@0.5": round(r["f1"], 3), "precision@top10%": round(r["p_at_10"], 3)}
        for n, r in results.items()
    ]
)
print("── 동일 모델·동일 조건부 분포, 양성 비율만 다른 3버전 ──")
print(table.to_string(index=False))
roc_spread = max(r["roc_auc"] for r in results.values()) - min(r["roc_auc"] for r in results.values())
pr_spread = max(r["pr_auc"] for r in results.values()) - min(r["pr_auc"] for r in results.values())
print(f"\nROC-AUC 격차: {roc_spread:.3f}  /  PR-AUC 격차: {pr_spread:.3f}")
print(f"양성 3%의 precision@top10% 산술 상한: {results['pos3']['prevalence'] / 0.10:.2f}")

# ── 플롯 2: ROC 커브 겹치기 ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 5))
for name, r in results.items():
    fpr, tpr, _ = roc_curve(r["y"], r["prob"])
    ax.plot(fpr, tpr, color=COLORS[name], lw=2,
            label=f"{VERSIONS[name]} (AUC={r['roc_auc']:.3f})")
ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1, label="무작위")
ax.set_xlabel("FPR")
ax.set_ylabel("TPR")
ax.set_title("ROC 커브 — 세 버전이 거의 겹친다")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(fig_dir / "fig2_roc_curves.png", dpi=150)

# ── 플롯 3: PR 커브 겹치기 ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 5))
for name, r in results.items():
    prec, rec, _ = precision_recall_curve(r["y"], r["prob"])
    ax.plot(rec, prec, color=COLORS[name], lw=2,
            label=f"{VERSIONS[name]} (PR-AUC={r['pr_auc']:.3f})")
    ax.axhline(r["prevalence"], color=COLORS[name], ls=":", lw=1, alpha=0.6)
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("PR 커브 — 같은 모델인데 유병률 따라 갈린다 (점선: 무작위 기준선)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(fig_dir / "fig3_pr_curves.png", dpi=150)

# ── 플롯 4: calibration 곡선 ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 5))
for name, r in results.items():
    frac_pos, mean_pred = calibration_curve(r["y"], r["prob"], n_bins=10, strategy="quantile")
    ax.plot(mean_pred, frac_pos, marker="o", ms=4, color=COLORS[name], lw=1.8,
            label=VERSIONS[name])
ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1, label="완벽 보정")
ax.set_xlabel("예측 확률 (구간 평균)")
ax.set_ylabel("실제 양성 비율")
ax.set_title("Calibration 곡선 (quantile 구간 10개)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(fig_dir / "fig4_calibration.png", dpi=150)

# ── 플롯 5: 양성 3% — 임계값별 혼동행렬 변화 ─────────────────────
r3 = results["pos3"]
thresholds = np.linspace(0.01, 0.99, 99)
counts = []
for t in thresholds:
    pred = r3["prob"] >= t
    tp = int((pred & (r3["y"] == 1)).sum())
    fp = int((pred & (r3["y"] == 0)).sum())
    fn = int((~pred & (r3["y"] == 1)).sum())
    counts.append((t, tp, fp, fn))
cm = pd.DataFrame(counts, columns=["t", "TP", "FP", "FN"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
ax1.plot(cm["t"], cm["TP"], color="tab:green", lw=2, label="TP (잡은 양성)")
ax1.plot(cm["t"], cm["FP"], color="tab:red", lw=2, label="FP (오경보)")
ax1.plot(cm["t"], cm["FN"], color="tab:gray", lw=2, label="FN (놓친 양성)")
ax1.set_xlabel("임계값")
ax1.set_ylabel("건수")
ax1.set_title(f"양성 3% — 임계값별 혼동행렬 성분 (테스트 {len(r3['y']):,}건)")
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

prec_line = cm["TP"] / (cm["TP"] + cm["FP"]).clip(lower=1)
rec_line = cm["TP"] / (cm["TP"] + cm["FN"]).clip(lower=1)
ax2.plot(cm["t"], prec_line, color="tab:blue", lw=2, label="Precision")
ax2.plot(cm["t"], rec_line, color="tab:orange", lw=2, label="Recall")
ax2.set_xlabel("임계값")
ax2.set_ylabel("값")
ax2.set_title("양성 3% — 임계값별 Precision / Recall")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(fig_dir / "fig5_threshold_cm.png", dpi=150)

# ── 플롯 6: precision@top-k% 커브 ────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 4.5))
fracs = np.arange(0.01, 0.51, 0.01)
for name, r in results.items():
    pk = [precision_at_top_frac(r["y"], r["prob"], f) for f in fracs]
    ax.plot(fracs * 100, pk, color=COLORS[name], lw=2, label=VERSIONS[name])
ax.axvline(10, color="gray", ls="--", lw=1, alpha=0.7)
ax.text(10.5, 0.02, "top 10%", fontsize=8, color="gray")
ax.set_xlabel("상위 k% (예측 확률 순)")
ax.set_ylabel("precision@top-k%")
ax.set_title("상위 k%를 조치한다면 그중 진짜 양성은 몇 %인가")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(fig_dir / "fig6_precision_at_k.png", dpi=150)

print(f"\n플롯 5장 저장 — {fig_dir}/fig2~6")
