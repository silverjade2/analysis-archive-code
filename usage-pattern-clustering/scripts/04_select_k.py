# k 선택: 로그+표준화 데이터로 k=2~10 스윕
#
# 확인할 것:
#   1. 엘보우(inertia)와 실루엣 점수가 명확한 k를 가리키는가
#      — 함정 2(night ↔ intermittent 경계 흐림) 때문에 애매하게 나올 것으로 예상
#   2. k=4 vs k=5: 03에서 확인한 트레이드오프(로그 변환이 노이즈 15대의 극단성을
#      눌러 allday_low에 흡수됨)가 k=5에서 해소되는가 — 군집 예산이 하나 늘면
#      노이즈가 독립 군집으로 복원되는지
#   3. 각 k에서 군집별 크기·정답 레이블 구성 — "지표가 고르는 k"와
#      "해석 가능성이 고르는 k"가 갈리는 지점을 기록
#
# 실행: .venv/bin/python scripts/04_select_k.py
# 출력: outputs/figures/fig5_k_sweep.png (엘보우 + 실루엣)
#       outputs/figures/fig6_k4_vs_k5.png (k=4/k=5 군집 구성 대조)

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

matplotlib.rcParams["font.family"] = "AppleGothic"
matplotlib.rcParams["axes.unicode_minus"] = False

SEED = 42
K_RANGE = range(2, 11)
base = Path(__file__).resolve().parents[1]
fig_dir = base / "outputs" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(base / "data" / "usage_profiles.csv")
hour_cols = [c for c in df.columns if c.startswith("u_")]
feature_cols = hour_cols + ["total_usage"]

X = StandardScaler().fit_transform(np.log1p(df[feature_cols].to_numpy()))
true = df["true_cluster"]

# ── k 스윕 ────────────────────────────────────────────────────────
records = []
preds = {}
for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    pred = km.fit_predict(X)
    preds[k] = pred
    records.append({
        "k": k,
        "inertia": km.inertia_,
        "silhouette": silhouette_score(X, pred),
        "ari": adjusted_rand_score(true, pred),
    })

sweep = pd.DataFrame(records)
print("── k 스윕 (로그+표준화, KMeans n_init=10) ──")
print(sweep.round(3).to_string(index=False))
print()

# ── 각 k에서 군집별 크기와 정답 레이블 구성 요약 ──────────────────
# 군집마다 최다 정답 레이블과 그 비율(순도)을 한 줄로 요약한다.
print("── 각 k의 군집 구성 (크기 / 최다 정답 레이블 / 순도) ──")
for k in K_RANGE:
    pred = preds[k]
    parts = []
    for c in range(k):
        mask = pred == c
        top = true[mask].value_counts()
        parts.append(f"c{c}:{mask.sum()}({top.index[0]} {top.iloc[0]/mask.sum():.0%})")
    print(f"k={k}: " + "  ".join(parts))
print()

# ── k=4 vs k=5: 노이즈 15대 추적 ──────────────────────────────────
print("── k=4 vs k=5 — 노이즈 소군집 15대 추적 ──")
for k in (4, 5):
    pred = preds[k]
    crosstab = pd.crosstab(true, pred, margins=True)
    print(f"[k={k}] ARI={sweep.loc[sweep['k'] == k, 'ari'].iloc[0]:.3f}")
    print(crosstab.to_string())
    noise_pred = pd.Series(pred)[true == "noise"]
    top_cluster = noise_pred.value_counts().idxmax()
    recall = (noise_pred == top_cluster).sum() / len(noise_pred)
    purity = (true[pred == top_cluster] == "noise").mean()
    print(f"노이즈 15대 → 예측 군집 {top_cluster} (재현율 {recall:.0%}, 순도 {purity:.0%})\n")

# ── 플롯 1: 엘보우 + 실루엣 ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].plot(sweep["k"], sweep["inertia"], marker="o", color="tab:blue")
axes[0].set_xlabel("k")
axes[0].set_ylabel("inertia (군집 내 제곱합)")
axes[0].set_title("elbow plot")
axes[0].grid(alpha=0.3)

axes[1].plot(sweep["k"], sweep["silhouette"], marker="o", color="tab:orange")
best_k = sweep.loc[sweep["silhouette"].idxmax(), "k"]
axes[1].axvline(best_k, color="tab:red", ls="--", alpha=0.6,
                label=f"silhouette 최대 k={best_k}")
axes[1].set_xlabel("k")
axes[1].set_ylabel("silhouette 점수")
axes[1].set_title("silhouette 점수")
axes[1].legend()
axes[1].grid(alpha=0.3)

for ax in axes:
    ax.set_xticks(list(K_RANGE))
fig.suptitle("k 선택 sweep — 로그+표준화 K-means (k=2~10)")
fig.tight_layout()
fig.savefig(fig_dir / "fig5_k_sweep.png", dpi=150)
print(f"플롯 저장 — {fig_dir / 'fig5_k_sweep.png'}")

# ── 플롯 2: k=4 vs k=5 군집 구성 대조 (정답 레이블 누적 막대) ─────
LABEL_ORDER = ["morning", "allday_low", "night", "intermittent", "noise"]
LABEL_COLORS = {
    "morning": "tab:blue",
    "allday_low": "tab:green",
    "night": "tab:purple",
    "intermittent": "tab:orange",
    "noise": "tab:red",
}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for ax, k in zip(axes, (4, 5)):
    pred = preds[k]
    comp = pd.crosstab(pred, true).reindex(columns=LABEL_ORDER, fill_value=0)
    bottom = np.zeros(len(comp))
    for label in LABEL_ORDER:
        ax.bar(comp.index, comp[label], bottom=bottom,
               color=LABEL_COLORS[label], label=label)
        bottom += comp[label].to_numpy()
    ari = sweep.loc[sweep["k"] == k, "ari"].iloc[0]
    ax.set_title(f"k={k} (ARI={ari:.3f})")
    ax.set_xlabel("예측 군집")
    ax.set_xticks(comp.index)
    ax.grid(alpha=0.3, axis="y")
axes[0].set_ylabel("기기 수")
axes[1].legend(title="정답 레이블", fontsize=8, loc="upper right")
fig.suptitle("k=4 vs k=5 — 예측 군집별 정답 레이블 구성 (노이즈 15대의 행방)")
fig.tight_layout()
fig.savefig(fig_dir / "fig6_k4_vs_k5.png", dpi=150)
print(f"플롯 저장 — {fig_dir / 'fig6_k4_vs_k5.png'}")

res_dir = base / "outputs" / "results"
res_dir.mkdir(parents=True, exist_ok=True)
sweep.round(3).to_csv(res_dir / "k_sweep.csv", index=False)
comp_rows = []
for k in K_RANGE:
    for c in range(k):
        mask = preds[k] == c
        top = true[mask].value_counts()
        comp_rows.append({"k": k, "cluster": c, "size": int(mask.sum()),
                          "top_label": top.index[0], "purity": round(top.iloc[0] / mask.sum(), 3)})
pd.DataFrame(comp_rows).to_csv(res_dir / "k_composition.csv", index=False)
for k in (4, 5):
    pd.crosstab(true, preds[k], margins=True).to_csv(res_dir / f"crosstab_k{k}.csv")
