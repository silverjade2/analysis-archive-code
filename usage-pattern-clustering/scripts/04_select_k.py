"""k 선택. log1p + 표준화, k=2~10

- elbow, silhouette가 어느 k를 가리키는지
- 03에서 allday_low에 흡수된 노이즈 15대가 k=5에서 독립 군집으로 돌아오는지
- k별 군집 크기와 정답 구성 기록
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import setup_font
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

setup_font()

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

records = []
preds = {}
for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    pred = km.fit_predict(X)
    preds[k] = pred
    records.append(
        {
            "k": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(X, pred),
            "ari": adjusted_rand_score(true, pred),
        }
    )

sweep = pd.DataFrame(records)
print("── k 스윕 (로그+표준화, KMeans n_init=10) ──")
print(sweep.round(3).to_string(index=False))
print()

print("── 각 k의 군집 구성 (크기 / 최다 정답 레이블 / 순도) ──")
for k in K_RANGE:
    pred = preds[k]
    parts = []
    for c in range(k):
        mask = pred == c
        top = true[mask].value_counts()
        parts.append(f"c{c}:{mask.sum()}({top.index[0]} {top.iloc[0] / mask.sum():.0%})")
    print(f"k={k}: " + "  ".join(parts))
print()

print("── k=4 vs k=5: 노이즈 소군집 15대 추적 ──")
for k in (4, 5):
    pred = preds[k]
    crosstab = pd.crosstab(true, pred, margins=True)
    print(f"[k={k}] ARI={sweep.loc[sweep['k'] == k, 'ari'].iloc[0]:.3f}")
    print(crosstab.to_string())
    noise_pred = pd.Series(pred)[true == "noise"]
    top_cluster = noise_pred.value_counts().idxmax()
    recall = (noise_pred == top_cluster).sum() / len(noise_pred)
    purity = (true[pred == top_cluster] == "noise").mean()
    print(f"노이즈 15대 -> 예측 군집 {top_cluster} (재현율 {recall:.0%}, 순도 {purity:.0%})\n")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].plot(sweep["k"], sweep["inertia"], marker="o", color="#5B6B7A")
axes[0].set_xlabel("k")
axes[0].set_ylabel("inertia (군집 내 제곱합)")
axes[0].set_title("elbow plot")
axes[0].grid(alpha=0.3)

axes[1].plot(sweep["k"], sweep["silhouette"], marker="o", color="#9E3D22")
best_k = sweep.loc[sweep["silhouette"].idxmax(), "k"]
axes[1].axvline(best_k, color="#9E3D22", ls="--", alpha=0.6, label=f"silhouette 최대 k={best_k}")
axes[1].set_xlabel("k")
axes[1].set_ylabel("silhouette 점수")
axes[1].set_title("silhouette 점수")
axes[1].legend()
axes[1].grid(alpha=0.3)

for ax in axes:
    ax.set_xticks(list(K_RANGE))
fig.suptitle("k sweep, 로그+표준화 K-means (k=2~10)")
fig.tight_layout()
fig.savefig(fig_dir / "fig5_k_sweep.png", dpi=150)
print(f"플롯 저장: {fig_dir / 'fig5_k_sweep.png'}")

LABEL_ORDER = ["morning", "allday_low", "night", "intermittent", "noise"]
LABEL_COLORS = {
    "morning": "#5B6B7A",
    "allday_low": "#8A8F98",
    "night": "tab:purple",
    "intermittent": "#9E3D22",
    "noise": "#9E3D22",
}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for ax, k in zip(axes, (4, 5)):
    pred = preds[k]
    comp = pd.crosstab(pred, true).reindex(columns=LABEL_ORDER, fill_value=0)
    bottom = np.zeros(len(comp))
    for label in LABEL_ORDER:
        ax.bar(comp.index, comp[label], bottom=bottom, color=LABEL_COLORS[label], label=label)
        bottom += comp[label].to_numpy()
    ari = sweep.loc[sweep["k"] == k, "ari"].iloc[0]
    ax.set_title(f"k={k} (ARI={ari:.3f})")
    ax.set_xlabel("예측 군집")
    ax.set_xticks(comp.index)
    ax.grid(alpha=0.3, axis="y")
axes[0].set_ylabel("기기 수")
axes[1].legend(title="정답 레이블", fontsize=8, loc="upper right")
fig.suptitle("k=4 vs k=5: 예측 군집별 정답 레이블 구성")
fig.tight_layout()
fig.savefig(fig_dir / "fig6_k4_vs_k5.png", dpi=150)
print(f"플롯 저장: {fig_dir / 'fig6_k4_vs_k5.png'}")

res_dir = base / "outputs" / "results"
res_dir.mkdir(parents=True, exist_ok=True)
sweep.round(3).to_csv(res_dir / "k_sweep.csv", index=False)
comp_rows = []
for k in K_RANGE:
    for c in range(k):
        mask = preds[k] == c
        top = true[mask].value_counts()
        comp_rows.append(
            {
                "k": k,
                "cluster": c,
                "size": int(mask.sum()),
                "top_label": top.index[0],
                "purity": round(top.iloc[0] / mask.sum(), 3),
            }
        )
pd.DataFrame(comp_rows).to_csv(res_dir / "k_composition.csv", index=False)
for k in (4, 5):
    pd.crosstab(true, preds[k], margins=True).to_csv(res_dir / f"crosstab_k{k}.csv")
