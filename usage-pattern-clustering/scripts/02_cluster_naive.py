"""1차 시도. scaling 없이 K-means, k=4.

"그냥 돌리면 어떻게 되는가"를 먼저 본다. 168개 셀은 0에서 10, total_usage는 0에서 5000 스케일 그대로 넣는다.
정답 라벨은 학습에 쓰지 않고 사후 대조에만 쓴다. ARI와 교차표 외에 feature별 분산을 비교해 total_usage 하나가
거리 계산을 지배하는지 숫자로 확인한다.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import setup_font
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

setup_font()

SEED = 42
K = 4
base = Path(__file__).resolve().parents[1]

df = pd.read_csv(base / "data" / "usage_profiles.csv")
hour_cols = [c for c in df.columns if c.startswith("u_")]
feature_cols = hour_cols + ["total_usage"]

X = df[feature_cols].to_numpy()

km = KMeans(n_clusters=K, random_state=SEED, n_init=10)
pred = km.fit_predict(X)
df = df.assign(pred_cluster=pred)

ari = adjusted_rand_score(df["true_cluster"], pred)
print(f"ARI (스케일링 없음, k={K}): {ari:.3f}")

crosstab = pd.crosstab(df["true_cluster"], df["pred_cluster"], margins=True)
print("\n교차표 (정답 레이블 × 예측 군집)")
print(crosstab.to_string())

usage_by_pred = df.groupby("pred_cluster")["total_usage"].agg(["mean", "std", "count"])
usage_by_pred = usage_by_pred.sort_values("mean")
print("\n예측 군집별 total_usage 평균 (오름차순)")
print(usage_by_pred.to_string())

var_hourly_sum = df[hour_cols].var().sum()
var_total_usage = df["total_usage"].var()
print(f"\n시간대 피처 168개 분산의 합: {var_hourly_sum:,.1f}")
print(f"total_usage 하나의 분산:      {var_total_usage:,.1f}")
print(f"total_usage가 전체 분산에서 차지하는 비율: {var_total_usage / (var_hourly_sum + var_total_usage):.1%}")
print(
    "→ K-means는 유클리드 거리를 쓰므로 분산이 큰 축이 군집 경계를 사실상 결정한다. "
    "total_usage 혼자 이 비율을 차지한다면, 예측 군집은 하루 사용 '모양'이 아니라 "
    "'총량 크기'로 갈렸을 가능성이 크다."
)

hourly = pd.DataFrame(
    df[hour_cols].to_numpy().reshape(len(df), 7, 24).mean(axis=1),
    columns=np.arange(24),
)
hourly["pred_cluster"] = pred

order = usage_by_pred.index.tolist()

fig = plt.figure(figsize=(13, 7))
gs = fig.add_gridspec(2, 3)

ax_box = fig.add_subplot(gs[:, 0])
box_data = [df.loc[df["pred_cluster"] == c, "total_usage"] for c in order]
ax_box.boxplot(box_data, tick_labels=[f"예측 {c}\n(n={(pred == c).sum()})" for c in order])
ax_box.set_ylabel("total_usage")
ax_box.set_title("예측 군집별 total_usage 분포")
ax_box.grid(alpha=0.3, axis="y")

profile_axes = [fig.add_subplot(gs[i // 2, 1 + i % 2]) for i in range(4)]
for ax, c in zip(profile_axes, order):
    sub = hourly[hourly["pred_cluster"] == c].drop(columns="pred_cluster")
    sample = sub.sample(min(15, len(sub)), random_state=SEED)
    for _, row in sample.iterrows():
        ax.plot(np.arange(24), row.values, color="tab:gray", alpha=0.3, lw=0.8)
    ax.plot(np.arange(24), sub.mean().values, color="tab:red", lw=2.5, label="군집 평균")
    ax.set_title(f"예측 군집 {c} — {len(sub)}대")
    ax.set_xticks(range(0, 24, 4))
    ax.grid(alpha=0.3)
profile_axes[0].legend(loc="upper right", fontsize=8)
for ax in profile_axes[2:]:
    ax.set_xlabel("시각")
for ax in [profile_axes[0], profile_axes[2]]:
    ax.set_ylabel("평균 사용 강도")

fig.suptitle(f"스케일링 없는 K-means(k={K}) — total_usage 오름차순 정렬 (ARI={ari:.3f})")
fig.tight_layout()
fig.savefig(base / "outputs" / "figures" / "fig2_naive_kmeans.png", dpi=150)
print(f"\n플롯 저장 — {base / 'figures' / 'fig2_naive_kmeans.png'}")

res_dir = base / "outputs" / "results"
res_dir.mkdir(parents=True, exist_ok=True)
pd.DataFrame(
    {
        "metric": ["ari", "var_hourly_sum", "var_total_usage", "var_share_total_usage"],
        "value": [ari, var_hourly_sum, var_total_usage, var_total_usage / (var_hourly_sum + var_total_usage)],
    }
).round(4).to_csv(res_dir / "naive_kmeans_summary.csv", index=False)
crosstab.to_csv(res_dir / "naive_crosstab.csv")
df.groupby("pred_cluster")["total_usage"].agg(["min", "max", "mean", "std", "count"]).sort_values("min").round(
    1
).to_csv(res_dir / "naive_usage_by_pred.csv")
