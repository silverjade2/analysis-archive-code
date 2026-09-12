"""1차 시도. scaling 없이 K-means, k=4

168셀(0~10)과 total_usage(0~5000)를 그대로. 정답 라벨은 사후 대조에만. feature별 분산도 같이 봄
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
    "K-means는 유클리드 거리라 분산 큰 축이 경계를 정함. "
    "total_usage 혼자 이 비율이면 군집은 사용 모양이 아니라 총량으로 갈린 것"
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
        ax.plot(np.arange(24), row.values, color="#B4B2A9", alpha=0.3, lw=0.8)
    ax.plot(np.arange(24), sub.mean().values, color="#9E3D22", lw=2.5, label="군집 평균")
    ax.set_title(f"예측 군집 {c} ({len(sub)}대)")
    ax.set_xticks(range(0, 24, 4))
    ax.grid(alpha=0.3)
profile_axes[0].legend(loc="upper right", fontsize=8)
for ax in profile_axes[2:]:
    ax.set_xlabel("시각")
for ax in [profile_axes[0], profile_axes[2]]:
    ax.set_ylabel("평균 사용 강도")

fig.suptitle(f"스케일링 없는 K-means (k={K}), total_usage 오름차순 (ARI={ari:.3f})")
fig.tight_layout()
fig.savefig(base / "outputs" / "figures" / "fig2_naive_kmeans.png", dpi=150)
# 아래 경로 표기 틀림 (실제 outputs/figures). 저장은 맞음
print(f"\n플롯 저장: {base / 'figures' / 'fig2_naive_kmeans.png'}")

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
