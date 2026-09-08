# 2호 글 — 클러스터링 1차 시도: 스케일링 없이 K-means(k=4)
#
# 목적: "그냥 돌리면 어떻게 되는가"를 먼저 확인한다. 함정 1(total_usage만
#   0~5000 스케일)이 어떤 식으로 결과를 지배하는지 정답 레이블과 대조해 진단한다.
#
# 입력 피처: u_{dow}_h{HH} 168개(0~10 스케일) + total_usage(0~5000 스케일).
#   스케일링을 전혀 적용하지 않은 원 스케일 그대로 KMeans에 투입한다.
# 평가: true_cluster는 학습에 쓰지 않고, 사후 대조에만 사용한다.
#   - ARI (adjusted_rand_score): 예측 군집이 정답 군집 구조와 얼마나 일치하는가
#   - 교차표: 예측 군집 × 정답 레이블
#   - 진단: 피처별 분산 비교로 total_usage가 거리 계산을 지배하는지 정량 확인
#
# 실행: .venv/bin/python notebooks/02_cluster_naive.py
# 출력: outputs/figures/fig2_naive_kmeans.png

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

matplotlib.rcParams["font.family"] = "AppleGothic"
matplotlib.rcParams["axes.unicode_minus"] = False

SEED = 42
K = 4
base = Path(__file__).resolve().parents[1]

df = pd.read_csv(base / "data" / "usage_profiles.csv")
hour_cols = [c for c in df.columns if c.startswith("u_")]
feature_cols = hour_cols + ["total_usage"]

X = df[feature_cols].to_numpy()

# ── K-means 원 스케일 그대로 ─────────────────────────────────────
km = KMeans(n_clusters=K, random_state=SEED, n_init=10)
pred = km.fit_predict(X)
df = df.assign(pred_cluster=pred)

ari = adjusted_rand_score(df["true_cluster"], pred)
print(f"ARI (스케일링 없음, k={K}): {ari:.3f}")

crosstab = pd.crosstab(df["true_cluster"], df["pred_cluster"], margins=True)
print("\n교차표 (정답 레이블 × 예측 군집)")
print(crosstab.to_string())

# ── 진단 1: 예측 군집이 total_usage로 정렬되는가 ─────────────────
usage_by_pred = df.groupby("pred_cluster")["total_usage"].agg(["mean", "std", "count"])
usage_by_pred = usage_by_pred.sort_values("mean")
print("\n예측 군집별 total_usage 평균 (오름차순)")
print(usage_by_pred.to_string())

# ── 진단 2: 피처별 분산 비교 — total_usage 하나가 168개 시간대 피처 합보다 큰가 ──
var_hourly_sum = df[hour_cols].var().sum()
var_total_usage = df["total_usage"].var()
print(f"\n시간대 피처 168개 분산의 합: {var_hourly_sum:,.1f}")
print(f"total_usage 하나의 분산:      {var_total_usage:,.1f}")
print(f"total_usage가 전체 분산에서 차지하는 비율: "
      f"{var_total_usage / (var_hourly_sum + var_total_usage):.1%}")
print(
    "→ K-means는 유클리드 거리를 쓰므로 분산이 큰 축이 군집 경계를 사실상 결정한다. "
    "total_usage 혼자 이 비율을 차지한다면, 예측 군집은 하루 사용 '모양'이 아니라 "
    "'총량 크기'로 갈렸을 가능성이 크다."
)

# ── 시각화: (1) 예측 군집별 total_usage 분포, (2~5) 예측 군집별 시간대 프로파일 ──
hourly = pd.DataFrame(
    df[hour_cols].to_numpy().reshape(len(df), 7, 24).mean(axis=1),
    columns=np.arange(24),
)
hourly["pred_cluster"] = pred

order = usage_by_pred.index.tolist()  # total_usage 오름차순으로 패널 정렬

fig = plt.figure(figsize=(13, 7))
gs = fig.add_gridspec(2, 3)

ax_box = fig.add_subplot(gs[:, 0])
box_data = [df.loc[df["pred_cluster"] == c, "total_usage"] for c in order]
ax_box.boxplot(box_data, tick_labels=[f"예측 {c}\n(n={ (pred==c).sum() })" for c in order])
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
