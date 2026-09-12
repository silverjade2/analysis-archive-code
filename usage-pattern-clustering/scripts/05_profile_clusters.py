"""최종 군집 (log1p + 표준화, k=5) 해석과 비즈니스 라벨

- 168셀 대신 시간대 5구간 평균, 주말 비율, 버스트성, total_usage로 접어 z-score
- 라벨은 군집별 평균 z-score 보고 손으로. 행동 세그먼트 4 + 이상집단 1 (행동이 아니라 상태)
- 정답 라벨 대조는 맨 뒤 사후 검증만
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import setup_font
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

setup_font()

SEED = 42
K = 5
base = Path(__file__).resolve().parents[1]
fig_dir = base / "outputs" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(base / "data" / "usage_profiles.csv")
hour_cols = [c for c in df.columns if c.startswith("u_")]
feature_cols = hour_cols + ["total_usage"]

X = StandardScaler().fit_transform(np.log1p(df[feature_cols].to_numpy()))
pred = KMeans(n_clusters=K, random_state=SEED, n_init=10).fit_predict(X)
df = pd.concat([df, pd.Series(pred, name="pred_cluster")], axis=1)

cube = df[hour_cols].to_numpy().reshape(len(df), 7, 24)  # (기기, 요일, 시각)
hourly_mean = cube.mean(axis=1)

BANDS = {
    "새벽 (0~5시)": range(0, 6),
    "아침 (6~10시)": range(6, 11),
    "낮 (11~16시)": range(11, 17),
    "저녁 (17~20시)": range(17, 21),
    "밤 (21~23시)": range(21, 24),
}
derived = pd.DataFrame({name: hourly_mean[:, list(hrs)].mean(axis=1) for name, hrs in BANDS.items()})
weekend = cube[:, 5:, :].mean(axis=(1, 2))
overall = cube.mean(axis=(1, 2))
derived["주말 비율"] = weekend / np.maximum(overall, 1e-9)
# 버스트성 = 상위 3시각 합 / 하루 합
top3 = np.sort(hourly_mean, axis=1)[:, -3:].sum(axis=1)
derived["버스트성 (상위3시간 집중도)"] = top3 / np.maximum(hourly_mean.sum(axis=1), 1e-9)
derived["총사용량 (total_usage)"] = df["total_usage"]

z = pd.DataFrame(StandardScaler().fit_transform(derived), columns=derived.columns, index=df.index)
z_by_cluster = z.groupby(df["pred_cluster"]).mean()

# 히트맵 보고 손으로 붙임. 번호는 KMeans가 준 것이라 seed나 데이터 바뀌면 다시 읽어야 함
LABELS = {
    0: "간헐 버스트형: 평소 무사용, 특정 시간대만 고강도",
    1: "상시 저강도형: 종일 낮은 강도로 고르게",
    2: "심야 집중형: 21시~새벽 2시 피크",
    3: "[이상집단] 상시 고강도: 데모/전시 기기 의심",
    4: "아침 집중형: 평일 6~10시 피크",
}

print("── 군집별 평균 z-score (해석용 파생 피처) ──")
print(z_by_cluster.round(2).to_string())
print()

print("── 군집 요약과 비즈니스 라벨 ──")
for c in range(K):
    n = (pred == c).sum()
    top = z_by_cluster.loc[c].abs().sort_values(ascending=False).head(3)
    signs = ", ".join(f"{name} {z_by_cluster.loc[c, name]:+.1f}" for name in top.index)
    print(f"군집 {c} (n={n:3d}): {LABELS[c]}")
    print(f"  근거 z-score: {signs}")
print()

print("── 사후 검증: 예측 군집 × 정답 레이블 ──")
print(pd.crosstab(df["pred_cluster"], df["true_cluster"]).to_string())

fig, (ax_hm, ax_prof) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.2, 1]})

order = list(range(K))
mat = z_by_cluster.loc[order].to_numpy()
im = ax_hm.imshow(mat, cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto")
ax_hm.set_xticks(range(len(derived.columns)))
ax_hm.set_xticklabels(derived.columns, rotation=30, ha="right", fontsize=9)
ax_hm.set_yticks(range(K))
ax_hm.set_yticklabels([f"군집 {c} (n={(pred == c).sum()})\n{LABELS[c].split(': ')[0]}" for c in order], fontsize=9)
for i in range(K):
    for j in range(len(derived.columns)):
        val = mat[i, j]
        ax_hm.text(
            j, i, f"{val:+.1f}", ha="center", va="center", fontsize=8, color="white" if abs(val) > 1.4 else "black"
        )
fig.colorbar(im, ax=ax_hm, label="평균 z-score", shrink=0.85)
ax_hm.set_title("군집별 평균 z-score")

CLUSTER_COLORS = ["#9E3D22", "#8A8F98", "tab:purple", "#9E3D22", "#5B6B7A"]
for c in order:
    prof = hourly_mean[pred == c].mean(axis=0)
    ax_prof.plot(np.arange(24), prof, lw=2.2, color=CLUSTER_COLORS[c], label=f"군집 {c}: {LABELS[c].split(': ')[0]}")
ax_prof.set_xticks(range(0, 24, 4))
ax_prof.set_xlabel("시각")
ax_prof.set_ylabel("평균 사용 강도 (요일 평균)")
ax_prof.set_title("군집별 24시간 평균 프로파일")
ax_prof.legend(fontsize=8)
ax_prof.grid(alpha=0.3)

fig.suptitle("최종 군집 (k=5) z-score 프로파일과 라벨")
fig.tight_layout()
fig.savefig(fig_dir / "fig7_cluster_zscore.png", dpi=150)
print(f"\n플롯 저장: {fig_dir / 'fig7_cluster_zscore.png'}")

res_dir = base / "outputs" / "results"
res_dir.mkdir(parents=True, exist_ok=True)
z_by_cluster.round(2).to_csv(res_dir / "cluster_zscore.csv")
pd.DataFrame([{"cluster": c, "n": int((pred == c).sum()), "label": LABELS[c]} for c in range(K)]).to_csv(
    res_dir / "cluster_labels.csv", index=False
)
pd.crosstab(df["pred_cluster"], df["true_cluster"]).to_csv(res_dir / "final_crosstab.csv")
