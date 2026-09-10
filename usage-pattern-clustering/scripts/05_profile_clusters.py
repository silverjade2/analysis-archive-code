# 클러스터 해석: 군집별 z-score 프로파일링과 비즈니스 라벨링
#
# 04에서 확정한 설정(로그+표준화, k=5)으로 최종 군집을 만들고,
# 각 군집을 "비즈니스가 읽을 수 있는 언어"로 번역한다.
#   1. 해석용 파생 피처 8개를 만들고 (시간대 5구간 평균, 주말 비율, 버스트성, total_usage)
#      전체 기기 기준으로 z-score화 → 군집별 평균 z-score 히트맵
#   2. 히트맵을 근거로 각 군집에 비즈니스 라벨을 붙인다
#      — 이때 "행동 세그먼트"(마케팅/UX 대상)와 "이상집단"(운영/QA 대상)의
#      층위를 구분하는 것이 핵심
#   3. 정답 레이블과 대조해 라벨링이 실제 구조와 일치하는지 확인 (사후 검증)
#
# 실행: .venv/bin/python scripts/05_profile_clusters.py
# 출력: outputs/figures/fig7_cluster_zscore.png (z-score 히트맵 + 24시간 프로파일)

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

matplotlib.rcParams["font.family"] = "AppleGothic"
matplotlib.rcParams["axes.unicode_minus"] = False

SEED = 42
K = 5
base = Path(__file__).resolve().parents[1]
fig_dir = base / "outputs" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(base / "data" / "usage_profiles.csv")
hour_cols = [c for c in df.columns if c.startswith("u_")]
feature_cols = hour_cols + ["total_usage"]

# ── 최종 클러스터링 (04와 동일 설정) ──────────────────────────────
X = StandardScaler().fit_transform(np.log1p(df[feature_cols].to_numpy()))
pred = KMeans(n_clusters=K, random_state=SEED, n_init=10).fit_predict(X)
df = pd.concat([df, pd.Series(pred, name="pred_cluster")], axis=1)

# ── 해석용 파생 피처 ──────────────────────────────────────────────
# 168개 시간대 셀을 그대로 보여주면 아무도 못 읽는다. 사람이 이해하는
# 단위(시간대 구간, 주말/주중, 버스트성)로 접어서 z-score를 계산한다.
cube = df[hour_cols].to_numpy().reshape(len(df), 7, 24)  # (기기, 요일, 시각)
hourly_mean = cube.mean(axis=1)                           # 요일 평균 24시간 프로파일

BANDS = {
    "새벽 (0~5시)": range(0, 6),
    "아침 (6~10시)": range(6, 11),
    "낮 (11~16시)": range(11, 17),
    "저녁 (17~20시)": range(17, 21),
    "밤 (21~23시)": range(21, 24),
}
derived = pd.DataFrame({name: hourly_mean[:, list(hrs)].mean(axis=1)
                        for name, hrs in BANDS.items()})
# 주말 비율: 주말 평균 강도 / 전체 평균 강도 (사용이 주말에 쏠렸는가)
weekend = cube[:, 5:, :].mean(axis=(1, 2))
overall = cube.mean(axis=(1, 2))
derived["주말 비율"] = weekend / np.maximum(overall, 1e-9)
# 버스트성: 상위 3개 시각의 강도 합이 하루 전체에서 차지하는 비율
top3 = np.sort(hourly_mean, axis=1)[:, -3:].sum(axis=1)
derived["버스트성 (상위3시간 집중도)"] = top3 / np.maximum(hourly_mean.sum(axis=1), 1e-9)
derived["총사용량 (total_usage)"] = df["total_usage"]

z = pd.DataFrame(StandardScaler().fit_transform(derived),
                 columns=derived.columns, index=df.index)
z_by_cluster = z.groupby(df["pred_cluster"]).mean()

# ── 비즈니스 라벨링 ───────────────────────────────────────────────
# 히트맵을 읽고 손으로 붙인 라벨. 자동화하지 않는 것이 의도다 —
# 라벨링은 z-score를 비즈니스 언어로 "번역"하는 판단 작업이기 때문.
# 층위 구분: 앞 4개는 행동 세그먼트, 마지막 1개는 행동이 아니라 상태(이상집단).
LABELS = {
    0: "간헐 버스트형 — 평소 무사용, 특정 시간대만 고강도",
    1: "상시 저강도형 — 종일 낮은 강도로 고르게",
    2: "심야 집중형 — 21시~새벽 2시 피크",
    3: "[이상집단] 상시 고강도 — 데모/전시 기기 의심",
    4: "아침 집중형 — 평일 6~10시 피크",
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

# ── 사후 검증: 라벨이 정답 구조와 일치하는가 ──────────────────────
print("── 사후 검증: 예측 군집 × 정답 레이블 ──")
print(pd.crosstab(df["pred_cluster"], df["true_cluster"]).to_string())

# ── 플롯: z-score 히트맵 + 군집별 24시간 프로파일 ─────────────────
fig, (ax_hm, ax_prof) = plt.subplots(
    1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.2, 1]})

order = list(range(K))
mat = z_by_cluster.loc[order].to_numpy()
im = ax_hm.imshow(mat, cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto")
ax_hm.set_xticks(range(len(derived.columns)))
ax_hm.set_xticklabels(derived.columns, rotation=30, ha="right", fontsize=9)
ax_hm.set_yticks(range(K))
ax_hm.set_yticklabels([f"군집 {c} (n={(pred == c).sum()})\n{LABELS[c].split(' — ')[0]}"
                       for c in order], fontsize=9)
for i in range(K):
    for j in range(len(derived.columns)):
        val = mat[i, j]
        ax_hm.text(j, i, f"{val:+.1f}", ha="center", va="center", fontsize=8,
                   color="white" if abs(val) > 1.4 else "black")
fig.colorbar(im, ax=ax_hm, label="평균 z-score", shrink=0.85)
ax_hm.set_title("군집별 평균 z-score — 어떤 축에서 평균과 다른가")

CLUSTER_COLORS = ["tab:orange", "tab:green", "tab:purple", "tab:red", "tab:blue"]
for c in order:
    prof = hourly_mean[pred == c].mean(axis=0)
    ax_prof.plot(np.arange(24), prof, lw=2.2, color=CLUSTER_COLORS[c],
                 label=f"군집 {c}: {LABELS[c].split(' — ')[0]}")
ax_prof.set_xticks(range(0, 24, 4))
ax_prof.set_xlabel("시각")
ax_prof.set_ylabel("평균 사용 강도 (요일 평균)")
ax_prof.set_title("군집별 24시간 평균 프로파일")
ax_prof.legend(fontsize=8)
ax_prof.grid(alpha=0.3)

fig.suptitle("최종 군집(k=5) 해석 — z-score 프로파일링과 비즈니스 라벨")
fig.tight_layout()
fig.savefig(fig_dir / "fig7_cluster_zscore.png", dpi=150)
print(f"\n플롯 저장 — {fig_dir / 'fig7_cluster_zscore.png'}")

res_dir = base / "outputs" / "results"
res_dir.mkdir(parents=True, exist_ok=True)
z_by_cluster.round(2).to_csv(res_dir / "cluster_zscore.csv")
pd.DataFrame([{"cluster": c, "n": int((pred == c).sum()), "label": LABELS[c]} for c in range(K)]) \
    .to_csv(res_dir / "cluster_labels.csv", index=False)
pd.crosstab(df["pred_cluster"], df["true_cluster"]).to_csv(res_dir / "final_crosstab.csv")
