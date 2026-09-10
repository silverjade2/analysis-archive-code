# 클러스터링 2차 시도: 스케일링 적용 후 재시도 (k=4 고정)
#
# 02에서 확인한 것: 스케일링 없이 돌리면 total_usage 혼자 분산의 99.8%를 차지해
# 예측 군집이 하루 사용 "모양"이 아니라 total_usage "크기"로 갈렸다.
# 이번엔 두 가지 스케일링을 비교한다.
#   (a) 표준화만 (StandardScaler) — 평균 0, 분산 1로 맞춘다. 분산 지배 문제는 없어지지만
#       치우친(skewed) 분포는 그대로 남는다.
#   (b) 로그 변환(log1p) + 표준화 — 치우친 피처를 먼저 정규분포에 가깝게 편 다음 표준화한다.
#       total_usage(0~5000, 오른쪽 꼬리 김)뿐 아니라 intermittent 프로파일의 시간대 피처도
#       "대부분 0 근처 + 가끔 버스트"라 치우쳐 있어, 로그 변환이 여기도 영향을 준다.
#
# 02와 동일하게 true_cluster는 사후 대조에만 쓴다.
#   - ARI, 교차표
#   - 노이즈 소군집(15대)이 어느 예측 군집에 배정되는지 추적 (재현율/순도)
#   - 예측 군집별 시간대 프로파일 플롯
#   - 1차(스케일링 없음) 대비 ARI 변화 요약표
#
# 실행: .venv/bin/python scripts/03_cluster_scaled.py
# 출력: outputs/figures/fig3_standard_kmeans.png
#       outputs/figures/fig4_log_standard_kmeans.png

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

matplotlib.rcParams["font.family"] = "AppleGothic"
matplotlib.rcParams["axes.unicode_minus"] = False

SEED = 42
K = 4
base = Path(__file__).resolve().parents[1]
fig_dir = base / "outputs" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(base / "data" / "usage_profiles.csv")
hour_cols = [c for c in df.columns if c.startswith("u_")]
feature_cols = hour_cols + ["total_usage"]
X_raw = df[feature_cols].to_numpy()

# ── 진단: 치우침(skewness) 비교 — total_usage만 치우친 게 아니다 ──────────
skew_hourly = skew(df[hour_cols].to_numpy(), axis=0)
skew_total = skew(df["total_usage"].to_numpy())
print("피처 치우침(skewness) — 0에 가까울수록 대칭")
print(f"  시간대 피처 168개: 평균 {skew_hourly.mean():.2f}, 최대 {skew_hourly.max():.2f} "
      f"(intermittent류의 '대부분 0 + 가끔 버스트' 셀)")
print(f"  total_usage:        {skew_total:.2f}")
print()


def run_kmeans(X: np.ndarray, method: str) -> dict:
    km = KMeans(n_clusters=K, random_state=SEED, n_init=10)
    pred = km.fit_predict(X)
    ari = adjusted_rand_score(df["true_cluster"], pred)

    crosstab = pd.crosstab(df["true_cluster"], pred, margins=True)
    print(f"── {method} — ARI={ari:.3f} ──")
    print(crosstab.to_string())

    noise_pred = pd.Series(pred)[df["true_cluster"] == "noise"]
    top_cluster = noise_pred.value_counts().idxmax()
    recall = (noise_pred == top_cluster).sum() / len(noise_pred)
    purity = (df["true_cluster"][pred == top_cluster] == "noise").mean()
    print(f"노이즈 소군집(15대) → 예측 군집 {top_cluster}로 최다 배정 "
          f"(재현율 {recall:.0%}, 그 군집 내 노이즈 순도 {purity:.0%})")
    print()

    return {"method": method, "ari": ari, "pred": pred,
            "noise_cluster": top_cluster, "noise_recall": recall, "noise_purity": purity}


def plot_result(pred: np.ndarray, method: str, fname: str) -> None:
    local = pd.concat([df, pd.Series(pred, name="pred_cluster")], axis=1)
    usage_by_pred = local.groupby("pred_cluster")["total_usage"].mean().sort_values()
    order = usage_by_pred.index.tolist()

    hourly = pd.DataFrame(
        df[hour_cols].to_numpy().reshape(len(df), 7, 24).mean(axis=1),
        columns=np.arange(24),
    )
    hourly["pred_cluster"] = pred

    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2, 3)

    ax_box = fig.add_subplot(gs[:, 0])
    box_data = [local.loc[local["pred_cluster"] == c, "total_usage"] for c in order]
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

    fig.suptitle(f"{method} — k={K} (total_usage 오름차순 정렬)")
    fig.tight_layout()
    fig.savefig(fig_dir / fname, dpi=150)
    print(f"플롯 저장 — {fig_dir / fname}\n")


# ── 0차 대조군: 02의 스케일링 없음 결과를 동일 SEED로 재계산 ─────────────
naive = run_kmeans(X_raw, "스케일링 없음 (02 재계산)")

# ── (a) 표준화만 ──────────────────────────────────────────────────
X_std = StandardScaler().fit_transform(X_raw)
result_std = run_kmeans(X_std, "표준화만 (StandardScaler)")
plot_result(result_std["pred"], "표준화만 (StandardScaler)", "fig3_standard_kmeans.png")

# ── (b) 로그 변환 + 표준화 ────────────────────────────────────────
X_log_std = StandardScaler().fit_transform(np.log1p(X_raw))
result_log = run_kmeans(X_log_std, "로그 변환(log1p) + 표준화")
plot_result(result_log["pred"], "로그 변환(log1p) + 표준화", "fig4_log_standard_kmeans.png")

# ── 요약표: 1차 대비 ARI 변화 ─────────────────────────────────────
summary = pd.DataFrame(
    [
        {"방법": r["method"], "ARI": round(r["ari"], 3),
         "노이즈 재현율": f"{r['noise_recall']:.0%}", "노이즈 순도": f"{r['noise_purity']:.0%}"}
        for r in (naive, result_std, result_log)
    ]
)
summary["ARI 변화(1차 대비)"] = summary["ARI"] - summary.loc[0, "ARI"]
print("── 요약: 스케일링 방법별 ARI 비교 ──")
print(summary.to_string(index=False))

res_dir = base / "outputs" / "results"
res_dir.mkdir(parents=True, exist_ok=True)
pd.DataFrame({"metric": ["skew_hourly_mean", "skew_hourly_max", "skew_total_usage"],
              "value": [skew_hourly.mean(), skew_hourly.max(), skew_total]}).round(3) \
    .to_csv(res_dir / "skewness.csv", index=False)
summary.to_csv(res_dir / "scaling_compare.csv", index=False)
for r, name in ((result_std, "standard"), (result_log, "log_standard")):
    pd.crosstab(df["true_cluster"], r["pred"], margins=True).to_csv(res_dir / f"crosstab_{name}.csv")
