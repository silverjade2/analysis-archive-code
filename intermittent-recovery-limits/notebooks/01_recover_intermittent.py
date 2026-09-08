# 4호 글 — 2호에서 못 찾은 intermittent 28대: 알고리즘 교체 vs 피처 재설계
#
# 2호 결론: 로그+표준화 K-means는 k=2~10 어디서도 night에 섞인 intermittent
# 28대(프로파일 혼합 기기)를 회수하지 못했다. 두 가설을 실험으로 가른다.
#   시도 1 (알고리즘 교체): 같은 피처(로그+표준화 169개)에서 GMM, DBSCAN
#   시도 2 (피처 재설계): 24×7 원시 프로파일 대신 버스트성 요약 피처 7개로 재클러스터링
#
# 평가:
#   - 기준선(2호 k=5 K-means)에서 미회수 28대를 특정하고, 각 방법에서
#     "그 28대가 intermittent 우세 군집으로 들어갔는가"(회수율)를 잰다
#   - 전체 구조 유지 여부는 ARI로 함께 확인 (28대만 잡고 나머지를 부수면 실패)
#
# 실행: .venv/bin/python notebooks/01_recover_intermittent.py
# 출력: outputs/figures/fig1_recovery_comparison.png (회수율·ARI 비교)
#       outputs/figures/fig2_burst_feature_space.png (새 피처 공간에서 28대의 위치)

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
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
true = df["true_cluster"]
cube = df[hour_cols].to_numpy().reshape(len(df), 7, 24)  # (기기, 요일, 시각)

X_log = StandardScaler().fit_transform(np.log1p(df[feature_cols].to_numpy()))


def evaluate(name: str, pred: np.ndarray, missed_idx: np.ndarray) -> dict:
    """전체 ARI + 미회수 28대의 회수율(간헐 우세 군집 배정 비율). -1(DBSCAN 노이즈)은 미회수 취급."""
    ari = adjusted_rand_score(true, pred)
    # 각 예측 군집의 우세 정답 레이블
    majority = {}
    for c in set(pred):
        if c == -1:
            continue
        labels = true[pred == c]
        majority[c] = labels.value_counts().index[0]
    recovered = sum(
        1 for i in missed_idx if pred[i] != -1 and majority.get(pred[i]) == "intermittent"
    )
    n_clusters = len([c for c in set(pred) if c != -1])
    return {"방법": name, "ARI": ari, "회수": recovered, "회수율": recovered / len(missed_idx),
            "군집 수": n_clusters, "pred": pred}


# ── 기준선: 2호의 로그+표준화 K-means(k=5) — 미회수 28대 특정 ─────
km_pred = KMeans(n_clusters=K, random_state=SEED, n_init=10).fit_predict(X_log)
km_majority = {c: true[km_pred == c].value_counts().index[0] for c in range(K)}
missed_mask = (true == "intermittent") & pd.Series(km_pred).map(km_majority).ne("intermittent").to_numpy()
missed_idx = np.flatnonzero(missed_mask)
print(f"기준선 K-means(k={K}) — 미회수 intermittent: {len(missed_idx)}대 "
      f"(배정 군집의 우세 레이블: {set(pd.Series(km_pred[missed_idx]).map(km_majority))})")

results = [evaluate("기준선 K-means (로그+표준화)", km_pred, missed_idx)]

# ── 시도 1a: GMM (같은 피처) ─────────────────────────────────────
gmm_pred = GaussianMixture(
    n_components=K, covariance_type="diag", random_state=SEED, n_init=3
).fit_predict(X_log)
results.append(evaluate("GMM k=5 (같은 피처)", gmm_pred, missed_idx))

# ── 시도 1b: DBSCAN (같은 피처) — eps 그리드에서 최선을 취한다 ────
best_db = None
for eps in (8, 10, 12, 14, 16, 18):
    db_pred = DBSCAN(eps=eps, min_samples=5).fit_predict(X_log)
    r = evaluate(f"DBSCAN eps={eps}", db_pred, missed_idx)
    noise_frac = (db_pred == -1).mean()
    print(f"  DBSCAN eps={eps}: 군집 {r['군집 수']}개, 노이즈 {noise_frac:.0%}, "
          f"ARI {r['ARI']:.3f}, 회수 {r['회수']}/{len(missed_idx)}")
    if best_db is None or r["ARI"] > best_db["ARI"]:
        best_db = r
best_db["방법"] = best_db["방법"] + " (같은 피처, ARI 최선)"
results.append(best_db)

# ── 시도 2: 피처 재설계 — 버스트성 요약 피처 7개 ─────────────────
hourly_mean = cube.mean(axis=1)          # 요일 평균 24시간 프로파일
daily_sum = cube.sum(axis=2)             # (기기, 요일) 일별 사용 합

burst = pd.DataFrame({
    # 거의 0인 셀 비율 — "평소 무사용"을 직접 잰다
    "zero_frac": (cube < 0.5).mean(axis=(1, 2)),
    # 버스트 강도: 상위 5% 셀 평균 (강한 시간대가 얼마나 강한가)
    "burst_intensity": np.sort(cube.reshape(len(df), -1), axis=1)[:, -8:].mean(axis=1),
    # 하루 안의 집중도: 상위 3시간이 차지하는 비율
    "top3_share": np.sort(hourly_mean, axis=1)[:, -3:].sum(axis=1)
                  / np.maximum(hourly_mean.sum(axis=1), 1e-9),
    # 요일 간 변동계수 — 가동률이 들쭉날쭉한가 (간헐 사용의 지문)
    "daily_cv": daily_sum.std(axis=1) / np.maximum(daily_sum.mean(axis=1), 1e-9),
    # 야간(21~02시) 비중 — night와의 대비 축은 남겨둔다
    "night_share": hourly_mean[:, [21, 22, 23, 0, 1, 2]].sum(axis=1)
                   / np.maximum(hourly_mean.sum(axis=1), 1e-9),
    # 아침(6~10시) 비중
    "morning_share": hourly_mean[:, 6:11].sum(axis=1)
                     / np.maximum(hourly_mean.sum(axis=1), 1e-9),
    # 총량 (로그)
    "log_total": np.log1p(df["total_usage"]),
})
X_burst = StandardScaler().fit_transform(burst.to_numpy())

burst_pred = KMeans(n_clusters=K, random_state=SEED, n_init=10).fit_predict(X_burst)
results.append(evaluate("K-means k=5 (버스트성 피처 7개)", burst_pred, missed_idx))

# ── 비교표 ────────────────────────────────────────────────────────
table = pd.DataFrame(
    [{k: (f"{v:.3f}" if k == "ARI" else f"{v:.0%}" if k == "회수율" else v)
      for k, v in r.items() if k != "pred"} for r in results]
)
print("\n── 알고리즘 교체 vs 피처 재설계 ──")
print(table.to_string(index=False))

print("\n── 버스트성 피처 K-means: 교차표 ──")
print(pd.crosstab(true, results[-1]["pred"], margins=True).to_string())
print("\n── GMM: 교차표 ──")
print(pd.crosstab(true, results[1]["pred"], margins=True).to_string())

# 28대가 버스트 피처 재설계에서 어디로 갔는지
bp = results[-1]["pred"]
b_majority = {c: true[bp == c].value_counts().index[0] for c in set(bp)}
print("\n버스트 피처에서 28대의 배정:", pd.Series(bp[missed_idx]).map(b_majority).value_counts().to_dict())

# ── 플롯 1: 회수율·ARI 비교 ──────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
names = [r["방법"].split(" (")[0] for r in results]
short = ["기준선\nK-means", "GMM", "DBSCAN\n(최선)", "버스트 피처\nK-means"]
recov = [r["회수율"] for r in results]
aris = [r["ARI"] for r in results]
colors = ["tab:gray", "tab:orange", "tab:purple", "tab:blue"]

ax1.bar(short, [v * 100 for v in recov], color=colors)
for i, v in enumerate(recov):
    ax1.text(i, v * 100 + 1.5, f"{results[i]['회수']}/{len(missed_idx)}", ha="center", fontsize=9)
ax1.set_ylabel("미회수 28대 회수율 (%)")
ax1.set_ylim(0, 108)
ax1.set_title("28대를 간헐 우세 군집으로 회수했는가")
ax1.grid(alpha=0.3, axis="y")

ax2.bar(short, aris, color=colors)
for i, v in enumerate(aris):
    ax2.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=9)
ax2.set_ylabel("ARI (전체 구조)")
ax2.set_ylim(0, 1.05)
ax2.set_title("전체 군집 구조는 유지되는가")
ax2.grid(alpha=0.3, axis="y")
fig.suptitle("알고리즘 교체 vs 피처 재설계 — 회수율과 전체 구조")
fig.tight_layout()
fig.savefig(fig_dir / "fig1_recovery_comparison.png", dpi=150)

# ── 플롯 2: 버스트 피처 공간에서 28대의 위치 ─────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
LABEL_COLORS = {"morning": "tab:blue", "allday_low": "tab:green",
                "night": "tab:purple", "intermittent": "tab:orange", "noise": "tab:red"}
for ax, (xcol, ycol) in zip(axes, [("daily_cv", "zero_frac"), ("top3_share", "night_share")]):
    for label, color in LABEL_COLORS.items():
        m = (true == label).to_numpy()
        ax.scatter(burst.loc[m, xcol], burst.loc[m, ycol], s=10, alpha=0.4,
                   color=color, label=label)
    ax.scatter(burst.loc[missed_idx, xcol], burst.loc[missed_idx, ycol], s=42,
               facecolors="none", edgecolors="black", lw=1.2, label="미회수 28대")
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=8, loc="lower right")
fig.suptitle("버스트성 피처 공간 — 원시 프로파일에서 안 보이던 축이 생긴다")
fig.tight_layout()
fig.savefig(fig_dir / "fig2_burst_feature_space.png", dpi=150)
print(f"\n플롯 저장 — {fig_dir}/fig1, fig2")
