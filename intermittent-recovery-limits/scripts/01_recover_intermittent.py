# 기준선 K-means(k=5)가 못 찾은 intermittent 28대: 알고리즘 교체(GMM, DBSCAN) vs 버스트성 feature 7개
# 평가: 28대 회수율(intermittent 우세 군집 배정), 전체 구조는 ARI
# 출력: fig1_recovery_comparison.png, fig2_burst_feature_space.png, outputs/results/*.csv

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import setup_font
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

setup_font()

SEED = 42
K = 5
base = Path(__file__).resolve().parents[1]
fig_dir = base / "outputs" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)
res_dir = base / "outputs" / "results"
res_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(base / "data" / "usage_profiles.csv")
hour_cols = [c for c in df.columns if c.startswith("u_")]
feature_cols = hour_cols + ["total_usage"]
true = df["true_cluster"]
cube = df[hour_cols].to_numpy().reshape(len(df), 7, 24)  # (기기, 요일, 시각)

X_log = StandardScaler().fit_transform(np.log1p(df[feature_cols].to_numpy()))


def evaluate(name: str, pred: np.ndarray, missed_idx: np.ndarray) -> dict:
    """전체 ARI + 미회수 28대의 회수율(간헐 우세 군집 배정 비율). -1(DBSCAN 노이즈)은 미회수 취급."""
    ari = adjusted_rand_score(true, pred)
    majority = {}
    for c in set(pred):
        if c == -1:
            continue
        labels = true[pred == c]
        majority[c] = labels.value_counts().index[0]
    recovered = sum(1 for i in missed_idx if pred[i] != -1 and majority.get(pred[i]) == "intermittent")
    n_clusters = len([c for c in set(pred) if c != -1])
    return {
        "방법": name,
        "ARI": ari,
        "회수": recovered,
        "회수율": recovered / len(missed_idx),
        "군집 수": n_clusters,
        "pred": pred,
    }


# 기준선: 로그+표준화 K-means(k=5), 미회수 28대 특정
km_pred = KMeans(n_clusters=K, random_state=SEED, n_init=10).fit_predict(X_log)
km_majority = {c: true[km_pred == c].value_counts().index[0] for c in range(K)}
missed_mask = (true == "intermittent") & pd.Series(km_pred).map(km_majority).ne("intermittent").to_numpy()
missed_idx = np.flatnonzero(missed_mask)
print(
    f"기준선 K-means(k={K}), 미회수 intermittent: {len(missed_idx)}대 "
    f"(배정 군집의 우세 라벨: {set(pd.Series(km_pred[missed_idx]).map(km_majority))})"
)

results = [evaluate("기준선 K-means (로그+표준화)", km_pred, missed_idx)]

# 시도 1a: GMM (같은 feature)
gmm_pred = GaussianMixture(n_components=K, covariance_type="diag", random_state=SEED, n_init=3).fit_predict(X_log)
results.append(evaluate("GMM k=5 (같은 feature)", gmm_pred, missed_idx))

# 시도 1b: DBSCAN (같은 feature), eps 그리드에서 ARI 최선 채택
best_db = None
db_rows = []
for eps in (8, 10, 12, 14, 16, 18):
    db_pred = DBSCAN(eps=eps, min_samples=5).fit_predict(X_log)
    r = evaluate(f"DBSCAN eps={eps}", db_pred, missed_idx)
    noise_frac = (db_pred == -1).mean()
    print(
        f"  DBSCAN eps={eps}: 군집 {r['군집 수']}개, 노이즈 {noise_frac:.0%}, "
        f"ARI {r['ARI']:.3f}, 회수 {r['회수']}/{len(missed_idx)}"
    )
    db_rows.append(
        {
            "eps": eps,
            "clusters": r["군집 수"],
            "noise_share": round(noise_frac, 3),
            "ari": round(r["ARI"], 3),
            "recovered": r["회수"],
        }
    )
    if best_db is None or r["ARI"] > best_db["ARI"]:
        best_db = r
best_db["방법"] = best_db["방법"] + " (같은 feature, ARI 최선)"
results.append(best_db)

# 시도 2: feature 재설계, 버스트성 요약 feature 7개
hourly_mean = cube.mean(axis=1)
daily_sum = cube.sum(axis=2)

burst = pd.DataFrame(
    {
        # 평소 무사용 비율
        "zero_frac": (cube < 0.5).mean(axis=(1, 2)),
        # 상위 5% 셀(168개 중 8개) 평균
        "burst_intensity": np.sort(cube.reshape(len(df), -1), axis=1)[:, -8:].mean(axis=1),
        "top3_share": np.sort(hourly_mean, axis=1)[:, -3:].sum(axis=1) / np.maximum(hourly_mean.sum(axis=1), 1e-9),
        # 요일 간 가동률 변동, 간헐 사용의 지문
        "daily_cv": daily_sum.std(axis=1) / np.maximum(daily_sum.mean(axis=1), 1e-9),
        # night와의 대비 축은 남겨둠
        "night_share": hourly_mean[:, [21, 22, 23, 0, 1, 2]].sum(axis=1) / np.maximum(hourly_mean.sum(axis=1), 1e-9),
        "morning_share": hourly_mean[:, 6:11].sum(axis=1) / np.maximum(hourly_mean.sum(axis=1), 1e-9),
        "log_total": np.log1p(df["total_usage"]),
    }
)
X_burst = StandardScaler().fit_transform(burst.to_numpy())

burst_pred = KMeans(n_clusters=K, random_state=SEED, n_init=10).fit_predict(X_burst)
results.append(evaluate("K-means k=5 (버스트성 feature 7개)", burst_pred, missed_idx))

# 비교표
table = pd.DataFrame(
    [
        {k: (f"{v:.3f}" if k == "ARI" else f"{v:.0%}" if k == "회수율" else v) for k, v in r.items() if k != "pred"}
        for r in results
    ]
)
pd.DataFrame(db_rows).to_csv(res_dir / "dbscan_eps_grid.csv", index=False)
pd.DataFrame(
    [
        {
            "method": r["방법"],
            "ari": round(r["ARI"], 3),
            "recovered": r["회수"],
            "missed_total": len(missed_idx),
            "clusters": r["군집 수"],
        }
        for r in results
    ]
).to_csv(res_dir / "recovery_comparison.csv", index=False)
print("\n[알고리즘 교체 vs feature 재설계]")
print(table.to_string(index=False))

print("\n[버스트성 feature K-means 교차표]")
print(pd.crosstab(true, results[-1]["pred"], margins=True).to_string())
print("\n[GMM 교차표]")
print(pd.crosstab(true, results[1]["pred"], margins=True).to_string())

bp = results[-1]["pred"]
b_majority = {c: true[bp == c].value_counts().index[0] for c in set(bp)}
pd.crosstab(true, bp).to_csv(res_dir / "burst_kmeans_crosstab.csv")
print("\n버스트 feature에서 28대의 배정:", pd.Series(bp[missed_idx]).map(b_majority).value_counts().to_dict())

# 플롯 1: 회수율, ARI
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
names = [r["방법"].split(" (")[0] for r in results]
short = ["기준선\nK-means", "GMM", "DBSCAN\n(최선)", "버스트 feature\nK-means"]
recov = [r["회수율"] for r in results]
aris = [r["ARI"] for r in results]
colors = ["tab:gray", "tab:orange", "tab:purple", "tab:blue"]

ax1.bar(short, [v * 100 for v in recov], color=colors)
for i, v in enumerate(recov):
    ax1.text(i, v * 100 + 1.5, f"{results[i]['회수']}/{len(missed_idx)}", ha="center", fontsize=9)
ax1.set_ylabel("미회수 28대 회수율 (%)")
ax1.set_ylim(0, 108)
ax1.set_title("미회수 28대 회수율")
ax1.grid(alpha=0.3, axis="y")

ax2.bar(short, aris, color=colors)
for i, v in enumerate(aris):
    ax2.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=9)
ax2.set_ylabel("ARI (전체 구조)")
ax2.set_ylim(0, 1.05)
ax2.set_title("전체 구조 (ARI)")
ax2.grid(alpha=0.3, axis="y")
fig.suptitle("알고리즘 교체 vs feature 재설계")
fig.tight_layout()
fig.savefig(fig_dir / "fig1_recovery_comparison.png", dpi=150)

# 플롯 2: 버스트 feature 공간에서 28대의 위치
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
LABEL_COLORS = {
    "morning": "tab:blue",
    "allday_low": "tab:green",
    "night": "tab:purple",
    "intermittent": "tab:orange",
    "noise": "tab:red",
}
for ax, (xcol, ycol) in zip(axes, [("daily_cv", "zero_frac"), ("top3_share", "night_share")]):
    for label, color in LABEL_COLORS.items():
        m = (true == label).to_numpy()
        ax.scatter(burst.loc[m, xcol], burst.loc[m, ycol], s=10, alpha=0.4, color=color, label=label)
    ax.scatter(
        burst.loc[missed_idx, xcol],
        burst.loc[missed_idx, ycol],
        s=42,
        facecolors="none",
        edgecolors="black",
        lw=1.2,
        label="미회수 28대",
    )
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=8, loc="lower right")
fig.suptitle("버스트성 feature 공간 (검은 테두리: 미회수 28대)")
fig.tight_layout()
fig.savefig(fig_dir / "fig2_burst_feature_space.png", dpi=150)
print(f"\n플롯 저장: {fig_dir}/fig1, fig2")
