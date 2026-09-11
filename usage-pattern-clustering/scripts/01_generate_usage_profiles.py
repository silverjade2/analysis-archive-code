"""기기 800대의 사용 프로파일을 만든다. 요일 7 × 시각 24 = 168개 셀과 total_usage, 그리고 정답 라벨.

클러스터링은 정답이 없어 결과가 틀려도 티가 나지 않는다. 그래서 정답을 심었다. 진짜 군집은 넷이다. 평일 아침
6시에서 10시 피크, 종일 낮은 강도, 21시에서 새벽 2시 피크, 평소 거의 0이고 특정 시간대에만 고강도 버스트.
함정은 셋이다. total_usage만 0에서 5000 스케일이라 scaling 없이는 이 축이 거리를 지배하고, night와
intermittent 일부 기기의 프로파일을 섞어 경계를 흐리며, 24시간 상시 고강도인 15대짜리 노이즈 소군집을 얹었다.
두 번째 함정의 애초 목적은 elbow를 애매하게 만드는 것이었는데 그 목적은 실패했고 글에 그렇게 적었다.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import setup_font

setup_font()

SEED = 42
N_DEVICES = 800

# 합계 800, 노이즈 소군집 15대 포함
CLUSTER_SIZES = {
    "morning": 230,
    "allday_low": 240,
    "night": 165,
    "intermittent": 150,
    "noise": 15,
}

DOWS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
HOURS = np.arange(24)

BLUR_FRAC = 0.35  # night와 intermittent에서 프로파일을 섞을 기기 비율
BLUR_ALPHA = (0.45, 0.75)  # 자기 군집 템플릿의 혼합 가중치 범위

CELL_NOISE_SD = 0.45  # 셀 단위 관측 노이즈
AMP_SIGMA = 0.15  # 기기별 전체 강도 편차, lognormal sigma
TOTAL_SCALE = 3.5  # total_usage = 셀 합 × TOTAL_SCALE + 노이즈. 이 배율이 스케일 차이를 만든다

rng = np.random.default_rng(SEED)
base = Path(__file__).resolve().parents[1]
data_dir = base / "data"
fig_dir = base / "outputs" / "figures"
data_dir.mkdir(parents=True, exist_ok=True)
fig_dir.mkdir(parents=True, exist_ok=True)

assert sum(CLUSTER_SIZES.values()) == N_DEVICES


def circ_gauss(hours: np.ndarray, center: float, width: float) -> np.ndarray:
    """24시간 원형 거리 기반 가우시안 범프 (자정 넘어가는 피크 처리용)."""
    d = np.abs(hours - center)
    d = np.minimum(d, 24 - d)
    return np.exp(-0.5 * (d / width) ** 2)


def morning_profile(r: np.random.Generator) -> np.ndarray:
    """아침집중: 평일 6~10시 피크, 주말은 절반 이하."""
    shift = r.normal(0, 0.7)
    hourly = 0.3 + 7.0 * circ_gauss(HOURS, 7.5 + shift, 1.6)
    dow_factor = np.array([1.0, 1.0, 1.0, 1.0, 0.95, 0.45, 0.40])
    return np.outer(dow_factor, hourly)


def allday_low_profile(r: np.random.Generator) -> np.ndarray:
    """종일저강도: 낮은 강도로 고르게, 낮 시간에 살짝 볼록."""
    level = r.uniform(1.0, 1.8)
    hourly = level + 0.6 * circ_gauss(HOURS, 14.0, 5.0)
    dow_factor = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.9])
    return np.outer(dow_factor, hourly)


def night_profile(r: np.random.Generator) -> np.ndarray:
    """야간집중: 21시~새벽 2시 피크 (자정을 넘는 원형 범프)."""
    shift = r.normal(0, 0.7)
    hourly = 0.4 + 6.5 * circ_gauss(HOURS, 23.5 + shift, 2.0)
    dow_factor = np.array([0.9, 0.9, 0.9, 0.95, 1.15, 1.2, 1.0])
    return np.outer(dow_factor, hourly)


def intermittent_profile(r: np.random.Generator) -> np.ndarray:
    """간헐고강도: 기기마다 2~4개 버스트 시간대. 버스트를 저녁에 치우치게 뽑아 night과의 경계를 한 번 더 흐린다."""
    n_burst = r.integers(2, 5)
    hour_weights = np.ones(24)
    hour_weights[17:24] = 4.0
    hour_weights /= hour_weights.sum()
    burst_hours = r.choice(24, size=n_burst, replace=False, p=hour_weights)

    hourly = np.full(24, 0.15)
    hourly[burst_hours] = r.uniform(7.0, 9.0, size=n_burst)
    active_frac = r.uniform(0.2, 0.7, size=7)  # 요일별 가동률. 집계값이 그만큼 깎인다
    return np.outer(active_frac, hourly) + 0.1


def noise_profile(r: np.random.Generator) -> np.ndarray:
    """노이즈성 소군집: 24시간 상시 고강도 (데모/전시용 기기 같은 이상 집단)."""
    level = r.normal(7.5, 0.4)
    return np.full((7, 24), level) + r.normal(0, 1.2, size=(7, 24))


PROFILE_FN = {
    "morning": morning_profile,
    "allday_low": allday_low_profile,
    "night": night_profile,
    "intermittent": intermittent_profile,
    "noise": noise_profile,
}

labels = np.repeat(list(CLUSTER_SIZES), list(CLUSTER_SIZES.values()))
order = rng.permutation(N_DEVICES)  # device_id와 군집이 정렬돼 있지 않도록 섞는다
labels = labels[order]

rows = []
for i, label in enumerate(labels):
    prof = PROFILE_FN[label](rng)

    # night와 intermittent 일부 기기는 상대 군집의 프로파일과 섞는다
    if label in ("night", "intermittent") and rng.random() < BLUR_FRAC:
        other = "intermittent" if label == "night" else "night"
        alpha = rng.uniform(*BLUR_ALPHA)
        prof = alpha * prof + (1 - alpha) * PROFILE_FN[other](rng)

    amp = rng.lognormal(0, AMP_SIGMA)
    prof = prof * amp + rng.normal(0, CELL_NOISE_SD, size=(7, 24))
    prof = np.clip(prof, 0, 10)

    cell_sum = prof.sum()
    total_usage = np.clip(cell_sum * TOTAL_SCALE + rng.normal(0, 30), 0, 5000)
    rows.append((f"D{i:04d}", prof, total_usage, label))

feature_cols = [f"u_{dow}_h{h:02d}" for dow in DOWS for h in HOURS]
df = pd.DataFrame(
    [
        {
            "device_id": did,
            **dict(zip(feature_cols, prof.ravel().round(3))),
            "total_usage": round(total, 1),
            "true_cluster": label,
        }
        for did, prof, total, label in rows
    ]
)
df.to_csv(data_dir / "usage_profiles.csv", index=False)

print(f"usage_profiles.csv 저장 — {len(df)}행 × {len(df.columns)}열")
print(df["true_cluster"].value_counts().to_string())
print(f"집계 피처 범위: {df[feature_cols].values.min():.2f} ~ {df[feature_cols].values.max():.2f}")
print(f"total_usage 범위: {df['total_usage'].min():.0f} ~ {df['total_usage'].max():.0f}")

# 검증 플롯. 요일 축을 평균으로 접어 24시간 프로파일로 본다
hourly = pd.DataFrame(
    np.stack([p.mean(axis=0) for _, p, _, _ in rows]),
    columns=HOURS,
)
hourly["true_cluster"] = labels

TITLES = {
    "morning": "아침집중 (morning)",
    "allday_low": "종일저강도 (allday_low)",
    "night": "야간집중 (night)",
    "intermittent": "간헐고강도 (intermittent)",
}

fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
for ax, (label, title) in zip(axes.ravel(), TITLES.items()):
    sub = hourly[hourly["true_cluster"] == label].drop(columns="true_cluster")
    sample = sub.sample(min(15, len(sub)), random_state=SEED)
    for _, row in sample.iterrows():
        ax.plot(HOURS, row.values, color="tab:gray", alpha=0.3, lw=0.8)
    ax.plot(HOURS, sub.mean().values, color="tab:blue", lw=2.5, label="군집 평균")
    ax.set_title(f"{title} — {len(sub)}대")
    ax.set_xticks(range(0, 24, 4))
    ax.grid(alpha=0.3)
axes[0, 0].legend(loc="upper right")
for ax in axes[1]:
    ax.set_xlabel("시각")
for ax in axes[:, 0]:
    ax.set_ylabel("평균 사용 강도 (요일 평균)")
fig.suptitle("정답 레이블 기준 시간대별 사용 프로파일 (노이즈 소군집 15대 제외)")
fig.tight_layout()
fig.savefig(fig_dir / "fig1_true_profiles.png", dpi=150)
print(f"검증 플롯 저장 — {fig_dir / 'fig1_true_profiles.png'}")

res_dir = base / "outputs" / "results"
res_dir.mkdir(parents=True, exist_ok=True)
df["true_cluster"].value_counts().rename_axis("true_cluster").rename("n").reset_index().to_csv(
    res_dir / "true_cluster_sizes.csv", index=False
)
