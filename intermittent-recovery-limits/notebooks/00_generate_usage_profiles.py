# 2호 글용 가상데이터 생성 — 사용 패턴 세그멘테이션(usage_profiles)
#
# 스키마: usage_profiles(device_id, u_{dow}_h{HH} ×168, total_usage, true_cluster)
#   - u_{dow}_h{HH}: 요일(dow)×시간대(HH) 평균 사용 강도, 0~10 스케일 (집계 피처 168개)
#   - total_usage:   전체 사용량 합산 지표, 0~5000 스케일  ← 함정 1 (혼자 스케일이 큼)
#   - true_cluster:  정답 레이블. 평가 전용이며 학습(클러스터링)에는 사용하지 않는다.
#
# 진짜 군집 4개 + 함정 3종:
#   morning      아침집중   — 평일 아침 6~10시 피크
#   allday_low   종일저강도 — 하루 종일 낮은 강도로 고르게 사용
#   night        야간집중   — 21시~새벽 2시 피크
#   intermittent 간헐고강도 — 평소 거의 0, 특정 시간대에만 고강도 버스트
#   (함정 1) total_usage만 0~5000으로 스케일이 큼 → 스케일링 없이 돌리면 이 축이 지배
#   (함정 2) night ↔ intermittent 일부 기기의 프로파일을 섞어 경계를 흐림 → 엘보우 애매
#   (함정 3) 15대짜리 노이즈성 소군집(24시간 상시 고강도) 추가
#
# 실행: .venv/bin/python notebooks/00_generate_usage_profiles.py
# 출력: data/usage_profiles.csv
#       outputs/figures/fig0_true_profiles.png (정답 레이블 기준 검증 플롯 4개)

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["font.family"] = "AppleGothic"
matplotlib.rcParams["axes.unicode_minus"] = False

SEED = 42
N_DEVICES = 800

# 군집 크기 — 합계 800 (노이즈 소군집 15대 포함)
CLUSTER_SIZES = {
    "morning": 230,
    "allday_low": 240,
    "night": 165,
    "intermittent": 150,
    "noise": 15,
}

DOWS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
HOURS = np.arange(24)

BLUR_FRAC = 0.35          # 함정 2: night/intermittent에서 프로파일을 섞을 기기 비율
BLUR_ALPHA = (0.45, 0.75) # 자기 군집 템플릿의 혼합 가중치 범위

CELL_NOISE_SD = 0.45      # 셀 단위 관측 노이즈
AMP_SIGMA = 0.15          # 기기별 전체 강도 편차 (lognormal sigma)
TOTAL_SCALE = 3.5         # total_usage = 셀 합 × TOTAL_SCALE + 노이즈

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
    """간헐고강도: 기기마다 2~4개 버스트 시간대, 요일별 가동률이 들쭉날쭉.

    버스트 시간대를 저녁(17~24시)에 치우치게 뽑아 night과의 경계를
    한 번 더 흐린다 (함정 2 보조).
    """
    n_burst = r.integers(2, 5)
    hour_weights = np.ones(24)
    hour_weights[17:24] = 4.0
    hour_weights /= hour_weights.sum()
    burst_hours = r.choice(24, size=n_burst, replace=False, p=hour_weights)

    hourly = np.full(24, 0.15)
    hourly[burst_hours] = r.uniform(7.0, 9.0, size=n_burst)
    active_frac = r.uniform(0.2, 0.7, size=7)  # 요일별 가동률 → 집계값이 깎임
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

# ── 기기별 프로파일 생성 ──────────────────────────────────────────
labels = np.repeat(list(CLUSTER_SIZES), list(CLUSTER_SIZES.values()))
order = rng.permutation(N_DEVICES)  # device_id와 군집이 정렬돼 있지 않도록 섞는다
labels = labels[order]

rows = []
for i, label in enumerate(labels):
    prof = PROFILE_FN[label](rng)

    # 함정 2: night ↔ intermittent 일부 기기는 상대 군집 프로파일과 혼합
    if label in ("night", "intermittent") and rng.random() < BLUR_FRAC:
        other = "intermittent" if label == "night" else "night"
        alpha = rng.uniform(*BLUR_ALPHA)
        prof = alpha * prof + (1 - alpha) * PROFILE_FN[other](rng)

    amp = rng.lognormal(0, AMP_SIGMA)  # 기기별 전체 강도 편차
    prof = prof * amp + rng.normal(0, CELL_NOISE_SD, size=(7, 24))
    prof = np.clip(prof, 0, 10)

    cell_sum = prof.sum()
    total_usage = np.clip(cell_sum * TOTAL_SCALE + rng.normal(0, 30), 0, 5000)
    rows.append((f"D{i:04d}", prof, total_usage, label))

feature_cols = [f"u_{dow}_h{h:02d}" for dow in DOWS for h in HOURS]
df = pd.DataFrame(
    [
        {"device_id": did, **dict(zip(feature_cols, prof.ravel().round(3))),
         "total_usage": round(total, 1), "true_cluster": label}
        for did, prof, total, label in rows
    ]
)
df.to_csv(data_dir / "usage_profiles.csv", index=False)

print(f"usage_profiles.csv 저장 — {len(df)}행 × {len(df.columns)}열")
print(df["true_cluster"].value_counts().to_string())
print(f"집계 피처 범위: {df[feature_cols].values.min():.2f} ~ {df[feature_cols].values.max():.2f}")
print(f"total_usage 범위: {df['total_usage'].min():.0f} ~ {df['total_usage'].max():.0f}")

# ── 검증: 정답 레이블 기준 시간대별 프로파일 플롯 4개 ────────────
# 요일 축을 평균으로 접어 24시간 프로파일로 본다.
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
fig.savefig(fig_dir / "fig0_true_profiles.png", dpi=150)
print(f"검증 플롯 저장 — {fig_dir / 'fig0_true_profiles.png'}")
