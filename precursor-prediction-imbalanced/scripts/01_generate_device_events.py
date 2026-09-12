"""기기 500대 x 180일 일별 로그 생성

- 진짜 전조 2개: 이벤트 7일 전부터 W3 램프, 14일 전부터 사용량 분산만 증가 (평균 유지)
- 함정: 기기 12%는 W7 상시 높음, 이벤트와 무관. severe 기기와 안 겹치게 뽑음 (겹치면 함정이 아니게 됨)
- 심각 이벤트는 기기당 1회. 이후 행은 03에서 제거
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_DEVICES = 500
N_DAYS = 180
START = pd.Timestamp("2025-01-01")

SEVERE_FRAC = 0.08
CONFOUNDER_FRAC = 0.12  # W7 상시 높음 군. severe와 안 겹침
MIN_EVENT_DAY = 30  # 전조 14일이 다 들어가게

# 일 평균 건수. W3, W7에 신호 얹힘
WARNING_BASE_RATES = {
    "W1": 0.40,
    "W2": 0.25,
    "W3": 0.30,
    "W4": 0.20,
    "W5": 0.15,
    "W6": 0.10,
    "W7": 0.30,
    "W8": 0.12,
    "W9": 0.08,
}

# 신호 1. W3 램프, -7일 +0.5 -> 당일 +4.0
W3_RAMP_DAYS = 7
W3_RAMP_START = 0.5
W3_RAMP_END = 4.0

# 신호 2. 사용량 lognormal sigma, -14일부터 선형 증가
USAGE_SIGMA_BASE = 0.08
USAGE_SIGMA_PEAK = 0.60
USAGE_VAR_DAYS = 14

# 함정. 교란군 W7 발생률
CONFOUNDER_W7_RANGE = (2.5, 4.0)

rng = np.random.default_rng(SEED)
out_dir = Path(__file__).resolve().parents[1] / "data"
out_dir.mkdir(parents=True, exist_ok=True)

device_ids = np.array([f"D{i:04d}" for i in range(N_DEVICES)])
n_severe = int(N_DEVICES * SEVERE_FRAC)
n_confounder = int(N_DEVICES * CONFOUNDER_FRAC)

shuffled = rng.permutation(N_DEVICES)
severe_idx = shuffled[:n_severe]
confounder_idx = shuffled[n_severe : n_severe + n_confounder]

group = np.full(N_DEVICES, "normal", dtype=object)
group[severe_idx] = "severe"
group[confounder_idx] = "confounder"

event_day = np.full(N_DEVICES, -1)
event_day[severe_idx] = rng.integers(MIN_EVENT_DAY, N_DAYS, size=n_severe)

usage_mu = np.clip(rng.normal(30, 8, size=N_DEVICES), 10, 60)
warn_mult = rng.lognormal(0, 0.3, size=(N_DEVICES, len(WARNING_BASE_RATES)))

# eps는 평균 1로 보정한 lognormal. sigma만 키우면 분산만 커짐
sigma = np.full((N_DEVICES, N_DAYS), USAGE_SIGMA_BASE)
for i in severe_idx:
    t = event_day[i]
    for d in range(max(0, t - USAGE_VAR_DAYS), t + 1):
        frac = 1 - (t - d) / USAGE_VAR_DAYS
        sigma[i, d] = USAGE_SIGMA_BASE + (USAGE_SIGMA_PEAK - USAGE_SIGMA_BASE) * frac

eps = rng.lognormal(mean=-(sigma**2) / 2, sigma=sigma)
usage_counts = rng.poisson(usage_mu[:, None] * eps)

warning_counts = {}
for j, (code, base) in enumerate(WARNING_BASE_RATES.items()):
    lam = np.full((N_DEVICES, N_DAYS), base) * warn_mult[:, j][:, None]

    if code == "W3":
        for i in severe_idx:
            t = event_day[i]
            for d in range(max(0, t - W3_RAMP_DAYS), t + 1):
                frac = 1 - (t - d) / W3_RAMP_DAYS
                lam[i, d] += W3_RAMP_START + (W3_RAMP_END - W3_RAMP_START) * frac

    if code == "W7":
        high = rng.uniform(*CONFOUNDER_W7_RANGE, size=n_confounder)
        lam[confounder_idx, :] = high[:, None]

    warning_counts[code] = rng.poisson(lam)

# count > 0인 행만. 빈 날은 03에서 0 채움
dates = START + pd.to_timedelta(np.arange(N_DAYS), unit="D")


def to_long(counts: np.ndarray, event_type: str, event_code: str) -> pd.DataFrame:
    dev, day = np.nonzero(counts)
    return pd.DataFrame(
        {
            "device_id": device_ids[dev],
            "event_date": dates[day],
            "event_type": event_type,
            "event_code": event_code,
            "count": counts[dev, day],
        }
    )


frames = [to_long(usage_counts, "usage", "U1")]
frames += [to_long(c, "warning", code) for code, c in warning_counts.items()]

severe_frame = pd.DataFrame(
    {
        "device_id": device_ids[severe_idx],
        "event_date": dates[event_day[severe_idx]],
        "event_type": "severe",
        "event_code": "S1",
        "count": 1,
    }
)
frames.append(severe_frame)

df = (
    pd.concat(frames, ignore_index=True)
    .sort_values(["device_id", "event_date", "event_type", "event_code"])
    .reset_index(drop=True)
)
df.to_csv(out_dir / "device_events.csv", index=False)

ground_truth = pd.DataFrame(
    {
        "device_id": device_ids,
        "group": group,
        "severe_event_date": [dates[d].date() if d >= 0 else None for d in event_day],
    }
)
ground_truth.to_csv(out_dir / "ground_truth.csv", index=False)

print(f"rows: {len(df):,}")
print(f"devices: {df['device_id'].nunique()} / days: {N_DAYS} ({dates[0].date()} ~ {dates[-1].date()})")
print(
    f"groups: severe={n_severe} ({n_severe / N_DEVICES:.0%}), "
    f"confounder={n_confounder}, normal={N_DEVICES - n_severe - n_confounder}"
)
print(df.groupby("event_type", observed=True)["count"].agg(["count", "sum"]))
print(f"saved: {out_dir / 'device_events.csv'}")
