"""기기 500대의 180일치 일별 로그를 만든다.

심각 이벤트 7일 전부터 W3 발생률이 선형으로 오르고, 14일 전부터는 사용량의 분산만 커지며 평균은 그대로다.
여기까지가 진짜 전조다. 그리고 기기 12%는 W7이 상시 높지만 심각 이벤트와는 무관하다. 모델이 물어야 할
함정으로 넣었다. 교란군과 심각 기기는 겹치지 않게 뽑았는데, 겹치면 W7이 진짜 신호처럼 보여서 함정이 함정이
아니게 된다. 심각 이벤트는 기기당 한 번이고, 그날 이후의 행은 03이 잘라낸다.
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_DEVICES = 500
N_DAYS = 180
START = pd.Timestamp("2025-01-01")

SEVERE_FRAC = 0.08  # 심각 이벤트 경험 기기 비율
CONFOUNDER_FRAC = 0.12  # 교란용(W7 상시 높음) 기기 비율, 심각 기기와 겹치지 않음
MIN_EVENT_DAY = 30  # 전조 구간(최대 14일)이 온전히 들어가도록 여유

# 경고 코드별 기본 발생률, 일 평균 건수. W3와 W7에 신호가 얹힌다
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

# 신호 (1): W3 발생률에 얹는 선형 램프. 7일 전 +0.5에서 당일 +4.0까지
W3_RAMP_DAYS = 7
W3_RAMP_START = 0.5
W3_RAMP_END = 4.0

# 신호 (2): 사용량 노이즈의 lognormal sigma를 14일 전부터 선형으로 키운다
USAGE_SIGMA_BASE = 0.08
USAGE_SIGMA_PEAK = 0.60
USAGE_VAR_DAYS = 14

# 신호 (3): 교란 기기군의 W7 발생률 범위
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

# 심각 이벤트 발생일 (기기당 1회)
event_day = np.full(N_DEVICES, -1)
event_day[severe_idx] = rng.integers(MIN_EVENT_DAY, N_DAYS, size=n_severe)

usage_mu = np.clip(rng.normal(30, 8, size=N_DEVICES), 10, 60)
warn_mult = rng.lognormal(0, 0.3, size=(N_DEVICES, len(WARNING_BASE_RATES)))

# 사용량은 Poisson(mu * eps), eps는 평균이 1이 되게 보정한 lognormal. sigma만 키우면 평균은 그대로고 분산만 커진다
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
        # 신호 (1): 이벤트 7일 전부터 선형 램프
        for i in severe_idx:
            t = event_day[i]
            for d in range(max(0, t - W3_RAMP_DAYS), t + 1):
                frac = 1 - (t - d) / W3_RAMP_DAYS
                lam[i, d] += W3_RAMP_START + (W3_RAMP_END - W3_RAMP_START) * frac

    if code == "W7":
        high = rng.uniform(*CONFOUNDER_W7_RANGE, size=n_confounder)
        lam[confounder_idx, :] = high[:, None]

    warning_counts[code] = rng.poisson(lam)

# count > 0인 행만 남긴다. 없는 날은 뒤에서 0으로 채운다
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
