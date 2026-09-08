# 1호 글용 가상데이터 생성 — device_events
#
# 스키마: device_events(device_id, event_date, event_type, event_code, count)
# 규모: 기기 500대 × 180일, 심각 이벤트(S1) 경험 기기 8%
#
# 심는 신호 3종:
#   (1) 심각 이벤트 7일 전부터 W3 경고 발생률이 선형 증가
#   (2) 심각 이벤트 14일 전부터 사용량(U1)의 분산이 증가 (평균은 유지)
#   (3) 심각 이벤트와 무관한 기기군(교란용)의 W7이 상시 높음
#
# 실행: .venv/bin/python notebooks/01_generate_device_events.py
# 출력: data/device_events.csv, data/ground_truth.csv

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_DEVICES = 500
N_DAYS = 180
START = pd.Timestamp("2025-01-01")

SEVERE_FRAC = 0.08        # 심각 이벤트 경험 기기 비율
CONFOUNDER_FRAC = 0.12    # 교란용(W7 상시 높음) 기기 비율 — 심각 기기와 겹치지 않음
MIN_EVENT_DAY = 30        # 전조 구간(최대 14일)이 온전히 들어가도록 여유를 둔다

# 경고 코드별 기본 발생률(일 평균 건수). W3/W7은 신호가 얹히는 코드.
WARNING_BASE_RATES = {
    "W1": 0.40, "W2": 0.25, "W3": 0.30, "W4": 0.20, "W5": 0.15,
    "W6": 0.10, "W7": 0.30, "W8": 0.12, "W9": 0.08,
}

# 신호 (1): W3 발생률에 얹는 선형 램프 (7일 전 +0.5 → 당일 +4.0)
W3_RAMP_DAYS = 7
W3_RAMP_START = 0.5
W3_RAMP_END = 4.0

# 신호 (2): 사용량 노이즈(lognormal sigma)를 14일 전부터 선형 증가
USAGE_SIGMA_BASE = 0.08
USAGE_SIGMA_PEAK = 0.60
USAGE_VAR_DAYS = 14

# 신호 (3): 교란 기기군의 W7 발생률 범위
CONFOUNDER_W7_RANGE = (2.5, 4.0)

rng = np.random.default_rng(SEED)
out_dir = Path(__file__).resolve().parents[1] / "data"
out_dir.mkdir(parents=True, exist_ok=True)

# ── 기기 그룹 배정 ────────────────────────────────────────────────
device_ids = np.array([f"D{i:04d}" for i in range(N_DEVICES)])
n_severe = int(N_DEVICES * SEVERE_FRAC)
n_confounder = int(N_DEVICES * CONFOUNDER_FRAC)

shuffled = rng.permutation(N_DEVICES)
severe_idx = shuffled[:n_severe]
confounder_idx = shuffled[n_severe:n_severe + n_confounder]

group = np.full(N_DEVICES, "normal", dtype=object)
group[severe_idx] = "severe"
group[confounder_idx] = "confounder"

# 심각 이벤트 발생일 (기기당 1회)
event_day = np.full(N_DEVICES, -1)
event_day[severe_idx] = rng.integers(MIN_EVENT_DAY, N_DAYS, size=n_severe)

# ── 기기별 베이스라인 ─────────────────────────────────────────────
usage_mu = np.clip(rng.normal(30, 8, size=N_DEVICES), 10, 60)          # U1 일 평균
warn_mult = rng.lognormal(0, 0.3, size=(N_DEVICES, len(WARNING_BASE_RATES)))

# ── 사용량(U1): Poisson(mu * eps), eps ~ lognormal(평균 1 보정) ────
# 신호 (2): 심각 기기의 이벤트 전 14일 동안 sigma를 선형으로 키운다 → 분산만 증가
sigma = np.full((N_DEVICES, N_DAYS), USAGE_SIGMA_BASE)
for i in severe_idx:
    t = event_day[i]
    for d in range(max(0, t - USAGE_VAR_DAYS), t + 1):
        frac = 1 - (t - d) / USAGE_VAR_DAYS
        sigma[i, d] = USAGE_SIGMA_BASE + (USAGE_SIGMA_PEAK - USAGE_SIGMA_BASE) * frac

eps = rng.lognormal(mean=-sigma**2 / 2, sigma=sigma)
usage_counts = rng.poisson(usage_mu[:, None] * eps)

# ── 경고(W1~W9): Poisson(기기별 발생률) ───────────────────────────
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
        # 신호 (3): 교란 기기군은 W7이 상시 높다 (이벤트와 무관)
        high = rng.uniform(*CONFOUNDER_W7_RANGE, size=n_confounder)
        lam[confounder_idx, :] = high[:, None]

    warning_counts[code] = rng.poisson(lam)

# ── long 포맷으로 변환 (count > 0 행만 기록) ──────────────────────
dates = START + pd.to_timedelta(np.arange(N_DAYS), unit="D")

def to_long(counts: np.ndarray, event_type: str, event_code: str) -> pd.DataFrame:
    dev, day = np.nonzero(counts)
    return pd.DataFrame({
        "device_id": device_ids[dev],
        "event_date": dates[day],
        "event_type": event_type,
        "event_code": event_code,
        "count": counts[dev, day],
    })

frames = [to_long(usage_counts, "usage", "U1")]
frames += [to_long(c, "warning", code) for code, c in warning_counts.items()]

severe_frame = pd.DataFrame({
    "device_id": device_ids[severe_idx],
    "event_date": dates[event_day[severe_idx]],
    "event_type": "severe",
    "event_code": "S1",
    "count": 1,
})
frames.append(severe_frame)

df = (
    pd.concat(frames, ignore_index=True)
    .sort_values(["device_id", "event_date", "event_type", "event_code"])
    .reset_index(drop=True)
)
df.to_csv(out_dir / "device_events.csv", index=False)

# 정답지 — 검증 플롯과 이후 성능 평가용 (모델 입력으로는 쓰지 않는다)
ground_truth = pd.DataFrame({
    "device_id": device_ids,
    "group": group,
    "severe_event_date": [dates[d].date() if d >= 0 else None for d in event_day],
})
ground_truth.to_csv(out_dir / "ground_truth.csv", index=False)

# ── 요약 ─────────────────────────────────────────────────────────
print(f"rows: {len(df):,}")
print(f"devices: {df['device_id'].nunique()} / days: {N_DAYS} ({dates[0].date()} ~ {dates[-1].date()})")
print(f"groups: severe={n_severe} ({n_severe / N_DEVICES:.0%}), "
      f"confounder={n_confounder}, normal={N_DEVICES - n_severe - n_confounder}")
print(df.groupby("event_type", observed=True)["count"].agg(["count", "sum"]))
print(f"saved: {out_dir / 'device_events.csv'}")
