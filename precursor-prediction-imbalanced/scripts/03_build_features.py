"""device-day 단위 feature 테이블. 각 기기의 각 날짜가 한 행이다.

feature는 그날을 포함한 과거 7일과 14일의 trailing window 집계이고, 타깃은 다음 날부터 7일 안에 심각 이벤트가
있는지다. 누수를 막기 위한 설계 네 가지:
- feature window는 t 이하만 쓴다. t-13에서 t까지
- 타깃 window는 t 초과만 쓴다. t+1에서 t+7까지. 두 구간이 겹치지 않는다
- 심각 이벤트 발생 이후의 행은 버린다. 이벤트 뒤의 데이터로 이벤트를 "예측"하는 누수를 막기 위해서다
- 타깃 window가 관측 기간을 벗어나는 마지막 7일 행도 버린다. 라벨이 불완전하기 때문이다
"""

from pathlib import Path

import numpy as np
import pandas as pd

WARNING_CODES = [f"W{i}" for i in range(1, 10)]
WINDOWS = (7, 14)
MAX_WINDOW = max(WINDOWS)
TARGET_HORIZON = 7  # t+1 ~ t+7

base = Path(__file__).resolve().parents[1]
df = pd.read_csv(base / "data" / "device_events.csv", parse_dates=["event_date"])
gt = pd.read_csv(base / "data" / "ground_truth.csv", parse_dates=["severe_event_date"])

start = df["event_date"].min()
n_days = (df["event_date"].max() - start).days + 1
df["day"] = (df["event_date"] - start).dt.days
gt["event_day"] = (gt["severe_event_date"] - start).dt.days  # 없으면 NaN

device_ids = gt["device_id"].to_numpy()
n_devices = len(device_ids)


def daily_matrix(code: str) -> np.ndarray:
    """기기 × 날짜 (n_devices, n_days) 그리드. 없는 날은 0건."""
    sub = df[df["event_code"] == code]
    wide = sub.pivot_table(index="device_id", columns="day", values="count", aggfunc="sum")
    return wide.reindex(index=device_ids, columns=range(n_days)).fillna(0).to_numpy()


def trailing(mat: np.ndarray, w: int) -> np.ndarray:
    """트레일링 윈도우 뷰: 결과 [:, t]는 원본 [:, t-w+1 : t+1]. t < w-1 구간은 NaN."""
    view = np.lib.stride_tricks.sliding_window_view(mat, w, axis=1).astype(float)
    pad = np.full((mat.shape[0], w - 1, w), np.nan)
    return np.concatenate([pad, view], axis=1)  # (n_devices, n_days, w)


def slope(win: np.ndarray) -> np.ndarray:
    """윈도우 내 일별 값의 최소제곱 기울기 (하루당 증가량)."""
    w = win.shape[-1]
    x = np.arange(w) - (w - 1) / 2
    return (win * x).sum(axis=-1) / (x**2).sum()


features: dict[str, np.ndarray] = {}  # 각 값은 (n_devices, n_days)

u1 = daily_matrix("U1")
for w in WINDOWS:
    win = trailing(u1, w)
    features[f"u1_mean_{w}d"] = win.mean(axis=-1)
    features[f"u1_std_{w}d"] = win.std(axis=-1, ddof=1)
    features[f"u1_slope_{w}d"] = slope(win)

for code in WARNING_CODES:
    mat = daily_matrix(code)
    for w in WINDOWS:
        features[f"{code.lower()}_cnt_{w}d"] = trailing(mat, w).sum(axis=-1)
    features[f"{code.lower()}_slope_7d"] = slope(trailing(mat, 7))

event_day = gt["event_day"].to_numpy()  # (n_devices,) NaN이면 이벤트 없음
days = np.arange(n_days)
with np.errstate(invalid="ignore"):
    target = ((event_day[:, None] > days[None, :]) & (event_day[:, None] <= days[None, :] + TARGET_HORIZON)).astype(
        int
    )  # NaN 비교는 False라 이벤트 없는 기기는 전부 0

valid = np.ones((n_devices, n_days), dtype=bool)
valid[:, : MAX_WINDOW - 1] = False  # 14일 윈도우가 안 차는 초기 구간
valid[:, n_days - TARGET_HORIZON :] = False  # 타깃 윈도우가 잘리는 마지막 7일
has_event = ~np.isnan(event_day)
post_event = has_event[:, None] & (days[None, :] >= np.nan_to_num(event_day, nan=np.inf)[:, None])
valid &= ~post_event  # 이벤트 발생일 이후 행 제거

dev_idx, day_idx = np.nonzero(valid)
table = pd.DataFrame(
    {
        "device_id": device_ids[dev_idx],
        "event_date": start + pd.to_timedelta(day_idx, unit="D"),
        "day": day_idx,
    }
)
for name, mat in features.items():
    table[name] = mat[dev_idx, day_idx]
table["target"] = target[dev_idx, day_idx]

assert not table.isna().any().any(), "피처에 NaN이 남아 있음"
out = base / "data" / "features.csv"
table.to_csv(out, index=False)

n_pos = int(table["target"].sum())
print(f"rows: {len(table):,} (device-day), features: {len(features)}")
print(f"positive: {n_pos:,} / negative: {len(table) - n_pos:,}")
print(f"타깃 불균형 비율: {n_pos / len(table):.4%} (1 : {(len(table) - n_pos) / n_pos:.0f})")
print(f"positive 기기 수: {table.loc[table['target'] == 1, 'device_id'].nunique()}")
print(f"saved: {out}")

res_dir = base / "outputs" / "results"
res_dir.mkdir(parents=True, exist_ok=True)
pd.DataFrame(
    {
        "metric": ["n_rows", "n_features", "n_positive", "positive_rate", "imbalance_ratio", "n_positive_devices"],
        "value": [
            len(table),
            len(features),
            n_pos,
            n_pos / len(table),
            (len(table) - n_pos) / n_pos,
            table.loc[table["target"] == 1, "device_id"].nunique(),
        ],
    }
).round(4).to_csv(res_dir / "feature_table_stats.csv", index=False)
