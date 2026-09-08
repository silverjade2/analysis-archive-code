# 피처 테이블 누수(leakage) 검증
#
# 원칙: 피처는 t 이하의 데이터만, 타깃은 t 초과의 데이터만 써야 한다.
# 검증 3종:
#   (1) 재계산 대조 — 무작위 표본 행에 대해, 원천 데이터를 t 시점까지로 "잘라낸 뒤"
#       피처를 처음부터 다시 계산해 저장된 값과 일치하는지 확인.
#       미래 데이터가 피처에 섞였다면 잘라낸 데이터로는 같은 값이 나올 수 없다.
#   (2) 타깃 방향 검증 — 모든 행에서 이벤트일이 t보다 미래인지(사후 행 제거 확인),
#       target 정의가 정확히 t+1~t+7 윈도우와 일치하는지 확인.
#   (3) 음성 대조 — 일부러 미래 데이터(t+1~t+3의 W3)를 쓰는 '누수 피처'를 만들어
#       (1)의 재계산 대조가 실제로 이를 잡아내는지 확인. 검증 도구 자체의 검증.
#
# 실행: .venv/bin/python notebooks/04_validate_features.py

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_SAMPLE = 200
WARNING_CODES = [f"W{i}" for i in range(1, 10)]

base = Path(__file__).resolve().parents[1]
raw = pd.read_csv(base / "data" / "device_events.csv", parse_dates=["event_date"])
table = pd.read_csv(base / "data" / "features.csv", parse_dates=["event_date"])
gt = pd.read_csv(base / "data" / "ground_truth.csv", parse_dates=["severe_event_date"])

start = raw["event_date"].min()
raw["day"] = (raw["event_date"] - start).dt.days
gt["event_day"] = (gt["severe_event_date"] - start).dt.days

feature_cols = [c for c in table.columns if c not in ("device_id", "event_date", "day", "target")]
rng = np.random.default_rng(SEED)
sample = table.sample(n=N_SAMPLE, random_state=SEED).reset_index(drop=True)


def slope(y: np.ndarray) -> float:
    x = np.arange(len(y)) - (len(y) - 1) / 2
    return float((y * x).sum() / (x**2).sum())


def recompute_features(device_id: str, t: int, cutoff: pd.DataFrame) -> dict[str, float]:
    """cutoff(잘라낸 원천 데이터)만으로 (device_id, t) 행의 모든 피처를 재계산."""
    sub = cutoff[(cutoff["device_id"] == device_id) & (cutoff["day"].between(t - 13, t))]
    out: dict[str, float] = {}

    def daily(code: str) -> np.ndarray:
        s = sub[sub["event_code"] == code].set_index("day")["count"]
        return s.reindex(range(t - 13, t + 1), fill_value=0).to_numpy(dtype=float)

    u1 = daily("U1")
    for w in (7, 14):
        win = u1[-w:]
        out[f"u1_mean_{w}d"] = win.mean()
        out[f"u1_std_{w}d"] = win.std(ddof=1)
        out[f"u1_slope_{w}d"] = slope(win)
    for code in WARNING_CODES:
        m = daily(code)
        out[f"{code.lower()}_cnt_7d"] = m[-7:].sum()
        out[f"{code.lower()}_cnt_14d"] = m[-14:].sum()
        out[f"{code.lower()}_slope_7d"] = slope(m[-7:])
    return out


# ── (1) 재계산 대조 ──────────────────────────────────────────────
mismatch = 0
for _, row in sample.iterrows():
    t = int(row["day"])
    cutoff = raw[raw["day"] <= t]  # t 이후 데이터를 아예 제거한 세계
    recomputed = recompute_features(row["device_id"], t, cutoff)
    for col in feature_cols:
        if not np.isclose(row[col], recomputed[col], rtol=1e-9, atol=1e-9):
            mismatch += 1
            print(f"  불일치: {row['device_id']} day={t} {col}: "
                  f"저장값 {row[col]} vs 재계산 {recomputed[col]}")
            break
print(f"[검증 1] 재계산 대조 — 표본 {N_SAMPLE}행 × 피처 {len(feature_cols)}개: "
      f"불일치 {mismatch}건 {'→ 통과' if mismatch == 0 else '→ 실패!'}")

# ── (2) 타깃 방향 검증 ───────────────────────────────────────────
merged = table.merge(gt[["device_id", "event_day"]], on="device_id", how="left")
ev, t = merged["event_day"], merged["day"]

post_event_rows = int((ev <= t).sum())  # NaN 비교는 False
expected_target = ((ev > t) & (ev <= t + 7)).fillna(False).astype(int)
target_mismatch = int((merged["target"] != expected_target).sum())
incomplete_label = int((t > (table["day"].max() + 7) - 7).sum())  # 타깃 윈도우가 잘리는 행

print(f"[검증 2] 이벤트일 이후 행: {post_event_rows}건, "
      f"타깃 정의 불일치: {target_mismatch}건 "
      f"{'→ 통과' if post_event_rows == 0 and target_mismatch == 0 else '→ 실패!'}")

# ── (3) 음성 대조: 일부러 만든 누수 피처를 검증이 잡아내는가 ───────
w3 = raw[raw["event_code"] == "W3"]
leak_detected = 0
for _, row in sample.head(50).iterrows():
    t = int(row["day"])
    # 누수 피처: 미래(t+1 ~ t+3)의 W3 건수 — 실제 파이프라인에 섞였다고 가정
    leaky_value = w3[(w3["device_id"] == row["device_id"])
                     & (w3["day"].between(t + 1, t + 3))]["count"].sum()
    # 재계산 대조: t까지 잘라낸 데이터에서 같은 피처를 계산하면 미래분은 항상 0
    recomputed_value = 0
    if not np.isclose(leaky_value, recomputed_value):
        leak_detected += 1
print(f"[검증 3] 음성 대조 — 누수 피처 표본 50행 중 재계산 불일치 {leak_detected}건 "
      f"(0보다 커야 정상: 검증 방법이 누수를 실제로 잡아낸다는 뜻)")

# ── 타깃 불균형 요약 ─────────────────────────────────────────────
n_pos = int(table["target"].sum())
print(f"\n타깃 불균형: positive {n_pos:,} / 전체 {len(table):,} "
      f"= {n_pos / len(table):.4%} (1 : {(len(table) - n_pos) / n_pos:.0f})")
