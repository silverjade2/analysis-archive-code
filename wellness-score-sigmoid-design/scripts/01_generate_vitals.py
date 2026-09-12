"""합성 1000명 + 엣지 케이스 생성

- 항목별 독립. 정상 70 / 경계~임계치 20 / 임계치 밖 8 / 결측 2 (%)
- 실제 바이탈 분포 아님. 점수 함수 스트레스 테스트용
"""

import numpy as np
import pandas as pd
from common import DATA, DECIMALS, ITEMS, N_USERS, PARAMS, rng

ZONE_P = {"normal": 0.70, "caution": 0.20, "warning": 0.08, "missing": 0.02}


def sample_item(g, p, n, nd):
    width = p["normal_high"] - p["normal_low"]
    zone = g.choice(list(ZONE_P), size=n, p=list(ZONE_P.values()))
    x = np.full(n, np.nan)

    m = zone == "normal"
    x[m] = g.uniform(p["normal_low"], p["normal_high"], m.sum())

    m = zone == "caution"
    low_ok = p["normal_low"] > p["lower_limit"]
    high_ok = p["upper_limit"] > p["normal_high"]
    side = g.random(m.sum()) < (0.5 if low_ok and high_ok else float(low_ok))
    lo = g.uniform(p["lower_limit"], p["normal_low"], m.sum())
    hi = g.uniform(p["normal_high"], p["upper_limit"], m.sum())
    x[m] = np.where(side, lo, hi)

    m = zone == "warning"  # spo2 >100, stress <0 은 불가라 반대편에 몰아줌
    low_ok = p["lower_limit"] > 0 or p["normal_low"] > p["lower_limit"]
    high_ok = p["upper_limit"] > p["normal_high"]
    side = g.random(m.sum()) < (0.5 if low_ok and high_ok else float(low_ok))
    lo = g.uniform(p["lower_limit"] - 0.5 * width, p["lower_limit"], m.sum())
    hi = g.uniform(p["upper_limit"], p["upper_limit"] + 0.5 * width, m.sum())
    x[m] = np.where(side, lo, hi)

    return np.round(x, nd)  # 반올림으로 구간 넘어가는 값 있음, 상태는 03이 다시 판정


g = rng()
users = pd.DataFrame({"user_id": [f"U{i:04d}" for i in range(1, N_USERS + 1)]})
for item in ITEMS:
    users[item] = sample_item(g, PARAMS[item], N_USERS, DECIMALS[item])
users.to_csv(DATA / "synthetic_vitals.csv", index=False)

mid = {i: (PARAMS[i]["normal_low"] + PARAMS[i]["normal_high"]) / 2 for i in ITEMS}


def case(name, expect, **over):
    row = {"case": name, "expected": expect, **mid}
    row.update(over)
    return row


edge = pd.DataFrame(
    [
        case("정상 경계 안쪽", "항목 점수 100", temperature=37.5),
        case("정상 경계 바로 밖", "재설계 항목 점수 99 이상, 원본은 50 근처", temperature=37.51),
        case("임계치 일치", "항목 상태 주의, 재설계 항목 점수 5 근처", spo2=90.0),
        case("임계치 초과 극단", "항목 점수 1 미만, 최종 단계 경고", temperature=42.0, spo2=70.0),
        case("결측 1개", "missing_count 1, 5항목으로 점수 계산", stress=np.nan),
        case("전부 결측", "데이터 없음", **{i: np.nan for i in ITEMS}),
        case("상방 이탈 불가 항목", "항목 점수 0", spo2=101.0),
        case("단일 항목 경고", "점수 단계는 안정, 최종 단계 경고", spo2=88.0),
        case("주의만 둘", "원본 보정 후 경고, 최종 단계 주의", pulse=105, bp_dia=91),
    ]
)
edge.to_csv(DATA / "edge_cases.csv", index=False)

print(users.describe().T[["count", "min", "max"]])
print(len(edge), "edge cases")
