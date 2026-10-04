# 가상 구직자 6,000명, 관측 180일 (20대 중후반~30대 초반)
#
# 잠재 유형 4개 + 비활성 (truth_* 는 평가 전용)
#   recent_career   최근 경력이 있는 신입: 경력 6~30개월, 퇴사 12개월 안
#   newgrad         경력 없는 신입: 인턴 정도, 지원 많음
#   mover           이직 준비 경력자: 경력 3년 이상, 공고 조회 많고 지원 적음
#   explorer        탐색형: 조회만, 지원 거의 없음
#   비활성 30%: 위 유형 중 하나로 활동하다 관측 중간에 멈춤
#     truth_outcome = hired(취업 완료) / lapsed(그냥 이탈). 멈춘 뒤 로그는 둘이 같음
#     hired 중 일부만 기업이 채용 진행 상태를 갱신해 offer_event 가 남음
# 입력 빈약 25%: 학력, 경력, 역량검사 칸이 비어 0으로 들어감 (유형과 무관)
#
# 출력: data/jobseekers.csv

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N = 6000
WINDOW = 180
DORMANT_P = 0.30
HIRED_P = 0.55  # 비활성 중 취업 완료
OFFER_EVENT_P = 0.35  # 취업 완료 중 채용 진행 상태가 플랫폼에 남는 비율
SPARSE_P = 0.25

TYPES = {
    # weight, 일 로그인, 로그인당 공고 조회, 로그인당 지원, 경력(개월) 범위, 역량검사 응시율
    "recent_career": (0.22, 0.30, 4.0, 0.12, (6, 30), 0.85),
    "newgrad": (0.32, 0.35, 3.0, 0.20, (0, 5), 0.70),
    "mover": (0.20, 0.20, 6.0, 0.05, (36, 96), 0.30),
    "explorer": (0.26, 0.12, 5.0, 0.01, (0, 24), 0.25),
}

base = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(SEED)

names = list(TYPES)
truth_type = rng.choice(names, size=N, p=[TYPES[t][0] for t in names])
age = rng.integers(25, 34, size=N)

rows = []
for i, t in enumerate(truth_type):
    _, login_rate, view_per, apply_per, (c_lo, c_hi), assess_p = TYPES[t]
    login_rate = login_rate * rng.lognormal(0, 0.35)

    dormant = rng.random() < DORMANT_P
    active_days = int(rng.integers(20, 150)) if dormant else WINDOW
    outcome = ("hired" if rng.random() < HIRED_P else "lapsed") if dormant else "active"

    logins = rng.poisson(login_rate * active_days)
    views = rng.poisson(view_per * logins)
    applies = rng.poisson(apply_per * logins)
    if logins > 0:
        gap = rng.geometric(min(1.0, login_rate)) - 1
        since_last = WINDOW - active_days + gap
    else:
        since_last = WINDOW
    since_last = min(since_last, WINDOW)

    career = int(rng.integers(c_lo, c_hi + 1))
    if t == "recent_career":
        since_job = int(rng.integers(0, 13))
    elif career > 0:
        since_job = int(rng.integers(0, 25))
    else:
        since_job = -1
    education = int(rng.choice([2, 3, 4], p=[0.15, 0.70, 0.15]))  # 2 전문대, 3 대졸, 4 석사
    assessed = rng.random() < assess_p
    score = float(np.clip(rng.normal(60, 12), 0, 100)) if assessed else 0.0
    completeness = float(rng.uniform(0.6, 1.0))

    sparse = rng.random() < SPARSE_P
    if sparse:
        career, since_job, education, assessed, score = 0, -1, 0, False, 0.0
        completeness = float(rng.uniform(0.1, 0.35))

    offer_event = outcome == "hired" and rng.random() < OFFER_EVENT_P

    rows.append(
        {
            "user_id": i,
            "age": int(age[i]),
            "logins_180d": logins,
            "views_180d": views,
            "applies_180d": applies,
            "days_since_login": since_last,
            "career_months": career,
            "months_since_job": since_job,
            "education": education,
            "assessed": int(assessed),
            "assess_score": round(score, 1),
            "profile_completeness": round(completeness, 3),
            "offer_event": int(offer_event),
            "truth_type": t,
            "truth_outcome": outcome,
            "truth_sparse": int(sparse),
        }
    )

df = pd.DataFrame(rows)
(base / "data").mkdir(exist_ok=True)
df.to_csv(base / "data" / "jobseekers.csv", index=False)
print(df["truth_type"].value_counts().to_dict())
print(df["truth_outcome"].value_counts().to_dict())
print(df.groupby("truth_outcome")[["logins_180d", "applies_180d", "days_since_login"]].mean().round(2))
