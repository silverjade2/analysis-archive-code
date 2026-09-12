"""경로, 항목 파라미터, 단계 구간

- PARAMS는 원본 params.json이 아님 (파일 없음). 설계 문서 표 + 테스트 출력에서 역산한 값
- pulse normal_high 100 (표는 120인데 115가 주의로 찍혔음), spo2 weight 20% (spo2만 0점일 때 종합 80.00)
- 나머지 weight 배분은 임의
"""

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RES = ROOT / "outputs" / "results"
FIG = ROOT / "outputs" / "figures"
for p in (DATA, RES, FIG):
    p.mkdir(parents=True, exist_ok=True)

SEED = 42
N_USERS = 1000

ITEMS = ["temperature", "pulse", "spo2", "stress", "bp_sys", "bp_dia"]
PARAMS = {
    "temperature": dict(normal_low=36.0, normal_high=37.5, lower_limit=35.0, upper_limit=40.0, steepness=5, weight=1.5),
    "pulse": dict(normal_low=50, normal_high=100, lower_limit=40, upper_limit=145, steepness=4, weight=1.5),
    "spo2": dict(normal_low=95.0, normal_high=100.0, lower_limit=90.0, upper_limit=100.0, steepness=6, weight=2.0),
    "stress": dict(normal_low=0, normal_high=60, lower_limit=0, upper_limit=100, steepness=3, weight=1.0),
    "bp_sys": dict(normal_low=100, normal_high=140, lower_limit=70, upper_limit=170, steepness=4, weight=2.0),
    "bp_dia": dict(normal_low=60, normal_high=90, lower_limit=45, upper_limit=120, steepness=4, weight=2.0),
}
DECIMALS = {"temperature": 1, "pulse": 0, "spo2": 1, "stress": 0, "bp_sys": 0, "bp_dia": 0}

STAGE_STABLE, STAGE_CAUTION = 70, 50  # 원본 값 그대로, 임상 근거 없음
PENALTY = {"경고": 0.5, "주의": 0.75, "안정": 1.0}
K_V2 = 6  # 재설계 k. 정규화 척도라 원본 항목별 k(3~6)와 무관


def rng(offset=0):
    return np.random.default_rng(SEED + offset)
