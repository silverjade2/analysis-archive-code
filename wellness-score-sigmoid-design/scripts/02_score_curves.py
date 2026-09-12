"""점수 함수 곡선 (데이터 없이 함수만). boundary_steps는 설계 문서에 손으로 적었던 숫자 재확인"""

import numpy as np
import pandas as pd
from common import ITEMS, PARAMS, RES
from scoring import item_score_current, item_score_v2, sigmoid

rows = []
for item, steps in {
    "temperature": [0.0, 0.01, 0.1, 0.5, 1.0, 2.5],
    "pulse": [0, 1, 2, 5, 15, 45],
    "stress": [0, 1, 2, 5, 20, 40],
    "bp_sys": [0, 1, 2, 5, 15, 30],
}.items():
    p = PARAMS[item]
    for s in steps:
        x = p["normal_high"] + s
        rows.append(
            dict(
                item=item,
                over_by=s,
                value=x,
                score_current=item_score_current(x, p) / p["weight"],
                score_v2=item_score_v2(x, p) / p["weight"],
            )
        )
steps = pd.DataFrame(rows).round(2)
steps.to_csv(RES / "boundary_steps.csv", index=False)

curve = []
for item in ITEMS:
    p = PARAMS[item]
    width = p["normal_high"] - p["normal_low"]
    lo = p["lower_limit"] - 0.5 * width
    hi = p["upper_limit"] + 0.5 * width
    for x in np.linspace(lo, hi, 601):
        curve.append(
            dict(
                item=item,
                value=x,
                score_current=item_score_current(x, p) / p["weight"],
                score_v2=item_score_v2(x, p) / p["weight"],
            )
        )
pd.DataFrame(curve).round(4).to_csv(RES / "item_curves.csv", index=False)

kc = []
for k in (4, 6, 8):
    for d in np.linspace(0, 2, 401):
        kc.append(dict(k=k, d=d, score=100 * (1 - sigmoid(k * (d - 0.5))) / (1 - sigmoid(-0.5 * k))))
kc = pd.DataFrame(kc).round(4)
kc.to_csv(RES / "k_curves.csv", index=False)

marks = kc[np.isclose(kc.d, 0.5) | np.isclose(kc.d, 1.0) | np.isclose(kc.d, 1.5)].pivot(
    index="k", columns="d", values="score"
)
marks.round(1).to_csv(RES / "k_curve_marks.csv")
print(steps.to_string(index=False))
print(marks.round(1))
