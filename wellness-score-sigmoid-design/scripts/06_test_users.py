"""테스트 사용자 7명 (A~G) 점수 계산. 적은 항목 말고는 정상범위 중앙값"""

import pandas as pd
from common import ITEMS, PARAMS, RES
from scoring import score_frame

mid = {i: (PARAMS[i]["normal_low"] + PARAMS[i]["normal_high"]) / 2 for i in ITEMS}
inputs = {
    "A": {},
    "B": {},
    "C": {},
    "D": {},
    "E": dict(temperature=37.6),
    "F": dict(pulse=115, bp_dia=91),
    "G": dict(spo2=88.0),
}

rows = []
for user, over in inputs.items():
    row = {"user": user, "inputs": ",".join(f"{k}={v}" for k, v in over.items()) or "-", **mid}
    row.update(over)
    rows.append(row)

scored = score_frame(pd.DataFrame(rows))
cols = ["user", "inputs", "score_current", "stage_current", "score_penalized", "stage_penalized", "item_stage_v2"]
out = scored[cols].rename(columns={"item_stage_v2": "item_stage"})
out.to_csv(RES / "test_users.csv", index=False)

print(out.to_string(index=False))
print("점수 단계 != 항목 단계:", (out.stage_current != out.item_stage).sum(), "/", len(out))
print("보정 단계 != 항목 단계:", (out.stage_penalized != out.item_stage).sum(), "/", len(out))
