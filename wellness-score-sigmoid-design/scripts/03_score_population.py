"""합성 1000명 + 엣지 케이스 채점. 엣지 케이스는 assert로 기대값 확인"""

import pandas as pd
from common import DATA, PARAMS, RES
from scoring import score_frame

users = pd.read_csv(DATA / "synthetic_vitals.csv")
scored = score_frame(users)
scored.to_csv(DATA / "scored_synthetic.csv", index=False)

edge = pd.read_csv(DATA / "edge_cases.csv")
scored_edge = score_frame(edge)
for i in ("temperature", "spo2"):
    scored_edge[f"item_v2_{i}"] = scored_edge[f"score_v2_{i}"] / PARAMS[i]["weight"]
cols = [
    "case",
    "expected",
    "score_current",
    "stage_current",
    "score_penalized",
    "stage_penalized",
    "score_v2",
    "stage_v2",
    "item_stage_original",
    "item_stage_v2",
    "missing_count",
    "final_stage",
    "item_v2_temperature",
    "item_v2_spo2",
]
scored_edge[cols].round(2).to_csv(RES / "edge_results.csv", index=False)

e = scored_edge.set_index("case")
assert e.loc["정상 경계 안쪽", "item_v2_temperature"] == 100
assert e.loc["정상 경계 바로 밖", "item_v2_temperature"] >= 99
assert e.loc["임계치 일치", "status_spo2"] == "주의" and abs(e.loc["임계치 일치", "item_v2_spo2"] - 5) < 0.1
assert e.loc["임계치 초과 극단", "item_v2_temperature"] < 1 and e.loc["임계치 초과 극단", "item_v2_spo2"] < 1
assert e.loc["임계치 초과 극단", "final_stage"] == "경고"
assert e.loc["결측 1개", "missing_count"] == 1 and e.loc["결측 1개", "score_v2"] == 100
assert e.loc["전부 결측", "final_stage"] == "데이터 없음"
assert e.loc["상방 이탈 불가 항목", "item_v2_spo2"] == 0
assert e.loc["단일 항목 경고", "stage_v2"] == "안정" and e.loc["단일 항목 경고", "final_stage"] == "경고"
assert e.loc["주의만 둘", "stage_penalized"] == "경고" and e.loc["주의만 둘", "final_stage"] == "주의"

print(scored_edge[cols].round(2).to_string(index=False))
print(scored[["score_current", "score_v2"]].describe().round(2))
