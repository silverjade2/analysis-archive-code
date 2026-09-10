"""
06. 기준일 스윕 — 라벨의 '오늘'을 움직이면 같은 회사의 라벨이 얼마나 바뀌는가.
- live: 그 시점까지 체결된 계약으로 라벨 (노트북을 나중에 새 데이터로 다시 돌린 경우)
- frozen: 2024-02 스냅샷을 그대로 두고 today만 바뀐 경우 (데이터 갱신 없이 재실행)
- 결과는 인터랙티브 위젯 데이터(outputs/results/ref_sweep.json)로도 저장한다.
"""
import json, numpy as np, pandas as pd
from _common import *

v1 = pd.read_csv(DATA / "features_v1.csv")
contracts = pd.read_csv(DATA / "contracts.csv", parse_dates=["contract_date", "start_date", "end_date"])
c = contracts[contracts.company_id.isin(v1.company_id)]
base_label = v1.set_index("company_id").y

def labels(ref, frozen):
    cut = REF_DATE if frozen else ref
    le = c[c.contract_date <= cut].groupby("company_id").end_date.max().reindex(v1.company_id)
    return churn_label(le, ref)

rows = []
for ref in pd.date_range("2023-03-15", DATA_END, freq="MS") + pd.Timedelta(days=14):
    for mode in ["live", "frozen"]:
        lab = labels(ref, mode == "frozen")
        flip = (lab.values != base_label.values)
        rows.append(dict(ref_date=ref.date(), mode=mode, churn_rate=lab.mean(), flip_rate=flip.mean(),
                         flip_to_churn=((lab.values == 1) & (base_label.values == 0)).mean(),
                         flip_to_customer=((lab.values == 0) & (base_label.values == 1)).mean()))
sw = pd.DataFrame(rows).round(4)
sw.to_csv(RESULTS / "reference_date_sweep.csv", index=False)
json.dump({"ref_date_base": str(REF_DATE.date()), "n": int(len(v1)), "base_churn_rate": round(float(base_label.mean()), 4),
           "rows": [{**r, "ref_date": str(r["ref_date"])} for r in sw.to_dict("records")]},
          open(RESULTS / "ref_sweep.json", "w"), ensure_ascii=False, indent=1)
key = sw[sw.ref_date.astype(str).isin(["2024-02-15", "2024-08-15", "2025-02-15", "2025-08-15", "2026-02-15"])]
print(key.to_string(index=False))
