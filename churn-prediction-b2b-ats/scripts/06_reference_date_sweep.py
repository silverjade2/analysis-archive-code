"""실행일 sweep. today를 옮기면 라벨이 얼마나 뒤집히는지

live = 그 시점까지 계약 반영 / frozen = 2024-02 스냅샷 고정, today만 이동
"""

import json

import pandas as pd
from common import DATA, DATA_END, REF_DATE, RESULTS, churn_label

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
        flip = lab.values != base_label.values
        rows.append(
            dict(
                ref_date=ref.date(),
                mode=mode,
                churn_rate=lab.mean(),
                flip_rate=flip.mean(),
                flip_to_churn=((lab.values == 1) & (base_label.values == 0)).mean(),
                flip_to_customer=((lab.values == 0) & (base_label.values == 1)).mean(),
            )
        )
sw = pd.DataFrame(rows).round(4)
sw.to_csv(RESULTS / "reference_date_sweep.csv", index=False)
json.dump(
    {
        "ref_date_base": str(REF_DATE.date()),
        "n": int(len(v1)),
        "base_churn_rate": round(float(base_label.mean()), 4),
        "rows": [{**r, "ref_date": str(r["ref_date"])} for r in sw.to_dict("records")],
    },
    open(RESULTS / "ref_sweep.json", "w"),
    ensure_ascii=False,
    indent=1,
)
key = sw[sw.ref_date.astype(str).isin(["2024-02-15", "2024-08-15", "2025-02-15", "2025-08-15", "2026-02-15"])]
print(key.to_string(index=False))
