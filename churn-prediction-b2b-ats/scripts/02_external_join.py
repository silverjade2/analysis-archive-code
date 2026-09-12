"""외부데이터 결합. 원본 절차(fillna 0 -> 연봉 median -> right join -> dropna) 그대로 + 탈락 편향 집계"""

import pandas as pd
from common import DATA, REF_DATE, RESULTS, churn_label

companies = pd.read_csv(DATA / "companies.csv")
contracts = pd.read_csv(DATA / "contracts.csv", parse_dates=["contract_date", "start_date", "end_date"])
ext = pd.read_csv(DATA / "external.csv")

snap = contracts[contracts.contract_date <= REF_DATE]
last_end = snap.groupby("company_id").end_date.max().rename("last_end")
base = companies.drop(
    columns=["hiring_rate", "attrition_rate", "headcount_growth", "hiring_growth", "attrition_growth"]
).merge(last_end, on="company_id")
base["churn_ref"] = churn_label(base.last_end, REF_DATE)

fill0 = [
    "monthly_hires",
    "monthly_leavers",
    "employees_ext",
    "hiring_rate",
    "attrition_rate",
    "company_revenue",
    "operating_profit",
]
ext2 = ext.copy()
n_blank_fin = ext2.company_revenue.isna().sum()
ext2[fill0] = ext2[fill0].fillna(0)
for c in ["avg_salary", "entry_salary", "industry_salary"]:
    ext2[c] = ext2[c].fillna(ext2[c].median())
merged = ext2.merge(base, on="company_id", how="right")
n_before = len(merged)
merged_kept = merged.dropna()  # 여기서 799사 빠짐
merged_kept.to_csv(DATA / "merged_v1.csv", index=False)

base["kept"] = base.company_id.isin(merged_kept.company_id)
bins = [0, 50, 100, 300, 1000, 10**9]
labels = ["<50", "50-99", "100-299", "300-999", "1000+"]
base["size_band"] = pd.cut(base.employees, bins=bins, labels=labels, right=False)
by_band = (
    base.groupby("size_band", observed=True)
    .agg(n=("company_id", "size"), kept_rate=("kept", "mean"), churn_all=("churn_ref", "mean"))
    .reset_index()
)
by_band["churn_kept"] = base[base.kept].groupby("size_band", observed=True).churn_ref.mean().values
summary = pd.DataFrame(
    {
        "metric": [
            "n_companies",
            "n_kept",
            "kept_rate",
            "churn_rate_all",
            "churn_rate_kept",
            "churn_rate_dropped",
            "n_fin_blank_filled_zero",
            "share_fin_zero_among_kept",
        ],
        "value": [
            n_before,
            len(merged_kept),
            len(merged_kept) / n_before,
            base.churn_ref.mean(),
            base[base.kept].churn_ref.mean(),
            base[~base.kept].churn_ref.mean(),
            n_blank_fin,
            (merged_kept.company_revenue == 0).mean(),
        ],
    }
).round(4)
by_band.round(4).to_csv(RESULTS / "join_bias_by_size.csv", index=False)
summary.to_csv(RESULTS / "join_bias_summary.csv", index=False)
print(summary.to_string(index=False))
print(by_band.round(3).to_string(index=False))
