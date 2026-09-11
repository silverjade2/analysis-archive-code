"""
03. v1 스냅샷 라벨·피처 테이블 (원본 노트북 구조, 피처 30열).
출력: data/features_v1.csv, snapshot_v1.csv, results/label_stats_v1.csv, churn_by_first_year.csv, future_revenue_leak.csv
"""
import numpy as np, pandas as pd
from _common import *

merged = pd.read_csv(DATA / "merged_v1.csv", parse_dates=["last_end"])
contracts = pd.read_csv(DATA / "contracts.csv", parse_dates=["contract_date", "start_date", "end_date"])
snap = contracts[contracts.contract_date <= REF_DATE].copy()

# 원본 파생 열
snap["first_contract"] = snap.groupby("company_id").contract_date.transform("min")
snap["gubun"] = np.where((snap.contract_date.dt.year == snap.first_contract.dt.year) &
                         (snap.contract_date.dt.month == snap.first_contract.dt.month), "신규", "재계약")
snap["year"] = snap.start_date.dt.year
last_row = snap.sort_values("end_date", ascending=False).drop_duplicates("company_id")[["company_id", "gubun", "first_contract"]]
rev_year = snap.pivot_table(index="company_id", columns="year", values="amount", aggfunc="sum", fill_value=0)
rev_year.columns = [f"rev_{c}" for c in rev_year.columns]
for y in range(2015, 2028):
    if f"rev_{y}" not in rev_year.columns:
        rev_year[f"rev_{y}"] = 0
agg = snap.groupby("company_id").agg(n_contracts=("contract_id", "nunique"), total_revenue=("amount", "sum"))

df = (merged.merge(last_row, on="company_id").merge(rev_year, on="company_id").merge(agg, on="company_id"))
df["y"] = churn_label(df.last_end, REF_DATE)
df["in_use"] = (df.last_end >= REF_DATE).astype(int)
df["first_year"] = df.first_contract.dt.year
df["first_month"] = df.first_contract.dt.month

# 원본 피처 30열 (y 포함)
onehot = pd.get_dummies(df[["size_class", "gubun"]], prefix=["size", "gubun"], dtype=int)
num = df[["firm_age", "avg_salary", "entry_salary", "industry_salary", "company_revenue", "operating_profit",
          "monthly_hires", "monthly_leavers", "employees_ext", "hiring_rate", "attrition_rate",
          "rev_2015", "rev_2024", "rev_2025", "rev_2026", "rev_2027",          # 2018~2023 드롭, 2015·2024~2027 유지 (원본 그대로)
          "n_contracts", "total_revenue", "first_year", "first_month",
          "attrition_growth", "hiring_growth", "headcount_growth"]]
X = pd.concat([df[["company_id"]], onehot, num, df[["y"]]], axis=1)
X.to_csv(DATA / "features_v1.csv", index=False)
df.drop(columns=[c for c in df.columns if c.startswith("truth_")]).to_csv(DATA / "snapshot_v1.csv", index=False)

# 라벨 통계, 미래 매출 열의 라벨 분리도
stats = [("n_rows", len(X)), ("n_features", X.shape[1] - 2), ("churn_rate", X.y.mean()),
         ("in_use_rate", df.in_use.mean()), ("gubun_new_rate", (df.gubun == "신규").mean())]
fut = []
for c in ["rev_2015", "rev_2024", "rev_2025", "rev_2026", "rev_2027", "n_contracts", "total_revenue"]:
    pos = df[c] > 0 if c.startswith("rev") else df[c] > df[c].median()
    fut.append(dict(feature=c, n_positive=int(pos.sum()), churn_rate_if_positive=df.y[pos].mean() if pos.any() else np.nan,
                    churn_rate_if_zero=df.y[~pos].mean()))
# 최초계약연도별 이탈률, 갱신 기회 횟수 병기
horizon = REF_DATE - pd.Timedelta(days=CHURN_GRACE_DAYS)
df["exposure_k"] = np.clip(np.floor((horizon - df.first_contract).dt.days / 365), 0, None)
by_year = df.groupby("first_year").agg(n=("company_id", "size"), churn_rate=("y", "mean"),
                                       mean_renewal_opportunities=("exposure_k", "mean")).reset_index()
by_year.round(3).to_csv(RESULTS / "churn_by_first_year.csv", index=False)
print(by_year.round(3))
pd.DataFrame(stats, columns=["metric", "value"]).round(4).to_csv(RESULTS / "label_stats_v1.csv", index=False)
fut.append(dict(feature="gubun_신규", n_positive=int((df.gubun == "신규").sum()),
                churn_rate_if_positive=df.y[df.gubun == "신규"].mean(), churn_rate_if_zero=df.y[df.gubun != "신규"].mean()))
pd.DataFrame(fut).round(4).to_csv(RESULTS / "future_revenue_leak.csv", index=False)
print(pd.DataFrame(stats)); print(pd.DataFrame(fut).round(3))
