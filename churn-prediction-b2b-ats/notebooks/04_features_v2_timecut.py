"""
04. v2 시간 절단 피처.
- 절단 시점 T = REF_DATE 12개월 전. 모집단은 T 시점에 '고객'인 회사(최종종료일이 T-90일 이후).
- 피처는 T 이전 정보만: 기업 속성·외부데이터·T까지의 계약 이력(건수, 재직 개월, 직전 12개월 매출,
  현재 계약 잔여 개월, 상품 구성). 미래 매출 열·총매출·사용유무는 없다.
- 라벨은 v1과 같은 REF_DATE 기준 이탈. 질문이 "T 시점 고객 중 누가 12개월 안에 이탈하는가"로 바뀐다.
- 오라클: 심어둔 갱신 확률 p와 T~REF-90 사이 갱신 결정 횟수 k로 1-p^k. 채점에만 쓴다.
"""
import numpy as np, pandas as pd
from _common import *

v1 = pd.read_csv(DATA / "features_v1.csv")
snap = pd.read_csv(DATA / "snapshot_v1.csv", parse_dates=["last_end", "first_contract"])
companies = pd.read_csv(DATA / "companies.csv")
contracts = pd.read_csv(DATA / "contracts.csv", parse_dates=["contract_date", "start_date", "end_date"])
T = CUTOFF_T

c_T = contracts[(contracts.contract_date <= T) & (contracts.company_id.isin(v1.company_id))]
g = c_T.groupby("company_id")
hist = pd.DataFrame({
    "n_contracts_T": g.contract_id.nunique(),
    "last_end_T": g.end_date.max(),
    "first_contract_T": g.contract_date.min(),
    "total_revenue_T": g.amount.sum(),
    "current_term": c_T.sort_values("end_date").groupby("company_id").term_months.last(),
}).reset_index()
last12 = c_T[c_T.start_date > T - pd.DateOffset(months=12)].groupby("company_id").amount.sum().rename("revenue_last12m_T")
hist = hist.merge(last12, on="company_id", how="left").fillna({"revenue_last12m_T": 0})
hist["tenure_months_T"] = ((T - hist.first_contract_T).dt.days / 30.44).round(1)
hist["remaining_months_T"] = ((hist.last_end_T - T).dt.days / 30.44).round(1)
hist["active_at_T"] = ((T - hist.last_end_T).dt.days <= CHURN_GRACE_DAYS).astype(int)

pop = hist[hist.active_at_T == 1].merge(snap, on="company_id", suffixes=("", "_snap"))
pop = pop.merge(companies[["company_id", "truth_renew_p"]], on="company_id")

# 오라클 이탈 확률
horizon = REF_DATE - pd.Timedelta(days=CHURN_GRACE_DAYS)
days_to_horizon = (horizon - pop.last_end_T).dt.days
k = np.where(days_to_horizon <= 0, 0, 1 + np.floor(days_to_horizon / 365))
pop["truth_churn_p"] = (1 - pop.truth_renew_p ** k).round(4)

onehot = pd.get_dummies(pop[["size_class", "product"]], prefix=["size", "product"], dtype=int)
num = pop[["firm_age", "avg_salary", "entry_salary", "industry_salary", "company_revenue", "operating_profit",
           "monthly_hires", "monthly_leavers", "employees_ext", "hiring_rate", "attrition_rate",
           "attrition_growth", "hiring_growth", "headcount_growth",
           "n_contracts_T", "tenure_months_T", "total_revenue_T", "revenue_last12m_T", "remaining_months_T", "current_term",
           "first_year", "first_month"]]
X2 = pd.concat([pop[["company_id"]], onehot, num, pop[["y"]]], axis=1)
X2.to_csv(DATA / "features_v2.csv", index=False)
pop[["company_id", "truth_renew_p", "truth_churn_p", "remaining_months_T", "y"]].to_csv(DATA / "truth_v2_population.csv", index=False)

from sklearn.metrics import roc_auc_score
stats = [("n_rows", len(X2)), ("n_features", X2.shape[1] - 2), ("churn_rate", X2.y.mean()),
         ("share_no_renewal_decision_in_window", (k == 0).mean()),
         ("oracle_auc_truth_churn_p", roc_auc_score(pop.y, pop.truth_churn_p)),
         ("oracle_auc_renew_p_only", roc_auc_score(pop.y, 1 - pop.truth_renew_p))]
pd.DataFrame(stats, columns=["metric", "value"]).round(4).to_csv(RESULTS / "label_stats_v2.csv", index=False)
print(pd.DataFrame(stats))
