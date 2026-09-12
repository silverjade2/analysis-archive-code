"""가상 고객사 2000사 + 계약 이력 + 외부데이터 생성

- truth_renew_p(연간 갱신 확률)만 심고, 이탈 라벨은 안 만듦 (03에서 원본 규칙으로)
- 외부데이터 존재 확률은 규모에 비례 -> 작은 회사가 결합에서 빠지도록
"""

import numpy as np
import pandas as pd
from common import DATA, DATA_END, rng

N = 2000
r = rng(1)


def z(x):
    return (x - x.mean()) / x.std()


employees = np.clip(np.exp(r.normal(np.log(300), 1.1, N)), 3, 40000).round().astype(int)
firm_age = np.clip(r.gamma(2.2, 12, N), 0, 130).round().astype(int)
hiring_rate = np.clip(r.beta(1.6, 40, N), 0.002, 0.6)  # 연간 채용/재직
attrition_rate = np.clip(hiring_rate * r.lognormal(0, 0.45, N), 0.002, 0.8)
headcount_growth = np.clip(r.normal(0.03, 0.10, N), -0.5, 1.5)
hiring_growth = np.clip(headcount_growth * 1.8 + r.normal(0, 0.25, N), -0.9, 3)
attrition_growth = np.clip(r.normal(0.05, 0.22, N), -0.9, 3)
product = r.choice(["diag", "posting", "both"], N, p=[0.45, 0.25, 0.30])
both = (product == "both").astype(int)

size_class = np.where(employees >= 1000, "LARGE", np.where(employees >= 300, "MEDIUM", "SMALL"))
size_class = np.where(r.random(N) < 0.05, "ETC", size_class)

# 갱신 확률 원인 6개. 부호는 cox_summary로 확인
logit = (
    0.30
    + 0.70 * z(headcount_growth)
    + 0.50 * z(np.log(hiring_rate))
    + 0.90 * both
    + 0.35 * z(np.log(employees))
    - 0.20 * z(firm_age)
    - 0.40 * z(np.log(attrition_rate))
    + r.normal(0, 0.5, N)
)
truth_renew_p = 1 / (1 + np.exp(-logit))

companies = pd.DataFrame(
    {
        "company_id": np.arange(1, N + 1),
        "employees": employees,
        "firm_age": firm_age,
        "size_class": size_class,
        "product": product,
        "hiring_rate": hiring_rate.round(4),
        "attrition_rate": attrition_rate.round(4),
        "headcount_growth": headcount_growth.round(3),
        "hiring_growth": hiring_growth.round(3),
        "attrition_growth": attrition_growth.round(3),
        "truth_renew_p": truth_renew_p.round(4),
    }
)

year_p = {2015: 0.03, 2018: 0.09, 2019: 0.13, 2020: 0.16, 2021: 0.20, 2022: 0.22, 2023: 0.17}
years, probs = zip(*year_p.items())
first_year = r.choice(years, N, p=probs)
rows = []
cid = 0
for i in range(N):
    p = truth_renew_p[i]
    start = pd.Timestamp(year=int(first_year[i]), month=int(r.integers(1, 13)), day=int(r.integers(1, 28)))
    base_amt = 3.2e6 * (employees[i] ** 0.45) * (1.6 if product[i] == "both" else 1.0)
    while start <= DATA_END:
        term = int(r.choice([12, 24, 36], p=[0.80, 0.13, 0.07]))
        amt = base_amt * (term / 12) * r.lognormal(0, 0.3)
        contract_date = start - pd.Timedelta(days=int(r.integers(0, 20)))
        end = start + pd.DateOffset(months=term) - pd.Timedelta(days=1)
        # 24/36개월은 연 단위 세금계산서 행으로 split (원본 데이터 형태)
        n_split = term // 12
        for k in range(n_split):
            rows.append(
                dict(
                    contract_id=cid,
                    company_id=i + 1,
                    contract_date=contract_date,
                    start_date=start + pd.DateOffset(years=k),
                    end_date=end,
                    term_months=term,
                    invoice_seq=k + 1,
                    amount=round(amt / n_split, -3),
                    product=product[i],
                )
            )
        # 추가 계약 10%
        if r.random() < 0.10:
            cid += 1
            rows.append(
                dict(
                    contract_id=cid,
                    company_id=i + 1,
                    contract_date=contract_date + pd.Timedelta(days=3),
                    start_date=start + pd.Timedelta(days=5),
                    end_date=end,
                    term_months=term,
                    invoice_seq=1,
                    amount=round(base_amt * 0.3 * r.lognormal(0, 0.3), -3),
                    product=product[i],
                )
            )
        cid += 1
        if r.random() > p:  # 갱신 안 함
            break
        gap = int(r.integers(0, 45)) if r.random() < 0.3 else 0
        start = end + pd.Timedelta(days=1 + gap)
contracts = pd.DataFrame(rows)

lz = z(np.log(employees))
p_ext = 1 / (1 + np.exp(-(0.55 + 1.3 * lz)))
has_ext = r.random(N) < p_ext
ind_salary = np.clip(r.normal(55, 12, N), 25, 120)
avg_salary = np.clip(ind_salary * r.lognormal(0.02, 0.25, N) + 2 * lz, 15, 200)
entry_salary = np.clip(avg_salary * r.uniform(0.45, 0.7, N), 10, 100)
rev = np.exp(r.normal(np.log(employees * 250.0), 0.8))  # 단위 백만
op = rev * r.normal(0.06, 0.09, N)
ext = pd.DataFrame(
    {
        "company_id": companies.company_id,
        "employees_ext": employees,
        "avg_salary": (avg_salary / 10).round() * 10,
        "entry_salary": (entry_salary / 10).round() * 10,
        "industry_salary": (ind_salary / 10).round() * 10,
        "company_revenue": rev.round(-1),
        "operating_profit": op.round(-1),
        "monthly_hires": (employees * hiring_rate / 12).round(),
        "monthly_leavers": (employees * attrition_rate / 12).round(),
        "hiring_rate": hiring_rate.round(3),
        "attrition_rate": attrition_rate.round(3),
        "attrition_growth": attrition_growth,
        "hiring_growth": hiring_growth,
        "headcount_growth": headcount_growth,
    }
)[has_ext].copy()
# 비상장 -> 매출/영업이익 결측, 연봉도 일부 결측
m = ext.shape[0]
blank_fin = r.random(m) < 0.55
blank_sal = r.random(m) < 0.30
ext.loc[blank_fin, ["company_revenue", "operating_profit"]] = np.nan
ext.loc[blank_sal, ["avg_salary", "entry_salary", "industry_salary"]] = np.nan
# 원본에 있던 단위 오류 1건 (industry_salary max 9010)
ext.iloc[0, ext.columns.get_loc("industry_salary")] = 9010

companies.to_csv(DATA / "companies.csv", index=False)
contracts.to_csv(DATA / "contracts.csv", index=False)
ext.to_csv(DATA / "external.csv", index=False)
print(
    f"companies {len(companies)}, contracts {len(contracts)}, external {len(ext)} ({len(ext) / N:.1%}), "
    f"renew_p mean {truth_renew_p.mean():.3f}"
)
