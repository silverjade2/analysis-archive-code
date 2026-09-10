"""
01. 가상 고객사·계약 이력·외부데이터 생성.
- 고객사 2,000사. 각 사에 잠재 갱신 확률 truth_renew_p(연간)를 부여하고,
  계약 종료 시점마다 베르누이로 갱신 여부를 결정해 계약 이력을 만든다.
- 이탈 라벨은 여기서 만들지 않는다. 03에서 원본 규칙으로 '도출'한다.
- 외부데이터(건강보험·공시)는 기업 규모가 클수록 존재 확률이 높게 심는다 → 결합 탈락 편향.
- 심어둔 진짜 원인: 인원 성장률(+), 연간 채용비율(+), 두 상품(A=diag, B=posting) 동시 사용(+),
  기업 규모(+), 업력(−), 퇴사비율(−). truth_* 열은 채점에만 쓴다.
"""
import numpy as np, pandas as pd
from _common import *

N = 2000
r = rng(1)
z = lambda x: (x - x.mean()) / x.std()

# ---- 고객사 속성 ----
employees = np.clip(np.exp(r.normal(np.log(300), 1.1, N)), 3, 40000).round().astype(int)
firm_age = np.clip(r.gamma(2.2, 12, N), 0, 130).round().astype(int)
hiring_rate = np.clip(r.beta(1.6, 40, N), 0.002, 0.6)            # 연간 신규채용/재직
attrition_rate = np.clip(hiring_rate * r.lognormal(0, 0.45, N), 0.002, 0.8)
headcount_growth = np.clip(r.normal(0.03, 0.10, N), -0.5, 1.5)
hiring_growth = np.clip(headcount_growth * 1.8 + r.normal(0, 0.25, N), -0.9, 3)
attrition_growth = np.clip(r.normal(0.05, 0.22, N), -0.9, 3)
product = r.choice(["diag", "posting", "both"], N, p=[0.45, 0.25, 0.30])
both = (product == "both").astype(int)

size_class = np.where(employees >= 1000, "LARGE", np.where(employees >= 300, "MEDIUM", "SMALL"))
size_class = np.where(r.random(N) < 0.05, "ETC", size_class)

# ---- 잠재 갱신 확률 (연간) ----
logit = (0.30 + 0.70 * z(headcount_growth) + 0.50 * z(np.log(hiring_rate)) + 0.90 * both
         + 0.35 * z(np.log(employees)) - 0.20 * z(firm_age) - 0.40 * z(np.log(attrition_rate))
         + r.normal(0, 0.5, N))
truth_renew_p = 1 / (1 + np.exp(-logit))

companies = pd.DataFrame({
    "company_id": np.arange(1, N + 1),
    "employees": employees, "firm_age": firm_age, "size_class": size_class, "product": product,
    "hiring_rate": hiring_rate.round(4), "attrition_rate": attrition_rate.round(4),
    "headcount_growth": headcount_growth.round(3), "hiring_growth": hiring_growth.round(3),
    "attrition_growth": attrition_growth.round(3),
    "truth_renew_p": truth_renew_p.round(4),
})

# ---- 계약 이력 (DATA_END까지 생성; 스냅샷 절단은 사용하는 쪽에서) ----
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
        # 다년 계약은 연 단위 세금계산서 행으로 분할 (원본: 공급가격/분할횟수, 시작일자 연도별 매출)
        n_split = term // 12
        for k in range(n_split):
            rows.append(dict(contract_id=cid, company_id=i + 1, contract_date=contract_date,
                             start_date=start + pd.DateOffset(years=k),
                             end_date=end, term_months=term, invoice_seq=k + 1,
                             amount=round(amt / n_split, -3), product=product[i]))
        # 같은 달 추가 계약 (상품 추가) 10%
        if r.random() < 0.10:
            cid += 1
            rows.append(dict(contract_id=cid, company_id=i + 1, contract_date=contract_date + pd.Timedelta(days=3),
                             start_date=start + pd.Timedelta(days=5), end_date=end, term_months=term, invoice_seq=1,
                             amount=round(base_amt * 0.3 * r.lognormal(0, 0.3), -3), product=product[i]))
        cid += 1
        if r.random() > p:          # 갱신 실패 → 이력 종료
            break
        gap = int(r.integers(0, 45)) if r.random() < 0.3 else 0
        start = end + pd.Timedelta(days=1 + gap)
contracts = pd.DataFrame(rows)

# ---- 외부데이터: 규모가 클수록 존재. 일부 필드는 빈값 ----
lz = z(np.log(employees))
p_ext = 1 / (1 + np.exp(-(0.55 + 1.3 * lz)))
has_ext = r.random(N) < p_ext
ind_salary = np.clip(r.normal(55, 12, N), 25, 120)
avg_salary = np.clip(ind_salary * r.lognormal(0.02, 0.25, N) + 2 * lz, 15, 200)
entry_salary = np.clip(avg_salary * r.uniform(0.45, 0.7, N), 10, 100)
rev = np.exp(r.normal(np.log(employees * 250.0), 0.8))       # 백만 원
op = rev * r.normal(0.06, 0.09, N)
ext = pd.DataFrame({
    "company_id": companies.company_id,
    "employees_ext": employees,
    "avg_salary": (avg_salary / 10).round() * 10, "entry_salary": (entry_salary / 10).round() * 10,
    "industry_salary": (ind_salary / 10).round() * 10,
    "company_revenue": rev.round(-1), "operating_profit": op.round(-1),
    "monthly_hires": (employees * hiring_rate / 12).round(), "monthly_leavers": (employees * attrition_rate / 12).round(),
    "hiring_rate": hiring_rate.round(3), "attrition_rate": attrition_rate.round(3),
    "attrition_growth": attrition_growth, "hiring_growth": hiring_growth, "headcount_growth": headcount_growth,
})[has_ext].copy()
# 공시 없는 비상장 다수: 매출·영업이익 빈값 (원본: fillna(0)) / 연봉 빈값 (원본: median)
m = ext.shape[0]
blank_fin = r.random(m) < 0.55
blank_sal = r.random(m) < 0.30
ext.loc[blank_fin, ["company_revenue", "operating_profit"]] = np.nan
ext.loc[blank_sal, ["avg_salary", "entry_salary", "industry_salary"]] = np.nan
# 단위 오류 이상치 1건 (원본 산업평균연봉 max 9,010)
ext.iloc[0, ext.columns.get_loc("industry_salary")] = 9010

companies.to_csv(DATA / "companies.csv", index=False)
contracts.to_csv(DATA / "contracts.csv", index=False)
ext.to_csv(DATA / "external.csv", index=False)
print(f"companies {len(companies)}, contracts {len(contracts)}, external {len(ext)} ({len(ext)/N:.1%}), renew_p mean {truth_renew_p.mean():.3f}")
