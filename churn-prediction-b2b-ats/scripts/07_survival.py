"""
07. 생존분석: Kaplan-Meier(상품 구성별), Cox PH(심어둔 원인 변수), truth_renew_p와 스피어만 상관으로 채점.
출력: results/km_by_product.csv, km_key_points.csv, cox_summary.csv
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import concordance_index
from scipy.stats import spearmanr
from _common import *

snap = pd.read_csv(DATA / "snapshot_v1.csv", parse_dates=["last_end", "first_contract"])
comp = pd.read_csv(DATA / "companies.csv")[["company_id", "truth_renew_p"]]
df = snap.merge(comp, on="company_id")
df["event"] = df.y
end_time = np.where(df.event == 1, df.last_end, REF_DATE)
df["duration_m"] = ((pd.to_datetime(end_time) - df.first_contract).dt.days / 30.44).clip(lower=0.5)

# KM by product
km_rows = []
grid = np.arange(0, 97, 1)
for prod, g in df.groupby("product"):
    kmf = KaplanMeierFitter().fit(g.duration_m, g.event, label=prod)
    sf = kmf.survival_function_at_times(grid).values
    ci = kmf.confidence_interval_survival_function_
    ci_lo = np.interp(grid, ci.index.values, ci.iloc[:, 0].values); ci_hi = np.interp(grid, ci.index.values, ci.iloc[:, 1].values)
    for t, s, lo, hi in zip(grid, sf, ci_lo, ci_hi):
        km_rows.append(dict(product=prod, month=t, survival=s, ci_lower=lo, ci_upper=hi, n=len(g),
                            at_risk=int((g.duration_m >= t).sum())))
km = pd.DataFrame(km_rows).round(4)
km.to_csv(RESULTS / "km_by_product.csv", index=False)
km_key = km[km.month.isin([12, 24, 36, 48])].pivot(index="product", columns="month", values="survival").reset_index()
km_key.to_csv(RESULTS / "km_key_points.csv", index=False)

# Cox PH
cox_df = pd.DataFrame({
    "duration_m": df.duration_m, "event": df.event,
    "headcount_growth": df.headcount_growth, "log_hiring_rate": np.log(df.hiring_rate.clip(lower=1e-3)),
    "log_attrition_rate": np.log(df.attrition_rate.clip(lower=1e-3)), "log_employees": np.log(df.employees),
    "firm_age": df.firm_age, "product_both": (df["product"] == "both").astype(int), "product_posting": (df["product"] == "posting").astype(int),
})
for c in ["headcount_growth", "log_hiring_rate", "log_attrition_rate", "log_employees", "firm_age"]:
    cox_df[c] = (cox_df[c] - cox_df[c].mean()) / cox_df[c].std()
cph = CoxPHFitter().fit(cox_df, "duration_m", "event")
summ = cph.summary[["coef", "exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].reset_index().rename(columns={"covariate": "feature"})
summ["concordance"] = cph.concordance_index_
rho = spearmanr(cph.predict_partial_hazard(cox_df), -df.truth_renew_p).correlation
summ["spearman_vs_truth"] = rho
summ.round(4).to_csv(RESULTS / "cox_summary.csv", index=False)
print(km_key.round(3)); print(summ.round(3).to_string(index=False))
