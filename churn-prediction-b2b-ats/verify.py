"""
검증 (SKILL 재현 코드 검증 절차 2·3): 누수 점검, 시간 절단 손검산 5사, 정상 범위.
파이프라인의 일부가 아니다. `python verify.py`로 실행.
"""
import sys; sys.path.insert(0, "notebooks")
import numpy as np, pandas as pd
from _common import *

v1 = pd.read_csv(DATA / "features_v1.csv"); v2 = pd.read_csv(DATA / "features_v2.csv")
contracts = pd.read_csv(DATA / "contracts.csv", parse_dates=["contract_date", "start_date", "end_date"])
truth = pd.read_csv(DATA / "truth_v2_population.csv")

# 1) 누수 점검
for name, df in [("v1", v1), ("v2", v2)]:
    assert not [c for c in df.columns if c.startswith("truth_")], name
    assert "y" in df.columns and df.columns.tolist().count("y") == 1
print("[1] truth_* 열 없음, y 1개 — OK")

# 2) 시간 절단 손검산: v2 5사. T 이후 체결 계약이 피처에 들어가지 않았는지
sample = v2.sample(5, random_state=1)
ok = True
for _, r in sample.iterrows():
    c = contracts[contracts.company_id == r.company_id]
    before = c[c.contract_date <= CUTOFF_T]; after = c[c.contract_date > CUTOFF_T]
    n_before = before.contract_id.nunique(); rev12 = before[before.start_date > CUTOFF_T - pd.DateOffset(months=12)].amount.sum()
    match = (n_before == r.n_contracts_T) and abs(rev12 - r.revenue_last12m_T) < 1
    ok &= match
    print(f"  company {int(r.company_id)}: T 이전 계약 {n_before} (피처 {int(r.n_contracts_T)}), T 이후 계약 {after.contract_id.nunique()}건 제외, "
          f"직전12개월 매출 {rev12:,.0f} (피처 {r.revenue_last12m_T:,.0f}) → {'OK' if match else 'MISMATCH'}")
assert ok
print("[2] 시간 절단 손검산 5사 — OK")

# 3) 정상 범위
ls1 = pd.read_csv(RESULTS / "label_stats_v1.csv").set_index("metric").value
ls2 = pd.read_csv(RESULTS / "label_stats_v2.csv").set_index("metric").value
mc = pd.read_csv(RESULTS / "model_compare.csv")
print(f"[3] v1 rows {int(ls1.n_rows)}, features {int(ls1.n_features)}, churn {ls1.churn_rate:.3f} | v2 rows {int(ls2.n_rows)}, churn {ls2.churn_rate:.3f}")
o = mc[mc.model == "truth_churn_p"].auc.iloc[0]
over = mc[(mc.variant == "v2 timecut (688)") & (mc.auc > o)]
print(f"    v2 AUC > 오라클(688) {o:.3f}인 모델: {len(over)}개 (0이어야 정상)")
print(f"    v1 스냅샷 AUC 최대 {mc[mc.variant=='v1 snapshot'].auc.max():.3f} > 오라클(1,141) {mc[mc.model=='truth_churn_p (exposure-aware)'].auc.iloc[0]:.3f} — 의도한 누수")
assert v1.isna().sum().sum() == 0 and v2.isna().sum().sum() == 0
print("    결측 0 — OK")
