"""
05. 모델 비교: 원본 회귀 재현, 분류기 4종, v1 ablation, 같은 688사 v1/v2, 오라클, SHAP, 캘리브레이션.
출력: results/model_compare.csv, shap_importance_*.csv, calibration.csv, risk_list_compare.csv, data/shap_values_*.csv
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, r2_score, mean_absolute_error, mean_squared_error, brier_score_loss
from sklearn.calibration import calibration_curve
import lightgbm as lgb, xgboost as xgb, shap
from _common import *

v1 = pd.read_csv(DATA / "features_v1.csv")
v2 = pd.read_csv(DATA / "features_v2.csv")
truth2 = pd.read_csv(DATA / "truth_v2_population.csv")

# 누수 점검
for name, df in [("v1", v1), ("v2", v2)]:
    bad = [c for c in df.columns if c.startswith("truth_") or c in ("in_use", "last_end", "churn_ref")]
    assert not bad, f"{name} leakage columns: {bad}"

FUTURE_COLS = ["rev_2024", "rev_2025", "rev_2026", "rev_2027"]
CUM_COLS = ["total_revenue", "n_contracts", "rev_2015"]

def split95(df):
    train = df.sample(frac=0.95, random_state=786)     # 원본 random_state
    return train.reset_index(drop=True)

def models():
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
        "XGBoost": xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.9,
                                     colsample_bytree=0.9, random_state=SEED, verbosity=0, n_jobs=-1),
        "LightGBM": lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, subsample=0.9,
                                       subsample_freq=1, colsample_bytree=0.9, random_state=SEED, verbose=-1, n_jobs=-1),
    }

def cv_auc(X, y, name, variant, rows):
    skf = StratifiedKFold(10, shuffle=True, random_state=123)
    oof = {}
    for mname, m in models().items():
        p = cross_val_predict(m, X, y, cv=skf, method="predict_proba")[:, 1]
        oof[mname] = p
        rows.append(dict(variant=variant, model=mname, task="classification", n=len(y), auc=roc_auc_score(y, p)))
    return oof

rows = []
# (1) v1 스냅샷: 원본 회귀(RF) 재현
tr = split95(v1)
X, y = tr.drop(columns=["company_id", "y"]), tr.y
kf = KFold(10, shuffle=True, random_state=123)
pred = cross_val_predict(RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1), X, y, cv=kf)
rows.append(dict(variant="v1 snapshot", model="Random Forest (regression, 원본)", task="regression", n=len(y),
                 auc=roc_auc_score(y, pred), r2=r2_score(y, pred), mae=mean_absolute_error(y, pred),
                 rmse=np.sqrt(mean_squared_error(y, pred))))
oof_v1 = cv_auc(X, y, "v1", "v1 snapshot", rows)
# (2) v1 ablation: 미래 매출·누적 열 제거
Xa = X.drop(columns=FUTURE_COLS + CUM_COLS)
cv_auc(Xa, y, "v1a", "v1 minus future/cumulative", rows)
# (2b) 계약 이력 열 단계적 제거 (LightGBM만)
skf10 = StratifiedKFold(10, shuffle=True, random_state=123)
for label, extra in [("v1 minus future/cumulative/gubun", ["gubun_신규", "gubun_재계약"]),
                     ("v1 minus all contract-history cols", ["gubun_신규", "gubun_재계약", "first_year", "first_month"])]:
    p = cross_val_predict(models()["LightGBM"], Xa.drop(columns=extra), y, cv=skf10, method="predict_proba")[:, 1]
    rows.append(dict(variant=label, model="LightGBM", task="classification", n=len(y), auc=roc_auc_score(y, p)))
# (3) v2 모집단(688사)에서 v1도 다시 채점
v1_688 = v1[v1.company_id.isin(v2.company_id)].set_index("company_id").loc[v2.company_id].reset_index()
X1s, y1s = v1_688.drop(columns=["company_id", "y"]), v1_688.y
cv_auc(X1s, y1s, "v1s", "v1 snapshot (688)", rows)
X2, y2 = v2.drop(columns=["company_id", "y"]), v2.y
oof_v2 = cv_auc(X2, y2, "v2", "v2 timecut (688)", rows)
rows.append(dict(variant="oracle (688)", model="truth_churn_p", task="oracle", n=len(y2), auc=roc_auc_score(y2, truth2.truth_churn_p)))
# 전체 오라클: 최초계약 이후 갱신 기회 k(12개월 단위)로 1 - p^k
comp = pd.read_csv(DATA / "companies.csv")[["company_id", "truth_renew_p"]]
snap = pd.read_csv(DATA / "snapshot_v1.csv", parse_dates=["first_contract"]).merge(comp, on="company_id")
horizon = REF_DATE - pd.Timedelta(days=CHURN_GRACE_DAYS)
k_full = np.clip(np.floor((horizon - snap.first_contract).dt.days / 365), 0, None)
snap["truth_churn_p_full"] = 1 - snap.truth_renew_p ** k_full
snap_tr = snap.set_index("company_id").loc[tr.company_id]
rows.append(dict(variant="oracle (1,141)", model="truth_churn_p (exposure-aware)", task="oracle", n=len(y),
                 auc=roc_auc_score(y, snap_tr.truth_churn_p_full)))
rows.append(dict(variant="oracle (1,141)", model="1 - truth_renew_p", task="oracle", n=len(y),
                 auc=roc_auc_score(y, 1 - snap_tr.truth_renew_p)))
res = pd.DataFrame(rows).round(4)
res.to_csv(RESULTS / "model_compare.csv", index=False)
print(res.to_string(index=False))

# SHAP: LightGBM 5-fold OOF
def oof_shap(X, y, tag):
    skf = StratifiedKFold(5, shuffle=True, random_state=7)
    sv = np.zeros(X.shape)
    for trn, val in skf.split(X, y):
        m = models()["LightGBM"].fit(X.iloc[trn], y.iloc[trn])
        ex = shap.TreeExplainer(m)
        s = ex.shap_values(X.iloc[val])
        sv[val] = s[1] if isinstance(s, list) else s
    pd.DataFrame(sv, columns=X.columns).to_csv(DATA / f"shap_values_{tag}.csv", index=False)
    imp = pd.DataFrame({"feature": X.columns, "mean_abs_shap": np.abs(sv).mean(0)}).sort_values("mean_abs_shap", ascending=False)
    imp["share"] = imp.mean_abs_shap / imp.mean_abs_shap.sum()
    imp.round(4).to_csv(RESULTS / f"shap_importance_{tag}.csv", index=False)
    return imp
imp1 = oof_shap(X, y, "v1"); imp2 = oof_shap(X2, y2, "v2")
print(imp1.head(8).round(3)); print(imp2.head(8).round(3))

# 캘리브레이션 (LightGBM OOF)
cal_rows = []
for tag, p, yy in [("v1", oof_v1["LightGBM"], y), ("v2", oof_v2["LightGBM"], y2)]:
    frac, mean_p = calibration_curve(yy, p, n_bins=8, strategy="quantile")
    for f, mp in zip(frac, mean_p):
        cal_rows.append(dict(variant=tag, mean_pred=mp, frac_pos=f))
    rows_b = dict(variant=tag, brier=brier_score_loss(yy, p), base_rate=yy.mean())
    cal_rows.append(dict(variant=tag, mean_pred=np.nan, frac_pos=np.nan, **{k: v for k, v in rows_b.items() if k != "variant"}))
pd.DataFrame(cal_rows).round(4).to_csv(RESULTS / "calibration.csv", index=False)

# 상위 20% 리스크 리스트 vs 오라클
sc = truth2.copy(); sc["p_v2"] = oof_v2["LightGBM"]; sc["p_v1"] = None
sc["p_v1"] = pd.Series(oof_v1["LightGBM"], index=tr.company_id).reindex(sc.company_id).values
# v1 OOF는 95% 분할 안에서만 존재 → 없는 회사는 제외하고 비교
sc = sc.dropna(subset=["p_v1"])
top = int(len(sc) * 0.2)
lst = []
for tag in ["p_v1", "p_v2"]:
    sel = sc.nlargest(top, tag)
    lst.append(dict(list=tag, n=top, actual_churn_rate=sel.y.mean(), mean_truth_churn_p=sel.truth_churn_p.mean(),
                    share_no_renewal_in_window=(sel.remaining_months_T > 9).mean()))
lst.append(dict(list="random", n=top, actual_churn_rate=sc.y.mean(), mean_truth_churn_p=sc.truth_churn_p.mean(),
                share_no_renewal_in_window=(sc.remaining_months_T > 9).mean()))
overlap = len(set(sc.nlargest(top, "p_v1").company_id) & set(sc.nlargest(top, "p_v2").company_id)) / top
lst.append(dict(list="overlap_v1_v2", n=top, actual_churn_rate=overlap))
pd.DataFrame(lst).round(4).to_csv(RESULTS / "risk_list_compare.csv", index=False)
print(pd.DataFrame(lst).round(3))
