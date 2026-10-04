# 구직자 군집: 해석이 안 되는 두 군집(비활성, 입력 빈약)이 어디서 막히는지
# - k sweep (log + 표준화 K-means)
# - 군집 x 잠재 유형 구성
# - 비활성 군집 안에서 취업 완료 / 이탈을 가를 수 있는가 (K-means, GMM, raw event)
# - 최근 경력 신입 규칙이 입력 빈약 때문에 놓치는 몫
# 출력: outputs/results/js_*.csv

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

SEED = 42
COUNT_COLS = ["logins_180d", "views_180d", "applies_180d", "days_since_login"]
PROFILE_COLS = ["career_months", "months_since_job", "education", "assessed", "assess_score", "profile_completeness"]

base = Path(__file__).resolve().parents[1]
res = base / "outputs" / "results"
res.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(base / "data" / "jobseekers.csv")
X = StandardScaler().fit_transform(np.column_stack([np.log1p(df[COUNT_COLS]), df[PROFILE_COLS]]))

sweep = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
    sil = silhouette_score(X, km.labels_, sample_size=3000, random_state=SEED)
    sweep.append({"k": k, "inertia": round(km.inertia_, 1), "silhouette": round(sil, 3)})
sweep = pd.DataFrame(sweep)
sweep.to_csv(res / "js_k_sweep.csv", index=False)
print(sweep.to_string(index=False))

K = 5  # silhouette 0.28~0.31로 평평. 입력 빈약 사용자가 활동 수준별로 갈라지는 가장 작은 k
df["cluster"] = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit_predict(X)
df["group"] = np.where(df["truth_outcome"] == "active", df["truth_type"], df["truth_outcome"])

comp = pd.crosstab(df["cluster"], df["group"])
comp["sparse_share"] = df.groupby("cluster")["truth_sparse"].mean().round(3)
comp["days_since_login_mean"] = df.groupby("cluster")["days_since_login"].mean().round(1)
comp.to_csv(res / "js_cluster_composition.csv")
print(comp)

dormant_c = comp["days_since_login_mean"].idxmax()
in_c = (df["cluster"] == dormant_c).to_numpy()
Xd = X[in_c]
d = df[in_c & df["truth_outcome"].isin(["hired", "lapsed"]).to_numpy()]
eval_mask = df.loc[in_c, "truth_outcome"].isin(["hired", "lapsed"]).to_numpy()  # 평가만 취업/이탈 행에서
outcome = (d["truth_outcome"] == "hired").astype(int)
split = []
for name, lab in [
    ("kmeans_k2", KMeans(n_clusters=2, n_init=10, random_state=SEED).fit_predict(Xd)),
    ("gmm_k2", GaussianMixture(n_components=2, random_state=SEED).fit(Xd).predict(Xd)),
]:
    split.append({"method": name, "ari_vs_outcome": round(adjusted_rand_score(outcome, lab[eval_mask]), 3)})
split = pd.DataFrame(split)
split.to_csv(res / "js_dormant_split.csv", index=False)
print(split.to_string(index=False))

# 멈추기 전 활동량 비교: 취업 완료와 이탈이 집계 feature에서 같은지
pre = d.groupby("truth_outcome")[COUNT_COLS + ["profile_completeness"]].mean().round(2)
pre.to_csv(res / "js_dormant_profile.csv")
print(pre)

hired = int(outcome.sum())
by_event = int(d["offer_event"].sum())
unknown = len(d) - by_event
known = pd.DataFrame(
    [
        {
            "cluster_size": int((df["cluster"] == dormant_c).sum()),
            "hired_or_lapsed": len(d),
            "hired": hired,
            "lapsed": len(d) - hired,
            "hired_by_offer_event": by_event,
            "unknown": unknown,
            "unknown_share": round(unknown / len(d), 3),
            "hired_share_in_unknown": round((hired - by_event) / unknown, 3),
        }
    ]
)
known.to_csv(res / "js_dormant_known.csv", index=False)
print(known.T)

# 최근 경력 신입 규칙: 경력 6~30개월이고 퇴사 12개월 안
rule = df["career_months"].between(6, 30) & df["months_since_job"].between(0, 12)
truth = df["truth_type"] == "recent_career"
target = pd.DataFrame(
    [
        {
            "truth_recent_career": int(truth.sum()),
            "rule_hit": int(rule.sum()),
            "rule_hit_true": int((rule & truth).sum()),
            "recall": round((rule & truth).sum() / truth.sum(), 3),
            "precision": round((rule & truth).sum() / rule.sum(), 3),
            "missed_sparse": int((truth & ~rule & (df["truth_sparse"] == 1)).sum()),
            "missed_other": int((truth & ~rule & (df["truth_sparse"] == 0)).sum()),
        }
    ]
)
target.to_csv(res / "js_target_rule.csv", index=False)
print(target.T)
