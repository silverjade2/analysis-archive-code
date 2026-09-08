"""09 — Leakage anatomy: the data behind figures 7–11.

Nothing here changes the pipeline; it reads data/ and the v1/v2 feature tables and writes
small CSVs that 10_site_figures_2.py draws. Five pieces:

  1. login rate around the (pseudo) consent day — consenters anchored on consent_day,
     non-consenters on their v2 cutoff + 1 (the same "profile + delay" formula as 04)
  2. a 12-user timeline sample (6 consented / 6 not) with login days and the v2 cutoff
  3. out-of-fold LightGBM scores for v1 and v2 (same CV / seed as 06) → ROC curves, checked
     against oof_auc_vs_oracle.csv
  4. non-consented users by score decile → mean planted consent propensity, checked against
     nudge_list_comparison.csv (top decile = the nudge list)
  5. non-consented users on the v1 × v2 score plane: the two top-decile lists as quadrants,
     checked against nudge_list_overlap.csv

Outputs (outputs/results/): login_around_consent.csv, timeline_sample.csv, oof_scores.csv,
roc_curves.csv, score_deciles.csv, score_quadrants.csv, leakage_anatomy_summary.csv (the
numbers quoted in the article's captions).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve
from lightgbm import LGBMClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RES, SEED, N_DAYS, NOTEBOOK_FEATURES, PREF_FEATURES, TARGET, CAT_COLS

PRE, POST = 30, 60          # event-study window (days before / after the anchor)
N_SAMPLE = 6                # users per class in the timeline figure
TOP = 0.10                  # nudge list = top decile of non-consented users (as in 06)
ROC_POINTS = 300

users = pd.read_csv(DATA / "users.csv")
logins = np.load(DATA / "logins.npz")["logins"]
v1 = pd.read_csv(DATA / "features_v1_snapshot.csv")
v2 = pd.read_csv(DATA / "features_v2_timecut.csv")
y = users[TARGET].values
pos = y == 1
join, profile, consent = users["join_day"].values, users["profile_day"].values, users["consent_day"].values
cutoff = v2["cutoff_day"].values
truth_p = users["truth_p_consent"].values
summary = {}

# ---------------------------------------------------------------- 1. login rate around the anchor
anchor = np.where(pos, consent, cutoff + 1)
rel = np.arange(-PRE, POST + 1)
idx = anchor[:, None] + rel[None, :]
valid = (idx >= join[:, None]) & (idx <= N_DAYS - 1)
idxc = np.clip(idx, 0, N_DAYS - 1)
out = pd.DataFrame({"rel_day": rel})
for label, m in [("consented", pos), ("not_consented", ~pos)]:
    r = np.flatnonzero(m)
    L = logins[r[:, None], idxc[r]].astype(float)
    V = valid[r]
    rate = (L * V).sum(0) / V.sum(0)
    out[f"{label}_rate"] = rate
    out[f"{label}_n"] = V.sum(0)
    pre = rate[(rel >= -PRE) & (rel <= -1)].mean()
    post = rate[(rel >= 8) & (rel <= POST)].mean()
    summary[f"{label}: users"] = len(r)
    summary[f"{label}: mean daily login rate, day -{PRE}..-1"] = pre
    summary[f"{label}: login rate on day +1"] = rate[rel == 1][0]
    summary[f"{label}: login rate on day +2"] = rate[rel == 2][0]
    summary[f"{label}: mean daily login rate, day +8..+{POST}"] = post
    summary[f"{label}: post/pre ratio"] = post / pre
for w in (f"mean daily login rate, day -{PRE}..-1", "login rate on day +1", f"mean daily login rate, day +8..+{POST}"):
    summary[f"ratio consented / not_consented: {w}"] = summary[f"consented: {w}"] / summary[f"not_consented: {w}"]
out.round(4).to_csv(RES / "login_around_consent.csv", index=False)
print("1. login_around_consent.csv")

# ---------------------------------------------------------------- 2. timeline sample
season = users["truth_season_joiner"].values
rng = np.random.default_rng(SEED)
room = (N_DAYS - 1 - join) >= 365
cand_pos = np.flatnonzero(pos & room & (consent - profile <= 30))
cand_neg = np.flatnonzero(~pos & room & (cutoff + 1 - profile <= 30))
sample = np.concatenate([rng.choice(cand_pos, N_SAMPLE, replace=False), rng.choice(cand_neg, N_SAMPLE, replace=False)])
rows = []
for u in sample:
    ld = np.flatnonzero(logins[u])
    ld = ld[ld <= join[u] + 365] - join[u]
    rows.append({"user": int(u), "consented": int(pos[u]), "season_joiner": int(season[u]),
                 "test_day": int(users["test1_day"][u] - join[u]), "profile_day": int(profile[u] - join[u]),
                 "consent_day": int(consent[u] - join[u]) if pos[u] else -1,
                 "v2_cutoff_day": int(cutoff[u] - join[u]),
                 "logins_365d": len(ld), "logins_after_cutoff_365d": int((ld > cutoff[u] - join[u]).sum()),
                 "login_days_since_join": ";".join(map(str, ld))})
pd.DataFrame(rows).to_csv(RES / "timeline_sample.csv", index=False)
print("2. timeline_sample.csv")

# ---------------------------------------------------------------- 3. out-of-fold scores → ROC
def encode(df, cols):
    cats = [c for c in cols if c in CAT_COLS]
    X = pd.get_dummies(df[cols], columns=cats, dtype=int)
    X.columns = [c.replace(" ", "_") for c in X.columns]
    return X


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
scores = {}
for vname, df, cols in [("v1", v1, NOTEBOOK_FEATURES), ("v2", v2, NOTEBOOK_FEATURES + PREF_FEATURES)]:
    m = LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=31, random_state=123, n_jobs=-1, verbose=-1)
    scores[vname] = cross_val_predict(m, encode(df, cols), y, cv=cv, method="predict_proba")[:, 1]
    print(f"   {vname}: out-of-fold scores done")
ref = pd.read_csv(RES / "oof_auc_vs_oracle.csv").set_index("model")["AUC"]
curves = {"v1 snapshot": scores["v1"], "v2 time cut": scores["v2"], "oracle (true propensity)": truth_p}
for label, s in curves.items():
    auc = roc_auc_score(y, s)
    assert abs(auc - ref[label]) < 5e-4, (label, auc, ref[label])   # must reproduce 06 exactly
    summary[f"AUC: {label}"] = auc
pd.DataFrame({"user": users["user"], TARGET: y, "truth_p_consent": truth_p,
              "score_v1": scores["v1"], "score_v2": scores["v2"]}).round(6).to_csv(RES / "oof_scores.csv", index=False)
rows = []
for label, s in curves.items():
    fpr, tpr, _ = roc_curve(y, s)
    keep = np.unique(np.linspace(0, len(fpr) - 1, ROC_POINTS).astype(int))
    rows += [(label, fpr[i], tpr[i]) for i in keep]
pd.DataFrame(rows, columns=["model", "fpr", "tpr"]).round(4).to_csv(RES / "roc_curves.csv", index=False)
print("3. oof_scores.csv, roc_curves.csv")

# ---------------------------------------------------------------- 4. non-consented users by score decile
neg = np.flatnonzero(~pos)
k = int(len(neg) * TOP)
nudge = pd.read_csv(RES / "nudge_list_comparison.csv").set_index("list")["mean true consent propensity"]
rows = []
for vname, s in scores.items():
    order = neg[np.argsort(-s[neg])]                       # same ordering as 06
    dec = np.minimum(np.arange(len(neg)) // k, 9) + 1
    for d in range(1, 11):
        r = order[dec == d]
        rows.append((vname, d, len(r), truth_p[r].mean(), s[r].mean()))
deciles = pd.DataFrame(rows, columns=["score", "decile", "n", "mean_truth_p", "mean_score"])
for vname, key in [("v1", "v1 top 10% (snapshot score)"), ("v2", "v2 top 10% (time-cut score)")]:
    top = deciles[(deciles["score"] == vname) & (deciles["decile"] == 1)]["mean_truth_p"].iloc[0]
    assert abs(top - nudge[key]) < 1e-4, (vname, top, nudge[key])   # decile 1 == the nudge list of 06
    bottom = deciles[(deciles["score"] == vname) & (deciles["decile"] == 10)]["mean_truth_p"].iloc[0]
    summary[f"deciles {vname}: top decile mean truth propensity"] = top
    summary[f"deciles {vname}: bottom decile mean truth propensity"] = bottom
    summary[f"deciles {vname}: top - bottom"] = top - bottom
summary["deciles: non-consented users"] = len(neg)
summary["deciles: non-consented mean truth propensity"] = truth_p[neg].mean()
deciles.round(4).to_csv(RES / "score_deciles.csv", index=False)
print("4. score_deciles.csv")

# ---------------------------------------------------------------- 5. v1 × v2 score plane, top-decile quadrants
lists = {vname: neg[np.argsort(-s[neg])[:k]] for vname, s in scores.items()}   # exactly the lists of 06
both = np.intersect1d(lists["v1"], lists["v2"])
v1_only = np.setdiff1d(lists["v1"], both)
v2_only = np.setdiff1d(lists["v2"], both)
neither = np.setdiff1d(neg, np.union1d(lists["v1"], lists["v2"]))
overlap_ref = pd.read_csv(RES / "nudge_list_overlap.csv").set_index("metric")["value"]["overlap between v1 and v2 top-10% lists"]
assert abs(len(both) / k - overlap_ref) < 1e-4
rows = []
for qname, r in [("both lists", both), ("v1 list only", v1_only), ("v2 list only", v2_only), ("neither", neither)]:
    rows.append({"quadrant": qname, "n": len(r), "share_of_list": len(r) / k, "mean_truth_p": truth_p[r].mean(),
                 "logged_in_30d_snapshot": (v1["days_since_last_login"].values[r] <= 30).mean(),
                 "preference_complete": users["truth_pref_complete"].values[r].mean(),
                 "mean_login_counts_snapshot": v1["login_counts"].values[r].mean()})
quad = pd.DataFrame(rows)
quad.round(4).to_csv(RES / "score_quadrants.csv", index=False)
for vname in scores:
    summary[f"quadrants: {vname} list threshold (min OOF score)"] = scores[vname][lists[vname]].min()
summary["quadrants: list size k"] = k
for _, r in quad.iterrows():
    summary[f"quadrants: {r['quadrant']} n"] = r["n"]
    summary[f"quadrants: {r['quadrant']} mean truth propensity"] = r["mean_truth_p"]
print("5. score_quadrants.csv")

s = pd.Series(summary, name="value")
s.to_frame().rename_axis("metric").to_csv(RES / "leakage_anatomy_summary.csv")
print("\n" + s.to_string())
