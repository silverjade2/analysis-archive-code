"""단계 분포, 점수 단계 vs 항목 단계 일치율, 희석/과판정 집계, k 민감도

- 일치율 분모는 6항목 완비 행. 결측 행은 missing_policy.csv에 따로
"""

import pandas as pd
from common import DATA, ITEMS, PARAMS, RES
from scoring import final_stage, score_frame, stage_from_score

df = pd.read_csv(DATA / "scored_synthetic.csv")
status_cols = [f"status_{i}" for i in ITEMS]
complete = df[df.missing_count == 0].copy()

METHODS = {
    "current": "stage_current",
    "current_penalized": "stage_penalized",
    "v2": "stage_v2",
    "v2_capped": "final_stage",
}
STAGES = ["안정", "주의", "경고", "데이터 없음"]

dist = []
for m, col in METHODS.items():
    vc = df[col].value_counts()
    dist.append(dict(method=m, **{s: int(vc.get(s, 0)) for s in STAGES}))
vc = df.item_stage_original.value_counts()
dist.append(dict(method="item_stage_original", **{s: int(vc.get(s, 0)) for s in STAGES}))
vc = df.item_stage_v2.value_counts()
dist.append(dict(method="item_stage_v2", **{s: int(vc.get(s, 0)) for s in STAGES}))
dist = pd.DataFrame(dist)
dist.to_csv(RES / "stage_distribution.csv", index=False)

agree = []
for m, col in METHODS.items():
    same = complete[col] == complete.item_stage_v2
    lenient = complete.apply(lambda r: final_stage(r[col], r.item_stage_v2) != r[col], axis=1)
    harsher = complete.apply(lambda r: final_stage(r[col], r.item_stage_v2) != r.item_stage_v2, axis=1)
    agree.append(
        dict(
            method=m,
            n=len(complete),
            agree=int(same.sum()),
            agree_rate=round(same.mean(), 4),
            score_more_lenient=int(lenient.sum()),
            score_harsher=int(harsher.sum()),
        )
    )
agree = pd.DataFrame(agree)
agree.to_csv(RES / "agreement.csv", index=False)

cross = []
for m, col in METHODS.items():
    ct = pd.crosstab(complete.item_stage_v2, complete[col]).reindex(index=STAGES[:3], columns=STAGES[:3], fill_value=0)
    for item_stage in STAGES[:3]:
        cross.append(
            dict(method=m, item_stage=item_stage, **{f"score_{s}": int(ct.loc[item_stage, s]) for s in STAGES[:3]})
        )
pd.DataFrame(cross).to_csv(RES / "stage_crosstab.csv", index=False)

n_warn = (df[status_cols] == "경고").sum(axis=1)
n_caut = (df[status_cols] == "주의").sum(axis=1)
single = df[(n_warn == 1) & (n_caut == 0) & (df.missing_count == 0)]
caution_only = df[(n_warn == 0) & (n_caut >= 1) & (df.missing_count == 0)]
one_off = df[((n_warn + n_caut) == 1) & (df.missing_count == 0)]
dil = pd.DataFrame(
    [
        dict(
            group="single_warning_rest_normal",
            n=len(single),
            v2_min=single.score_v2.min(),
            v2_mean=single.score_v2.mean(),
            v2_stage_stable=int((single.stage_v2 == "안정").sum()),
            current_min=single.score_current.min(),
            current_stage_stable=int((single.stage_current == "안정").sum()),
            penalized_stage_warning=int((single.stage_penalized == "경고").sum()),
        ),
        dict(
            group="caution_only",
            n=len(caution_only),
            v2_min=caution_only.score_v2.min(),
            v2_mean=caution_only.score_v2.mean(),
            v2_stage_stable=int((caution_only.stage_v2 == "안정").sum()),
            current_min=caution_only.score_current.min(),
            current_stage_stable=int((caution_only.stage_current == "안정").sum()),
            penalized_stage_warning=int((caution_only.stage_penalized == "경고").sum()),
        ),
        dict(
            group="one_item_off_only",
            n=len(one_off),
            v2_min=one_off.score_v2.min(),
            v2_mean=one_off.score_v2.mean(),
            v2_stage_stable=int((one_off.stage_v2 == "안정").sum()),
            current_min=one_off.score_current.min(),
            current_stage_stable=int((one_off.stage_current == "안정").sum()),
            penalized_stage_warning=int((one_off.stage_penalized == "경고").sum()),
        ),
    ]
).round(2)
dil.to_csv(RES / "dilution_groups.csv", index=False)

band = caution_only[(caution_only.score_current >= 50) & (caution_only.score_current < 200 / 3)]
over = pd.DataFrame(
    [
        dict(
            n_caution_only=len(caution_only),
            n_base_below_50=int((caution_only.score_current < 50).sum()),
            n_base_50_to_66_7=len(band),
            n_penalized_below_50=int((caution_only.score_penalized < 50).sum()),
            n_v2_below_50=int((caution_only.score_v2 < 50).sum()),
            n_v2_capped_warning=int((caution_only.final_stage == "경고").sum()),
        )
    ]
)
over.to_csv(RES / "penalty_overjudge.csv", index=False)

by_stage = complete.groupby("item_stage_v2").score_v2.describe()[["count", "25%", "50%", "75%", "min"]].round(2)
by_stage.to_csv(RES / "score_by_item_stage.csv")

summ = df[["score_current", "score_penalized", "score_v2"]].describe().T.round(2)
summ.index.name = "score"
summ.to_csv(RES / "score_summary.csv")

miss = pd.DataFrame(
    [
        dict(
            n_rows=len(df),
            n_with_missing=int((df.missing_count > 0).sum()),
            n_all_missing=int((df.missing_count == len(ITEMS)).sum()),
            item_stage_original_no_data=int((df.item_stage_original == "데이터 없음").sum()),
            item_stage_v2_no_data=int((df.item_stage_v2 == "데이터 없음").sum()),
        )
    ]
)
miss.to_csv(RES / "missing_policy.csv", index=False)

users = pd.read_csv(DATA / "synthetic_vitals.csv")
sens = []
for k in (4, 6, 8):
    s = score_frame(users, k=k)
    s = s[s.missing_count == 0]
    st = s.score_v2.map(stage_from_score)
    sens.append(
        dict(
            k=k,
            n=len(s),
            v2_agree_rate=round((st == s.item_stage_v2).mean(), 4),
            capped_agree_rate=round((s.final_stage == s.item_stage_v2).mean(), 4),
            v2_mean=round(s.score_v2.mean(), 2),
            v2_median=round(s.score_v2.median(), 2),
        )
    )
pd.DataFrame(sens).to_csv(RES / "k_sensitivity.csv", index=False)

status_share = pd.DataFrame(
    {i: df[f"status_{i}"].value_counts(normalize=True).reindex(STAGES, fill_value=0) for i in ITEMS}
).T.round(3)
status_share.index.name = "item"
status_share.to_csv(RES / "item_status_share.csv")

print(dist.to_string(index=False))
print(agree.to_string(index=False))
print(dil.to_string(index=False))
print(over.to_string(index=False))
print(pd.DataFrame(sens).to_string(index=False))
print(miss.to_string(index=False))
print(sum(PARAMS[i]["weight"] for i in ITEMS), "total weight")
