"""의도 군집 세기: 겹침 허용, 첫 일치, 규칙 순서를 바꾼 첫 일치

- 겹침 허용은 한 메시지가 여러 군집에 걸리면 모두 셈. 합이 메시지 수보다 큼
- 첫 일치는 RULES 순서대로 처음 걸린 군집 하나. 순서를 무작위로 N_PERM 번 바꿔 범위를 봄
"""

import numpy as np
import pandas as pd
from common import CLUSTERS, DATA, LABEL, RES, SEED, first_match, matches

N_PERM = 200

msgs = pd.read_csv(DATA / "messages.csv", keep_default_na=False)
msgs["matches"] = msgs["text"].map(lambda t: ";".join(matches(t)))
msgs["first"] = msgs["text"].map(first_match)
human = msgs[msgs["drop"] == ""].copy()
human[["msg_id", "project", "matches", "first"]].to_csv(DATA / "labels.csv", index=False)

hit = {k: human["matches"].str.split(";").map(lambda m, k=k: k in m) for k in CLUSTERS}
counts = pd.DataFrame(
    {
        "cluster": CLUSTERS,
        "label": [LABEL[k] for k in CLUSTERS],
        "overlap_n": [int(hit[k].sum()) for k in CLUSTERS],
        "first_n": [int((human["first"] == k).sum()) for k in CLUSTERS],
        "first_n_before_2nd": [int((msgs["first"] == k).sum()) for k in CLUSTERS],
    }
)

rng = np.random.default_rng(SEED)
match_sets = [m.split(";") if m else [] for m in human["matches"]]
perm = []
for _ in range(N_PERM):
    order = list(rng.permutation(CLUSTERS))
    rank = {k: i for i, k in enumerate(order)}
    firsts = pd.Series([min(m, key=rank.get) if m else "other" for m in match_sets])
    perm.append(firsts.value_counts().reindex(CLUSTERS, fill_value=0))
perm = pd.DataFrame(perm)
counts["perm_min"] = perm.min().values
counts["perm_median"] = perm.median().values
counts["perm_max"] = perm.max().values
counts = counts.sort_values("overlap_n", ascending=False)

other = pd.DataFrame(
    [
        {
            "cluster": "other",
            "label": LABEL["other"],
            "overlap_n": int((human["matches"] == "").sum()),
            "first_n": int((human["first"] == "other").sum()),
            "first_n_before_2nd": int((msgs["first"] == "other").sum()),
            "perm_min": int((human["matches"] == "").sum()),
            "perm_median": int((human["matches"] == "").sum()),
            "perm_max": int((human["matches"] == "").sum()),
        }
    ]
)
counts = pd.concat([counts, other], ignore_index=True)
counts.to_csv(RES / "cluster_counts.csv", index=False)

n_hits = human["matches"].map(lambda m: len(m.split(";")) if m else 0)
mult = n_hits.clip(upper=3).value_counts().sort_index()
mult = pd.DataFrame({"n_matches": ["0", "1", "2", "3+"], "messages": mult.reindex([0, 1, 2, 3], fill_value=0).values})
mult["share"] = (mult["messages"] / len(human)).round(3)
mult.to_csv(RES / "match_multiplicity.csv", index=False)

summary = pd.DataFrame(
    [
        {"item": "사람 메시지", "value": len(human)},
        {"item": "1차 통과 메시지", "value": len(msgs)},
        {"item": "사람 메시지가 있는 본 세션", "value": int(human["session"].nunique())},
        {"item": "겹침 허용 합", "value": int(counts["overlap_n"].iloc[:-1].sum())},
        {"item": "군집 2개 이상 걸린 메시지", "value": int((n_hits >= 2).sum())},
        {"item": "어느 군집에도 안 걸린 메시지", "value": int((n_hits == 0).sum())},
    ]
)
summary.to_csv(RES / "count_summary.csv", index=False)

by_project = pd.DataFrame({k: hit[k].groupby(human["project"]).sum() for k in CLUSTERS}).T
by_project.index.name = "cluster"
by_project.to_csv(RES / "cluster_by_project.csv")

print(counts[["cluster", "overlap_n", "first_n", "first_n_before_2nd", "perm_min", "perm_max"]].to_string(index=False))
print(summary.to_string(index=False))
