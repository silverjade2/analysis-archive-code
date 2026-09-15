"""주제별 우선순위 점수

점수는 단위 없는 곱이라 순위에만 쓴다. 분모를 두 번 잡는다. priority_all은 원문이 있는 유저 발화 전체이고
미매칭을 한 줄로 세운다. priority_matched는 주제가 붙은 발화만 분모로 둔, 처음 설계한 표다.
"""

import pandas as pd
from common import DATA, RES, TOPICS, UNMATCHED

pairs = pd.read_csv(DATA / "turn_pairs.csv.gz", keep_default_na=False, usecols=["turn_id", "text"])
d = pairs.merge(pd.read_csv(DATA / "topic_match.csv.gz"), on="turn_id").merge(
    pd.read_csv(DATA / "turn_flags.csv.gz", keep_default_na=False), on="turn_id"
)
d = d[~d["empty"]]


def score_table(frame: pd.DataFrame, denom: int) -> pd.DataFrame:
    t = frame.groupby("topic").agg(
        n=("turn_id", "size"), fresh_rate=("fresh", "mean"), fail_rate=("state", lambda s: (s != "normal").mean())
    )
    t["demand_share"] = t["n"] / denom
    t["score"] = t["demand_share"] * t["fresh_rate"] * t["fail_rate"]
    t["rank"] = t["score"].rank(ascending=False, method="min").astype(int)
    t["validity_h"] = t.index.map(TOPICS)
    return t.sort_values("rank").reset_index()[
        ["topic", "n", "demand_share", "fresh_rate", "fail_rate", "score", "rank", "validity_h"]
    ]


matched = d[d["topic"] != UNMATCHED]
priority_matched = score_table(matched, len(matched))
priority_all = score_table(d, len(d))
top_topic = priority_all.loc[priority_all["topic"] != UNMATCHED, "score"].max()
priority_all["score_vs_top_topic"] = priority_all["score"] / top_topic
priority_matched.to_csv(RES / "priority_matched.csv", index=False)
priority_all.to_csv(RES / "priority_all.csv", index=False)

fresh = d[d["fresh"]]
coverage = pd.DataFrame(
    {
        "metric": ["matched_share", "short_only_share_of_matched", "unmatched_share_of_fresh"],
        "value": [(d["topic"] != UNMATCHED).mean(), matched["short_only"].mean(), (fresh["topic"] == UNMATCHED).mean()],
    }
)
coverage.to_csv(RES / "coverage.csv", index=False)
print(coverage.round(4).to_string(index=False))
print(priority_matched.round(4).to_string(index=False))
print(priority_all.round(4).to_string(index=False))
