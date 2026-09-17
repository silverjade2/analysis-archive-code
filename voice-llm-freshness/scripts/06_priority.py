"""주제별 점수표 두 벌 (수요 x 최신성 요구율 x 실패율, 순위용)

- priority_matched: 처음 설계. 주제 붙은 발화만 분모
- priority_all: 원문 있는 발화 전체가 분모, 미매칭을 한 줄로
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
