"""truth_stale 대비 proxy 3종(최신성 신호, 불만, 재시도)의 recall / precision. 정답 컬럼은 07, 08만 읽음"""

import pandas as pd
from common import DATA, RES, TOPICS, UNMATCHED

logs = pd.read_csv(
    DATA / "logs.csv.gz", usecols=["row_id", "truth_misfire", "truth_topic", "truth_fresh", "truth_stale"]
)
pairs = pd.read_csv(DATA / "turn_pairs.csv.gz", keep_default_na=False, usecols=["turn_id", "session_id", "ts", "text"])
d = (
    pairs.merge(pd.read_csv(DATA / "topic_match.csv.gz"), on="turn_id")
    .merge(pd.read_csv(DATA / "turn_flags.csv.gz", keep_default_na=False), on="turn_id")
    .merge(pd.read_csv(DATA / "turn_retry.csv.gz"), on="turn_id")
    .merge(logs.rename(columns={"row_id": "turn_id"}), on="turn_id")
    .sort_values(["session_id", "ts", "turn_id"])
)
d["truth_topic"] = d["truth_topic"].fillna("")
# 불만은 낡은 답 다음 턴에 나옴
d["next_complaint"] = d.groupby("session_id")["complaint"].shift(-1, fill_value=False).astype(bool)
d = d[~d["empty"]]

rows = []
for name, g in [("전체", d)] + [(t, d[d["topic"] == t]) for t in list(TOPICS) + [UNMATCHED]]:
    stale = g["truth_stale"] == 1
    row = {"topic": name, "n": len(g), "stale_rate": stale.mean()}
    for proxy in ["fresh", "next_complaint", "retry"]:
        flag = g[proxy].astype(bool)
        row[f"{proxy}_rate"] = flag.mean()
        row[f"{proxy}_recall"] = flag[stale].mean() if stale.any() else float("nan")
        row[f"{proxy}_precision"] = stale[flag].mean() if flag.any() else float("nan")
    rows.append(row)
oracle = pd.DataFrame(rows)
oracle.to_csv(RES / "oracle_compare.csv", index=False)

cov = (
    d[d["truth_topic"] != ""]
    .assign(hit=lambda x: x["topic"] == x["truth_topic"])
    .groupby("truth_topic")
    .agg(n=("hit", "size"), keyword_coverage=("hit", "mean"), stale_rate=("truth_stale", "mean"))
    .reset_index()
)
cov.to_csv(RES / "coverage_by_truth_topic.csv", index=False)
print(oracle.round(4).to_string(index=False))
print(cov.round(4).to_string(index=False))
