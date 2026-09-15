"""주제 매칭

부분 문자열 매칭이다. 제외 규칙은 발화 전체를 버리지 않고 걸린 키워드 히트 하나만 취소한다.
"경기도 가는 길에 날씨 어때"면 경기는 취소되고 날씨는 남는다.

불만 발화("최신으로 알려줘")에는 주제 키워드가 없다. 같은 세션 직전 유저 발화의 주제를 가져온다.
직전 발화가 없거나 그 발화도 미매칭이면 미매칭으로 둔다.
"""

import pandas as pd
from common import COMPLAINT, DATA, EXCLUDE_RULES, GZ, RES, TOPIC_KEYWORDS, TOPICS, UNMATCHED

pairs = pd.read_csv(DATA / "turn_pairs.csv.gz", keep_default_na=False)
text = pairs["text"]

hits = []
for topic, kws in TOPIC_KEYWORDS.items():
    for kw in kws:
        m = text.str.contains(kw, regex=False)
        for ex_kw, ex_pat in EXCLUDE_RULES:
            if ex_kw == kw:
                cancel = m & text.str.contains(ex_pat, regex=False)
                hits.append({"keyword": kw, "exclude": ex_pat, "cancelled": int(cancel.sum())})
                m &= ~cancel
        pairs[f"hit:{topic}:{kw}"] = m

exclude_hits = pd.DataFrame(hits).groupby(["keyword", "exclude"], as_index=False)["cancelled"].sum()
exclude_hits.to_csv(RES / "exclude_hits.csv", index=False)

hit_cols = [c for c in pairs.columns if c.startswith("hit:")]
H = pairs[hit_cols]
kw_len = pd.Series([len(c.split(":")[2]) for c in hit_cols], index=hit_cols)
topic_of = pd.Series([c.split(":")[1] for c in hit_cols], index=hit_cols)

# 여러 주제에 걸리면 가장 긴 키워드의 주제. 길이가 같으면 TOPICS 순서
order = sorted(hit_cols, key=lambda c: (-kw_len[c], list(TOPICS).index(topic_of[c])))
pairs["topic"] = UNMATCHED
for c in reversed(order):
    pairs.loc[H[c], "topic"] = topic_of[c]
pairs["n_hits"] = H.sum(axis=1)
pairs["short_only"] = (H & (kw_len <= 2)).sum(axis=1).eq(pairs["n_hits"]) & pairs["n_hits"].gt(0)

pairs["complaint"] = text.str.contains(COMPLAINT)
pairs = pairs.sort_values(["session_id", "ts", "turn_id"])
prev_topic = pairs.groupby("session_id")["topic"].shift(1)
inherit = pairs["complaint"] & pairs["topic"].eq(UNMATCHED) & prev_topic.notna()
pairs.loc[inherit, "topic"] = prev_topic[inherit]

out = pairs[["turn_id", "topic", "n_hits", "short_only", "complaint"]]
out.to_csv(DATA / "topic_match.csv.gz", index=False, compression=GZ)

nonempty = pairs["text"] != ""
matched = nonempty & (pairs["n_hits"] > 0)
share, short = matched.sum() / nonempty.sum(), pairs.loc[matched, "short_only"].mean()
print(f"matched share of non-empty {share:.4f}, short-only {short:.3f}")
print(f"complaints {pairs.complaint.sum()}, inherited topic {inherit.sum()}")
print(exclude_hits.to_string(index=False))
