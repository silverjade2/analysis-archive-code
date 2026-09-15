"""주제 키워드에 안 걸린 발화의 n-gram

형태소 분석기 없이 공백으로 자르고 흔한 조사만 뗀다. 최신성 신호 단어와 "알려줘", "틀어줘" 같은 요청 어미는
뺀다. 이것들을 두면 상위가 전부 "오늘", "지금"으로 채워져 누락 주제가 안 보인다.
"""

from collections import Counter

import pandas as pd
from common import DATA, FRESH_SIGNALS, RES, UNMATCHED

PARTICLES = ("은", "는", "을", "를", "에", "에서", "도", "로", "가")
STOP = {w for ws in FRESH_SIGNALS.values() for w in ws} | {
    "알려줘",
    "틀어줘",
    "뭐야",
    "어때",
    "해줘",
    "줘",
    "좀",
    "다시",
}

pairs = pd.read_csv(DATA / "turn_pairs.csv.gz", keep_default_na=False, usecols=["turn_id", "text"])
d = pairs.merge(pd.read_csv(DATA / "topic_match.csv.gz"), on="turn_id")
d = d[(d["text"] != "") & (d["topic"] == UNMATCHED)]


def tokens(s: str) -> list[str]:
    out = []
    for w in s.split():
        for p in PARTICLES:
            if len(w) > len(p) + 1 and w.endswith(p):
                w = w[: -len(p)]
                break
        if len(w) > 1 and w not in STOP:
            out.append(w)
    return out


cnt = Counter()
for s in d["text"]:
    tk = tokens(s)
    grams = set(tk) | {" ".join(tk[i : i + 2]) for i in range(len(tk) - 1)}
    cnt.update(grams)

# 동률은 문자열 순. set 순회 순서는 실행마다 달라진다
top = pd.DataFrame(sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:30], columns=["ngram", "n_turns"])
top["share_of_unmatched"] = top["n_turns"] / len(d)
top.to_csv(RES / "uncovered_ngram.csv", index=False)
print(len(d))
print(top.head(20).round(4).to_string(index=False))
