"""미매칭 발화 1~2gram 상위 30. 형태소 분석기 없이 공백 분리 + 조사 제거"""

from collections import Counter

import pandas as pd
from common import DATA, FRESH_SIGNALS, RES, UNMATCHED

PARTICLES = ("은", "는", "을", "를", "에", "에서", "도", "로", "가")
STOP = {w for ws in FRESH_SIGNALS.values() for w in ws} | {  # 안 빼면 상위가 "오늘", "지금"으로 채워짐
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
