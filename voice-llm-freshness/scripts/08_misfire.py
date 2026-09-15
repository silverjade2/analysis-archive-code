"""오호출 상한과 하한, 분모를 바꾼 뒤의 비율

하한은 기기가 스스로 소음이나 오호출로 판정한 발화뿐이다. 상한은 거기에 빈 발화와 "짧고 의도 미분류이고
fallback"을 더한다. 짧은 기준 3글자는 파편 표현(아니, 엄마, 이거)을 덮는 가장 작은 값이다.

규칙은 truth_misfire를 보지 않고 만든다. 규칙이 정해진 다음에만 정답과 대조한다.
"""

import pandas as pd
from common import DATA, RES

pairs = pd.read_csv(DATA / "turn_pairs.csv.gz", keep_default_na=False, usecols=["turn_id", "text"])
d = pairs.merge(pd.read_csv(DATA / "turn_flags.csv.gz", keep_default_na=False), on="turn_id")

short = d["text"].str.replace(" ", "").str.len().between(1, 3)
rules = pd.DataFrame(
    {
        "self_judged": d["fallback_cat"].isin(["소음 판정", "오호출 판정"]),
        "empty": d["empty"],
        "short_other_fallback": short & d["intent"].eq("기타") & d["state"].eq("fallback"),
    }
)
rules["lower"] = rules["self_judged"]
rules["upper"] = rules[["self_judged", "empty", "short_other_fallback"]].any(axis=1)

truth = pd.read_csv(DATA / "logs.csv.gz", usecols=["row_id", "truth_misfire"]).set_index("row_id")["truth_misfire"]
t = d["turn_id"].map(truth).astype(bool)

bounds = []
for name in rules.columns:
    r = rules[name]
    bounds.append(
        {
            "rule": name,
            "share_all_turns": r.mean(),
            "share_nonempty": r[~d["empty"]].mean(),
            "precision": t[r].mean(),
            "recall": r[t].mean(),
        }
    )
bounds.append({"rule": "truth_misfire", "share_all_turns": t.mean(), "share_nonempty": t[~d["empty"]].mean()})
pd.DataFrame(bounds).to_csv(RES / "misfire_bounds.csv", index=False)

base = ~d["empty"]
denoms = {
    "nonempty": base,
    "nonempty_minus_lower": base & ~rules["lower"],
    "nonempty_minus_upper": base & ~rules["upper"],
    "nonempty_minus_truth": base & ~t,
}
rates = []
for name, m in denoms.items():
    x = d[m]
    rates.append(
        {
            "denominator": name,
            "n": len(x),
            "fallback_rate": (x["state"] == "fallback").mean(),
            "none_rate": (x["state"] == "none").mean(),
            "info_query_share": (x["intent"] == "정보질의").mean(),
            "fresh_share": x["fresh"].mean(),
        }
    )
rates = pd.DataFrame(rates)
rates.to_csv(RES / "rates_by_denominator.csv", index=False)
print(pd.DataFrame(bounds).round(4).to_string(index=False))
print(rates.round(4).to_string(index=False))
