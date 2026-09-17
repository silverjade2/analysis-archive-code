"""turn_flags: 응답 상태(normal/fallback/none), 최신성 신호, 의도"""

import pandas as pd
from common import DATA, FRESH_SIGNALS, GZ, INTENT_RULES, RES, fallback_lookup

pairs = pd.read_csv(DATA / "turn_pairs.csv.gz", keep_default_na=False)
lut = fallback_lookup()

pairs["state"] = "normal"
pairs.loc[pairs["resp_row_id"] == "", "state"] = "none"
# 사전 밖 응답은 전부 normal. 낡은 답이 여기서 정상으로 잡힘
pairs.loc[pairs["resp_text"].isin(lut), "state"] = "fallback"
pairs["fallback_cat"] = pairs["resp_text"].map(lut).fillna("")

pairs["fresh_cat"] = ""
for cat in reversed(list(FRESH_SIGNALS)):  # 앞 카테고리 우선. "어제 누가 이겼어"는 시점 아니고 결과
    m = pairs["text"].str.contains("|".join(FRESH_SIGNALS[cat]))
    pairs.loc[m, "fresh_cat"] = cat
pairs["fresh"] = pairs["fresh_cat"] != ""

pairs["intent"] = "기타"
for name, pat in reversed(INTENT_RULES):
    pairs.loc[pairs["text"].str.contains(pat), "intent"] = name

pairs["empty"] = pairs["text"] == ""
flags = pairs[["turn_id", "empty", "state", "fallback_cat", "fresh", "fresh_cat", "intent"]]
flags.to_csv(DATA / "turn_flags.csv.gz", index=False, compression=GZ)

ne = pairs[~pairs["empty"]]
share = pd.crosstab(ne["fresh"], ne["state"], normalize="index").reset_index()
share.insert(1, "n_share", ne["fresh"].value_counts(normalize=True).reindex(share["fresh"]).to_numpy())
share.to_csv(RES / "state_by_fresh.csv", index=False)

intent = pd.crosstab(ne["intent"], ne["state"], normalize="index").reset_index()
intent.insert(1, "n_share", ne["intent"].value_counts(normalize=True).reindex(intent["intent"]).to_numpy())
intent.to_csv(RES / "state_by_intent.csv", index=False)

fb = ne[ne["state"] == "fallback"]
phr = fb["resp_text"].value_counts().rename_axis("phrase").reset_index(name="n")
phr["share"] = phr["n"] / len(fb)
phr["category"] = phr["phrase"].map(lut)
phr["rank"] = range(1, len(phr) + 1)
phr.to_csv(RES / "fallback_phrases.csv", index=False)

cat = fb["fallback_cat"].value_counts(normalize=True).rename_axis("category").reset_index(name="share")
cat.to_csv(RES / "fallback_categories.csv", index=False)

overview = pd.DataFrame(
    {
        "metric": [
            "empty_share",
            "normal",
            "fallback",
            "none",
            "fresh_share",
            "fresh_time_only_share",
            "fresh_result_schedule_share",
        ],
        "value": [
            pairs["empty"].mean(),
            (ne["state"] == "normal").mean(),
            (ne["state"] == "fallback").mean(),
            (ne["state"] == "none").mean(),
            ne["fresh"].mean(),
            (ne.loc[ne["fresh"], "fresh_cat"] == "시점").mean(),
            ne.loc[ne["fresh"], "fresh_cat"].isin(["결과", "일정"]).mean(),
        ],
    }
)
overview.to_csv(RES / "overview.csv", index=False)
print(overview.round(4).to_string(index=False))
print(share.round(4).to_string(index=False))
print(intent.round(4).to_string(index=False))
print(
    phr.head(15)[["rank", "share", "category"]].round(4).to_string(index=False),
    "top15",
    round(phr.head(15).share.sum(), 4),
)
print(cat.round(4).to_string(index=False))
