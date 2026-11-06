"""ground truth 평가: 거르기, 군집별 precision/recall, 기타의 실제 구성, 후보별 실제 크기

truth_parts.csv 는 여기서만 읽음
"""

import pandas as pd
from common import CANDIDATES, CLUSTERS, DATA, LABEL, RES

truth = pd.read_csv(DATA / "truth_parts.csv", keep_default_na=False)
truth["msg_id"] = truth["uuid"] + ":" + truth["part"].astype(str)
msgs = pd.read_csv(DATA / "messages.csv", keep_default_na=False)
labels = pd.read_csv(DATA / "labels.csv", keep_default_na=False)
labels = labels.merge(pd.read_csv(DATA / "labels_ctx.csv", keep_default_na=False), on="msg_id")

stage = truth[["msg_id", "truth_source"]].merge(msgs[["msg_id", "drop"]], on="msg_id", how="left")
stage["result"] = (
    stage["drop"].map({"": "남음", "ide": "2차에서 빠짐", "bg_first": "2차에서 빠짐"}).fillna("1차에서 빠짐")
)
filt = pd.crosstab(stage["truth_source"], stage["result"]).reset_index()
filt.to_csv(RES / "eval_filter.csv", index=False)

df = labels.merge(truth[["msg_id", "truth_intent", "truth_form"]], on="msg_id", how="left")
assert df["truth_intent"].ne("").all(), "사람 메시지가 아닌 것이 남음"
df["truth"] = df["truth_intent"].where(df["truth_intent"].isin(CLUSTERS), "other")
sets = df["matches"].map(lambda m: set(m.split(";")) if m else set())


def pr(pred: pd.Series, true: pd.Series) -> tuple[int, float, float]:
    tp = int((pred & true).sum())
    n = int(pred.sum())
    return n, round(tp / n, 3) if n else float("nan"), round(tp / int(true.sum()), 3)


rows = []
for k in CLUSTERS:
    true = df["truth"] == k
    o = pr(sets.map(lambda s, k=k: k in s), true)
    f = pr(df["first"] == k, true)
    c = pr(df["ctx"] == k, true)
    rows.append(
        {
            "cluster": k,
            "label": LABEL[k],
            "truth_n": int(true.sum()),
            "overlap_n": o[0],
            "overlap_precision": o[1],
            "overlap_recall": o[2],
            "first_n": f[0],
            "first_precision": f[1],
            "first_recall": f[2],
            "ctx_n": c[0],
            "ctx_precision": c[1],
            "ctx_recall": c[2],
        }
    )
ev = pd.DataFrame(rows).sort_values("truth_n", ascending=False)
ev.to_csv(RES / "eval_clusters.csv", index=False)

other = df[df["first"] == "other"]
comp = other.assign(truth_label=other["truth_intent"].map(lambda t: LABEL.get(t, "기능 구현/수정 지시")))
comp = comp.groupby(["truth_intent", "truth_label", "truth_form"]).size().rename("messages").reset_index()
comp = comp.sort_values("messages", ascending=False)
comp["share_of_other"] = (comp["messages"] / len(other)).round(3)
comp.to_csv(RES / "eval_other.csv", index=False)

attached = df[df["ctx_source"].isin(["assistant", "previous"])]
summary = pd.DataFrame(
    [
        {"item": "첫 일치 정확도", "value": round(float((df["first"] == df["truth"]).mean()), 3)},
        {"item": "짧은 응답 붙인 뒤 정확도", "value": round(float((df["ctx"] == df["truth"]).mean()), 3)},
        {"item": "붙인 짧은 응답 수", "value": len(attached)},
        {"item": "붙인 짧은 응답 정확도", "value": round(float((attached["ctx"] == attached["truth"]).mean()), 3)},
        {
            "item": "직전 assistant 로 붙인 정확도",
            "value": round(
                float((attached["ctx"] == attached["truth"])[attached["ctx_source"] == "assistant"].mean()), 3
            ),
        },
        {
            "item": "앞 메시지로 붙인 정확도",
            "value": round(
                float((attached["ctx"] == attached["truth"])[attached["ctx_source"] == "previous"].mean()), 3
            ),
        },
        {"item": "기타 중 실제로는 군집이 있는 비율", "value": round(float((other["truth"] != "other").mean()), 3)},
        {"item": "실제 짧은 응답 수", "value": int((df["truth_form"] == "short").sum())},
    ]
)
summary.to_csv(RES / "eval_summary.csv", index=False)

cand = pd.read_csv(RES / "candidates.csv")
cand["truth"] = [int(df["truth"].isin(c).sum()) for _, _, c in CANDIDATES]
cand["rank_truth"] = cand["truth"].rank(ascending=False, method="min").astype(int)
cols = ["candidate", "name", "sum_overlap", "union_overlap", "first", "first_ctx", "truth", "rank_truth"]
cand[cols].to_csv(RES / "eval_candidates.csv", index=False)

# /clear 직전 메시지: 붙인 군집 x 실제 의도
cb = pd.read_csv(DATA / "clear_before.csv", keep_default_na=False).merge(df[["msg_id", "truth"]], on="msg_id")
pd.crosstab(cb["ctx"], cb["truth"]).reset_index().to_csv(RES / "eval_clear_context.csv", index=False)

print(filt.to_string(index=False))
print(ev.to_string(index=False))
print(comp.head(12).to_string(index=False))
print(summary.to_string(index=False))
print(cand[cols].to_string(index=False))
