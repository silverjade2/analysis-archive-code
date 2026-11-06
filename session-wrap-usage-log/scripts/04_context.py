"""기타 군집 들여다보기: 짧은 응답에 앞 턴의 군집을 붙임, /clear 직전 메시지의 군집

- 짧은 응답(SHORT_LEN 이하)이 기타면 직전 assistant 텍스트를 같은 규칙으로 분류
- 거기서도 기타면 같은 세션 바로 앞 사람 메시지의 군집
"""

import pandas as pd
from common import CLUSTERS, DATA, LABEL, RES, first_match, is_short

msgs = pd.read_csv(DATA / "messages.csv", keep_default_na=False)
labels = pd.read_csv(DATA / "labels.csv", keep_default_na=False)
df = labels.merge(msgs[["msg_id", "file", "order", "text", "prev_assistant"]], on="msg_id").sort_values(
    ["file", "order"]
)
df["short"] = df["text"].map(is_short)

ctx, source = [], []
prev = {}
for row in df.itertuples():
    label, src = row.first, "rule"
    if label == "other" and row.short:
        label, src = first_match(row.prev_assistant), "assistant"
        if label == "other":
            label, src = prev.get(row.file, "other"), "previous"
        if label == "other":
            src = "none"
    ctx.append(label)
    source.append(src)
    prev[row.file] = label
df["ctx"] = ctx
df["ctx_source"] = source
df[["msg_id", "short", "ctx", "ctx_source"]].to_csv(DATA / "labels_ctx.csv", index=False)

other = df[df["first"] == "other"]
breakdown = pd.DataFrame(
    [
        {"part": "기타 전체", "messages": len(other)},
        {"part": "짧은 응답", "messages": int(other["short"].sum())},
        {"part": "짧지 않은 메시지", "messages": int((~other["short"]).sum())},
        {"part": "짧은 응답 중 직전 assistant 로 붙음", "messages": int((other["ctx_source"] == "assistant").sum())},
        {"part": "짧은 응답 중 앞 메시지로 붙음", "messages": int((other["ctx_source"] == "previous").sum())},
        {"part": "짧은 응답 중 못 붙음", "messages": int((other["ctx_source"] == "none").sum())},
    ]
)
breakdown["share_of_all"] = (breakdown["messages"] / len(df)).round(3)
breakdown.to_csv(RES / "other_breakdown.csv", index=False)

keys = CLUSTERS + ["other"]
cc = pd.DataFrame(
    {
        "cluster": keys,
        "label": [LABEL[k] for k in keys],
        "first_n": df["first"].value_counts().reindex(keys, fill_value=0).values,
        "ctx_n": df["ctx"].value_counts().reindex(keys, fill_value=0).values,
    }
)
cc["moved_in"] = cc["ctx_n"] - cc["first_n"]
cc.to_csv(RES / "context_counts.csv", index=False)

slash = pd.read_csv(DATA / "slash.csv", keep_default_na=False)
clears = slash[(slash["command"] == "/clear") & ~slash["file"].str.contains("/subagents/")]
before, before_id = [], []
for c in clears.itertuples():
    earlier = df[(df["file"] == c.file) & (df["order"] < c.order)]
    before.append(earlier["ctx"].iloc[-1] if len(earlier) else "other")
    before_id.append(earlier["msg_id"].iloc[-1] if len(earlier) else "")
pd.DataFrame({"msg_id": before_id, "ctx": before}).to_csv(DATA / "clear_before.csv", index=False)
clear_ctx = pd.Series(before).value_counts().rename_axis("cluster").rename("clears").reset_index()
clear_ctx["label"] = clear_ctx["cluster"].map(LABEL)
clear_ctx["share"] = (clear_ctx["clears"] / len(clears)).round(3)
clear_ctx.to_csv(RES / "clear_context.csv", index=False)

print(breakdown.to_string(index=False))
print(cc.to_string(index=False))
print(clear_ctx.to_string(index=False))
