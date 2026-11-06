"""후보 도구 비교: 덮는 메시지 수를 세는 방법 넷, 퍼진 프로젝트 수, 스킬 호출 대비 대화 건수"""

import pandas as pd
from common import CANDIDATES, DATA, LABEL, MIN_PROJECT_MSGS, RES

labels = pd.read_csv(DATA / "labels.csv", keep_default_na=False)
labels = labels.merge(pd.read_csv(DATA / "labels_ctx.csv", keep_default_na=False), on="msg_id")
sets = labels["matches"].map(lambda m: set(m.split(";")) if m else set())

rows = []
for key, name, clusters in CANDIDATES:
    union = sets.map(lambda s, c=clusters: bool(s & set(c)))
    per_project = union.groupby(labels["project"]).sum()
    rows.append(
        {
            "candidate": key,
            "name": name,
            "clusters": ";".join(clusters),
            "sum_overlap": int(sum(sets.map(lambda s, k=k: k in s).sum() for k in clusters)),
            "union_overlap": int(union.sum()),
            "first": int(labels["first"].isin(clusters).sum()),
            "first_ctx": int(labels["ctx"].isin(clusters).sum()),
            "projects": int((per_project >= MIN_PROJECT_MSGS).sum()),
        }
    )
cand = pd.DataFrame(rows)
cand["double_counted"] = cand["sum_overlap"] - cand["union_overlap"]
for m in ["sum_overlap", "union_overlap", "first", "first_ctx"]:
    cand[f"rank_{m}"] = cand[m].rank(ascending=False, method="min").astype(int)
cand.to_csv(RES / "candidates.csv", index=False)

# 군집을 맡는 스킬. 커밋/푸시는 맡는 스킬이 없음
SKILL_OF = {
    "commit": "",
    "report": "artifact-design",
    "ui": "frontend-design",
    "wording": "term-check",
    "notebook": "notebook-check",
    "sql": "sql-check",
}
calls = pd.read_csv(RES / "skill_calls.csv").set_index("skill")["calls"]
sk = pd.DataFrame(
    [
        {
            "cluster": c,
            "label": LABEL[c],
            "messages": int((labels["first"] == c).sum()),
            "skill": s,
            "calls": int(calls.get(s, 0)) if s else 0,
        }
        for c, s in SKILL_OF.items()
    ]
)
sk.to_csv(RES / "skill_vs_dialog.csv", index=False)

print(cand.to_string(index=False))
print(sk.to_string(index=False))
