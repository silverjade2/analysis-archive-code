"""사람이 입력한 메시지 거르기, 슬래시 명령과 스킬 호출 세기

- 1차: tool_result, 서브에이전트 기록, isMeta, 슬래시 명령, 자동 삽입 접두어(AUTO) 제외
- 2차: 1차를 통과한 IDE 자동 삽입, 다른 세션이 띄운 세션의 첫 지시문 제외
"""

import collections
import re

import pandas as pd
from common import AUTO, DATA, IDE, PROJECT_NAME, RES, read_transcripts

removed = collections.Counter()
part_types = collections.Counter()
skills = collections.Counter()
rows, slash = [], []
last_assistant: dict[str, str] = {}
bg_seen: set[str] = set()

for order, (file, d) in enumerate(read_transcripts()):
    project = file.split("/")[0]
    sub = "/subagents/" in file
    if d["type"] == "assistant":
        for c in d["message"]["content"]:
            if c["type"] == "tool_use" and c["name"] == "Skill":
                skills[c["input"]["skill"]] += 1
            elif c["type"] == "text":
                last_assistant[file] = c["text"]
        continue
    content = d["message"]["content"]
    parts = [{"type": "text", "text": content}] if isinstance(content, str) else content
    for i, x in enumerate(parts):
        part_types[x["type"]] += 1
        if x["type"] != "text":
            continue
        t = x["text"]
        cmd = re.search(r"<command-name>(.*?)</command-name>", t)
        if sub or d.get("isSidechain"):
            removed["서브에이전트 기록"] += 1
            continue
        if d.get("isMeta"):
            removed["isMeta"] += 1
            continue
        if cmd:
            args = re.search(r"<command-args>(.*?)</command-args>", t, re.S).group(1).strip()
            slash.append({"file": file, "project": project, "order": order, "command": cmd.group(1), "args": args})
            if not args:
                removed["슬래시 명령"] += 1
                continue
            t = f"{cmd.group(1)} {args}"
        if t.strip().startswith(AUTO):
            removed["자동 삽입"] += 1
            continue
        t = re.sub(r"<system-reminder>.*?</system-reminder>", "", t, flags=re.S).strip()
        if not t:
            removed["빈 텍스트"] += 1
            continue
        drop = ""
        if t.startswith(IDE):
            drop = "ide"
        elif d.get("sessionKind") == "bg" and file not in bg_seen:
            drop = "bg_first"
        if d.get("sessionKind") == "bg":
            bg_seen.add(file)
        rows.append(
            {
                "msg_id": f"{d['uuid']}:{i}",
                "file": file,
                "project": project,
                "session": d["sessionId"],
                "order": order,
                "ts": d["timestamp"],
                "text": t,
                "prev_assistant": last_assistant.get(file, ""),
                "drop": drop,
            }
        )

msgs = pd.DataFrame(rows)
msgs.to_csv(DATA / "messages.csv", index=False)
pd.DataFrame(slash).to_csv(DATA / "slash.csv", index=False)

RES.mkdir(parents=True, exist_ok=True)
total = sum(part_types.values())
steps = [("user 내용 조각", 0), ("tool_result", part_types["tool_result"]), ("이미지", part_types["image"])]
steps += [(k, removed[k]) for k in ["서브에이전트 기록", "isMeta", "슬래시 명령", "자동 삽입", "빈 텍스트"]]
steps += [
    ("IDE 자동 삽입", int((msgs["drop"] == "ide").sum())),
    ("다른 세션이 쓴 첫 지시문", int((msgs["drop"] == "bg_first").sum())),
]
left, funnel = total, []
for name, n in steps:
    left -= n
    funnel.append({"step": name, "removed": n, "remaining": left})
funnel = pd.DataFrame(funnel)
funnel["removed_share"] = (funnel["removed"] / total).round(3)
funnel["remaining_share"] = (funnel["remaining"] / total).round(3)
funnel.to_csv(RES / "funnel.csv", index=False)
pd.Series(part_types, name="n").rename_axis("type").reset_index().to_csv(RES / "part_types.csv", index=False)

sl = pd.DataFrame(slash)
sl[~sl["file"].str.contains("/subagents/")].groupby("command").size().rename("n").sort_values(
    ascending=False
).reset_index().to_csv(RES / "slash_commands.csv", index=False)
pd.Series(skills, name="calls").rename_axis("skill").sort_values(ascending=False).reset_index().to_csv(
    RES / "skill_calls.csv", index=False
)

human = msgs[msgs["drop"] == ""]
proj = human.groupby("project").agg(messages=("msg_id", "size"), sessions=("session", "nunique")).reset_index()
proj["name"] = proj["project"].map(PROJECT_NAME)
proj = proj.sort_values("messages", ascending=False)
proj[["project", "name", "messages", "sessions"]].to_csv(RES / "project_messages.csv", index=False)

print(funnel.to_string(index=False))
print(f"1차 통과 {len(msgs)}, 사람 메시지 {len(human)}")
