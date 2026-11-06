"""가상 대화 기록 생성: 프로젝트 10개, 30일, 본 세션 + 서브에이전트 기록

- 형식은 Claude Code 대화 기록 jsonl 을 줄인 것. 원래 파일 경로는 레코드의 _file 에
- 사람 메시지의 의도와 형태는 data/truth_parts.csv 에만 (truth_intent, truth_form)
"""

import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from common import DATA, N_DAYS, SEED, START, TRANSCRIPTS

rng = np.random.default_rng(SEED)

# 세션 수, IDE 세션 비율, 저장소 여부, 작업 주제 비율. 커밋과 세션 마감은 작업 흐름에서 따로 넣음
PLAN = {
    "analysis-project": (
        75,
        0.8,
        True,
        "sql 16, relay 14, notebook 9, report 9, question 10, summary 7, build 7, mac 7, "
        "files 5, tooling 5, wording 4, ui 2",
    ),
    "portfolio-site": (
        52,
        0.2,
        True,
        "report 16, build 12, wording 10, tooling 10, ui 9, question 9, career 6, files 4, mac 4",
    ),
    "web-service": (38, 0.5, True, "build 28, ui 18, question 10, tooling 9, mac 5, report 4, files 4"),
    "job-scraper": (29, 0.3, True, "career 20, build 20, scrape 12, ui 7, question 7, tooling 6, files 4"),
    "family-app": (11, 0.4, True, "build 30, ui 25, question 10, files 6, mac 5"),
    "booking-scraper": (7, 0.3, True, "scrape 25, build 25, question 8, mac 8, files 4"),
    "writing-tool": (6, 0.3, True, "build 25, wording 20, question 12, tooling 8, ui 8"),
    "calendar": (5, 0.3, True, "build 30, ui 20, question 12, tooling 8"),
    "photo-select": (4, 0.0, False, "files 50, question 20, build 15, ui 5"),  # 저장소 없는 폴더
    "budget": (2, 0.3, True, "build 45, ui 20, question 10"),
}
N_BG = 12  # 다른 세션이 띄운 세션. 첫 지시문은 Claude 가 씀
TASKS = 2.0  # 세션당 작업 수 = 1 + Poisson
MSGS_PER_TASK = 1.0  # 작업당 메시지 수 = 1 + Poisson
COMMIT_AFTER_TASK = 0.3  # 작업 하나 끝날 때 커밋/푸시 요청
COMMIT_AT_END = 0.35
WRAP_AT_END = 0.65  # 저장소 없는 폴더는 0.2
CLEANUP_AT_END = 0.15  # 켜 둔 것 끄기
CLEAR_AFTER_WRAP = 0.6
CLEAR_OTHER = 0.15
ACK = 0.05  # 질문 없이도 짧게 답하는 확률
SHORT_AFTER_ASK = 0.7
# assistant 가 확인 질문을 할 확률. 커밋/푸시가 가장 잦음 (푸시, 배포 승인). 나머지 0.12
ASK = {"commit": 0.4, "ui": 0.3, "report": 0.25, "wrapup": 0.25, "career": 0.2, "build": 0.2}


def read_phrases(path: Path) -> dict[str, list[str]]:
    out, key = {}, None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        if re.fullmatch(r"\[(human|ask) \w+\]|\[short\]", line):  # [Image #1] 은 문장
            key = line.strip("[]")
            out[key] = []
        else:
            out[key].append(line)
    return out


TABLE = read_phrases(Path(__file__).with_name("phrases.txt"))
PHRASES = {k.split()[1]: v for k, v in TABLE.items() if k.startswith("human ")}
QUESTIONS = {k.split()[1]: v for k, v in TABLE.items() if k.startswith("ask ")}
SHORT = TABLE["short"]
PREFIX = ["", "", "", "", "이거 ", "그럼 ", "아 그리고 ", "음 "]
DONE = ["고쳤어요.", "끝났어요.", "반영했어요.", "확인했어요. 문제없어요.", "바꿔 뒀어요."]
TOOLS = ["Bash", "Read", "Edit", "Grep", "Write", "Agent"]
TOOL_P = [0.35, 0.25, 0.2, 0.1, 0.05, 0.05]
SKILL_P = {
    "report": ("artifact-design", 0.10),
    "ui": ("frontend-design", 0.08),
    "wording": ("term-check", 0.15),
    "notebook": ("notebook-check", 0.04),
    "sql": ("sql-check", 0.01),
}
BG_TASK = (
    "다음 작업을 맡는다.\n1. {p} 저장소에서 보고서 그림을 다시 만들고 숫자를 결과 파일과 대조한다\n"
    "2. 끝나면 저장소 규칙대로 커밋하고, 푸시가 확인 대상이면 멈추고 보고한다\n3. 남은 작업과 다음 할 일을 정리한다"
)
REMINDER = "<system-reminder>\nThe task tools haven't been used recently.\n</system-reminder>"
IDE_TEXT = (
    "<ide_opened_file>The user opened the file {f} in the IDE. "
    "This may or may not be related to the current task.</ide_opened_file>"
)
IDE_FILES = ["src/app.ts", "scripts/02_build.py", "README.md", "notebooks/eda.ipynb", "docs/PROGRESS.md"]

lines: list[dict] = []
truth: list[dict] = []


def pick(xs):
    return xs[int(rng.integers(len(xs)))]


def new_id() -> str:
    return rng.bytes(8).hex()


class Session:
    def __init__(self, project: str, sid: str, t0: pd.Timestamp, entry: str, bg: bool):
        self.project, self.sid, self.t, self.entry, self.bg = project, sid, t0, entry, bg
        self.file = f"{project}/{sid}.jsonl"
        self.n_agents = 0

    def tick(self, lo: float, hi: float):
        self.t += pd.Timedelta(seconds=float(rng.uniform(lo, hi)))

    def emit(self, typ: str, content, file: str | None = None, side: bool = False, meta: bool = False) -> str:
        uid = new_id()
        rec = {
            "_file": file or self.file,
            "type": typ,
            "uuid": uid,
            "sessionId": self.sid,
            "timestamp": self.t.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "isSidechain": side,
            "entrypoint": self.entry,
        }
        if meta:
            rec["isMeta"] = True
        if self.bg:
            rec["sessionKind"] = "bg"
        rec["message"] = {"role": "user" if typ == "user" else "assistant", "content": content}
        lines.append(rec)
        return uid

    def user_text(self, text: str, source: str, intent: str = "", form: str = "", **kw):
        uid = self.emit("user", text, **kw)
        truth.append({"uuid": uid, "part": 0, "truth_source": source, "truth_intent": intent, "truth_form": form})

    def human(self, text: str, intent: str, form: str = "normal"):
        self.tick(20, 240)
        parts, src = [], []
        if "[Image #" in text:
            parts.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": ""}})
            src.append(None)
        elif self.entry == "claude-vscode" and rng.random() < 0.12:
            parts.append({"type": "text", "text": IDE_TEXT.format(f=pick(IDE_FILES))})
            src.append("ide")
        if rng.random() < 0.05:
            text = text + "\n" + REMINDER
        parts.append({"type": "text", "text": text})
        src.append("human")
        uid = self.emit("user", parts if len(parts) > 1 else text)
        for i, s in enumerate(src):
            if s:
                truth.append(
                    {
                        "uuid": uid,
                        "part": i,
                        "truth_source": s,
                        "truth_intent": intent if s == "human" else "",
                        "truth_form": form if s == "human" else "",
                    }
                )
        self.noise()

    def noise(self):
        r = rng.random()
        if r < 0.12:
            self.user_text(REMINDER, "auto")
        elif r < 0.15:
            self.user_text("<task-notification>\n<status>completed</status>\n</task-notification>", "auto")
        elif r < 0.19:
            self.user_text("[Request interrupted by user]", "auto")
        elif r < 0.21:
            self.user_text("<bash-input>git status</bash-input>", "auto")
            self.user_text("<bash-stdout>nothing to commit</bash-stdout>", "auto")
        elif r < 0.215:
            self.user_text('<teammate-message from="peer">done</teammate-message>', "auto")

    def work(self, intent: str, ask: bool):
        for _ in range(1 + int(rng.poisson(3.5))):
            self.tick(3, 40)
            tool = TOOLS[int(rng.choice(len(TOOLS), p=TOOL_P))]
            tid = "toolu_" + new_id()
            self.emit("assistant", [{"type": "tool_use", "id": tid, "name": tool, "input": {}}])
            self.tick(1, 20)
            self.emit("user", [{"type": "tool_result", "tool_use_id": tid, "content": "ok"}])
            if tool == "Agent":
                self.subagent(intent)
        if intent in SKILL_P and rng.random() < SKILL_P[intent][1]:
            self.skill(SKILL_P[intent][0])
        self.tick(3, 30)
        if ask:
            q = pick(QUESTIONS[intent]).format(n=int(rng.integers(2, 6)))
            self.emit("assistant", [{"type": "text", "text": pick(DONE) + " " + q}])
        else:
            self.emit("assistant", [{"type": "text", "text": pick(DONE)}])

    def skill(self, name: str, file: str | None = None, side: bool = False):
        tid = "toolu_" + new_id()
        self.emit("assistant", [{"type": "tool_use", "id": tid, "name": "Skill", "input": {"skill": name}}], file, side)
        self.emit(
            "user", [{"type": "tool_result", "tool_use_id": tid, "content": f"Launching skill: {name}"}], file, side
        )
        uid = self.emit("user", f"Base directory for this skill: ~/.claude/skills/{name}\n\n# {name}", file, side, True)
        truth.append(
            {
                "uuid": uid,
                "part": 0,
                "truth_source": "subagent" if side else "meta",
                "truth_intent": "",
                "truth_form": "",
            }
        )

    def subagent(self, intent: str):
        self.n_agents += 1
        file = f"{self.project}/{self.sid}/subagents/agent-{self.n_agents}.jsonl"
        uid = self.emit("user", f"조사해서 결론만 보고: {self.project} 저장소의 관련 파일과 규칙", file, True)
        truth.append({"uuid": uid, "part": 0, "truth_source": "subagent", "truth_intent": "", "truth_form": ""})
        for _ in range(2 + int(rng.poisson(4))):
            tid = "toolu_" + new_id()
            self.emit(
                "assistant",
                [{"type": "tool_use", "id": tid, "name": pick(["Bash", "Read", "Grep"]), "input": {}}],
                file,
                True,
            )
            self.emit("user", [{"type": "tool_result", "tool_use_id": tid, "content": "ok"}], file, True)
        if intent in SKILL_P and rng.random() < SKILL_P[intent][1] / 2:
            self.skill(SKILL_P[intent][0], file, True)
        self.emit("assistant", [{"type": "text", "text": "결론: 확인한 파일 3개, 규칙 표 첨부."}], file, True)

    def slash(self, cmd: str, args: str = ""):
        self.tick(5, 60)
        self.user_text(
            "<local-command-caveat>Caveat: The messages below were generated by the user while running "
            "local commands.</local-command-caveat>",
            "auto",
            meta=True,
        )
        text = (
            f"<command-name>{cmd}</command-name>\n<command-message>{cmd[1:]}</command-message>\n"
            f"<command-args>{args}</command-args>"
        )
        if args:
            self.user_text(text, "human", "tooling", "slash")
        else:
            self.user_text(text, "slash")
        self.user_text("<local-command-stdout></local-command-stdout>", "auto")

    def turn(self, intent: str):
        """사람 메시지 하나와 그 뒤 작업. 질문을 받으면 짧게 답함"""
        text = pick(PREFIX) + pick(PHRASES[intent]).format(nums=numbers())
        self.human(text, intent)
        ask = rng.random() < ASK.get(intent, 0.12)
        self.work(intent, ask)
        if (ask and rng.random() < SHORT_AFTER_ASK) or (not ask and rng.random() < ACK):
            self.human(pick(SHORT), intent, "short")
            self.work(intent, False)


def numbers() -> str:
    n = int(rng.integers(3, 6))
    vals = [f"{rng.uniform(0, 2000):.0f}" if rng.random() < 0.5 else f"{rng.uniform(0, 1):.2f}" for _ in range(n)]
    return pick([" ", " / ", ", "]).join(vals)


def draw(mix: str) -> str:
    pairs = [x.split() for x in mix.split(",")]
    keys = [k for k, _ in pairs]
    p = np.array([float(w) for _, w in pairs])
    return keys[int(rng.choice(len(keys), p=p / p.sum()))]


def run_session(project: str, bg: bool):
    _, ide_share, has_repo, mix = PLAN[project]
    day = int(rng.integers(N_DAYS))
    hour = float(rng.choice([10, 11, 13, 14, 15, 16, 17, 20, 21, 22, 23]))
    t0 = START + pd.Timedelta(days=day, hours=hour, minutes=float(rng.uniform(0, 59)))
    entry = "claude-vscode" if rng.random() < ide_share else "cli"
    s = Session(project, new_id(), t0, entry, bg)
    if rng.random() < 0.03:
        s.user_text("This session is being continued from a previous conversation that ran out of context.", "auto")
    if bg:
        s.user_text(BG_TASK.format(p=project), "bg_instruction")
        s.work("build", False)
    n_tasks = 1 + int(rng.poisson(TASKS))
    model_at = int(rng.integers(n_tasks)) if rng.random() < 0.15 else -1
    committed = False
    for i in range(n_tasks):
        if i == model_at:
            s.slash("/model")
        if rng.random() < 0.02:
            s.slash("/rename", "새 이름")
        topic = draw(mix)
        for _ in range(1 + int(rng.poisson(MSGS_PER_TASK))):
            s.turn(topic)
        committed = has_repo and rng.random() < COMMIT_AFTER_TASK
        if committed:
            for _ in range(1 + int(rng.random() < 0.4)):
                s.turn("commit")
    if rng.random() < CLEANUP_AT_END:
        s.turn("mac")
    if has_repo and not committed and rng.random() < COMMIT_AT_END:
        for _ in range(1 + int(rng.random() < 0.4)):
            s.turn("commit")
    wrapped = rng.random() < (WRAP_AT_END if has_repo else 0.2)
    if wrapped:
        for _ in range(1 + int(rng.random() < 0.6)):
            s.turn("wrapup")
    if rng.random() < (CLEAR_AFTER_WRAP if wrapped else CLEAR_OTHER):
        s.slash("/clear")
    elif rng.random() < 0.03:
        s.slash("/exit")
    elif rng.random() < 0.015:
        s.slash("/compact")


plan = [p for p, v in PLAN.items() for _ in range(v[0])]
bg_idx = set(rng.choice(len(plan), N_BG, replace=False).tolist())
for i, p in enumerate(plan):
    run_session(p, i in bg_idx)

DATA.mkdir(parents=True, exist_ok=True)
with open(TRANSCRIPTS, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as gz:
    for rec in lines:
        gz.write((json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8"))
pd.DataFrame(truth).to_csv(DATA / "truth_parts.csv", index=False)

files = {r["_file"] for r in lines}
print(f"lines {len(lines)}, files {len(files)} (subagent {sum('/subagents/' in f for f in files)})")
t = pd.DataFrame(truth)
print(t["truth_source"].value_counts().to_string())
