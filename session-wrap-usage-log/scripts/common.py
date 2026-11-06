"""경로, 프로젝트표, 의도 규칙, 메시지 거르기 기준. 스크립트 위치 기준이라 어디서 실행해도 됨"""

import gzip
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RES = ROOT / "outputs" / "results"
TRANSCRIPTS = DATA / "transcripts.jsonl.gz"

SEED = 42
START = pd.Timestamp("2025-09-15")
N_DAYS = 30

# key, 이름
PROJECTS = [
    ("analysis-project", "분석 프로젝트"),
    ("portfolio-site", "포트폴리오 사이트"),
    ("web-service", "웹 서비스"),
    ("job-scraper", "채용 공고 수집"),
    ("family-app", "가족 앱"),
    ("booking-scraper", "예약 스크래퍼"),
    ("writing-tool", "작문 도구"),
    ("calendar", "캘린더"),
    ("photo-select", "사진 셀렉"),
    ("budget", "개인 앱"),
]
PROJECT_NAME = dict(PROJECTS)

# 첫 일치는 이 순서. 세션 관리 규칙이 커밋보다 앞에 있는 순서를 그대로 둠
RULES = [
    (
        "mac",
        "Mac/시스템 관리",
        r"caffeinate|잠자기|화면 꺼|배터리|발열|dev 서버|개발 서버|서버 꺼|메모리|CPU|단축키|모니터",
    ),
    (
        "tooling",
        "Claude Code 사용법/설정",
        r"세션|에이전트|워크트리|worktree|플러그인|plugin|컨텍스트|권한|hook|훅|스킬|IDE|/model|statusline",
    ),
    ("wrapup", "세션 마감/인계", r"클리어|/clear|이어서|이어하기|인계|다음 세션"),
    ("commit", "커밋/푸시/배포", r"커밋|푸시|푸쉬|push|commit|배포|deploy|잔여|남은 작업|남은 일"),
    ("relay", "결과 캡처/숫자 전달", r"\[Image #|캡처|캡쳐|결과값|측정값|값 줄게|^[\d\s,./:-]{6,}$"),
    ("sql", "SQL 쿼리", r"쿼리|SQL|sql|테이블|컬럼|조인"),
    ("notebook", "노트북 실행 오류", r"노트북|ipynb|셀|커널|에러가|오류|codec|NameError"),
    ("report", "보고서/아티팩트/덱", r"보고|아티팩트|artifact|html|HTML|pdf|PDF|덱|슬라이드|장표|페이지|리포트|아티클"),
    ("summary", "회의/메일/한 줄 요약", r"회의|미팅|안건|메일|한 줄|한줄|브리핑|주간"),
    ("wording", "용어/문체 다듬기", r"용어|조어|음차|표현|문구|말투|어색|문체"),
    ("ui", "UI 시안/미세 조정", r"시안|UI|폰트|색|컬러|정렬|너비|간격|버튼|모바일|레이아웃"),
    ("career", "채용/면접/CV", r"공고|면접|이력서|CV|포트폴리오|지원서|채용|자소서"),
    ("scrape", "데이터 수집/스크래핑", r"수집|크롤|스크래|새로고침"),
    ("files", "파일/폴더 정리", r"폴더|파일|다운로드|삭제|지워|휴지통|중복"),
    ("question", "설명/개념 질문", r"뭐야|뭐지|무슨|의미|왜|어떻게|설명|이해"),
]
CLUSTERS = [r[0] for r in RULES]
LABEL = {r[0]: r[1] for r in RULES} | {"other": "기타/짧은 응답"}
PATTERN = {r[0]: re.compile(r[2], re.M) for r in RULES}

# 후보 도구와 덮는 군집
CANDIDATES = [
    ("A", "세션 마감 도구", ["commit", "wrapup"]),
    ("B", "보고 페이지 생성기", ["report"]),
    ("C", "결과 회수 폼", ["relay"]),
    ("D", "상태 메뉴바", ["mac"]),
]

# 사람이 쓴 것이 아닌 user 텍스트. 1차 거르기
AUTO = (
    "<system-reminder",
    "<local-command",
    "<command-",
    "Caveat:",
    "[Request interrupted",
    "<task-notification",
    "<bash-",
    "<teammate-message",
    "This session is being continued",
)
IDE = ("<ide_opened_file", "<ide_selection")  # 1차에서 빠지지 않아 2차에서 거름
SHORT_LEN = 10  # 공백 뺀 글자 수가 이 이하면 짧은 응답
MIN_PROJECT_MSGS = 3  # 후보가 그 프로젝트에 퍼져 있다고 볼 메시지 수


def read_transcripts():
    """(파일 경로, 레코드) 를 기록 순서대로"""
    with gzip.open(TRANSCRIPTS, "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            yield rec.pop("_file"), rec


def matches(text: str) -> list[str]:
    return [k for k in CLUSTERS if PATTERN[k].search(text)]


def first_match(text: str, order: list[str] = CLUSTERS) -> str:
    for k in order:
        if PATTERN[k].search(text):
            return k
    return "other"


def is_short(text: str) -> bool:
    return len(re.sub(r"\s", "", text)) <= SHORT_LEN
