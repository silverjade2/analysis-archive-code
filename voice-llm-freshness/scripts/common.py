"""경로, 사전, 판정 규칙. 02 이후 스크립트는 전부 여기 사전만 보고 판정한다."""

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RES = ROOT / "outputs" / "results"
FIG = ROOT / "outputs" / "figures"
DATA.mkdir(exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)
GZ = {"method": "gzip", "mtime": 0}  # mtime을 고정해야 두 번 실행한 파일이 byte 단위로 같다

SEED = 42
START = pd.Timestamp("2025-10-01")
END = pd.Timestamp("2026-07-31")
CUTOFF = pd.Timestamp("2025-01-01")  # 분석 대상 모델 카드의 knowledge cutoff(2025년 1월)
FOLLOW_LAUNCH = pd.Timestamp("2026-03-02")  # 따라와 기능 출시. 의도 규칙에는 없다

USER, DEVICE, CONTROL = 0, 1, 2  # schema에 있는 log_type. 3, 4는 음성 원본 참조 행
PAIR_MAX_SEC = 120
RETRY_WINDOW_SEC = 30
RETRY_SIM = 0.5

# 답의 유효 기간(시간). 생성기의 truth_stale 판정과 그림 x축에만 쓴다
TOPICS = {
    "날씨/대기질": 3,
    "시세/지표": 24,
    "스포츠": 72,
    "뉴스/시사": 336,
    "트렌드/화제": 720,
}
UNMATCHED = "미매칭"

TOPIC_KEYWORDS = {
    "날씨/대기질": ["날씨", "기온", "미세먼지", "황사", "장마", "태풍", "일기예보", "강수량"],
    "시세/지표": ["환율", "주가", "금리", "유가", "코스피", "비트코인", "시세", "증시"],
    "스포츠": ["야구", "축구", "농구", "배구", "경기", "수영", "유도", "월드컵", "올림픽", "골프 메이저", "그랜드슬램"],
    "뉴스/시사": ["뉴스", "속보", "선거", "국회", "사회", "복지", "정치"],
    "트렌드/화제": ["유행", "인기", "화제", "신곡", "트렌드"],
}

# (키워드, 같은 발화에 있으면 그 키워드 히트만 취소하는 표현)
EXCLUDE_RULES = [
    ("경기", "경기도"),
    ("경기", "경기가 안 좋"),
    ("사회", "사회복지사"),
    ("복지", "사회복지사"),
    ("수영", "수영하"),
    ("유도", "유도 보내"),
    ("인기", "인기척"),
    ("뉴스", "뉴스 틀어"),
]

FRESH_SIGNALS = {
    "결과": ["결과", "이겼", "졌어", "몇 대 몇"],
    "일정": ["일정", "언제 해", "몇 시에 해"],
    "시세": ["시세", "얼마 올랐", "떨어졌"],
    "화제": ["유행", "요즘 핫", "새로 나온", "속보"],
    "시점": ["오늘", "지금", "요즘", "최근", "어제", "내일", "이번 주", "주말"],
}

COMPLAINT = re.compile(r"옛날 정보|지난 거잖아|최신으로|최신 정보|업데이트 안")

INTENT_RULES = [
    ("기기제어", re.compile(r"틀어|꺼줘|켜줘|볼륨|청소|충전|멈춰|알람|깨워")),
    ("정보질의", re.compile(r"얼마|어때|알려|누구|몇 시|몇 도|언제|어디|이겼|결과|며칠|무슨 요일|뭐 있어|뭐야")),
]

FALLBACK_PHRASES = {
    "인식/이해 실패": [
        "죄송하지만 이해하기 어려워요. 다음에 다시 말씀해주시겠어요?",
        "잘 못 알아들었어요. 다시 한번 말씀해 주세요.",
        "무슨 말씀인지 잘 모르겠어요.",
        "다시 한번 말씀해 주시겠어요?",
        "죄송해요, 제가 이해하지 못했어요.",
        "조금 더 크게 말씀해 주세요.",
        "잘 들리지 않았어요.",
        "제가 도와드릴 수 없는 요청이에요.",
        "죄송해요, 다시 말해 주세요.",
        "음, 무슨 뜻인지 모르겠어요.",
        "천천히 다시 말씀해 주세요.",
        "제가 아직 배우는 중이라 잘 모르겠어요.",
        "이해하지 못했어요. 다른 방식으로 말씀해 주세요.",
        "말씀을 놓쳤어요. 한 번만 더 말해 주실래요?",
    ],
    "소음 판정": [
        "주변이 시끄러워서 잘 못 들었어요.",
        "소리가 섞여서 잘 안 들렸어요.",
        "주변 소음 때문에 알아듣기 어려워요.",
    ],
    "오호출 판정": ["저를 부르신 게 아닌 것 같아요.", "혹시 저를 부르셨나요?"],
    "정보 부재": ["요청하신 지역을 찾을 수 없어요."],
}
FALLBACK_TAIL_N = 100  # 인식 실패 계열의 긴 꼬리 변형 수. 상위 15개가 전체를 다 덮지 않게


def fallback_tail(i: int) -> str:
    return f"죄송해요, 그 부분은 잘 모르겠어요 ({i:02d})"


def fallback_lookup() -> dict:
    lut = {p: cat for cat, ps in FALLBACK_PHRASES.items() for p in ps}
    lut.update({fallback_tail(i): "인식/이해 실패" for i in range(FALLBACK_TAIL_N)})
    return lut


def bigram_sim(a: str, b: str) -> float:
    ga = {a[i : i + 2] for i in range(len(a) - 1)} or {a}
    gb = {b[i : i + 2] for i in range(len(b) - 1)} or {b}
    return len(ga & gb) / len(ga | gb)


def pick_font():
    from matplotlib import font_manager

    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ["AppleGothic", "Pretendard", "NanumGothic"]:
        if name in have:
            return name
    return "DejaVu Sans"
