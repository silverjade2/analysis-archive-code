"""가짜 소스 세 개: 스코어 시트(탭 4개), 캘린더 서버(캘린더 6개의 iCalendar), 설문 DB(식별 열이 있는 응답 원본)

truth_*.csv 는 04 만 읽음
"""

import json
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd
from common import DATA, GOLF_TABS, HOME_HDCP, HOME_PARS, LOOKAHEAD_DAYS, LOOKBACK_DAYS, SEED, SOURCE, TODAY

rng = np.random.default_rng(SEED)

# ---------- 스코어 시트 ----------
# 값은 전부 문자열이고 Sheets API 는 행 끝의 빈 칸을 잘라서 돌려줌
PLAYERS = [
    ["p1", "하늘", "2026-09-21T21:04:11+09:00"],
    ["p2", "  바다 ", "2026-09-21T21:05:40+09:00"],  # 앞뒤 공백은 stg 에서 정리
    ["p3", "선수3", ""],  # updatedAt 이 비면 아직 안 바꾼 기본 이름
    ["p4", "선수4", ""],
]
SKILL = {"p1": 0.9, "p2": 1.3, "p3": 1.7, "p4": 1.1}  # 홀당 평균 par 대비 타수
BLANK_P = 0.015  # 입력했다가 지운 칸


def played_order(first_hole: int) -> list[int]:
    """친 순서 1~18 -> 실제 홀 번호. IN 출발이면 10번 홀이 첫 칸"""
    return [(first_hole - 1 + i) % 18 + 1 for i in range(18)]


def pars_text(first_hole: int) -> str:
    return ",".join(str(HOME_PARS[h - 1]) for h in played_order(first_hole))


COURSES = [
    ["c_home_out", "Maple Ridge CC OUT", pars_text(1), "2026-09-01T10:00:00+09:00", "", ""],
    ["c_home_in", "Maple Ridge CC IN", pars_text(10), "2026-09-01T10:01:00+09:00", "", ""],
    ["c_home_out9", "Maple Ridge CC OUT 9H", pars_text(1), "2026-09-01T10:02:00+09:00", "", ""],
    ["c_home_in9", "Maple Ridge CC IN 9H", pars_text(10), "2026-09-01T10:03:00+09:00", "", ""],
    ["c_test", "TEST", ",".join(["4"] * 18), "2026-09-01T10:04:00+09:00", "", "2026-09-21T20:00:00+09:00"],
    ["c_range", "연습 코스", "3,3,3", "2026-09-19T18:00:00+09:00", "", ""],  # pars 형식이 어긋남 -> stg 에서 전부 4
]
FIRST_HOLE = {"c_home_out": 1, "c_home_in": 10, "c_home_out9": 1, "c_home_in9": 10}
HOLES = {"c_home_out9": 9, "c_home_in9": 9}

# id, 날짜, 코스, 메모, 시작 시각, 삭제, 입력한 칸 수(친 순서 1부터)
ROUNDS = [
    ("r01", "2026-09-20", "c_range", "연습", "07:10", True, 18),
    ("r02", "2026-09-21", "c_test", "앱 테스트", "20:30", True, 3),
    ("r03", "2026-09-22", "c_home_out", "", "08:40", False, 18),
    ("r04", "2026-09-23", "c_home_in", "", "07:50", False, 18),
    ("r05", "2026-09-23", "c_home_in", "중복 생성", "07:51", True, 0),
    ("r06", "2026-09-23", "c_home_out9", "", "13:20", False, 9),
    ("r07", "2026-09-24", "c_home_in", "코스 잘못 고름", "08:00", True, 2),
    ("r08", "2026-09-24", "c_home_out", "", "08:05", False, 18),
    ("r09", "2026-09-25", "c_home_in", "", "08:10", False, 18),
    ("r10", "2026-09-25", "c_home_in9", "", "13:40", False, 9),
    ("r11", "2026-09-26", "c_home_out", "", "08:30", False, 18),
    ("r12", "2026-09-26", "c_home_out", "두 번 입력", "08:31", True, 18),
    ("r13", "2026-09-27", "c_home_out", "비로 중단", "07:40", True, 7),
    ("r14", "2026-09-27", "c_home_out9", "", "11:00", False, 9),
]


def stroke(player: str, hole_no: int, par: int) -> int:
    hard = 0.3 if HOME_HDCP[hole_no - 1] <= 6 else -0.2 if HOME_HDCP[hole_no - 1] >= 13 else 0.0
    return par + int(np.clip(round(rng.normal(SKILL[player] + hard, 0.85)), -1, 4))


round_rows, score_rows, truth_scores = [], [], []
copied = {}
for rid, day, course, memo, start, deleted, n_cells in ROUNDS:
    t0 = datetime.fromisoformat(f"{day}T{start}:00+09:00")
    holes = HOLES.get(course, 18)
    first = FIRST_HOLE.get(course, 1)
    pars = [int(x) for x in COURSES[[c[0] for c in COURSES].index(course)][2].split(",")]
    pars = pars if len(pars) == 18 else [4] * 18
    end_at = t0 + timedelta(hours=4.5 if holes == 18 else 2.2)
    round_rows.append(
        [
            rid,
            day,
            course,
            memo,
            t0.isoformat(),
            end_at.isoformat() if n_cells else "",
            (end_at + timedelta(minutes=20)).isoformat() if deleted else "",
        ]
    )
    for order in range(1, n_cells + 1):
        hole_no = played_order(first)[order - 1]
        for pid, *_ in PLAYERS:
            if rid == "r12":  # 같은 라운드를 다른 기기에서 한 번 더 입력
                s = copied[("r11", pid, order)]
            else:
                s = stroke(pid, hole_no, pars[order - 1])
                copied[(rid, pid, order)] = s
            blank = rng.random() < BLANK_P
            at = t0 + timedelta(minutes=14 * order)
            score_rows.append(
                [f"s_{rid}_{pid}_{order:02d}", rid, pid, str(order), "" if blank else str(s), at.isoformat()]
            )
            reason = "deleted_round" if deleted else "blank" if blank else "beyond_holes" if order > holes else "kept"
            truth_scores.append(
                {
                    "id": f"s_{rid}_{pid}_{order:02d}",
                    "round_id": rid,
                    "player_id": pid,
                    "play_order": order,
                    "truth_strokes": None if blank else s,
                    "truth_reason": reason,
                }
            )

# Sheets 에 쌓이는 순서: 입력한 시각순
score_rows.sort(key=lambda r: (r[5], r[0]))


def trim(row: list[str]) -> list[str]:
    """Sheets API values 응답처럼 행 끝 빈 칸을 자름"""
    while row and row[-1] == "":
        row = row[:-1]
    return row


sheet = {
    t: [trim(list(r)) for r in rows]
    for t, rows in {"Player": PLAYERS, "Course": COURSES, "Round": round_rows, "Score": score_rows}.items()
}
(SOURCE / "golf").mkdir(parents=True, exist_ok=True)
(SOURCE / "golf" / "spreadsheet.json").write_text(
    json.dumps({"tabs": list(GOLF_TABS), "values": sheet}, ensure_ascii=False, indent=1) + "\n",
    encoding="utf-8",
)
pd.DataFrame(truth_scores).to_csv(DATA / "truth_golf_scores.csv", index=False)
pd.DataFrame([{"round_id": r[0], "truth_deleted": r[5], "truth_holes": HOLES.get(r[2], 18)} for r in ROUNDS]).to_csv(
    DATA / "truth_golf_rounds.csv", index=False
)

# ---------- 캘린더 서버 ----------
# 서버는 일정 1개 = VCALENDAR 1개로 돌려줌
# 반복 일정의 회차 수정(RECURRENCE-ID)은 같은 VCALENDAR 안 두 번째 VEVENT
WIN_START = TODAY - timedelta(days=LOOKBACK_DAYS)
WIN_END = TODAY + timedelta(days=LOOKAHEAD_DAYS + 1)  # 끝은 포함하지 않음
WIN_DAYS = (WIN_END - WIN_START).days
CALENDARS = [
    "개인",
    "업무",
    "운동",
    "가족",
    "스터디",
    "미리 알림",
]  # 미리 알림은 할 일만 있어 일정 조회 결과가 비어 있음
TITLES = {
    "개인": ["병원", "은행", "친구 약속", "미용실", "저녁 약속", "택배 수령"],
    "업무": ["주간 회의", "1:1", "코드 리뷰", "외부 미팅", "발표 준비", "면담"],
    "운동": ["러닝", "등산", "수영"],
    "가족": ["가족 식사", "부모님 댁", "장보기"],
    "스터디": ["강의", "독서 모임", "발표 연습"],
}
N_TIMED = {"개인": 16, "업무": 20, "운동": 4, "가족": 6, "스터디": 4}
N_ALLDAY = {"개인": 2, "업무": 0, "운동": 0, "가족": 2, "스터디": 2}


def fmt_local(d: datetime) -> str:
    return d.strftime("%Y%m%dT%H%M%S")


def vevent(uid: str, title: str, start, end, *, kind: str = "local", extra: list[str] | None = None) -> list[str]:
    if kind == "allday":
        ds, de = f"DTSTART;VALUE=DATE:{start:%Y%m%d}", f"DTEND;VALUE=DATE:{end:%Y%m%d}"
    elif kind == "utc":
        ds, de = f"DTSTART:{fmt_local(start)}Z", f"DTEND:{fmt_local(end)}Z"
    elif kind == "floating":  # 시간대 없는 값
        ds, de = f"DTSTART:{fmt_local(start)}", f"DTEND:{fmt_local(end)}"
    else:
        ds, de = f"DTSTART;TZID=Asia/Seoul:{fmt_local(start)}", f"DTEND;TZID=Asia/Seoul:{fmt_local(end)}"
    return [
        "BEGIN:VEVENT",
        f"UID:{uid}",
        "DTSTAMP:20261020T000000Z",
        ds,
        de,
        f"SUMMARY:{title}",
        *(extra or []),
        "END:VEVENT",
    ]


def vcalendar(*events: list[str]) -> str:
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//repro//calendar//KO"]
    for e in events:
        lines += e
    return "\r\n".join(lines + ["END:VCALENDAR"]) + "\r\n"


server = {c: [] for c in CALENDARS}
truth_occ = []


def add(cal: str, uid: str, title: str, start, end, all_day: bool, recurring: bool, status: str = "CONFIRMED"):
    truth_occ.append(
        {
            "calendar": cal,
            "uid": uid,
            "title": title,
            "truth_all_day": all_day,
            "truth_recurring": recurring,
            "truth_status": status,
            "truth_start_local": start.isoformat(),
            "truth_end_local": end.isoformat(),
        }
    )


def random_slot() -> datetime:
    day = WIN_START + timedelta(days=int(rng.integers(0, WIN_DAYS)))
    return datetime.combine(day, time(int(rng.integers(8, 21)), int(rng.choice([0, 30]))))


n = 0
for cal in CALENDARS[:-1]:
    for _ in range(N_TIMED[cal]):
        n += 1
        uid, title = f"ev{n:03d}", str(rng.choice(TITLES[cal]))
        s = random_slot()
        e = s + timedelta(minutes=int(rng.choice([30, 60, 60, 90, 120, 180])))
        server[cal].append(vcalendar(vevent(uid, title, s, e)))
        add(cal, uid, title, s, e, False, False)
    for _ in range(N_ALLDAY[cal]):
        n += 1
        uid, title = f"ev{n:03d}", {"개인": "휴가", "가족": "가족 행사", "스터디": "과제 마감"}[cal]
        s = WIN_START + timedelta(days=int(rng.integers(0, WIN_DAYS)))
        e = s + timedelta(days=1)
        server[cal].append(vcalendar(vevent(uid, title, s, e, kind="allday")))
        add(cal, uid, title, s, e, True, False)

# 시간대가 다르게 적힌 일정, 취소된 일정, 여러 날 종일 일정
s = datetime(2026, 10, 12, 1, 0)
server["업무"].append(vcalendar(vevent("ev-utc", "해외 미팅", s, s + timedelta(hours=1), kind="utc")))
add("업무", "ev-utc", "해외 미팅", s + timedelta(hours=9), s + timedelta(hours=10), False, False)
s = datetime(2026, 10, 14, 9, 0)
server["개인"].append(vcalendar(vevent("ev-float", "건강검진", s, s + timedelta(hours=2), kind="floating")))
add("개인", "ev-float", "건강검진", s, s + timedelta(hours=2), False, False)
s = datetime(2026, 10, 16, 15, 0)
server["업무"].append(
    vcalendar(vevent("ev-cancel", "고객사 방문", s, s + timedelta(hours=2), extra=["STATUS:CANCELLED"]))
)
add("업무", "ev-cancel", "고객사 방문", s, s + timedelta(hours=2), False, False, "CANCELLED")
for uid, cal, title, s, e in [
    ("ev-trip", "가족", "골프 여행", date(2026, 9, 22), date(2026, 9, 28)),
    ("ev-biz", "업무", "출장", date(2026, 11, 4), date(2026, 11, 7)),
]:
    server[cal].append(vcalendar(vevent(uid, title, s, e, kind="allday")))
    add(cal, uid, title, s, e, True, False)
# 조회 기간 밖 일정 (서버는 돌려주지만 전개 단계에서 빠짐)
s = datetime(2026, 8, 10, 19, 0)
server["개인"].append(vcalendar(vevent("ev-old", "저녁 약속", s, s + timedelta(hours=2))))
s = datetime(2027, 2, 3, 10, 0)
server["업무"].append(vcalendar(vevent("ev-far", "주간 회의", s, s + timedelta(hours=1))))

# 반복 일정 1: 매주 화요일 8회, 여행 주는 EXDATE, 10-06 회차는 다음 날로 옮김(RECURRENCE-ID)
s0 = datetime(2026, 9, 1, 19, 30)
master = vevent(
    "rec-pilates",
    "필라테스",
    s0,
    s0 + timedelta(minutes=50),
    extra=["RRULE:FREQ=WEEKLY;BYDAY=TU;COUNT=8", "EXDATE;TZID=Asia/Seoul:20260922T193000"],
)
moved = vevent(
    "rec-pilates",
    "필라테스",
    datetime(2026, 10, 7, 20, 0),
    datetime(2026, 10, 7, 20, 50),
    extra=["RECURRENCE-ID;TZID=Asia/Seoul:20261006T193000"],
)
server["운동"].append(vcalendar(master, moved))
for k in range(8):
    s = s0 + timedelta(days=7 * k)
    if s.date() == date(2026, 9, 22):
        continue
    if s.date() == date(2026, 10, 6):
        s = datetime(2026, 10, 7, 20, 0)
    add("운동", "rec-pilates", "필라테스", s, s + timedelta(minutes=50), False, True)


def second_saturday(y: int, m: int) -> date:
    d = date(y, m, 1)
    return d + timedelta(days=(5 - d.weekday()) % 7 + 7)


# 반복 일정 2: 매월 둘째 토요일 4회
s0 = datetime(2026, 9, 12, 14, 0)
server["스터디"].append(
    vcalendar(
        vevent("rec-study", "스터디", s0, s0 + timedelta(hours=2), extra=["RRULE:FREQ=MONTHLY;BYDAY=2SA;COUNT=4"])
    )
)
for m in (9, 10, 11, 12):
    s = datetime.combine(second_saturday(2026, m), time(14, 0))
    add("스터디", "rec-study", "스터디", s, s + timedelta(hours=2), False, True)

# 반복 일정 3: 매년 생일 (종일)
server["가족"].append(
    vcalendar(
        vevent("rec-bday", "생일", date(2020, 11, 14), date(2020, 11, 15), kind="allday", extra=["RRULE:FREQ=YEARLY"])
    )
)
add("가족", "rec-bday", "생일", date(2026, 11, 14), date(2026, 11, 15), True, True)

(SOURCE / "calendar").mkdir(parents=True, exist_ok=True)
(SOURCE / "calendar" / "server.json").write_text(
    json.dumps({"calendars": [{"name": c, "objects": server[c]} for c in CALENDARS]}, ensure_ascii=False, indent=1)
    + "\n",
    encoding="utf-8",
)
pd.DataFrame(truth_occ).to_csv(DATA / "truth_calendar.csv", index=False)

# ---------- 설문 DB ----------
# 원본 테이블 열 구성 그대로 (사주 풀이 jsonb 열 몇 개는 뺌). 닉네임, 생년월일시, 4주, 보정 시각 같은 식별 열이 있고
# 수집은 DB 쪽 view(식별 열 없는 7개 열)만 읽음. mbti, saju_mbti 는 NOT NULL
MBTI_W = {
    "ISFJ": 9,
    "INFP": 9,
    "ENFP": 8,
    "ESFJ": 7,
    "ISTJ": 7,
    "ISFP": 7,
    "INFJ": 6,
    "ESTJ": 6,
    "INTJ": 5,
    "INTP": 5,
    "ENTP": 5,
    "ENFJ": 5,
    "ESFP": 5,
    "ISTP": 5,
    "ENTJ": 4,
    "ESTP": 4,
}
STEMS = list("갑을병정무기경신임계")
BRANCHES = list("자축인묘진사오미신유술해")
N_RESPONSES = 340
N_TEST = 12
LAUNCH = date(2026, 8, 3)
SYL = list("하나별빛구름바람달솔봄가을물결은새")

types = list(MBTI_W)
weights = np.array(list(MBTI_W.values()), dtype=float)
days = (TODAY - LAUNCH).days
day_w = np.exp(-np.arange(days) / 25)  # 공개 직후에 몰리고 점점 줄어듦
day_w[45:48] += 1.2  # 중간에 한 번 공유되어 튐


def flip(letter: str) -> str:
    return {"E": "I", "I": "E", "S": "N", "N": "S", "T": "F", "F": "T", "J": "P", "P": "J"}[letter]


def uuid() -> str:
    h = "".join(rng.choice(list("0123456789abcdef"), 32))
    return f"{h[:8]}-{h[8:12]}-4{h[13:16]}-a{h[17:20]}-{h[20:]}"


subs = []
for i in range(N_RESPONSES + N_TEST):
    day = LAUNCH + timedelta(days=int(rng.choice(days, p=day_w / day_w.sum())))
    created = datetime.combine(day, time(int(rng.integers(0, 24)), int(rng.integers(0, 60)), int(rng.integers(0, 60))))
    mbti = str(rng.choice(types, p=weights / weights.sum()))
    saju = "".join(c if rng.random() < 0.55 else flip(c) for c in mbti)
    pillars = [str(rng.choice(STEMS)) + str(rng.choice(BRANCHES)) for _ in range(4)]
    hour_known = bool(rng.random() < 0.72)
    birth = datetime.combine(
        date(1970, 1, 1) + timedelta(days=int(rng.integers(0, 13000))),
        time(int(rng.integers(0, 24)), int(rng.integers(0, 60))),
    )
    is_test = i >= N_RESPONSES
    subs.append(
        {
            "id": uuid(),
            "slug": "".join(rng.choice(list("0123456789abcdefghjkmnpqrstuvwxyz"), 10)),
            "created_at": created.isoformat() + "+09:00",
            "nickname": "test" if is_test else "".join(rng.choice(SYL, 2)) + str(int(rng.integers(1, 99))),
            "birth_date": birth.date().isoformat(),
            "birth_time": f"{birth:%H:%M}" if hour_known else "",
            "hour_known": hour_known,
            "calendar_type": "lunar" if rng.random() < 0.15 else "solar",
            "is_leap_month": False,
            "corrected_datetime": f"{birth - timedelta(minutes=32):%Y-%m-%dT%H:%M}" if hour_known else "",
            "mbti": mbti,
            "pillars": json.dumps(dict(zip(["year", "month", "day", "hour"], pillars)), ensure_ascii=False),
            "day_master": pillars[2][0],
            "saju_mbti": saju,
            "overall_score": int(np.clip(round(rng.normal(74, 9)), 41, 99)),
            "mapping_version": "2026.08" if day < date(2026, 9, 10) else "2026.09",
            "referrer_id": "",
            "source": str(rng.choice(["", "instagram", "kakao", "blog"])),
            "user_agent": str(
                rng.choice(["Mozilla/5.0 (iPhone)", "Mozilla/5.0 (Linux; Android)", "Mozilla/5.0 (Macintosh)"])
            ),
            "is_test": is_test,
        }
    )
subs = pd.DataFrame(subs).sort_values(["created_at", "slug"]).reset_index(drop=True)
shared = rng.random(len(subs)) < 0.2  # 친구가 보낸 링크로 들어온 응답
subs.loc[shared, "referrer_id"] = [
    subs["id"].iloc[int(rng.integers(0, k))] if k else "" for k in np.flatnonzero(shared)
]
(SOURCE / "saju").mkdir(parents=True, exist_ok=True)
subs.to_csv(SOURCE / "saju" / "submissions.csv", index=False)

print(f"스코어 시트: 플레이어 {len(PLAYERS)}, 코스 {len(COURSES)}, 라운드 {len(ROUNDS)}, 점수 {len(score_rows)}행")
print(
    f"캘린더: {len(CALENDARS)}개, 일정 원본 {sum(len(v) for v in server.values())}개, 조회 기간 회차 {len(truth_occ)}"
)
print(f"설문: 응답 {N_RESPONSES} + 테스트 {N_TEST}")
