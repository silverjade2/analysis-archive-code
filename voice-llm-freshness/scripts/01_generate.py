"""가상 음성 대화 로그

빈 발화는 두 곳에서 나온다. 호출어 오인식으로 켜졌는데 앞에 사람이 없는 경우와, 사람이 말했지만 기기가 멀리
있거나 움직이는 중이라 STT가 아무것도 받지 못한 경우다. 오호출은 유저 발화의 40% 안팎으로 두고, 운영 로그의
빈 발화 57%를 채우는 나머지는 사람 쪽 STT 누락으로 넣었다. 두 비율 모두 생성기의 가정이다.

사람도 파편을 말한다. 답을 끊는 "아니", "이거 말고"는 기기 명령이 아니지만 사람이 기기에 한 말이다.
미매칭 n-gram 상위의 파편이 전부 오호출은 아니다.

세션은 기기가 끊는다. fallback 뒤에는 대부분 세션을 닫고, 무응답이면 계속 듣는다. 재시도가 다음 세션으로
넘어가는 구조는 여기서 생긴다.

truth_misfire, truth_stale, truth_topic, truth_fresh는 평가용이다. 02~06, 09는 이 컬럼을 읽지 않는다.
"""

import numpy as np
import pandas as pd
from common import (
    CONTROL,
    CUTOFF,
    DATA,
    DEVICE,
    END,
    FALLBACK_PHRASES,
    FALLBACK_TAIL_N,
    FOLLOW_LAUNCH,
    GZ,
    SEED,
    START,
    TOPICS,
    USER,
    fallback_tail,
)

rng = np.random.default_rng(SEED)

N_EPISODES = 64000
P_MISFIRE_EPISODE = 0.35
P_EMPTY_MISFIRE = 0.78
P_EMPTY_PERSON = 0.48  # 이동형 기기라 거리와 주행 소음으로 STT가 비는 사람 발화
P_REF_ROW = 0.28  # 유저 발화에 음성 원본 참조 행이 붙는 비율

# (문장, 실제 주제, 최신 정보 필요, 시계로 답하는 질문, 가중치)
PERSON = {
    "제어": [
        ("노래 틀어줘", None, 0, 0, 18),
        ("음악 틀어줘", None, 0, 0, 12),
        ("음악 꺼줘", None, 0, 0, 14),
        ("볼륨 올려줘", None, 0, 0, 10),
        ("볼륨 낮춰줘", None, 0, 0, 8),
        ("청소 시작해", None, 0, 0, 12),
        ("충전하러 가", None, 0, 0, 8),
        ("멈춰", None, 0, 0, 10),
        ("라디오 틀어줘", None, 0, 0, 6),
        ("뉴스 틀어줘", None, 0, 0, 5),
        ("지금 노래 틀어줘", None, 0, 0, 5),
        ("새로 나온 노래 틀어줘", None, 0, 0, 3),
        ("내일 아침 7시에 깨워줘", None, 0, 0, 5),
        ("오늘 알람 꺼줘", None, 0, 0, 2),
    ],
    "잡담": [
        ("심심해", None, 0, 0, 5),
        ("사랑해", None, 0, 0, 4),
        ("고마워", None, 0, 0, 5),
        ("잘 자", None, 0, 0, 4),
        ("안녕", None, 0, 0, 5),
        ("노래 불러줘", None, 0, 0, 3),
        ("재밌는 얘기 해줘", None, 0, 0, 3),
        ("배고파", None, 0, 0, 3),
        ("너 이름이 뭐야", None, 0, 0, 3),
        ("나 왔어", None, 0, 0, 4),
        ("졸려", None, 0, 0, 3),
        ("힘들다", None, 0, 0, 3),
        ("뭐 하고 놀까", None, 0, 0, 3),
        ("칭찬해줘", None, 0, 0, 3),
        ("웃긴 얘기 해줘", None, 0, 0, 2),
        ("잘했어", None, 0, 0, 3),
        ("미안해", None, 0, 0, 3),
        ("넌 몇 살이야", None, 0, 0, 2),
        ("춤춰 봐", None, 0, 0, 3),
        ("같이 놀자", None, 0, 0, 3),
        ("기분 좋다", None, 0, 0, 2),
        ("보고 싶었어", None, 0, 0, 2),
        ("수고했어", None, 0, 0, 3),
        ("착하다", None, 0, 0, 2),
        ("오늘 기분 안 좋아", None, 0, 0, 1),
        ("지금 뭐 해", None, 0, 0, 1),
        ("오늘 뭐 먹지", None, 0, 0, 1),
        ("날씨 좋다", None, 0, 0, 0.8),
        ("야구 보자", None, 0, 0, 0.45),
        ("축구하러 가자", None, 0, 0, 0.35),
        ("너 인기 많아", None, 0, 0, 0.2),
        ("사회 숙제 도와줘", None, 0, 0, 0.1),
    ],
    "정정": [
        ("아니", None, 0, 0, 30),
        ("아니야", None, 0, 0, 20),
        ("아니 그거 말고", None, 0, 0, 20),
        ("이거 말고", None, 0, 0, 15),
        ("아니 됐어", None, 0, 0, 15),
    ],
    "시계": [
        ("지금 몇 시야", None, 1, 1, 20),
        ("오늘 무슨 요일이야", None, 1, 1, 8),
        ("오늘 며칠이야", None, 1, 1, 8),
    ],
    "정보": [
        ("오늘 날씨 어때", "날씨·대기질", 1, 0, 0.9),
        ("내일 날씨 알려줘", "날씨·대기질", 1, 0, 0.35),
        ("지금 미세먼지 어때", "날씨·대기질", 1, 0, 0.3),
        ("오늘 비 와", "날씨·대기질", 1, 0, 8),
        ("우산 챙겨야 돼", "날씨·대기질", 1, 0, 6),
        ("밖에 추워", "날씨·대기질", 1, 0, 6),
        ("지금 환율 얼마야", "시세·지표", 1, 0, 0.4),
        ("오늘 주가 어때", "시세·지표", 1, 0, 0.3),
        ("금리 올랐어", "시세·지표", 1, 0, 0.2),
        ("비트코인 시세 알려줘", "시세·지표", 1, 0, 0.2),
        ("달러 얼마야", "시세·지표", 1, 0, 4),
        ("요즘 금값 어때", "시세·지표", 1, 0, 3),
        ("어제 야구 누가 이겼어", "스포츠", 1, 0, 0.2),
        ("오늘 축구 몇 시에 해", "스포츠", 1, 0, 0.1),
        ("어제 한국 이겼어", "스포츠", 1, 0, 0.2),
        ("오늘 한국 몇 대 몇이야", "스포츠", 1, 0, 0.1),
        ("요즘 뉴스 뭐 있어", "뉴스·시사", 1, 0, 0.4),
        ("오늘 속보 알려줘", "뉴스·시사", 1, 0, 0.2),
        ("대통령 누구야", "뉴스·시사", 1, 0, 4),
        ("요즘 무슨 일 있어", "뉴스·시사", 1, 0, 3),
        ("요즘 유행하는 노래 뭐야", "트렌드·화제", 1, 0, 0.35),
        ("요즘 인기 드라마 뭐야", "트렌드·화제", 1, 0, 0.2),
        ("새로 나온 영화 뭐 있어", "트렌드·화제", 1, 0, 3),
        ("요즘 핫한 거 뭐야", "트렌드·화제", 1, 0, 3),
        ("요즘 볼만한 영화 추천해줘", "트렌드·화제", 1, 0, 4),
        ("오늘 운세 알려줘", None, 1, 0, 5),
        ("지금 차 막혀", None, 1, 0, 3),
        ("로또 번호 알려줘", None, 0, 0, 3),
        ("김치찌개 레시피 알려줘", None, 0, 0, 6),
        ("고양이는 몇 살까지 살아", None, 0, 0, 4),
        ("영어로 사과가 뭐야", None, 0, 0, 5),
        ("경기도 가는 길 막혀", None, 1, 0, 1),
        ("사회복지사 시험 언제야", None, 0, 0, 1),
    ],
    "따라와": [("따라와", None, 0, 0, 6), ("나 따라와", None, 0, 0, 3), ("따라와 봐", None, 0, 0, 1)],
}
KIND_P = {"제어": 0.10, "잡담": 0.60, "정정": 0.11, "시계": 0.018, "정보": 0.10, "따라와": 0.07}

COMPLAINTS = [
    "그거 옛날 정보잖아",
    "최신으로 알려줘",
    "최신 정보로 다시 알려줘",
    "그거 지난 거잖아",
    "업데이트 안 됐어",
]

FRAGMENTS = [
    ("아니", 30),
    ("엄마", 30),
    ("아니야", 20),
    ("이거", 11),
    ("아빠", 8),
    ("그래서", 7),
    ("진짜", 7),
    ("아니 근데", 6),
    ("엄마 이거 뭐야", 6),
    ("이거 봐", 5),
    ("밥 먹자", 5),
    ("어디 가", 4),
    ("빨리 와", 4),
    ("그만해", 4),
    ("잠깐만", 4),
    ("다음 소식입니다", 3),
    ("잠시 후 계속됩니다", 3),
    ("요즘 경기가 안 좋아서", 1),
    ("수영하고 올게", 0.5),
    ("유도 보내자", 0.5),
    ("지금 가", 0.3),
    ("날씨 좋다", 0.4),
    ("경기 봐", 0.3),
    ("인기척 났어", 0.5),
    ("오늘의 날씨였습니다", 0.1),
    ("경기 결과 전해드립니다", 0.2),
]

# (normal, fallback, none)
RESP_P = {
    "misfire": (0.42, 0.40, 0.18),
    "empty": (0.05, 0.60, 0.35),
    "제어": (0.95, 0.033, 0.017),
    "따라와": (0.30, 0.66, 0.04),
    "잡담": (0.76, 0.19, 0.05),
    "정정": (0.40, 0.55, 0.05),
    "시계": (0.85, 0.145, 0.005),
    "정보": (0.76, 0.235, 0.005),
    "complaint": (0.85, 0.14, 0.01),
}
P_NOTICE = {"날씨·대기질": 0.10, "시세·지표": 0.08, "스포츠": 0.12, "뉴스·시사": 0.05, "트렌드·화제": 0.02}

rec = FALLBACK_PHRASES["인식·이해 실패"]
# 운영 로그의 상위 두 문구 비중(39.8%, 17.6%)에 맞춘 가중치. 나머지는 긴 꼬리
REC_W = np.array(
    [39.8, 17.6, 1.2, 1.1, 1.0, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.52] + [0.309] * FALLBACK_TAIL_N
)
REC_ALL = rec + [fallback_tail(i) for i in range(FALLBACK_TAIL_N)]

total_sec = int((END + pd.Timedelta(days=1) - START).total_seconds())
rows = []
sid = 0


def pick(items):
    w = np.array([x[-1] for x in items], dtype=float)
    return items[rng.choice(len(items), p=w / w.sum())]


def fallback_phrase(cause):
    r = rng.random()
    if cause == "misfire" and r < 0.016:
        return str(rng.choice(FALLBACK_PHRASES["소음 판정"]))
    if cause == "misfire" and r < 0.0235:
        return str(rng.choice(FALLBACK_PHRASES["오호출 판정"]))
    if cause == "정보" and r < 0.12:
        return FALLBACK_PHRASES["정보 부재"][0]
    return REC_ALL[rng.choice(len(REC_ALL), p=REC_W / REC_W.sum())]


def emit_turn(t, text, kind, topic, fresh, clock, misfire):
    """유저 발화 1건과 기기 쪽 행을 쓰고 (응답 상태, 낡음 여부)를 돌려준다."""
    rows.append((sid, USER, t, text, int(misfire), topic or "", int(fresh), 0))
    uidx = len(rows) - 1
    if rng.random() < P_REF_ROW:
        rows.append(
            (
                sid,
                int(rng.choice([3, 4])),
                t,
                f"k={rng.integers(16**6):06x};f={rng.integers(10**7):07d}.wav",
                0,
                "",
                0,
                0,
            )
        )

    rkind = "empty" if text == "" else kind
    state = ["normal", "fallback", "none"][rng.choice(3, p=RESP_P[rkind])]
    if kind == "제어" and state == "normal":
        rows.append((sid, CONTROL, t + 1, f"CMD_{rng.integers(100, 999)}", 0, "", 0, 0))
    lat = float(np.exp(rng.normal(np.log(2.0), 0.45)))
    if rng.random() < 0.01:
        lat += float(rng.uniform(121, 400))  # 적재 지연. 02의 120초 컷에 걸린다
    if state == "normal":
        rows.append((sid, DEVICE, t + lat, "네, 알려드릴게요.", 0, "", 0, 0))
    elif state == "fallback":
        cause = "misfire" if misfire else kind
        rows.append((sid, DEVICE, t + lat, fallback_phrase(cause), 0, "", 0, 0))

    stale = 0
    if state == "normal" and fresh and not clock:
        validity_h = TOPICS.get(topic, 24) * float(np.exp(rng.normal(0, 0.5)))
        # LLM 답은 cutoff 시점에 정해진 정보. 발화 시각까지 유효 기간이 지났으면 낡은 답
        age_h = (START + pd.Timedelta(seconds=t) - CUTOFF).total_seconds() / 3600
        stale = int(age_h > validity_h)
        r = list(rows[uidx])
        r[7] = stale
        rows[uidx] = tuple(r)
    return state, stale


for _ in range(N_EPISODES):
    t = float(rng.integers(0, total_sec))
    sid += 1
    if rng.random() < P_MISFIRE_EPISODE:
        for k in range(1 + rng.geometric(0.6) - 1):
            text = "" if rng.random() < P_EMPTY_MISFIRE else pick(FRAGMENTS)[0]
            emit_turn(t, text, "misfire", None, 0, 0, True)
            gap = float(rng.uniform(3, 40))
            if rng.random() > 0.5:
                sid += 1
                gap += float(rng.uniform(20, 300))
            t += gap
        continue

    kinds = list(KIND_P)
    p = np.array(list(KIND_P.values()))
    now = START + pd.Timedelta(seconds=t)
    if now < FOLLOW_LAUNCH:
        p[kinds.index("따라와")] = 0
    kind = kinds[rng.choice(len(kinds), p=p / p.sum())]
    text, topic, fresh, clock, _w = pick(PERSON[kind])
    if rng.random() < P_EMPTY_PERSON:
        text = ""

    for _step in range(8):
        state, stale = emit_turn(t, text, kind, topic, fresh, clock, False)
        r = rng.random()
        if state == "none":
            nxt, same = ("retry", 0.85) if r < 0.20 else (None, 0)
        elif state == "fallback":
            nxt, same = ("retry", 0.03) if r < 0.22 else (None, 0)
        elif stale and rng.random() < P_NOTICE.get(topic, 0.03):
            nxt, same = "complaint", 0.7
        else:
            nxt, same = ("retry", 0.55) if r < 0.025 else (("follow", 0.8) if r < 0.32 else (None, 0))
        if nxt is None:
            break
        t += float(rng.uniform(4, 15))
        if rng.random() > same:
            sid += 1
            t += float(rng.uniform(10, 90))
        if nxt == "retry":
            if text == "":
                text, topic, fresh, clock, _w = pick(PERSON[kind])
        elif nxt == "complaint":
            text, kind, fresh, clock = str(rng.choice(COMPLAINTS)), "complaint", 0, 0
        else:
            kind = kinds[rng.choice(len(kinds), p=p / p.sum())]
            text, topic, fresh, clock, _w = pick(PERSON[kind])
            if rng.random() < P_EMPTY_PERSON:
                text = ""

logs = pd.DataFrame(
    rows, columns=["session_id", "log_type", "t", "text", "truth_misfire", "truth_topic", "truth_fresh", "truth_stale"]
)
logs["ts"] = (START + pd.to_timedelta(logs["t"].round(1), unit="s")).dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-5]
logs = logs.sort_values(["session_id", "t"], kind="stable").reset_index(drop=True)
logs.insert(0, "row_id", np.arange(len(logs)))
logs = logs.drop(columns="t")
DATA.mkdir(exist_ok=True)
logs.to_csv(DATA / "logs.csv.gz", index=False, compression=GZ)
print(logs.shape, logs.log_type.value_counts().sort_index().to_dict())
