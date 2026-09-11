"""가상 발화 생성. 일상 대화와 상담 스크립트, 두 출처의 발화를 같은 스키마로 만든다.

심어둔 구조는 셋이다.
A. 문체는 출처를 따르고 우울 내용은 truth_depressed를 따른다. 둘이 따로 종속된다. 문체는 5% 확률로 출처와
   어긋나게 했는데, 출처 분류가 100%가 되지 않게 하는 장치다. 상담 스크립트 20,000건 중 4,000건은 일상 의도라
   우울이 아니고, 원 프로젝트는 이걸 학습에서 뺐다. 일상 대화의 sadness 발화 일부는 실제로 우울하다.
B. 화자마다 음성 지문과 감정 분포 편향이 있다. 음성 채널이 감정에 대해 가진 진짜 정보는 고각성 감정인 angry,
   fear, surprise에 몰려 있고 neutral과 disgust에는 거의 없다.
C. 텍스트와 음성 신호가 정답을 가리키는지를 truth_*_informative에 기록해 oracle 상한을 계산할 수 있게 한다.
truth_* 열은 채점 전용이다. feature로 쓰지 않는다.
"""

import numpy as np
import pandas as pd
from common import AUDIO_COLS, DATA, EMOTION_P, EMOTIONS, N_COUNSEL_NORMAL, N_COUNSEL_SYMPTOM, N_DAILY, N_SPEAKERS, SEED

rng = np.random.default_rng(SEED)

COUNSEL_FILLER = ["요즘", "자꾸", "선생님", "계속", "사실은", "그래서", "제가", "아무래도", "며칠째"]
COUNSEL_END = ["요", "어요", "습니다", "거든요", "네요", "것 같아요", "더라고요"]
DAILY_FILLER = ["진짜", "완전", "오늘", "어제", "나", "그냥", "아", "근데", "약간"]
DAILY_END = ["어", "네", "야", "지", "다", "ㅋㅋ", "ㅠㅠ", "임"]

DEPRESSED = [
    "무의미하",
    "외롭",
    "무기력하",
    "우울하",
    "죽고 싶",
    "눈물이 나",
    "잠이 안 오",
    "입맛이 없",
    "희망이 없",
    "지쳤",
    "숨이 막히",
    "다 포기하고 싶",
    "가라앉",
    "의욕이 없",
    "쓸모없",
    "아무것도 하기 싫",
    "혼자인 것 같",
]
NORMAL_LIFE = [
    "여행 가",
    "맛집 갔",
    "주말에 쉬",
    "친구 만났",
    "운동했",
    "영화 봤",
    "강아지 산책",
    "날씨 좋",
    "회사 갔",
    "프로젝트 끝났",
    "밥 먹었",
    "카페 갔",
    "책 읽었",
    "청소했",
    "장 봤",
    "부모님 뵀",
    "시험 봤",
    "이사했",
]
EMOTION_TOKENS = {
    "happiness": ["기분 좋", "신나", "행복하", "웃음이 나", "설레", "즐거웠", "고마웠", "뿌듯하"],
    "angry": ["화가 나", "짜증 나", "열받", "빡치", "억울하", "참을 수가 없", "어이없"],
    "neutral": ["그랬", "했", "갔다 왔", "있었", "확인했", "정리했", "보냈"],
    "sadness": ["슬프", "울었", "서운하", "그립", "허전하", "속상하", "마음이 아프"],
    "disgust": ["역겹", "싫", "더럽", "꼴 보기 싫", "질렸", "구역질 나", "정 떨어지"],
    "surprise": ["깜짝 놀랐", "믿기지 않", "헐", "갑자기", "예상 못 했", "소름", "대박"],
    "fear": ["무서웠", "두렵", "불안하", "떨렸", "겁이 나", "걱정돼", "심장이 뛰"],
}
SYMPTOM_INTENT = [
    "정신증상/초조함",
    "정신증상/불안",
    "정신증상/무기력",
    "정신증상/기분우울",
    "정신증상/식욕저하",
    "정신증상/수면문제",
    "정신증상/자책",
    "정신증상/집중력저하",
]
NORMAL_INTENT = ["일상/인사", "일상/취미", "일상/가족", "일상/직장", "일상/감사", "일상/날씨"]

P_DAILY_SAD_DEPRESSED = 0.25  # 일상 sadness 발화 중 실제 우울 비율
Q_DEP_TEXT = 0.80  # 우울 내용 토큰이 정답을 가리킬 확률
Q_EMO_TEXT = 0.82  # 감정 토큰이 정답 감정을 가리킬 확률 (나머지는 다른 감정 토큰)
Q_EMO_AUDIO = 0.70  # 음성 오프셋이 실제로 실리는 확률
# 감정별 음성 오프셋. 고각성 감정은 rms와 zcr에 크게, 저각성은 작게. 값이 큰 감정일수록 음성 채널이 그 감정을
# 잘 구분한다. 심어둔 구조 B의 핵심 파라미터
AUDIO_OFFSET = {
    "angry": {"rms_mean": 2.2, "rms_std": 1.4, "zcr_mean": 1.5, "mfcc1_mean": 1.2},
    "fear": {"rms_mean": 1.0, "zcr_mean": 2.0, "mfcc2_mean": 1.6, "zcr_std": 1.2},
    "surprise": {"rms_std": 2.0, "zcr_std": 1.6, "mfcc3_mean": 1.4},
    "happiness": {"rms_mean": 1.0, "mfcc4_mean": 1.4, "mfcc1_mean": -0.6},
    "sadness": {"rms_mean": -1.4, "mfcc5_mean": -1.2, "zcr_mean": -0.8},
    "disgust": {"mfcc6_mean": 0.9},
    "neutral": {},
}
SPEAKER_FP_SD = 1.5  # 화자 음성 지문 크기. 감정 오프셋보다 크다
# 화자 지문은 음색 계열인 MFCC 7~13, chroma, duration에 실린다. 감정 오프셋은 에너지와 피치 계열에 실린다
SPEAKER_FP_COLS = [f"mfcc{i}_mean" for i in range(7, 14)] + ["chroma_mean", "chroma_std", "duration"]
AUDIO_NOISE_SD = 1.0
SPEAKER_EMO_CONC = 0.4  # 화자별 감정 분포 편향. 작을수록 편향 큼
P_STYLE_CROSS = 0.05  # 출처와 다른 문체로 말하는 발화 비율. 경어체와 반말이 바뀐다


def pick(pool, k=1):
    idx = rng.choice(len(pool), size=k, replace=False)
    return [pool[i] for i in idx]


def build_text(source, truth_dep, emotion, informative_dep, informative_emo):
    style = source if rng.random() >= P_STYLE_CROSS else ("daily" if source == "counsel" else "counsel")
    if style == "counsel":
        filler, ends = COUNSEL_FILLER, COUNSEL_END
        n_fill = rng.integers(1, 4)
    else:
        filler, ends = DAILY_FILLER, DAILY_END
        n_fill = rng.integers(0, 3)
    parts = pick(filler, n_fill) if n_fill else []
    if informative_dep:
        parts += pick(DEPRESSED if truth_dep else NORMAL_LIFE, rng.integers(1, 3))
    else:
        parts += pick(NORMAL_LIFE if rng.random() < 0.5 else DEPRESSED, 1)
    if emotion is not None:
        emo_src = emotion if informative_emo else rng.choice([e for e in EMOTIONS if e != emotion])
        parts += pick(EMOTION_TOKENS[emo_src], rng.integers(1, 3))
    rng.shuffle(parts)
    return " ".join(parts) + pick(ends)[0]


rows = []

N_COUNSEL = N_COUNSEL_SYMPTOM + N_COUNSEL_NORMAL
counsel_truth = np.concatenate([np.ones(N_COUNSEL_SYMPTOM, int), np.zeros(N_COUNSEL_NORMAL, int)])
rng.shuffle(counsel_truth)
for i in range(N_COUNSEL):
    truth_dep = int(counsel_truth[i])
    intent = rng.choice(SYMPTOM_INTENT if truth_dep else NORMAL_INTENT)
    inf_dep = int(rng.random() < Q_DEP_TEXT)
    rows.append(
        dict(
            utt_id=f"c{i:05d}",
            source="counsel",
            speaker_id=None,
            intent=intent,
            emotion_label=None,
            text=build_text("counsel", truth_dep, None, inf_dep, None),
            truth_depressed=truth_dep,
            truth_text_informative_dep=inf_dep,
            truth_text_informative_emo=None,
            truth_audio_informative=None,
        )
    )

speaker_w = rng.dirichlet(np.ones(N_SPEAKERS) * 2.0)
speaker_ids = rng.choice(N_SPEAKERS, size=N_DAILY, p=speaker_w)
speaker_emo_p = rng.dirichlet(EMOTION_P * len(EMOTIONS) * SPEAKER_EMO_CONC, size=N_SPEAKERS)
speaker_fp = np.zeros((N_SPEAKERS, len(AUDIO_COLS)))
fp_idx = [AUDIO_COLS.index(c) for c in SPEAKER_FP_COLS]
speaker_fp[:, fp_idx] = rng.normal(0, SPEAKER_FP_SD, size=(N_SPEAKERS, len(fp_idx)))

audio_rows = []
for i in range(N_DAILY):
    s = int(speaker_ids[i])
    emotion = EMOTIONS[rng.choice(len(EMOTIONS), p=speaker_emo_p[s])]
    truth_dep = int(emotion == "sadness" and rng.random() < P_DAILY_SAD_DEPRESSED)
    inf_dep = int(rng.random() < Q_DEP_TEXT)
    inf_emo = int(rng.random() < Q_EMO_TEXT)
    inf_audio = int(rng.random() < Q_EMO_AUDIO)
    text = build_text("daily", truth_dep, emotion, inf_dep, inf_emo)

    a = speaker_fp[s] + rng.normal(0, AUDIO_NOISE_SD, size=len(AUDIO_COLS))
    if inf_audio:
        for col, w in AUDIO_OFFSET[emotion].items():
            a[AUDIO_COLS.index(col)] += w
    rows.append(
        dict(
            utt_id=f"d{i:05d}",
            source="daily",
            speaker_id=f"spk{s:03d}",
            intent=None,
            emotion_label=emotion,
            text=text,
            truth_depressed=truth_dep,
            truth_text_informative_dep=inf_dep,
            truth_text_informative_emo=inf_emo,
            truth_audio_informative=inf_audio,
        )
    )
    audio_rows.append(a)

df = pd.DataFrame(rows)
audio = pd.DataFrame(np.full((len(df), len(AUDIO_COLS)), np.nan), columns=AUDIO_COLS)
audio.iloc[N_COUNSEL:] = np.array(audio_rows)
df = pd.concat([df, audio.round(4)], axis=1)
df.to_csv(DATA / "utterances.csv", index=False)

print(df.shape)
print(df.source.value_counts())
print("counsel truth_depressed rate:", df[df.source == "counsel"].truth_depressed.mean().round(3))
print("daily truth_depressed rate:", df[df.source == "daily"].truth_depressed.mean().round(4))
print(df[df.source == "daily"].emotion_label.value_counts(normalize=True).round(3))
print(df.sample(5, random_state=SEED)[["source", "text", "truth_depressed", "emotion_label"]].to_string())
