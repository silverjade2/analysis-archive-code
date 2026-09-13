"""03. 감정 코퍼스 생성 (D2 감성 대화 말뭉치 스키마).
문장 24,000개: 사용자 발화 2문장(상황 사건 + 감정 표현), 6대 감정, 상황 7종, 연령대, 성별.
직장 상황 비중이 가장 크다(서비스 타깃). 상황별로 감정 분포가 다르다."""

import _path  # noqa: F401
import lexicon as L
import numpy as np
import pandas as pd
from config import AGE_GROUPS, DATA, EMOTIONS, N_EMOTION_SENT, SEED, SITUATION_P, SITUATIONS

rng = np.random.default_rng(SEED + 2)
rows = []
for i in range(N_EMOTION_SENT):
    sit = SITUATIONS[rng.choice(len(SITUATIONS), p=SITUATION_P)]
    emo = EMOTIONS[rng.choice(len(EMOTIONS), p=L.SITUATION_EMOTION_P[sit])]
    events = L.SITUATION_EVENTS[sit]
    pos = [e for e in events if any(k in e for k in L.POSITIVE_EVENT_KEYWORDS)]
    neg = [e for e in events if e not in pos]
    pool = pos if emo == "기쁨" else neg
    ev = pool[rng.integers(len(pool))]
    if rng.random() < L.P_AMBIGUOUS:
        text = ev + ". " + L.AMBIGUOUS_REACTIONS[rng.integers(len(L.AMBIGUOUS_REACTIONS))]
    else:
        tmpl = L.EMOTION_TEMPLATES[emo][rng.integers(len(L.EMOTION_TEMPLATES[emo]))]
        text = tmpl.format(ev=ev)
    age = AGE_GROUPS[rng.choice(3, p=[0.55, 0.33, 0.12])]
    if age == "노년" and rng.random() < 0.6:
        text = text.replace("같아.", "같아요.").replace("나.", "나요.").replace("어.", "어요.").replace("해.", "해요.")
    rows.append(
        dict(
            sent_id=f"E{i:05d}",
            situation=sit,
            emotion=emo,
            age_group=age,
            sex=("F" if rng.random() < 0.58 else "M"),
            text=text,
        )
    )
df = pd.DataFrame(rows)
df.to_csv(DATA / "emotion_corpus.csv", index=False)
print(df.shape)
print(pd.crosstab(df.situation, df.emotion))
print(df.sample(6, random_state=SEED)[["situation", "emotion", "text"]].to_string())
