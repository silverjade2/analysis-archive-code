"""Kiwi 토큰화 캐시 (녹음 STT + 감정 코퍼스)"""

import _path  # noqa: F401
import pandas as pd
from config import DATA
from features import tokenize

rec = pd.read_csv(DATA / "recordings.csv")
rec_tok = pd.DataFrame(dict(recording_id=rec.recording_id, tokens=tokenize(rec.stt_text)))
rec_tok.to_csv(DATA / "recordings_tokens.csv", index=False)
emo = pd.read_csv(DATA / "emotion_corpus.csv")
emo_tok = pd.DataFrame(dict(sent_id=emo.sent_id, tokens=tokenize(emo.text)))
emo_tok.to_csv(DATA / "emotion_tokens.csv", index=False)
print(rec_tok.tokens.iloc[0])
print(emo_tok.tokens.iloc[0])
print(
    "mean tokens:",
    rec_tok.tokens.str.split().str.len().mean().round(1),
    emo_tok.tokens.str.split().str.len().mean().round(1),
)
