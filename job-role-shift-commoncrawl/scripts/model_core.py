"""분류 모델의 feature, 점수 계산 (numpy와 표준 라이브러리만). 학습과 Spark 적용이 같은 코드를 쓴다

feature: 소문자, 글자만 2자 이상 토큰, 단어 1~2-gram, 1 + log(tf) x idf, L2 정규화
점수: 유형마다 logistic regression 계수와 절편. 확률이 유형별 threshold 이상이면 그 유형을 붙인다
"""

import html
import json
import re
from pathlib import Path

import numpy as np

MAX_CHARS = 4000  # 라벨을 붙일 때 본 범위와 같게
_TAG = re.compile(r"<[^>]*>")
_TAG_TAIL = re.compile(r"<[^>]*$")  # 4,000자에서 잘린 태그 꼬리
_WS = re.compile(r"[ \t\r\f\v\u00a0]+")
_NL = re.compile(r"\n\s*\n+")


def plain_text(raw: str | None) -> str:
    """본문 앞 4,000자의 태그 제거, 엔티티 풀기, 공백 정리"""
    t = (raw or "")[:MAX_CHARS]
    t = _TAG_TAIL.sub(" ", _TAG.sub(" ", t))
    t = html.unescape(t)
    t = _WS.sub(" ", t)
    return _NL.sub("\n", t).strip()


_EN_STOP = set(
    "the and of to in a for with on is are as be or an by at from that this will you we our your have has it their "
    "can all work team experience skills".split()
)
_WORD = re.compile(r"[^\W\d_]+")


def english_ratio(text: str) -> float:
    w = _WORD.findall(text.lower())
    return sum(x in _EN_STOP for x in w) / len(w) if w else 0.0


def is_english(text: str) -> bool:
    """불용어 비율 추정 (영어 본문은 보통 0.25 이상, 독일어, 프랑스어는 0.05 안팎)"""
    return english_ratio(text) >= 0.12


TOKEN = re.compile(r"[^\W\d_]{2,}")  # 글자만 2자 이상. 숫자는 빠져 python3은 python


def analyze(text: str) -> list[str]:
    """TfidfVectorizer(analyzer=analyze)로 넘기는 분석기. 1-gram + 2-gram"""
    t = TOKEN.findall((text or "").lower())
    return t + [t[i] + " " + t[i + 1] for i in range(len(t) - 1)]


class Model:
    def __init__(
        self,
        vocab: dict[str, int],
        idf: np.ndarray,
        coef: np.ndarray,
        intercept: np.ndarray,
        thresholds: np.ndarray,
        task_ids: list[str],
    ):
        self.vocab, self.idf, self.coef, self.intercept = (
            vocab,
            idf.astype(np.float64),
            coef.astype(np.float64),
            intercept,
        )
        self.thresholds, self.task_ids = thresholds, task_ids

    @classmethod
    def load(cls, folder: Path) -> "Model":
        z = np.load(folder / "model.npz")
        meta = json.loads((folder / "vocab.json").read_text(encoding="utf-8"))
        return cls(meta["vocab"], z["idf"], z["coef"], z["intercept"], z["thresholds"], meta["task_ids"])

    @classmethod
    def from_blobs(cls, npz_bytes: bytes, vocab_text: str) -> "Model":
        """파일 대신 바이트로 읽는다 (Spark 작업자에 모델을 실어 보낼 때)"""
        import io

        z = np.load(io.BytesIO(npz_bytes))
        meta = json.loads(vocab_text)
        return cls(meta["vocab"], z["idf"], z["coef"], z["intercept"], z["thresholds"], meta["task_ids"])

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            folder / "model.npz", idf=self.idf, coef=self.coef, intercept=self.intercept, thresholds=self.thresholds
        )
        (folder / "vocab.json").write_text(
            json.dumps({"vocab": self.vocab, "task_ids": self.task_ids}, ensure_ascii=False), encoding="utf-8"
        )

    def features(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        """(어휘 번호, 값) 희소 벡터. 어휘 밖 토큰은 버리고 남은 것으로 L2 정규화"""
        counts: dict[int, int] = {}
        for tok in analyze(text):
            j = self.vocab.get(tok)
            if j is not None:
                counts[j] = counts.get(j, 0) + 1
        if not counts:
            return np.empty(0, dtype=np.int64), np.empty(0)
        idx = np.fromiter(counts.keys(), dtype=np.int64, count=len(counts))
        tf = np.fromiter(counts.values(), dtype=np.float64, count=len(counts))
        v = (1.0 + np.log(tf)) * self.idf[idx]
        return idx, v / np.sqrt((v * v).sum())

    def proba(self, texts) -> np.ndarray:
        """(문서 수 x 유형 수) 확률"""
        out = np.empty((len(texts), len(self.task_ids)))
        for i, t in enumerate(texts):
            idx, v = self.features(t)
            z = self.coef[:, idx] @ v + self.intercept if len(idx) else self.intercept.copy()
            out[i] = 1.0 / (1.0 + np.exp(-z))
        return out

    def decide(self, p: np.ndarray) -> tuple[list[str], str]:
        """한 공고의 확률로 (tasks, primary_task). 넘은 유형이 없으면 none, 대표는 넘은 유형 중 확률 최대"""
        over = np.flatnonzero(p >= self.thresholds)
        if len(over) == 0:
            return [], "none"
        best = over[np.argmax(p[over])]
        return [self.task_ids[j] for j in over], self.task_ids[best]
