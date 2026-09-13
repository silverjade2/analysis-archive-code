"""텍스트 피처. Kiwi 형태소 분석 → "형태소/품사" 토큰. 결과는 data/에 캐시한다."""

from kiwipiepy import Kiwi

_kiwi = None


def kiwi():
    global _kiwi
    if _kiwi is None:
        _kiwi = Kiwi()
    return _kiwi


def tokenize(texts):
    """각 문장을 '형태소/품사' 토큰의 공백 구분 문자열로."""
    out = []
    for toks in kiwi().tokenize(list(texts)):
        out.append(" ".join(f"{t.form}/{t.tag}" for t in toks if t.tag not in ("SF", "SP", "SS", "SE", "SO", "SW")))
    return out


# 치료 맥락 어휘 (환자군 shortcut 진단용 마스킹 목록).
# 면담에서 치료 맥락을 말할 때 쓰는 표현들을 형태소 분석해 내용 형태소(명사·동사·형용사·부사)만 모은다.
_TREATMENT_PHRASES = [
    "병원 다니기 시작하고 나서는",
    "약 먹고 나서부터는",
    "선생님이 말씀하신 대로",
    "진료받은 뒤로는",
    "상담 받으면서",
    "약 바꾸고 나서",
    "여기 처음 왔을 때랑 비교하면",
    "약 먹으면서 많이 좋아졌어요",
    "선생님 덕분에 요즘은 괜찮아요",
    "진료 받고 나서 좀 나아졌어요",
    "약을 꾸준히 먹고 있어요",
    "다음 진료 때 말씀드리려고요",
    "병원 오는 길에",
    "선생님이 지난번에 보여주신 그림",
    "진료 기다리면서",
    "처방",
    "상담사",
    "정신과",
    "치료",
]
_CONTENT_TAGS = {"NNG", "NNP", "VV", "VA", "MAG", "XSN", "XR"}
_STOP = {"많이", "좀", "요즘", "때", "그림", "길", "생각", "처음", "다음", "지난번"}


def treatment_morphs():
    out = set()
    for toks in kiwi().tokenize(_TREATMENT_PHRASES):
        for t in toks:
            if t.tag in _CONTENT_TAGS and t.form not in _STOP:
                out.add(t.form)
    return out


_TM = None


def mask_treatment(token_str):
    global _TM
    if _TM is None:
        _TM = treatment_morphs()
    return " ".join(t for t in token_str.split() if t.split("/")[0] not in _TM)
