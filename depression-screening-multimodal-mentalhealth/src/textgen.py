"""서술 과제 STT 텍스트 생성. 템플릿 선택 P(t | z) ~ exp(TEXT_SIGNAL * w_t * z), z = PHQ-9 표준화 점수"""

import re

import lexicon as L
import numpy as np
from config import P_IDIOLECT, P_STT_ERROR, P_TREATMENT_MENTION, TEXT_SIGNAL

POOLS = {
    "when": L.WHEN,
    "who": L.WHO,
    "place": L.PLACE,
    "activity": L.ACTIVITY,
    "event_pos": L.EVENT_POS,
    "detail_pos": L.DETAIL_POS,
    "detail_neutral": L.DETAIL_NEUTRAL,
    "feel_pos": L.FEEL_POS,
    "feel_look": L.FEEL_LOOK,
    "feel_pos_short": L.FEEL_POS_SHORT,
    "anhedonia": L.ANHEDONIA,
    "anhedonia_long": L.ANHEDONIA_LONG,
    "event_neg": L.EVENT_NEG,
    "detail_neg": L.DETAIL_NEG,
    "coping": L.COPING,
    "symptom": L.SYMPTOM,
    "symptom_long": L.SYMPTOM_LONG,
    "selfblame": L.SELFBLAME,
    "hopeless": L.HOPELESS,
    "pos_obs": L.POS_OBS,
    "neutral_obs": L.NEUTRAL_OBS,
    "pic_neg_read": L.PIC_NEG_READ,
    "neg_obs": L.NEG_OBS,
    "neu_obs": L.NEU_OBS,
}
TEMPLATES = {
    "happy_memory": L.HAPPY,
    "hard_memory": L.HARD,
    "pic_positive": L.PIC_POS,
    "pic_negative": L.PIC_NEG,
    "pic_neutral": L.PIC_NEU,
}
SLOT_RE = re.compile(r"\{(\w+)\}")


def _choose_template(task, z, rng):
    ws = np.array([w for w, _ in TEMPLATES[task]])
    logits = TEXT_SIGNAL * ws * z
    p = np.exp(logits - logits.max())
    p /= p.sum()
    i = rng.choice(len(ws), p=p)
    return TEMPLATES[task][i][1], i, ws[i]


def _fill(template, z, rng):
    def rep(m):
        key = m.group(1)
        if key == "fill":
            # 머뭇거림은 라벨과 무관하게 절반씩. 반응 지연이 우울 신호로 새지 않게
            pool = L.FILLER_HIGH if rng.random() < 0.5 else L.FILLER_LOW
            return pool[rng.integers(len(pool))]
        pool = POOLS[key]
        return pool[rng.integers(len(pool))]

    s = SLOT_RE.sub(rep, template)
    return re.sub(r"\s+", " ", s).strip()


def _add_treatment(text, task, group, rng):
    """치료 맥락 언급. 환자군에서 훨씬 자주, 우울 여부와는 무관"""
    p = P_TREATMENT_MENTION[group] * (1.0 if task in ("happy_memory", "hard_memory") else 0.7)
    if rng.random() >= p:
        return text, 0
    if task in ("happy_memory", "hard_memory") and rng.random() < 0.5:
        return L.TREATMENT[rng.integers(len(L.TREATMENT))] + " " + text, 1
    pool = L.TREATMENT_TAIL if task in ("happy_memory", "hard_memory") else L.TREATMENT_PIC_TAIL
    return text + " " + pool[rng.integers(len(pool))], 1


def _stt_noise(text, rng):
    words = text.split(" ")
    out = []
    for w in words:
        if w in L.STT_CONFUSIONS and rng.random() < P_STT_ERROR * 4:
            out.append(L.STT_CONFUSIONS[w])
        elif rng.random() < P_STT_ERROR * 0.3:
            continue  # 어절 탈락
        else:
            out.append(w)
    return " ".join(out)


def _add_idiolect(text, tics, rng):
    if not tics or rng.random() >= P_IDIOLECT:
        return text
    tic = tics[rng.integers(len(tics))]
    sents = text.split(". ")
    i = rng.integers(len(sents))
    sents[i] = tic + " " + sents[i]
    return ". ".join(sents)


def generate_narrative(task, z, group, rng, tics=()):
    template, t_idx, w = _choose_template(task, z, rng)
    text = _fill(template, z, rng)
    text = _add_idiolect(text, tics, rng)
    text, treated = _add_treatment(text, task, group, rng)
    text = _stt_noise(text, rng)
    return dict(text=text, template_idx=int(t_idx), template_weight=float(w), treatment_mention=treated)
