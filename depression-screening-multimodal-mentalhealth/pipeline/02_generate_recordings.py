"""02. 녹음 생성: 참가자 × 과제 8개 = 16,000건. STT 텍스트 + 음향 피처 20차원.
텍스트: 서술 과제 5개는 템플릿(가중치·PHQ z), 숫자 세기·낭독은 고정 텍스트.
음성: 화자 지문 + 과제 효과 + 우울 효과 + 잡음.
truth_* 컬럼(template_weight 등)은 생성기의 잠재 변수. 채점·진단에만 쓴다."""

import _path  # noqa: F401
import lexicon as L
import numpy as np
import pandas as pd
from audiogen import recording_features, speaker_fingerprint
from config import DATA, NARRATIVE_TASKS, SEED, TASK_KO, TASKS
from textgen import generate_narrative

rng = np.random.default_rng(SEED + 1)
cohort = pd.read_csv(DATA / "participants.csv")
rows = []
for _, p in cohort.iterrows():
    fp = speaker_fingerprint(rng, p.sex)
    tics = [L.IDIOLECT[i] for i in rng.choice(len(L.IDIOLECT), size=rng.integers(1, 3), replace=False)]
    for task in TASKS:
        if task in NARRATIVE_TASKS:
            g = generate_narrative(task, p.phq9_z, p.group, rng, tics)
            text, tw, ti, treat = g["text"], g["template_weight"], g["template_idx"], g["treatment_mention"]
        else:
            text = {"count_up": L.COUNT_UP, "count_down": L.COUNT_DOWN, "reading": L.READING}[task]
            tw, ti, treat = 0.0, -1, 0
        n_words = len(text.split())
        feats = recording_features(fp, task, p.phq9_z, n_words, rng)
        rows.append(
            dict(
                recording_id=f"{p.participant_id}_{task}",
                participant_id=p.participant_id,
                task=task,
                stt_text=text,
                n_words=n_words,
                **{k: round(v, 4) for k, v in feats.items()},
                truth_template_weight=tw,
                truth_template_idx=ti,
                truth_treatment_mention=treat,
            )
        )
rec = pd.DataFrame(rows)
rec.to_csv(DATA / "recordings.csv", index=False)
print(rec.shape)
print(rec.groupby("task").n_words.agg(["mean", "min", "max"]).round(1))
m = rec.merge(cohort[["participant_id", "label_depressed", "group"]], on="participant_id")
for task in NARRATIVE_TASKS:
    sub = m[m.task == task]
    print(f"\n[{TASK_KO[task]}]")
    for lab in [0, 1]:
        print(f"  label={lab}:", sub[sub.label_depressed == lab].sample(2, random_state=SEED).stt_text.tolist())
