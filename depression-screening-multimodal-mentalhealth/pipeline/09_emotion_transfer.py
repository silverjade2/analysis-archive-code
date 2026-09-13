"""09. 감정 모델과 스크리닝으로의 전이.
(1) D2 스타일 코퍼스로 6감정 텍스트 분류기(5-fold). 전체·상황별 정확도, 혼동행렬.
(2) 그 분류기를 D1 서술 과제 STT에 적용해 부정 정서 점수 = P(슬픔)+P(불안)+P(상처)를 만들고,
    참가자 평균이 PHQ-9 라벨을 얼마나 가르는지(AUC), PHQ-9 점수와 상관(Spearman)을 본다.
    같은 데이터로 직접 학습한 텍스트 모델(08)과 비교한다."""

import _path  # noqa: F401
import numpy as np
import pandas as pd
from config import DATA, EMOTIONS, N_FOLDS, NARRATIVE_TASKS, RESULTS, SEED, TASK_KO
from evalutil import load_recordings
from models import TextModel
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

emo = pd.read_csv(DATA / "emotion_corpus.csv").merge(pd.read_csv(DATA / "emotion_tokens.csv"), on="sent_id")
y = emo.emotion.values
oof = np.empty(len(emo), dtype=object)
proba = np.zeros((len(emo), len(EMOTIONS)))
for tr, te in StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(emo, y):
    m = TextModel(SEED).fit(list(emo.tokens.iloc[tr]), y[tr])
    p = m.predict_proba(list(emo.tokens.iloc[te]))
    order = [list(m.clf.classes_).index(e) for e in EMOTIONS]
    proba[te] = p[:, order]
    oof[te] = m.clf.classes_[p.argmax(1)]
overall = pd.DataFrame(
    [dict(scope="all", n=len(emo), accuracy=accuracy_score(y, oof), f1_macro=f1_score(y, oof, average="macro"))]
)
by_sit = (
    emo.assign(pred=oof)
    .groupby("situation")
    .apply(
        lambda g: pd.Series(
            dict(
                n=len(g),
                accuracy=accuracy_score(g.emotion, g.pred),
                f1_macro=f1_score(g.emotion, g.pred, average="macro"),
            )
        ),
        include_groups=False,
    )
    .reset_index()
    .rename(columns={"situation": "scope"})
)
acc = pd.concat([overall, by_sit]).round(4)
acc.to_csv(RESULTS / "emotion_accuracy.csv", index=False)
cm = pd.DataFrame(confusion_matrix(y, oof, labels=EMOTIONS), index=EMOTIONS, columns=EMOTIONS)
cm.to_csv(RESULTS / "emotion_confusion.csv")
print(acc.to_string(index=False))
print(cm)

# (2) 전이: 전체 코퍼스로 학습한 감정 모델을 D1 서술 STT에 적용
full = TextModel(SEED).fit(list(emo.tokens), y)
df = load_recordings()
nar = df[df.task.isin(NARRATIVE_TASKS)].reset_index(drop=True)
P = full.predict_proba(list(nar.tokens))
cls = list(full.clf.classes_)
nar["p_neg_affect"] = P[:, [cls.index(e) for e in ["슬픔", "불안", "상처"]]].sum(1)
nar["p_joy"] = P[:, cls.index("기쁨")]
nar["emotion_pred"] = full.clf.classes_[P.argmax(1)]
par = (
    nar.groupby("participant_id")
    .agg(y=("label_depressed", "first"), phq=("phq9", "first"), neg=("p_neg_affect", "mean"), joy=("p_joy", "mean"))
    .reset_index()
)
rho, _ = spearmanr(par.phq, par.neg)
rows = [
    dict(
        signal="emotion_model_neg_affect (transfer)",
        level="participant",
        auc=roc_auc_score(par.y, par.neg),
        spearman_phq=rho,
    ),
    dict(
        signal="emotion_model_neg_affect (transfer)",
        level="recording",
        auc=roc_auc_score(nar.label_depressed, nar.p_neg_affect),
        spearman_phq=spearmanr(nar.phq9, nar.p_neg_affect)[0],
    ),
]
ldc = pd.read_csv(RESULTS / "label_definition_comparison.csv")
rows.append(
    dict(
        signal="dedicated_text_model (08, phq_label)",
        level="participant",
        auc=ldc[(ldc.trained_on == "phq_label") & (ldc.tokens == "full")].par_auc_vs_phq.iloc[0],
        spearman_phq=np.nan,
    )
)
rows.append(
    dict(
        signal="dedicated_text_model (08, phq_label)",
        level="recording",
        auc=ldc[(ldc.trained_on == "phq_label") & (ldc.tokens == "full")].auc_vs_phq_label.iloc[0],
        spearman_phq=np.nan,
    )
)
tr_ = pd.DataFrame(rows).round(4)
tr_.to_csv(RESULTS / "emotion_transfer.csv", index=False)
# 과제별 감정 예측 분포 (전이 모델이 D1 텍스트를 어떻게 읽는지)
dist = pd.crosstab(nar.task.map(TASK_KO), nar.emotion_pred, normalize="index").round(4)
dist.to_csv(RESULTS / "emotion_transfer_pred_by_task.csv")
par.round(4).to_csv(RESULTS / "emotion_transfer_participant_scores.csv", index=False)
print(tr_.to_string(index=False))
print(dist)
