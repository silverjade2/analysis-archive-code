"""라벨 정의 비교. 환자군 여부 vs PHQ-9 >= 10으로 각각 학습, 둘 다 PHQ 라벨로 채점. 치료 어휘 마스킹 포함

관심은 환자군인데 비우울인 351명의 예측 확률
"""

import _path  # noqa: F401
import numpy as np
import pandas as pd
from config import NARRATIVE_TASKS, RESULTS, SEED
from evalutil import cross_val_predict, fp_text, load_recordings, metrics, participant_level
from features import mask_treatment
from models import TextModel

df = load_recordings()
df = df[df.task.isin(NARRATIVE_TASKS)].reset_index(drop=True)
df["label_group"] = (df.group == "patient").astype(int)
df["tokens_masked"] = df.tokens.map(mask_treatment)

rows = []
probs = {}
for label_name, ycol in [("group_label", "label_group"), ("phq_label", "label_depressed")]:
    for tok_name, tcol in [("full", "tokens"), ("treatment_masked", "tokens_masked")]:

        def fp(tr, te, s, ycol=ycol, tcol=tcol):
            return fp_text(tr, te, s, y_col=ycol, token_col=tcol)

        oof = cross_val_predict(df, fp, y_col=ycol, split="participant", seed=SEED)
        probs[(label_name, tok_name)] = oof
        m_own = metrics(df[ycol], oof)
        m_phq = metrics(df.label_depressed, oof)
        par = participant_level(df, oof)
        sub = df[(df.group == "patient") & (df.label_depressed == 0)]
        rows.append(
            dict(
                trained_on=label_name,
                tokens=tok_name,
                auc_vs_own_label=m_own["auc"],
                auc_vs_phq_label=m_phq["auc"],
                par_auc_vs_phq=par["auc"],
                mean_prob_patient_not_depressed=oof[sub.index].mean(),
                mean_prob_control_not_depressed=oof[
                    df[(df.group == "control") & (df.label_depressed == 0)].index
                ].mean(),
                mean_prob_patient_depressed=oof[df[(df.group == "patient") & (df.label_depressed == 1)].index].mean(),
            )
        )
        print(label_name, tok_name, round(m_own["auc"], 4), round(m_phq["auc"], 4))
res = pd.DataFrame(rows).round(4)
res.to_csv(RESULTS / "label_definition_comparison.csv", index=False)

# 그림용 분위. 그룹 x 라벨 4셀
cells = []
for (label_name, tok_name), oof in probs.items():
    if tok_name != "full":
        continue
    for g in ["control", "patient"]:
        for lab in [0, 1]:
            m = ((df.group == g) & (df.label_depressed == lab)).values
            for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
                cells.append(
                    dict(
                        trained_on=label_name,
                        group=g,
                        label_depressed=lab,
                        quantile=q,
                        prob=np.quantile(oof[m], q),
                        n=int(m.sum()),
                    )
                )
pd.DataFrame(cells).round(4).to_csv(RESULTS / "label_definition_prob_quantiles.csv", index=False)
for label_name, ycol in [("group_label", "label_group"), ("phq_label", "label_depressed")]:
    tf = TextModel(SEED).fit(list(df.tokens), df[ycol].values).top_features(25)
    tf.round(4).to_csv(RESULTS / f"top_features_{label_name}.csv", index=False)
    print(label_name, tf.head(12).feature.tolist())
print(res.to_string(index=False))
