"""우울 진단 v1. 원 프로젝트 방식.

상담 스크립트의 증상 의도 발화는 1, 일상 대화는 0. 상담 스크립트의 일상 의도 발화 4,000건은 학습에서 뺀다.
홀드아웃 7:3. 같은 테스트셋을 학습 라벨과 truth_depressed 두 기준으로 채점하고, 학습에서 뺀 상담 일상 의도
발화, 즉 모델이 한 번도 본 적 없는 4,000건을 넣어 우울로 찍는 비율을 본다.
"""

import numpy as np
import pandas as pd
from common import DATA, RESULTS, SEED, depression_split
from models import TextClf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

df = pd.read_csv(DATA / "utterances.csv", low_memory=False)
df["is_symptom"] = df.intent.fillna("").str.startswith("정신증상").astype(int)
df["label_v1"] = df.is_symptom  # 상담 증상 = 1, 나머지 0
tr, te = depression_split(df)
tr_v1 = tr[~((tr.source == "counsel") & (tr.is_symptom == 0))]  # 원 방식: 상담 일상 의도는 학습에서 제외
te_v1 = te[~((te.source == "counsel") & (te.is_symptom == 0))]  # 원 프로젝트가 채점한 범위

m = TextClf(SEED).fit(tr_v1.text, tr_v1.label_v1)


def metrics(y, p):
    return dict(
        accuracy=accuracy_score(y, p),
        precision=precision_score(y, p, zero_division=0),
        recall=recall_score(y, p, zero_division=0),
        f1=f1_score(y, p, zero_division=0),
    )


rows = []
p_v1 = m.predict(te_v1.text)
rows.append(dict(scoring="train_label", subset="test_as_original", n=len(te_v1), **metrics(te_v1.label_v1, p_v1)))
rows.append(
    dict(scoring="truth_depressed", subset="test_as_original", n=len(te_v1), **metrics(te_v1.truth_depressed, p_v1))
)
p_all = m.predict(te.text)
rows.append(dict(scoring="truth_depressed", subset="test_all", n=len(te), **metrics(te.truth_depressed, p_all)))
cm = (te.source == "counsel").values
rows.append(
    dict(
        scoring="truth_depressed",
        subset="test_counsel_only",
        n=int(cm.sum()),
        **metrics(te.truth_depressed[cm], p_all[cm]),
    )
)
res = pd.DataFrame(rows).round(4)
res.to_csv(RESULTS / "depression_v1_metrics.csv", index=False)

# 상담 일상 의도는 학습에 쓰지 않았으므로 4,000건 전부에 넣어 본다
dropped = df[(df.source == "counsel") & (df.is_symptom == 0)]
p_dropped = m.predict(dropped.text)
d = te[te.source == "daily"]
p_d = p_all[(te.source == "daily").values]
c1 = te[(te.source == "counsel") & (te.is_symptom == 1)]
p_c1 = p_all[((te.source == "counsel") & (te.is_symptom == 1)).values]
detail = pd.DataFrame(
    [
        dict(
            group="counsel & normal intent (excluded from training, all 4,000)",
            n=len(dropped),
            pred_depressed_rate=p_dropped.mean(),
        ),
        dict(group="counsel & symptom intent (test)", n=len(c1), pred_depressed_rate=p_c1.mean()),
        dict(
            group="daily & truth_depressed=1 (test)",
            n=int((d.truth_depressed == 1).sum()),
            pred_depressed_rate=p_d[(d.truth_depressed == 1).values].mean(),
        ),
        dict(
            group="daily & truth_depressed=0 (test)",
            n=int((d.truth_depressed == 0).sum()),
            pred_depressed_rate=p_d[(d.truth_depressed == 0).values].mean(),
        ),
    ]
).round(4)
detail.to_csv(RESULTS / "depression_v1_by_group.csv", index=False)

# oracle. 내용 토큰이 정답을 가리키면 맞히고, 아니면 출처 안의 다수 클래스로 찍는다. 상담 발화의 80%가 우울이므로
# 상담이면 1, 일상이면 0. 생성 모델을 아는 판정자의 상한이다
inf = te.truth_text_informative_dep.values.astype(bool)
prior = (te.source == "counsel").astype(int).values
op = np.where(inf, te.truth_depressed.values, prior)
pd.DataFrame(
    [
        dict(
            subset="test_all",
            oracle_accuracy=accuracy_score(te.truth_depressed, op),
            oracle_f1=f1_score(te.truth_depressed, op),
        )
    ]
).round(4).to_csv(RESULTS / "depression_oracle.csv", index=False)

feats = m.top_features(40)
pd.DataFrame(
    [dict(direction="depressed(+)", feature=f, coef=c) for f, c in feats["pos"]]
    + [dict(direction="daily(-)", feature=f, coef=c) for f, c in feats["neg"]]
).to_csv(RESULTS / "depression_v1_top_features.csv", index=False)
print(res.to_string())
print(detail.to_string())
print("top:", [f for f, _ in feats["pos"][:12]])
