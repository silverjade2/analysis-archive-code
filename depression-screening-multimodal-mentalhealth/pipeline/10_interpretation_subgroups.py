"""10. 해석과 하위군 성능.
(1) PHQ 라벨 텍스트 모델의 상위 피처를 유형(증상·무쾌감·머뭇거림·치료맥락·기타)으로 분류.
(2) 융합 모델(참가자 분할)의 참가자 단위 AUC를 직장인 여부 / 성별 / 연령대 / 그룹별로. 부트스트랩 95% CI.
(3) 예측 확률 보정(calibration) 10분위."""

import _path  # noqa: F401
import numpy as np
import pandas as pd
from config import RESULTS, SEED
from evalutil import load_recordings
from sklearn.metrics import roc_auc_score

df = load_recordings()
oof = pd.read_csv(RESULTS / "split_comparison_oof_seed0.csv")
df = df.merge(
    oof[["recording_id", "participant__fusion", "participant__text", "participant__audio_gbm"]], on="recording_id"
)
par = (
    df.groupby("participant_id")
    .agg(
        y=("label_depressed", "first"),
        phq=("phq9", "first"),
        employed=("employed", "first"),
        sex=("sex", "first"),
        age=("age", "first"),
        group=("group", "first"),
        p_fusion=("participant__fusion", "mean"),
        p_text=("participant__text", "mean"),
        p_audio=("participant__audio_gbm", "mean"),
    )
    .reset_index()
)
par["age_band"] = pd.cut(par.age, [0, 29, 44, 59, 100], labels=["17-29", "30-44", "45-59", "60+"]).astype(str)
par["subgroup_employed"] = np.where(par.employed == 1, "직장인", "비직장인")

rng = np.random.default_rng(SEED)


def boot_auc(y, p, n=500):
    y = np.asarray(y)
    p = np.asarray(p)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        vals.append(roc_auc_score(y[i], p[i]))
    return roc_auc_score(y, p), np.percentile(vals, 2.5), np.percentile(vals, 97.5)


rows = []
for col in ["subgroup_employed", "sex", "age_band", "group"]:
    for val, g in par.groupby(col):
        if g.y.nunique() < 2:
            continue
        for model in ["p_fusion", "p_text", "p_audio"]:
            a, lo, hi = boot_auc(g.y, g[model])
            rows.append(
                dict(
                    dimension=col,
                    subgroup=val,
                    model=model.replace("p_", ""),
                    n=len(g),
                    depressed_share=g.y.mean(),
                    auc=a,
                    ci_low=lo,
                    ci_high=hi,
                )
            )
a, lo, hi = boot_auc(par.y, par.p_fusion)
rows.append(
    dict(
        dimension="all",
        subgroup="all",
        model="fusion",
        n=len(par),
        depressed_share=par.y.mean(),
        auc=a,
        ci_low=lo,
        ci_high=hi,
    )
)
sub = pd.DataFrame(rows).round(4)
sub.to_csv(RESULTS / "subgroup_auc.csv", index=False)

# 보정: 참가자 확률 10분위 vs 실제 우울 비율
par["decile"] = pd.qcut(par.p_fusion, 10, labels=False) + 1
cal = (
    par.groupby("decile")
    .agg(n=("y", "size"), mean_pred=("p_fusion", "mean"), observed=("y", "mean"), mean_phq=("phq", "mean"))
    .round(4)
    .reset_index()
)
cal.to_csv(RESULTS / "calibration_deciles.csv", index=False)

# 상위 피처 유형 분류
tf = pd.read_csv(RESULTS / "top_features_phq_label.csv")
KIND = {
    "증상": ["잠", "입맛", "눈물", "집중", "무겁", "누워", "새벽", "뒤척", "힘들", "아프", "배고프"],
    "무쾌감·무망": [
        "재미",
        "귀찮",
        "의욕",
        "즐겁",
        "의미",
        "웃",
        "버티",
        "싫",
        "포기",
        "희망",
        "소용",
        "끝나",
        "탓",
        "부족",
        "망치",
    ],
    "머뭇거림": ["음", "어", "글쎄요", "그냥", "뭔가", "모르", "딱히", "잘", "ᆫ가"],
    "치료맥락": ["병원", "약", "선생", "진료", "상담", "처방", "덕분", "꾸준히"],
    "긍정회상": ["행복", "즐거웠", "좋", "웃음", "사진", "신나", "뿌듯", "편하", "괜찮", "나아지", "단단"],
}


def kind(f):
    forms = [t.split("/")[0] for t in f.split()]
    for k, ws in KIND.items():
        if any(w in x for x in forms for w in ws):
            return k
    return "기타"


tf["kind"] = tf.feature.map(kind)
tf.to_csv(RESULTS / "top_features_phq_label_kinds.csv", index=False)
print(sub[sub.model == "fusion"].to_string(index=False))
print(cal.to_string(index=False))
print(tf.head(15).to_string(index=False))
