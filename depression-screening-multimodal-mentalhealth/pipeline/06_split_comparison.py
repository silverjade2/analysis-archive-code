"""06. 분할 단위 비교. 같은 16,000건을 (a) 녹음 단위 랜덤 5-fold, (b) 참가자 단위 5-fold로 나눠
텍스트 / 음성(GBM, k-NN) / 융합 모델을 평가한다. 시드 3개.
화자 사전확률(학습셋의 같은 참가자 라벨)로 누수 상한을 함께 둔다.
모든 지표는 녹음 단위와 참가자 단위(8건 확률 평균) 둘 다 낸다."""

import _path  # noqa: F401
import pandas as pd
from config import RESULTS, SEED
from evalutil import (
    cross_val_predict,
    fp_audio,
    fp_audio_knn,
    fp_fusion,
    fp_speaker_prior,
    fp_text,
    load_recordings,
    metrics,
    participant_level,
)

df = load_recordings()
rows = []
oofs = {}
for seed in range(3):
    for split in ["recording_random", "participant"]:
        for model, fp in [
            ("text", fp_text),
            ("audio_gbm", fp_audio),
            ("audio_knn", fp_audio_knn),
            ("fusion", fp_fusion),
            ("speaker_prior", fp_speaker_prior),
        ]:
            if model == "speaker_prior" and split == "participant":
                continue
            if model == "fusion":

                def fp_(tr, te, s, grouped=(split == "participant")):
                    return fp_fusion(tr, te, s, grouped=grouped)
            else:
                fp_ = fp
            oof = cross_val_predict(df, fp_, split=split, seed=SEED + seed)
            if seed == 0:
                oofs[(split, model)] = oof
            m_rec = metrics(df.label_depressed, oof)
            m_par = participant_level(df, oof)
            rows.append(
                dict(
                    seed=SEED + seed,
                    split=split,
                    model=model,
                    **{f"rec_{k}": v for k, v in m_rec.items()},
                    **{f"par_{k}": v for k, v in m_par.items()},
                )
            )
            print(seed, split, model, round(m_rec["auc"], 4), round(m_par["auc"], 4))
res = pd.DataFrame(rows).round(4)
res.to_csv(RESULTS / "split_comparison_by_seed.csv", index=False)
agg = (
    res.groupby(["split", "model"])
    .agg(
        rec_auc_mean=("rec_auc", "mean"),
        rec_auc_sd=("rec_auc", "std"),
        par_auc_mean=("par_auc", "mean"),
        par_auc_sd=("par_auc", "std"),
        par_f1_mean=("par_f1", "mean"),
        rec_f1_mean=("rec_f1", "mean"),
    )
    .round(4)
    .reset_index()
)
agg.to_csv(RESULTS / "split_comparison.csv", index=False)
pd.DataFrame({f"{s}__{m}": v for (s, m), v in oofs.items()}).assign(recording_id=df.recording_id.values).to_csv(
    RESULTS / "split_comparison_oof_seed0.csv", index=False
)
print(agg.to_string(index=False))
