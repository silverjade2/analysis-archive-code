"""위젯용 JSON. split_comparison.csv를 모델 × 분할로 펼친다."""

import json

import _path  # noqa: F401
import pandas as pd
from config import RESULTS

sc = pd.read_csv(RESULTS / "split_comparison.csv")
points = [
    dict(
        model=r.model,
        split=r.split,
        rec=dict(mean=float(r.rec_auc_mean), sd=float(r.rec_auc_sd)),
        par=dict(mean=float(r.par_auc_mean), sd=float(r.par_auc_sd)),
    )
    for r in sc.itertuples(index=False)
]
out = dict(meta=dict(seeds=3, nParticipants=2000, nRecordings=16000, folds=5), points=points)
(RESULTS / "split_comparison.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(f"{len(points)} points -> split_comparison.json")
