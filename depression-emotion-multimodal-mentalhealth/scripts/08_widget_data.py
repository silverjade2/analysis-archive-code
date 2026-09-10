"""08. 글의 인터랙티브 위젯(학습 N 탐색기)용 JSON. 05의 N 곡선 CSV 두 개를 합칠 뿐 숫자를 다시 계산하지 않는다.
출력: outputs/results/multimodal_n_curve.json → 사이트 저장소 src/data/multimodal-n-curve.json으로 복사해 정적 임베드."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, N_DAILY, N_MULTIMODAL, TEST_SIZE

curve = pd.read_csv(RESULTS / "multimodal_n_curve.csv")
gain = pd.read_csv(RESULTS / "multimodal_n_curve_gain.csv")
points = []
for (n, split), g in curve.groupby(["n", "split"], sort=True):
    row = {m: g[g.model == m].iloc[0] for m in ("text", "multimodal")}
    gg = gain[(gain.n == n) & (gain.split == split)].iloc[0]
    assert np.isclose(gg.gain, round(gg.multimodal - gg.text, 4), atol=1e-4), (n, split)
    assert np.isclose(gg.text, row["text"]["mean"]) and np.isclose(gg.multimodal, row["multimodal"]["mean"]), (n, split)
    points.append(dict(n=int(n), split=split,
                       text=dict(mean=float(row["text"]["mean"]), std=float(row["text"]["std"])),
                       multimodal=dict(mean=float(row["multimodal"]["mean"]), std=float(row["multimodal"]["std"])),
                       gain=float(gg.gain)))
out = dict(meta=dict(seeds=3, nTotal=N_DAILY, nOriginal=N_MULTIMODAL, testSize=TEST_SIZE,
                     model="char n-gram TF-IDF + LR / 음성 표준화 + LR / late fusion",
                     ns=sorted({p["n"] for p in points})),
           points=points)
(RESULTS / "multimodal_n_curve.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(f"{len(points)} points → multimodal_n_curve.json")
