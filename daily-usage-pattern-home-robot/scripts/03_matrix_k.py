"""기기-일 행렬 (기능 9종 x 2시간 구간 12 = 108열) 과 군집 수 선택.

- 정규화 전후로 기능별 분산 몫을 비교 (켜 두는 기능이 거리를 독차지하는지)
- k 3 ~ 8 의 inertia 와 silhouette. 규칙은 silhouette 최대값과 0.03 안 중 가장 작은 k
"""

import numpy as np
import pandas as pd
from common import BLOCK_HOURS, DATA, FEAT, MIN_EVENTS, NAME, RES, day_matrix, k_sweep, pick_k


def target_keys(c: pd.DataFrame) -> pd.DataFrame:
    n = c.groupby(["serial", "dt"])["start_cnt"].sum()
    return n[n >= MIN_EVENTS].reset_index()[["serial", "dt"]].sort_values(["serial", "dt"]).reset_index(drop=True)


def variance_share(x: np.ndarray) -> np.ndarray:
    nb = 24 // BLOCK_HOURS
    v = x.var(0).reshape(len(FEAT), nb).sum(1)
    return v / v.sum()


def main():
    c = pd.read_csv(DATA / "cells.csv.gz", parse_dates=["dt"])
    keys = target_keys(c)
    keys.to_csv(DATA / "target_days.csv", index=False, date_format="%Y-%m-%d")
    xn = day_matrix(c, keys, FEAT, normalize=True)
    xr = day_matrix(c, keys, FEAT, normalize=False)

    vs = pd.DataFrame(
        {
            "feature": FEAT,
            "name": [NAME[f] for f in FEAT],
            "raw_share": variance_share(xr).round(3),
            "norm_share": variance_share(xn).round(3),
        }
    )
    vs.to_csv(RES / "feature_variance_share.csv", index=False)

    sw = k_sweep(xn)
    sw.to_csv(RES / "k_sweep.csv", index=False)
    swr = k_sweep(xr)
    swr.to_csv(RES / "k_sweep_raw.csv", index=False)
    k = pick_k(sw)
    pd.DataFrame(
        [
            {
                "matrix": "정규화",
                "k": k,
                "sil_max": sw["silhouette"].max(),
                "k_at_max": int(sw.loc[sw["silhouette"].idxmax(), "k"]),
            }
        ]
    ).to_csv(RES / "k_choice.csv", index=False)
    print(f"target days {len(keys):,}, matrix {xn.shape}")
    print(vs.to_string(index=False))
    print(sw.to_string(index=False))
    print(f"pick k = {k}")


if __name__ == "__main__":
    main()
