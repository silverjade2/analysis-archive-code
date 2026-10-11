"""기기 1대 = 1행 프로파일로 묶기 (기기-일 패턴의 일관성이 높아 기기 단위가 더 자연스러운지 확인).

프로파일 13항목: 기능 9종 쓴 날 비율, 활동 날 비율 (군집 대상 날 / 20), 제자리 모드 하루 점유 시간 (20일 평균),
제자리 모드 중 18시 이후 비중, 음성 쓴 날 하루 호출 수 (log). 표준화 뒤 k-means, k 규칙은 03 과 같음
"""

import numpy as np
import pandas as pd
from common import DATA, DAYS, FEAT, NAME, RES, k_sweep, kmeans, pick_k, relabel_by_size, silhouette


def main():
    c = pd.read_csv(DATA / "cells.csv.gz", parse_dates=["dt"])
    tgt = pd.read_csv(DATA / "target_days.csv")
    day_p = pd.read_csv(RES / "pattern_by_day.csv")
    nd = len(DAYS)
    sers = sorted(c["serial"].unique())
    used = c.groupby(["serial", "feature"])["dt"].nunique().unstack().reindex(index=sers, columns=FEAT).fillna(0) / nd
    used.columns = [f"use_{f}" for f in FEAT]
    fx = c[c["feature"] == "fixed"]
    prof = used.assign(
        active_days=tgt.groupby("serial").size().reindex(sers).fillna(0) / nd,
        fixed_hours=fx.groupby("serial")["occ_sec"].sum().reindex(sers).fillna(0) / 3600 / nd,
        fixed_evening=(fx[fx["hr"] >= 18].groupby("serial")["occ_sec"].sum() / fx.groupby("serial")["occ_sec"].sum())
        .reindex(sers)
        .fillna(0),
        voice_per_day=np.log1p(
            (
                c[c["feature"] == "voice"].groupby("serial")["start_cnt"].sum()
                / c[c["feature"] == "voice"].groupby("serial")["dt"].nunique()
            )
            .reindex(sers)
            .fillna(0)
        ),
    )
    z = ((prof - prof.mean()) / prof.std(ddof=0).replace(0, 1)).to_numpy()
    sw = k_sweep(z)
    sw.to_csv(RES / "device_k_sweep.csv", index=False)
    k = pick_k(sw)
    lab, _, _ = kmeans(z, k, n_init=10)
    t = relabel_by_size(lab)
    out = prof.assign(device_type=t)
    out.index.name = "serial"
    out.round(4).to_csv(RES / "device_type.csv")

    mean = out.groupby("device_type").mean()
    mean["voice_per_day"] = np.expm1(mean["voice_per_day"])
    mean.insert(0, "devices", out["device_type"].value_counts().sort_index())
    mean.round(3).to_csv(RES / "device_type_profile.csv")
    zt = pd.DataFrame(z, index=prof.index, columns=prof.columns).assign(device_type=t).groupby("device_type").mean()
    zt.round(3).to_csv(RES / "device_type_z.csv")

    main_p = day_p.groupby("serial")["pattern"].agg(lambda s: s.value_counts().sort_index().idxmax())
    xt = pd.crosstab(out["device_type"], main_p.reindex(out.index).fillna(0).astype(int), normalize="index").round(3)
    xt.to_csv(RES / "device_type_x_pattern.csv")

    sil_cmp = pd.DataFrame(
        [
            {
                "unit": "기기-일 (기능 9종)",
                "k": int(pd.read_csv(RES / "k_choice.csv")["k"].iloc[0]),
                "silhouette": float(
                    pd.read_csv(RES / "k_sweep.csv")
                    .set_index("k")
                    .loc[int(pd.read_csv(RES / "k_choice.csv")["k"].iloc[0]), "silhouette"]
                ),
            },
            {"unit": "기기 (프로파일 13항목)", "k": k, "silhouette": round(silhouette(z, t), 3)},
        ]
    )
    sil_cmp.to_csv(RES / "silhouette_by_unit.csv", index=False)
    print(sw.to_string(index=False))
    print(mean.round(2).T.to_string())
    print(xt.to_string())
    print(sil_cmp.to_string(index=False))
    assert all(f"use_{f}" in prof.columns for f in FEAT) and len(NAME) == len(FEAT)


if __name__ == "__main__":
    main()
