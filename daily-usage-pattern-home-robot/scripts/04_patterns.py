"""선택한 k 로 기기-일 패턴을 묶고, 패턴마다 기능 사용률, 시각 분포, 비중, 일관성을 정리. 순서는 05"""

import pandas as pd
from common import DATA, FEAT, OCC, RES, day_matrix, kmeans, relabel_by_size


def main():
    c = pd.read_csv(DATA / "cells.csv.gz", parse_dates=["dt", "first_start_at"])
    keys = pd.read_csv(DATA / "target_days.csv", parse_dates=["dt"])
    x = day_matrix(c, keys, FEAT, normalize=True)
    k = int(pd.read_csv(RES / "k_choice.csv")["k"].iloc[0])
    lab, _, _ = kmeans(x, k, n_init=10)
    keys["pattern"] = relabel_by_size(lab)
    keys.to_csv(RES / "pattern_by_day.csv", index=False, date_format="%Y-%m-%d")

    ck = c.merge(keys, on=["serial", "dt"])
    n_by_p = keys["pattern"].value_counts().sort_index()

    # 기능 사용률 (그날 쓴 기기-일 비율)
    used = ck.groupby(["serial", "dt", "feature"])["pattern"].first().reset_index()
    rate = used.groupby(["pattern", "feature"]).size().unstack().reindex(columns=FEAT).fillna(0).div(n_by_p, axis=0)
    rate.loc["전체"] = used.groupby("feature").size().reindex(FEAT).fillna(0) / len(keys)
    rate.round(3).to_csv(RES / "pattern_feature_use.csv")

    # 시각 분포 (그 시각에 그 기능을 쓴 기기-일 비율)
    hs = ck.groupby(["pattern", "feature", "hr"]).size().div(n_by_p, level="pattern").rename("share").reset_index()
    hs.round(4).to_csv(RES / "pattern_hour_share.csv", index=False)

    # 비중: 기기-일 기준 / 기기 기준 (가장 자주 속한 패턴) / 일관성
    main_p = keys.groupby("serial")["pattern"].agg(lambda s: s.value_counts().sort_index().idxmax())
    cons = keys.merge(main_p.rename("main"), on="serial").assign(same=lambda d: d["pattern"] == d["main"])
    cons = cons.groupby(["main", "serial"])["same"].mean().groupby("main").mean()
    share = pd.DataFrame(
        {
            "device_days": n_by_p,
            "day_share": (n_by_p / len(keys)).round(3),
            "devices": main_p.value_counts().sort_index(),
            "device_share": (main_p.value_counts().sort_index() / len(main_p)).round(3),
            "consistency": cons.round(3),
        }
    )
    share["days_per_device"] = (share["device_days"] / share["devices"]).round(1)
    share.index.name = "pattern"
    share.to_csv(RES / "pattern_share.csv")

    # 패턴별 요약 (시작 건수, 기능 수, 제자리 점유, 스케줄 있는 날)
    dd = (
        ck.groupby(["serial", "dt", "pattern"])
        .agg(starts=("start_cnt", "sum"), n_feat=("feature", "nunique"))
        .reset_index()
    )
    fx = ck[ck["feature"] == "fixed"].groupby(["serial", "dt"])["occ_sec"].sum().div(3600).rename("fixed_h")
    sc = ck[ck["feature"] == "schedule"].groupby(["serial", "dt"]).size().gt(0).rename("has_schedule")
    dd = dd.join(fx, on=["serial", "dt"]).join(sc, on=["serial", "dt"]).fillna({"fixed_h": 0.0, "has_schedule": False})
    prof = (
        dd.groupby("pattern")
        .agg(
            starts_median=("starts", "median"),
            features_mean=("n_feat", "mean"),
            fixed_hours_mean=("fixed_h", "mean"),
            schedule_day_rate=("has_schedule", "mean"),
        )
        .round(3)
    )

    prof.to_csv(RES / "pattern_profile.csv")

    print(share.to_string())
    print(rate.round(2).to_string())
    print(prof.to_string())
    assert set(OCC) <= set(FEAT)


if __name__ == "__main__":
    main()
