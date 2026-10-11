"""ground truth 평가. truth_* 열은 여기서만 씀

- 기기-일 패턴이 가상데이터에 넣은 하루 상태를 얼마나 되찾나 (정규화 전후, ARI)
- 기기 유형이 가상데이터에 넣은 기기 유형과 선호 상태를 얼마나 되찾나
- 기본 스케줄 휴리스틱의 정밀도와 재현율, 기기 기준 스케줄 기기 비율의 부풀림
"""

import numpy as np
import pandas as pd
from common import DATA, FEAT, RES, SCHEDULE_HOURS, SCHEDULE_MAX_MIN, ari, day_matrix, kmeans, relabel_by_size


def main():
    dev = pd.read_csv(DATA / "devices.csv")
    e = pd.read_csv(DATA / "events.csv.gz", parse_dates=["dt", "start_ts"])
    keys = pd.read_csv(RES / "pattern_by_day.csv", parse_dates=["dt"])
    k = int(pd.read_csv(RES / "k_choice.csv")["k"].iloc[0])

    truth_day = e[e["truth_state"] != "idle"].groupby(["serial", "dt"])["truth_state"].first()
    kd = keys.join(truth_day, on=["serial", "dt"])
    cells = pd.read_csv(DATA / "cells.csv.gz", parse_dates=["dt"])
    xr = day_matrix(cells, keys[["serial", "dt"]], FEAT, normalize=False)
    raw_lab, _, _ = kmeans(xr, k, n_init=10)
    kd["raw_pattern"] = relabel_by_size(raw_lab)
    ok = kd["truth_state"].notna()
    rows = [
        {"what": "기기-일 패턴 (정규화)", "ari": round(ari(kd.loc[ok, "pattern"], kd.loc[ok, "truth_state"]), 3)},
        {
            "what": "기기-일 패턴 (정규화 전)",
            "ari": round(ari(kd.loc[ok, "raw_pattern"], kd.loc[ok, "truth_state"]), 3),
        },
    ]
    pd.crosstab(kd["pattern"], kd["truth_state"], normalize="index").round(3).to_csv(RES / "eval_pattern_x_state.csv")
    pd.crosstab(kd["raw_pattern"], kd["truth_state"], normalize="index").round(3).to_csv(
        RES / "eval_raw_pattern_x_state.csv"
    )

    # 정규화 전 패턴은 무엇으로 갈렸나: 패턴별 제자리 모드 시작 시각 중앙값
    fx = e[e["feature"] == "fixed"].groupby(["serial", "dt"])["start_ts"].min().dt.hour.rename("fixed_first_hour")
    rp = (
        kd.join(fx, on=["serial", "dt"])
        .groupby("raw_pattern")
        .agg(device_days=("serial", "size"), fixed_first_hour_median=("fixed_first_hour", "median"))
    )
    rp.to_csv(RES / "eval_raw_pattern_profile.csv")

    dt = pd.read_csv(RES / "device_type.csv").merge(dev, on="serial")
    rows += [
        {
            "what": "기기 유형 vs 기기 유형 (저활동, 활동, 공간 스캔)",
            "ari": round(ari(dt["device_type"], dt["truth_type"]), 3),
        },
        {
            "what": "기기 유형 vs 활동 기기의 선호 상태",
            "ari": round(ari(dt["device_type"], np.where(dt["truth_type"] == "low", "low", dt["truth_pref"])), 3),
        },
    ]
    pd.crosstab(
        dt["device_type"],
        np.where(dt["truth_type"] == "low", "low", dt["truth_type"] + "_" + dt["truth_pref"]),
        normalize="index",
    ).round(3).to_csv(RES / "eval_type_x_truth.csv")

    # 기본 스케줄 휴리스틱
    pat = e[e["feature"] == "patrol"].copy()
    pat["pred"] = pat["start_ts"].dt.hour.isin(SCHEDULE_HOURS) & (pat["dur_sec"] < SCHEDULE_MAX_MIN * 60)
    tp = int((pat["pred"] & (pat["truth_sched"] == 1)).sum())
    sched = pd.DataFrame(
        [
            {
                "precision": round(tp / pat["pred"].sum(), 3),
                "recall": round(tp / (pat["truth_sched"] == 1).sum(), 3),
                "false_schedule_events": int((pat["pred"] & (pat["truth_sched"] == 0)).sum()),
                "devices_truth_on": round(dev["truth_schedule_on"].mean(), 3),
                "devices_any_schedule": round(pat.loc[pat["pred"], "serial"].nunique() / dev.shape[0], 3),
                "devices_schedule_half_days": round(
                    (pat[pat["pred"]].groupby("serial")["dt"].nunique() >= 10).sum() / dev.shape[0], 3
                ),
            }
        ]
    )
    sched.to_csv(RES / "eval_schedule_heuristic.csv", index=False)

    # 일관성: 가상데이터에 넣은 값과 잰 값
    share = pd.read_csv(RES / "pattern_share.csv")
    rows.append(
        {
            "what": "가상데이터에 넣은 일관성 평균 (활동 기기)",
            "ari": round(dev.loc[dev["truth_type"] != "low", "truth_consistency"].mean(), 3),
        }
    )
    rows.append({"what": "잰 일관성 범위", "ari": f"{share['consistency'].min():.3f}~{share['consistency'].max():.3f}"})
    out = pd.DataFrame(rows)
    out.to_csv(RES / "eval_summary.csv", index=False)
    print(out.to_string(index=False))
    print(sched.to_string(index=False))
    print(rp.to_string())
    print(pd.read_csv(RES / "eval_pattern_x_state.csv").to_string(index=False))


if __name__ == "__main__":
    main()
