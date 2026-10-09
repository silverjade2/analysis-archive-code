"""이벤트를 (기기, 날짜, 시간, 기능) 셀로 접기. 실제 분석의 집계 쿼리와 같은 모양.

- 기본 스케줄은 이동 모드 중 9, 12, 15, 18시에 시작한 10분 미만으로 다시 가름 (휴리스틱)
- 켜 두는 기능은 시작부터 끝까지를 시간마다 잘라 점유 초로, 시작 건수는 시작 시간에만
- 출력: 기능별 규모, 기기-일 구성, 군집 대상 전체의 기능 x 시각 사용 비율
"""

import numpy as np
import pandas as pd
from common import DATA, DAYS, FEAT, MIN_EVENTS, NAME, OCC, RES, relabel_schedule


def to_cells(e: pd.DataFrame) -> pd.DataFrame:
    rows = []
    start_h = e["start_ts"].dt.hour.to_numpy()
    sec0 = (e["start_ts"] - e["start_ts"].dt.normalize()).dt.total_seconds().to_numpy()
    dur = e["dur_sec"].to_numpy()
    for i, r in enumerate(e[["serial", "dt", "feature", "start_ts"]].itertuples(index=False)):
        if r.feature in OCC:
            s, t = sec0[i], min(sec0[i] + dur[i], 86400)
            for hr in range(int(s // 3600), int(np.ceil(t / 3600)) if t > s else int(s // 3600) + 1):
                occ = max(0.0, min(t, (hr + 1) * 3600) - max(s, hr * 3600))
                rows.append(
                    (
                        r.serial,
                        r.dt,
                        hr,
                        r.feature,
                        int(hr == start_h[i]),
                        occ,
                        r.start_ts if hr == start_h[i] else pd.NaT,
                    )
                )
        else:
            rows.append((r.serial, r.dt, start_h[i], r.feature, 1, 0.0, r.start_ts))
    c = pd.DataFrame(rows, columns=["serial", "dt", "hr", "feature", "start_cnt", "occ_sec", "first_start_at"])
    c = c.groupby(["serial", "dt", "hr", "feature"], as_index=False).agg(
        start_cnt=("start_cnt", "sum"), occ_sec=("occ_sec", "sum"), first_start_at=("first_start_at", "min")
    )
    c["occ_sec"] = c["occ_sec"].clip(upper=3600).round().astype(int)
    return c


def main():
    e = pd.read_csv(DATA / "events.csv.gz", parse_dates=["dt", "start_ts"])
    e = relabel_schedule(e)
    c = to_cells(e)
    c.to_csv(
        DATA / "cells.csv.gz", index=False, date_format="%Y-%m-%d %H:%M:%S", compression={"method": "gzip", "mtime": 0}
    )
    n_dev = c["serial"].nunique()

    # 기능별 규모
    starts = e.groupby("feature").size()
    used = c.groupby(["serial", "dt", "feature"])["occ_sec"].sum().reset_index()
    scale = pd.DataFrame(
        {
            "feature": FEAT,
            "name": [NAME[f] for f in FEAT],
            "starts": [int(starts.get(f, 0)) for f in FEAT],
            "devices": [int(c.loc[c["feature"] == f, "serial"].nunique()) for f in FEAT],
            "device_days": [int((used["feature"] == f).sum()) for f in FEAT],
            "occ_hours_per_used_day": [
                round(used.loc[used["feature"] == f, "occ_sec"].mean() / 3600, 3) if f in OCC else 0.0 for f in FEAT
            ],
        }
    )
    scale["device_share"] = (scale["devices"] / n_dev).round(3)
    scale.to_csv(RES / "feature_scale.csv", index=False)

    # 기기-일 구성: 분모 = 로그가 한 번이라도 있는 기기 x 평일 20일
    day_n = c.groupby(["serial", "dt"])["start_cnt"].sum()
    total = n_dev * len(DAYS)
    comp = pd.DataFrame(
        [
            {"group": "무사용 날", "device_days": total - len(day_n)},
            {"group": "기록 1건 날", "device_days": int((day_n < MIN_EVENTS).sum())},
            {"group": "군집 대상 날", "device_days": int((day_n >= MIN_EVENTS).sum())},
        ]
    )
    comp["share"] = (comp["device_days"] / total).round(3)
    comp.to_csv(RES / "day_composition.csv", index=False)

    # 군집 대상 전체의 기능 x 시각 사용 비율 (그 시각에 그 기능을 쓴 기기-일 비율)
    tgt = day_n[day_n >= MIN_EVENTS].index
    ct = c.set_index(["serial", "dt"]).loc[lambda x: x.index.isin(tgt)].reset_index()
    hs = ct.groupby(["feature", "hr"]).size().div(len(tgt)).rename("share").reset_index()
    hs = (
        hs.pivot(index="feature", columns="hr", values="share")
        .reindex(index=FEAT, columns=range(24))
        .fillna(0)
        .round(4)
    )
    hs.to_csv(RES / "overall_hour_share.csv")
    use = ct.groupby("feature")[["serial", "dt"]].apply(lambda g: len(g.drop_duplicates())).div(len(tgt))
    use.reindex(FEAT).fillna(0).round(3).rename("use_rate").to_csv(RES / "overall_use_rate.csv")

    # 휴리스틱 점검 재료 (평가는 07)
    print(f"cells {len(c):,} / devices {n_dev:,} / target days {len(tgt):,}")
    print(scale.to_string(index=False))
    print(comp.to_string(index=False))


if __name__ == "__main__":
    main()
