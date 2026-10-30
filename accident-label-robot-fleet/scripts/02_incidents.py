"""알림 행을 사건으로 묶고, 앞뒤 가동과 주행 기록으로 유형 A~D를 가름

- incidents.csv: 사건 목록. 첫 알림 시각, 코드, 직전 가동, 뒤 멈춤, 직전 주행 기록, 유형
- type_share.csv: 유형별 사건 수와 비중 (기본 창)
- type_sensitivity.csv: 직전 주행 창, 뒤 멈춤 창을 바꿀 때 유형 비중
- type_by_period.csv: 코드 분리(SPLIT) 전후 유형 비중
"""

import numpy as np
import pandas as pd
from common import (
    ALERT,
    BUNDLE_MIN,
    DRIVE,
    POST_GRID,
    POST_WIN,
    PRE_DRIVE_GRID,
    PRE_DRIVE_MIN,
    PRE_RUN_MIN,
    RES,
    SPLIT,
    TYPE_ORDER,
    Timeline,
    load_events,
    load_sessions,
    minutes,
)


def bundle(alerts):
    a = alerts.sort_values(["serial", "ts"], kind="stable").copy()
    gap = a.groupby("serial")["ts"].diff().dt.total_seconds() / 60
    a["inc"] = (gap.isna() | (gap > BUNDLE_MIN)).cumsum()
    inc = a.groupby("inc").agg(
        serial=("serial", "first"),
        ts=("ts", "first"),
        code=("code", "first"),
        n_alerts=("id", "size"),
        truth_kind=("truth_kind", "first"),
    )
    return inc.reset_index(drop=True)


def classify(inc, tl, pre_drive=PRE_DRIVE_MIN, post=POST_WIN):
    t = minutes(inc["ts"]) + tl.offset(inc["serial"])
    pre_run = tl.run_overlap(t - PRE_RUN_MIN, t)
    post_run = tl.run_overlap(t + post[0], t + post[1])
    pre_drv = tl.drive_count(t - pre_drive, t) > 0
    typ = np.select([~pre_run, ~post_run, pre_drv], ["D", "A", "B"], default="C")
    return pre_run, post_run, pre_drv, typ


def share(typ):
    s = pd.Series(typ).value_counts().reindex(TYPE_ORDER, fill_value=0)
    return pd.DataFrame({"type": TYPE_ORDER, "n": s.to_numpy(), "share": (s / s.sum()).round(4).to_numpy()})


def main():
    RES.mkdir(parents=True, exist_ok=True)
    ev = load_events()
    ses = load_sessions()
    drive = ev[ev["code"].isin(DRIVE)]
    tl = Timeline(ses, drive)
    inc = bundle(ev[ev["code"].isin(ALERT)])
    inc["pre_run"], inc["post_run"], inc["pre_drive"], inc["type"] = classify(inc, tl)
    inc.insert(0, "incident_id", np.arange(1, len(inc) + 1))
    inc.to_csv(RES / "incidents.csv", index=False)

    base = share(inc["type"])
    base.to_csv(RES / "type_share.csv", index=False)

    rows = []
    for pd_min in PRE_DRIVE_GRID:
        for post in POST_GRID:
            typ = classify(inc, tl, pd_min, post)[3]
            s = share(typ).set_index("type")["share"]
            rows.append({"pre_drive_min": pd_min, "post": f"{post[0] // 60}~{post[1] // 60}시간", **s.to_dict()})
    sens = pd.DataFrame(rows)
    sens.to_csv(RES / "type_sensitivity.csv", index=False)

    per = inc.assign(period=np.where(inc["ts"] < SPLIT, "분리 전", "분리 뒤"))
    tab = pd.crosstab(per["period"], per["type"], normalize="index").reindex(columns=TYPE_ORDER).round(4)
    tab.reset_index().to_csv(RES / "type_by_period.csv", index=False)

    print(f"incidents {len(inc):,}")
    print(base.to_string(index=False))
    print(f"A 비중 범위 {sens['A'].min():.3f} ~ {sens['A'].max():.3f}")
    print(tab.to_string())


if __name__ == "__main__":
    main()
