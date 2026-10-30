"""A 신호 점검: 알림 뒤 멈춤이 같은 달, 같은 시각대의 평소 멈춤보다 얼마나 많은가

- stop_baseline.csv: 달 x 시각별 평소 멈춤 비율. 가동 중인 매시 정각에서 1~6시간 뒤 가동이 없던 비율
- a_signal_monthly.csv: 달별 직전 가동 사건 수, 뒤 멈춤 수, 기대 멈춤, 배수, 기대를 넘는 몫
"""

import numpy as np
import pandas as pd
from common import DRIVE, POST_WIN, RES, START, Timeline, load_events, load_sessions, minutes


def baseline_points(ses):
    """세션마다 안에 든 매시 정각. 평소 멈춤의 분모"""
    s = minutes(ses["start"])
    e = minutes(ses["end"])
    first = np.ceil(s / 60) * 60
    n = np.maximum(0, np.floor((e - 1e-9 - first) / 60).astype(int) + 1)
    idx = np.repeat(np.arange(len(ses)), n)
    step = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    t = first[idx] + step * 60
    return ses["serial"].to_numpy()[idx], t


def main():
    ev = load_events()
    ses = load_sessions()
    tl = Timeline(ses, ev[ev["code"].isin(DRIVE)])

    serial, t = baseline_points(ses)
    off = pd.Series(serial).map(tl.key).to_numpy()
    stop = ~tl.run_overlap(t + off + POST_WIN[0], t + off + POST_WIN[1])
    ts = START + pd.to_timedelta(t, unit="m")
    base = (
        pd.DataFrame({"month": ts.to_period("M").astype(str), "hour": ts.hour, "stop": stop})
        .groupby(["month", "hour"])["stop"]
        .agg(points="size", stop_rate="mean")
        .reset_index()
    )
    base["stop_rate"] = base["stop_rate"].round(4)
    base.to_csv(RES / "stop_baseline.csv", index=False)

    inc = pd.read_csv(RES / "incidents.csv", parse_dates=["ts"])
    pre = inc[inc["pre_run"]].copy()
    pre["month"] = pre["ts"].dt.to_period("M").astype(str)
    pre["hour"] = pre["ts"].dt.hour
    pre = pre.merge(base[["month", "hour", "stop_rate"]], on=["month", "hour"], how="left")
    m = (
        pre.assign(stopped=~pre["post_run"])
        .groupby("month")
        .agg(pre_run=("incident_id", "size"), stopped=("stopped", "sum"), expected=("stop_rate", "sum"))
    )
    m["ratio"] = (m["stopped"] / m["expected"]).round(3)
    m["excess_share"] = ((m["stopped"] - m["expected"]) / m["pre_run"]).round(3)
    m["expected"] = m["expected"].round(1)
    m.reset_index().to_csv(RES / "a_signal_monthly.csv", index=False)

    hours = base.groupby("hour")["stop_rate"].mean()
    print(m.to_string())
    print(f"2배 이상 달 {(m['ratio'] >= 2).sum()} / {len(m)}")
    print("시각대별 평소 멈춤 (달 평균)")
    print(hours.round(3).to_string())


if __name__ == "__main__":
    main()
