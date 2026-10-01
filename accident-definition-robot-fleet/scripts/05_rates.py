"""노출로 나눈 사고율과 경과 구성 보정

- monthly_rate.csv: 월별 사고, 노출, 1,000대당 하루. 기본 판과 들림 포함 판
- tenure_rate.csv: 설치 후 경과 구간별
- cohort_rate.csv: 설치 월 x 경과 30일 구간
- sir.csv: 경과 구성을 맞춘 기대 대비 관측 (간접 표준화)

출시 초기 달에는 90일 넘은 기기가 없어 직접 표준화나 Kitagawa 분해는 구간이 안 겹침. 간접 표준화만 씀
"""

import numpy as np
import pandas as pd
from common import A_START, AGE_EDGES, DATA, END, K_DAYS, RES, SPLIT, X_MIN, Y_MIN, device_days_per_month

count = __import__("04_count_accidents")  # 파일명이 숫자로 시작해 import 문으로는 못 씀


def obs_window(dev):
    w = dev[["serial", "install_date"]].copy()
    w["obs_from"] = w["install_date"].clip(lower=A_START)
    w["obs_to"] = END
    return w[w["obs_from"] < w["obs_to"]].copy()


def accidents_with_lift():
    """들림 코드를 전도 트랙에 넣고 같은 규칙으로 다시 센 판 (상한)"""
    dev = pd.read_csv(DATA / "devices.csv", parse_dates=["install_date"])
    ev = pd.read_csv(DATA / "events.csv.gz", parse_dates=["ts"], dtype={"level": str})
    ev = ev[ev["code"].isin(["TIP", "DROP", "LIFT"])].merge(dev[["serial", "install_date"]], on="serial")
    ev = ev[(ev["ts"] >= ev["install_date"])].copy()
    ev["code"] = ev["code"].replace({"LIFT": "TIP"})
    ev = ev.sort_values(["serial", "ts", "id"]).reset_index(drop=True)
    ep = count.episodes(ev, X_MIN)
    return count.apply_k(count.merge_tracks(ep, Y_MIN), K_DAYS)


def age_month_exposure(w):
    """달력월 x 경과 구간 노출. 기기별 관측 구간을 경과일 축으로 옮겨 겹침을 잰다"""
    labels = [f"{a}~{b}" if b < 10000 else f"{a}+" for a, b in zip(AGE_EDGES[:-1], AGE_EDGES[1:])]
    inst = w["install_date"].to_numpy("datetime64[ns]")
    f, t = w["obs_from"].to_numpy("datetime64[ns]"), w["obs_to"].to_numpy("datetime64[ns]")
    out = {}
    for m0 in pd.date_range(A_START.to_period("M").to_timestamp(), END, freq="MS"):
        m1 = m0 + pd.offsets.MonthBegin(1)
        lo = (np.maximum(f, np.datetime64(m0)) - inst) / np.timedelta64(1, "D")
        hi = (np.minimum(t, np.datetime64(m1)) - inst) / np.timedelta64(1, "D")
        out[m0] = [
            float(np.clip(np.minimum(hi, b) - np.maximum(lo, a), 0, None).sum())
            for a, b in zip(AGE_EDGES[:-1], AGE_EDGES[1:])
        ]
    me = pd.DataFrame(out, index=labels).T
    me.index = pd.DatetimeIndex(me.index)
    return me


def age_month_count(anc, me):
    labels = list(me.columns)
    x = anc.assign(
        month=anc["s"].dt.to_period("M").dt.to_timestamp(),
        age_bin=pd.cut((anc["s"] - anc["install_date"]).dt.days, bins=AGE_EDGES, right=False, labels=labels),
    )
    n = x.pivot_table(index="month", columns="age_bin", values="serial", aggfunc="size", observed=False)
    return n.reindex(index=me.index, columns=labels).fillna(0.0)


def sir(me, nm):
    age_rate = (nm.sum() / me.sum().replace(0, np.nan)).fillna(0.0)
    expected = (me * age_rate).sum(axis=1)
    return nm.sum(axis=1) / expected.replace(0, np.nan)


def main():
    dev = pd.read_csv(DATA / "devices.csv", parse_dates=["install_date"])
    w = obs_window(dev)
    acc = pd.read_csv(RES / "accidents.csv", parse_dates=["s", "e"])
    acc = acc[acc["is_anchor"]].merge(dev[["serial", "install_date"]], on="serial")
    lift = accidents_with_lift()
    lift = lift[lift["is_anchor"]].merge(dev[["serial", "install_date"]], on="serial")

    expo = device_days_per_month(w)
    mon = pd.DataFrame(
        {
            "accidents": acc.set_index("s").resample("MS").size().reindex(expo.index, fill_value=0),
            "device_days": expo.round(0),
        }
    )
    mon["rate"] = (mon["accidents"] / expo * 1000).round(3)
    mon["accidents_lift"] = lift.set_index("s").resample("MS").size().reindex(expo.index, fill_value=0)
    mon["rate_lift"] = (mon["accidents_lift"] / expo * 1000).round(3)
    mon["partial"] = expo < expo[expo > 0].mean() * 0.5  # 관측 창 양 끝의 부분 달
    mon.to_csv(RES / "monthly_rate.csv", index_label="month")

    me = age_month_exposure(w)
    nm = age_month_count(acc, me)
    ten = pd.DataFrame({"accidents": nm.sum().astype(int), "device_days": me.sum().round(0)})
    ten["rate"] = (ten["accidents"] / ten["device_days"] * 1000).round(3)
    ten.to_csv(RES / "tenure_rate.csv", index_label="age_days")
    overall = float(ten["accidents"].sum() / ten["device_days"].sum() * 1000)

    s = pd.DataFrame(
        {"sir_basic": sir(me, nm).round(3), "sir_lift_included": sir(me, age_month_count(lift, me)).round(3)}
    )
    s["rate_basic"] = mon["rate"]
    s["expected_rate_basic"] = (mon["rate"] / s["sir_basic"]).round(3)  # 경과 구성만으로 기대되는 사고율
    s["partial"] = mon["partial"]
    s.to_csv(RES / "sir.csv", index_label="month")

    # 설치 월 코호트 x 경과 30일 구간. 코드 분리 전 코호트끼리 첫 30일을 비교하는 근거
    edges = np.arange(0, 301, 30)
    w2 = w.copy()
    w2["cohort"] = w2["install_date"].dt.to_period("M").astype(str)
    lo = (w2["obs_from"] - w2["install_date"]).dt.days.to_numpy()[:, None]
    hi = (w2["obs_to"] - w2["install_date"]).dt.days.to_numpy()[:, None]
    ov = np.clip(np.minimum(hi, edges[1:][None, :]) - np.maximum(lo, edges[:-1][None, :]), 0, None)
    lab = [f"m{i}" for i in range(len(edges) - 1)]
    ce = pd.DataFrame(ov, columns=lab).groupby(w2["cohort"].to_numpy()).sum()
    a2 = acc.assign(
        cohort=acc["install_date"].dt.to_period("M").astype(str), age=(acc["s"] - acc["install_date"]).dt.days
    )
    a2["bin"] = pd.cut(a2["age"], bins=edges, right=False, labels=lab)
    cc = (
        a2.pivot_table(index="cohort", columns="bin", values="serial", aggfunc="size", observed=False)
        .reindex(index=ce.index, columns=lab)
        .fillna(0)
    )
    cr = (cc / ce.replace(0, np.nan) * 1000).where(ce >= 300).round(2)
    cr["devices"] = w2.groupby("cohort").size()
    cr["first30_after_split"] = [pd.Period(c, "M").end_time + pd.Timedelta(days=30) > SPLIT for c in cr.index]
    cr.to_csv(RES / "cohort_rate.csv", index_label="cohort")

    print(f"accidents {len(acc):,}  exposure {expo.sum():,.0f}  overall {overall:.3f} per 1,000 device-days")
    print(mon[["accidents", "rate", "rate_lift"]].to_string())
    print(ten.to_string())
    print(s.to_string())


if __name__ == "__main__":
    main()
