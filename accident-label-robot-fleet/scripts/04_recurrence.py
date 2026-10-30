"""재발 위험 곡선과 K 재선정. A, B 사건만 대상

- recurrence_curve.csv: 직전 사건 뒤 경과일별 위험 기기 수, 재발 수, 재발률 (1,000 기기-일당), 7일 평균
- recurrence_summary.csv: 평소 발생률, 장기 수준, K 후보 (평소 기준, 장기 수준 기준)
- k_anchors.csv: K별 예측 기준 시점 수 (A, B 사건과 전체 사건)
- funnel.csv: 알림 사건에서 K30 기준 시점까지
- repeat_by_code.csv: 코드 분리 뒤 전도, 낙하 사건의 30일 안 재발 비율 (전체 사건, A, B 사건)
"""

import numpy as np
import pandas as pd
from common import CURVE_DAYS, END, K_DAYS, K_GRID, LONG_RUN, RES, SPLIT, load_devices


def curve(inc, obs_days):
    """사건마다 다음 사건까지 날을 세고, 다음 사건이 나면 시계를 그 사건으로 넘김"""
    d = inc.sort_values(["serial", "ts"]).copy()
    d["day"] = d["ts"].dt.normalize()
    d["next"] = d.groupby("serial")["day"].shift(-1)
    gap = (d["next"] - d["day"]).dt.days  # 다음 사건까지. 같은 날이면 0
    left = (END - d["day"]).dt.days - 1  # 관측 끝까지 남은 날
    at_risk = np.zeros(CURVE_DAYS + 1)
    events = np.zeros(CURVE_DAYS + 1)
    for k in range(1, CURVE_DAYS + 1):
        alive = (left >= k) & (gap.isna() | (gap >= k))
        at_risk[k] = alive.sum()
        events[k] = (alive & (gap == k)).sum()
    c = pd.DataFrame({"day": np.arange(1, CURVE_DAYS + 1), "at_risk": at_risk[1:], "events": events[1:]})
    c["rate"] = c["events"] / c["at_risk"] * 1000
    # d부터 7일 평균 (앞으로 보는 창)
    c["rate_7d"] = c["rate"][::-1].rolling(7, min_periods=7).mean()[::-1]
    base = len(inc) / obs_days * 1000
    return c, base


def first_below(c, level, floor=7):
    hit = c[(c["day"] >= floor) & (c["rate_7d"] <= level)]
    return int(hit["day"].iloc[0]) if len(hit) else None


def anchors(inc, k):
    """직전 사건(같은 묶음)에서 K일 안의 사건은 재발로 접음. 날짜 단위, 연쇄"""
    d = inc.sort_values(["serial", "ts"])
    since = (d["ts"].dt.normalize() - d.groupby("serial")["ts"].shift().dt.normalize()).dt.days
    return len(d) if k is None else int((since.isna() | (since > k)).sum())


def repeat_by_code(inc, k=K_DAYS):
    """코드 분리 뒤 전도, 낙하 사건이 k일 안에 같은 기기에서 또 나는 비율. 낙하 센서 오탐이면 낙하가 높아야 함"""
    d = inc.sort_values(["serial", "ts"]).copy()
    nxt = d.groupby("serial")["ts"].shift(-1)
    d["repeat"] = (nxt.dt.normalize() - d["ts"].dt.normalize()).dt.days <= k
    d = d[(d["ts"] >= SPLIT) & (d["ts"] < END - pd.Timedelta(days=k)) & d["code"].isin(["TIP", "DROP"])]
    out = d.groupby("code")["repeat"].agg(n="size", repeat_rate="mean").reset_index()
    out["repeat_rate"] = out["repeat_rate"].round(3)
    return out


def main():
    dev = load_devices()
    obs_days = float((END - dev["install_date"]).dt.days.sum())
    inc = pd.read_csv(RES / "incidents.csv", parse_dates=["ts"])
    ab = inc[inc["type"].isin(["A", "B"])]

    c, base = curve(ab, obs_days)
    long_run = c.loc[c["day"].between(*LONG_RUN), "rate"].mean()
    c.round(3).to_csv(RES / "recurrence_curve.csv", index=False)
    summ = pd.DataFrame(
        [
            {
                "base_rate": round(base, 3),
                "long_run_rate": round(long_run, 3),
                "long_run_ratio": round(long_run / base, 3),
                "rate_d1_7": round(c.loc[c["day"] <= 7, "rate"].mean(), 3),
                "k_by_base": first_below(c, base),
                "k_by_long_run": first_below(c, long_run),
            }
        ]
    )
    summ.to_csv(RES / "recurrence_summary.csv", index=False)

    rows = [{"k": "없음" if k is None else k, "all": anchors(inc, k), "ab": anchors(ab, k)} for k in K_GRID]
    ka = pd.DataFrame(rows)
    ka.to_csv(RES / "k_anchors.csv", index=False)

    funnel = pd.DataFrame(
        [
            ("알림 사건", len(inc)),
            ("A, B 사건", len(ab)),
            (f"K{K_DAYS} 기준 시점", anchors(ab, K_DAYS)),
        ],
        columns=["step", "n"],
    )
    funnel["share_of_prev"] = (funnel["n"] / funnel["n"].shift()).round(3)
    funnel.to_csv(RES / "funnel.csv", index=False)

    # 전도 코드에는 사람 조작이 섞여 있어 같은 유형끼리도 봄
    rep = pd.concat(
        [repeat_by_code(inc).assign(scope="전체 알림 사건"), repeat_by_code(ab).assign(scope="A, B 사건")],
        ignore_index=True,
    )[["scope", "code", "n", "repeat_rate"]]
    rep.to_csv(RES / "repeat_by_code.csv", index=False)

    print(summ.to_string(index=False))
    print(ka.to_string(index=False))
    print(funnel.to_string(index=False))
    print(rep.to_string(index=False))


if __name__ == "__main__":
    main()
