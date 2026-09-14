"""일별 집계, 시작일 귀속과 자정 분할 두 벌

- is_ota 는 채점용. 03은 이 열을 안 읽는다
"""

import pandas as pd
from common import DATA, RES, START

RES.mkdir(parents=True, exist_ok=True)

sess = pd.read_csv(DATA / "sessions.csv", parse_dates=["start", "end"])
ota = pd.read_csv(DATA / "ota_days.csv", parse_dates=["ota_dt"])
sess["is_fixed"] = (sess.kind == "fixed").astype(int)
sess["start_dt"] = sess.start.dt.normalize()

# 첫 시작 시각은 두 방식 모두 그날 시작한 세션 기준
first_fixed = (
    sess[sess.is_fixed == 1]
    .groupby(["start_dt", "serial"])
    .start.min()
    .dt.hour.rename("first_fixed_hour")
    .reset_index()
    .rename(columns={"start_dt": "dt"})
)


def build(dev_day: pd.DataFrame, name: str):
    daily = dev_day.groupby("dt").agg(active_n=("serial", "size"), off_n=("fixed_only", "sum")).reset_index()
    daily["off_ratio"] = (1 - daily.off_n / daily.active_n).round(3)
    ff = first_fixed[first_fixed.dt.isin(daily.dt)]
    fixed_n = ff.groupby("dt").size().rename("fixed_n")
    hc = ff.groupby(["dt", "first_fixed_hour"]).size().rename("n").reset_index()
    mode = hc.sort_values(["dt", "n"], ascending=[True, False]).drop_duplicates("dt")
    mode = mode.merge(fixed_n, on="dt")
    mode["mode_share"] = (mode.n / mode.fixed_n).round(3)
    daily = daily.merge(
        mode[["dt", "first_fixed_hour", "mode_share", "fixed_n"]].rename(columns={"first_fixed_hour": "mode_hour"}),
        on="dt",
        how="left",
    )
    daily["weekday"] = daily.dt.dt.dayofweek  # 0=Mon
    daily["is_ota"] = daily.dt.isin(ota.ota_dt).astype(int)
    daily = daily[["dt", "weekday", "active_n", "off_n", "off_ratio", "fixed_n", "mode_hour", "mode_share", "is_ota"]]
    daily.to_csv(RES / f"daily_{name}.csv", index=False)
    return daily


dd_start = (
    sess[sess.start >= START]
    .groupby(["start_dt", "serial"])
    .is_fixed.min()
    .rename("fixed_only")
    .reset_index()
    .rename(columns={"start_dt": "dt"})
)
d1 = build(dd_start, "startdate")

# 자정 분할: 세션이 걸친 날마다 한 행
span = sess[["serial", "is_fixed", "start", "end"]].copy()
span["d0"] = span.start.dt.normalize().clip(lower=START)
span["d1"] = (span.end - pd.Timedelta(seconds=1)).dt.normalize()
span["n_days"] = ((span.d1 - span.d0).dt.days + 1).clip(lower=1)
span = span.loc[span.index.repeat(span.n_days)]
span["dt"] = span.d0 + pd.to_timedelta(span.groupby(level=0).cumcount(), unit="D")
span = span[span.dt <= sess.start.max().normalize()]
dd_mid = span.groupby(["dt", "serial"]).is_fixed.min().rename("fixed_only").reset_index()
d2 = build(dd_mid, "midnight")

long = sess[(sess.is_fixed == 1) & (sess.duration_min >= 1440)]
pd.DataFrame(
    {
        "metric": ["devices_with_24h_fixed_session", "sessions_24h_plus", "share_of_fixed_sessions_24h_plus"],
        "value": [long.serial.nunique(), len(long), round(len(long) / (sess.is_fixed == 1).sum(), 4)],
    }
).to_csv(RES / "long_sessions.csv", index=False)

print(d1.describe().loc[["50%", "max"], ["active_n", "off_n", "off_ratio", "mode_share"]])
print(d2.describe().loc[["50%", "max"], ["active_n", "off_n", "off_ratio", "mode_share"]])
