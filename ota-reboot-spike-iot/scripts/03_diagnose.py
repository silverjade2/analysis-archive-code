"""스파이크 검출과 후보 (a)~(e) 진단. is_ota, truth_* 를 읽지 않는다"""

import numpy as np
import pandas as pd
from common import DATA, RES, START, robust_z

Z_CUT = 3.0
NEW_DAYS = 3  # 설치 후 3일 이내 = 신규
DORMANT_DAYS = 7  # 직전 로그와 7일 이상 간격 = 재활성

sess = pd.read_csv(DATA / "sessions.csv", parse_dates=["start", "end"])
dev = pd.read_csv(DATA / "devices.csv", parse_dates=["install_date"])
d_start = pd.read_csv(RES / "daily_startdate.csv", parse_dates=["dt"])
d_mid = pd.read_csv(RES / "daily_midnight.csv", parse_dates=["dt"])


spikes = []
for name, d in [("startdate", d_start), ("midnight", d_mid)]:
    z = robust_z(d.off_n.values)
    base = np.median(d.off_n)
    s = d.assign(
        robust_z=z.round(2),
        off_n_multiple=(d.off_n / base).round(2),
        off_ratio_drop=(np.median(d.off_ratio) - d.off_ratio).round(3),
        attribution=name,
    )
    spikes.append(s[s.robust_z >= Z_CUT])
spikes = pd.concat(spikes)[
    [
        "attribution",
        "dt",
        "weekday",
        "off_n",
        "off_n_multiple",
        "robust_z",
        "off_ratio",
        "off_ratio_drop",
        "mode_hour",
        "mode_share",
    ]
]
spikes.to_csv(RES / "spike_days.csv", index=False)
spike_dts = sorted(spikes[spikes.attribution == "startdate"].dt.unique())

# (b)(c): 시작일 귀속, 고정 세션만 있는 기기
sess["dt"] = sess.start.dt.normalize()
sess = sess[sess.start >= START]
dev_day = sess.groupby(["dt", "serial"]).agg(fixed_only=("kind", lambda k: int((k == "fixed").all()))).reset_index()
dev_day = dev_day.merge(dev[["serial", "install_date"]], on="serial")
dev_day = dev_day.sort_values(["serial", "dt"])
dev_day["prev_dt"] = dev_day.groupby("serial").dt.shift(1)
dev_day["gap"] = (dev_day.dt - dev_day.prev_dt).dt.days
dev_day["seg"] = np.select(
    [(dev_day.dt - dev_day.install_date).dt.days < NEW_DAYS, dev_day.gap >= DORMANT_DAYS],
    ["new", "reactivated"],
    default="existing",
)
off = dev_day[dev_day.fixed_only == 1]
comp = (
    off.groupby(["dt", "seg"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=["new", "reactivated", "existing"], fill_value=0)
)
comp["off_n"] = comp.sum(axis=1)
base = comp[~comp.index.isin(spike_dts)].median().rename("baseline_median")
rows = [base.to_frame().T.assign(dt="baseline_median")]
for dt in spike_dts:
    r = comp.loc[dt].to_frame().T.assign(dt=dt.date())
    rows.append(r)
decomp = pd.concat(rows)[["dt", "new", "reactivated", "existing", "off_n"]]
decomp["excess_vs_baseline"] = decomp.off_n - base["off_n"]
for c in ["new", "reactivated", "existing"]:
    decomp[f"{c}_excess"] = decomp[c] - base[c]
decomp.to_csv(RES / "spike_decomposition.csv", index=False)

wk = d_start.groupby("weekday").off_n.median().rename("off_n_median").to_frame()
wk["spike_days"] = pd.Series([d.dayofweek for d in spike_dts]).value_counts().reindex(range(7), fill_value=0).values
wk.reset_index().to_csv(RES / "weekday_profile.csv", index=False)

ff = sess[sess.kind == "fixed"].groupby(["dt", "serial"]).start.min().dt.hour.rename("h").reset_index()
ff["group"] = np.where(ff.dt.isin(spike_dts), "spike", "normal")
hp = ff.groupby(["group", "h"]).size().unstack(0, fill_value=0).reindex(range(24), fill_value=0)
hp = (hp / hp.sum()).round(4).reset_index().rename(columns={"h": "hour"})
hp.to_csv(RES / "start_hour_profile.csv", index=False)

sp_mid = spikes[spikes.attribution == "midnight"]
sp_sd = spikes[spikes.attribution == "startdate"]
normal = d_start[~d_start.dt.isin(spike_dts)]
sp = d_start[d_start.dt.isin(spike_dts)]
mid_keep = d_mid[d_mid.dt.isin(spike_dts)]
tbl = pd.DataFrame(
    [
        [
            "(a) 요일 생활 패턴",
            "스파이크일 요일 종류 / 스파이크 수",
            f"{len(set(d.dayofweek for d in spike_dts))} / {len(spike_dts)}",
            "요일별 off_n 중앙값 범위",
            f"{wk.off_n_median.min():.0f}~{wk.off_n_median.max():.0f}",
        ],
        [
            "(b) 신규 설치 배치 유입",
            "스파이크일 증가분 중 신규 기기 비중(중앙값)",
            f"{(decomp.iloc[1:].new_excess / decomp.iloc[1:].excess_vs_baseline).median():.3f}",
            "",
            "",
        ],
        [
            "(c) 휴면 기기 일괄 복귀",
            "스파이크일 증가분 중 재활성 기기 비중(중앙값)",
            f"{(decomp.iloc[1:].reactivated_excess / decomp.iloc[1:].excess_vs_baseline).median():.3f}",
            "재활성 기기수 스파이크일/평일 중앙값",
            f"{decomp.iloc[1:].reactivated.median():.0f} / {base['reactivated']:.0f}",
        ],
        [
            "(d) 자동 트리거",
            "스파이크일 최빈시 집중 비율(최소~최대)",
            f"{sp.mode_share.min():.3f}~{sp.mode_share.max():.3f}",
            "평일 최빈시 집중 비율 중앙값(최대)",
            f"{normal.mode_share.median():.3f} ({normal.mode_share.max():.3f})",
        ],
        [
            "(e) 자정 분할 이월",
            "시작일 귀속에서도 스파이크로 남는 날 수",
            f"{len(spike_dts)} / {len(spikes[spikes.attribution == 'midnight'])}",
            "스파이크일 off_n 배수 midnight vs startdate",
            f"{sp_mid.off_n_multiple.median():.2f} vs {sp_sd.off_n_multiple.median():.2f}",
        ],
    ],
    columns=["hypothesis", "metric_1", "value_1", "metric_2", "value_2"],
)
tbl.to_csv(RES / "hypothesis_table.csv", index=False)

print(spikes.to_string(index=False))
print(decomp.to_string(index=False))
print(tbl.to_string(index=False))
