"""채점. 정답(is_ota, truth_source)을 읽는 유일한 스크립트"""

import pandas as pd
from common import DATA, RES

sess = pd.read_csv(DATA / "sessions.csv", parse_dates=["start", "end"])
ota = pd.read_csv(DATA / "ota_days.csv", parse_dates=["ota_dt"])
batches = pd.read_csv(DATA / "install_batches.csv", parse_dates=["batch_dt"])
d_start = pd.read_csv(RES / "daily_startdate.csv", parse_dates=["dt"])
d_mid = pd.read_csv(RES / "daily_midnight.csv", parse_dates=["dt"])
spikes = pd.read_csv(RES / "spike_days.csv", parse_dates=["dt"])

rows = []
for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
    for name, d in [("startdate", d_start), ("midnight", d_mid)]:
        flag = d.mode_share >= thr
        tp = int((flag & (d.is_ota == 1)).sum())
        fp = int((flag & (d.is_ota == 0)).sum())
        fn = int((~flag & (d.is_ota == 1)).sum())
        rows.append(
            [
                thr,
                name,
                tp,
                fp,
                fn,
                round(tp / max(tp + fp, 1), 3),
                round(tp / max(tp + fn, 1), 3),
                ", ".join(d.loc[flag & (d.is_ota == 0), "dt"].dt.strftime("%m-%d")) or "-",
            ]
        )
pd.DataFrame(rows, columns=["threshold", "attribution", "tp", "fp", "fn", "precision", "recall", "fp_days"]).to_csv(
    RES / "threshold_sensitivity.csv", index=False
)

sp = spikes.merge(d_start[["dt", "is_ota"]], on="dt")
spike_vs_ota = (
    sp.groupby("attribution").agg(spike_days=("dt", "size"), spike_days_on_ota=("is_ota", "sum")).reset_index()
)
spike_vs_ota["ota_days_total"] = len(ota)
spike_vs_ota.to_csv(RES / "spike_vs_ota.csv", index=False)

rows = []
for name, d in [("startdate", d_start), ("midnight", d_mid)]:
    base = d[d.is_ota == 0]
    for r in d[d.is_ota == 1].itertuples():
        rows.append(
            [
                name,
                r.dt.date(),
                r.off_n,
                round(r.off_n / base.off_n.median(), 2),
                round(base.off_ratio.median() - r.off_ratio, 3),
                r.mode_share,
                int(r.dt in set(spikes[spikes.attribution == name].dt)),
            ]
        )
pd.DataFrame(
    rows,
    columns=["attribution", "ota_dt", "off_n", "off_n_multiple", "off_ratio_drop", "mode_share", "detected_as_spike"],
).to_csv(RES / "carryover_comparison.csv", index=False)

# OTA 후 15분 내 시작한 고정 세션. 15분은 운영 분석에서 쓴 창
rows = []
for r in ota.itertuples():
    T = r.ota_dt + pd.Timedelta(hours=int(r.ota_hour))
    w = sess[(sess.kind == "fixed") & (sess.start >= T) & (sess.start < T + pd.Timedelta(minutes=15))]
    for src, g in w.groupby("truth_source"):
        rows.append(
            [
                r.ota_dt.date(),
                src,
                len(g),
                round(len(g) / len(w), 3),
                round(g.duration_min.median(), 1),
                round((g.duration_min <= 120).mean(), 3),
            ]
        )
ps = pd.DataFrame(rows, columns=["ota_dt", "truth_source", "n", "share", "median_duration_min", "share_under_2h"])
ps.to_csv(RES / "path_share.csv", index=False)

# 설치 배치일과 다음 날
base = d_start[d_start.is_ota == 0]
dd = d_start[d_start.dt.isin(batches.batch_dt) | d_start.dt.isin(batches.batch_dt + pd.Timedelta(days=1))]
dist = dd.assign(off_n_multiple=(dd.off_n / base.off_n.median()).round(2))[
    ["dt", "off_n", "off_n_multiple", "mode_hour", "mode_share"]
]
dist.to_csv(RES / "distractor_days.csv", index=False)

ota_rows = d_start[d_start.is_ota == 1]
normal = d_start[d_start.is_ota == 0]
summary = pd.DataFrame(
    [
        ["n_days", len(d_start)],
        ["n_ota_days", len(ota)],
        ["baseline_off_n_median_startdate", normal.off_n.median()],
        ["baseline_off_n_median_midnight", d_mid[d_mid.is_ota == 0].off_n.median()],
        ["baseline_off_ratio_median", normal.off_ratio.median()],
        ["ota_off_n_multiple_min", round((ota_rows.off_n / normal.off_n.median()).min(), 2)],
        ["ota_off_n_multiple_max", round((ota_rows.off_n / normal.off_n.median()).max(), 2)],
        ["ota_off_ratio_min", ota_rows.off_ratio.min()],
        ["ota_mode_share_min", ota_rows.mode_share.min()],
        ["ota_mode_share_max", ota_rows.mode_share.max()],
        ["normal_mode_share_median", normal.mode_share.median()],
        ["normal_mode_share_max", normal.mode_share.max()],
        ["ota_mode_hour_all_equal_to_ota_hour", int((ota_rows.mode_hour == ota.ota_hour.iloc[0]).all())],
        ["path1_resume_share_median", ps[ps.truth_source == "ota_resume"].share.median()],
        ["path2_lmk_share_median", ps[ps.truth_source == "ota_lmk"].share.median()],
        ["path2_lmk_median_duration_min", ps[ps.truth_source == "ota_lmk"].median_duration_min.median()],
        ["path1_resume_median_duration_min", ps[ps.truth_source == "ota_resume"].median_duration_min.median()],
    ],
    columns=["metric", "value"],
)
summary.to_csv(RES / "summary.csv", index=False)

print(pd.read_csv(RES / "threshold_sensitivity.csv").to_string(index=False))
print(ps.to_string(index=False))
print(dist.to_string(index=False))
print(summary.to_string(index=False))
