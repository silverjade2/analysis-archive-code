"""하루 사용 순서. 기능을 새로 시작할 때마다 사용 1회 (다른 기능이 켜져 있어도), 계속 켜져 있던 같은 기능만 이어 봄.

- 하루 사용 횟수 분포, 패턴별 횟수와 기능 종류 수, 앞 4개로 자르면 잘리는 날
- 패턴별 하루 순서 상위 (같은 기능 연속은 ×n 으로 줄여 표시, 자르지 않음), 다음에 쓰는 기능 (전이) 상위
- 같은 기기의 날끼리 순서가 얼마나 닮았나 (전이 집합 Jaccard, 다른 기기와 비교)
- 기기마다 쓰는 시각: 기능별 첫 사용 시각의 기기 안 흩어짐과 기기 사이 흩어짐 (주 패턴 날만)
"""

import numpy as np
import pandas as pd
from common import CONT_GAP_SEC, DATA, NAME, RES, SEED, relabel_schedule

TRUNC = 4  # 실제 분석의 순서표가 본 앞 개수
MIN_DAYS = 5  # 기기 안 비교에 쓰는 기기의 최소 대상 날


def uses(e: pd.DataFrame) -> pd.DataFrame:
    """이벤트를 사용 1회로 묶기. 같은 기능이 앞 사용이 끝난 뒤 CONT_GAP_SEC 안에 다시 시작되면 이어 켜진 것"""
    e = e.sort_values(["serial", "dt", "feature", "start_ts"], kind="stable").copy()
    e["end_ts"] = e["start_ts"] + pd.to_timedelta(e["dur_sec"], unit="s")
    run_end = e.groupby(["serial", "dt", "feature"])["end_ts"].transform(lambda s: s.cummax().shift())
    e["new_use"] = run_end.isna() | (e["start_ts"] > run_end + pd.Timedelta(seconds=CONT_GAP_SEC))
    u = e[e["new_use"]].sort_values(["serial", "dt", "start_ts", "feature"], kind="stable")
    return u[["serial", "dt", "start_ts", "feature"]].reset_index(drop=True)


def compress(seq: list) -> str:
    out = []
    for f in seq:
        if out and out[-1][0] == f:
            out[-1][1] += 1
        else:
            out.append([f, 1])
    return " -> ".join(NAME[f] + (f" ×{n}" if n > 1 else "") for f, n in out)


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


def main():
    e = relabel_schedule(pd.read_csv(DATA / "events.csv.gz", parse_dates=["dt", "start_ts"]))
    keys = pd.read_csv(RES / "pattern_by_day.csv", parse_dates=["dt"])
    u = uses(e).merge(keys, on=["serial", "dt"])

    day = (
        u.groupby(["serial", "dt", "pattern"])["feature"]
        .agg(list)
        .rename("seq")
        .reset_index()
        .assign(n_uses=lambda d: d["seq"].str.len(), n_kinds=lambda d: d["seq"].map(lambda s: len(set(s))))
    )
    day["seq_text"] = day["seq"].map(compress)
    day.drop(columns="seq").to_csv(
        DATA / "day_sequences.csv.gz", index=False, date_format="%Y-%m-%d", compression={"method": "gzip", "mtime": 0}
    )

    # 하루 사용 횟수 분포 (10 이상은 한 칸)
    day["bin"] = day["n_uses"].clip(upper=10)
    steps = pd.crosstab(day["bin"], day["pattern"], normalize="columns")
    steps["전체"] = day["bin"].value_counts(normalize=True).sort_index()
    steps.index.name = "n_uses"
    steps.round(3).to_csv(RES / "seq_steps.csv")

    prof = day.groupby("pattern").agg(
        device_days=("n_uses", "size"),
        uses_median=("n_uses", "median"),
        uses_mean=("n_uses", "mean"),
        over5_share=("n_uses", lambda s: (s > 5).mean()),
        cut_by_trunc_share=("n_uses", lambda s: (s > TRUNC).mean()),
        kinds_mean=("n_kinds", "mean"),
    )
    prof.loc["전체"] = [
        len(day),
        day["n_uses"].median(),
        day["n_uses"].mean(),
        (day["n_uses"] > 5).mean(),
        (day["n_uses"] > TRUNC).mean(),
        day["n_kinds"].mean(),
    ]
    prof.round(3).to_csv(RES / "seq_profile.csv")

    # 패턴별 하루 순서 상위 5 (자르지 않음)
    top = (
        day.groupby(["pattern", "seq_text"])
        .size()
        .rename("device_days")
        .reset_index()
        .sort_values(["pattern", "device_days", "seq_text"], ascending=[True, False, True])
        .groupby("pattern")
        .head(5)
    )
    top["share"] = (top["device_days"] / top["pattern"].map(day["pattern"].value_counts())).round(3)
    top.to_csv(RES / "seq_top.csv", index=False)

    # 전이: 사용 1회 바로 다음에 온 사용 1회
    tr = day[["pattern", "seq"]].copy()
    tr["pair"] = tr["seq"].map(lambda s: list(zip(s[:-1], s[1:])))
    tr = tr.explode("pair").dropna(subset=["pair"])
    tr["from"] = tr["pair"].str[0].map(NAME)
    tr["to"] = tr["pair"].str[1].map(NAME)
    trans = (
        tr.groupby(["pattern", "from", "to"])
        .size()
        .rename("n")
        .reset_index()
        .assign(share=lambda d: d["n"] / d.groupby("pattern")["n"].transform("sum"))
        .sort_values(["pattern", "n", "from", "to"], ascending=[True, False, True, True])
        .groupby("pattern")
        .head(6)
    )
    trans.round(3).to_csv(RES / "seq_transitions.csv", index=False)

    # 같은 기기의 날끼리 vs 다른 기기의 날끼리: 전이 집합 Jaccard
    rng = np.random.default_rng(SEED)
    day["pairs"] = day["seq"].map(lambda s: set(zip(s[:-1], s[1:])) | {(s[0], "start")})
    main_p = day.groupby("serial")["pattern"].agg(lambda s: s.value_counts().sort_index().idxmax())
    rows = []
    for p in sorted(day["pattern"].unique()):
        devs = [s for s, g in day.groupby("serial") if main_p[s] == p and len(g) >= MIN_DAYS]
        within, between = [], []
        by_dev = {s: day.loc[day["serial"] == s, "pairs"].tolist() for s in devs}
        for s in devs:
            d = by_dev[s]
            i, j = rng.choice(len(d), 2, replace=False)
            within.append(jaccard(d[i], d[j]))
            other = devs[rng.integers(len(devs))]
            while other == s:
                other = devs[rng.integers(len(devs))]
            between.append(jaccard(d[i], by_dev[other][rng.integers(len(by_dev[other]))]))
        seq_mode = (
            day[day["serial"].isin(devs)].groupby("serial")["seq_text"].agg(lambda s: s.value_counts().iloc[0] / len(s))
        )
        rows.append(
            {
                "pattern": p,
                "devices": len(devs),
                "within_device": round(float(np.mean(within)), 3),
                "between_device": round(float(np.mean(between)), 3),
                "same_full_seq_share": round(float(seq_mode.mean()), 3),
            }
        )
    pd.DataFrame(rows).to_csv(RES / "seq_similarity.csv", index=False)

    # 기기마다 쓰는 시각: 기능별 하루 첫 사용 시각 (시). 기기의 주 패턴 날끼리만 비교
    first = u.groupby(["serial", "dt", "pattern", "feature"])["start_ts"].min()
    first = (first.dt.hour + first.dt.minute / 60).rename("hour").reset_index()
    first = first[first["pattern"] == first["serial"].map(main_p)]
    hrs = []
    for f in ["fixed", "patrol", "voice", "autocall"]:
        g = first[first["feature"] == f]
        n = g.groupby("serial")["hour"].size()
        g = g[g["serial"].isin(n[n >= MIN_DAYS].index)]
        med = g.groupby("serial")["hour"].median()
        hrs.append(
            {
                "feature": f,
                "name": NAME[f],
                "devices": int(len(med)),
                "within_sd_h": round(float(g.groupby("serial")["hour"].std().mean()), 2),
                "between_sd_h": round(float(med.std()), 2),
                "median_p10_h": round(float(med.quantile(0.1)), 1),
                "median_p90_h": round(float(med.quantile(0.9)), 1),
            }
        )
    pd.DataFrame(hrs).to_csv(RES / "personal_hours.csv", index=False)
    v = first[first["feature"] == "voice"]
    n = v.groupby("serial").size()
    vmed = v[v["serial"].isin(n[n >= MIN_DAYS].index)].groupby("serial")["hour"].median()
    vh = (vmed // 2 * 2).astype(int).value_counts().reindex(range(0, 24, 2), fill_value=0).rename("devices")
    vh.index.name = "voice_median_block"
    vh.to_csv(RES / "personal_voice_hour.csv")

    print(prof.round(2).to_string())
    print(steps.round(3).to_string())
    print(top.to_string(index=False))
    print(trans.round(3).to_string(index=False))
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(hrs).to_string(index=False))


if __name__ == "__main__":
    main()
