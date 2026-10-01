"""로그 행에서 사고 건수로: 연속 기록 묶기(X), 전도-낙하 병합(Y), 재발 접기(K)

- gap_bins.csv: 같은 기기 전도 로그의 직전 로그와의 간격 분포 (X 근거)
- x_sensitivity.csv: X를 바꿀 때 사건 수
- k_sensitivity.csv: K별 사고 건수, 기기 쏠림
- joint_vs_separate.csv: 전도만, 낙하만, 함께 센 값
- accidents.csv: 사건 목록. is_anchor = K14 기준 사고
"""

import numpy as np
import pandas as pd
from common import A_START, ACCIDENT, DATA, K_DAYS, K_GRID, RES, START, X_GRID, X_MIN, Y_MIN

BINS = [
    (0, "0초", 0, 0),
    (1, "0~1분", 0, 1),
    (2, "1~2분", 1, 2),
    (3, "2~5분", 2, 5),
    (4, "5~10분", 5, 10),
    (5, "10~20분", 10, 20),
    (6, "20~30분", 20, 30),
    (7, "30분~1시간", 30, 60),
    (8, "1~2시간", 60, 120),
    (9, "2~3시간", 120, 180),
    (10, "3~6시간", 180, 360),
    (11, "6~12시간", 360, 720),
    (12, "12~24시간", 720, 1440),
    (13, "1~2일", 1440, 2880),
    (14, "2~3일", 2880, 4320),
    (15, "3~7일", 4320, 10080),
    (16, "7~14일", 10080, 20160),
    (17, "14~30일", 20160, 43200),
    (18, "30~60일", 43200, 86400),
    (19, "60일 초과", 86400, np.nan),
]


def load_accident_rows():
    dev = pd.read_csv(DATA / "devices.csv", parse_dates=["install_date"])
    ev = pd.read_csv(DATA / "events.csv.gz", parse_dates=["ts"], dtype={"level": str})
    ev = ev[ev["code"].isin(ACCIDENT)].merge(dev[["serial", "install_date"]], on="serial")
    ev = ev[(ev["ts"] >= ev["install_date"]) & (ev["ts"] >= START)]
    return ev.sort_values(["serial", "ts", "id"]).reset_index(drop=True)


def episodes(df, x_by_code):
    """같은 기기, 같은 코드에서 직전 로그와 X분 안이면 같은 사건"""
    out = []
    for (ser, code), g in df.groupby(["serial", "code"], sort=False):
        start = end = None
        n = 0
        kinds = set()
        for t, k in zip(g["ts"], g["truth_kind"]):
            if start is None or (t - end).total_seconds() > x_by_code[code] * 60:
                if start is not None:
                    out.append((ser, code, start, end, n, "|".join(sorted(kinds))))
                start, end, n, kinds = t, t, 1, {k}
            else:
                end, n = t, n + 1
                kinds.add(k)
        out.append((ser, code, start, end, n, "|".join(sorted(kinds))))
    return pd.DataFrame(out, columns=["serial", "code", "s", "e", "rows", "truth_kinds"])


def merge_tracks(ep, y_min):
    """전도 사건과 낙하 사건이 Y분 안에 이어지면 한 사건"""
    out = []
    for ser, g in ep.sort_values(["serial", "s", "e"]).groupby("serial", sort=False):
        cur = None
        for r in g.itertuples(index=False):
            if cur is None or (r.s - cur["e"]).total_seconds() > y_min * 60:
                if cur is not None:
                    out.append(cur)
                cur = {"serial": ser, "code": r.code, "s": r.s, "e": r.e, "rows": r.rows, "truth_kinds": r.truth_kinds}
            else:
                cur["e"] = max(cur["e"], r.e)
                cur["rows"] += r.rows
                cur["truth_kinds"] = "|".join(
                    sorted(set(cur["truth_kinds"].split("|")) | set(r.truth_kinds.split("|")))
                )
        out.append(cur)
    return pd.DataFrame(out)


def apply_k(ep2, k_days):
    """직전 사건 종료일부터 K일 안의 사건은 재발. 날짜 단위, 연쇄"""
    ep2 = ep2.sort_values(["serial", "s"]).copy()
    ep2["days_since_prev"] = (ep2["s"].dt.normalize() - ep2.groupby("serial")["e"].shift().dt.normalize()).dt.days
    ep2 = ep2[ep2["s"] >= A_START].copy()
    if k_days is None:
        ep2["is_anchor"] = True
    else:
        ep2["is_anchor"] = ep2["days_since_prev"].isna() | (ep2["days_since_prev"] > k_days)
    return ep2


def gap_table(df):
    tip = df[df["code"] == "TIP"].copy()
    tip["gap_min"] = tip.groupby("serial")["ts"].diff().dt.total_seconds() / 60
    g = tip.dropna(subset=["gap_min"])
    edges = np.array([b[3] for b in BINS[1:19]])
    b = np.where(g["gap_min"] == 0, 0, np.searchsorted(edges, g["gap_min"], side="left") + 1)
    cnt = pd.Series(b).value_counts().reindex(range(len(BINS)), fill_value=0)
    out = pd.DataFrame(BINS, columns=["bin_no", "bin_label", "lo_min", "hi_min"])
    out["gaps"] = cnt.to_numpy()
    out["share"] = (out["gaps"] / out["gaps"].sum()).round(4)
    mid = out.index[1:19]
    width = np.log10(out.loc[mid, "hi_min"] / out.loc[mid, "lo_min"].clip(lower=1 / 60))
    out["density"] = (out.loc[mid, "share"] / width).round(4)  # 좁은 구간이 꺼져 보이지 않게 log 폭으로 나눔
    out["cum_share"] = out["share"].cumsum().round(4)
    return out


def main():
    RES.mkdir(parents=True, exist_ok=True)
    df = load_accident_rows()
    gap_table(df).to_csv(RES / "gap_bins.csv", index=False)

    rows = []
    for x in X_GRID:
        ep = episodes(df, {"TIP": x, "DROP": X_MIN["DROP"]})
        rows.append({"x_min": x, "tip_episodes": int((ep["code"] == "TIP").sum())})
    xs = pd.DataFrame(rows)
    base = xs.loc[xs["x_min"] == X_MIN["TIP"], "tip_episodes"].iloc[0]
    xs["vs_default"] = (xs["tip_episodes"] / base - 1).round(3)
    xs.to_csv(RES / "x_sensitivity.csv", index=False)

    ep = episodes(df, X_MIN)
    tip_only, drop_only = ep[ep["code"] == "TIP"], ep[ep["code"] == "DROP"]
    joint = merge_tracks(ep, Y_MIN)
    rows, jrows = [], []
    for k in K_GRID:
        a = apply_k(joint, k)
        anc = a[a["is_anchor"]]
        per = anc.groupby("serial").size().sort_values(ascending=False)
        top = int(np.ceil(len(per) * 0.1))
        rows.append(
            {
                "k_days": "none" if k is None else k,
                "episodes": len(a),
                "accidents": int(len(anc)),
                "devices": int(len(per)),
                "per_device": round(float(per.mean()), 2),
                "max_per_device": int(per.max()),
                "top10pct_share": round(float(per.head(top).sum() / len(anc)), 3),
            }
        )
        sep_t = int(apply_k(tip_only, k)["is_anchor"].sum())
        sep_d = int(apply_k(drop_only, k)["is_anchor"].sum())
        jrows.append(
            {
                "k_days": "none" if k is None else k,
                "tip_separate": sep_t,
                "drop_separate": sep_d,
                "joint": int(len(anc)),
                "folded": sep_t + sep_d - int(len(anc)),
            }
        )
    pd.DataFrame(rows).to_csv(RES / "k_sensitivity.csv", index=False)
    pd.DataFrame(jrows).to_csv(RES / "joint_vs_separate.csv", index=False)

    acc = apply_k(joint, K_DAYS)
    acc["track"] = np.where(acc["code"] == "TIP", "tip", "drop")
    acc.to_csv(RES / "accidents.csv", index=False)
    n_rows = len(df[df["ts"] >= A_START])
    ev_all = pd.read_csv(DATA / "events.csv.gz", parse_dates=["ts"], dtype={"level": str})
    n_all = int((ev_all["code"].isin(ACCIDENT) & (ev_all["ts"] >= A_START)).sum())
    funnel = [
        ("사고 코드 로그", n_all),
        ("설치 후", n_rows),
        ("사건 (X, Y)", len(acc)),
        (f"사고 (K{K_DAYS})", int(acc["is_anchor"].sum())),
    ]
    pd.DataFrame(funnel, columns=["step", "count"]).to_csv(RES / "funnel.csv", index=False)
    print(f"rows {n_rows:,} -> episodes {len(acc):,} -> accidents(K{K_DAYS}) {int(acc['is_anchor'].sum()):,}")
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(jrows).to_string(index=False))


if __name__ == "__main__":
    main()
