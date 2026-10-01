"""로그 전수 구성: 코드별 행 수, coverage, 기기당 행 수, 3단계 비중

- code_profile.csv: 코드 한 줄씩
- tier_share.csv: 코드 수 비중과 로그 비중이 반대 방향인지
- concentration.csv: 묶음별 상위 10% 기기 비중
"""

import numpy as np
import pandas as pd
from common import A_START, ACCIDENT, CODE_DF, DATA, DRIVE, RES, TIER_ORDER


def load():
    dev = pd.read_csv(DATA / "devices.csv", parse_dates=["install_date"])
    ev = pd.read_csv(DATA / "events.csv.gz", parse_dates=["ts"], dtype={"level": str})
    ev = ev.merge(dev[["serial", "install_date"]], on="serial")
    ev["pre"] = ev["ts"] < ev["install_date"]
    return dev, ev


def main():
    RES.mkdir(parents=True, exist_ok=True)
    dev, ev = load()
    n_dev = ev["serial"].nunique()
    tot = len(ev)
    g = ev.groupby("code")
    top1 = ev.groupby(["code", "serial"]).size().groupby("code").max()
    prof = pd.DataFrame({"rows": g.size(), "devices": g["serial"].nunique()})
    prof["share"] = (prof["rows"] / tot).round(5)
    prof["coverage"] = (prof["devices"] / n_dev).round(3)
    prof["rows_per_device"] = (prof["rows"] / prof["devices"]).round(1)
    prof["top1_device_share"] = (top1 / prof["rows"]).round(3)
    prof = prof.reindex(CODE_DF.index).fillna({"rows": 0, "devices": 0, "share": 0}).join(CODE_DF)
    prof = prof.sort_values("rows", ascending=False)
    prof.to_csv(RES / "code_profile.csv", index_label="code")

    tier = prof.groupby("tier").agg(
        codes=("rows", "size"), codes_with_logs=("rows", lambda v: int((v > 0).sum())), rows=("rows", "sum")
    )
    tier = tier.reindex(TIER_ORDER)
    tier["code_share"] = (tier["codes"] / tier["codes"].sum()).round(3)
    tier["log_share"] = (tier["rows"] / tot).round(4)
    tier.to_csv(RES / "tier_share.csv")

    cat = prof.groupby("category").agg(codes=("rows", "size"), rows=("rows", "sum"))
    cat["log_share"] = (cat["rows"] / tot).round(4)
    cat.sort_values("rows", ascending=False).to_csv(RES / "category_share.csv")

    post = ev[~ev["pre"] & (ev["ts"] >= A_START)]
    groups = {
        "drive": post["code"].isin(DRIVE),
        "accident": post["code"].isin(ACCIDENT),
        "wifi": post["code"].eq("W1"),
        "other": ~post["code"].isin(DRIVE + ACCIDENT + ["W1"]),
    }
    rows = []
    for name, m in groups.items():
        per = post.loc[m].groupby("serial").size().reindex(dev["serial"], fill_value=0).sort_values(ascending=False)
        k = int(np.ceil(len(per) * 0.1))
        rows.append(
            {"group": name, "rows": int(per.sum()), "top10pct_share": round(float(per.head(k).sum() / per.sum()), 3)}
        )
    pd.DataFrame(rows).to_csv(RES / "concentration.csv", index=False)

    drive_share = prof.loc[DRIVE, "share"].sum()
    acc_share = prof.loc[ACCIDENT, "share"].sum()
    print(f"rows {tot:,}  codes {int((prof['rows'] > 0).sum())}  devices {n_dev:,}")
    print(f"drive {drive_share:.1%}  accident codes {acc_share:.2%}  W1 top1 {prof.loc['W1', 'top1_device_share']:.3f}")
    print(tier[["codes", "code_share", "log_share"]].to_string())


if __name__ == "__main__":
    main()
