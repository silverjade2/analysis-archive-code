"""사고 정의 두 가지를 나란히: 관제 등급 1 기준 vs 코드 의미 기준

- level_timeline.csv: 전도, 낙하 로그의 월별 등급 구성
- definition_compare.csv: 월별 건수. 등급 기준은 정책이 바뀐 달에 끊긴다
"""

import pandas as pd
from common import A_START, ACCIDENT, DATA, RES


def main():
    dev = pd.read_csv(DATA / "devices.csv", parse_dates=["install_date"])
    ev = pd.read_csv(DATA / "events.csv.gz", parse_dates=["ts"], dtype={"level": str})
    ev = ev.merge(dev[["serial", "install_date"]], on="serial")
    ev = ev[(ev["ts"] >= ev["install_date"]) & (ev["ts"] >= A_START)].copy()
    ev["month"] = ev["ts"].dt.to_period("M").dt.to_timestamp()

    acc = ev[ev["code"].isin(ACCIDENT)]
    lv = acc.pivot_table(index="month", columns="level", values="id", aggfunc="size").fillna(0).astype(int)
    lv.columns = [f"level_{c}" for c in lv.columns]
    lv["rows"] = lv.sum(axis=1)
    for c in [c for c in lv.columns if c.startswith("level_")]:
        lv[c + "_share"] = (lv[c] / lv["rows"]).round(3)
    lv.to_csv(RES / "level_timeline.csv", index_label="month")

    # 정의 1: 관제 등급 1 (배터리 경고 제외) / 정의 2: 전도, 낙하 코드
    lvl1 = ev[(ev["level"] == "1") & ~ev["code"].isin(["B2"])]
    cmp = (
        pd.DataFrame(
            {
                "by_level1_rows": lvl1.groupby("month").size(),
                "by_level1_all_rows": ev[ev["level"] == "1"].groupby("month").size(),
                "by_code_rows": acc.groupby("month").size(),
            }
        )
        .fillna(0)
        .astype(int)
    )
    cmp.to_csv(RES / "definition_compare.csv", index_label="month")
    print(lv[[c for c in lv.columns if c.endswith("_share")]].to_string())
    print(cmp.to_string())


if __name__ == "__main__":
    main()
