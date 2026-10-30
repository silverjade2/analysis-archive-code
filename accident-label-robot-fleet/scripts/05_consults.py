"""상담 기록 대조: 상담이 유형 A를 검증해 줄 수 있나

- consult_match.csv: 상담 에러코드가 인입 7일 전 ~ 1일 뒤 같은 기기 알림에 있는 비율, 넘어짐 서술 비율, 처리 결과 구성
- consult_link.csv: 유형별 사건 수, 7일 안 상담이 붙은 비율 (자동 상담 전후), A / C
"""

import numpy as np
import pandas as pd
from common import ALERT, AUTO_CONSULT, RES, TYPE_ORDER, load_events

A_OVER_C_MIN = 1.5  # 이 배수 이상이면 A가 실제 이상을 담는다고 봄 (결과 전 고정)


def code_match(con, alerts):
    """상담 하나마다 창 안에 같은 코드 알림이 있나"""
    m = con[["consult_id", "serial", "ts", "error_code"]].merge(
        alerts[["serial", "ts", "code"]], on="serial", suffixes=("", "_a")
    )
    lag = (m["ts"] - m["ts_a"]).dt.total_seconds() / 86400
    m = m[(lag <= 7) & (lag >= -1)]
    hit = m[m["error_code"] == m["code"]].groupby("consult_id").size()
    return con["consult_id"].isin(hit.index)


def link(inc, con):
    """사건 시작부터 7일 안 상담. 상담 하나는 그 앞 가장 가까운 사건 하나에만 붙임"""
    c = con.sort_values("ts")
    i = inc.sort_values("ts")
    j = pd.merge_asof(c, i[["incident_id", "serial", "ts"]], on="ts", by="serial", direction="backward")
    j = j.merge(i[["incident_id", "ts"]].rename(columns={"ts": "inc_ts"}), on="incident_id", how="left")
    ok = (j["ts"] - j["inc_ts"]) <= pd.Timedelta(days=7)
    return inc["incident_id"].isin(j.loc[ok, "incident_id"])


def main():
    ev = load_events()
    alerts = ev[ev["code"].isin(ALERT)]
    con = pd.read_csv(RES.parents[1] / "data" / "consults.csv", parse_dates=["ts"])
    inc = pd.read_csv(RES / "incidents.csv", parse_dates=["ts"])

    con["code_match"] = code_match(con, alerts)
    out = con["outcome"].value_counts(normalize=True).round(4)
    match = pd.DataFrame(
        [("consults", len(con)), ("code_match", round(con["code_match"].mean(), 4))]
        + [("fall_note", round(con["fall_note"].mean(), 4))]
        + [(f"outcome:{k}", v) for k, v in out.items()],
        columns=["item", "value"],
    )
    match.to_csv(RES / "consult_match.csv", index=False)

    inc["linked"] = link(inc, con)
    inc["period"] = np.where(inc["ts"] < AUTO_CONSULT, "자동 상담 전", "자동 상담 뒤")
    tab = (
        inc.groupby(["period", "type"])["linked"]
        .agg(n="size", link_rate="mean")
        .reset_index()
        .assign(link_rate=lambda d: d["link_rate"].round(4))
    )
    allp = inc.groupby("type")["linked"].agg(n="size", link_rate="mean").reset_index().assign(period="전체")
    allp["link_rate"] = allp["link_rate"].round(4)
    tab = pd.concat([allp, tab], ignore_index=True)[["period", "type", "n", "link_rate"]]
    tab["type"] = pd.Categorical(tab["type"], TYPE_ORDER)
    tab = tab.sort_values(["period", "type"], kind="stable")
    tab.to_csv(RES / "consult_link.csv", index=False)

    r = allp.set_index("type")["link_rate"]
    a_over_c = r["A"] / r["C"]
    print(match.to_string(index=False))
    print(tab.to_string(index=False))
    verdict = "A가 실제 이상을 담음" if a_over_c >= A_OVER_C_MIN else "A는 상한"
    print(f"A / C {a_over_c:.2f} (기준 {A_OVER_C_MIN}) -> {verdict}")
    pd.DataFrame([{"a_over_c": round(a_over_c, 3), "threshold": A_OVER_C_MIN}]).to_csv(
        RES / "consult_a_over_c.csv", index=False
    )


if __name__ == "__main__":
    main()
