"""ground truth 평가. 판정(02~06)에는 truth_* 열을 쓰지 않음

- eval_type_truth.csv: 유형 x 실제 사건 종류 교차표
- eval_a_purity.csv: A 안 실제 넘어짐 비중, 실제 넘어짐 중 A로 잡힌 비중, A 신호의 기대를 넘는 몫과 비교
- eval_recurrence_truth.csv: 실제 fall + stuck만으로 그린 재발 곡선의 장기 수준 배수
- eval_consult_truth.csv: A를 실제 넘어짐과 나머지로 나눈 상담 연결률. A가 순수했다면 A ÷ C가 얼마였나
"""

from importlib import import_module

import pandas as pd
from common import DATA, END, LONG_RUN, RES, TYPE_ORDER, load_devices

recur = import_module("04_recurrence")
consult = import_module("05_consults")


def main():
    inc = pd.read_csv(RES / "incidents.csv", parse_dates=["ts"])
    inc["truth"] = inc["truth_kind"].str.split("_").str[0]
    tab = pd.crosstab(inc["type"], inc["truth"]).reindex(TYPE_ORDER, fill_value=0)
    tab.reset_index().to_csv(RES / "eval_type_truth.csv", index=False)

    a = inc[inc["type"] == "A"]
    falls = inc[inc["truth"] == "fall"]
    sig = pd.read_csv(RES / "a_signal_monthly.csv")
    pre = inc[inc["pre_run"]]
    purity = pd.DataFrame(
        [
            {
                "a_true_fall_share": round((a["truth"] == "fall").mean(), 3),
                "fall_caught_as_a": round((falls["type"] == "A").mean(), 3),
                "true_fall_share_of_pre_run": round((pre["truth"] == "fall").mean(), 3),
                "excess_share_signal": round((sig["stopped"].sum() - sig["expected"].sum()) / sig["pre_run"].sum(), 3),
            }
        ]
    )
    purity.to_csv(RES / "eval_a_purity.csv", index=False)

    dev = load_devices()
    obs_days = float((END - dev["install_date"]).dt.days.sum())
    c, base = recur.curve(inc[inc["truth"] != "handling"], obs_days)
    lr = c.loc[c["day"].between(*LONG_RUN), "rate"].mean()
    rt = pd.DataFrame([{"base_rate": round(base, 3), "long_run_rate": round(lr, 3), "ratio": round(lr / base, 3)}])
    rt.to_csv(RES / "eval_recurrence_truth.csv", index=False)

    con = pd.read_csv(DATA / "consults.csv", parse_dates=["ts"])
    inc["linked"] = consult.link(inc, con)
    c_link = inc.loc[inc["type"] == "C", "linked"].mean()
    a_fall = a["truth"] == "fall"
    a_link = inc.loc[a.index, "linked"]
    ct = pd.DataFrame(
        [
            {
                "a_fall_link": round(a_link[a_fall].mean(), 4),
                "a_other_link": round(a_link[~a_fall].mean(), 4),
                "c_link": round(c_link, 4),
                "true_fall_a_over_c": round(a_link[a_fall].mean() / c_link, 3),
            }
        ]
    )
    ct.to_csv(RES / "eval_consult_truth.csv", index=False)

    print(tab.to_string())
    print(purity.to_string(index=False))
    print(rt.to_string(index=False))
    print(ct.to_string(index=False))


if __name__ == "__main__":
    main()
