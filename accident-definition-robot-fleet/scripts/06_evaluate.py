"""ground truth로 집계 규칙과 하락 해석을 평가. truth_* 는 여기서만 읽음

- eval_counting.csv: K별로 센 사고 중 실제 사고, 재발, 들림이 몇 건인가
- eval_contamination.csv: 전도 코드 행에 섞인 들림, 검사 로그 비중 (배포 전후)
- eval_truth_rate.csv: 실제 사고의 경과 구간별 발생률, 전반기 vs 후반기
- eval_truth_sir.csv: 실제 사고로 같은 월별 SIR
- eval_truth_mix_share.csv: 실제 사고로 같은 구성 효과 몫
"""

import numpy as np
import pandas as pd
from common import A_START, DATA, END, K_GRID, RES, SPLIT, X_MIN, Y_MIN, device_days_per_month

count = __import__("04_count_accidents")
rates = __import__("05_rates")


def main():
    dev = pd.read_csv(DATA / "devices.csv", parse_dates=["install_date"])
    ev = pd.read_csv(DATA / "events.csv.gz", parse_dates=["ts"], dtype={"level": str})
    ev = ev.merge(dev[["serial", "install_date"]], on="serial")

    tip = ev[ev["code"] == "TIP"].copy()
    tip["period"] = np.where(tip["ts"] < SPLIT, "before_split", "after_split")
    tip["post_install"] = tip["ts"] >= tip["install_date"]
    cont = tip.groupby(["period", "post_install", "truth_kind"]).size().unstack(fill_value=0)
    cont["rows"] = cont.sum(axis=1)
    for c in ["accident_tip", "accident_drop", "lift", "qa"]:
        if c in cont:
            cont[c + "_share"] = (cont[c] / cont["rows"]).round(3)
    cont.to_csv(RES / "eval_contamination.csv")

    df = count.load_accident_rows()
    joint = count.merge_tracks(count.episodes(df, X_MIN), Y_MIN)
    truth_events = ev[
        (ev["truth_kind"].str.startswith("accident")) & (ev["ts"] >= ev["install_date"]) & (ev["ts"] >= A_START)
    ]
    n_true = truth_events["truth_event"].nunique()
    rows = []
    for k in K_GRID:
        a = count.apply_k(joint, k)
        anc = a[a["is_anchor"]]
        kinds = anc["truth_kinds"]
        rows.append(
            {
                "k_days": "none" if k is None else k,
                "counted": len(anc),
                "true_events": n_true,
                "with_true_accident": int(kinds.str.contains("accident").sum()),
                "lift_only": int((kinds == "lift").sum()),
                "qa_only": int((kinds == "qa").sum()),
                "lift_only_share": round(float((kinds == "lift").mean()), 3),
            }
        )
    pd.DataFrame(rows).to_csv(RES / "eval_counting.csv", index=False)

    # 실제 사고의 hazard는 달력 시간에 불변으로 넣었다. 경과 구간별 발생률이 전후반에서 같게 나와야 분해가 맞게 읽힌다
    first = truth_events.drop_duplicates("truth_event")
    mid = A_START + (END - A_START) / 2
    w = dev[["serial", "install_date"]].copy()
    w["obs_from"] = w["install_date"].clip(lower=A_START)
    edges = [0, 30, 90, 10000]
    lab = ["0~30", "30~90", "90+"]
    out = []
    for half, (lo_t, hi_t) in {"first_half": (A_START, mid), "second_half": (mid, END)}.items():
        ww = w.copy()
        ww["obs_from"], ww["obs_to"] = ww["obs_from"].clip(lower=lo_t), hi_t
        ww = ww[ww["obs_from"] < ww["obs_to"]]
        lo = (ww["obs_from"] - ww["install_date"]).dt.days.to_numpy()[:, None]
        hi = (ww["obs_to"] - ww["install_date"]).dt.days.to_numpy()[:, None]
        e = np.array(edges)
        expo = np.clip(np.minimum(hi, e[1:][None, :]) - np.maximum(lo, e[:-1][None, :]), 0, None).sum(axis=0)
        f = first[(first["ts"] >= lo_t) & (first["ts"] < hi_t)]
        age = (f["ts"] - f["install_date"]).dt.days
        n = pd.cut(age, bins=edges, right=False, labels=lab).value_counts().reindex(lab).to_numpy()
        for i, b in enumerate(lab):
            out.append(
                {
                    "half": half,
                    "age_days": b,
                    "true_accidents": int(n[i]),
                    "device_days": round(float(expo[i])),
                    "rate": round(n[i] / expo[i] * 1000, 3),
                }
            )
    pd.DataFrame(out).to_csv(RES / "eval_truth_rate.csv", index=False)

    # 실제 사고(첫 행, 재발 포함)의 SIR. hazard가 달력 시간에 불변이라 1 근처에서 잡음만큼 흔들려야 함
    ww = rates.obs_window(dev)
    me = rates.age_month_exposure(ww)
    tr = first.rename(columns={"ts": "s"})[["serial", "s", "install_date"]]
    nm = rates.age_month_count(tr, me)
    mon = pd.read_csv(RES / "monthly_rate.csv", parse_dates=["month"]).set_index("month")
    td = pd.DataFrame({"sir_truth": rates.sir(me, nm).round(3), "true_accidents": nm.sum(axis=1).astype(int)})
    td["partial"] = mon["partial"]
    td.to_csv(RES / "eval_truth_sir.csv", index_label="month")
    full = mon.index[~mon["partial"]]
    tm = pd.DataFrame([rates.mix_share("truth", me, nm, full[:2], full[-2:])])
    tm.to_csv(RES / "eval_truth_mix_share.csv", index=False)

    expo_m = device_days_per_month(w.assign(obs_to=END))
    print(f"true accidents {n_true:,}  exposure {expo_m.sum():,.0f}  true rate {n_true / expo_m.sum() * 1000:.3f}")
    print(td.to_string())
    print(tm.to_string(index=False))
    print(cont[[c for c in cont.columns if c.endswith("_share") or c == "rows"]].to_string())
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(out).to_string(index=False))


if __name__ == "__main__":
    main()
