"""최종 posting_share 표: 유형 14개 + 묶음 5개, 전체, 영어, 비영어 층, 칸과 직무군 단위, 보정값 + 95% 구간

같은 평가 표본으로 보고 단위만 나눠 다시 계산한다 (다시 학습하지 않음)
묶음 상태: 평가 표본 가중 F1 0.60 이상이면 통과, 아니면 참고용
"""

import numpy as np
import pandas as pd
from common import DATA, GROUP_OF, GROUPS, IDS, RESULTS, multi_hot, primary_hot

K = len(IDS)
UNITS = list(IDS) + [f"[그룹] {g}" for g in GROUPS]
U = len(UNITS)
N_BOOT = 2000
RNG = np.random.default_rng(42)
MAIN = ["DA", "BA/BI", "DS"]


def to_units(M: np.ndarray) -> np.ndarray:
    """(n x 14) 유형 행렬을 (n x 19) 유형 + 묶음으로 (묶음은 멤버 중 하나라도 있으면 1)"""
    cols = [M[:, j] for j in range(K)]
    for g in GROUPS:
        cols.append(M[:, [IDS.index(t) for t in IDS if GROUP_OF[t] == g]].max(1))
    return np.column_stack(cols)


def main() -> None:
    pred = pd.read_parquet(DATA / "task_pred.parquet")
    labels = pd.read_csv(DATA / "eval_labels.csv")
    te = labels.merge(
        pred[["doc_id", "tasks", "primary_task"]].rename(columns={"tasks": "p_tasks", "primary_task": "p_primary"}),
        on="doc_id",
    )
    assert len(te) == 510

    status = pd.read_csv(RESULTS / "task1_b_test_by_type.csv").set_index("type").f1_weighted
    gstat = {g: ("통과" if status[f"[그룹] {g}"] >= 0.6 else "참고용") for g in GROUPS}
    typ_pass = {t: bool(status[t] >= 0.6) for t in IDS}

    # 모집단 (예측)과 평가 표본 (라벨, 예측)의 유형 + 묶음 행렬
    pred_t = to_units(multi_hot(pred.tasks))
    pred_p = to_units(primary_hot(pred.primary_task))
    pred_nn = (pred.primary_task != "none").values.astype(float)
    te_y = {"tasks": to_units(multi_hot(te.tasks)), "primary": to_units(primary_hot(te.primary_task))}
    te_f = {"tasks": to_units(multi_hot(te.p_tasks)), "primary": to_units(primary_hot(te.p_primary))}
    te_d, te_g = (te.primary_task != "none").values.astype(float), (te.p_primary != "none").values.astype(float)
    pop_f = {"tasks": pred_t, "primary": pred_p}

    rows = []
    for sl, pm, tm in [
        ("전체", np.ones(len(pred), bool), np.ones(len(te), bool)),
        ("영어", pred.english.values, te.english.values),
        ("비영어", ~pred.english.values, ~te.english.values),
    ]:
        cells = {}
        for (role, crawl), idx in pred[pm].groupby(["role_family", "crawl"]).indices.items():
            ii = np.flatnonzero(pm)[idx]
            cells[(role, crawl)] = {"N": len(ii), "ii": ii, "g": pred_nn[ii].mean()}
        tcell = {k: np.flatnonzero(tm & (te.role_family == k[0]).values & (te.crawl == k[1]).values) for k in cells}
        sels = [((r, c), [(r, c)]) for (r, c) in cells]
        sels += [((r, "ALL"), [k for k in cells if k[0] == r]) for r in sorted({k[0] for k in cells})]
        for kind in ["tasks", "primary"]:
            Y, F = te_y[kind], te_f[kind]
            # 칸마다 부트스트랩 잔차 평균 (rep x U), (rep)을 한 번에
            comp = {}
            for k, c in cells.items():
                t = tcell[k]
                if len(t) == 0:
                    continue
                draws = RNG.integers(0, len(t), (N_BOOT, len(t)))
                R = (Y[t] - F[t])[draws].mean(1)
                RD = (te_d[t] - te_g[t])[draws].mean(1)
                comp[k] = (R, RD, (Y[t] - F[t]).mean(0), (te_d[t] - te_g[t]).mean(), Y[t].mean(0), te_d[t].mean())
            for key, sel in sels:
                have = all(k in comp for k in sel)
                n_pop = sum(cells[k]["N"] for k in sel)
                f_c = {k: pop_f[kind][cells[k]["ii"]].mean(0) for k in sel}
                den_raw = sum(cells[k]["N"] * cells[k]["g"] for k in sel)
                raw_s = sum(cells[k]["N"] * f_c[k] for k in sel) / den_raw
                if have:
                    num = sum(cells[k]["N"] * (f_c[k] + comp[k][2]) for k in sel)
                    den = sum(cells[k]["N"] * (cells[k]["g"] + comp[k][3]) for k in sel)
                    ppi = num / den
                    lab_s = sum(cells[k]["N"] * comp[k][4] for k in sel) / sum(cells[k]["N"] * comp[k][5] for k in sel)
                    bn = sum(cells[k]["N"] * (f_c[k] + comp[k][0]) for k in sel)
                    bd = sum(cells[k]["N"] * (cells[k]["g"] + comp[k][1]) for k in sel)[:, None]
                    lo, hi = np.percentile(bn / bd, [2.5, 97.5], axis=0)
                for u, name in enumerate(UNITS):
                    grp = GROUP_OF[name] if name in IDS else name.replace("[그룹] ", "")
                    rows.append(
                        {
                            "slice": sl,
                            "kind": kind,
                            "role_family": key[0],
                            "crawl": key[1],
                            "unit": name,
                            "unit_type": "type" if name in IDS else "group",
                            "task_group": grp,
                            "group_status": gstat[grp],
                            "type_pass_f1_060": typ_pass.get(name, np.nan),
                            "n_pop": n_pop,
                            "n_test": sum(len(tcell[k]) for k in sel),
                            "share_raw": raw_s[u],
                            "share_label_test": lab_s[u] if have else np.nan,
                            "share_ppi": float(np.clip(ppi[u], 0, 1)) if have else np.nan,
                            "ci_lo": max(lo[u], 0) if have else np.nan,
                            "ci_hi": min(hi[u], 1) if have else np.nan,
                        }
                    )
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "task1_b_share_final.csv", index=False)
    old = pd.read_csv(RESULTS / "task1_b_test_share.csv")
    a = df[(df.slice == "전체") & (df.kind == "tasks") & (df.unit_type == "type")].merge(
        old[old.kind == "tasks"],
        left_on=["role_family", "crawl", "unit"],
        right_on=["role_family", "crawl", "task_id"],
        suffixes=("", "_old"),
    )
    print(
        f"행 {len(df):,}. 이전 표(유형, 전체 층)와 보정값 최대 차이 "
        f"{np.nanmax(np.abs(a.share_ppi - a.share_ppi_old)):.2e}"
    )
    g = df[
        (df.slice == "전체")
        & (df.kind == "tasks")
        & (df.crawl == "ALL")
        & (df.role_family == "DA")
        & (df.unit_type == "group")
    ]
    print(
        g[["unit", "group_status", "share_raw", "share_label_test", "share_ppi", "ci_lo", "ci_hi"]]
        .round(3)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
