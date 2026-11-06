"""평가 재계산: 주 대상 기준 F1, 오차 분포, 보정값과 라벨만 쓴 구간, 회차 차이

예측과 평가 라벨만 읽고 다시 학습하지 않는다. 10_eval_test.py의 지표 코드는 쓰지 않는다
"""

import json

import numpy as np
import pandas as pd
from common import GROUP_OF, GROUPS, IDS, RESULTS, SEAL, T1, wdc_doc_id

ROLES = ["DA", "BA/BI", "DS"]
CRAWLS = ["2022-12", "2023-12", "2024-12"]
N_BOOT = 1000
GROUP_COLS = {g: [IDS.index(t) for t in IDS if GROUP_OF[t] == g] for g in GROUPS}
UNITS = {t: [j] for j, t in enumerate(IDS)} | {f"[그룹] {g}": js for g, js in GROUP_COLS.items()}


def multi_hot(tasks: pd.Series) -> np.ndarray:
    Y = np.zeros((len(tasks), len(IDS)))
    for i, x in enumerate(tasks):
        for t in json.loads(x) if isinstance(x, str) else list(x):
            Y[i, IDS.index(t)] = 1
    return Y


def f1(y: np.ndarray, h: np.ndarray, w: np.ndarray) -> float:
    tp, fp, fn = (w * y * h).sum(), (w * (1 - y) * h).sum(), (w * y * (1 - h)).sum()
    return float(2 * tp / (2 * tp + fp + fn)) if tp else 0.0


def main() -> None:
    rng = np.random.default_rng(7)
    pred = pd.read_parquet(T1 / "task_pred_b", columns=["doc_id", "role_family", "crawl", "tasks", "primary_task"])
    man = pd.read_parquet(SEAL / "manifest_test.parquet")
    lab = pd.read_parquet(SEAL / "labels_test.parquet")
    te = (
        man[["doc_id", "role_family", "crawl"]]
        .merge(lab[["doc_id", "tasks", "primary_task"]], on="doc_id")
        .merge(
            pred[["doc_id", "tasks", "primary_task"]].rename(columns={"tasks": "p_tasks", "primary_task": "p_primary"}),
            on="doc_id",
        )
    )
    assert len(te) == 510, len(te)
    Y, H = multi_hot(te.tasks), multi_hot(te.p_tasks)
    cell_n = pred.groupby(["role_family", "crawl"]).size()
    te["w"] = [cell_n[(r, c)] / 50 if r != "DA 인접" else cell_n[(r, c)] / 20 for r, c in zip(te.role_family, te.crawl)]
    print("가중치 중 DA 인접 비중", round(te[te.role_family == "DA 인접"].w.sum() / te.w.sum(), 3))
    print("라벨 none", te.groupby("role_family").primary_task.apply(lambda s: round((s == "none").mean(), 3)).to_dict())
    print("예측 none", te.groupby("role_family").p_primary.apply(lambda s: round((s == "none").mean(), 3)).to_dict())

    # 학습과 같은 조직을 뺀 평가
    tr = pd.read_parquet(T1 / "labels_work_b/train_table.parquet")
    orgs = set(tr[tr.source == "wdc"].org.dropna().str.lower())
    d = pd.read_parquet(T1 / "wdc_jobs_dedup.parquet", columns=["crawl", "page_url", "node", "org_name"])
    d["doc_id"] = wdc_doc_id(d)
    te_org = te[["doc_id"]].merge(d[["doc_id", "org_name"]], on="doc_id", how="left").org_name
    same_org = te_org.fillna("").str.lower().isin(orgs).values

    # F1: 전체 510, 주 대상, 조직 겹침 제외, 직무군별
    is_main = te.role_family.isin(ROLES).values
    masks = {
        "전체 510": np.ones(len(te), bool),
        "주 대상 450": is_main,
        "주 대상, 학습과 같은 조직 제외": is_main & ~same_org,
    }
    masks |= {r: (te.role_family == r).values for r in [*ROLES, "DA 인접"]}
    rows = []
    for name, m in masks.items():
        w = te.w.values[m]
        for u, js in UNITS.items():
            score = f1(Y[m][:, js].max(1), H[m][:, js].max(1), w)
            rows.append({"평가 범위": name, "n": int(m.sum()), "unit": u, "f1": score})
    fdf = pd.DataFrame(rows)
    fdf.to_csv(RESULTS / "task1_b_recheck_f1.csv", index=False)
    print(fdf.pivot(index="unit", columns="평가 범위", values="f1").loc[list(UNITS)].round(2)[list(masks)].to_string())

    # 직무군 x 단위: 예측만, 라벨만, 보정, 구간 폭
    pop, nn_pop = multi_hot(pred.tasks), (pred.primary_task != "none").values.astype(float)
    pop_u = {u: pop[:, js].max(1) for u, js in UNITS.items()}
    y_u = {u: Y[:, js].max(1) for u, js in UNITS.items()}
    h_u = {u: H[:, js].max(1) for u, js in UNITS.items()}
    d_lab, d_pred = (te.primary_task != "none").values.astype(float), (te.p_primary != "none").values.astype(float)
    cell_idx = {(r, c): np.flatnonzero(((te.role_family == r) & (te.crawl == c)).values) for r in ROLES for c in CRAWLS}
    pop_mask = {(r, c): ((pred.role_family == r) & (pred.crawl == c)).values for r in ROLES for c in CRAWLS}

    def est(u, keys, idx, mode):
        num = den = 0.0
        for k in keys:
            i, n = idx[k], pop_mask[k].sum()
            if mode == "ppi":
                num += n * (pop_u[u][pop_mask[k]].mean() + (y_u[u][i] - h_u[u][i]).mean())
                den += n * (nn_pop[pop_mask[k]].mean() + (d_lab[i] - d_pred[i]).mean())
            elif mode == "label":
                num, den = num + n * y_u[u][i].mean(), den + n * d_lab[i].mean()
            else:
                num, den = num + n * pop_u[u][pop_mask[k]].mean(), den + n * nn_pop[pop_mask[k]].mean()
        return num / den

    boots = [{k: rng.choice(v, len(v)) for k, v in cell_idx.items()} for _ in range(N_BOOT)]

    def width(a):
        return float(np.subtract(*np.percentile(a, [97.5, 2.5])) * 100)

    rows = []
    for r in ROLES:
        keys = [(r, c) for c in CRAWLS]
        for u in UNITS:
            raw = est(u, keys, cell_idx, "raw")
            label = est(u, keys, cell_idx, "label")
            ppi = est(u, keys, cell_idx, "ppi")
            rows.append(
                {
                    "role_family": r,
                    "unit": u,
                    "share_raw": raw,
                    "share_label_test": label,
                    "share_ppi": ppi,
                    "err_pp": abs(raw - label) * 100,
                    "width_ppi_pp": width([est(u, keys, b, "ppi") for b in boots]),
                    "width_label_pp": width([est(u, keys, b, "label") for b in boots]),
                }
            )
    role = pd.DataFrame(rows)
    role.to_csv(RESULTS / "task1_b_recheck_role.csv", index=False)
    ty = role[~role.unit.str.startswith("[")]
    common = ty[ty.share_label_test >= 0.10]
    print(
        f"오차 평균 {ty.err_pp.mean():.2f} 중앙값 {ty.err_pp.median():.2f} 최대 {ty.err_pp.max():.1f}, "
        f"5%p 초과 {(ty.err_pp > 5).sum()}, 10%p 초과 {(ty.err_pp > 10).sum()} / {len(ty)}"
    )
    rare = ty[ty.share_label_test < 0.10]
    print(
        f"라벨 비율 10% 이상 {len(common)}칸 평균 {common.err_pp.mean():.2f}, "
        f"10% 미만 {len(ty) - len(common)}칸 평균 {rare.err_pp.mean():.2f}"
    )
    print(
        role[role.share_label_test >= 0.10]
        .groupby("role_family")[["width_ppi_pp", "width_label_pp"]]
        .mean()
        .round(1)
        .to_string()
    )

    # 2022-12와 2024-12 차이
    rows = []
    for r in ROLES:
        for u in UNITS:

            def diff(idx, m, u=u, r=r):
                return est(u, [(r, "2024-12")], idx, m) - est(u, [(r, "2022-12")], idx, m)

            lo, hi = np.percentile([diff(b, "ppi") for b in boots], [2.5, 97.5])
            rows.append(
                {
                    "role_family": r,
                    "unit": u,
                    "diff_raw_pp": diff(cell_idx, "raw") * 100,
                    "diff_label_pp": diff(cell_idx, "label") * 100,
                    "diff_ppi_pp": diff(cell_idx, "ppi") * 100,
                    "ci_lo_pp": lo * 100,
                    "ci_hi_pp": hi * 100,
                    "ci_excludes_0": bool(lo > 0 or hi < 0),
                }
            )
    tr_df = pd.DataFrame(rows)
    tr_df.to_csv(RESULTS / "task1_b_recheck_trend.csv", index=False)
    opposite = (
        (np.sign(tr_df.diff_raw_pp) != np.sign(tr_df.diff_ppi_pp))
        & (tr_df.diff_raw_pp.abs() >= 2)
        & (tr_df.diff_ppi_pp.abs() >= 2)
    )
    width_mean = (tr_df.ci_hi_pp - tr_df.ci_lo_pp).mean()
    print(
        f"구간이 0을 벗어난 칸 {tr_df.ci_excludes_0.sum()} / {len(tr_df)}, 구간 폭 평균 {width_mean:.1f}%p, "
        f"예측만과 방향이 반대 {opposite.sum()}칸"
    )
    print(tr_df[tr_df.ci_excludes_0].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
