"""평가 표본 510건으로 최종 모델을 한 번만 평가하고 posting_share를 보정

평가 표본을 평가에 쓰는 유일한 곳. 결과 파일이 있으면 멈춘다. 평가 뒤 다시 학습하지 않는다
보정: 칸의 전체 예측 평균 + 평가 표본의 (라벨 - 예측) 평균으로 분자, 분모를 각각 보정 (prediction-powered inference)
95% 구간은 칸 안에서 평가 표본을 복원 추출한 부트스트랩 2,000회
"""

import hashlib
import json
import sys

import numpy as np
import pandas as pd
from common import (
    DATA,
    DATA_TITLE,
    GROUP_OF,
    GROUPS,
    IDS,
    RESULTS,
    SEAL,
    T1,
    english_ratio,
    multi_hot,
    plain_text,
    primary_hot,
    read_jsonl,
)

OUT = T1 / "model_b"
RESULT = OUT / "test_result.json"
K = len(IDS)
RNG = np.random.default_rng(42)
N_BOOT = 2000


def wf1(y, h, w) -> float:
    tp = (w * ((y == 1) & (h == 1))).sum()
    fp = (w * ((y == 0) & (h == 1))).sum()
    fn = (w * ((y == 1) & (h == 0))).sum()
    return float(2 * tp / (2 * tp + fp + fn)) if tp else 0.0


def type_rows(Y, H, w, boot_idx=None) -> pd.DataFrame:
    rows = []
    cols = [(t, [j]) for j, t in enumerate(IDS)]
    cols += [(f"[그룹] {g}", [IDS.index(t) for t in IDS if GROUP_OF[t] == g]) for g in GROUPS]
    for name, js in cols:
        y, h = Y[:, js].max(1), H[:, js].max(1)
        r = {
            "type": name,
            "n_pos": int(y.sum()),
            "f1_weighted": wf1(y, h, w),
            "f1_unweighted": wf1(y, h, np.ones(len(y))),
            "precision": float((w * ((y == 1) & (h == 1))).sum() / max((w * (h == 1)).sum(), 1e-9)),
            "recall": float((w * ((y == 1) & (h == 1))).sum() / max((w * (y == 1)).sum(), 1e-9)),
        }
        if boot_idx is not None:
            bs = [wf1(y[i], h[i], w[i]) for i in boot_idx]
            r["f1_lo"], r["f1_hi"] = np.percentile(bs, [2.5, 97.5])
        rows.append(r)
    return pd.DataFrame(rows)


def main() -> None:
    if RESULT.exists():
        sys.exit(f"{RESULT} 이 있다. 평가 표본은 한 번만 본다. 새 평가가 필요하면 새 표본을 뽑는다")
    pred = pd.read_parquet(T1 / "task_pred_b")
    pred["P"] = list(np.vstack(pred.probs.values))
    man = pd.read_parquet(SEAL / "manifest_test.parquet")[["doc_id", "cell", "role_family", "crawl"]]
    lab = pd.read_parquet(SEAL / "labels_test.parquet")[["doc_id", "tasks", "primary_task"]]
    raw = pd.DataFrame([r for f in sorted((SEAL / "in").glob("*.jsonl")) for r in read_jsonl(f)])
    te = man.merge(lab, on="doc_id", suffixes=("", "_lab")).merge(
        pred[["doc_id", "tasks", "primary_task", "P"]].rename(
            columns={"tasks": "p_tasks", "primary_task": "p_primary"}
        ),
        on="doc_id",
    )
    te = te.merge(raw[["doc_id", "title", "description"]], on="doc_id")
    assert len(te) == len(man) == 510, f"평가 표본 {len(te)}건 / 매니페스트 {len(man)}건: 라벨 또는 예측이 빠짐"
    te["english"] = te.description.map(plain_text).map(lambda t: english_ratio(t) >= 0.12)

    # 모집단 칸 (직무군 x 회차)
    pred["H"] = list(multi_hot(pred.tasks))
    pred["PH"] = list(primary_hot(pred.primary_task))
    pred["nn"] = (pred.primary_task != "none").astype(float)
    cell_n = pred.groupby(["role_family", "crawl"]).size()
    te["N"] = [cell_n[(r, c)] for r, c in zip(te.role_family, te.crawl)]
    te["w"] = te.N / te.groupby("cell").doc_id.transform("size")

    Y, H = multi_hot(te.tasks), multi_hot(te.p_tasks)
    w = te.w.values
    # 유형별, 그룹별 F1 (가중 + 구간)
    boot_idx = [
        np.concatenate([RNG.choice(np.flatnonzero(te.cell == c), (te.cell == c).sum()) for c in te.cell.unique()])
        for _ in range(500)
    ]
    by_type = type_rows(Y, H, w, boot_idx)
    by_type["pass_f1_060"] = by_type.f1_weighted >= 0.60
    by_type.to_csv(RESULTS / "task1_b_test_by_type.csv", index=False)

    # 한 공고 단위: 대표 업무 일치, 목록 자카드, 자른 단위별
    jac = (Y * H).sum(1) / np.maximum(((Y + H) > 0).sum(1), 1)
    jac[(Y.sum(1) == 0) & (H.sum(1) == 0)] = 1.0
    agree = (te.primary_task.values == te.p_primary.values).astype(float)
    cuts = []
    for name, col in [
        ("role_family", te.role_family),
        ("crawl", te.crawl),
        ("english", te.english),
        ("in_title", te.title.fillna("").str.lower().str.contains(DATA_TITLE)),
    ]:
        for v in sorted(col.unique(), key=str):
            m = (col == v).values
            t = type_rows(Y[m], H[m], w[m])
            ty = t[~t.type.str.startswith("[그룹]") & (t.n_pos >= 5)]
            g = t[t.type.str.startswith("[그룹]")]
            cuts.append(
                {
                    "cut": name,
                    "value": str(v),
                    "n": int(m.sum()),
                    "types_n_pos>=5": len(ty),
                    "macro_f1_weighted": ty.f1_weighted.mean(),
                    "group_macro_f1": g.f1_weighted.mean(),
                    "primary_agree": float(np.average(agree[m], weights=w[m])),
                    "jaccard": float(np.average(jac[m], weights=w[m])),
                }
            )
    pd.DataFrame(cuts).to_csv(RESULTS / "task1_b_test_cuts.csv", index=False)

    # posting_share: 칸, 직무군 단위. 예측만, 라벨 (평가 표본), 보정 + 구간
    def share_tables(kind: str) -> pd.DataFrame:
        Ycol = Y if kind == "tasks" else primary_hot(te.primary_task)
        Fte = H if kind == "tasks" else primary_hot(te.p_primary)
        dte = (te.primary_task != "none").values.astype(float)
        gte = (te.p_primary != "none").values.astype(float)
        pop_f = np.vstack(pred.H.values) if kind == "tasks" else np.vstack(pred.PH.values)
        cells = {}
        for (role, crawl), g in pred.groupby(["role_family", "crawl"]):
            idx = g.index.values
            cells[(role, crawl)] = {"N": len(g), "f": pop_f[idx].mean(0), "g": g.nn.mean()}
        rows = []
        # 평가 표본이 있는 칸의 보정 구성 요소
        comp = {}
        for (role, crawl), c in cells.items():
            m = ((te.role_family == role) & (te.crawl == crawl)).values
            if m.sum():
                comp[(role, crawl)] = (np.flatnonzero(m), c)

        def est(sel, idx_map=None):
            num = np.zeros(K)
            den = 0.0
            for key in sel:
                m, c = comp[key]
                m = m if idx_map is None else idx_map[key]
                num += c["N"] * (c["f"] + (Ycol[m] - Fte[m]).mean(0))
                den += c["N"] * (c["g"] + (dte[m] - gte[m]).mean())
            return num / den

        def raw_est(sel):
            return sum(cells[k]["N"] * cells[k]["f"] for k in sel) / sum(cells[k]["N"] * cells[k]["g"] for k in sel)

        def lab_est(sel):
            num = sum(cells[k]["N"] * Ycol[comp[k][0]].mean(0) for k in sel)
            return num / sum(cells[k]["N"] * dte[comp[k][0]].mean() for k in sel)

        groups = [((role, crawl), [(role, crawl)]) for (role, crawl) in cells]
        groups += [((role, "ALL"), [k for k in cells if k[0] == role]) for role in sorted({k[0] for k in cells})]
        boots = [{k: RNG.choice(comp[k][0], len(comp[k][0])) for k in comp} for _ in range(N_BOOT)]
        for (role, crawl), sel in groups:
            have = all(k in comp for k in sel)
            raw_s = raw_est(sel)
            if have:
                ppi, lab_s = est(sel), lab_est(sel)
                bs = np.array([est(sel, b) for b in boots])
                lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
            for j, t in enumerate(IDS):
                rows.append(
                    {
                        "kind": kind,
                        "role_family": role,
                        "crawl": crawl,
                        "task_id": t,
                        "n_pop": sum(cells[k]["N"] for k in sel),
                        "n_test": sum(len(comp[k][0]) for k in sel if k in comp),
                        "share_raw": raw_s[j],
                        "share_label_test": lab_s[j] if have else np.nan,
                        "share_ppi": np.clip(ppi[j], 0, 1) if have else np.nan,
                        "ci_lo": max(lo[j], 0) if have else np.nan,
                        "ci_hi": min(hi[j], 1) if have else np.nan,
                    }
                )
        return pd.DataFrame(rows)

    sh = pd.concat([share_tables("tasks"), share_tables("primary")])
    sh.to_csv(RESULTS / "task1_b_test_share.csv", index=False)
    s = sh[(sh.kind == "tasks") & sh.share_ppi.notna()].copy()
    s["abs_err_pp"] = (s.share_raw - s.share_label_test).abs() * 100
    err_role = s[(s.crawl == "ALL") & s.role_family.isin(["DA", "BA/BI", "DS"])].abs_err_pp.mean()
    err_cell = s[(s.crawl != "ALL") & s.role_family.isin(["DA", "BA/BI", "DS"])].abs_err_pp.mean()

    ty = by_type[~by_type.type.str.startswith("[그룹]")]
    model_bytes = (OUT / "final/model.npz").read_bytes() + (OUT / "final/vocab.json").read_bytes()
    model_hash = hashlib.sha256(model_bytes).hexdigest()[:16]
    res = {
        "model_sha256_16": model_hash,
        "n_test": len(te),
        "macro_f1_weighted": float(ty.f1_weighted.mean()),
        "types_pass_f1_060": int((ty.f1_weighted >= 0.6).sum()),
        "types_n_pos_lt_10": ty[ty.n_pos < 10].type.tolist(),
        "primary_agree_weighted": float(np.average(agree, weights=w)),
        "jaccard_weighted": float(np.average(jac, weights=w)),
        "share_err_role_pp": float(err_role),
        "share_err_cell_pp": float(err_cell),
        "share_err_role_pass_5pp": bool(err_role <= 5),
    }
    RESULT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    # 11, 13이 저장소만으로 돌도록 원문과 URL을 뺀 표. label_row는 라벨 파일 순서 (부트스트랩이 행 순서를 탄다)
    label_row = {d: i for i, d in enumerate(lab.doc_id)}
    eval_cols = ["doc_id", "role_family", "crawl", "cell", "tasks", "primary_task", "english"]
    te[eval_cols].assign(label_row=te.doc_id.map(label_row)).to_csv(DATA / "eval_labels.csv", index=False)
    pred_cols = ["doc_id", "crawl", "role_family", "tasks", "primary_task", "english"]
    pred[pred_cols].to_parquet(DATA / "task_pred.parquet", index=False, compression="zstd")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
