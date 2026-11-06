"""업무 유형 분류 모델 학습, 교차 검증, time-based split, ablation, 최종 모델

평가 표본은 읽지 않는다. 바깥 5-fold StratifiedGroupKFold 3회 (seed 42, 43, 44)
안쪽 3-fold로 유형마다 C, class_weight, threshold 선택
모델: TF-IDF (단어 1~2-gram, 본문만) + 유형별 logistic regression (liblinear)
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from common import GROUP_OF, GROUPS, IDS, RESULTS, T1, WORK
from joblib import Parallel, delayed
from model_core import Model, analyze
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

OUT = T1 / "model_b"
NJOBS = int(os.environ.get("SPARK_CORES", "4"))
GRID = [(c, cw) for c in (0.3, 1, 3, 10, 30, 100) for cw in (None, "balanced")]
THR = np.round(np.arange(0.10, 0.91, 0.05), 2)
MIN_POS = 6  # 학습 fold 안 양성이 이보다 적으면 튜닝하지 않고 기본값 (C=10, balanced, 0.5)
DEFAULT = (10, "balanced", 0.5)


def load(with_jobkko: bool = False) -> pd.DataFrame:
    df = pd.read_parquet(WORK / "train_table.parquet")
    if not with_jobkko:
        df = df[df.source == "wdc"]
    df = df.reset_index(drop=True)
    df["toks"] = df.text.map(analyze)
    return df


def multi_hot(df: pd.DataFrame) -> np.ndarray:
    Y = np.zeros((len(df), len(IDS)), dtype=int)
    for i, t in enumerate(df.tasks.map(json.loads)):
        for x in t:
            Y[i, IDS.index(x)] = 1
    return Y


def f1(y: np.ndarray, p: np.ndarray) -> float:
    tp = int(((y == 1) & (p == 1)).sum())
    return 2 * tp / (2 * tp + int(((y == 0) & (p == 1)).sum()) + int(((y == 1) & (p == 0)).sum())) if tp else 0.0


def vectorizer() -> TfidfVectorizer:
    # 입력은 이미 분석한 토큰 목록. fold마다 다시 분석하지 않으려고
    return TfidfVectorizer(analyzer=lambda x: x, min_df=3, sublinear_tf=True, norm="l2", dtype=np.float64)


def fit_lr(X, y, C, cw):
    if y.sum() < 1 or y.sum() == len(y):
        return float(y.mean())
    return LogisticRegression(C=C, class_weight=cw, solver="liblinear", max_iter=300).fit(X, y)


def prob(m, X) -> np.ndarray:
    return np.full(X.shape[0], m) if isinstance(m, float) else m.predict_proba(X)[:, 1]


def tune_one(X, y, strat, groups, seed):
    """한 유형의 (C, class_weight, threshold). 학습 fold 안의 3-fold OOF F1이 최대인 조합"""
    if y.sum() < MIN_POS:
        return DEFAULT
    folds = list(StratifiedGroupKFold(3, shuffle=True, random_state=seed).split(X, strat, groups))
    best, best_f = DEFAULT, -1.0
    for C, cw in GRID:
        oof = np.zeros(len(y))
        for a, b in folds:
            oof[b] = prob(fit_lr(X[a], y[a], C, cw), X[b])
        for t in THR:
            f = f1(y, (oof >= t).astype(int))
            if f > best_f + 1e-12:
                best, best_f = (C, cw, float(t)), f
    return best


def fit_all(X, Y, strat, groups, seed):
    """유형 14개를 각각 튜닝해 학습"""
    cfg = Parallel(n_jobs=NJOBS)(delayed(tune_one)(X, Y[:, j], strat, groups, seed) for j in range(Y.shape[1]))
    models = [fit_lr(X, Y[:, j], c[0], c[1]) for j, c in enumerate(cfg)]
    return models, cfg


def predict(models, cfg, X) -> tuple[np.ndarray, np.ndarray]:
    P = np.column_stack([prob(m, X) for m in models])
    return P, (P >= np.array([c[2] for c in cfg])).astype(int)


def strat_label(df: pd.DataFrame) -> np.ndarray:
    return df.primary_task.values


def run_cv(df: pd.DataFrame, seed: int, text_col: str = "toks") -> tuple[np.ndarray, np.ndarray, list]:
    """바깥 5-fold 한 번. OOF 확률, OOF 예측, fold별 설정"""
    Y = multi_hot(df)
    P, H, cfgs = np.zeros(Y.shape), np.zeros(Y.shape, dtype=int), []
    for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(df, strat_label(df), df.group):
        vec = vectorizer()
        Xa = vec.fit_transform(df[text_col].iloc[a])
        Xb = vec.transform(df[text_col].iloc[b])
        models, cfg = fit_all(Xa, Y[a], strat_label(df.iloc[a]), df.group.values[a], seed)
        P[b], H[b] = predict(models, cfg, Xb)
        cfgs.append(cfg)
    return P, H, cfgs


def type_table(Y: np.ndarray, H: np.ndarray, mask=None) -> pd.DataFrame:
    if mask is not None:
        Y, H = Y[mask], H[mask]
    rows = []
    for j, t in enumerate(IDS):
        tp = int(((Y[:, j] == 1) & (H[:, j] == 1)).sum())
        fp = int(((Y[:, j] == 0) & (H[:, j] == 1)).sum())
        fn = int(((Y[:, j] == 1) & (H[:, j] == 0)).sum())
        rows.append(
            {
                "type": t,
                "n_pos": int(Y[:, j].sum()),
                "precision": tp / (tp + fp) if tp + fp else 0.0,
                "recall": tp / (tp + fn) if tp + fn else 0.0,
                "f1": f1(Y[:, j], H[:, j]),
            }
        )
    gj = {g: [IDS.index(t) for t in IDS if GROUP_OF[t] == g] for g in GROUPS}
    for g, cols in gj.items():
        y, h = Y[:, cols].max(1), H[:, cols].max(1)
        rows.append(
            {
                "type": f"[그룹] {g}",
                "n_pos": int(y.sum()),
                "precision": float(((y == 1) & (h == 1)).sum() / max(h.sum(), 1)),
                "recall": float(((y == 1) & (h == 1)).sum() / max(y.sum(), 1)),
                "f1": f1(y, h),
            }
        )
    return pd.DataFrame(rows)


def primary_of(P: np.ndarray, H: np.ndarray) -> np.ndarray:
    out = np.array(["none"] * len(P), dtype=object)
    for i in range(len(P)):
        over = np.flatnonzero(H[i])
        if len(over):
            out[i] = IDS[over[np.argmax(P[i, over])]]
    return out


def summary(df: pd.DataFrame, Y: np.ndarray, P: np.ndarray, H: np.ndarray) -> dict:
    t = type_table(Y, H)
    types = t[~t.type.str.startswith("[그룹]")]
    jac = (Y & H).sum(1) / np.maximum((Y | H).sum(1), 1)
    jac[(Y.sum(1) == 0) & (H.sum(1) == 0)] = 1.0  # 둘 다 none
    return {
        "macro_f1": types.f1.mean(),
        "n_types_f1_ge_060": int((types.f1 >= 0.60).sum()),
        "primary_agree": float((primary_of(P, H) == df.primary_task.values).mean()),
        "jaccard": float(jac.mean()),
    }


def share_error(df: pd.DataFrame, Y: np.ndarray, H: np.ndarray) -> float:
    """직무군 x 회차 칸 (라벨 10건 이상)에서 유형별 라벨 비율과 예측 비율의 차 평균 (%p)"""
    errs = []
    for (_, _), idx in df.groupby(["role_family", "crawl"]).indices.items():
        if len(idx) < 10:
            continue
        errs += list(np.abs(Y[idx].mean(0) - H[idx].mean(0)) * 100)
    return float(np.mean(errs))


def cmd_cv() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = load()
    Y = multi_hot(df)
    print(f"학습 {len(df)}건 (WDC), group {df.group.nunique()}개, 양성 수 {dict(zip(IDS, Y.sum(0)))}", flush=True)

    # 기본 모델 교차 검증 3회
    runs, tables, shares = [], [], []
    Ps = []
    for seed in (42, 43, 44):
        P, H, _ = run_cv(df, seed)
        Ps.append(P)
        runs.append({"seed": seed, **summary(df, Y, P, H)})
        tables.append(type_table(Y, H).assign(seed=seed))
        shares.append(share_error(df, Y, H))
        np.save(OUT / f"cv_oof_H_{seed}.npy", H)
        print(f"[cv seed {seed}] {runs[-1]} 칸 오차 {shares[-1]:.1f}%p ({time.time() - t0:.0f}초)", flush=True)
    pd.DataFrame(runs).assign(share_err_pp=shares).to_csv(RESULTS / "task1_b_cv_runs.csv", index=False)
    tt = pd.concat(tables)
    base = tt.groupby("type", sort=False)[["n_pos", "precision", "recall", "f1"]].mean()
    base["f1_sd"] = tt.groupby("type", sort=False).f1.std()
    base.to_csv(RESULTS / "task1_b_cv_by_type.csv")
    np.save(OUT / "cv_oof_P_mean.npy", np.mean(Ps, axis=0))
    P0 = np.load(OUT / "cv_oof_P_mean.npy")
    H0 = np.load(OUT / "cv_oof_H_42.npy")

    # 자른 단위별 (seed 42 OOF)
    cuts = []
    for name, col in [
        ("role_family", df.role_family),
        ("crawl", df.crawl),
        ("english", df.english),
        ("in_title", df.in_title),
        ("kind", df.kind),
    ]:
        for v in col.unique():
            m = (col == v).values
            if m.sum() >= 20:
                t = type_table(Y, H0, m)
                tt2 = t[~t.type.str.startswith("[그룹]")]
                cuts.append(
                    {
                        "cut": name,
                        "value": str(v),
                        "n": int(m.sum()),
                        "macro_f1_types_n>=5": tt2[tt2.n_pos >= 5].f1.mean(),
                        "primary_agree": float((primary_of(P0[m], H0[m]) == df.primary_task.values[m]).mean()),
                    }
                )
    pd.DataFrame(cuts).to_csv(RESULTS / "task1_b_cv_cuts.csv", index=False)

    # time-based split: 두 회차로 학습, 남은 회차로 평가
    tr = []
    for held in sorted(df.crawl.unique()):
        a, b = np.flatnonzero(df.crawl != held), np.flatnonzero(df.crawl == held)
        vec = vectorizer()
        Xa, Xb = vec.fit_transform(df.toks.iloc[a]), vec.transform(df.toks.iloc[b])
        models, cfg = fit_all(Xa, Y[a], strat_label(df.iloc[a]), df.group.values[a], 42)
        P, H = predict(models, cfg, Xb)
        t = type_table(Y[b], H)
        s = summary(df.iloc[b], Y[b], P, H)
        err = share_error(df.iloc[b].reset_index(drop=True), Y[b], H)
        tr.append({"held_out": held, "n_train": len(a), "n_test": len(b), **s, "share_err_pp": err})
        t.assign(held_out=held).to_csv(OUT / f"time_split_{held}.csv", index=False)
        print(f"[time-based split {held}] {tr[-1]}", flush=True)
    pd.DataFrame(tr).to_csv(RESULTS / "task1_b_time_split.csv", index=False)

    # ablation (seed 42 한 번): 제목 + 본문, 국내 라벨 추가
    ab = [{"variant": "본문만 (기준)", **runs[0], "share_err_pp": shares[0]}]
    d2 = df.assign(toks=[analyze("Title: " + (t or "") + "\n" + x) for t, x in zip(df.title, df.text)])
    P, H, _ = run_cv(d2, 42)
    ab.append({"variant": "제목 + 본문", **summary(d2, Y, P, H), "share_err_pp": share_error(d2, Y, H)})
    type_table(Y, H).to_csv(OUT / "cv_ablation_title_by_type.csv", index=False)
    print(f"[ablation 제목] {ab[-1]}", flush=True)
    d3 = load(with_jobkko=True)
    Y3 = multi_hot(d3)
    P3, H3, _ = run_cv(d3, 42)
    w = (d3.source == "wdc").values  # 평가는 WDC 행만
    ab.append(
        {
            "variant": "jobkko 한글 라벨 추가 (WDC 행만 평가)",
            **summary(d3[w], Y3[w], P3[w], H3[w]),
            "share_err_pp": share_error(d3[w].reset_index(drop=True), Y3[w], H3[w]),
        }
    )
    type_table(Y3[w], H3[w]).to_csv(OUT / "cv_ablation_jobkko_by_type.csv", index=False)
    print(f"[ablation jobkko] {ab[-1]}", flush=True)
    pd.DataFrame(ab).to_csv(RESULTS / "task1_b_cv_ablation.csv", index=False)
    print(f"끝 ({time.time() - t0:.0f}초)")


def cmd_final() -> None:
    """전체 WDC 학습 데이터로 최종 모델. 설정과 threshold는 OOF로 정한다"""
    df = load()
    Y = multi_hot(df)
    vec = vectorizer()
    X = vec.fit_transform(df.toks)
    models, cfg = fit_all(X, Y, strat_label(df), df.group.values, 42)
    coef = np.zeros((len(IDS), X.shape[1]))
    icpt = np.zeros(len(IDS))
    for j, m in enumerate(models):
        if isinstance(m, float):  # 양성이 없는 유형은 항상 음성
            icpt[j] = -20.0
        else:
            coef[j], icpt[j] = m.coef_[0], m.intercept_[0]
    vocab = {w: int(i) for w, i in vec.vocabulary_.items()}
    model = Model(vocab, vec.idf_, coef, icpt, np.array([c[2] for c in cfg]), list(IDS))
    model.save(OUT / "final")
    per_type = {t: {"C": c[0], "class_weight": c[1], "threshold": c[2]} for t, c in zip(IDS, cfg)}
    config = {"n_train": len(df), "vocab": len(vocab), "per_type": per_type}
    (OUT / "final" / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=1), encoding="utf-8")
    # sklearn과 model_core의 확률이 같은지 1,000건에서 확인
    s = df.sample(min(1000, len(df)), random_state=1)
    Xs = vec.transform(s.toks)
    ref = np.column_stack([prob(m, Xs) for m in models])
    got = model.proba(s.text.tolist())
    diff = np.abs(ref - got).max()
    print(f"최종 모델: 학습 {len(df)}건, 어휘 {len(vocab):,}, 확률 최대 차이 (sklearn vs core) {diff:.2e}")


if __name__ == "__main__":
    {"cv": cmd_cv, "final": cmd_final}[sys.argv[1]]()
