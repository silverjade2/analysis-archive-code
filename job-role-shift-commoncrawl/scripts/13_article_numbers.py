"""글의 수치 표: 직무군별 업무 구성, 평가 표본 라벨 구성, 기술 등장률 변화와 민감도, 국내 비교, 직함 변경 상한

인자로 표 이름을 주면 그 표만 쓴다 (task_mix, label_mix, rename_bound는 저장소만으로 돈다)
"""

import ast
import json
import re
import sys

import numpy as np
import pandas as pd
from candidates import load
from common import DATA, GROUPS, RESULTS, T1, WORK_DIR
from common import US as US_SQL
from roles import MIN_BODY_CHARS

SHARE = RESULTS / "task1_b_share_final.csv"
MAIN = ["DA", "BA/BI", "DS"]
A, B = "2022-12", "2024-12"
US = ast.literal_eval(US_SQL)


def task_mix() -> pd.DataFrame:
    d = pd.read_csv(SHARE)
    d = d[(d.crawl == "ALL") & d.role_family.isin(MAIN) & d.slice.isin(["전체", "영어"])]
    cols = [
        "slice",
        "kind",
        "role_family",
        "unit",
        "unit_type",
        "task_group",
        "group_status",
        "n_pop",
        "n_test",
        "share_raw",
        "share_label_test",
        "share_ppi",
        "ci_lo",
        "ci_hi",
    ]
    return d[cols]


def task_groups() -> dict[str, str]:
    x = pd.read_csv(SHARE).query("unit_type == 'type'")
    return dict(zip(x.unit, x.task_group))


def label_mix() -> pd.DataFrame:
    """평가 표본 라벨만 쓴 직무군 구성. 칸의 모집단 크기로 가중하고 칸 안에서 복원 추출"""
    tg = task_groups()
    x = pd.read_csv(DATA / "eval_labels.csv").sort_values("label_row")
    x["tset"] = x.tasks.map(json.loads)
    x["none"] = x.tset.map(len) == 0
    for g in GROUPS:
        x[g] = x.tset.map(lambda s, g=g: any(tg.get(i) == g for i in s))
    x["분석만"] = x["분석"] & ~x["모델링"] & ~x["데이터 엔지니어링"] & ~x["AI 엔지니어링"] & ~x["연구"]
    x["모델링만"] = x["모델링"] & ~x["분석"] & ~x["데이터 엔지니어링"] & ~x["AI 엔지니어링"] & ~x["연구"]
    x["분석+모델링"] = x["분석"] & x["모델링"]
    x["유형 수"] = x.tset.map(len)
    pop = pd.read_csv(SHARE).query("slice == '전체' and crawl != 'ALL'")[["role_family", "crawl", "n_pop"]]
    pop = pop.drop_duplicates(["role_family", "crawl"])
    x = x.merge(pop, on=["role_family", "crawl"])
    rng = np.random.default_rng(42)
    rows = []

    def est(fr: pd.DataFrame, col: str, over_nonnone: bool) -> float:
        w = fr.groupby("crawl").n_pop.first()
        num = den = 0.0
        for c, g in fr.groupby("crawl"):
            base = g[~g.none] if over_nonnone else g
            num += w[c] * base[col].sum() / len(g)
            den += w[c] * len(base) / len(g)
        return num / den if den else float("nan")

    measures = [("none", False)] + [(k, True) for k in [*GROUPS, "분석만", "모델링만", "분석+모델링", "유형 수"]]
    for role in [*MAIN, "DA 인접"]:
        fr = x[x.role_family == role]
        for col, nn in measures:
            v = est(fr, col, nn)
            bs = []
            for _ in range(2000):
                s = pd.concat(
                    [g.sample(len(g), replace=True, random_state=rng.integers(1 << 31)) for _, g in fr.groupby("crawl")]
                )
                bs.append(est(s, col, nn))
            rows.append(
                dict(
                    role=role,
                    measure=col,
                    over="none 제외" if nn else "전체",
                    n=len(fr),
                    n_nonnone=int((~fr.none).sum()),
                    value=v,
                    lo=np.nanpercentile(bs, 2.5),
                    hi=np.nanpercentile(bs, 97.5),
                )
            )
    return pd.DataFrame(rows)


def rate_row(ok: pd.DataFrame, col: str) -> tuple[int, float]:
    return len(ok), float(ok[col].mean()) if len(ok) else float("nan")


def skill_trend() -> pd.DataFrame:
    cands = load()
    hits = pd.read_parquet(T1 / "c_skill_hits.parquet")
    hits["role_family"] = hits.role_family.astype(str)
    w = pd.read_parquet(T1 / "wdc_jobs_dedup.parquet", columns=["crawl", "page_url", "node", "host", "country"])
    x = hits[hits.body_ok].merge(w, on=["crawl", "page_url", "node"], how="left", validate="m:1")
    x["is_us"] = x.country.fillna("").str.strip().str.lower().isin(US)
    all_host = w.groupby("host").crawl.nunique()
    x["fixed_host"] = x.host.map(all_host) == 3
    scopes = {"전체": x, "미국": x[x.is_us], "사이트 고정": x[x.fixed_host]}
    rows = []
    for scope, fr in scopes.items():
        for role in [*MAIN, "DE", "ML/AI 엔지니어"]:
            r = fr[fr.role_family == role]
            a, b, mid = r[r.crawl == A], r[r.crawl == B], r[r.crawl == "2023-12"]
            for i, c in enumerate(cands):
                col = f"s{i:03d}"
                na, pa = rate_row(a, col)
                nb, pb = rate_row(b, col)
                _, pm = rate_row(mid, col)
                se = np.sqrt(pa * (1 - pa) / max(na, 1) + pb * (1 - pb) / max(nb, 1))
                rows.append(
                    dict(
                        scope=scope,
                        role=role,
                        skill=c["skill"],
                        source=c["source"],
                        n_2022=na,
                        n_2024=nb,
                        rate_2022=pa,
                        rate_2023=pm,
                        rate_2024=pb,
                        diff_pp=(pb - pa) * 100,
                        diff_lo=(pb - pa - 1.96 * se) * 100,
                        diff_hi=(pb - pa + 1.96 * se) * 100,
                    )
                )
    return pd.DataFrame(rows)


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n or np.isnan(p):
        return (float("nan"), float("nan"))
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def rename_bound() -> pd.DataFrame:
    """직함 변경의 상한. 미국 전체 공고 감소율을 그대로 받았을 때의 기대 건수와 실제 건수의 차"""
    r = pd.read_csv(RESULTS / "task1_role_share.csv").query("scope == '미국'").set_index("g")
    keep = r.loc[["DA", "BA/BI", "DE", "DS", "ML/AI 엔지니어"], [A, B]].copy()
    ratio = r.loc["전체 공고", B] / r.loc["전체 공고", A]
    keep["expected_2024"] = keep[A] * ratio
    keep["gap"] = keep[B] - keep.expected_2024
    keep["total_ratio"] = ratio
    return keep.reset_index().rename(columns={"g": "role"})


def jobkko() -> pd.DataFrame:
    """국내 공고 그룹 (본문이 가장 긴 공고 하나)의 직무군별 기술 등장률과 라벨 업무 구성"""
    tg = task_groups()
    pool = pd.read_parquet(T1 / "label_pool.parquet")
    pool = pool[pool.source == "jobkko"]
    jk = pd.read_parquet(WORK_DIR / "jobkko/jobkko_postings.parquet")
    jk["doc_id"] = "jobkko-" + jk.final_group_id.fillna(jk.group_id).fillna(jk.posting_id).astype("int64").astype(str)
    jk["txt"] = (jk.title.fillna("") + "\n" + jk.description.fillna("")).str.lower()
    jk["l"] = jk.description.fillna("").str.len()
    one = jk.sort_values("l").groupby("doc_id").tail(1)
    one = one[one.l >= MIN_BODY_CHARS][["doc_id", "txt"]]  # WDC 등장률과 같은 분모
    x = pool[["doc_id", "role_family", "is_main"]].merge(one, on="doc_id")
    lab = pd.read_parquet(T1 / "labels_agent.parquet")[["doc_id", "tasks", "primary_task"]]
    x = x.merge(lab, on="doc_id", how="left")
    x = x[x.role_family.isin(MAIN)]  # 라벨이 적은 직무군은 공고 한 건의 라벨이 드러나 뺀다
    rows = []
    for role, g in x.groupby("role_family"):
        rows.append(dict(role=role, kind="n", name="공고 그룹", value=len(g), n=len(g)))
        lg = g[g.primary_task.notna()]
        rows.append(dict(role=role, kind="n", name="라벨 공고", value=len(lg), n=len(g)))
        if len(lg):
            nn = lg[lg.primary_task != "none"]
            rows.append(dict(role=role, kind="label", name="none", value=(lg.primary_task == "none").mean(), n=len(lg)))
            for gr in GROUPS:
                v = (
                    nn.tasks.map(lambda s, gr=gr: any(tg.get(i) == gr for i in json.loads(s))).mean()
                    if len(nn)
                    else np.nan
                )
                lo, hi = wilson(v, len(nn))
                rows.append(dict(role=role, kind="label", name=gr, value=v, n=len(nn), lo=lo, hi=hi))
        for c in load():
            v = g.txt.map(lambda t, p=re.compile(c["pattern"]): bool(p.search(t))).mean()
            lo, hi = wilson(v, len(g))
            rows.append(dict(role=role, kind="skill", name=c["skill"], value=v, n=len(g), lo=lo, hi=hi))
    return pd.DataFrame(rows)


TABLES = {
    "task_mix": task_mix,
    "label_mix": label_mix,
    "skill_trend": skill_trend,
    "jobkko": jobkko,
    "rename_bound": rename_bound,
}


def main() -> None:
    for name in sys.argv[1:] or TABLES:
        TABLES[name]().round(4).to_csv(RESULTS / f"task1_article2_{name}.csv", index=False)


if __name__ == "__main__":
    main()
