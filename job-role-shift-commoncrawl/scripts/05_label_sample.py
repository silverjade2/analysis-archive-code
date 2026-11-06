"""라벨 표본 추출: 비교군 + 학습 보강 + 평가 표본, seed 42

비교군 193, 학습 보강 300 (제목 규칙 밖 DA 100, BA/BI 100, 희소 유형 키워드 100)
평가 510 (주 대상 9칸 50건씩, DA 인접 3회차 20건씩). 같은 본문은 겹치지 않게 한 건만
이미 만든 폴더가 있으면 멈춘다
"""

import re
import sys
from pathlib import Path

import pandas as pd
from common import (
    CROSS,
    MAX_CHARS,
    MIN_BODY,
    SEAL,
    T1,
    WORK,
    existing_keys,
    load_population,
    md5,
    norm_key,
    plain_text,
    write_jsonl,
)
from roles import role_family

SEED = 42
MAIN = ["DA", "BA/BI", "DS"]
KEYWORDS = {  # 건수가 적은 유형. 영어 본문에서 찾는다
    "llm_agent_dev": (
        r"\bllms?\b|large language model|\brag\b|retrieval[- ]augmented|ai agents?|agentic|prompt engineering|langchain"
    ),
    "recommendation_search": (
        r"recommend(er|ation)|search ranking|learning to rank|personali[sz]ation|information retrieval"
        r"|search relevance"
    ),
    "optimization_operations": (
        r"operations research|linear programming|mixed[- ]integer|optimi[sz]ation (model|problem)|simulation|routing"
        r"|scheduling"
    ),
    "research": (
        r"peer[- ]reviewed|publications?|\bph\.?d\b|publish (papers|research)|research scientist"
        r"|novel (algorithms|methods)"
    ),
    "ai_adoption_automation": (
        r"ai adoption|process automation|workflow automation|digital transformation|\brpa\b|ai transformation"
        r"|automate (business|manual)"
    ),
    "metric_design": (
        r"define (kpis?|metrics)|metrics? framework|kpi framework|event tracking|instrumentation|north star"
        r"|tracking plan"
    ),
    "ml_engineering": (
        r"mlops|model (deployment|serving)|deploy (ml|models)|ml pipelines?|production ml|model monitoring|sagemaker"
    ),
}
KW_N = {
    "llm_agent_dev": 14,
    "recommendation_search": 14,
    "optimization_operations": 14,
    "research": 14,
    "ai_adoption_automation": 14,
    "metric_design": 15,
    "ml_engineering": 15,
}


def pick(df: pd.DataFrame, n: int, used: set, seed: int = SEED) -> pd.DataFrame:
    """무작위로 n건. 이미 쓴 dedup_key를 빼고 같은 본문은 한 건만"""
    d = df[~df.dedup_key.isin(used)].sample(frac=1, random_state=seed).drop_duplicates("dedup_key").head(n)
    used |= set(d.dedup_key)
    return d


def write_chunks(df: pd.DataFrame, base: Path, prefix: str, size: int = 50) -> list[str]:
    """입력 jsonl과 라벨을 붙일 때 읽을 5건씩 텍스트 파일. 청크 수는 ceil(n / size)"""
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)  # 칸과 종류가 청크마다 섞이게
    k = -(-len(df) // size)
    names = []
    for c in range(k):
        part = df.iloc[c::k]
        name = f"{prefix}_{c + 1:02d}"
        names.append(name)
        write_jsonl(
            base / "in" / f"{name}.jsonl",
            [
                {"doc_id": r.doc_id, "title": r.title, "description": (r.raw or "")[:MAX_CHARS]}
                for r in part.itertuples()
            ],
        )
        rows = list(part.itertuples())
        for j in range(0, len(rows), 5):
            blocks = [
                f"=== DOC {j + i + 1} ===\ndoc_id: {r.doc_id}\nTitle: {r.title or ''}\n"
                f"Description:\n{plain_text(r.raw)}\n"
                for i, r in enumerate(rows[j : j + 5])
            ]
            p = base / "txt" / f"{name}_{j // 5 + 1:02d}.txt"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("\n".join(blocks), encoding="utf-8")
    return names


def main() -> None:
    if WORK.exists() or SEAL.exists():
        sys.exit(f"{WORK} 또는 {SEAL} 이 이미 있다. 지우고 다시 하거나 그대로 쓴다")
    pop = load_population()
    pool = pd.read_parquet(T1 / "label_pool.parquet")
    labeled_keys, ex = existing_keys(pop, pool)
    print(f"적용 대상 {len(pop):,}건. 기존 WDC 라벨 {len(ex)}건 중 기준 데이터와 연결 {ex.dedup_key.notna().sum()}건")

    # 평가 표본 먼저. 기존 라벨과 같은 본문은 뺀다
    used = set(labeled_keys)
    ev = []
    for crawl in CROSS:
        for role in MAIN:
            c = pick(pop[(pop.role_family == role) & (pop.crawl == crawl)], 50, used)
            ev.append(c.assign(cell=f"{role}|{crawl}"))
        c = pick(pop[(pop.role_family == "DA 인접") & (pop.crawl == crawl)], 20, used)
        ev.append(c.assign(cell=f"DA 인접|{crawl}"))
    ev = pd.concat(ev)

    # 비교군 193: label_pool 중 라벨 없고 새 규칙으로 DE, ML/AI
    done = set(pd.read_parquet(T1 / "labels_agent.parquet").doc_id)
    pl = pool[~pool.doc_id.isin(done)].copy()
    pl["new_role"] = pl.title.map(role_family)
    pl["body"] = pl.description.fillna("").map(plain_text).str.len()
    cmp_ = pl[pl.new_role.isin(["DE", "ML/AI 엔지니어"]) & (pl.body >= MIN_BODY)].copy()
    print(f"비교군 후보 {len(cmp_)}건: {cmp_.groupby(['source', 'new_role']).size().to_dict()}")
    k2 = pop.drop_duplicates("key2").set_index("key2").dedup_key
    cmp_["dedup_key"] = [
        k2.get(md5(norm_key(t) + norm_key((d or "")[:MAX_CHARS])), "pool-" + i)
        for t, d, i in zip(cmp_.title, cmp_.description, cmp_.doc_id)
    ]
    cmp_ = cmp_[~cmp_.dedup_key.isin(used)].drop_duplicates("dedup_key")
    used |= set(cmp_.dedup_key)

    # 학습 보강: DATA_TITLE 밖 DA 100 + BA/BI 100 (회차 고르게), 희소 유형 키워드 100
    boost = []
    for role in ["DA", "BA/BI"]:
        for crawl, n in zip(CROSS, [33, 33, 34]):
            c = pick(pop[(pop.role_family == role) & (pop.crawl == crawl) & ~pop.in_title], n, used)
            boost.append(c.assign(kind=f"boost_title_{role}"))
    base = pop[pop.role_family.isin(MAIN + ["DE", "ML/AI 엔지니어"])].copy()
    base["lower"] = base.description.map(plain_text).str.lower()
    for name, pat in KEYWORDS.items():
        hit = base[base.lower.str.contains(re.compile(pat))]
        c = pick(hit, KW_N[name], used)
        print(f"키워드 {name}: 후보 {len(hit):,}건 중 {len(c)}건")
        boost.append(c.assign(kind=f"boost_kw_{name}"))
    boost = pd.concat(boost)

    tr = pd.concat(
        [
            cmp_.assign(
                raw=cmp_.description, kind="comparison", crawl=cmp_.crawl, role_family=cmp_.new_role, in_title=False
            )[["doc_id", "title", "raw", "kind", "crawl", "role_family", "dedup_key"]],
            boost.assign(raw=boost.description)[
                ["doc_id", "title", "raw", "kind", "crawl", "role_family", "dedup_key"]
            ],
        ],
        ignore_index=True,
    )
    ev_cols = ["doc_id", "title", "raw", "kind", "crawl", "role_family", "dedup_key", "cell"]
    ev = ev.assign(raw=ev.description, kind="eval")[ev_cols]
    tr_names = write_chunks(tr, WORK, "tr")
    ev_names = write_chunks(ev, SEAL, "ev")
    tr.drop(columns="raw").to_parquet(WORK / "manifest_train.parquet", index=False)
    ev.drop(columns="raw").to_parquet(SEAL / "manifest_test.parquet", index=False)
    print(f"학습용 새 라벨 {len(tr)}건 {len(tr_names)}묶음, 평가 {len(ev)}건 {len(ev_names)}묶음")
    print(tr.kind.value_counts().to_string())
    print(ev.cell.value_counts().sort_index().to_string())
    assert not (set(tr.dedup_key) & set(ev.dedup_key)), "학습과 평가가 같은 본문을 공유"


if __name__ == "__main__":
    main()
