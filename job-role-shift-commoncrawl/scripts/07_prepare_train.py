"""학습용 표: 기존 WDC 라벨 + 새 라벨에 본문, 직무군, 교차 검증 group을 붙인다

평가 표본은 읽지 않는다. group = 같은 본문이거나 같은 조직이면 같은 묶음 (연결 요소)
"""

import pandas as pd
from common import (
    DATA_TITLE,
    T1,
    WORK,
    english_ratio,
    existing_keys,
    load_population,
    plain_text,
    read_jsonl,
)
from roles import role_family

BIG_ORG = 25  # 이 이상 라벨이 걸린 조직 (채용 대행, 게시판 이름 등)은 묶지 않는다


def components(pairs: list[tuple[str, str | None]]) -> list[int]:
    """(본문 key, 조직) 쌍마다 연결 요소 번호"""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for k, o in pairs:
        find("k:" + k)
        if isinstance(o, str) and o:
            parent[find("k:" + k)] = find("o:" + o)
    roots = [find("k:" + k) for k, _ in pairs]
    ids = {r: i for i, r in enumerate(dict.fromkeys(roots))}
    return [ids[r] for r in roots]


def main() -> None:
    pop = load_population()
    pool = pd.read_parquet(T1 / "label_pool.parquet")
    _, ex = existing_keys(pop, pool)

    # 기존 라벨 (WDC 437 + 국내 276)
    lab = pd.read_parquet(T1 / "labels_agent.parquet")[["doc_id", "tasks", "primary_task"]].merge(pool, on="doc_id")
    lab["kind"] = "existing"
    lab["raw"] = lab.description
    lab = lab.merge(ex[["doc_id", "dedup_key"]], on="doc_id", how="left")
    # 새 라벨
    new = pd.read_parquet(WORK / "labels_train_new.parquet")[["doc_id", "tasks", "primary_task"]]
    man = pd.read_parquet(WORK / "manifest_train.parquet")
    nl = new.merge(man, on="doc_id")
    nl_in = pd.concat([pd.DataFrame(read_jsonl(f)) for f in sorted((WORK / "in").glob("*.jsonl"))])
    nl = nl.merge(nl_in[["doc_id", "description"]].rename(columns={"description": "raw"}), on="doc_id")
    nl["source"] = nl.doc_id.str.startswith("jobkko-").map({True: "jobkko", False: "wdc"})

    cols = ["doc_id", "source", "kind", "title", "raw", "crawl", "tasks", "primary_task", "dedup_key"]
    df = pd.concat([lab[cols], nl[cols]], ignore_index=True)
    meta = pop.drop_duplicates("doc_id").set_index("doc_id")
    df["org"] = df.doc_id.map(meta.org_name).where(lambda s: s.notna() & (s.str.len() > 0), df.doc_id.map(meta.host))
    df["org"] = df.org.str.lower().str.strip()
    df["dedup_key"] = df.dedup_key.fillna("own-" + df.doc_id)
    df["role_family"] = df.title.map(role_family)  # label_pool의 role_family는 옛 규칙이라 다시 붙임
    df["text"] = df.raw.map(plain_text)
    df["english"] = df.text.map(lambda t: english_ratio(t) >= 0.12)
    df["in_title"] = df.title.fillna("").str.lower().str.contains(DATA_TITLE)
    big = df.org.value_counts()
    df.loc[df.org.map(big).fillna(0) >= BIG_ORG, "org"] = None
    df["group"] = components(list(zip(df.dedup_key, df.org)))
    df = df.drop(columns="raw")
    df.to_parquet(WORK / "train_table.parquet", index=False)
    g = df.group.value_counts()
    print(
        f"학습표 {len(df)}건 (WDC {(df.source == 'wdc').sum()}, jobkko {(df.source == 'jobkko').sum()}), "
        f"group {len(g)}개, 가장 큰 group {g.iloc[0]}건"
    )
    print(df.groupby(["source", "kind"]).size().to_string())
    print(df.groupby(["source", "role_family"]).size().unstack(0).fillna(0).astype(int).to_string())


if __name__ == "__main__":
    main()
