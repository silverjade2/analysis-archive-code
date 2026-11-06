"""기준 데이터에서 기술 후보 96개의 회차별, 직무군별 등장률

본문 판정은 제목 + 본문 소문자. 패턴은 lookaround가 없어 DuckDB (RE2)로 센다
"""

import os

import duckdb
import pandas as pd
from candidates import load
from common import RESULTS, T1
from roles import MIN_BODY_CHARS

SRC = T1 / "wdc_jobs_dedup.parquet"
HITS = T1 / "c_skill_hits.parquet"
OUT = RESULTS / "task1_c_skill_rates.csv"
MAIN = ["DA", "BA/BI", "DS"]


def sql_str(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def build_hits(cands: list[dict]) -> None:
    cols = ",\n".join(f"regexp_matches(txt, {sql_str(c['pattern'])}) AS s{i:03d}" for i, c in enumerate(cands))
    con = duckdb.connect()
    cores = os.environ.get("SPARK_CORES", "4")
    con.execute(f"SET threads={cores}; SET memory_limit='8GB'; SET temp_directory='{T1}/duck_tmp';")
    con.execute(f"""
    COPY (
      SELECT crawl, role_family, page_url, node, body_chars >= {MIN_BODY_CHARS} AS body_ok, {cols}
      FROM (
        SELECT *, lower(coalesce(title, '') || chr(10) || coalesce(description, '')) AS txt FROM read_parquet('{SRC}')
      )
    ) TO '{HITS}' (FORMAT parquet, COMPRESSION zstd)
    """)


def build_rates(cands: list[dict]) -> pd.DataFrame:
    con = duckdb.connect()
    names = [f"s{i:03d}" for i in range(len(cands))]
    hits = con.execute(f"SELECT * FROM '{HITS}'").df()
    hits["role_family"] = hits["role_family"].astype(str)
    main = hits[hits.role_family.isin(MAIN)].assign(role_family="주 대상 합")
    both = pd.concat([hits, main], ignore_index=True)
    rows = []
    for (crawl, role), g in both.groupby(["crawl", "role_family"]):
        ok = g[g.body_ok]
        for c, n in zip(cands, names):
            rows.append(
                dict(
                    crawl=crawl,
                    role=role,
                    skill=c["skill"],
                    source=c["source"],
                    n_post=len(ok),
                    n_hit=int(ok[n].sum()),
                    rate=ok[n].mean() if len(ok) else float("nan"),
                    n_post_all=len(g),
                    rate_all=g[n].mean(),
                )
            )
    return pd.DataFrame(rows)


def main() -> None:
    cands = load()
    build_hits(cands)
    r = build_rates(cands)
    r.to_csv(OUT, index=False, float_format="%.4f")
    print(f"사전 후보 {len(cands)}개, 행 {len(r):,}개, {OUT}")


if __name__ == "__main__":
    main()
